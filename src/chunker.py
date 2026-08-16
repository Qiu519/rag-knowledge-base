"""文本分块（chunking）—— RAG 效果的第一道关口。

为什么分块如此重要：
    向量模型把“一整块文本”压缩成一个向量，块太大则主题混杂、向量失焦，
    块太小则上下文割裂。分块边界切得好不好，直接决定检索上限。

三种策略（评估实验的对比对象之一）：
    fixed      纯字符滑窗。简单基线：按 chunk_size 切、相邻块重叠 overlap。
    recursive  层级切分（默认）。优先在 Markdown 标题 → 空行分段 → 句号 →
               分号/逗号等“天然边界”处下刀，块内语义尽量完整。
    sentence   先切整句，再把相邻句拼到预算内。适合无结构的纯文本。

块 ID 规则：{文档source}::{序号}，与文档加载顺序绑定——加载侧已排序，
所以同一批语料多次入库，ID 恒定，评测集里的相关性标注不会失效。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.config import CONFIG, ChunkConfig
from src.document_loader import LoadedDoc

# recursive 策略的层级分隔符：从上到下优先级递减。
# 前缀 \n 的设计：保留分隔符本身在块首（如 "\n# "），标题不丢字。
# 注意：列表不能以空串结尾——所有分隔符都未命中时递归自然返回原文，
# 超长段落由 _chunks_recursive 里的滑窗兜底。
_SENTENCE_END = "。"  # 中文句号；英文句点对中文语料干扰大，暂不启用
_RECURSIVE_SEPS = ["\n# ", "\n\n", f"{_SENTENCE_END}\n", _SENTENCE_END, "；", "，", "\n", " "]

# 匹配 Markdown 标题行（# 到 ######），用于给块打“所属标题”标签
_HEADING_RE = re.compile(r"^#{1,6}\s+")


@dataclass
class Chunk:
    """一个文本块。

    heading: 该块所属的最近一级标题文本（无标题则为空串），
             检索结果里展示它，用户能立刻判断片段来自哪个章节。
    """

    chunk_id: str      # 全局唯一："{source}::{序号}"
    doc_source: str    # 来源文档（溯源用）
    heading: str
    text: str

    @property
    def display(self) -> str:
        """给大模型/界面看的展示文本：带上下文标签。"""
        prefix = f"[{self.heading}] " if self.heading else ""
        return f"{prefix}{self.text}"


# ---------------------------------------------------------------------------
# 三种策略的实现
# ---------------------------------------------------------------------------

def _chunks_fixed(text: str, cfg: ChunkConfig) -> list[str]:
    """字符滑窗切分（基线策略）。

    步长 = chunk_size - overlap，保证相邻块有重叠、边界信息不丢失。
    """
    step = max(cfg.chunk_size - cfg.overlap, 1)
    return [text[i:i + cfg.chunk_size]
            for i in range(0, len(text), step) if text[i:i + cfg.chunk_size].strip()]


def _split_by_seps(text: str, seps: list[str]) -> list[str]:
    """按分隔符列表层级切分（recursive 策略的核心）。

    逻辑：取第一个在文本中出现的分隔符，把文本切开；
    若某段仍超长且还有下一级分隔符，则对该段递归下钻。
    带换行前缀的分隔符（如 "\\n# "）会补回后续各段的开头，
    这样段与段合并回去时，标题/段落边界不丢失。
    """
    if not seps:
        return [text]
    sep, rest = seps[0], seps[1:]

    if sep not in text:
        return _split_by_seps(text, rest)

    parts = text.split(sep)
    if sep.startswith("\n"):
        # 换行类：分隔符归到后续段落的开头（首段之前没有分隔符，不补）
        parts = [parts[0]] + [sep + p for p in parts[1:]]
    else:
        # 句读类（。；，）：分隔符归还给前一段的结尾，保持句子完整
        parts = [p + sep for p in parts[:-1]] + parts[-1:]

    out: list[str] = []
    for part in parts:
        out.extend(_split_by_seps(part, rest))
    return [p for p in out if p.strip()]


def _chunks_recursive(text: str, cfg: ChunkConfig) -> list[str]:
    """层级切分 + 小段合并。

    两步走：
      1) _split_by_seps 尽量在天然边界切开（可能产生很多小段）；
      2) 贪心合并相邻小段，直到接近 chunk_size 上限。
    这样本能产出“边界自然、长度均匀”的块。
    """
    pieces = _split_by_seps(text, list(_RECURSIVE_SEPS))
    merged: list[str] = []
    buffer = ""
    for piece in pieces:
        # 单段本身超长（如无任何标点的长串）：退化为滑窗硬切
        if len(piece) > cfg.chunk_size:
            if buffer:
                merged.append(buffer)
                buffer = ""
            merged.extend(_chunks_fixed(piece, cfg))
            continue
        if len(buffer) + len(piece) <= cfg.chunk_size:
            buffer += piece
        else:
            if buffer:
                merged.append(buffer)
            buffer = piece
    if buffer:
        merged.append(buffer)
    return [m.strip() for m in merged if m.strip()]


def _split_sentences(text: str) -> list[str]:
    """按中文句号切句，保留句尾标点。"""
    parts = text.split(_SENTENCE_END)
    return [p + _SENTENCE_END for p in parts[:-1]] + ([parts[-1]] if parts[-1].strip() else [])


def _chunks_sentence(text: str, cfg: ChunkConfig) -> list[str]:
    """整句打包：句子为最小单元，贪心装进 chunk_size 预算。"""
    sentences = _split_sentences(text)
    packed: list[str] = []
    buffer = ""
    for s in sentences:
        if len(buffer) + len(s) <= cfg.chunk_size:
            buffer += s
        else:
            if buffer:
                packed.append(buffer)
            buffer = s
    if buffer:
        packed.append(buffer)
    return packed


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------

def chunk_document(doc: LoadedDoc, cfg: ChunkConfig | None = None) -> list[Chunk]:
    """把单个文档切块，并跟踪每块所属的最近标题。

    标题跟踪：逐行扫描，遇到 Markdown 标题行就更新“当前标题”，
    后续块打上该标签——这样正文块也能携带章节上下文。
    """
    cfg = cfg or CONFIG.chunk
    strategy_fn = {
        "fixed": _chunks_fixed,
        "recursive": _chunks_recursive,
        "sentence": _chunks_sentence,
    }.get(cfg.strategy)
    if strategy_fn is None:
        raise ValueError(f"未知分块策略: {cfg.strategy}（可选 fixed/recursive/sentence）")

    # 逐行维护“当前标题”，把正文按标题分段后分别送入策略函数，
    # 避免跨章节合并导致 heading 标注失真。
    lines = doc.text.split("\n")
    sections: list[tuple[str, str]] = []  # (heading, 正文)
    current_heading, buf = "", []
    for line in lines:
        if _HEADING_RE.match(line):
            if buf:
                sections.append((current_heading, "\n".join(buf)))
                buf = []
            current_heading = _HEADING_RE.sub("", line).strip()
        else:
            buf.append(line)
    if buf:
        sections.append((current_heading, "\n".join(buf)))
    if not sections:  # 全文都是标题的极端情况
        sections = [("", doc.text)]

    chunks: list[Chunk] = []
    for heading, body in sections:
        if not body.strip():
            continue
        for piece in strategy_fn(body, cfg):
            chunks.append(Chunk(
                chunk_id="",  # 占位，下面统一编号
                doc_source=doc.source,
                heading=heading,
                text=piece,
            ))

    # 统一编号生成稳定的块 ID
    for i, c in enumerate(chunks):
        c.chunk_id = f"{doc.source}::{i:04d}"
    return chunks


def chunk_documents(docs: list[LoadedDoc], cfg: ChunkConfig | None = None) -> list[Chunk]:
    """批量切块。"""
    cfg = cfg or CONFIG.chunk
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, cfg))
    return all_chunks
