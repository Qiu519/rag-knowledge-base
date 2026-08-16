"""文档加载与清洗。

职责：把 data/raw/ 下的原始文档读成统一的 LoadedDoc 列表，
后续分块、向量化只面对统一结构，不关心文件格式。

支持格式：.md / .txt / .pdf
    - Markdown 保留 # 标题符号（chunker 依赖它识别结构）；
    - PDF 用 pypdf 逐页抽取文本（扫描版 PDF 无文字层，抽取结果为空，
      本项目暂不做 OCR，遇到会打印警告并跳过）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from src.config import CONFIG

# 允许入库的扩展名（小写）。新增格式在这里和 load_single 里同步扩展。
SUPPORTED_EXTS = {".md", ".txt", ".pdf"}


@dataclass
class LoadedDoc:
    """一份加载完成的文档。

    text    清洗后的正文（保留 Markdown 标记）
    source  相对 data/raw 的相对路径，作为溯源标识贯穿全流程
    title   文档标题：Markdown 取首个一级标题，其余取文件名
    meta    预留扩展字段（如页数、加载时间等）
    """

    text: str
    source: str
    title: str
    meta: dict = field(default_factory=dict)


def clean_text(text: str) -> str:
    """正文清洗：去 BOM/零宽字符、归一换行、压缩连续空行。

    注意保持克制——只清理明显的噪声，不碰正文内容本身，
    避免误伤代码块、表格这类对空白敏感的结构。
    """
    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)      # 3+ 连续空行压成 1 个
    text = re.sub(r"[ \t]+\n", "\n", text)       # 行尾空白
    return text.strip()


def _extract_title(text: str, fallback: str) -> str:
    """取文档标题：优先首个 '# ' 一级标题，否则用文件名（去扩展名）。"""
    for line in text.split("\n"):
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _load_pdf(path: Path) -> str:
    """逐页抽取 PDF 文字层并拼接。空结果（扫描件）由调用方告警跳过。"""
    from pypdf import PdfReader  # 延迟导入：无 PDF 场景不需要这个依赖

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def load_single(path: Path, raw_root: Path) -> LoadedDoc:
    """加载单个文档为 LoadedDoc。不支持的扩展名抛 ValueError。"""
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"不支持的文件类型: {path.name}（支持 {SUPPORTED_EXTS}）")

    if ext == ".pdf":
        text = _load_pdf(path)
    else:
        # md/txt 统一按 UTF-8 读（errors='ignore' 容忍混入的个别坏字节）
        text = path.read_text(encoding="utf-8", errors="ignore")

    text = clean_text(text)
    rel_source = path.relative_to(raw_root).as_posix()
    return LoadedDoc(
        text=text,
        source=rel_source,
        title=_extract_title(text, path.stem),
        meta={"ext": ext},
    )


def load_documents(raw_dir: Path | None = None) -> list[LoadedDoc]:
    """递归扫描目录，加载全部受支持文档。

    返回按 source 排序的列表，保证多次构建索引时块 ID 顺序稳定，
    这是实验可复现的前提（块 ID 不因文件系统遍历顺序而变）。
    """
    raw_dir = raw_dir or CONFIG.paths.raw_docs_dir
    docs: list[LoadedDoc] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        doc = load_single(path, raw_dir)
        if not doc.text:
            print(f"[loader] 跳过空文档: {doc.source}")
            continue
        docs.append(doc)
    return docs
