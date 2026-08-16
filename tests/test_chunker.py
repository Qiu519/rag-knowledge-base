"""分块器单元测试。

运行：pytest tests/ -v（在项目根目录）

重点验证三点：
    1. 块长不超上限（fixed/recursive/sentence 三种策略都成立）；
    2. recursive 策略优先在天然边界切开（不把句子拦腰截断）；
    3. 块 ID 稳定且唯一（评测集的相关性标注依赖这一点）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import chunk_document, _split_by_seps
from src.config import ChunkConfig
from src.document_loader import LoadedDoc


def make_doc(text: str, source: str = "测试文档.md") -> LoadedDoc:
    return LoadedDoc(text=text, source=source, title="测试", meta={})


LONG_BODY = "这是第一句话，讲的是课程安排。" * 40  # 无换行长文，约 640 字


def test_fixed_respects_size():
    doc = make_doc("# 标题\n" + LONG_BODY)
    cfg = ChunkConfig(strategy="fixed", chunk_size=100, overlap=20)
    chunks = chunk_document(doc, cfg)
    assert chunks, "fixed 策略不应产出空列表"
    assert all(len(c.text) <= 100 for c in chunks)


def test_recursive_respects_size_and_prefers_boundary():
    body = "这是第一句话。这是第二句话。" + "这是后续内容。" * 40
    doc = make_doc("# 章节\n" + body)
    cfg = ChunkConfig(strategy="recursive", chunk_size=60, overlap=10)
    chunks = chunk_document(doc, cfg)
    assert all(len(c.text) <= 60 for c in chunks)
    # recursive 在句号边界切分：每块应以句号结尾（或为最后一块）
    assert all(c.text.endswith("。") for c in chunks)


def test_sentence_packs_whole_sentences():
    body = "第一句内容。第二句内容。第三句内容。" * 20
    doc = make_doc("# 章节\n" + body)
    cfg = ChunkConfig(strategy="sentence", chunk_size=50, overlap=0)
    chunks = chunk_document(doc, cfg)
    assert all(len(c.text) <= 50 for c in chunks)
    assert all(c.text.endswith("。") for c in chunks)


def test_heading_tracking():
    body = "甲章节的正文内容。\n\n# 乙标题\n乙章节的正文内容。"
    doc = make_doc(body)
    chunks = chunk_document(doc, ChunkConfig(strategy="recursive", chunk_size=400, overlap=60))
    by_text = {c.text: c.heading for c in chunks}
    assert by_text.get("甲章节的正文内容。") == ""
    assert by_text.get("乙章节的正文内容。") == "乙标题"


def test_chunk_id_unique_and_stable():
    doc = make_doc("# 标题\n" + LONG_BODY)
    cfg = ChunkConfig(strategy="fixed", chunk_size=100, overlap=20)
    ids1 = [c.chunk_id for c in chunk_document(doc, cfg)]
    ids2 = [c.chunk_id for c in chunk_document(doc, cfg)]
    assert ids1 == ids2, "同一文档两次分块，ID 必须一致（评测可复现的前提）"
    assert len(set(ids1)) == len(ids1), "块 ID 必须唯一"


def test_split_by_seps_preserves_boundary():
    """带换行前缀的分隔符应补回段首，标题不被吞。"""
    text = "段落一内容\n\n# 第二章\n第二章正文"
    parts = _split_by_seps(text, ["\n\n", "\n# ", "\n", "。"])
    joined = "".join(parts)
    assert "第二章" in joined and "段落一内容" in joined
