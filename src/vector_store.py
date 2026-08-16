"""FAISS 向量索引与元数据持久化。

设计：
    - 索引类型 IndexFlatIP（精确内积检索）。
      本项目语料规模在数千~数万块，Flat 精确检索延迟毫秒级，
      不需要 IVF/HNSW 等近似索引——引入倒排反而增加调参负担和召回损失。
      （语料涨到百万级再换 IndexHNSWFlat，接口不用动。）
    - 向量与元数据分开存：
        index.faiss  只有向量（FAISS 原生格式）
        metas.json   块的元数据列表，行号即 FAISS 内部 id
      好处：FAISS 的 id 语义极简，重建/合并索引都不用维护映射表。
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src.config import CONFIG
from src.chunker import Chunk

_INDEX_FILE = "index.faiss"
_META_FILE = "metas.json"


class VectorStore:
    """向量库：add 入库 / search 检索 / save+load 持久化。"""

    def __init__(self, dim: int, index_dir: Path | None = None) -> None:
        self.dim = dim
        self.index_dir = index_dir or CONFIG.paths.index_dir
        self.index = faiss.IndexFlatIP(dim)
        self.metas: list[dict] = []  # 每行对应一个向量：chunk_id/source/heading/text

    def __len__(self) -> int:
        return self.index.ntotal

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """批量入库。chunks 与 embeddings 行对行对应。"""
        assert embeddings.shape == (len(chunks), self.dim), \
            f"向量形状 {embeddings.shape} 与块数 {len(chunks)} 不匹配"
        self.index.add(embeddings)
        self.metas.extend(
            {"chunk_id": c.chunk_id, "source": c.doc_source,
             "heading": c.heading, "text": c.text}
            for c in chunks
        )

    def search(self, query_vec: np.ndarray, top_k: int) -> list[dict]:
        """精确检索 top_k。返回按相似度降序的结果，score 为余弦相似度。"""
        scores, ids = self.index.search(query_vec.reshape(1, -1), top_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:  # 库内向量不足 top_k 时 FAISS 补 -1
                continue
            meta = self.metas[idx]
            results.append({**meta, "score": float(score), "faiss_id": int(idx)})
        return results

    # ---- 持久化 ----

    def save(self) -> Path:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_dir / _INDEX_FILE))
        (self.index_dir / _META_FILE).write_text(
            json.dumps(self.metas, ensure_ascii=False, indent=1), encoding="utf-8")
        return self.index_dir

    @classmethod
    def load(cls, index_dir: Path | None = None) -> "VectorStore":
        index_dir = index_dir or CONFIG.paths.index_dir
        index = faiss.read_index(str(index_dir / _INDEX_FILE))
        store = cls.__new__(cls)  # 绕过 __init__（dim 从文件读取）
        store.index_dir = index_dir
        store.index = index
        store.dim = index.d
        store.metas = json.loads(
            (index_dir / _META_FILE).read_text(encoding="utf-8"))
        return store
