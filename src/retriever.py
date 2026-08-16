"""混合检索：稠密向量（语义）+ BM25（关键词），RRF 融合。

为什么要混合：
    - 向量检索强在“语义改写”（问“挂科了还能保研吗”能召回写“学业警告…
      取消推免资格”的段落），但对专有名词、课程代码、数字不敏感；
    - BM25 是经典词频检索，对精确术语命中极稳，但不懂同义改写；
    - 两者错误类型互补，融合后召回率上限高于任何单路（实验会给数据）。

RRF（Reciprocal Rank Fusion）融合公式：
    score(doc) = Σ_rank 1 / (rrf_k + rank(doc))
    只用“排名”不用“分值”，天然规避两路分数量纲不可比的问题，
    是无训练融合的标准做法。
"""

from __future__ import annotations

import jieba
from rank_bm25 import BM25Okapi

from src.chunker import Chunk
from src.config import CONFIG, RetrievalConfig
from src.embedder import Embedder
from src.vector_store import VectorStore


def chunk_repr(meta: dict) -> str:
    """块的“检索表示”：正文前拼上所属标题。

    为什么：分块按标题切段后，纯正文块可能完全不含主题词——例如
    “二、基本申请条件”一节的正文是编号列表，通篇没有“申请条件/保研”
    字样，向量与重排都无法把它和“保研要求”类问题关联（实测踩坑）。
    拼上标题后语义完整，这是上下文增强分块（contextual chunk）的标准做法。
    注意：该表示只用于 embedding / BM25 / 重排打分；喂给 LLM 的上下文
    仍用干净正文（见 llm.build_context），避免重复噪音。
    """
    heading = meta.get("heading") or ""
    return f"{heading}\n{meta['text']}" if heading else meta["text"]


def _tokenize_zh(text: str) -> list[str]:
    """中文分词 + 轻量停用词过滤（BM25 的索引单元）。

    停用词表刻意极简：BM25 的 IDF 本身会压低高频词权重，
    大规模停用反而伤召回，这里只去掉纯标点与单字虚词。
    """
    STOP = set("，。、；：？！“”‘’（）《》的了呢吧啊吗之")
    return [t for t in jieba.lcut(text) if t.strip() and t not in STOP]


class HybridRetriever:
    """两路召回 + RRF 融合的检索器。

    BM25 索引不落盘：语料直接取自 VectorStore 的元数据，
    加载 FAISS 索引后现场重建（数千块重建耗时秒级，省去同步两份缓存）。
    """

    def __init__(self, store: VectorStore, embedder: Embedder,
                 cfg: RetrievalConfig | None = None) -> None:
        self.store = store
        self.embedder = embedder
        self.cfg = cfg or CONFIG.retrieval

        texts = [chunk_repr(m) for m in store.metas]
        tokenized = [_tokenize_zh(t) for t in texts]
        # BM25Okapi 不接受空文档，兜底插入占位符
        tokenized = [toks if toks else ["空"] for toks in tokenized]
        self.bm25 = BM25Okapi(tokenized)
        self._tokenized = tokenized

    def _dense_search(self, query: str, k: int) -> list[dict]:
        return self.store.search(self.embedder.encode_query(query), k)

    def _bm25_search(self, query: str, k: int) -> list[dict]:
        scores = self.bm25.get_scores(_tokenize_zh(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [{**self.store.metas[i], "bm25_score": float(scores[i])} for i in top if scores[i] > 0]

    def search(self, query: str, k: int | None = None) -> list[dict]:
        """对外检索入口。返回融合排名后的候选列表（含各路分数）。

        use_hybrid=False 时退化为纯向量检索（消融实验用）。
        """
        cfg = self.cfg
        k = k or cfg.final_k

        dense_hits = self._dense_search(query, cfg.dense_k)
        if not cfg.use_hybrid:
            return dense_hits[:k]

        bm25_hits = self._bm25_search(query, cfg.bm25_k)
        # RRF 融合：rank 从 1 计
        rrf: dict[str, float] = {}
        for hits in (dense_hits, bm25_hits):
            for rank, hit in enumerate(hits, start=1):
                cid = hit["chunk_id"]
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (cfg.rrf_k + rank)

        # 按 chunk_id 找回元数据（两路结果里必有其一）
        meta_by_id = {h["chunk_id"]: h for h in dense_hits}
        meta_by_id.update({h["chunk_id"]: h for h in bm25_hits})
        fused = [
            {**meta_by_id[cid], "rrf_score": s}
            for cid, s in sorted(rrf.items(), key=lambda x: x[1], reverse=True)
        ]
        return fused[:k]

    @classmethod
    def from_chunks(cls, chunks: list[Chunk], embedder: Embedder,
                    cfg: RetrievalConfig | None = None) -> "HybridRetriever":
        """从块列表构建（入库流程用）。

        向量编码用“标题+正文”的检索表示（见 chunk_repr），与检索/
        重排口径一致，保证“同一文本怎么编码就怎么比对”。
        """
        store = VectorStore(embedder.dim)
        store.add(chunks, embedder.encode_corpus(
            [chunk_repr({"heading": c.heading, "text": c.text}) for c in chunks]))
        return cls(store, embedder, cfg)
