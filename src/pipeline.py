"""端到端 RAG 管线编排。

两类入口：
    ingest()  入库：读文档 → 分块 → 向量化 → 建索引 → 落盘
    load()    加载已建索引（含 BM25 重建），之后 query() 问答

query 的完整数据流与埋点：
    问题 → [检索 embed] → 混合召回 → [重排] → [LLM 生成] → 答案+引用
    每个阶段记录耗时（毫秒），界面上直接展示——
    “检索 45ms / 重排 80ms / 生成 1.2s”这种数字既是体验指标，
    也是实验报告里延迟分析的数据来源。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.chunker import Chunk, chunk_documents
from src.config import CONFIG, AppConfig, ChunkConfig, RetrievalConfig
from src.document_loader import load_documents
from src.embedder import Embedder
from src.llm import LLMClient
from src.reranker import Reranker
from src.retriever import HybridRetriever
from src.vector_store import VectorStore

_CHUNKS_FILE = "chunks.jsonl"  # 分块明细落盘（评测与调试用）


@dataclass
class QueryResult:
    """一次问答的完整结果。"""

    question: str
    answer: str
    contexts: list[dict]          # 最终送入 LLM 的片段（含分数与来源）
    mode: str                     # api / extractive
    timings_ms: dict = field(default_factory=dict)  # 各阶段耗时

    @property
    def total_ms(self) -> float:
        return sum(self.timings_ms.values())


class RAGPipeline:
    """RAG 主类。持有全部重型资源（模型、索引），进程内只初始化一次。"""

    def __init__(self, cfg: AppConfig | None = None) -> None:
        self.cfg = cfg or CONFIG
        self.embedder: Embedder | None = None
        self.retriever: HybridRetriever | None = None
        self.reranker: Reranker | None = None
        self.llm: LLMClient | None = None

    # ------------------------------------------------------------------
    # 入库
    # ------------------------------------------------------------------

    def ingest(self, chunk_cfg: ChunkConfig | None = None) -> int:
        """全量构建索引，返回块数量。

        简单起见采用“全量重建”而非增量更新：本场景语料更换频率低、
        构建耗时分钟级，全量重建保证索引与语料强一致，免去增量同步的
        复杂度（这是规模换简单的有意取舍）。
        """
        chunk_cfg = chunk_cfg or self.cfg.chunk
        docs = load_documents()
        if not docs:
            raise RuntimeError(
                f"未在 {self.cfg.paths.raw_docs_dir} 找到可入库文档（支持 md/txt/pdf）")

        chunks = chunk_documents(docs, chunk_cfg)
        print(f"[pipeline] 文档 {len(docs)} 份 → 分块 {len(chunks)} 块"
              f"（策略 {chunk_cfg.strategy}, size={chunk_cfg.chunk_size}）")

        self.embedder = self.embedder or Embedder()
        self.retriever = HybridRetriever.from_chunks(chunks, self.embedder, self.cfg.retrieval)
        self.retriever.store.save()

        # 分块明细同步落盘：评测集标注、坏例分析都靠它
        chunks_file = self.cfg.paths.processed_dir / _CHUNKS_FILE
        with chunks_file.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(
                    {"chunk_id": c.chunk_id, "source": c.doc_source,
                     "heading": c.heading, "text": c.text},
                    ensure_ascii=False) + "\n")
        print(f"[pipeline] 索引与分块明细已写入 {self.cfg.paths.index_dir}")
        return len(chunks)

    # ------------------------------------------------------------------
    # 加载与问答
    # ------------------------------------------------------------------

    def load(self, use_rerank: bool | None = None) -> "RAGPipeline":
        """加载已构建的索引与各模型。问答前必须先 load 或 ingest。"""
        if use_rerank is not None:
            self.cfg.retrieval.use_rerank = use_rerank

        self.embedder = self.embedder or Embedder()
        store = VectorStore.load()
        self.retriever = HybridRetriever(store, self.embedder, self.cfg.retrieval)
        if self.cfg.retrieval.use_rerank:
            self.reranker = self.reranker or Reranker()
        self.llm = self.llm or LLMClient()
        print(f"[pipeline] 就绪：{len(store)} 个块 | "
              f"混合检索 {'开' if self.cfg.retrieval.use_hybrid else '关'} | "
              f"重排 {'开' if self.reranker else '关'} | 生成 {self.llm.mode}")
        return self

    def query(self, question: str, top_k: int | None = None) -> QueryResult:
        """端到端问答。"""
        if self.retriever is None:
            raise RuntimeError("管线未加载，请先调用 load() 或 ingest()")

        t_retrieval = time.perf_counter()
        candidates = self.retriever.search(question, k=self.cfg.retrieval.dense_k
                                           if self.cfg.retrieval.use_rerank else top_k)
        t_rerank = time.perf_counter()
        # 有重排器时：召回多取候选，重排后截断到 top_k；否则直接截断
        if self.reranker is not None:
            candidates = self.reranker.rerank(question, candidates,
                                              top_k or self.cfg.retrieval.final_k)
        elif top_k:
            candidates = candidates[:top_k]
        t_generate = time.perf_counter()
        self.llm = self.llm or LLMClient()
        answer, gen_sec = self.llm.answer(question, candidates)
        t_end = time.perf_counter()

        return QueryResult(
            question=question,
            answer=answer,
            contexts=candidates,
            mode=self.llm.mode,
            timings_ms={
                "retrieval_ms": round((t_rerank - t_retrieval) * 1000, 1),
                "rerank_ms": round((t_generate - t_rerank) * 1000, 1),
                "generate_ms": round(gen_sec * 1000, 1),
                "total_ms": round((t_end - t_retrieval) * 1000, 1),
            },
        )
