"""检索质量评估 —— 本项目的“数据科学”核心。

评估口径（设计决策，面试常被追问）：
    相关性判在“来源文档”级而非“块”级：
        不同实验的分块策略/块大小不同，块 ID 天然不同；而“这个问题
        的答案在哪个文档里”是稳定的人工标注。来源级判定让跨策略对比
        公平成立——这是实验设计里最重要的一步。

指标定义：
    HitRate@k    top-k 中是否至少命中一个相关来源（0/1，按查询平均）
    Recall@k     命中的相关来源数 / 全部相关来源数（按查询平均）
    MRR          首个相关结果的排名倒数的平均（衡量“排得多靠前”）
    latency      检索/重排耗时（毫秒，端到端体验指标）
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.config import CONFIG


def load_eval_set(path: Path | None = None) -> list[dict]:
    """读评测集 jsonl。每行：{"query":..., "relevant": ["xx.md"], "reference": "..."}"""
    path = path or CONFIG.paths.eval_dir / "eval_set.jsonl"
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        assert {"query", "relevant"} <= set(r), f"评测集缺字段: {r}"
    return rows


def _retrieved_sources(contexts: list[dict], k: int) -> list[str]:
    """top-k 去重后的来源序列（保持出现顺序，供排名类指标用）。"""
    seen: list[str] = []
    for c in contexts[:k]:
        if c["source"] not in seen:
            seen.append(c["source"])
    return seen


def query_metrics(contexts: list[dict], relevant: list[str], k: int) -> dict:
    """单条查询的指标。"""
    retrieved = _retrieved_sources(contexts, k)
    hits = [s for s in retrieved if s in relevant]
    first_rank = next((i + 1 for i, s in enumerate(retrieved) if s in relevant), None)
    return {
        f"hit@{k}": 1.0 if hits else 0.0,
        f"recall@{k}": len(hits) / len(relevant),
        f"mrr@{k}": (1.0 / first_rank) if first_rank else 0.0,
    }


def evaluate_retrieval(pipeline, eval_set: list[dict], k: int = 5) -> dict:
    """在评测集上整体评估某条管线配置的检索质量。

    注意：只调 pipeline 的检索+重排部分，不走 LLM 生成——
    检索指标要纯粹反映检索本身，不能被生成端的随机性污染。
    同时输出 @1 与 @k 两档指标：语料较易时 recall@k 饱和，
    hit@1 / MRR 才是拉开配置差距的主指标。
    """
    per_query = []
    for row in eval_set:
        t0 = time.perf_counter()
        candidates = pipeline.retriever.search(row["query"], k=pipeline.cfg.retrieval.dense_k
                                               if pipeline.reranker else k)
        if pipeline.reranker is not None:
            candidates = pipeline.reranker.rerank(row["query"], candidates, top_k=k)
        else:
            candidates = candidates[:k]
        latency = (time.perf_counter() - t0) * 1000
        m = query_metrics(candidates, row["relevant"], k=1)   # 严格档：第一名就要对
        m.update(query_metrics(candidates, row["relevant"], k=k))
        m["latency_ms"] = latency
        per_query.append(m)

    n = len(per_query)
    agg = {key: sum(p[key] for p in per_query) / n for key in per_query[0]}
    agg["n_queries"] = n
    return agg
