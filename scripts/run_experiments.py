"""批量对照实验：系统回答“每一步优化到底值多少分”。

实验矩阵（四组，每组只变一个因子——控制变量法）：
    A 分块策略   fixed / recursive / sentence（size=400 对齐）
    B 块大小     200 / 400 / 600（recursive）
    C 检索消融   混合检索 on/off × 重排 on/off（recursive/400）
    D 向量模型   bge-small-zh-v1.5 vs bge-base-zh-v1.5

产出：
    outputs/results.csv          全部指标明细
    docs/figures/*.png           对比图（写入 README/实验报告）

用法：python scripts/run_experiments.py [--k 5] [--quick]
    --quick 只跑 C 组消融（约 1 次建库 + 4 轮评估，快速验证评估链路）
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.chunker import ChunkConfig
from src.config import CONFIG
from src.evaluate import evaluate_retrieval, load_eval_set
from src.pipeline import RAGPipeline
from src.reranker import Reranker

SMALL_EMB = "BAAI/bge-small-zh-v1.5"
BASE_EMB = "BAAI/bge-base-zh-v1.5"


def build_experiments(quick: bool) -> list[dict]:
    """生成实验配置列表。每项：{名字, chunk配置, 检索配置, embedding模型}。"""
    base_rc = dict(CONFIG.retrieval.__dict__)  # 默认检索配置的拷贝

    def rc(**overrides) -> dict:
        cfg = dict(base_rc)
        cfg.update(overrides)
        return cfg

    exps: list[dict] = []

    # A 组：分块策略
    for strat in ["fixed", "recursive", "sentence"]:
        exps.append(dict(
            group="A_分块策略", name=f"strategy={strat}",
            chunk=dict(strategy=strat, chunk_size=400, overlap=60),
            retrieval=rc(), embedding=BASE_EMB))

    # B 组：块大小（recursive）
    for size in [200, 400, 600]:
        exps.append(dict(
            group="B_块大小", name=f"size={size}",
            chunk=dict(strategy="recursive", chunk_size=size, overlap=60),
            retrieval=rc(), embedding=BASE_EMB))

    # C 组：检索消融
    for hybrid, rerank in itertools.product([True, False], repeat=2):
        exps.append(dict(
            group="C_检索消融", name=f"hybrid={'on' if hybrid else 'off'},rerank={'on' if rerank else 'off'}",
            chunk=dict(strategy="recursive", chunk_size=400, overlap=60),
            retrieval=rc(use_hybrid=hybrid, use_rerank=rerank), embedding=BASE_EMB))

    # D 组：向量模型
    for emb in [SMALL_EMB, BASE_EMB]:
        exps.append(dict(
            group="D_向量模型", name=emb.split("/")[-1],
            chunk=dict(strategy="recursive", chunk_size=400, overlap=60),
            retrieval=rc(), embedding=emb))

    if quick:  # 快速模式：只保留 C 组
        exps = [e for e in exps if e["group"] == "C_检索消融"]
    return exps


def apply_config(exp: dict, pipeline: RAGPipeline, cache: dict) -> None:
    """把实验配置写回全局 CONFIG 并处理模型复用。

    优化点：embedding 模型与上一个实验相同时不重载（省 10s 级加载时间）；
    重排器只要 C 组没有全关就一直复用。config 写回后由 ingest 重建索引。
    """
    CONFIG.chunk = ChunkConfig(**exp["chunk"])
    for key, val in exp["retrieval"].items():
        setattr(CONFIG.retrieval, key, val)
    if cache.get("embedding") != exp["embedding"]:
        pipeline.embedder = None               # 触发 Embedder 重建
        CONFIG.models.embedding_model = exp["embedding"]
        cache["embedding"] = exp["embedding"]
    pipeline.reranker = Reranker() if CONFIG.retrieval.use_rerank and pipeline.reranker is None else \
        (None if not CONFIG.retrieval.use_rerank else pipeline.reranker)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="指标截断位置（默认 5）")
    parser.add_argument("--quick", action="store_true", help="只跑 C 组消融")
    args = parser.parse_args()

    eval_set = load_eval_set()
    print(f"评测集 {len(eval_set)} 条 | 截断 k={args.k}")

    pipeline = RAGPipeline()
    cache: dict = {}
    rows = []
    for exp in build_experiments(args.quick):
        label = f"[{exp['group']}] {exp['name']}"
        print(f"\n>>> 运行实验 {label}")
        apply_config(exp, pipeline, cache)
        t0 = time.perf_counter()
        n_chunks = pipeline.ingest()           # 按当前配置重建索引
        pipeline.load()                        # 重建 BM25/加载生成端
        metrics = evaluate_retrieval(pipeline, eval_set, k=args.k)
        row = dict(experiment=exp["name"], group=exp["group"],
                   strategy=exp["chunk"]["strategy"], size=exp["chunk"]["chunk_size"],
                   hybrid=exp["retrieval"]["use_hybrid"], rerank=exp["retrieval"]["use_rerank"],
                   embedding=exp["embedding"].split("/")[-1],
                   n_chunks=n_chunks, **metrics)
        rows.append(row)
        print(f"    块数 {n_chunks} | recall@{args.k}={metrics[f'recall@{args.k}']:.3f}"
              f" | mrr@{args.k}={metrics[f'mrr@{args.k}']:.3f}"
              f" | 建库+评估 {time.perf_counter()-t0:.0f}s")

    out = CONFIG.paths.outputs_dir
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False, encoding="utf-8-sig")
    print(f"\n结果已写入 {out / 'results.csv'}")
    print(df[["experiment", f"recall@{args.k}", f"mrr@{args.k}", "latency_ms"]].to_string(index=False))


if __name__ == "__main__":
    main()
