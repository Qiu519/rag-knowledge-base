"""本地 vs 云端生成后端对比实验（实验报告第六章数据源）。

对比口径：同一套检索端（本地 GPU 的向量/BM25/重排完全一致），只切换
生成端——Ollama Qwen2.5-7B（本地、免费、离线） vs 阿里云 qwen3.7-max（API）。
每个后端各自完成"查询改写 + 生成"两步（改写也吃模型能力，端到端才公平）。

题目设计（6 题覆盖 4 类难度）：
    事实题（单点查询）/ 同义改写题（口语→术语）/ 数字辨析题（跨文档易混值）
    / 拒答题（语料外，考察防幻觉）/ 条件综合题（多条件罗列）

用法：
    CLOUD_API_KEY=sk-xxx python scripts/compare_backends.py
云端凭据从环境变量读取，避免提交进仓库。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG, LLMConfig
from src.llm import LLMClient
from src.pipeline import RAGPipeline

QUESTIONS = [
    ("事实题", "毕业论文查重不能超过多少"),
    ("同义改写题", "保研对绩点有什么要求"),
    ("数字辨析题", "奖学金评定里综合素质测评占多少分"),
    ("数字辨析题", "申请辅修要求绩点达到多少"),
    ("条件综合题", "转专业需要满足哪些条件"),
    ("拒答题", "学校的食堂哪家好吃"),
]

LOCAL = dict(base_url="http://localhost:11434/v1", model="qwen2.5:7b", api_key="ollama")
CLOUD = dict(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
             model="qwen3.7-max", api_key=os.getenv("CLOUD_API_KEY", ""))


def run_backend(pipeline: RAGPipeline, cfg_dict: dict, name: str) -> list[dict]:
    """切换生成端配置后逐题问答。"""
    CONFIG.llm = LLMConfig(**cfg_dict, temperature=0.3, max_tokens=1024,
                           enable_thinking=False)
    pipeline.llm = None  # 强制按新配置重建客户端
    pipeline.llm = LLMClient()
    results = []
    for kind, q in QUESTIONS:
        t0 = time.perf_counter()
        r = pipeline.query(q)
        results.append(dict(kind=kind, question=q, backend=name,
                            answer=r.answer.strip(),
                            rewritten=r.rewritten_query,
                            contexts=[c["source"] for c in r.contexts[:3]],
                            total_ms=round((time.perf_counter() - t0) * 1000)))
        print(f"[{name}] {kind}: {q} -> {r.total_ms}ms")
    return results


def main() -> None:
    if not CLOUD["api_key"]:
        sys.exit("请先设置 CLOUD_API_KEY 环境变量（云端后端凭据）")

    pipeline = RAGPipeline().load()
    out = {"local": run_backend(pipeline, LOCAL, "local"),
           "cloud": run_backend(pipeline, CLOUD, "cloud")}

    path = CONFIG.paths.outputs_dir / "comparison_local_vs_cloud.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"原始数据已写入 {path}")

    # 生成并排 Markdown 便于人工评阅
    md = ["# 本地 vs 云端生成后端对比（原始记录）", ""]
    for (l, c) in zip(out["local"], out["cloud"]):
        md += [f"## [{l['kind']}] {l['question']}",
               f"- 本地改写：{l['rewritten'] or '（未改写）'}",
               f"- 云端改写：{c['rewritten'] or '（未改写）'}",
               f"- 检索 top3：{' / '.join(l['contexts'])}",
               "",
               "**本地 qwen2.5:7b：**", l["answer"], "",
               "**云端 qwen3.7-max：**", c["answer"], "",
               f"耗时：本地 {l['total_ms']}ms ｜ 云端 {c['total_ms']}ms", ""]
    (CONFIG.paths.outputs_dir / "comparison_local_vs_cloud.md").write_text(
        "\n".join(md), encoding="utf-8")
    print("并排评阅稿已写入 outputs/comparison_local_vs_cloud.md")


if __name__ == "__main__":
    main()
