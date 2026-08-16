"""命令行问答：快速验证检索与生成效果。

用法：
    python scripts/ask.py "保研的绩点要求是什么？"          # 单问
    python scripts/ask.py                                   # 交互模式（exit 退出）
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import RAGPipeline


def print_result(result) -> None:
    """终端友好输出：答案 + 引用来源 + 各阶段耗时。"""
    bar = "=" * 64
    print(f"\n{bar}\n问题：{result.question}\n{bar}")
    print(result.answer)
    print(f"\n--- 引用片段（{len(result.contexts)} 条）---")
    for i, c in enumerate(result.contexts, start=1):
        score = c.get("rerank_score", c.get("rrf_score", c.get("score", 0)))
        heading = f" [{c['heading']}]" if c.get("heading") else ""
        print(f"  [{i}] {c['source']}{heading}（得分 {score:.3f}）")
    t = result.timings_ms
    print(f"--- 耗时：检索 {t.get('retrieval_ms')}ms | 重排 {t.get('rerank_ms')}ms"
          f" | 生成 {t.get('generate_ms')}ms | 总计 {t.get('total_ms')}ms"
          f" | 模式 {result.mode} ---")


def main() -> None:
    if len(sys.argv) > 1:
        pipeline = RAGPipeline().load()
        print_result(pipeline.query(" ".join(sys.argv[1:])))
        return

    pipeline = RAGPipeline().load()
    print("交互问答已就绪（输入 exit 退出）")
    while True:
        try:
            question = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"exit", "quit", "q"}:
            break
        print_result(pipeline.query(question))


if __name__ == "__main__":
    main()
