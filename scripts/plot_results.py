"""从 outputs/results.csv 生成实验对比图（写入 docs/figures/）。

用法：python scripts/plot_results.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd

from src.config import CONFIG

# 中文字体（Windows 自带微软雅黑；Linux 环境可换 Noto Sans CJK）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_group(df: pd.DataFrame, group: str, metrics: list[str],
               title: str, out_name: str) -> None:
    sub = df[df["group"] == group].set_index("experiment")
    if sub.empty:
        return
    ax = sub[metrics].plot(kind="bar", figsize=(9, 4.5), width=0.7)
    ax.set_title(title)
    ax.set_ylabel("指标值")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out = CONFIG.paths.outputs_dir / out_name
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"已生成 {out}")


def main() -> None:
    df = pd.read_csv(CONFIG.paths.outputs_dir / "results.csv")
    plot_group(df, "A_分块策略", ["hit@1", "recall@5", "mrr@5"],
               "A组：分块策略对检索质量的影响", "figA_分块策略.png")
    plot_group(df, "B_块大小", ["hit@1", "recall@5", "mrr@5"],
               "B组：块大小对检索质量的影响", "figB_块大小.png")
    plot_group(df, "C_检索消融", ["hit@1", "recall@5", "mrr@5"],
               "C组：混合检索×重排消融", "figC_检索消融.png")
    plot_group(df, "D_向量模型", ["hit@1", "recall@5", "mrr@5"],
               "D组：向量模型对比", "figD_向量模型.png")

    # 延迟单独一张（量纲不同，不与准确率混轴）
    sub = df.set_index("experiment")[["latency_ms"]]
    ax = sub.plot(kind="barh", figsize=(9, 5), legend=False)
    ax.set_title("各配置检索+重排延迟（ms，评测集 38 题均值）")
    ax.set_xlabel("延迟 (ms)")
    plt.tight_layout()
    out = CONFIG.paths.outputs_dir / "figE_延迟.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"已生成 {out}")


if __name__ == "__main__":
    main()
