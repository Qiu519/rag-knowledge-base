"""入库脚本：构建/重建向量索引。

用法（在项目根目录）：
    python scripts/ingest.py                          # 默认配置入库
    python scripts/ingest.py --strategy fixed         # 指定分块策略
    python scripts/ingest.py --strategy recursive --size 300 --overlap 50

语料放 data/raw/（md/txt/pdf 均可），跑完生成：
    data/processed/index/index.faiss + metas.json   向量索引
    data/processed/chunks.jsonl                      分块明细
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 保证从任意目录执行都能 import src 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import ChunkConfig
from src.config import CONFIG
from src.pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 RAG 向量索引")
    parser.add_argument("--strategy", choices=["fixed", "recursive", "sentence"],
                        default=CONFIG.chunk.strategy, help="分块策略")
    parser.add_argument("--size", type=int, default=CONFIG.chunk.chunk_size,
                        help="块大小（字符数）")
    parser.add_argument("--overlap", type=int, default=CONFIG.chunk.overlap,
                        help="相邻块重叠字符数")
    parser.add_argument("--embedding", type=str, default=CONFIG.models.embedding_model,
                        help="embedding 模型名（实验用，如 BAAI/bge-small-zh-v1.5）")
    args = parser.parse_args()

    CONFIG.chunk = ChunkConfig(strategy=args.strategy, chunk_size=args.size,
                               overlap=args.overlap)
    CONFIG.models.embedding_model = args.embedding

    n = RAGPipeline().ingest()
    print(f"完成：共 {n} 块已入库。接下来可运行 python scripts/ask.py 测试问答。")


if __name__ == "__main__":
    main()
