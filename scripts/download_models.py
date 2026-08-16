"""模型下载脚本：把本项目的三个本地模型拉到 S 盘缓存。

模型清单：
    BAAI/bge-base-zh-v1.5     默认向量模型（768 维，约 400MB）
    BAAI/bge-small-zh-v1.5    轻量向量模型（512 维，约 100MB，实验 D 组对照组）
    BAAI/bge-reranker-base    交叉编码器重排模型（约 1.1GB）

缓存位置由 src/config.py 统一重定向到项目内 .hf-home/（S 盘），
下载源默认走 hf-mirror.com 镜像（也写在 config.py）。
已存在的模型会秒过（huggingface_hub 断点续传/校验）。

用法：python scripts/download_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# import config 会设置 HF_HOME / HF_ENDPOINT，必须在 huggingface_hub 之前
from src.config import CONFIG  # noqa: F401

from huggingface_hub import snapshot_download

MODELS = [
    "BAAI/bge-base-zh-v1.5",
    "BAAI/bge-small-zh-v1.5",
    "BAAI/bge-reranker-base",
]


def main() -> None:
    for name in MODELS:
        print(f">>> 下载 {name} ...")
        # 只拉推理必需文件：跳过 onnx/、*.bin（与 safetensors 内容重复），
        # reranker 模型能省约 2GB
        path = snapshot_download(
            name,
            allow_patterns=["*.json", "*.txt", "*.safetensors",
                            "sentencepiece.bpe.model", "vocab*"],
        )
        print(f"    完成：{path}")


if __name__ == "__main__":
    main()
