"""文本向量化（embedding）—— 把文本映射为语义向量。

选型：BAAI/bge-base-zh-v1.5（768 维）
    - 中文语义检索的主流开源基线，质量/速度均衡；
    - v1.5 版本显著缓解了“相似度分数分布异常”的老问题。

工程要点：
    1. L2 归一化：向量归一化后，内积（IndexFlatIP）== 余弦相似度，
       FAISS 侧无需额外换算；
    2. 查询侧指令：bge 系列 s2q（短查询→段落）检索官方建议给查询加
       前缀指令，可小幅提升召回（文档侧不加）。做成开关便于消融实验；
    3. GPU 批量编码：encode 走 cuda + batch，万级文本分块在
       RTX 4060 上分钟级完成，这是“本地 GPU 加速”卖点的主体。
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import CONFIG, ModelConfig

# bge 官方建议的中文查询指令（对称检索/文档侧不加）
QUERY_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："


class Embedder:
    """向量模型封装：加载一次，全文编码 / 查询编码复用。"""

    def __init__(self, model_name: str | None = None, cfg: ModelConfig | None = None) -> None:
        self.cfg = cfg or CONFIG.models
        model_name = model_name or self.cfg.embedding_model

        # 显式传入 device；torch.cuda 不可用时 SentenceTransformer 会对
        # "cuda" 抛错，这里先探测再决定，保证无 GPU 机器也能跑通全流程。
        import torch
        device = self.cfg.device if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("[embedder] 警告：未检测到 CUDA，使用 CPU 编码（速度慢一个量级）")

        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[embedder] 已加载 {model_name} | 维度 {self.dim} | 设备 {device}")

    def encode_corpus(self, texts: list[str]) -> np.ndarray:
        """文档块编码（入库时用）：批量、L2 归一化、float32。"""
        vecs = self.model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=True,   # 归一化后内积即余弦相似度
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        return vecs

    def encode_query(self, query: str, use_instruction: bool = True) -> np.ndarray:
        """单条查询编码（检索时用）：可选拼接指令前缀。"""
        text = QUERY_INSTRUCTION_ZH + query if use_instruction else query
        vec = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        return vec[0]
