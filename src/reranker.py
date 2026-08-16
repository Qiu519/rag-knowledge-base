"""交叉编码器重排（rerank）—— 检索质量的第二级放大器。

双塔 vs 交叉编码器：
    双塔（embedding）：query 和文档各自独立编码成向量再比对，
        可离线建库、在线毫秒级检索，但交互浅，精度有天花板；
    交叉编码器：把 (query, doc) 拼在一起送进模型逐 token 交叉注意力，
        精度显著更高，但必须逐对在线计算，无法预建索引。

所以工程上的标准打法是漏斗结构：
    稠密+BM25 召回 top-20（快而粗） → 交叉编码器精排留 top-5（慢而准）。
    重排只在 20 个候选上算，延迟可控（GPU 上几十毫秒）。
"""

from __future__ import annotations

from src.config import CONFIG, ModelConfig


class Reranker:
    """bge-reranker-base 封装。score 为相关性 logits，越大越相关。"""

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        self.cfg = cfg or CONFIG.models
        from sentence_transformers import CrossEncoder

        import torch
        device = self.cfg.device if torch.cuda.is_available() else "cpu"
        # bge-reranker-base 单语料场景 fp16 精度足够，减半显存、约提 1 倍吞吐
        self.max_length = 512
        self.model = CrossEncoder(
            self.cfg.reranker_model,
            device=device,
            max_length=self.max_length,
            automodel_args={"torch_dtype": torch.float16} if device == "cuda" else {},
        )
        print(f"[reranker] 已加载 {self.cfg.reranker_model} | 设备 {device}")

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """对候选列表重排，返回 top_k。

        保留原检索分数字段（rrf_score/score），新增 rerank_score，
        便于在界面上对比“重排前后排序变化”——这本身就是个好演示。
        """
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs, batch_size=16)
        ranked = sorted(
            ({**c, "rerank_score": float(s)} for c, s in zip(candidates, scores)),
            key=lambda x: x["rerank_score"], reverse=True,
        )
        return ranked[:top_k]
