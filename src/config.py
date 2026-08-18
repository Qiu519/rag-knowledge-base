"""全局配置模块。

设计原则：
    1. 所有可调参数集中在此，改配置不需要翻代码；
    2. 路径全部基于项目根目录推导，项目整体拷走也能跑；
    3. 敏感信息（API Key）从 .env 读取，绝不硬编码；
    4. HF_HOME 在此统一重定向到项目内的 S 盘缓存——必须在
       transformers/huggingface_hub 导入之前设置才会生效，
       所以任何模块要“先 import config，再 import transformers”。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# HuggingFace 模型缓存重定向到项目目录（S 盘），避免默认写入 C 盘用户目录。
# 注意：必须先于 transformers 相关 import 执行——本模块被 import 时即生效。
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf-home"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内镜像站
# 本机 VPN 以系统代理方式运行（127.0.0.1:17892），Python 的 requests 会读取
# 注册表代理去连 hf-mirror，握手会失败；直连反而通畅，故显式绕过代理。
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")
# Windows 未开开发者模式时 hf 缓存不支持符号链接，仅是提示性警告，静默之
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# 加载 .env（若存在）。里面放 LLM_API_KEY 等敏感配置。
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class PathConfig:
    """数据与产物的目录规划。

    raw       存放原始文档（用户要问答的资料，md/txt/pdf）
    processed 存放分块结果与 FAISS 索引
    eval      存放评测集与实验产出
    """

    raw_docs_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    index_dir: Path = PROJECT_ROOT / "data" / "processed" / "index"
    eval_dir: Path = PROJECT_ROOT / "data" / "eval"
    outputs_dir: Path = PROJECT_ROOT / "outputs"

    def ensure(self) -> None:
        """创建全部目录（幂等）。"""
        for d in (self.raw_docs_dir, self.processed_dir, self.index_dir,
                  self.eval_dir, self.outputs_dir):
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class ChunkConfig:
    """分块参数。

    chunk_size 以“字符数”计（中文场景按字符比按词直观）：
        - 过大 → 单块主题混杂，向量表达被稀释，检索精度下降；
        - 过小 → 上下文割裂，召回的片段不足以支撑回答；
        400 字符（约一至两段）是中文文档的常用起点，实验环节会对比不同取值。
    overlap: 相邻块重叠字符数，防止答案恰好被切断在边界上。
    strategy: fixed（纯滑窗）/ recursive（按标题-段落-句子层级切）/ sentence（整句打包）。
    """

    strategy: str = "recursive"
    chunk_size: int = 400
    overlap: int = 60


@dataclass
class RetrievalConfig:
    """检索参数。

    管线是两级漏斗：
        第一级 候选召回  dense_k / bm25_k 各自取较多候选；
        第二级 精排      rerank 后只留 final_k 条给大模型。
    use_hybrid: 是否启用 BM25+向量混合检索（RRF 融合）。
                向量检索擅长语义改写，BM25 擅长术语/编号的精确匹配，互补。
    rrf_k:      RRF 公式 1/(rrf_k+rank) 的平滑常数，60 是论文常用值。
    """

    dense_k: int = 20
    bm25_k: int = 20
    final_k: int = 7   # 送入生成的片段数。5 会漏掉"差一名"的关键块（实测案例：
                       # 保研问题下"基本申请条件"块常排第 6），7 能覆盖且 token 成本可控
    use_hybrid: bool = True
    use_rerank: bool = True
    rrf_k: int = 60


@dataclass
class ModelConfig:
    """本地模型选型（都跑在 GPU 上）。

    embedding: bge-base-zh-v1.5，中文检索代表性模型，768 维。
               （实验环节会换 bge-small-zh-v1.5 做对比）
    reranker:  bge-reranker-base，交叉编码器，对 query-doc 对打分。
    """

    embedding_model: str = "BAAI/bge-base-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    device: str = "cuda"  # 无 GPU 时 Embedder 会自动回退到 cpu
    batch_size: int = 64


@dataclass
class LLMConfig:
    """生成端配置。

    三个环境变量都没配时进入 extractive（抽取式）模式：
    不调用任何外部服务，直接把检索结果整理成带引用的草稿答案。
    用途：无 Key 联调、演示检索质量；正式评估与生产必须配 API。
    """

    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", ""))
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    temperature: float = 0.3  # 事实型问答要“稳”，温度调低
    max_tokens: int = 1024
    # 思考型模型（如 qwen3.7-max）的推理开关：RAG 场景上下文已给足依据，
    # 关闭思考可把生成耗时从约 6-12s 压到 1-2s，事实型回答质量几乎无损；
    # 复杂推理题可在 .env 里设 LLM_ENABLE_THINKING=true。
    # 默认 false：对非思考型模型该字段会被忽略，无副作用。
    enable_thinking: bool = field(default_factory=lambda: (
        os.getenv("LLM_ENABLE_THINKING", "false").lower() in {"1", "true", "yes"}))

    @property
    def api_ready(self) -> bool:
        """三者齐备才认为 API 可用。"""
        return bool(self.base_url and self.api_key and self.model)


@dataclass
class AppConfig:
    """汇总配置。用法：`from src.config import CONFIG`。"""

    paths: PathConfig = field(default_factory=PathConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    def __post_init__(self) -> None:
        self.paths.ensure()


# 版本标识（开发日志见 docs/开发手册.md §8）。界面与文档统一引用，便于辨识当前能力。
# v0.4.0–v0.4.9 的表格统计旁路已移除，系统回到纯文档问答 RAG 核心（基线 v0.3.0）。
PROJECT_VERSION = "v0.3.0"

# 全局单例：整个项目共享一份配置
CONFIG = AppConfig()
