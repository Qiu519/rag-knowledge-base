# 本地知识库问答系统（RAG）

> 基于 RTX 4060 本地 GPU 的中文文档问答：混合检索 + 交叉编码器重排 + 大模型生成，
> 答案带引用溯源，并配套 30 题评测集与四组对照实验。数据科学专业课程作品 / 实习项目。

## 效果一览

- **检索质量**（30 题人工评测集，来源级判定）：见 [docs/实验报告.md](docs/实验报告.md)
- **响应延迟**：检索+重排 <100ms（本地 GPU），生成 1-3s（API）
- **全离线可用**：embedding / 重排全部本地推理；未配置 LLM Key 时自动降级抽取式回答

## 快速开始

```bash
# 1. 环境：conda 创建 py3.11，装 GPU 版 torch（详见 docs/开发手册.md）
conda create -n rag python=3.11 -y
pip install <torch-cu121-wheel-url>       # 见开发手册 §2.3
pip install -r requirements.txt

# 2. 模型：下载三个本地模型（bge-base / bge-small / bge-reranker）
python scripts/download_models.py

# 3. 配置 LLM（可选）：复制 .env.example 为 .env 填入 Key
#    支持 DeepSeek / 硅基流动 / 本地 Ollama（OpenAI 兼容协议）

# 4. 跑起来
python scripts/ingest.py                   # data/raw/ 的语料建入索引
python scripts/ask.py "保研对绩点有什么要求？"
python app/web_ui.py                       # Web 界面 → http://127.0.0.1:7860
```

## 技术架构

```
入库：文档(md/txt/pdf) → 清洗 → 层级分块 → GPU 向量化(bge-base-zh) → FAISS
问答：问题 → 向量top20 + BM25top20 → RRF融合 → bge-reranker精排 → top5
           → LLM 生成带引用回答（防幻觉 prompt 约束）
```

特性：

- **两级检索漏斗**：双塔召回（快而粗）→ 交叉编码器精排（慢而准）
- **混合检索**：向量懂语义改写，BM25 稳术语命中，RRF 无参融合
- **评估驱动**：HitRate@k / Recall@k / MRR / 延迟四指标，控制变量实验定位每个组件的贡献
- **工程细节**：块 ID 稳定化（实验可复现）、模型缓存重定向（不污染系统盘）、无 Key 回退

## 项目结构

```
src/        核心管线（加载/分块/向量化/索引/检索/重排/生成/评估/配置）
scripts/    CLI：ingest 入库 · ask 问答 · run_experiments 实验 · download_models
app/        Gradio Web 界面
data/raw/   语料（示例：12 篇高校培养方案文档，可直接替换为自己的资料）
data/eval/  30 题评测集（jsonl：问题 + 相关来源 + 参考答案）
tests/      分块器单元测试
docs/       开发手册 · 架构设计 · 实验报告
```

## 文档

- [开发手册](docs/开发手册.md)——环境复现、配置系统、踩坑记录
- [架构设计](docs/架构设计.md)——数据流、设计决策与取舍
- [实验报告](docs/实验报告.md)——四组对照实验的数据与结论

## 环境

Python 3.11 · PyTorch 2.5.1+cu121 · sentence-transformers 3.3 · FAISS 1.15 · Gradio 5
GPU：NVIDIA RTX 4060 Laptop（8GB），无 GPU 自动回退 CPU
