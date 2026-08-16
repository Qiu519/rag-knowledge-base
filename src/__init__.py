"""RAG 知识库问答系统源码包。

模块划分（按数据流顺序）：
    document_loader  文档加载与清洗（PDF / Markdown / TXT）
    chunker          分块策略（fixed / recursive / sentence）
    embedder         文本向量化（bge 系列，GPU 加速）
    vector_store     FAISS 向量索引与元数据持久化
    retriever        混合检索（稠密向量 + BM25，RRF 融合）
    reranker         交叉编码器重排（bge-reranker）
    llm              大模型生成（OpenAI 兼容 API / 离线抽取式回退）
    pipeline         端到端编排（入库 ingest / 问答 query）
    evaluate         检索质量评估（Recall@k / MRR / 延迟）
    config           全局配置（所有可调参数集中于此）
"""
