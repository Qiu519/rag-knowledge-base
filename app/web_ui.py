"""Gradio Web 界面 —— 演示与体验入口。

布局：
    左侧  参数面板（top-k / 重排开关 / 混合检索开关 / 上传文档重建索引）
    右侧  问答区（答案 Markdown + 引用片段表格 + 各阶段耗时）

启动：python app/web_ui.py   然后浏览器打开 http://127.0.0.1:7860
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr

from src.config import CONFIG
from src.pipeline import RAGPipeline

PIPELINE: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """进程内单例：模型与索引只在首次调用时加载（约十几秒）。"""
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = RAGPipeline().load()
    return PIPELINE


def answer_question(question: str, top_k: int, use_rerank: bool,
                    use_hybrid: bool) -> tuple[str, str, str]:
    """问答回调：返回 (答案Markdown, 引用表格CSV串, 耗时说明)。"""
    if not question.strip():
        return "请在上方输入问题。", "", ""
    pipeline = get_pipeline()
    # 参数即时生效（同一配置对象，检索器与管线共享引用）
    pipeline.cfg.retrieval.use_hybrid = use_hybrid
    if use_rerank and pipeline.reranker is None:
        from src.reranker import Reranker
        pipeline.reranker = Reranker()  # 首次勾选重排时才加载模型

    result = pipeline.query(question, top_k=top_k)

    cite_rows = ["\t来源\t章节\t得分\t内容摘要"]
    for i, c in enumerate(result.contexts, start=1):
        score = c.get("rerank_score", c.get("rrf_score", c.get("score", 0)))
        snippet = c["text"][:60].replace("\n", " ") + "…"
        heading = c.get("heading", "")
        cite_rows.append(f"[{i}]\t{c['source']}\t{heading}\t{score:.3f}\t{snippet}")

    t = result.timings_ms
    timing = (f"检索 **{t.get('retrieval_ms')}ms** · 重排 **{t.get('rerank_ms')}ms** · "
              f"生成 **{t.get('generate_ms')}ms** · 总计 **{t.get('total_ms')}ms** · 模式 `{result.mode}`")
    return result.answer, "\n".join(cite_rows), timing


def rebuild_index(files: list[str]) -> str:
    """上传文档入库回调：存到 data/raw/uploads/ 后全量重建索引。"""
    if not files:
        return "未选择文件。"
    upload_dir = CONFIG.paths.raw_docs_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy(f, upload_dir / Path(f).name)
    n = get_pipeline().ingest()
    return f"已入库 {len(files)} 个文件，当前索引共 {n} 块。"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="本地知识库问答系统") as demo:
        gr.Markdown(
            "# 📚 本地知识库问答（RAG）\n"
            "基于本地 GPU 的中文文档问答：混合检索 + 重排 + 大模型生成，"
            "答案附引用来源。语料放在 `data/raw/`，界面左侧可上传补充。")
        with gr.Row():
            with gr.Column(scale=1):
                top_k = gr.Slider(1, 10, value=CONFIG.retrieval.final_k, step=1,
                                  label="返回片段数 top-k")
                use_rerank = gr.Checkbox(value=CONFIG.retrieval.use_rerank, label="启用重排（bge-reranker）")
                use_hybrid = gr.Checkbox(value=CONFIG.retrieval.use_hybrid, label="混合检索（BM25+向量）")
                gr.Markdown("### 📄 上传文档（md/txt/pdf）")
                uploader = gr.File(file_count="multiple",
                                   file_types=[".md", ".txt", ".pdf"])
                rebuild_btn = gr.Button("重建索引")
                rebuild_status = gr.Markdown()
                rebuild_btn.click(rebuild_index, inputs=uploader, outputs=rebuild_status)
            with gr.Column(scale=2):
                question = gr.Textbox(label="问题", placeholder="例如：保研对绩点有什么要求？",
                                      lines=2)
                ask_btn = gr.Button("提问", variant="primary")
                answer_md = gr.Markdown(label="回答")
                timing_md = gr.Markdown()
                cites = gr.Textbox(label="引用片段", lines=10)
                ask_btn.click(answer_question,
                              inputs=[question, top_k, use_rerank, use_hybrid],
                              outputs=[answer_md, cites, timing_md])
                question.submit(answer_question,
                                inputs=[question, top_k, use_rerank, use_hybrid],
                                outputs=[answer_md, cites, timing_md])
    return demo


if __name__ == "__main__":
    build_ui().queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
