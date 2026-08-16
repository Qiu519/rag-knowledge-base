"""大模型生成层 —— 两级回退的答案生成。

三种运行模式（按配置自动选择）：
    api        .env 配置了 LLM_BASE_URL/KEY/MODEL 时启用。
               OpenAI 兼容协议，DeepSeek / 硅基流动 / Ollama 通吃。
    extractive 未配置 API 时的回退：不联网，把 top 检索结果整理成
               带编号引用的“草稿答案”。用于无 Key 联调和演示，
               它同时是评估环节里“纯检索无生成”的基线。

Prompt 设计要点（防幻觉三件套）：
    1. 明确指令“只依据给定资料回答”，禁止调动模型自身知识；
    2. 要求引用编号 [n]，答案可溯源到具体片段；
    3. 资料中没有答案时必须直说“资料中未提及”——
       宁可拒答，不可编造（RAG 场景最忌一本正经地胡说）。
"""

from __future__ import annotations

import time

from src.config import CONFIG, LLMConfig

SYSTEM_PROMPT = (
    "你是一个严谨的知识库问答助手。你只能依据下面提供的编号资料回答问题，"
    "禁止使用资料以外的知识。回答要求：\n"
    "1. 用中文回答，先给结论，再作简要说明；\n"
    "2. 引用依据时标注资料编号，如 [1][3]；\n"
    "3. 如果资料中没有足够信息回答问题，明确回答“根据现有资料无法回答该问题”，"
    "不要猜测或编造。"
)


def build_context(candidates: list[dict]) -> str:
    """把检索结果拼装成带编号的上下文文本。"""
    parts = []
    for i, c in enumerate(candidates, start=1):
        heading = f"[{c['heading']}] " if c.get("heading") else ""
        parts.append(f"[{i}] （来源：{c['source']}）{heading}{c['text']}")
    return "\n\n".join(parts)


class LLMClient:
    """生成客户端。对外只暴露 answer()，内部自动选择模式。"""

    def __init__(self, cfg: LLMConfig | None = None) -> None:
        self.cfg = cfg or CONFIG.llm
        self._client = None
        if self.cfg.api_ready:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.cfg.base_url,
                api_key=self.cfg.api_key,
                timeout=60,
            )
            print(f"[llm] API 模式：{self.cfg.base_url} | 模型 {self.cfg.model}")
        else:
            print("[llm] 未配置 API，使用抽取式回退模式（不调用外部服务）")

    @property
    def mode(self) -> str:
        return "api" if self._client else "extractive"

    # ---- API 模式 ----

    def _chat(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.cfg.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        return resp.choices[0].message.content or ""

    # ---- 抽取式模式 ----

    @staticmethod
    def _extractive_answer(question: str, candidates: list[dict]) -> str:
        """无 API 回退：直接整理最相关片段作为“草稿答案”。

        说明：这不是真正的生成，只是把 top-3 片段按相关度罗列。
        它的价值在于（a）无 Key 时全链路仍可跑通；（b）作为
        “检索质量”的裸基线，与 LLM 生成效果形成对照。
        """
        if not candidates:
            return ("（未检索到相关内容。请先运行 scripts/ingest.py 构建索引，"
                    "或检查问题是否超出语料范围。）")
        top = candidates[:3]
        lines = [f"【抽取式模式 · 未配置 LLM API】问题：{question}",
                 "", "以下是与问题最相关的原文片段（按相关度排序）：", ""]
        for i, c in enumerate(top, start=1):
            heading = f"[{c['heading']}] " if c.get("heading") else ""
            lines.append(f"[{i}] {heading}{c['text']}")
            lines.append(f"    —— 来源：{c['source']}")
            lines.append("")
        lines.append("提示：在 .env 中配置 LLM API 后可获得连贯的生成式回答。")
        return "\n".join(lines)

    # ---- 对外入口 ----

    def answer(self, question: str, candidates: list[dict]) -> tuple[str, float]:
        """基于检索结果生成回答。返回 (回答文本, 生成耗时秒)。"""
        t0 = time.perf_counter()
        if self._client:
            context = build_context(candidates)
            user_prompt = f"问题：{question}\n\n编号资料：\n{context}"
            text = self._chat(SYSTEM_PROMPT, user_prompt)
        else:
            text = self._extractive_answer(question, candidates)
        return text, time.perf_counter() - t0
