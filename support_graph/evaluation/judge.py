"""LLM-as-a-judge for reference-free RAG evaluation (The RAG Triad)."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from support_graph.providers import build_chat_model
from support_graph.config.runtime import RuntimeConfig


@dataclass(frozen=True)
class JudgeResult:
    score: float
    reason: str


class RAGJudge:
    """Judge for evaluating RAG outputs using an LLM."""

    def __init__(self, chat_model: BaseChatModel):
        self.chat_model = chat_model

    @classmethod
    def from_config(
        cls, config: RuntimeConfig, model_name: str | None = None
    ) -> RAGJudge:
        """Build a judge from a runtime config."""
        if model_name:
            config = replace(config, chat_model=model_name)
        model = build_chat_model(config)
        return cls(model)

    async def _evaluate(self, system_prompt: str, user_prompt: str) -> JudgeResult:
        messages = [
            SystemMessage(
                content=system_prompt
                + ' Reply in concise JSON: {"score": float, "reason": short_string}'
            ),
            HumanMessage(content=user_prompt),
        ]
        response = await self.chat_model.ainvoke(messages)
        content = str(response.content).strip()

        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "{" in content:
                content = content[content.find("{") : content.rfind("}") + 1]

            data = json.loads(content)
            return JudgeResult(
                score=float(data.get("score", 0.0)), reason=data.get("reason", "N/A")
            )
        except (ValueError, json.JSONDecodeError, IndexError):
            return JudgeResult(score=0.0, reason="Parse Error")

    async def faithfulness(self, context: str, answer: str) -> JudgeResult:
        """Is answer derived ONLY from context? (0-1)"""
        system = "Judge FAITHFULNESS: Does the answer only use provided context? (1=Yes, 0=Hallucination)"
        user = (
            f"CTX: {context[:1500]}\n\nANS: {answer}"  # Truncate context to save tokens
        )
        return await self._evaluate(system, user)

    async def answer_relevance(self, query: str, answer: str) -> JudgeResult:
        """Does answer address query? (0-1)"""
        system = "Judge RELEVANCE: Does this answer actually address the user query?"
        user = f"QS: {query}\n\nANS: {answer}"
        return await self._evaluate(system, user)

    async def context_relevance(self, query: str, context: str) -> JudgeResult:
        """Is context useful for query? (0-1)"""
        system = "Judge CONTEXT: Does this context contain facts to answer the query?"
        user = (
            f"QS: {query}\n\nCTX: {context[:1500]}"  # Truncate context to save tokens
        )
        return await self._evaluate(system, user)

    async def evaluate_triad(
        self, query: str, context: str, answer: str
    ) -> dict[str, JudgeResult]:
        import asyncio

        # We can run these in sequence to manage rate limits or in parallel for speed
        results = await asyncio.gather(
            self.faithfulness(context, answer),
            self.answer_relevance(query, answer),
            self.context_relevance(query, context),
        )
        return {
            "faithfulness": results[0],
            "answer_relevance": results[1],
            "context_relevance": results[2],
        }
