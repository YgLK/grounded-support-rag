from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from langchain_core.prompts import ChatPromptTemplate

from support_graph.config.runtime import RuntimeConfig
from support_graph.runtime import nodes
from support_graph.runtime.llm_policy import LLMCallTimeoutError
from support_graph.runtime.prompts import PromptSet
from support_graph.runtime.schemas import GraphState, Runtime, fallback_metadata


def _prompt_set() -> PromptSet:
    template = ChatPromptTemplate.from_messages([("system", "test"), ("human", "{x}")])
    return PromptSet(
        version="v-test",
        route_query=template,
        evidence_grade=template,
        answer=template,
        non_answer=template,
        streaming_answer=template,
    )


def _runtime(*, chat_model: object, llm_timeout_seconds: float = 60.0) -> Runtime:
    config = RuntimeConfig(
        chat_model="chat-model",
        llm_max_retries=1,
        llm_timeout_seconds=llm_timeout_seconds,
        llm_retry_base_delay_seconds=0.0,
        llm_retry_max_delay_seconds=0.0,
    )
    return Runtime(
        config=config,
        vectorstore=None,
        keyword_retriever=None,
        chat_model=chat_model,
        chunk_records_by_doc={},
        prompts=_prompt_set(),
        trace_path=Path("/tmp/run-test.jsonl"),
        run_id="run-test",
    )


def _state(*, verdict: str = "partial") -> GraphState:
    return {
        "example_id": "kubernetes::ex::turn_1",
        "domain": "kubernetes",
        "conversation": [
            {"role": "user", "utterance": "My deployment stalled so what should I do?"}
        ],
        "latest_user_utterance": "My deployment stalled so what should I do?",
        "retrieved_chunks": [
            {
                "chunk_id": "kubernetes::doc::sec::2::sub::0",
                "doc_id": "doc",
                "section_id": "2",
                "section_title": "Deployment rollout guidance",
                "span_ids": ["2", "3"],
                "text": "Restore coverage immediately to avoid unavailable replicas.",
                "token_count": 12,
            }
        ],
        "evidence_grade": {
            "verdict": verdict,
            "reason": "needs review",
            "missing_information": ["rollout status"] if verdict == "partial" else [],
        },
    }


def test_query_example_from_state_uses_domain_field() -> None:
    query_example = nodes._query_example_from_state(_state())

    assert query_example == {
        "domain": "kubernetes",
        "conversation": [
            {"role": "user", "utterance": "My deployment stalled so what should I do?"}
        ],
        "latest_user_turn_id": None,
        "latest_user_utterance": "My deployment stalled so what should I do?",
    }


def test_ainvoke_structured_prompt_rejects_non_mapping_results() -> None:
    class FakePrompt:
        def __or__(self, other):
            return other

    class FakeStructuredChain:
        async def ainvoke(self, payload):
            return "not-a-mapping"

    class FakeChatModel:
        def with_structured_output(self, schema, method="json_schema"):
            return FakeStructuredChain()

    with pytest.raises(TypeError, match="Expected structured output"):
        asyncio.run(
            nodes._ainvoke_structured_prompt(
                runtime=_runtime(chat_model=FakeChatModel()),
                prompt=FakePrompt(),
                schema=nodes.ResponseModel,
                payload={"latest_user_utterance": "deployment stalled"},
            )
        )


def test_grade_evidence_falls_back_when_structured_output_parse_fails(
    monkeypatch,
) -> None:
    async def broken_structured_prompt(**kwargs):
        raise json.JSONDecodeError("Expecting value", "", 0)

    monkeypatch.setattr(nodes.logger, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        nodes,
        "_ainvoke_structured_prompt",
        broken_structured_prompt,
        raising=False,
    )

    result = asyncio.run(
        nodes.grade_evidence(
            state=_state(verdict="partial"),
            runtime=_runtime(chat_model=object()),
        )
    )

    fallback = fallback_metadata(result)
    assert result["verdict"] == "sufficient"
    assert fallback is not None
    assert fallback["node"] == "grade_evidence"
    assert fallback["exception_type"] == "JSONDecodeError"
    assert "Expecting value" in fallback["error"]


def test_ainvoke_structured_prompt_raises_timeout_without_retrying() -> None:
    class FakePrompt:
        def __or__(self, other):
            return other

    class SlowStructuredChain:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, payload, config=None):
            del payload, config
            self.calls += 1
            await asyncio.sleep(0.02)
            return {
                "decision": "answer",
                "response_text": "late",
                "citation_chunk_ids": [],
                "confidence_label": "low",
            }

    class SlowChatModel:
        def __init__(self) -> None:
            self.chain = SlowStructuredChain()

        def with_structured_output(self, schema, method="json_schema"):
            del schema, method
            return self.chain

    chat_model = SlowChatModel()

    with pytest.raises(LLMCallTimeoutError, match="timed out after"):
        asyncio.run(
            nodes._ainvoke_structured_prompt(
                runtime=_runtime(chat_model=chat_model, llm_timeout_seconds=0.001),
                prompt=FakePrompt(),
                schema=nodes.ResponseModel,
                payload={"latest_user_utterance": "deployment stalled"},
            )
        )

    assert chat_model.chain.calls == 1


def test_generate_response_falls_back_when_structured_output_fails(
    monkeypatch,
) -> None:
    async def broken_structured_prompt(**kwargs):
        raise TypeError("Expected structured output as BaseModel or mapping, got str.")

    monkeypatch.setattr(nodes.logger, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        nodes,
        "_ainvoke_structured_prompt",
        broken_structured_prompt,
        raising=False,
    )

    result = asyncio.run(
        nodes.generate_response(
            state=_state(verdict="sufficient"),
            runtime=_runtime(chat_model=object()),
        )
    )

    fallback = fallback_metadata(result)
    assert result["decision"] == "answer"
    assert result["citation_chunk_ids"] == ["kubernetes::doc::sec::2::sub::0"]
    assert fallback is not None
    assert fallback["node"] == "generate_response"
    assert fallback["exception_type"] == "TypeError"


def test_generate_response_propagates_timeout_errors(monkeypatch) -> None:
    async def timeout_structured_prompt(**kwargs):
        del kwargs
        raise LLMCallTimeoutError(1.0)

    monkeypatch.setattr(
        nodes,
        "_ainvoke_structured_prompt",
        timeout_structured_prompt,
        raising=False,
    )

    with pytest.raises(LLMCallTimeoutError, match="1.0s"):
        asyncio.run(
            nodes.generate_response(
                state=_state(verdict="sufficient"),
                runtime=_runtime(chat_model=object()),
            )
        )


def test_resolve_without_answer_falls_back_when_structured_output_fails(
    monkeypatch,
) -> None:
    async def broken_structured_prompt(**kwargs):
        raise RuntimeError("schema validation failed")

    monkeypatch.setattr(nodes.logger, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        nodes,
        "_ainvoke_structured_prompt",
        broken_structured_prompt,
        raising=False,
    )

    result = asyncio.run(
        nodes.resolve_without_answer(
            state=_state(verdict="partial"),
            runtime=_runtime(chat_model=object()),
        )
    )

    fallback = fallback_metadata(result)
    assert result["decision"] == "clarify"
    assert result["citation_chunk_ids"] == ["kubernetes::doc::sec::2::sub::0"]
    assert fallback is not None
    assert fallback["node"] == "resolve_without_answer"
    assert fallback["exception_type"] == "RuntimeError"
