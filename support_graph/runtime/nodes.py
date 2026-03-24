"""Runtime node implementations and shared helpers."""

from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

from langchain_community.retrievers import BM25Retriever
from pydantic import BaseModel

from support_graph.config.runtime import RuntimeConfig
from support_graph.logging_utils import get_logger
from support_graph.providers import build_chat_model as build_provider_chat_model
from support_graph.providers import (
    chat_provider as resolved_chat_provider,
    embedding_provider as resolved_embedding_provider,
)
from support_graph.retrieval.index import load_chunk_records
from support_graph.retrieval.retrieve import build_legacy_query
from support_graph.retrieval.retrieve import build_query as build_retrieval_query
from support_graph.retrieval.retrieve import (
    build_keyword_retriever,
    build_query_context,
    get_vectorstore,
    retrieve_chunks,
)
from support_graph.runtime.llm_policy import (
    LLMCallTimeoutError,
    ainvoke_with_retry,
)
from support_graph.runtime.observability import build_observability
from support_graph.runtime.prompts import resolve_prompt_set
from support_graph.runtime.schemas import (
    EvidenceGradeModel,
    FallbackResponseModel,
    GraphState,
    GraphStreamEvent,
    ResponseModel,
    RouteModel,
    Runtime,
    RuntimeResources,
    attach_fallback_metadata,
)
from support_graph.runtime.traces import write_trace_event


logger = get_logger(__name__)
QueryMode = Literal["legacy_transcript", "latest_user_only", "structured"]


def build_chat_model(
    config: RuntimeConfig,
    chat_model_cls: type[Any] | None = None,
) -> Any:
    return build_provider_chat_model(
        config,
        chat_model_cls=chat_model_cls,
    )


async def _emit_graph_event(runtime: Runtime, event: GraphStreamEvent) -> None:
    if runtime.event_sink is None:
        return
    result = runtime.event_sink(event)
    if inspect.isawaitable(result):
        await result


async def _trace(runtime: Runtime, node: str, event: dict) -> None:
    await asyncio.to_thread(
        write_trace_event,
        runtime.trace_path,
        {
            "node": node,
            **event,
        },
    )


def _response_started_event(
    *,
    state: GraphState,
    runtime: Runtime,
    node_name: str,
    decision: str | None,
) -> GraphStreamEvent:
    return {
        "kind": "response_started",
        "node": node_name,
        "run_id": runtime.run_id,
        "example_id": state.get("example_id"),
        "decision": decision,
    }


def _response_deltas(text: str) -> list[str]:
    stripped = str(text or "")
    if not stripped:
        return []
    deltas = re.findall(r"\S+\s*", stripped)
    return deltas or [stripped]


def _stream_item_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping):
        return str(item.get("text") or item.get("content") or "")
    text = getattr(item, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(item, "content", None)
    if isinstance(content, str):
        return content
    raise TypeError(f"Unsupported stream item type: {type(item).__name__}")


def _stream_chunk_text(chunk: Any) -> str:
    if chunk is None:
        return ""
    if isinstance(chunk, str):
        return chunk
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        return str(content.get("text") or content.get("content") or "")
    if isinstance(content, list):
        return "".join(_stream_item_text(item) for item in content)
    text = getattr(chunk, "text", None)
    if isinstance(text, str):
        return text
    raise TypeError(f"Unsupported stream chunk type: {type(chunk).__name__}")


async def _emit_buffered_response_preview(
    *,
    state: GraphState,
    runtime: Runtime,
    node_name: str,
    payload: dict[str, Any],
) -> None:
    if not runtime.stream_responses:
        return

    await _emit_graph_event(
        runtime,
        _response_started_event(
            state=state,
            runtime=runtime,
            node_name=node_name,
            decision=payload.get("decision"),
        ),
    )
    for delta in _response_deltas(str(payload.get("response_text", ""))):
        await _emit_graph_event(
            runtime,
            {
                "kind": "response_delta",
                "node": node_name,
                "run_id": runtime.run_id,
                "example_id": state.get("example_id"),
                "delta": delta,
            },
        )


async def _emit_streaming_answer_preview(
    *,
    state: GraphState,
    runtime: Runtime,
    node_name: str,
    payload: dict[str, Any],
) -> str | None:
    if not runtime.stream_responses or runtime.chat_model is None:
        return None
    if payload.get("decision") != "answer":
        return None

    astream = getattr(runtime.chat_model, "astream", None)
    if not callable(astream):
        return None

    await _emit_graph_event(
        runtime,
        _response_started_event(
            state=state,
            runtime=runtime,
            node_name=node_name,
            decision=payload.get("decision"),
        ),
    )
    messages = runtime.prompts.streaming_answer.format_messages(
        conversation=_render_conversation(state.get("conversation", [])),
        latest_user_utterance=state.get("latest_user_utterance") or "",
        evidence_grade=state.get("evidence_grade", {}),
        retrieved_chunks=_render_chunks(_reasoning_chunks(state)),
    )
    deltas: list[str] = []
    try:
        async for chunk in astream(messages):
            delta = _stream_chunk_text(chunk)
            if not delta:
                continue
            deltas.append(delta)
            await _emit_graph_event(
                runtime,
                {
                    "kind": "response_delta",
                    "node": node_name,
                    "run_id": runtime.run_id,
                    "example_id": state.get("example_id"),
                    "delta": delta,
                },
            )
    except Exception as exc:
        logger.warning(
            "%s preview streaming fell back to buffered output: %s",
            node_name,
            exc,
            exc_info=exc,
        )
        return None

    streamed_text = "".join(deltas).strip()
    return streamed_text or None


def _render_conversation(conversation: list[dict], max_turns: int = 8) -> str:
    lines = []
    for turn in conversation[-max_turns:]:
        role = str(turn.get("role", "")).title()
        utterance = str(turn.get("utterance", "")).strip()
        if utterance:
            lines.append(f"{role}: {utterance}")
    return "\n".join(lines)


def _render_chunks(chunks: list[dict], limit: int = 8) -> str:
    parts = []
    for chunk in chunks[:limit]:
        title = str(chunk.get("section_title") or "").strip()
        title_line = f"Title: {title}\n" if title else ""
        parts.append(
            f'<chunk id="{chunk.get("chunk_id")}" doc="{chunk.get("doc_id")}">\n'
            f"{title_line}"
            f"Text: {chunk.get('text', '')}\n"
            "</chunk>"
        )
    return "\n".join(parts)


def _as_payload(value: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(
        f"Expected structured output as BaseModel or mapping, got {type(value).__name__}."
    )


async def _ainvoke_structured_prompt(
    *,
    runtime: Runtime,
    prompt: Any,
    schema: type[BaseModel],
    payload: dict[str, Any],
) -> dict[str, Any]:
    chain = prompt | runtime.chat_model.with_structured_output(
        schema, method="json_schema"
    )

    async def invoke_chain() -> Any:
        try:
            return await chain.ainvoke(
                payload,
                config={"run_name": f"support_graph.{schema.__name__}"},
            )
        except TypeError as exc:
            if "unexpected keyword argument 'config'" not in str(exc):
                raise
            return await chain.ainvoke(payload)

    result = await ainvoke_with_retry(
        invoke_chain,
        timeout_seconds=getattr(runtime.config, "llm_timeout_seconds", None),
        max_attempts=runtime.config.llm_max_retries,
        base_delay_seconds=runtime.config.llm_retry_base_delay_seconds,
        max_delay_seconds=runtime.config.llm_retry_max_delay_seconds,
    )
    return _as_payload(result)


def _log_llm_fallback(node_name: str, exc: Exception) -> None:
    logger.warning(
        "%s falling back to heuristics after structured LLM output failed: %s",
        node_name,
        exc,
        exc_info=exc,
    )


def _heuristic_evidence_grade(retrieved_chunks: list[dict]) -> dict:
    if not retrieved_chunks:
        return {
            "verdict": "insufficient",
            "reason": "No relevant documentation was retrieved.",
            "missing_information": [],
        }
    best_chunk = _best_retrieved_chunk(retrieved_chunks)
    if best_chunk is not None and _is_substantive_chunk(best_chunk):
        return {
            "verdict": "sufficient",
            "reason": "A substantive retrieved chunk supports a grounded next-step answer.",
            "missing_information": [],
        }
    return {
        "verdict": "partial",
        "reason": "Retrieved evidence needs model review before answering safely.",
        "missing_information": [],
    }


def _answer_payload(chunk: dict) -> dict:
    chunk_id = chunk.get("chunk_id")
    return {
        "decision": "answer",
        "response_text": chunk.get("text", "").strip()
        or "Relevant documentation was found.",
        "citation_chunk_ids": [chunk_id] if chunk_id else [],
        "confidence_label": "medium",
    }


def _clarify_payload(chunks: list[dict], missing: str) -> dict:
    return {
        "decision": "clarify",
        "response_text": f"Need one detail before answering: {missing}.",
        "citation_chunk_ids": [chunks[0].get("chunk_id")] if chunks else [],
        "confidence_label": "low",
    }


def _abstain_payload() -> dict:
    return {
        "decision": "abstain",
        "response_text": "No sufficient support was found in the indexed documentation.",
        "citation_chunk_ids": [],
        "confidence_label": "low",
    }


def _heuristic_response(state: GraphState) -> dict:
    grade = state.get("evidence_grade", {})
    chunks = _reasoning_chunks(state)
    match grade.get("verdict"):
        case "sufficient":
            if not chunks:
                return _abstain_payload()
            chunk = _best_retrieved_chunk(chunks)
            if chunk is None:
                raise ValueError(
                    "Missing best retrieved chunk for sufficient evidence."
                )
            return _answer_payload(chunk)
        case "partial":
            missing = grade.get("missing_information") or ["one missing detail"]
            return _clarify_payload(chunks, missing[0])
        case "insufficient" | None:
            return _abstain_payload()
    raise ValueError(f"Unknown evidence verdict: {grade.get('verdict')}")


def _heuristic_non_answer_response(state: GraphState) -> dict:
    grade = state.get("evidence_grade", {})
    chunks = _reasoning_chunks(state)
    match grade.get("verdict"):
        case "partial":
            missing = grade.get("missing_information") or [
                "what condition changed in your DMV case"
            ]
            return _clarify_payload(chunks, missing[0])
        case "sufficient" | "insufficient" | None:
            return _abstain_payload()
    raise ValueError(f"Unknown evidence verdict: {grade.get('verdict')}")


def _query_example_from_state(state: GraphState) -> dict:
    return {
        "domain": state.get("domain"),
        "conversation": state.get("conversation", []),
        "latest_user_turn_id": state.get("latest_user_turn_id"),
        "latest_user_utterance": state.get("latest_user_utterance"),
    }


def _reasoning_chunks(state: GraphState) -> list[dict]:
    return list(state.get("retrieved_chunks", []))


def _llm_available(runtime: Runtime) -> bool:
    return runtime.chat_model is not None


def _ablation_bool(state: GraphState, key: str, default: bool) -> bool:
    ablation_options = state.get("ablation_options", {})
    if key in ablation_options:
        return bool(ablation_options[key])
    return default


def _query_mode(state: GraphState) -> QueryMode:
    mode = state.get("ablation_options", {}).get("query_mode", "structured")
    match mode:
        case "legacy_transcript" | "latest_user_only" | "structured":
            return cast(QueryMode, mode)
    raise ValueError(f"Unknown query_mode: {mode}")


def _is_title_chunk(chunk: dict) -> bool:
    return str(chunk.get("section_id", "")).startswith("t_")


def _content_only_reasoning_enabled(*, state: GraphState, runtime: Runtime) -> bool:
    return _ablation_bool(
        state,
        "content_only_reasoning",
        bool(runtime.config.content_only_reasoning),
    )


def _neighbor_expansion_enabled(*, state: GraphState, runtime: Runtime) -> bool:
    return _ablation_bool(
        state,
        "neighbor_expansion",
        bool(runtime.config.neighbor_expansion),
    )


def _content_only_chunks(chunks: list[dict]) -> list[dict]:
    filtered = [chunk for chunk in chunks if not _is_title_chunk(chunk)]
    return filtered or list(chunks)


def _normalize_chunk_record(chunk: dict) -> dict:
    return {
        "chunk_id": chunk.get("chunk_id"),
        "domain": chunk.get("domain"),
        "doc_id": chunk.get("doc_id"),
        "doc_title": chunk.get("doc_title"),
        "section_id": chunk.get("section_id"),
        "section_title": chunk.get("section_title"),
        "parent_titles": chunk.get("parent_titles", []),
        "span_ids": chunk.get("span_ids", []),
        "token_count": chunk.get("token_count"),
        "text": chunk.get("text", ""),
        "subchunk_index": chunk.get("subchunk_index"),
        "start_sec": chunk.get("start_sec"),
        "end_sec": chunk.get("end_sec"),
    }


def _numeric_section_id(section_id: Any) -> int | None:
    text = str(section_id or "").strip()
    if re.fullmatch(r"\d+", text):
        return int(text)
    return None


def expand_neighbor_sections(
    chunks: list[dict],
    chunk_records_by_doc: dict[str, list[dict]],
    *,
    limit: int = 8,
) -> list[dict]:
    """Expand direct hits with adjacent numeric sections from the same document."""
    if not chunks:
        return []
    if not chunk_records_by_doc:
        return list(chunks[:limit])

    direct_chunks_by_id = {
        chunk.get("chunk_id"): chunk for chunk in chunks if chunk.get("chunk_id")
    }
    doc_order: list[str] = []
    selected_chunk_ids_by_doc: dict[str, set[str]] = {}

    for chunk in chunks:
        doc_id = chunk.get("doc_id")
        chunk_id = chunk.get("chunk_id")
        if not doc_id:
            continue
        if doc_id not in selected_chunk_ids_by_doc:
            selected_chunk_ids_by_doc[doc_id] = set()
            doc_order.append(str(doc_id))
        if chunk_id:
            selected_chunk_ids_by_doc[doc_id].add(str(chunk_id))

    for anchor in chunks:
        if _is_title_chunk(anchor):
            continue
        doc_id = anchor.get("doc_id")
        section_number = _numeric_section_id(anchor.get("section_id"))
        if not doc_id or section_number is None:
            continue
        for candidate in chunk_records_by_doc.get(str(doc_id), []):
            candidate_section = _numeric_section_id(candidate.get("section_id"))
            candidate_chunk_id = candidate.get("chunk_id")
            if candidate_chunk_id and candidate_section in {
                section_number - 1,
                section_number + 1,
            }:
                selected_chunk_ids_by_doc.setdefault(str(doc_id), set()).add(
                    str(candidate_chunk_id)
                )

    expanded: list[dict] = []
    seen: set[str] = set()
    for doc_id in doc_order:
        doc_chunks = chunk_records_by_doc.get(doc_id, [])
        if not doc_chunks:
            for chunk in chunks:
                chunk_id = chunk.get("chunk_id")
                if chunk.get("doc_id") != doc_id or not chunk_id or chunk_id in seen:
                    continue
                expanded.append(chunk)
                seen.add(str(chunk_id))
                if len(expanded) >= limit:
                    return expanded
            continue

        selected_chunk_ids = selected_chunk_ids_by_doc.get(doc_id, set())
        for chunk in doc_chunks:
            chunk_id = chunk.get("chunk_id")
            if not chunk_id or chunk_id not in selected_chunk_ids or chunk_id in seen:
                continue
            expanded.append(
                direct_chunks_by_id.get(chunk_id, _normalize_chunk_record(chunk))
            )
            seen.add(str(chunk_id))
            if len(expanded) >= limit:
                return expanded
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk.get("doc_id") != doc_id or not chunk_id or chunk_id in seen:
                continue
            if chunk_id not in selected_chunk_ids:
                continue
            expanded.append(chunk)
            seen.add(str(chunk_id))
            if len(expanded) >= limit:
                return expanded

    return expanded[:limit]


def _normalize_citation_from_chunk(chunk: dict) -> dict:
    return {
        "doc_id": chunk.get("doc_id"),
        "chunk_id": chunk.get("chunk_id"),
        "span_ids": chunk.get("span_ids", []),
    }


def _best_retrieved_chunk(chunks: list[dict]) -> dict | None:
    if not chunks:
        return None
    return next(
        (
            candidate
            for candidate in chunks
            if not str(candidate.get("section_id", "")).startswith("t_")
            and (
                len(str(candidate.get("text", "")).split()) > 12
                or len(candidate.get("span_ids", [])) > 1
            )
        ),
        chunks[0],
    )


def _is_substantive_chunk(chunk: dict) -> bool:
    if _is_title_chunk(chunk):
        return False
    text = str(chunk.get("text", "")).strip()
    span_count = len(chunk.get("span_ids", []))
    return len(text.split()) > 12 or span_count > 1


def _best_chunk_for_answer(state: GraphState) -> dict | None:
    return _best_retrieved_chunk(_reasoning_chunks(state))


def _grounded_answer_from_chunk(chunk: dict) -> str:
    text = " ".join(str(chunk.get("text", "")).split()).strip()
    if not text:
        return "Relevant DMV documentation was retrieved."
    if text.endswith((".", "!", "?")):
        return text
    return f"{text}."


def _normalize_citations(payload: dict, chunk_map: dict[str, dict]) -> list[dict]:
    citations: list[dict] = []
    seen: set[str] = set()

    for raw_citation in payload.get("citations", []):
        chunk_id = raw_citation.get("chunk_id")
        if not chunk_id or chunk_id in seen:
            continue
        chunk = chunk_map.get(chunk_id)
        if chunk is None:
            continue
        citations.append(
            {
                "doc_id": chunk.get("doc_id"),
                "chunk_id": chunk.get("chunk_id"),
                "span_ids": raw_citation.get("span_ids") or chunk.get("span_ids", []),
            }
        )
        seen.add(chunk_id)

    for chunk_id in payload.get("citation_chunk_ids", []):
        if not chunk_id or chunk_id in seen:
            continue
        chunk = chunk_map.get(chunk_id)
        if chunk is None:
            continue
        citations.append(_normalize_citation_from_chunk(chunk))
        seen.add(chunk_id)

    return citations


def _build_evidence_chunks(
    *, state: GraphState, runtime: Runtime, ranked_chunks: list[dict]
) -> list[dict]:
    evidence_chunks = list(ranked_chunks)
    if _content_only_reasoning_enabled(state=state, runtime=runtime):
        evidence_chunks = _content_only_chunks(evidence_chunks)
    if _neighbor_expansion_enabled(state=state, runtime=runtime):
        evidence_chunks = expand_neighbor_sections(
            evidence_chunks,
            runtime.chunk_records_by_doc,
            limit=8,
        )
    return evidence_chunks[:8]


async def route_query(*, state: GraphState, runtime: Runtime) -> dict:
    """Classify the user intent into chitchat or document_query."""
    if not _llm_available(runtime):
        return {"intent": "document_query", "reason": "No LLM available for routing."}

    try:
        return await _ainvoke_structured_prompt(
            runtime=runtime,
            prompt=runtime.prompts.route_query,
            schema=RouteModel,
            payload={
                "conversation": _render_conversation(state.get("conversation", [])),
                "latest_user_utterance": state.get("latest_user_utterance") or "",
            },
        )
    except Exception as exc:
        _log_llm_fallback("route_query", exc)
        return {"intent": "document_query", "reason": f"Routing failed: {exc}"}


def prepare_query(*, state: GraphState, runtime: Runtime) -> tuple[str, dict]:
    """Build the retrieval query string and context from conversation state.

    Returns a tuple of (query_string, query_context) based on the configured
    query mode in ablation_options.
    """
    del runtime
    query_example = _query_example_from_state(state)
    query_context = build_query_context(query_example)
    mode = _query_mode(state)
    match mode:
        case "legacy_transcript":
            return build_legacy_query(
                query_example, history_turn_limit=4
            ), query_context
        case "latest_user_only":
            return (
                build_retrieval_query(
                    query_example,
                    history_turn_limit=0,
                    include_history=False,
                ),
                query_context,
            )
        case "structured":
            return build_retrieval_query(
                query_example, history_turn_limit=4
            ), query_context


async def retrieve_docs(*, state: GraphState, runtime: Runtime) -> list[dict]:
    ablation_options = state.get("ablation_options", {})
    query = state.get("refined_query") or state.get("query")
    if query is None:
        raise ValueError("Missing retrieval query.")
    return await asyncio.to_thread(
        retrieve_chunks,
        example=_query_example_from_state(state),
        vectorstore=runtime.vectorstore,
        keyword_retriever=runtime.keyword_retriever,
        config=runtime.config,
        top_k=runtime.config.retrieval_top_k,
        candidate_k=int(
            ablation_options.get(
                "retrieval_candidate_k",
                runtime.config.retrieval_candidate_k,
            )
        ),
        query=query,
        query_context=state.get("query_context"),
        rerank=_ablation_bool(
            state,
            "retrieval_rerank",
            bool(runtime.config.retrieval_rerank),
        ),
    )


async def grade_evidence(*, state: GraphState, runtime: Runtime) -> dict:
    retrieved_chunks = _reasoning_chunks(state)
    if not retrieved_chunks:
        return _heuristic_evidence_grade(retrieved_chunks)
    if not _llm_available(runtime):
        return _heuristic_evidence_grade(retrieved_chunks)

    try:
        return await _ainvoke_structured_prompt(
            runtime=runtime,
            prompt=runtime.prompts.evidence_grade,
            schema=EvidenceGradeModel,
            payload={
                "conversation": _render_conversation(state.get("conversation", [])),
                "latest_user_utterance": state.get("latest_user_utterance") or "",
                "retrieved_chunks": _render_chunks(retrieved_chunks),
            },
        )
    except LLMCallTimeoutError:
        raise
    except Exception as exc:
        _log_llm_fallback("grade_evidence", exc)
        return attach_fallback_metadata(
            _heuristic_evidence_grade(retrieved_chunks),
            node="grade_evidence",
            exc=exc,
        )


def refine_query(*, state: GraphState, runtime: Runtime) -> str:
    del runtime
    current_query = state.get("query") or ""
    grade = state.get("evidence_grade", {})
    ablation_options = state.get("ablation_options", {})
    if ablation_options.get("query_mode") == "latest_user_only":
        return current_query.strip()
    missing = "; ".join(grade.get("missing_information", []))
    retrieved_chunks = _reasoning_chunks(state)
    focus_titles: list[str] = []
    seen_titles: set[str] = set()
    for chunk in retrieved_chunks:
        if _is_title_chunk(chunk):
            continue
        title = chunk.get("section_title")
        if title and title not in seen_titles:
            seen_titles.add(title)
            focus_titles.append(str(title))
        if len(focus_titles) >= 2:
            break

    if missing:
        return f"{current_query}\nMissing condition: {missing}".strip()
    if focus_titles:
        return f"{current_query}\nFocus sections: {'; '.join(focus_titles)}".strip()
    return current_query.strip()


async def generate_response(*, state: GraphState, runtime: Runtime) -> dict:
    retrieved_chunks = _reasoning_chunks(state)
    grade = state.get("evidence_grade", {})
    if not _llm_available(runtime):
        payload = _heuristic_response(state)
        await _emit_buffered_response_preview(
            state=state,
            runtime=runtime,
            node_name="generate_response",
            payload=payload,
        )
        return payload

    try:
        payload = await _ainvoke_structured_prompt(
            runtime=runtime,
            prompt=runtime.prompts.answer,
            schema=ResponseModel,
            payload={
                "conversation": _render_conversation(state.get("conversation", [])),
                "latest_user_utterance": state.get("latest_user_utterance") or "",
                "evidence_grade": grade,
                "retrieved_chunks": _render_chunks(retrieved_chunks),
            },
        )
        streamed_text = await _emit_streaming_answer_preview(
            state=state,
            runtime=runtime,
            node_name="generate_response",
            payload=payload,
        )
        if streamed_text is not None:
            return {**payload, "response_text": streamed_text}
        await _emit_buffered_response_preview(
            state=state,
            runtime=runtime,
            node_name="generate_response",
            payload=payload,
        )
        return payload
    except LLMCallTimeoutError:
        raise
    except Exception as exc:
        _log_llm_fallback("generate_response", exc)
        payload = attach_fallback_metadata(
            _heuristic_response(state),
            node="generate_response",
            exc=exc,
        )
        await _emit_buffered_response_preview(
            state=state,
            runtime=runtime,
            node_name="generate_response",
            payload=payload,
        )
        return payload


async def resolve_without_answer(*, state: GraphState, runtime: Runtime) -> dict:
    retrieved_chunks = _reasoning_chunks(state)
    grade = state.get("evidence_grade", {})
    ablation_options = state.get("ablation_options", {})
    if (
        ablation_options.get("answer_forward_grounding")
        and grade.get("verdict") == "partial"
    ):
        best_chunk = _best_chunk_for_answer(state)
        if best_chunk is not None and best_chunk.get("chunk_id"):
            payload = {
                "decision": "answer",
                "response_text": _grounded_answer_from_chunk(best_chunk),
                "citation_chunk_ids": [best_chunk.get("chunk_id")],
                "confidence_label": "low",
            }
            await _emit_buffered_response_preview(
                state=state,
                runtime=runtime,
                node_name="resolve_without_answer",
                payload=payload,
            )
            return payload
    if not _llm_available(runtime):
        payload = _heuristic_non_answer_response(state)
        await _emit_buffered_response_preview(
            state=state,
            runtime=runtime,
            node_name="resolve_without_answer",
            payload=payload,
        )
        return payload

    try:
        payload = await _ainvoke_structured_prompt(
            runtime=runtime,
            prompt=runtime.prompts.non_answer,
            schema=FallbackResponseModel,
            payload={
                "conversation": _render_conversation(state.get("conversation", [])),
                "latest_user_utterance": state.get("latest_user_utterance") or "",
                "evidence_grade": grade,
                "retrieved_chunks": _render_chunks(retrieved_chunks),
            },
        )
        await _emit_buffered_response_preview(
            state=state,
            runtime=runtime,
            node_name="resolve_without_answer",
            payload=payload,
        )
        return payload
    except LLMCallTimeoutError:
        raise
    except Exception as exc:
        _log_llm_fallback("resolve_without_answer", exc)
        payload = attach_fallback_metadata(
            _heuristic_non_answer_response(state),
            node="resolve_without_answer",
            exc=exc,
        )
        await _emit_buffered_response_preview(
            state=state,
            runtime=runtime,
            node_name="resolve_without_answer",
            payload=payload,
        )
        return payload


def finalize(*, state: GraphState, payload: dict) -> dict:
    payload = {
        **payload,
        "response_text": str(payload.get("response_text", "")).strip(),
    }
    best_chunk = _best_chunk_for_answer(state)
    chunk_map = {
        chunk.get("chunk_id"): chunk
        for chunk in state.get("retrieved_chunks", [])
        if chunk.get("chunk_id")
    }
    citations = _normalize_citations(payload, chunk_map)
    verdict = state.get("evidence_grade", {}).get("verdict")
    decision = payload.get("decision")
    ablation_options = state.get("ablation_options", {})

    match verdict:
        case "sufficient":
            if decision == "answer" and payload.get("response_text"):
                if not citations and best_chunk is not None:
                    citations = [_normalize_citation_from_chunk(best_chunk)]
                return {**payload, "citations": citations}
            payload = {**payload, **_heuristic_response(state)}
            return {**payload, "citations": _normalize_citations(payload, chunk_map)}
        case "partial":
            if decision == "answer":
                grounded_answer_allowed = bool(
                    ablation_options.get("answer_forward_grounding")
                    and payload.get("response_text")
                    and (citations or best_chunk is not None)
                )
                if not grounded_answer_allowed:
                    payload = {**payload, **_heuristic_response(state)}
                    return {
                        **payload,
                        "citations": _normalize_citations(payload, chunk_map),
                    }
            elif decision != "clarify":
                payload = {**payload, "decision": "clarify"}
            if not citations and best_chunk is not None:
                citations = [_normalize_citation_from_chunk(best_chunk)]
            return {**payload, "citations": citations}
        case "insufficient":
            if decision == "abstain":
                return {**payload, "citations": citations}
            payload = {**payload, **_heuristic_response(state)}
            return {**payload, "citations": _normalize_citations(payload, chunk_map)}
    raise ValueError(f"Unknown evidence verdict: {verdict}")


def _chunk_sort_key(chunk: dict) -> tuple[int, int, str, str]:
    start_sec = chunk.get("start_sec")
    section_number = _numeric_section_id(chunk.get("section_id"))
    return (
        int(start_sec)
        if start_sec is not None
        else (section_number if section_number is not None else 10**9),
        int(chunk.get("subchunk_index") or 0),
        str(chunk.get("section_id", "")),
        str(chunk.get("chunk_id", "")),
    )


def _load_chunk_records_by_doc(config: RuntimeConfig) -> dict[str, list[dict]]:
    chunk_records = config.chunk_records
    if chunk_records is None:
        chunk_artifact_path = config.chunk_artifact_path
        if not chunk_artifact_path:
            return {}
        path = Path(chunk_artifact_path)
        if not path.exists():
            return {}
        chunk_records = load_chunk_records(path)

    by_doc: dict[str, list[dict]] = {}
    for record in chunk_records:
        doc_id = record.get("doc_id")
        if not doc_id:
            continue
        by_doc.setdefault(str(doc_id), []).append(_normalize_chunk_record(record))
    for doc_id, records in by_doc.items():
        by_doc[doc_id] = sorted(records, key=_chunk_sort_key)
    return by_doc


async def build_runtime_async(
    config: RuntimeConfig,
    *,
    run_id: str,
    trace_path: str | Path,
    vectorstore: Any = None,
    keyword_retriever: BM25Retriever | None = None,
    chat_model: Any = None,
    resources: RuntimeResources | None = None,
    event_sink: Any = None,
    stream_responses: bool = False,
) -> Runtime:
    resolved_trace_path = Path(trace_path)
    logger.info(
        "Preparing runtime %s for domain=%s provider=%s trace_path=%s",
        run_id,
        config.domain,
        resolved_chat_provider(config),
        resolved_trace_path,
    )
    resolved_resources = resources
    if resolved_resources is None:
        resolved_resources = await resolve_runtime_resources_async(
            config,
            vectorstore=vectorstore,
            keyword_retriever=keyword_retriever,
            chat_model=chat_model,
        )
    logger.info(
        "Runtime %s ready with chunk_docs=%s chat_model=%s vectorstore=%s",
        run_id,
        len(resolved_resources.chunk_records_by_doc),
        "enabled" if resolved_resources.chat_model is not None else "disabled",
        "enabled" if resolved_resources.vectorstore is not None else "disabled",
    )
    return Runtime(
        config=config,
        vectorstore=resolved_resources.vectorstore,
        keyword_retriever=resolved_resources.keyword_retriever,
        chat_model=resolved_resources.chat_model,
        chunk_records_by_doc=resolved_resources.chunk_records_by_doc,
        prompts=resolved_resources.prompts,
        llm_semaphore=resolved_resources.llm_semaphore,
        trace_path=resolved_trace_path,
        run_id=run_id,
        event_sink=event_sink,
        stream_responses=stream_responses,
        observability=build_observability(config),
    )


async def resolve_runtime_resources_async(
    config: RuntimeConfig,
    *,
    vectorstore: Any = None,
    keyword_retriever: BM25Retriever | None = None,
    chat_model: Any = None,
) -> RuntimeResources:
    resolved_vectorstore = vectorstore
    if resolved_vectorstore is None and config.postgres_dsn and config.embedding_model:
        logger.info(
            "Connecting vectorstore for collection=%s embedding_provider=%s embedding_model=%s",
            config.collection_name,
            resolved_embedding_provider(config),
            config.embedding_model,
        )
        resolved_vectorstore = await asyncio.to_thread(get_vectorstore, config)

    resolved_keyword_retriever = keyword_retriever
    if resolved_keyword_retriever is None and resolved_vectorstore is not None:
        logger.info("Initializing keyword retriever")
        chunk_records = None
        if config.chunk_artifact_path and config.chunk_artifact_path.exists():
            chunk_records = await asyncio.to_thread(
                load_chunk_records, config.chunk_artifact_path
            )

        resolved_keyword_retriever = build_keyword_retriever(
            config,
            chunk_records=chunk_records,
        )

    resolved_chat_model = chat_model
    if resolved_chat_model is None and config.chat_model:
        logger.info(
            "Initializing chat model %s via %s",
            config.chat_model,
            resolved_chat_provider(config),
        )
        resolved_chat_model = build_chat_model(config)

    chunk_records_by_doc = await asyncio.to_thread(_load_chunk_records_by_doc, config)
    logger.info(
        "Loaded chunk metadata for %s documents using prompt_version=%s",
        len(chunk_records_by_doc),
        config.prompt_version,
    )
    return RuntimeResources(
        vectorstore=resolved_vectorstore,
        keyword_retriever=resolved_keyword_retriever,
        chat_model=resolved_chat_model,
        chunk_records_by_doc=chunk_records_by_doc,
        prompts=resolve_prompt_set(config.prompt_version),
    )


__all__ = [
    "_build_evidence_chunks",
    "_emit_graph_event",
    "_trace",
    "build_chat_model",
    "build_runtime_async",
    "expand_neighbor_sections",
    "finalize",
    "generate_response",
    "grade_evidence",
    "prepare_query",
    "refine_query",
    "resolve_runtime_resources_async",
    "resolve_without_answer",
    "retrieve_docs",
    "route_query",
]
