"""Runtime node implementations and shared helpers."""

from __future__ import annotations

import asyncio
import inspect
import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain_ollama import ChatOllama
from pydantic import BaseModel

from support_graph.config.runtime import RuntimeConfigLike
from support_graph.logging_utils import get_logger
from support_graph.retrieval.index import load_chunk_records
from support_graph.retrieval.retrieve import build_legacy_query
from support_graph.retrieval.retrieve import build_query as build_retrieval_query
from support_graph.retrieval.retrieve import build_query_context
from support_graph.retrieval.retrieve import get_vectorstore, retrieve_chunks
from support_graph.runtime.llm_policy import ainvoke_with_retry, shared_llm_semaphore
from support_graph.runtime.prompts import resolve_prompt_set
from support_graph.runtime.schemas import (
    EvidenceGradeModel,
    FallbackResponseModel,
    GraphState,
    GraphStreamEvent,
    ResponseModel,
    Runtime,
    attach_fallback_metadata,
)


logger = get_logger(__name__)


def build_chat_model(
    config: RuntimeConfigLike, chat_model_cls: type[ChatOllama] = ChatOllama
) -> Any:
    return chat_model_cls(
        model=config.chat_model,
        base_url=config.ollama_base_url,
        temperature=0,
    )


async def _emit_stream_event(runtime: Runtime, event: GraphStreamEvent) -> None:
    if runtime.event_sink is None:
        return
    result = runtime.event_sink(event)
    if inspect.isawaitable(result):
        await result


def _response_deltas(text: str) -> list[str]:
    stripped = str(text or "")
    if not stripped:
        return []
    deltas = re.findall(r"\S+\s*", stripped)
    return deltas or [stripped]


async def _emit_response_preview(
    *,
    state: GraphState,
    runtime: Runtime,
    node_name: str,
    payload: dict[str, Any],
) -> None:
    if not runtime.stream_responses:
        return

    await _emit_stream_event(
        runtime,
        {
            "kind": "response_started",
            "node": node_name,
            "run_id": runtime.run_id,
            "example_id": state.get("example_id"),
            "decision": payload.get("decision"),
        },
    )
    for delta in _response_deltas(str(payload.get("response_text", ""))):
        await _emit_stream_event(
            runtime,
            {
                "kind": "response_delta",
                "node": node_name,
                "run_id": runtime.run_id,
                "example_id": state.get("example_id"),
                "delta": delta,
            },
        )


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
        parts.append(
            "\n".join(
                [
                    f"Chunk ID: {chunk.get('chunk_id')}",
                    f"Doc ID: {chunk.get('doc_id')}",
                    f"Spans: {', '.join(chunk.get('span_ids', []))}",
                    f"Text: {chunk.get('text', '')}",
                ]
            )
        )
    return "\n\n".join(parts)


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
    result = await ainvoke_with_retry(
        lambda: chain.ainvoke(payload),
        semaphore=runtime.llm_semaphore,
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
    return {
        "verdict": "partial",
        "reason": "Retrieved evidence needs model review.",
        "missing_information": [],
    }


def _heuristic_response(state: GraphState) -> dict:
    grade = state.get("evidence_grade", {})
    chunks = _reasoning_chunks(state)
    if grade.get("verdict") == "sufficient" and chunks:
        chunk = _best_retrieved_chunk(chunks)
        return {
            "decision": "answer",
            "response_text": chunk.get("text", "").strip()
            or "Relevant documentation was found.",
            "citation_chunk_ids": [chunk.get("chunk_id")]
            if chunk and chunk.get("chunk_id")
            else [],
            "confidence_label": "medium",
        }
    if grade.get("verdict") == "partial":
        missing = grade.get("missing_information") or ["one missing detail"]
        return {
            "decision": "clarify",
            "response_text": f"Need one detail before answering: {missing[0]}.",
            "citation_chunk_ids": [chunks[0].get("chunk_id")] if chunks else [],
            "confidence_label": "low",
        }
    return {
        "decision": "abstain",
        "response_text": "No sufficient support was found in the indexed documentation.",
        "citation_chunk_ids": [],
        "confidence_label": "low",
    }


def _heuristic_non_answer_response(state: GraphState) -> dict:
    grade = state.get("evidence_grade", {})
    chunks = _reasoning_chunks(state)
    if grade.get("verdict") == "partial":
        missing = grade.get("missing_information") or [
            "what condition changed in your DMV case"
        ]
        return {
            "decision": "clarify",
            "response_text": f"Need one detail before answering: {missing[0]}.",
            "citation_chunk_ids": [chunks[0].get("chunk_id")] if chunks else [],
            "confidence_label": "low",
        }
    return {
        "decision": "abstain",
        "response_text": "No sufficient support was found in the indexed documentation.",
        "citation_chunk_ids": [],
        "confidence_label": "low",
    }


def _query_example_from_state(state: GraphState) -> dict:
    return {
        "domain": state.get("domain"),
        "domain_hint": state.get("domain"),
        "conversation": state.get("conversation", []),
        "latest_user_turn_id": state.get("latest_user_turn_id"),
        "latest_user_utterance": state.get("latest_user_utterance"),
    }


def _reasoning_chunks(state: GraphState) -> list[dict]:
    return list(state.get("retrieved_chunks", []))


def _is_title_chunk(chunk: dict) -> bool:
    return str(chunk.get("section_id", "")).startswith("t_")


def _content_only_reasoning_enabled(*, state: GraphState, runtime: Runtime) -> bool:
    ablation_options = state.get("ablation_options", {})
    if "content_only_reasoning" in ablation_options:
        return bool(ablation_options["content_only_reasoning"])
    return bool(runtime.config.content_only_reasoning)


def _neighbor_expansion_enabled(*, state: GraphState, runtime: Runtime) -> bool:
    ablation_options = state.get("ablation_options", {})
    if "neighbor_expansion" in ablation_options:
        return bool(ablation_options["neighbor_expansion"])
    return bool(runtime.config.neighbor_expansion)


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


def prepare_query(*, state: GraphState, runtime: Runtime) -> tuple[str, dict]:
    del runtime
    ablation_options = state.get("ablation_options", {})
    query_example = _query_example_from_state(state)
    if ablation_options.get("query_mode") == "legacy_transcript":
        return build_legacy_query(
            query_example, history_turn_limit=4
        ), build_query_context(query_example)
    if ablation_options.get("query_mode") == "latest_user_only":
        return (
            build_retrieval_query(
                query_example, history_turn_limit=0, include_history=False
            ),
            build_query_context(query_example),
        )
    return build_retrieval_query(
        query_example, history_turn_limit=4
    ), build_query_context(query_example)


async def retrieve_docs(*, state: GraphState, runtime: Runtime) -> list[dict]:
    ablation_options = state.get("ablation_options", {})
    rerank_enabled = bool(
        ablation_options.get("retrieval_rerank", runtime.config.retrieval_rerank)
    )
    candidate_k = int(
        ablation_options.get(
            "retrieval_candidate_k",
            runtime.config.retrieval_candidate_k,
        )
    )
    return await asyncio.to_thread(
        retrieve_chunks,
        example={
            "domain": state.get("domain"),
            "conversation": state.get("conversation", []),
            "latest_user_turn_id": state.get("latest_user_turn_id"),
            "latest_user_utterance": state.get("latest_user_utterance"),
        },
        vectorstore=runtime.vectorstore,
        config=runtime.config,
        top_k=runtime.config.retrieval_top_k,
        candidate_k=candidate_k,
        query=state.get("refined_query") or state.get("query"),
        query_context=state.get("query_context"),
        rerank=rerank_enabled,
    )


async def grade_evidence(*, state: GraphState, runtime: Runtime) -> dict:
    retrieved_chunks = _reasoning_chunks(state)
    if not retrieved_chunks:
        return _heuristic_evidence_grade(retrieved_chunks)
    if not runtime.config.chat_model or runtime.chat_model is None:
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
    if not runtime.config.chat_model or runtime.chat_model is None:
        payload = _heuristic_response(state)
        await _emit_response_preview(
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
        await _emit_response_preview(
            state=state,
            runtime=runtime,
            node_name="generate_response",
            payload=payload,
        )
        return payload
    except Exception as exc:
        _log_llm_fallback("generate_response", exc)
        payload = attach_fallback_metadata(
            _heuristic_response(state),
            node="generate_response",
            exc=exc,
        )
        await _emit_response_preview(
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
            await _emit_response_preview(
                state=state,
                runtime=runtime,
                node_name="resolve_without_answer",
                payload=payload,
            )
            return payload
    if not runtime.config.chat_model or runtime.chat_model is None:
        payload = _heuristic_non_answer_response(state)
        await _emit_response_preview(
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
        await _emit_response_preview(
            state=state,
            runtime=runtime,
            node_name="resolve_without_answer",
            payload=payload,
        )
        return payload
    except Exception as exc:
        _log_llm_fallback("resolve_without_answer", exc)
        payload = attach_fallback_metadata(
            _heuristic_non_answer_response(state),
            node="resolve_without_answer",
            exc=exc,
        )
        await _emit_response_preview(
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

    if verdict == "sufficient":
        if decision != "answer" or not payload.get("response_text"):
            fallback = _heuristic_response(state)
            payload = {**payload, **fallback}
            citations = _normalize_citations(payload, chunk_map)
        elif not citations and best_chunk is not None:
            citations = [_normalize_citation_from_chunk(best_chunk)]
    elif verdict == "partial":
        if decision == "answer":
            if not (
                ablation_options.get("answer_forward_grounding")
                and payload.get("response_text")
                and (citations or best_chunk is not None)
            ):
                fallback = _heuristic_response(state)
                payload = {**payload, **fallback}
                citations = _normalize_citations(payload, chunk_map)
            elif not citations and best_chunk is not None:
                citations = [_normalize_citation_from_chunk(best_chunk)]
        elif decision != "clarify":
            payload = {**payload, "decision": "clarify"}
        if not citations and best_chunk is not None:
            citations = [_normalize_citation_from_chunk(best_chunk)]
    elif verdict == "insufficient":
        if decision != "abstain":
            fallback = _heuristic_response(state)
            payload = {**payload, **fallback}
            citations = _normalize_citations(payload, chunk_map)

    return {**payload, "citations": citations}


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


def _load_chunk_records_by_doc(config: RuntimeConfigLike) -> dict[str, list[dict]]:
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
    config: RuntimeConfigLike,
    trace_dir: str | Path | None = None,
    vectorstore: Any = None,
    chat_model: Any = None,
    *,
    event_sink: Any = None,
    stream_responses: bool = False,
) -> Runtime:
    run_id = f"run-{uuid.uuid4().hex[:12]}"
    resolved_trace_dir = Path(trace_dir or config.trace_dir)
    resolved_vectorstore = vectorstore
    if resolved_vectorstore is None and config.postgres_dsn and config.embedding_model:
        resolved_vectorstore = await asyncio.to_thread(get_vectorstore, config)

    resolved_chat_model = chat_model
    if resolved_chat_model is None and config.chat_model:
        resolved_chat_model = build_chat_model(config)

    chunk_records_by_doc = await asyncio.to_thread(_load_chunk_records_by_doc, config)
    return Runtime(
        config=config,
        vectorstore=resolved_vectorstore,
        chat_model=resolved_chat_model,
        chunk_records_by_doc=chunk_records_by_doc,
        prompts=resolve_prompt_set(config.prompt_version),
        llm_semaphore=shared_llm_semaphore(config),
        trace_dir=resolved_trace_dir,
        run_id=run_id,
        event_sink=event_sink,
        stream_responses=stream_responses,
    )


__all__ = [
    "build_chat_model",
    "build_runtime_async",
    "expand_neighbor_sections",
    "finalize",
    "generate_response",
    "grade_evidence",
    "prepare_query",
    "refine_query",
    "resolve_without_answer",
    "retrieve_docs",
]
