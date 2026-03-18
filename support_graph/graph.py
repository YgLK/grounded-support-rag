"""Phase 3 graph runtime."""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from support_graph.index import load_chunk_records
from support_graph.retrieve import build_legacy_query
from support_graph.retrieve import build_query as build_retrieval_query
from support_graph.retrieve import build_query_context
from support_graph.retrieve import get_vectorstore, retrieve_chunks
from support_graph.traces import write_trace_event


Decision = Literal["answer", "clarify", "abstain"]
EvidenceVerdict = Literal["sufficient", "partial", "insufficient"]


class EvidenceGradeModel(BaseModel):
    verdict: EvidenceVerdict
    reason: str
    missing_information: list[str] = Field(default_factory=list)


class ResponseModel(BaseModel):
    decision: Decision
    response_text: str
    citation_chunk_ids: list[str] = Field(default_factory=list)
    confidence_label: Literal["high", "medium", "low"]


class FallbackResponseModel(BaseModel):
    decision: Literal["clarify", "abstain"]
    response_text: str
    citation_chunk_ids: list[str] = Field(default_factory=list)
    confidence_label: Literal["medium", "low"]


class GraphState(TypedDict, total=False):
    example: dict
    example_id: str
    domain: str
    conversation: list[dict]
    latest_user_turn_id: int | None
    latest_user_utterance: str | None
    query_context: dict
    query: str
    refined_query: str
    retrieval_ranked_chunks: list[dict]
    retrieved_chunks: list[dict]
    retrieval_attempts: int
    evidence_grade: dict
    decision: Decision
    response_text: str
    citations: list[dict]
    confidence_label: str
    graph_path: list[str]
    final_query: str
    run_id: str
    trace_dir: str
    max_attempts: int
    response_payload: dict
    final_output: dict
    ablation_options: dict


@dataclass
class Runtime:
    config: Any
    vectorstore: Any
    chat_model: Any
    chunk_records_by_doc: dict[str, list[dict]]
    trace_dir: Path
    run_id: str


def build_chat_model(config: Any, chat_model_cls: type[ChatOllama] = ChatOllama) -> Any:
    return chat_model_cls(
        model=getattr(config, "chat_model"),
        base_url=getattr(config, "ollama_base_url"),
        temperature=0,
    )


def _ablation_options(config: Any) -> dict:
    return dict(getattr(config, "ablation_options", {}) or {})


def _conversation_from_example(example: dict) -> list[dict]:
    if example.get("conversation"):
        return list(example.get("conversation", []))
    if example.get("turns_before_target"):
        return list(example.get("turns_before_target", []))
    return []


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


def _as_payload(value: Any) -> dict:
    if isinstance(value, BaseModel):
        return value.model_dump()
    return dict(value)


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
            "response_text": chunk.get("text", "").strip() or "Relevant documentation was found.",
            "citation_chunk_ids": [chunk.get("chunk_id")] if chunk.get("chunk_id") else [],
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
        missing = grade.get("missing_information") or ["what condition changed in your DMV case"]
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
    return bool(getattr(runtime.config, "content_only_reasoning", True))


def _neighbor_expansion_enabled(*, state: GraphState, runtime: Runtime) -> bool:
    ablation_options = state.get("ablation_options", {})
    if "neighbor_expansion" in ablation_options:
        return bool(ablation_options["neighbor_expansion"])
    return bool(getattr(runtime.config, "neighbor_expansion", True))


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
        chunk.get("chunk_id"): chunk
        for chunk in chunks
        if chunk.get("chunk_id")
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
            if candidate_chunk_id and candidate_section in {section_number - 1, section_number + 1}:
                selected_chunk_ids_by_doc.setdefault(str(doc_id), set()).add(str(candidate_chunk_id))

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
            expanded.append(direct_chunks_by_id.get(chunk_id, _normalize_chunk_record(chunk)))
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


def _retrieval_trace_chunks(chunks: list[dict], limit: int = 5) -> list[dict]:
    summaries: list[dict] = []
    for chunk in chunks[:limit]:
        summaries.append(
            {
                "rank": chunk.get("rank"),
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "section_title": chunk.get("section_title"),
                "span_ids": chunk.get("span_ids", []),
                "score": chunk.get("score"),
            }
        )
    return summaries


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
            and (len(str(candidate.get("text", "")).split()) > 12 or len(candidate.get("span_ids", [])) > 1)
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


def _trace(runtime: Runtime, node: str, event: dict) -> None:
    write_trace_event(
        runtime.trace_dir,
        runtime.run_id,
        {
            "node": node,
            **event,
        },
    )


def _build_evidence_chunks(*, state: GraphState, runtime: Runtime, ranked_chunks: list[dict]) -> list[dict]:
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
    ablation_options = state.get("ablation_options", {})
    query_example = _query_example_from_state(state)
    if ablation_options.get("query_mode") == "legacy_transcript":
        return build_legacy_query(query_example, history_turn_limit=4), build_query_context(query_example)
    if ablation_options.get("query_mode") == "latest_user_only":
        return (
            build_retrieval_query(query_example, history_turn_limit=0, include_history=False),
            build_query_context(query_example),
        )
    return build_retrieval_query(query_example, history_turn_limit=4), build_query_context(query_example)


def retrieve_docs(*, state: GraphState, runtime: Runtime) -> list[dict]:
    ablation_options = state.get("ablation_options", {})
    rerank_enabled = bool(ablation_options.get("retrieval_rerank", getattr(runtime.config, "retrieval_rerank", True)))
    candidate_k = int(ablation_options.get("retrieval_candidate_k", getattr(runtime.config, "retrieval_candidate_k", 12)))
    return retrieve_chunks(
        example={
            "domain": state.get("domain"),
            "conversation": state.get("conversation", []),
            "latest_user_turn_id": state.get("latest_user_turn_id"),
            "latest_user_utterance": state.get("latest_user_utterance"),
        },
        vectorstore=runtime.vectorstore,
        config=runtime.config,
        top_k=getattr(runtime.config, "retrieval_top_k", 5),
        candidate_k=candidate_k,
        query=state.get("refined_query") or state.get("query"),
        query_context=state.get("query_context"),
        rerank=rerank_enabled,
    )


def grade_evidence(*, state: GraphState, runtime: Runtime) -> dict:
    retrieved_chunks = _reasoning_chunks(state)
    if not retrieved_chunks:
        return _heuristic_evidence_grade(retrieved_chunks)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You grade whether retrieved documentation is enough to answer a support question safely. "
                "Return sufficient, partial, or insufficient. "
                "Mark sufficient when the documentation supports a safe next-step answer, even if it does not cover every possible alternative. "
                "Mark partial only when one specific missing condition blocks any safe answer. "
                "When partial, include exactly one concrete missing_information item when possible.",
            ),
            (
                "human",
                "Conversation:\n{conversation}\n\n"
                "Latest user need:\n{latest_user_utterance}\n\n"
                "Retrieved chunks:\n{retrieved_chunks}\n",
            ),
        ]
    )
    try:
        chain = prompt | runtime.chat_model.with_structured_output(EvidenceGradeModel, method="json_schema")
        result = chain.invoke(
            {
                "conversation": _render_conversation(state.get("conversation", [])),
                "latest_user_utterance": state.get("latest_user_utterance") or "",
                "retrieved_chunks": _render_chunks(retrieved_chunks),
            }
        )
        return _as_payload(result)
    except Exception:
        return _heuristic_evidence_grade(retrieved_chunks)


def refine_query(*, state: GraphState, runtime: Runtime) -> str:
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


def generate_response(*, state: GraphState, runtime: Runtime) -> dict:
    retrieved_chunks = _reasoning_chunks(state)
    grade = state.get("evidence_grade", {})
    if not getattr(runtime.config, "chat_model", None):
        return _heuristic_response(state)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are SupportGraph, a calm documentation-grounded support operator. "
                "Only answer from the provided evidence. "
                "Choose answer, clarify, or abstain. "
                "If the evidence supports a safe conditional next step, answer instead of clarifying. "
                "Do not repeat a retrieved question heading as the answer. Prefer substantive content chunks. "
                "If clarifying, ask one concrete missing-condition question. "
                "If abstaining, explain what is missing. "
                "Do not include inline bracket citations in response_text. "
                "Use citation_chunk_ids from the provided chunk IDs only.",
            ),
            (
                "human",
                "Conversation:\n{conversation}\n\n"
                "Latest user need:\n{latest_user_utterance}\n\n"
                "Evidence grade:\n{evidence_grade}\n\n"
                "Retrieved chunks:\n{retrieved_chunks}\n",
            ),
        ]
    )
    try:
        chain = prompt | runtime.chat_model.with_structured_output(ResponseModel, method="json_schema")
        result = chain.invoke(
            {
                "conversation": _render_conversation(state.get("conversation", [])),
                "latest_user_utterance": state.get("latest_user_utterance") or "",
                "evidence_grade": grade,
                "retrieved_chunks": _render_chunks(retrieved_chunks),
            }
        )
        return _as_payload(result)
    except Exception:
        return _heuristic_response(state)


def resolve_without_answer(*, state: GraphState, runtime: Runtime) -> dict:
    retrieved_chunks = _reasoning_chunks(state)
    grade = state.get("evidence_grade", {})
    ablation_options = state.get("ablation_options", {})
    if ablation_options.get("answer_forward_grounding") and grade.get("verdict") == "partial":
        best_chunk = _best_chunk_for_answer(state)
        if best_chunk is not None and best_chunk.get("chunk_id"):
            return {
                "decision": "answer",
                "response_text": _grounded_answer_from_chunk(best_chunk),
                "citation_chunk_ids": [best_chunk.get("chunk_id")],
                "confidence_label": "low",
            }
    if not getattr(runtime.config, "chat_model", None):
        return _heuristic_non_answer_response(state)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are SupportGraph, a calm documentation-grounded support operator. "
                "The evidence is not sufficient for a direct answer. "
                "Choose only clarify or abstain. "
                "If clarifying, ask exactly one concrete missing-condition question. "
                "If abstaining, explain briefly what support is missing. "
                "Do not answer the user's underlying question. "
                "Do not include inline bracket citations in response_text. "
                "Use citation_chunk_ids from the provided chunk IDs only.",
            ),
            (
                "human",
                "Conversation:\n{conversation}\n\n"
                "Latest user need:\n{latest_user_utterance}\n\n"
                "Evidence grade:\n{evidence_grade}\n\n"
                "Retrieved chunks:\n{retrieved_chunks}\n",
            ),
        ]
    )
    try:
        chain = prompt | runtime.chat_model.with_structured_output(FallbackResponseModel, method="json_schema")
        result = chain.invoke(
            {
                "conversation": _render_conversation(state.get("conversation", [])),
                "latest_user_utterance": state.get("latest_user_utterance") or "",
                "evidence_grade": grade,
                "retrieved_chunks": _render_chunks(retrieved_chunks),
            }
        )
        return _as_payload(result)
    except Exception:
        return _heuristic_non_answer_response(state)


def finalize(*, state: GraphState, payload: dict) -> dict:
    payload = {**payload, "response_text": str(payload.get("response_text", "")).strip()}
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


def _normalize_final_output(state: GraphState, payload: dict) -> dict:
    graph_path = state.get("graph_path", []) + ["finalize"]
    return {
        "example_id": state.get("example_id"),
        "decision": payload.get("decision"),
        "response_text": str(payload.get("response_text", "")).strip(),
        "citations": payload.get("citations", []),
        "confidence_label": payload.get("confidence_label", "low"),
        "latest_user_utterance": state.get("latest_user_utterance"),
        "retrieval_ranked_chunks": state.get("retrieval_ranked_chunks", []),
        "retrieved_chunks": state.get("retrieved_chunks", []),
        "evidence_grade": state.get("evidence_grade", {}),
        "trace_summary": {
            "retrieval_attempts": state.get("retrieval_attempts", 0),
            "final_query": state.get("final_query") or state.get("refined_query") or state.get("query"),
            "graph_path": graph_path,
            "latency_ms": state.get("total_latency_ms", 0.0),
        },
    }


def _route_after_grade(state: GraphState) -> str:
    grade = state.get("evidence_grade", {})
    attempts = state.get("retrieval_attempts", 0)
    max_attempts = state.get("max_attempts", 2)
    if grade.get("verdict") == "sufficient":
        return "generate_response"
    if grade.get("verdict") == "partial" and attempts < max_attempts:
        return "refine_query"
    return "resolve_without_answer"


def _chunk_sort_key(chunk: dict) -> tuple[int, int, str, str]:
    start_sec = chunk.get("start_sec")
    section_number = _numeric_section_id(chunk.get("section_id"))
    return (
        int(start_sec) if start_sec is not None else (section_number if section_number is not None else 10**9),
        int(chunk.get("subchunk_index") or 0),
        str(chunk.get("section_id", "")),
        str(chunk.get("chunk_id", "")),
    )


def _load_chunk_records_by_doc(config: Any) -> dict[str, list[dict]]:
    chunk_records = getattr(config, "chunk_records", None)
    if chunk_records is None:
        chunk_artifact_path = getattr(config, "chunk_artifact_path", None)
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


def _build_runtime(config: Any, trace_dir: str | Path | None = None, vectorstore: Any = None, chat_model: Any = None) -> Runtime:
    run_id = f"run-{uuid.uuid4().hex[:12]}"
    resolved_trace_dir = Path(trace_dir or getattr(config, "trace_dir"))
    resolved_vectorstore = vectorstore
    if resolved_vectorstore is None and getattr(config, "postgres_dsn", None) and getattr(config, "embedding_model", None):
        resolved_vectorstore = get_vectorstore(config)

    resolved_chat_model = chat_model
    if resolved_chat_model is None and getattr(config, "chat_model", None):
        resolved_chat_model = build_chat_model(config)
    return Runtime(
        config=config,
        vectorstore=resolved_vectorstore,
        chat_model=resolved_chat_model,
        chunk_records_by_doc=_load_chunk_records_by_doc(config),
        trace_dir=resolved_trace_dir,
        run_id=run_id,
    )


def run_graph(
    *,
    example: dict,
    config: Any,
    max_attempts: int | None = None,
    trace_dir: str | Path | None = None,
    vectorstore: Any = None,
    chat_model: Any = None,
) -> dict:
    run_started = time.perf_counter()
    runtime = _build_runtime(config, trace_dir=trace_dir, vectorstore=vectorstore, chat_model=chat_model)
    conversation = _conversation_from_example(example)

    initial_state: GraphState = {
        "example": example,
        "example_id": example.get("example_id"),
        "domain": example.get("domain") or getattr(config, "domain", "dmv"),
        "conversation": conversation,
        "latest_user_turn_id": example.get("latest_user_turn_id"),
        "latest_user_utterance": example.get("latest_user_utterance"),
        "retrieval_attempts": 0,
        "graph_path": [],
        "run_id": runtime.run_id,
        "trace_dir": str(runtime.trace_dir),
        "max_attempts": max_attempts or getattr(config, "max_retrieval_attempts", 2),
        "ablation_options": _ablation_options(config),
    }

    graph_builder = StateGraph(GraphState)

    def prepare_query_node(state: GraphState) -> GraphState:
        started = time.perf_counter()
        prepared = prepare_query(state=state, runtime=runtime)
        if isinstance(prepared, tuple):
            query, query_context = prepared
        else:
            query = str(prepared)
            query_context = build_query_context(_query_example_from_state(state))
        path = state.get("graph_path", []) + ["prepare_query"]
        _trace(
            runtime,
            "prepare_query",
            {
                "query": query,
                "query_context": query_context,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return {
            "query": query,
            "query_context": query_context,
            "final_query": query,
            "graph_path": path,
        }

    def retrieve_docs_node(state: GraphState) -> GraphState:
        started = time.perf_counter()
        ranked_chunks = retrieve_docs(state=state, runtime=runtime)
        evidence_chunks = _build_evidence_chunks(
            state=state,
            runtime=runtime,
            ranked_chunks=ranked_chunks,
        )
        attempts = state.get("retrieval_attempts", 0) + 1
        path = state.get("graph_path", []) + ["retrieve_docs"]
        _trace(
            runtime,
            "retrieve_docs",
            {
                "query": state.get("refined_query") or state.get("query"),
                "retrieval_attempts": attempts,
                "retrieval_ranked_count": len(ranked_chunks),
                "retrieved_count": len(evidence_chunks),
                "retrieval_ranked_chunks": _retrieval_trace_chunks(ranked_chunks),
                "retrieved_chunks": _retrieval_trace_chunks(evidence_chunks, limit=8),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return {
            "retrieval_ranked_chunks": ranked_chunks,
            "retrieved_chunks": evidence_chunks,
            "retrieval_attempts": attempts,
            "graph_path": path,
        }

    def grade_evidence_node(state: GraphState) -> GraphState:
        started = time.perf_counter()
        grade = grade_evidence(state=state, runtime=runtime)
        path = state.get("graph_path", []) + ["grade_evidence"]
        _trace(
            runtime,
            "grade_evidence",
            {
                "evidence_grade": grade,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return {"evidence_grade": grade, "graph_path": path}

    def refine_query_node(state: GraphState) -> GraphState:
        started = time.perf_counter()
        query = refine_query(state=state, runtime=runtime)
        path = state.get("graph_path", []) + ["refine_query"]
        _trace(
            runtime,
            "refine_query",
            {
                "refined_query": query,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return {"refined_query": query, "final_query": query, "graph_path": path}

    def generate_response_node(state: GraphState) -> GraphState:
        started = time.perf_counter()
        response = generate_response(state=state, runtime=runtime)
        path = state.get("graph_path", []) + ["generate_response"]
        _trace(
            runtime,
            "generate_response",
            {
                "decision": response.get("decision"),
                "citation_chunk_ids": response.get("citation_chunk_ids", []),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return {
            "decision": response.get("decision"),
            "response_text": response.get("response_text"),
            "confidence_label": response.get("confidence_label"),
            "citations": response.get("citation_chunk_ids", []),
            "response_payload": response,
            "graph_path": path,
        }

    def resolve_without_answer_node(state: GraphState) -> GraphState:
        started = time.perf_counter()
        response = resolve_without_answer(state=state, runtime=runtime)
        path = state.get("graph_path", []) + ["resolve_without_answer"]
        _trace(
            runtime,
            "resolve_without_answer",
            {
                "decision": response.get("decision"),
                "citation_chunk_ids": response.get("citation_chunk_ids", []),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return {
            "decision": response.get("decision"),
            "response_text": response.get("response_text"),
            "confidence_label": response.get("confidence_label"),
            "citations": response.get("citation_chunk_ids", []),
            "response_payload": response,
            "graph_path": path,
        }

    def finalize_node(state: GraphState) -> GraphState:
        started = time.perf_counter()
        payload = state.get("response_payload", {})
        finalized_payload = finalize(state=state, payload=payload)
        finalized = _normalize_final_output(
            {**state, "total_latency_ms": round((time.perf_counter() - run_started) * 1000, 2)},
            finalized_payload,
        )
        _trace(
            runtime,
            "finalize",
            {
                "decision": finalized.get("decision"),
                "citations": finalized.get("citations", []),
                "total_latency_ms": finalized.get("trace_summary", {}).get("latency_ms", 0.0),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return {"final_output": finalized}

    graph_builder.add_node("prepare_query", prepare_query_node)
    graph_builder.add_node("retrieve_docs", retrieve_docs_node)
    graph_builder.add_node("grade_evidence", grade_evidence_node)
    graph_builder.add_node("refine_query", refine_query_node)
    graph_builder.add_node("generate_response", generate_response_node)
    graph_builder.add_node("resolve_without_answer", resolve_without_answer_node)
    graph_builder.add_node("finalize", finalize_node)

    graph_builder.add_edge(START, "prepare_query")
    graph_builder.add_edge("prepare_query", "retrieve_docs")
    graph_builder.add_edge("retrieve_docs", "grade_evidence")
    graph_builder.add_conditional_edges(
        "grade_evidence",
        _route_after_grade,
        {
            "generate_response": "generate_response",
            "refine_query": "refine_query",
            "resolve_without_answer": "resolve_without_answer",
        },
    )
    graph_builder.add_edge("refine_query", "retrieve_docs")
    graph_builder.add_edge("generate_response", "finalize")
    graph_builder.add_edge("resolve_without_answer", "finalize")
    graph_builder.add_edge("finalize", END)

    app = graph_builder.compile()
    result = app.invoke(initial_state)
    return result["final_output"]
