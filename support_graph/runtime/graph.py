"""Phase 3 graph wiring and public runtime entrypoints."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from support_graph.config.runtime import RuntimeConfigLike
from support_graph.retrieval.retrieve import build_query_context
from support_graph.runtime.nodes import (
    _build_evidence_chunks,
    build_chat_model,
    build_runtime,
    expand_neighbor_sections,
    finalize,
    generate_response,
    grade_evidence,
    prepare_query,
    refine_query,
    resolve_without_answer,
    retrieve_docs,
)
from support_graph.runtime.schemas import (
    GraphState,
    Runtime,
    fallback_metadata,
    strip_internal_fields,
)
from support_graph.runtime.traces import trace_file_path, write_trace_event


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


def _trace(runtime: Runtime, node: str, event: dict) -> None:
    write_trace_event(
        runtime.trace_dir,
        runtime.run_id,
        {
            "node": node,
            **event,
        },
    )


def _normalize_final_output(state: GraphState, payload: dict) -> dict:
    graph_path = state.get("graph_path", []) + ["finalize"]
    trace_path = trace_file_path(state.get("trace_dir", ""), state.get("run_id", ""))
    fallback_events = list(state.get("fallback_events", []))
    fallback_nodes = [
        str(event.get("node"))
        for event in fallback_events
        if event.get("node") is not None
    ]
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
            "final_query": state.get("final_query")
            or state.get("refined_query")
            or state.get("query"),
            "graph_path": graph_path,
            "latency_ms": state.get("total_latency_ms", 0.0),
            "trace_path": str(trace_path),
            "fallback_count": len(fallback_events),
            "fallback_nodes": fallback_nodes,
            "fallbacks": fallback_events,
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


def _updated_fallback_events(state: GraphState, payload: dict) -> list[dict] | None:
    fallback = fallback_metadata(payload)
    if fallback is None:
        return None
    return list(state.get("fallback_events", [])) + [dict(fallback)]


def run_graph(
    *,
    example: dict,
    config: RuntimeConfigLike,
    max_attempts: int | None = None,
    trace_dir: str | Path | None = None,
    vectorstore: Any = None,
    chat_model: Any = None,
) -> dict:
    run_started = time.perf_counter()
    runtime = build_runtime(
        config, trace_dir=trace_dir, vectorstore=vectorstore, chat_model=chat_model
    )

    initial_state: GraphState = {
        "example": example,
        "example_id": example.get("example_id"),
        "domain": example.get("domain") or config.domain,
        "conversation": list(example.get("conversation") or [])
        or list(example.get("turns_before_target") or []),
        "latest_user_turn_id": example.get("latest_user_turn_id"),
        "latest_user_utterance": example.get("latest_user_utterance"),
        "retrieval_attempts": 0,
        "graph_path": [],
        "run_id": runtime.run_id,
        "trace_dir": str(runtime.trace_dir),
        "max_attempts": max_attempts or config.max_retrieval_attempts,
        "ablation_options": dict(config.ablation_options or {}),
        "fallback_events": [],
    }

    graph_builder = StateGraph(GraphState)

    def prepare_query_node(state: GraphState) -> GraphState:
        started = time.perf_counter()
        prepared = prepare_query(state=state, runtime=runtime)
        if isinstance(prepared, tuple):
            query, query_context = prepared
        else:
            query = str(prepared)
            query_context = build_query_context(
                {
                    "domain": state.get("domain"),
                    "conversation": state.get("conversation", []),
                    "latest_user_turn_id": state.get("latest_user_turn_id"),
                    "latest_user_utterance": state.get("latest_user_utterance"),
                }
            )
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
        raw_grade = grade_evidence(state=state, runtime=runtime)
        grade = strip_internal_fields(raw_grade)
        path = state.get("graph_path", []) + ["grade_evidence"]
        event = {
            "evidence_grade": grade,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        fallback_events = _updated_fallback_events(state, raw_grade)
        if fallback_events is not None:
            event["fallback"] = fallback_events[-1]
        _trace(runtime, "grade_evidence", event)
        result: GraphState = {"evidence_grade": grade, "graph_path": path}
        if fallback_events is not None:
            result["fallback_events"] = fallback_events
        return result

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
        raw_response = generate_response(state=state, runtime=runtime)
        response = strip_internal_fields(raw_response)
        path = state.get("graph_path", []) + ["generate_response"]
        event = {
            "decision": response.get("decision"),
            "citation_chunk_ids": response.get("citation_chunk_ids", []),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        fallback_events = _updated_fallback_events(state, raw_response)
        if fallback_events is not None:
            event["fallback"] = fallback_events[-1]
        _trace(runtime, "generate_response", event)
        result: GraphState = {
            "decision": response.get("decision"),
            "response_text": response.get("response_text"),
            "confidence_label": response.get("confidence_label"),
            "citations": response.get("citation_chunk_ids", []),
            "response_payload": response,
            "graph_path": path,
        }
        if fallback_events is not None:
            result["fallback_events"] = fallback_events
        return result

    def resolve_without_answer_node(state: GraphState) -> GraphState:
        started = time.perf_counter()
        raw_response = resolve_without_answer(state=state, runtime=runtime)
        response = strip_internal_fields(raw_response)
        path = state.get("graph_path", []) + ["resolve_without_answer"]
        event = {
            "decision": response.get("decision"),
            "citation_chunk_ids": response.get("citation_chunk_ids", []),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        fallback_events = _updated_fallback_events(state, raw_response)
        if fallback_events is not None:
            event["fallback"] = fallback_events[-1]
        _trace(runtime, "resolve_without_answer", event)
        result: GraphState = {
            "decision": response.get("decision"),
            "response_text": response.get("response_text"),
            "confidence_label": response.get("confidence_label"),
            "citations": response.get("citation_chunk_ids", []),
            "response_payload": response,
            "graph_path": path,
        }
        if fallback_events is not None:
            result["fallback_events"] = fallback_events
        return result

    def finalize_node(state: GraphState) -> GraphState:
        started = time.perf_counter()
        payload = state.get("response_payload", {})
        finalized_payload = finalize(state=state, payload=payload)
        finalized = _normalize_final_output(
            {
                **state,
                "total_latency_ms": round(
                    (time.perf_counter() - run_started) * 1000, 2
                ),
            },
            finalized_payload,
        )
        _trace(
            runtime,
            "finalize",
            {
                "decision": finalized.get("decision"),
                "citations": finalized.get("citations", []),
                "total_latency_ms": finalized.get("trace_summary", {}).get(
                    "latency_ms", 0.0
                ),
                "fallback_count": finalized.get("trace_summary", {}).get(
                    "fallback_count", 0
                ),
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


__all__ = [
    "build_chat_model",
    "expand_neighbor_sections",
    "finalize",
    "generate_response",
    "grade_evidence",
    "prepare_query",
    "refine_query",
    "resolve_without_answer",
    "retrieve_docs",
    "run_graph",
    "write_trace_event",
]
