"""Phase 3 graph wiring and async runtime entrypoints."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph

from support_graph.config.runtime import RuntimeConfig
from support_graph.logging_utils import get_logger
from support_graph.retrieval.retrieve import build_query_context
from support_graph.runtime.nodes import (
    _build_evidence_chunks,
    _emit_graph_event,
    _trace,
    build_chat_model,
    build_runtime_async,
    expand_neighbor_sections,
    finalize,
    generate_response,
    grade_evidence,
    prepare_query,
    refine_query,
    resolve_runtime_resources_async,
    resolve_without_answer,
    retrieve_docs,
)
from support_graph.runtime.observability import graph_run_context, span_context
from support_graph.runtime.schemas import (
    GraphEventSink,
    GraphState,
    GraphStreamEvent,
    Runtime,
    RuntimeResources,
    fallback_metadata,
    strip_internal_fields,
)


GraphRoute = Literal["generate_response", "refine_query", "resolve_without_answer"]
logger = get_logger(__name__)


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


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _latency_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _graph_path(state: GraphState, node_name: str) -> list[str]:
    return state.get("graph_path", []) + [node_name]


def _query_example(state: GraphState) -> dict:
    return {
        "domain": state.get("domain"),
        "conversation": state.get("conversation", []),
        "latest_user_turn_id": state.get("latest_user_turn_id"),
        "latest_user_utterance": state.get("latest_user_utterance"),
    }


def _query_and_context(prepared: Any, state: GraphState) -> tuple[str, dict]:
    if isinstance(prepared, tuple):
        return prepared
    return str(prepared), build_query_context(_query_example(state))


def _response_state_update(
    response: dict,
    *,
    path: list[str],
    fallback_events: list[dict] | None,
) -> GraphState:
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


def _normalize_final_output(
    state: GraphState, payload: dict, *, runtime: Runtime
) -> dict:
    graph_path = state.get("graph_path", []) + ["finalize"]
    fallback_events = list(state.get("fallback_events", []))
    fallback_nodes = [
        str(event.get("node"))
        for event in fallback_events
        if event.get("node") is not None
    ]
    observability_summary = (
        runtime.observability.summary() if runtime.observability is not None else {}
    )
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
            "trace_path": str(runtime.trace_path),
            "fallback_count": len(fallback_events),
            "fallback_nodes": fallback_nodes,
            "fallbacks": fallback_events,
            **(
                {"observability": observability_summary}
                if observability_summary
                else {}
            ),
        },
    }


def _span_attributes(
    runtime: Runtime,
    state: GraphState,
    node_name: str,
) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        "support_graph.node": node_name,
        "support_graph.run_id": runtime.run_id,
        "support_graph.provider_type": str(runtime.config.provider_type),
    }
    if state.get("example_id") is not None:
        attributes["support_graph.example_id"] = str(state.get("example_id"))
    if state.get("domain") is not None:
        attributes["support_graph.domain"] = str(state.get("domain"))
    return attributes


def _route_after_grade(state: GraphState) -> GraphRoute:
    grade = state.get("evidence_grade", {})
    attempts = state.get("retrieval_attempts", 0)
    max_attempts = state.get("max_attempts", 2)
    verdict = grade.get("verdict")
    match verdict:
        case "sufficient":
            return "generate_response"
        case "partial":
            if attempts < max_attempts:
                return "refine_query"
            return "resolve_without_answer"
        case "insufficient":
            return "resolve_without_answer"
    raise ValueError(f"Unknown evidence verdict: {verdict}")


def _updated_fallback_events(state: GraphState, payload: dict) -> list[dict] | None:
    fallback = fallback_metadata(payload)
    if fallback is None:
        return None
    return list(state.get("fallback_events", [])) + [dict(fallback)]


async def _emit_fallback_event(
    *,
    runtime: Runtime,
    state: GraphState,
    node_name: str,
    fallback_events: list[dict] | None,
) -> None:
    if not fallback_events:
        return
    await _emit_graph_event(
        runtime,
        {
            "kind": "fallback",
            "node": node_name,
            "run_id": runtime.run_id,
            "example_id": state.get("example_id"),
            "fallback": fallback_events[-1],
        },
    )


def _initial_state(
    *,
    example: dict,
    config: RuntimeConfig,
    runtime: Runtime,
    max_attempts: int | None,
) -> GraphState:
    conversation = list(example.get("conversation") or []) or list(
        example.get("turns_before_target") or []
    )
    return {
        "example": example,
        "example_id": example.get("example_id"),
        "domain": example.get("domain") or config.domain,
        "conversation": conversation,
        "latest_user_turn_id": example.get("latest_user_turn_id"),
        "latest_user_utterance": example.get("latest_user_utterance"),
        "retrieval_attempts": 0,
        "graph_path": [],
        "run_id": runtime.run_id,
        "max_attempts": max_attempts or config.max_retrieval_attempts,
        "ablation_options": dict(config.ablation_options or {}),
        "fallback_events": [],
    }


def _build_graph_app(runtime: Runtime, *, run_started: float) -> Any:
    graph_builder = StateGraph(GraphState)

    async def prepare_query_node(state: GraphState) -> GraphState:
        with span_context(
            runtime.observability,
            "support_graph.prepare_query",
            attributes=_span_attributes(runtime, state, "prepare_query"),
        ):
            started = time.perf_counter()
            prepared = await _maybe_await(prepare_query(state=state, runtime=runtime))
            query, query_context = _query_and_context(prepared, state)
            path = _graph_path(state, "prepare_query")
            await _trace(
                runtime,
                "prepare_query",
                {
                    "query": query,
                    "query_context": query_context,
                    "latency_ms": _latency_ms(started),
                },
            )
            await _emit_graph_event(
                runtime,
                {
                    "kind": "query_ready",
                    "node": "prepare_query",
                    "run_id": runtime.run_id,
                    "example_id": state.get("example_id"),
                    "query": query,
                    "query_context": query_context,
                },
            )
            return {
                "query": query,
                "query_context": query_context,
                "final_query": query,
                "graph_path": path,
            }

    async def retrieve_docs_node(state: GraphState) -> GraphState:
        with span_context(
            runtime.observability,
            "support_graph.retrieve_docs",
            attributes=_span_attributes(runtime, state, "retrieve_docs"),
        ):
            started = time.perf_counter()
            ranked_chunks = await _maybe_await(
                retrieve_docs(state=state, runtime=runtime)
            )
            evidence_chunks = _build_evidence_chunks(
                state=state,
                runtime=runtime,
                ranked_chunks=ranked_chunks,
            )
            attempts = state.get("retrieval_attempts", 0) + 1
            path = _graph_path(state, "retrieve_docs")
            await _trace(
                runtime,
                "retrieve_docs",
                {
                    "query": state.get("refined_query") or state.get("query"),
                    "retrieval_attempts": attempts,
                    "retrieval_ranked_count": len(ranked_chunks),
                    "retrieved_count": len(evidence_chunks),
                    "retrieval_ranked_chunks": _retrieval_trace_chunks(ranked_chunks),
                    "retrieved_chunks": _retrieval_trace_chunks(
                        evidence_chunks, limit=8
                    ),
                    "latency_ms": _latency_ms(started),
                },
            )
            await _emit_graph_event(
                runtime,
                {
                    "kind": "retrieval_complete",
                    "node": "retrieve_docs",
                    "run_id": runtime.run_id,
                    "example_id": state.get("example_id"),
                    "retrieval_attempts": attempts,
                    "retrieval_ranked_chunks": ranked_chunks,
                    "retrieved_chunks": evidence_chunks,
                },
            )
            return {
                "retrieval_ranked_chunks": ranked_chunks,
                "retrieved_chunks": evidence_chunks,
                "retrieval_attempts": attempts,
                "graph_path": path,
            }

    async def grade_evidence_node(state: GraphState) -> GraphState:
        with span_context(
            runtime.observability,
            "support_graph.grade_evidence",
            attributes=_span_attributes(runtime, state, "grade_evidence"),
        ):
            started = time.perf_counter()
            raw_grade = await _maybe_await(grade_evidence(state=state, runtime=runtime))
            grade = strip_internal_fields(raw_grade)
            path = _graph_path(state, "grade_evidence")
            event = {
                "evidence_grade": grade,
                "latency_ms": _latency_ms(started),
            }
            fallback_events = _updated_fallback_events(state, raw_grade)
            if fallback_events is not None:
                event["fallback"] = fallback_events[-1]
            await _trace(runtime, "grade_evidence", event)
            await _emit_graph_event(
                runtime,
                {
                    "kind": "evidence_graded",
                    "node": "grade_evidence",
                    "run_id": runtime.run_id,
                    "example_id": state.get("example_id"),
                    "evidence_grade": grade,
                },
            )
            await _emit_fallback_event(
                runtime=runtime,
                state=state,
                node_name="grade_evidence",
                fallback_events=fallback_events,
            )
            result: GraphState = {"evidence_grade": grade, "graph_path": path}
            if fallback_events is not None:
                result["fallback_events"] = fallback_events
            return result

    async def refine_query_node(state: GraphState) -> GraphState:
        with span_context(
            runtime.observability,
            "support_graph.refine_query",
            attributes=_span_attributes(runtime, state, "refine_query"),
        ):
            started = time.perf_counter()
            query = await _maybe_await(refine_query(state=state, runtime=runtime))
            path = _graph_path(state, "refine_query")
            await _trace(
                runtime,
                "refine_query",
                {
                    "refined_query": query,
                    "latency_ms": _latency_ms(started),
                },
            )
            await _emit_graph_event(
                runtime,
                {
                    "kind": "query_refined",
                    "node": "refine_query",
                    "run_id": runtime.run_id,
                    "example_id": state.get("example_id"),
                    "refined_query": query,
                },
            )
            return {"refined_query": query, "final_query": query, "graph_path": path}

    async def generate_response_node(state: GraphState) -> GraphState:
        with span_context(
            runtime.observability,
            "support_graph.generate_response",
            attributes=_span_attributes(runtime, state, "generate_response"),
        ):
            started = time.perf_counter()
            raw_response = await _maybe_await(
                generate_response(state=state, runtime=runtime)
            )
            response = strip_internal_fields(raw_response)
            path = _graph_path(state, "generate_response")
            event = {
                "decision": response.get("decision"),
                "citation_chunk_ids": response.get("citation_chunk_ids", []),
                "latency_ms": _latency_ms(started),
            }
            fallback_events = _updated_fallback_events(state, raw_response)
            if fallback_events is not None:
                event["fallback"] = fallback_events[-1]
            await _trace(runtime, "generate_response", event)
            await _emit_fallback_event(
                runtime=runtime,
                state=state,
                node_name="generate_response",
                fallback_events=fallback_events,
            )
            return _response_state_update(
                response,
                path=path,
                fallback_events=fallback_events,
            )

    async def resolve_without_answer_node(state: GraphState) -> GraphState:
        with span_context(
            runtime.observability,
            "support_graph.resolve_without_answer",
            attributes=_span_attributes(runtime, state, "resolve_without_answer"),
        ):
            started = time.perf_counter()
            raw_response = await _maybe_await(
                resolve_without_answer(state=state, runtime=runtime)
            )
            response = strip_internal_fields(raw_response)
            path = _graph_path(state, "resolve_without_answer")
            event = {
                "decision": response.get("decision"),
                "citation_chunk_ids": response.get("citation_chunk_ids", []),
                "latency_ms": _latency_ms(started),
            }
            fallback_events = _updated_fallback_events(state, raw_response)
            if fallback_events is not None:
                event["fallback"] = fallback_events[-1]
            await _trace(runtime, "resolve_without_answer", event)
            await _emit_fallback_event(
                runtime=runtime,
                state=state,
                node_name="resolve_without_answer",
                fallback_events=fallback_events,
            )
            return _response_state_update(
                response,
                path=path,
                fallback_events=fallback_events,
            )

    async def finalize_node(state: GraphState) -> GraphState:
        with span_context(
            runtime.observability,
            "support_graph.finalize",
            attributes=_span_attributes(runtime, state, "finalize"),
        ):
            started = time.perf_counter()
            payload = state.get("response_payload", {})
            finalized_payload = await _maybe_await(
                finalize(state=state, payload=payload)
            )
            finalized = _normalize_final_output(
                {
                    **state,
                    "total_latency_ms": round(
                        (time.perf_counter() - run_started) * 1000, 2
                    ),
                },
                finalized_payload,
                runtime=runtime,
            )
            await _trace(
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
                    "latency_ms": _latency_ms(started),
                },
            )
            await _emit_graph_event(
                runtime,
                {
                    "kind": "response_completed",
                    "node": "finalize",
                    "run_id": runtime.run_id,
                    "example_id": state.get("example_id"),
                    "decision": finalized.get("decision"),
                    "response_text": finalized.get("response_text"),
                    "citations": finalized.get("citations", []),
                    "confidence_label": finalized.get("confidence_label"),
                    "trace_summary": finalized.get("trace_summary", {}),
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
    return graph_builder.compile()


async def run_graph_async(
    *,
    example: dict,
    config: RuntimeConfig,
    run_id: str,
    trace_path: str | Path,
    max_attempts: int | None = None,
    vectorstore: Any = None,
    chat_model: Any = None,
    _event_sink: GraphEventSink | None = None,
    _stream_responses: bool = False,
    _runtime_resources: RuntimeResources | None = None,
) -> dict:
    run_started = time.perf_counter()
    logger.info(
        "Starting graph run for example=%s domain=%s",
        example.get("example_id"),
        example.get("domain") or config.domain,
    )
    runtime = await build_runtime_async(
        config,
        run_id=run_id,
        trace_path=trace_path,
        vectorstore=vectorstore,
        chat_model=chat_model,
        resources=_runtime_resources,
        event_sink=_event_sink,
        stream_responses=_stream_responses,
    )
    initial_state = _initial_state(
        example=example,
        config=config,
        runtime=runtime,
        max_attempts=max_attempts,
    )
    app = _build_graph_app(runtime, run_started=run_started)
    with graph_run_context(
        runtime.observability,
        project_name=(
            runtime.observability.langsmith_project if runtime.observability else None
        ),
        tags=[
            "support-graph",
            f"provider:{config.provider_type}",
            f"domain:{initial_state.get('domain')}",
        ],
        metadata={
            "run_id": runtime.run_id,
            "example_id": initial_state.get("example_id"),
            "provider_type": config.provider_type,
        },
    ):
        with span_context(
            runtime.observability,
            "support_graph.run_graph_async",
            attributes={
                "support_graph.run_id": runtime.run_id,
                "support_graph.example_id": str(initial_state.get("example_id")),
                "support_graph.provider_type": str(config.provider_type),
                "support_graph.domain": str(initial_state.get("domain")),
            },
        ):
            result = await app.ainvoke(initial_state)
    final_output = cast(dict, result["final_output"])
    logger.info(
        "Completed graph run %s example=%s decision=%s latency_ms=%.2f",
        runtime.run_id,
        initial_state.get("example_id"),
        final_output.get("decision"),
        float(final_output.get("trace_summary", {}).get("latency_ms", 0.0)),
    )
    return final_output


async def astream_graph_events(
    *,
    example: dict,
    config: RuntimeConfig,
    run_id: str,
    trace_path: str | Path,
    max_attempts: int | None = None,
    vectorstore: Any = None,
    chat_model: Any = None,
) -> AsyncIterator[GraphStreamEvent]:
    queue: asyncio.Queue[GraphStreamEvent | None] = asyncio.Queue()

    async def event_sink(event: GraphStreamEvent) -> None:
        await queue.put(event)

    async def runner() -> None:
        try:
            await run_graph_async(
                example=example,
                config=config,
                run_id=run_id,
                trace_path=trace_path,
                max_attempts=max_attempts,
                vectorstore=vectorstore,
                chat_model=chat_model,
                _event_sink=event_sink,
                _stream_responses=True,
            )
        except Exception as exc:
            await event_sink(
                {
                    "kind": "error",
                    "node": "run_graph_async",
                    "example_id": example.get("example_id"),
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                }
            )
        finally:
            await queue.put(None)

    task = asyncio.create_task(runner())
    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            yield event
    finally:
        await task


__all__ = [
    "astream_graph_events",
    "build_chat_model",
    "expand_neighbor_sections",
    "finalize",
    "generate_response",
    "grade_evidence",
    "prepare_query",
    "refine_query",
    "resolve_runtime_resources_async",
    "resolve_without_answer",
    "retrieve_docs",
    "run_graph_async",
]
