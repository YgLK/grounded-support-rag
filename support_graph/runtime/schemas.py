"""Runtime schemas and internal payload metadata."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from pydantic import BaseModel, Field

from support_graph.config.runtime import RuntimeConfig
from support_graph.runtime.observability import Observability
from support_graph.runtime.prompts import PromptSet


Decision = Literal["answer", "clarify", "abstain"]
Intent = Literal["chitchat", "document_query"]
EvidenceVerdict = Literal["sufficient", "partial", "insufficient"]
ResponseConfidence = Literal["high", "medium", "low"]
FallbackConfidence = Literal["medium", "low"]
GraphStreamEventKind = Literal[
    "query_ready",
    "query_refined",
    "retrieval_complete",
    "evidence_graded",
    "response_started",
    "response_delta",
    "response_completed",
    "fallback",
    "error",
]

_FALLBACK_KEY: Literal["_fallback"] = "_fallback"


class EvidenceGradeModel(BaseModel):
    verdict: EvidenceVerdict
    reason: str
    missing_information: list[str] = Field(default_factory=list)


class RouteModel(BaseModel):
    intent: Intent
    reason: str


class ResponseModel(BaseModel):
    decision: Decision
    response_text: str
    citation_chunk_ids: list[str] = Field(default_factory=list)
    confidence_label: ResponseConfidence


class FallbackResponseModel(BaseModel):
    decision: Literal["clarify", "abstain"]
    response_text: str
    citation_chunk_ids: list[str] = Field(default_factory=list)
    confidence_label: FallbackConfidence


class FallbackTrace(TypedDict, total=False):
    used: bool
    node: str
    mode: Literal["heuristic"]
    exception_type: str
    error: str


class GraphStreamEvent(TypedDict, total=False):
    kind: GraphStreamEventKind
    node: str
    run_id: str
    example_id: str
    query: str
    refined_query: str
    query_context: dict
    retrieval_attempts: int
    retrieval_ranked_chunks: list[dict]
    retrieved_chunks: list[dict]
    evidence_grade: dict
    decision: Decision
    response_text: str
    delta: str
    citations: list[dict]
    confidence_label: str
    trace_summary: dict
    fallback: FallbackTrace
    error: str
    exception_type: str


GraphEventSink = Callable[[GraphStreamEvent], Awaitable[None] | None]


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
    max_attempts: int
    intent: Intent
    response_payload: dict
    final_output: dict
    ablation_options: dict
    fallback_events: list[FallbackTrace]


@dataclass(slots=True)
class Runtime:
    config: RuntimeConfig
    vectorstore: Any
    chat_model: Any
    chunk_records_by_doc: dict[str, list[dict]]
    prompts: PromptSet
    llm_semaphore: asyncio.Semaphore
    trace_path: Path
    run_id: str
    event_sink: GraphEventSink | None = None
    stream_responses: bool = False
    observability: Observability | None = None


@dataclass(slots=True)
class RuntimeResources:
    vectorstore: Any
    chat_model: Any
    chunk_records_by_doc: dict[str, list[dict]]
    prompts: PromptSet
    llm_semaphore: asyncio.Semaphore


def attach_fallback_metadata(
    payload: dict[str, Any],
    *,
    node: str,
    exc: Exception,
) -> dict[str, Any]:
    message = str(exc).strip()
    return {
        **payload,
        _FALLBACK_KEY: {
            "used": True,
            "node": node,
            "mode": "heuristic",
            "exception_type": type(exc).__name__,
            "error": message or repr(exc),
        },
    }


def fallback_metadata(payload: dict[str, Any]) -> FallbackTrace | None:
    raw = payload.get(_FALLBACK_KEY)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError("Fallback metadata must be a dict.")
    return cast(FallbackTrace, dict(raw))


def strip_internal_fields(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != _FALLBACK_KEY}


__all__ = [
    "Decision",
    "Intent",
    "EvidenceGradeModel",
    "EvidenceVerdict",
    "FallbackResponseModel",
    "FallbackTrace",
    "GraphEventSink",
    "GraphStreamEvent",
    "GraphStreamEventKind",
    "GraphState",
    "FallbackConfidence",
    "ResponseConfidence",
    "ResponseModel",
    "RouteModel",
    "Runtime",
    "RuntimeResources",
    "attach_fallback_metadata",
    "fallback_metadata",
    "strip_internal_fields",
]
