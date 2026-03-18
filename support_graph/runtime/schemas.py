"""Runtime schemas and internal payload metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from pydantic import BaseModel, Field

from support_graph.config.runtime import RuntimeConfigLike


Decision = Literal["answer", "clarify", "abstain"]
EvidenceVerdict = Literal["sufficient", "partial", "insufficient"]

_FALLBACK_KEY = "_fallback"


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


class FallbackTrace(TypedDict, total=False):
    used: bool
    node: str
    mode: Literal["heuristic"]
    exception_type: str
    error: str


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
    fallback_events: list[FallbackTrace]


@dataclass(slots=True)
class Runtime:
    config: RuntimeConfigLike
    vectorstore: Any
    chat_model: Any
    chunk_records_by_doc: dict[str, list[dict]]
    trace_dir: Path
    run_id: str


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
    if not isinstance(raw, dict):
        return None
    return cast(FallbackTrace, dict(raw))


def strip_internal_fields(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != _FALLBACK_KEY}


__all__ = [
    "Decision",
    "EvidenceGradeModel",
    "EvidenceVerdict",
    "FallbackResponseModel",
    "FallbackTrace",
    "GraphState",
    "ResponseModel",
    "Runtime",
    "attach_fallback_metadata",
    "fallback_metadata",
    "strip_internal_fields",
]
