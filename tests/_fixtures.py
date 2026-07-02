"""Typed fixture builders for tests.

These builders return fully-typed TypedDict instances so that test call sites
satisfy production TypedDict contracts without resorting to ``cast`` or
``# type: ignore``.
"""

from __future__ import annotations

from typing import Any

from support_graph.types import (
    AnswerType,
    Citation,
    ChunkRecord,
    DialogueTurn,
    Document,
    DocumentSpan,
    Example,
    NormalizedRetrievalHit,
    QueryContext,
    QueryContextTurn,
    Reference,
    TargetMode,
    TurnRole,
)


def _span(
    span_id: str,
    *,
    tag: str = "section",
    text_sp: str = "Span text.",
    title: str = "Section",
    parent_titles: list[str] | None = None,
    id_sec: str = "section",
    text_sec: str | None = None,
    start_sp: int | None = 0,
    end_sp: int | None = 10,
    start_sec: int | None = 0,
    end_sec: int | None = 10,
) -> DocumentSpan:
    return {
        "id_sp": span_id,
        "tag": tag,
        "start_sp": start_sp,
        "end_sp": end_sp,
        "text_sp": text_sp,
        "title": title,
        "parent_titles": parent_titles if parent_titles is not None else [],
        "id_sec": id_sec,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "text_sec": text_sec if text_sec is not None else text_sp,
    }


def _document(
    doc_id: str,
    *,
    domain: str = "kubernetes",
    title: str = "Doc",
    doc_text: str = "",
    spans: list[DocumentSpan] | None = None,
    doc_html_ts: str = "",
    doc_html_raw: str = "",
    raw_spans: dict[str, Any] | None = None,
) -> Document:
    return {
        "domain": domain,
        "doc_id": doc_id,
        "title": title,
        "doc_text": doc_text,
        "doc_html_ts": doc_html_ts,
        "doc_html_raw": doc_html_raw,
        "spans": spans if spans is not None else [],
        "raw_spans": raw_spans if raw_spans is not None else {},
    }


def _chunk(
    chunk_id: str,
    *,
    domain: str = "kubernetes",
    doc_id: str = "doc",
    doc_title: str = "Doc",
    section_id: str = "section",
    section_title: str = "Section",
    parent_titles: list[str] | None = None,
    subchunk_index: int = 0,
    text: str = "Chunk text.",
    span_ids: list[str] | None = None,
    token_count: int = 4,
    start_sec: int | None = 0,
    end_sec: int | None = 10,
) -> ChunkRecord:
    return {
        "chunk_id": chunk_id,
        "domain": domain,
        "doc_id": doc_id,
        "doc_title": doc_title,
        "section_id": section_id,
        "section_title": section_title,
        "parent_titles": parent_titles if parent_titles is not None else [],
        "subchunk_index": subchunk_index,
        "text": text,
        "span_ids": span_ids if span_ids is not None else [],
        "token_count": token_count,
        "start_sec": start_sec,
        "end_sec": end_sec,
    }


def _hit(
    chunk_id: str | None = None,
    *,
    rank: int = 1,
    original_rank: int | None = None,
    domain: str | None = "kubernetes",
    doc_id: str | None = "doc",
    doc_title: str | None = "Doc",
    section_id: str | None = "section",
    section_title: str | None = "Section",
    parent_titles: list[str] | None = None,
    span_ids: list[str] | None = None,
    token_count: int | None = 4,
    text: str = "Chunk text.",
    score: float | None = 0.5,
    vector_distance: float | None = 0.5,
) -> NormalizedRetrievalHit:
    return {
        "rank": rank,
        "original_rank": original_rank if original_rank is not None else rank,
        "chunk_id": chunk_id,
        "domain": domain,
        "doc_id": doc_id,
        "doc_title": doc_title,
        "section_id": section_id,
        "section_title": section_title,
        "parent_titles": parent_titles if parent_titles is not None else [],
        "span_ids": span_ids if span_ids is not None else [],
        "token_count": token_count,
        "text": text,
        "score": score,
        "vector_distance": vector_distance,
    }


def _citation(
    *,
    doc_id: str | None = "doc",
    chunk_id: str | None = None,
    span_ids: list[str] | None = None,
) -> Citation:
    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "span_ids": span_ids if span_ids is not None else [],
    }


def _query_context_turn(
    *,
    role: str = "user",
    utterance: str = "What is a Pod?",
) -> QueryContextTurn:
    return {"role": role, "utterance": utterance}


def _dialogue_turn(
    *,
    turn_id: int | None = 1,
    role: TurnRole = "user",
    da: str = "",
    utterance: str = "",
    references: list[Reference] | None = None,
) -> DialogueTurn:
    return {
        "turn_id": turn_id,
        "role": role,
        "da": da,
        "utterance": utterance,
        "references": references if references is not None else [],
    }


def _query_context(
    *,
    domain: str = "kubernetes",
    latest_user_need: str = "What is a Pod?",
    last_agent_question: str = "",
    carry_forward_context: list[QueryContextTurn] | None = None,
) -> QueryContext:
    return {
        "domain": domain,
        "latest_user_need": latest_user_need,
        "last_agent_question": last_agent_question,
        "carry_forward_context": carry_forward_context
        if carry_forward_context is not None
        else [],
    }


def _example(
    example_id: str,
    *,
    domain: str = "kubernetes",
    dial_id: str = "dial",
    target_turn_id: int = 1,
    turns_before_target: list[DialogueTurn] | None = None,
    latest_user_turn_id: int | None = 1,
    latest_user_utterance: str | None = "What is a Pod?",
    target_turn: DialogueTurn | None = None,
    target_mode: TargetMode = "answer",
    gold_doc_ids: list[str] | None = None,
    gold_span_ids: list[str] | None = None,
    expected_sources: list[str] | None = None,
    acceptable_sources: list[str] | None = None,
    required_points: list[str | list[str]] | None = None,
    acceptable_span_ids: list[str] | None = None,
    forbidden_claims: list[str] | None = None,
    answer_type: AnswerType | None = None,
) -> Example:
    example: Example = {
        "example_id": example_id,
        "domain": domain,
        "dial_id": dial_id,
        "target_turn_id": target_turn_id,
        "turns_before_target": turns_before_target
        if turns_before_target is not None
        else [],
        "latest_user_turn_id": latest_user_turn_id,
        "latest_user_utterance": latest_user_utterance,
        "target_turn": target_turn if target_turn is not None else _dialogue_turn(),
        "target_mode": target_mode,
        "gold_doc_ids": gold_doc_ids if gold_doc_ids is not None else [],
        "gold_span_ids": gold_span_ids if gold_span_ids is not None else [],
    }
    if expected_sources is not None:
        example["expected_sources"] = expected_sources
    if acceptable_sources is not None:
        example["acceptable_sources"] = acceptable_sources
    if required_points is not None:
        example["required_points"] = required_points
    if acceptable_span_ids is not None:
        example["acceptable_span_ids"] = acceptable_span_ids
    if forbidden_claims is not None:
        example["forbidden_claims"] = forbidden_claims
    if answer_type is not None:
        example["answer_type"] = answer_type
    return example
