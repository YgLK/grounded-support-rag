"""Corpus-grounded eval example authoring tool.

Drafts new eval examples by running real retrieval against the live index,
picking the best retrieved chunk as the gold span, and asking the chat model
to draft a reference answer plus required-point alias groups constrained to
the retrieved chunk text. Every candidate records a ``provenance`` block
(seed, retrieved chunk_ids, source span text snippet) so labels are anchored
to the real pinned corpus and a human can verify them.

The drafting functions accept injectable retrieval and chat-callables so the
deterministic parts can be unit-tested with mock fixtures (no live LLM or
Postgres in the default suite).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from support_graph.evaluation.eval_examples import (
    Provenance,
    append_candidates,
    load_candidates,
)
from support_graph.logging_utils import get_logger
from support_graph.retrieval.retrieve import retrieve_chunks
from support_graph.types import AnswerType

__all__ = [
    "SeedTopic",
    "DraftedExampleFields",
    "DraftResult",
    "RetrieverFunc",
    "ChatDraftFunc",
    "load_seed_topics",
    "write_seed_topics",
    "build_seed_example",
    "select_gold_chunk",
    "draft_example",
    "draft_examples",
    "append_candidate_rows",
    "load_candidate_rows",
    "build_default_chat_draft",
    "DRAFT_PROMPT_TEMPLATE",
]

logger = get_logger(__name__)


@dataclass(slots=True)
class SeedTopic:
    """A seed question/topic for drafting a new eval example."""

    seed_id: str
    question: str
    answer_type: AnswerType
    domain: str = "kubernetes"
    expected_doc_id: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "seed_id": self.seed_id,
            "question": self.question,
            "answer_type": self.answer_type,
            "domain": self.domain,
        }
        if self.expected_doc_id:
            payload["expected_doc_id"] = self.expected_doc_id
        if self.notes:
            payload["notes"] = self.notes
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SeedTopic":
        return cls(
            seed_id=str(payload["seed_id"]),
            question=str(payload["question"]),
            answer_type=str(payload.get("answer_type") or "definition"),  # type: ignore[arg-type]
            domain=str(payload.get("domain") or "kubernetes"),
            expected_doc_id=payload.get("expected_doc_id"),
            notes=payload.get("notes"),
        )


def load_seed_topics(path: str | Path) -> list[SeedTopic]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Seed file not found: {path}")
    topics: list[SeedTopic] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            topics.append(SeedTopic.from_dict(json.loads(line)))
    return topics


def write_seed_topics(topics: list[SeedTopic], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for topic in topics:
            handle.write(json.dumps(topic.to_dict(), ensure_ascii=True))
            handle.write("\n")


class DraftedExampleFields(BaseModel):
    """Structured output schema for the drafting chat model."""

    reference_answer: str = Field(
        description="A short grounded reference answer to the seed question."
    )
    required_points: list[list[str]] = Field(
        description=(
            "Required-point alias groups. Each group is a list of alternative "
            "phrases; at least one phrase per group must use tokens that appear "
            "in the provided source chunk text."
        )
    )
    expected_sources: list[str] = Field(
        description="doc_ids that should be retrieved/cited (from the source chunks)."
    )
    acceptable_sources: list[str] = Field(
        default_factory=list,
        description="Alternate doc_ids that are valid but not required.",
    )
    forbidden_claims: list[str] = Field(
        default_factory=list,
        description="Claims the answer must NOT make (e.g., common misconceptions).",
    )


@dataclass(slots=True)
class DraftResult:
    candidate: dict[str, Any]
    provenance: Provenance


class RetrieverFunc(Protocol):
    def __call__(
        self, *, example: dict[str, Any], config: Any, top_k: int, candidate_k: int
    ) -> list[dict[str, Any]]: ...


class ChatDraftFunc(Protocol):
    async def __call__(
        self, *, seed: SeedTopic, chunks: list[dict[str, Any]], config: Any
    ) -> DraftedExampleFields: ...


def build_seed_example(seed: SeedTopic) -> dict[str, Any]:
    """Build a minimal retrieval example dict from a seed topic."""
    return {
        "domain": seed.domain,
        "latest_user_utterance": seed.question,
        "latest_user_turn_id": 1,
        "turns_before_target": [
            {"turn_id": 1, "role": "user", "utterance": seed.question}
        ],
        "target_mode": "answer",
    }


def select_gold_chunk(
    chunks: list[dict[str, Any]],
    *,
    expected_doc_id: str | None = None,
) -> dict[str, Any] | None:
    """Pick the best retrieved chunk as the gold span source.

    Prefers a chunk whose doc_id matches ``expected_doc_id`` when provided;
    otherwise takes the top-ranked chunk with at least one span_id.
    """
    if not chunks:
        return None
    if expected_doc_id:
        for chunk in chunks:
            if chunk.get("doc_id") == expected_doc_id and chunk.get("span_ids"):
                return chunk
    for chunk in chunks:
        if chunk.get("span_ids"):
            return chunk
    return chunks[0]


def _neighbor_span_ids(
    chunks: list[dict[str, Any]], gold_chunk: dict[str, Any]
) -> list[str]:
    """Span IDs from neighboring chunks in the same document as the gold chunk."""
    gold_doc_id = gold_chunk.get("doc_id")
    gold_section_id = gold_chunk.get("section_id")
    gold_span_ids = {str(sid) for sid in gold_chunk.get("span_ids", []) if sid}
    neighbors: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.get("doc_id") != gold_doc_id:
            continue
        if chunk.get("section_id") == gold_section_id:
            continue
        for span_id in chunk.get("span_ids", []) or []:
            sid = str(span_id)
            if sid and sid not in gold_span_ids and sid not in seen:
                neighbors.append(sid)
                seen.add(sid)
    return neighbors


def _candidate_id(seed: SeedTopic) -> str:
    return f"kubernetes::draft::{seed.seed_id}"


def _build_candidate(
    seed: SeedTopic,
    *,
    gold_chunk: dict[str, Any],
    acceptable_span_ids: list[str],
    drafted: DraftedExampleFields,
    provenance: Provenance,
) -> dict[str, Any]:
    gold_doc_id = str(gold_chunk.get("doc_id") or "")
    gold_span_ids = [str(sid) for sid in gold_chunk.get("span_ids", []) if sid][:1]
    expected_sources = list(drafted.expected_sources or [gold_doc_id])
    if gold_doc_id and gold_doc_id not in expected_sources:
        expected_sources.insert(0, gold_doc_id)
    reference_answer = drafted.reference_answer.strip()
    return {
        "example_id": _candidate_id(seed),
        "domain": seed.domain,
        "dial_id": f"draft-{seed.seed_id}",
        "target_turn_id": 2,
        "turns_before_target": [
            {
                "turn_id": 1,
                "role": "user",
                "da": "query_solution",
                "utterance": seed.question,
                "references": [],
            }
        ],
        "latest_user_turn_id": 1,
        "latest_user_utterance": seed.question,
        "target_turn": {
            "turn_id": 2,
            "role": "agent",
            "da": "respond_solution",
            "utterance": reference_answer,
            "references": [
                {
                    "label": "solution",
                    "id_sp": gold_span_ids[0] if gold_span_ids else "",
                    "doc_id": gold_doc_id,
                }
            ],
        },
        "target_mode": "answer",
        "gold_doc_ids": [gold_doc_id] if gold_doc_id else [],
        "gold_span_ids": gold_span_ids,
        "acceptable_span_ids": acceptable_span_ids,
        "expected_sources": expected_sources,
        "acceptable_sources": list(drafted.acceptable_sources or []),
        "required_points": drafted.required_points,
        "forbidden_claims": list(drafted.forbidden_claims or []),
        "answer_type": seed.answer_type,
        "provenance": provenance,
    }


def _default_retriever(
    *, example: dict[str, Any], config: Any, top_k: int, candidate_k: int
) -> list[dict[str, Any]]:
    return retrieve_chunks(
        example=example,
        config=config,
        top_k=top_k,
        candidate_k=candidate_k,
        rerank=True,
    )


async def draft_example(
    seed: SeedTopic,
    *,
    config: Any,
    retriever: RetrieverFunc | None = None,
    chat_draft: ChatDraftFunc,
    top_k: int = 5,
    candidate_k: int = 12,
) -> DraftResult | None:
    """Draft a single candidate example for one seed topic."""
    resolve_retriever = retriever or _default_retriever
    example = build_seed_example(seed)
    chunks = resolve_retriever(
        example=example, config=config, top_k=top_k, candidate_k=candidate_k
    )
    gold_chunk = select_gold_chunk(chunks, expected_doc_id=seed.expected_doc_id)
    if gold_chunk is None:
        logger.warning("Seed %s retrieved no chunks; skipping.", seed.seed_id)
        return None
    acceptable_span_ids = _neighbor_span_ids(chunks, gold_chunk)
    drafted = await chat_draft(seed=seed, chunks=chunks, config=config)
    gold_span_id = next((str(sid) for sid in gold_chunk.get("span_ids", []) if sid), "")
    provenance: Provenance = {
        "seed": seed.to_dict(),
        "retrieved_chunk_ids": [
            str(chunk.get("chunk_id")) for chunk in chunks if chunk.get("chunk_id")
        ],
        "gold_chunk_id": str(gold_chunk.get("chunk_id") or ""),
        "gold_doc_id": str(gold_chunk.get("doc_id") or ""),
        "gold_span_id": gold_span_id,
        "source_span_text_snippet": str(gold_chunk.get("text", ""))[:400],
    }
    candidate = _build_candidate(
        seed,
        gold_chunk=gold_chunk,
        acceptable_span_ids=acceptable_span_ids,
        drafted=drafted,
        provenance=provenance,
    )
    return DraftResult(candidate=candidate, provenance=provenance)


async def draft_examples(
    seeds: list[SeedTopic],
    *,
    config: Any,
    retriever: RetrieverFunc | None = None,
    chat_draft: ChatDraftFunc,
    top_k: int = 5,
    candidate_k: int = 12,
) -> list[DraftResult]:
    """Draft candidate examples for a list of seed topics."""
    results: list[DraftResult] = []
    for seed in seeds:
        result = await draft_example(
            seed,
            config=config,
            retriever=retriever,
            chat_draft=chat_draft,
            top_k=top_k,
            candidate_k=candidate_k,
        )
        if result is not None:
            results.append(result)
    return results


def append_candidate_rows(
    results: list[DraftResult], path: str | Path
) -> list[dict[str, Any]]:
    return append_candidates([result.candidate for result in results], path)


def load_candidate_rows(path: str | Path) -> list[dict[str, Any]]:
    return load_candidates(path)


DRAFT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You draft evaluation examples for a Kubernetes support RAG system. "
                "Use ONLY the provided source chunk text to ground the answer and the "
                "required-point alias groups. Each required-point alias group is a list "
                "of alternative phrases; at least one phrase per group MUST use tokens "
                "that literally appear in the source text. Do not invent doc_ids or "
                "span IDs; use only the doc_ids listed in the source chunks. "
                "Keep the reference answer short and grounded."
            ),
        ),
        (
            "user",
            (
                "Seed question: {question}\n"
                "Answer type: {answer_type}\n\n"
                "Source chunks (doc_id | text):\n{chunks_text}\n\n"
                "Draft the eval example fields."
            ),
        ),
    ]
)


def _render_chunks_text(chunks: list[dict[str, Any]], *, limit: int = 2000) -> str:
    parts: list[str] = []
    total = 0
    for chunk in chunks:
        doc_id = chunk.get("doc_id") or ""
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        line = f"{doc_id} | {text}"
        if total + len(line) > limit:
            line = line[: max(0, limit - total)]
        parts.append(line)
        total += len(line)
        if total >= limit:
            break
    return "\n".join(parts)


def build_default_chat_draft(chat_model: Any) -> ChatDraftFunc:
    """Build a ChatDraftFunc that uses a chat model with structured output.

    The chat model must support ``with_structured_output(DraftedExampleFields,
    method="json_schema")`` (the same structured-output contract the runtime
    nodes rely on). This is the live-LLM path; unit tests inject a mock
    ChatDraftFunc instead.
    """
    chain = DRAFT_PROMPT_TEMPLATE | chat_model.with_structured_output(
        DraftedExampleFields, method="json_schema"
    )

    async def _draft(
        *, seed: SeedTopic, chunks: list[dict[str, Any]], config: Any
    ) -> DraftedExampleFields:
        payload = {
            "question": seed.question,
            "answer_type": seed.answer_type,
            "chunks_text": _render_chunks_text(chunks),
        }
        try:
            return await chain.ainvoke(
                payload, config={"run_name": "support_graph.DraftedExampleFields"}
            )
        except TypeError as exc:
            if "unexpected keyword argument 'config'" not in str(exc):
                raise
            return await chain.ainvoke(payload)

    return _draft
