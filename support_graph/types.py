"""Shared enums and literal aliases for public SupportGraph contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Iterable, Literal, Protocol, Self, TypeAlias, TypedDict


class ChoiceStrEnum(StrEnum):
    @classmethod
    def values(cls) -> tuple[str, ...]:
        return tuple(item.value for item in cls)

    @classmethod
    def value_set(cls) -> frozenset[str]:
        return frozenset(cls.values())

    @classmethod
    def parse(cls, value: str | Self) -> Self:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            supported = ", ".join(sorted(cls.value_set()))
            raise ValueError(
                f"Unsupported {cls.__name__.lower()} '{value}'. Expected one of: {supported}"
            ) from exc


class Domain(ChoiceStrEnum):
    DMV = "dmv"
    KUBERNETES = "kubernetes"
    SSA = "ssa"
    STUDENTAID = "studentaid"
    VA = "va"


class DatasetSplit(ChoiceStrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class EvalSubset(ChoiceStrEnum):
    SMOKE = "smoke"
    FROZEN_EXPERIMENT = "frozen_experiment"
    FULL_VALIDATION = "full_validation"


DomainLike: TypeAlias = Domain | str
DatasetSplitLike: TypeAlias = DatasetSplit | str
EvalSubsetLike: TypeAlias = EvalSubset | str
TargetMode = Literal["answer", "follow_up"]
TurnRole = Literal["agent", "user"]
QueryContextRole = Literal["agent", "user", "unknown"]


class Reference(TypedDict):
    """Reference to a document span in MultiDoc2Dial."""

    label: str
    id_sp: str
    doc_id: str


class DialogueTurn(TypedDict):
    """A single turn in a dialogue with normalized fields."""

    turn_id: int | None
    role: TurnRole
    da: str
    utterance: str
    references: list[Reference]


class Dialogue(TypedDict):
    """A complete dialogue with domain and turns."""

    domain: str
    dial_id: str
    turns: list[DialogueTurn]


class DocumentSpan(TypedDict):
    """A span within a document with position and hierarchy information."""

    id_sp: str
    tag: str
    start_sp: int | None
    end_sp: int | None
    text_sp: str
    title: str
    parent_titles: list[str]
    id_sec: str
    start_sec: int | None
    end_sec: int | None
    text_sec: str


class Document(TypedDict):
    """A complete document with metadata and spans."""

    domain: str
    doc_id: str
    title: str
    doc_text: str
    doc_html_ts: str
    doc_html_raw: str
    spans: list[DocumentSpan]
    raw_spans: dict[str, Any]


class ChunkRecord(TypedDict):
    """A retrieval chunk with section and span metadata."""

    chunk_id: str
    domain: str
    doc_id: str
    doc_title: str
    section_id: str
    section_title: str
    parent_titles: list[str]
    subchunk_index: int
    text: str
    span_ids: list[str]
    token_count: int
    start_sec: int | None
    end_sec: int | None


class Example(TypedDict):
    """A turn-level example for evaluation."""

    example_id: str
    domain: str
    dial_id: str
    target_turn_id: int
    turns_before_target: list[DialogueTurn]
    latest_user_turn_id: int | None
    latest_user_utterance: str | None
    target_turn: DialogueTurn
    target_mode: TargetMode
    gold_doc_ids: list[str]
    gold_span_ids: list[str]


class QueryContextTurn(TypedDict):
    """A single turn in the dialogue history used for search context."""

    role: str
    utterance: str


class QueryContext(TypedDict):
    """The structured representation of dialogue context used for retrieval."""

    domain: str
    latest_user_need: str
    last_agent_question: str
    carry_forward_context: list[QueryContextTurn]


class NormalizedRetrievalHit(TypedDict):
    """A standardized format for document chunks returned by any retriever."""

    rank: int
    original_rank: int
    chunk_id: str | None
    domain: str | None
    doc_id: str | None
    doc_title: str | None
    section_id: str | None
    section_title: str | None
    parent_titles: list[str]
    span_ids: list[str]
    token_count: int | None
    text: str
    score: float | None
    vector_distance: float | None


class Citation(TypedDict):
    """A citation to a document chunk with span IDs."""

    doc_id: str | None
    chunk_id: str | None
    span_ids: list[str]


class TraceSummary(TypedDict, total=False):
    """Summary of trace events for a prediction run."""

    retrieval_attempts: int
    final_query: str
    graph_path: list[str]
    latency_ms: float | None
    trace_path: str
    fallback_count: int
    fallback_nodes: list[str]
    fallbacks: list[dict[str, Any]]


class Prediction(TypedDict, total=False):
    """Model prediction with decision, response, and metadata."""

    decision: str
    response_text: str
    citations: list[Citation]
    retrieval_ranked_chunks: list[NormalizedRetrievalHit]
    retrieved_chunks: list[NormalizedRetrievalHit]
    trace_summary: TraceSummary
    latest_user_utterance: str | None
    query_context: QueryContext | None
    query: str | None
    refined_query: str | None


class MetricsDict(TypedDict, total=False):
    """Evaluation metrics for a prediction."""

    doc_recall_at_1: float | None
    doc_recall_at_3: float | None
    doc_recall_at_5: float | None
    doc_recall_at_10: float | None
    span_recall_at_5: float | None
    mrr_at_5: float | None
    rouge_l: float
    token_f1: float
    exact_match: float
    sacrebleu: float
    citation_coverage: float | None
    citations_valid: float
    end_to_end_success: float


class PredictionRecord(TypedDict, total=False):
    """Complete evaluation record for a prediction."""

    example_id: str
    target_mode: TargetMode
    target_turn_id: int
    latest_user_utterance: str | None
    gold_doc_ids: list[str]
    gold_span_ids: list[str]
    target_text: str
    decision: str
    response_text: str
    citations: list[Citation]
    retrieval_ranked_chunks: list[NormalizedRetrievalHit]
    retrieved_chunks: list[NormalizedRetrievalHit]
    trace_summary: TraceSummary
    metrics: MetricsDict
    failure_label: str | None
    runtime_error: dict[str, Any] | None


class VectorStoreLike(Protocol):
    """Protocol for vector store implementations."""

    def similarity_search_with_score(
        self,
        query: str,
        *,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[Any]: ...


class EmbeddingsLike(Protocol):
    """Protocol for embedding model implementations."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class ChatModelLike(Protocol):
    """Protocol for chat/LLM model implementations."""

    def invoke(self, messages: Any, **kwargs: Any) -> Any: ...
    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any: ...


class RetrieverLike(Protocol):
    """Protocol for retriever implementations (e.g., BM25)."""

    def invoke(self, query: str, **kwargs: Any) -> list[Any]: ...

    k: int


def parse_domain(value: DomainLike) -> Domain:
    return Domain.parse(value)


def parse_domains(values: Iterable[DomainLike]) -> tuple[Domain, ...]:
    ordered: list[Domain] = []
    seen: set[Domain] = set()
    for value in values:
        domain = parse_domain(value)
        if domain in seen:
            continue
        seen.add(domain)
        ordered.append(domain)
    return tuple(ordered)


def parse_dataset_split(value: DatasetSplitLike) -> DatasetSplit:
    return DatasetSplit.parse(value)


def parse_eval_subset(value: EvalSubsetLike) -> EvalSubset:
    return EvalSubset.parse(value)


__all__ = [
    "ChatModelLike",
    "ChoiceStrEnum",
    "ChunkRecord",
    "Citation",
    "DatasetSplit",
    "DatasetSplitLike",
    "Dialogue",
    "DialogueTurn",
    "Document",
    "DocumentSpan",
    "Domain",
    "DomainLike",
    "EmbeddingsLike",
    "EvalSubset",
    "EvalSubsetLike",
    "Example",
    "MetricsDict",
    "NormalizedRetrievalHit",
    "Prediction",
    "PredictionRecord",
    "QueryContext",
    "QueryContextRole",
    "QueryContextTurn",
    "Reference",
    "RetrieverLike",
    "TargetMode",
    "TraceSummary",
    "TurnRole",
    "VectorStoreLike",
    "parse_dataset_split",
    "parse_domain",
    "parse_domains",
    "parse_eval_subset",
]
