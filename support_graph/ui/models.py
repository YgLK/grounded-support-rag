"""Pydantic models for Workbench artifact loading and API responses."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Decision = Literal["answer", "clarify", "abstain"]
TargetMode = Literal["answer", "follow_up"]
EvidenceVerdict = Literal["sufficient", "partial", "insufficient"]
ResponseConfidence = Literal["high", "medium", "low"]
FailureLabel = Literal[
    "bad_clarification",
    "abstained_with_evidence",
    "wrong_doc",
    "missed_history",
    "right_doc_wrong_section",
    "weak_citations",
    "unsupported_answer",
    "runtime_error",
]
ReportType = Literal["ablation_summary", "comparison_report"]
SortOrder = Literal["asc", "desc"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderSnapshot(StrictModel):
    type: str
    chat_base_url: str | None = None
    embedding_type: str | None = None
    embedding_base_url: str | None = None
    chat_model: str | None = None
    embedding_model: str | None = None


class ChunkingConfig(StrictModel):
    strategy: str
    max_tokens_per_chunk: int


class RetrievalConfig(StrictModel):
    top_k: int | None
    candidate_k: int
    max_attempts: int
    use_history: bool
    content_only_reasoning: bool
    neighbor_expansion: bool


class GraphConfig(StrictModel):
    enable_retry: bool
    decision_policy_version: str


class CitationRecord(StrictModel):
    doc_id: str
    chunk_id: str
    span_ids: list[str] = Field(default_factory=list)


class ChunkRecordView(StrictModel):
    chunk_id: str
    doc_id: str
    doc_title: str | None
    section_id: str | None
    section_title: str | None
    parent_titles: list[str] = Field(default_factory=list)
    span_ids: list[str] = Field(default_factory=list)
    text: str
    domain: str | None = None
    rank: int | None = None
    score: float | None = None
    token_count: int | None = None
    subchunk_index: int | None = None
    start_sec: int | None = None
    end_sec: int | None = None


class RankedChunkRecordView(ChunkRecordView):
    rank: int


class FallbackRecord(StrictModel):
    used: bool
    node: str
    mode: str
    exception_type: str
    error: str


class TraceSummary(StrictModel):
    retrieval_attempts: int
    final_query: str
    graph_path: list[str] = Field(default_factory=list)
    latency_ms: float | None
    trace_path: str
    fallback_count: int
    fallback_nodes: list[str] = Field(default_factory=list)
    fallbacks: list[FallbackRecord] = Field(default_factory=list)
    observability: dict[str, object] | None = None


class PerExampleMetrics(StrictModel):
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


class RuntimeErrorInfo(StrictModel):
    exception_type: str
    error: str


class EvalRunManifest(StrictModel):
    run_id: str
    created_at: datetime
    dataset_root: str
    domains: list[str] = Field(default_factory=list)
    split: str
    eval_subset: str
    subset_label: str
    target_modes: list[str] = Field(default_factory=list)
    provider: ProviderSnapshot
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    graph: GraphConfig
    prompt_version: str
    notes: str
    ablation: dict[str, object] | None = None


class EvalRetrievalMetrics(StrictModel):
    doc_recall_at_1: float | None
    doc_recall_at_3: float | None
    doc_recall_at_5: float | None
    doc_recall_at_10: float | None
    span_recall_at_5: float | None
    mrr_at_5: float | None


class EvalGenerationAnswerMetrics(StrictModel):
    rouge_l: float | None
    token_f1: float | None
    exact_match: float | None
    sacrebleu: float | None
    citation_coverage: float | None
    end_to_end_success_rate: float | None


class EvalCounts(StrictModel):
    examples: int
    answer_examples: int
    follow_up_examples: int


class EvalGenerationMetrics(StrictModel):
    answer: EvalGenerationAnswerMetrics


class EvalRetrievalBreakdown(StrictModel):
    answer: EvalRetrievalMetrics
    follow_up: EvalRetrievalMetrics
    overall: EvalRetrievalMetrics


class DecisionDistribution(StrictModel):
    overall: dict[str, float] = Field(default_factory=dict)
    answer: dict[str, float] = Field(default_factory=dict)
    follow_up: dict[str, float] = Field(default_factory=dict)


class EvalLatencyMetrics(StrictModel):
    average: float | None
    p95: float | None


class EvalRunMetrics(StrictModel):
    counts: EvalCounts
    retrieval: EvalRetrievalBreakdown
    generation: EvalGenerationMetrics
    decision_distribution: DecisionDistribution
    latency_ms: EvalLatencyMetrics
    failure_counts: dict[str, int] = Field(default_factory=dict)


class TraceIndexEntry(StrictModel):
    example_id: str
    trace_file: str
    graph_path: list[str] = Field(default_factory=list)
    retrieval_attempts: int
    final_query: str
    decision: Decision
    failure_label: FailureLabel | None
    total_latency_ms: float
    node_latency_ms: dict[str, list[float]] = Field(default_factory=dict)
    retrieval_ranked_count: int
    retrieved_count: int
    fallback_count: int
    fallback_nodes: list[str] = Field(default_factory=list)


class PredictionRecord(StrictModel):
    example_id: str
    target_mode: TargetMode
    target_turn_id: int
    latest_user_utterance: str | None
    gold_doc_ids: list[str] = Field(default_factory=list)
    gold_span_ids: list[str] = Field(default_factory=list)
    target_text: str
    decision: Decision
    response_text: str
    citations: list[CitationRecord] = Field(default_factory=list)
    retrieval_ranked_chunks: list[RankedChunkRecordView] = Field(default_factory=list)
    retrieved_chunks: list[ChunkRecordView] = Field(default_factory=list)
    trace_summary: TraceSummary
    metrics: PerExampleMetrics
    failure_label: FailureLabel | None
    runtime_error: RuntimeErrorInfo | None = None


class RetrievalExampleRecord(StrictModel):
    example_id: str
    target_mode: TargetMode
    latest_user_utterance: str | None
    target_text: str
    gold_doc_ids: list[str] = Field(default_factory=list)
    gold_span_ids: list[str] = Field(default_factory=list)
    final_query: str | None
    retrieval_attempts: int | None
    retrieval_ranked_chunks: list[RankedChunkRecordView] = Field(default_factory=list)
    retrieved_chunks: list[ChunkRecordView] = Field(default_factory=list)
    doc_recall_at_3: float | None
    span_recall_at_5: float | None
    mrr_at_5: float | None
    failure_label: FailureLabel | None
    decision: Decision


class EvalReportManifest(StrictModel):
    report_id: str
    created_at: datetime
    report_type: ReportType
    title: str
    related_run_ids: list[str] = Field(default_factory=list)
    domain: str | None = None
    split: str | None = None
    subset_label: str | None = None
    notes: str | None = None


class StandaloneRunManifest(StrictModel):
    run_id: str
    created_at: datetime
    example_id: str
    domain: str
    provider: ProviderSnapshot
    prompt_version: str | None = None
    notes: str | None = None


class EvidenceGrade(StrictModel):
    verdict: EvidenceVerdict
    reason: str
    missing_information: list[str] = Field(default_factory=list)


class TraceEventSummary(StrictModel):
    graph_path: list[str] = Field(default_factory=list)
    retrieval_attempts: int
    final_query: str
    decision: Decision
    evidence_grade: dict[str, object] | None = None
    total_latency_ms: float
    node_latency_ms: dict[str, list[float]] = Field(default_factory=dict)
    retrieval_ranked_count: int
    retrieved_count: int
    fallback_count: int
    fallback_nodes: list[str] = Field(default_factory=list)
    fallbacks: list[FallbackRecord] = Field(default_factory=list)
    observability: dict[str, object] | None = None
    trace_path: str


class StandaloneRunResult(StrictModel):
    example_id: str
    latest_user_utterance: str | None
    decision: Decision
    response_text: str
    citations: list[CitationRecord] = Field(default_factory=list)
    confidence_label: ResponseConfidence
    retrieval_ranked_chunks: list[RankedChunkRecordView] = Field(default_factory=list)
    retrieved_chunks: list[ChunkRecordView] = Field(default_factory=list)
    evidence_grade: EvidenceGrade
    trace_summary: TraceSummary


class FailureRecordView(StrictModel):
    example_id: str
    target_mode: TargetMode
    failure_label: FailureLabel
    latest_user_utterance: str | None
    target_text: str
    final_decision: Decision
    response_text: str
    gold_doc_ids: list[str] = Field(default_factory=list)
    gold_span_ids: list[str] = Field(default_factory=list)
    doc_recall_at_3: float | None
    span_recall_at_5: float | None
    mrr_at_5: float | None
    citation_coverage: float | None
    citations_valid: float
    end_to_end_success: float
    trace_path: str


class EvalRunSummary(StrictModel):
    run_id: str
    created_at: datetime
    domains: list[str] = Field(default_factory=list)
    split: str
    eval_subset: str
    subset_label: str
    provider: ProviderSnapshot
    headline_retrieval: EvalRetrievalMetrics
    headline_generation: EvalGenerationAnswerMetrics
    failure_counts: dict[str, int] = Field(default_factory=dict)
    artifact_paths: dict[str, str] = Field(default_factory=dict)


class StandaloneRunSummary(StrictModel):
    run_id: str
    created_at: datetime
    example_id: str
    domain: str
    provider: ProviderSnapshot
    decision: Decision
    final_query: str
    trace_path: str
    artifact_paths: dict[str, str] = Field(default_factory=dict)


class EvalReportSummary(StrictModel):
    report_id: str
    created_at: datetime
    report_type: ReportType
    title: str
    related_run_ids: list[str] = Field(default_factory=list)
    domain: str | None = None
    split: str | None = None
    subset_label: str | None = None
    artifact_paths: dict[str, str] = Field(default_factory=dict)


class ArtifactExplorerView(StrictModel):
    eval_runs: list[EvalRunSummary] = Field(default_factory=list)
    standalone_runs: list[StandaloneRunSummary] = Field(default_factory=list)
    reports: list[EvalReportSummary] = Field(default_factory=list)


class EvalRunListView(StrictModel):
    items: list[EvalRunSummary] = Field(default_factory=list)


class StandaloneRunListView(StrictModel):
    items: list[StandaloneRunSummary] = Field(default_factory=list)


class EvalReportListView(StrictModel):
    items: list[EvalReportSummary] = Field(default_factory=list)


class EvalRunDetailView(StrictModel):
    summary: EvalRunSummary
    manifest: EvalRunManifest
    metrics: EvalRunMetrics
    summary_markdown: str
    summary_html: str
    failures: list[FailureRecordView] = Field(default_factory=list)
    trace_index_entries: list[TraceIndexEntry] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)


class EvalFailureTableView(StrictModel):
    run: EvalRunSummary
    items: list[FailureRecordView] = Field(default_factory=list)
    total_failures: int
    failure_counts: dict[str, int] = Field(default_factory=dict)
    applied_failure_label: FailureLabel | None = None
    applied_target_mode: TargetMode | None = None


class TraceEventView(StrictModel):
    timestamp: str | None = None
    node: str | None = None
    payload: dict[str, object] = Field(default_factory=dict)


class ExampleDetailView(StrictModel):
    run: EvalRunSummary
    prediction: PredictionRecord
    retrieval_example: RetrievalExampleRecord | None = None
    trace_index_entry: TraceIndexEntry
    trace_summary: TraceEventSummary
    trace_events: list[TraceEventView] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)


class StandaloneRunDetailView(StrictModel):
    summary: StandaloneRunSummary
    manifest: StandaloneRunManifest
    result: StandaloneRunResult
    trace_summary: TraceEventSummary
    trace_events: list[TraceEventView] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)


class EvalReportDetailView(StrictModel):
    summary: EvalReportSummary
    manifest: EvalReportManifest
    report_markdown: str
    report_html: str
    artifact_paths: dict[str, str] = Field(default_factory=dict)
