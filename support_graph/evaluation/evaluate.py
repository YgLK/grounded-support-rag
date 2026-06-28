"""Phase 4 evaluation harness."""

from __future__ import annotations

import asyncio
import csv
import inspect
import json
import math
import re
from collections import Counter
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from statistics import mean
from typing import Any

from sacrebleu.metrics import BLEU

from support_graph.artifacts import (
    build_trace_file,
    eval_run_artifacts,
    project_relative_path,
    resolve_project_path,
)
from support_graph.config.runtime import (
    RuntimeConfig,
    RuntimeExperimentOverrides,
    apply_runtime_experiment_overrides,
)
from support_graph.config.settings import Settings
from support_graph.logging_utils import get_logger
from support_graph.data.dataset import load_dialogues
from support_graph.data.eval_subsets import load_subset_jsonl
from support_graph.data.examples import (
    build_turn_examples,
    load_examples_jsonl,
    write_examples_jsonl,
)
from support_graph.providers import (
    chat_provider as resolved_chat_provider,
    chat_provider_base_url,
    embedding_provider as resolved_embedding_provider,
    embedding_provider_base_url,
)
from support_graph.runtime.graph import resolve_runtime_resources_async, run_graph_async
from support_graph.runtime.traces import (
    load_trace_events,
    summarize_trace_events,
    write_trace_event,
)
from support_graph.types import (
    Citation,
    DatasetSplit,
    DatasetSplitLike,
    DomainLike,
    EvalSubset,
    EvalSubsetLike,
    Example,
    NormalizedRetrievalHit,
    parse_dataset_split,
    parse_domain,
    parse_eval_subset,
)

__all__ = [
    "load_eval_examples",
    "build_eval_config",
    "with_config_overrides",
    "build_run_id",
    "evaluate_examples_async",
    "evaluate_split_async",
    "doc_recall_at_k",
    "span_recall_at_k",
    "mrr_at_k",
    "citation_coverage",
    "citations_map_to_retrieved",
    "rouge_l_f1",
    "token_f1",
    "exact_match",
    "sacrebleu_score",
    "hit_at_k",
    "precision_at_k",
    "graded_mrr_at_k",
    "ndcg_at_k",
    "required_point_coverage",
    "forbidden_claims_hit",
    "has_rag_eval_fields",
    "RAGTriadJudge",
]


END_TO_END_TEXT_THRESHOLD = 0.35
_SACREBLEU = BLEU(effective_order=True)
logger = get_logger(__name__)
MANUAL_REVIEW_COLUMNS = [
    "run_id",
    "example_id",
    "target_mode",
    "decision",
    "failure_label",
    "latest_user_utterance",
    "response_text",
    "gold_doc_ids",
    "gold_span_ids",
    "expected_sources",
    "acceptable_sources",
    "required_points",
    "forbidden_claims",
    "answer_type",
    "final_query",
    "retrieval_attempts",
    "retrieval_ranked_chunk_ids",
    "retrieved_chunk_ids",
    "citation_chunk_ids",
    "context_relevance",
    "faithfulness",
    "answer_relevance",
    "required_points_covered",
    "forbidden_claims_present",
    "judge_rationale",
    "decision_correct",
    "evidence_relevant",
    "no_unsupported_claims",
    "clear",
    "citations_useful",
    "reviewer_notes",
]


def _normalize_text(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    predicted_tokens = _normalize_text(prediction)
    reference_tokens = _normalize_text(reference)
    if not predicted_tokens or not reference_tokens:
        return 0.0
    lcs = _lcs_length(predicted_tokens, reference_tokens)
    precision = lcs / len(predicted_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, reference: str) -> float:
    predicted_tokens = Counter(_normalize_text(prediction))
    reference_tokens = Counter(_normalize_text(reference))
    if not predicted_tokens or not reference_tokens:
        return 0.0
    overlap = sum((predicted_tokens & reference_tokens).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(predicted_tokens.values())
    recall = overlap / sum(reference_tokens.values())
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    normalized_prediction = " ".join(_normalize_text(prediction))
    normalized_reference = " ".join(_normalize_text(reference))
    if not normalized_prediction or not normalized_reference:
        return 0.0
    return 1.0 if normalized_prediction == normalized_reference else 0.0


def sacrebleu_score(prediction: str, reference: str) -> float:
    cleaned_prediction = str(prediction).strip()
    cleaned_reference = str(reference).strip()
    if not cleaned_prediction or not cleaned_reference:
        return 0.0
    return (
        _SACREBLEU.sentence_score(cleaned_prediction, [cleaned_reference]).score / 100.0
    )


def doc_recall_at_k(
    gold_doc_ids: list[str], retrieved_chunks: list[NormalizedRetrievalHit], k: int = 3
) -> float | None:
    gold = {doc_id for doc_id in gold_doc_ids if doc_id}
    if not gold:
        return None
    retrieved = {
        chunk.get("doc_id") for chunk in retrieved_chunks[:k] if chunk.get("doc_id")
    }
    return 1.0 if gold & retrieved else 0.0


def span_recall_at_k(
    gold_span_ids: list[str], retrieved_chunks: list[NormalizedRetrievalHit], k: int = 5
) -> float | None:
    gold = {span_id for span_id in gold_span_ids if span_id}
    if not gold:
        return None
    retrieved_spans: set[str] = set()
    for chunk in retrieved_chunks[:k]:
        retrieved_spans.update(str(span_id) for span_id in chunk.get("span_ids", []))
    return len(gold & retrieved_spans) / len(gold)


def mrr_at_k(
    gold_doc_ids: list[str], retrieved_chunks: list[NormalizedRetrievalHit], k: int = 5
) -> float | None:
    gold = {doc_id for doc_id in gold_doc_ids if doc_id}
    if not gold:
        return None
    for rank, chunk in enumerate(retrieved_chunks[:k], start=1):
        if chunk.get("doc_id") in gold:
            return 1.0 / rank
    return 0.0


def _graded_relevance_map(
    expected_sources: list[str], acceptable_sources: list[str]
) -> dict[str, int]:
    """Map doc_id -> graded relevance: expected=2, acceptable=1, else 0."""
    grades: dict[str, int] = {}
    for doc_id in acceptable_sources:
        if doc_id:
            grades[doc_id] = 1
    for doc_id in expected_sources:
        if doc_id:
            grades[doc_id] = 2
    return grades


def hit_at_k(
    expected_sources: list[str],
    acceptable_sources: list[str],
    retrieved_chunks: list[NormalizedRetrievalHit],
    k: int = 5,
) -> float | None:
    """1.0 if any expected or acceptable source appears in top-k, else 0.0."""
    relevant = {doc_id for doc_id in (expected_sources + acceptable_sources) if doc_id}
    if not relevant:
        return None
    retrieved = {
        chunk.get("doc_id") for chunk in retrieved_chunks[:k] if chunk.get("doc_id")
    }
    return 1.0 if relevant & retrieved else 0.0


def precision_at_k(
    expected_sources: list[str],
    acceptable_sources: list[str],
    retrieved_chunks: list[NormalizedRetrievalHit],
    k: int = 5,
) -> float | None:
    """Fraction of top-k retrieved docs that are expected or acceptable sources."""
    relevant = {doc_id for doc_id in (expected_sources + acceptable_sources) if doc_id}
    if not relevant:
        return None
    top_k = retrieved_chunks[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for chunk in top_k if chunk.get("doc_id") in relevant)
    return hits / len(top_k)


def graded_mrr_at_k(
    expected_sources: list[str],
    acceptable_sources: list[str],
    retrieved_chunks: list[NormalizedRetrievalHit],
    k: int = 5,
) -> float | None:
    """MRR using graded relevance: first hit weighted by grade (2 / rank for expected, 1 / rank for acceptable)."""
    grades = _graded_relevance_map(expected_sources, acceptable_sources)
    if not grades:
        return None
    best = 0.0
    for rank, chunk in enumerate(retrieved_chunks[:k], start=1):
        doc_id = chunk.get("doc_id")
        if doc_id in grades:
            best = grades[doc_id] / rank
            break
    # Normalize to [0, 1] using max possible grade (2).
    return best / 2.0


def ndcg_at_k(
    expected_sources: list[str],
    acceptable_sources: list[str],
    retrieved_chunks: list[NormalizedRetrievalHit],
    k: int = 5,
) -> float | None:
    """Discounted cumulative gain for expected sources plus alternate sources.

    Expected sources are the ideal target. Acceptable sources are valid alternates
    that should earn partial credit when expected sources are not retrieved, but
    they are not extra required documents in the ideal ranking.
    """
    grades = _graded_relevance_map(expected_sources, acceptable_sources)
    if not grades:
        return None
    top_k = retrieved_chunks[:k]
    dcg = 0.0
    seen_doc_ids: set[str] = set()
    for rank, chunk in enumerate(top_k, start=1):
        doc_id = chunk.get("doc_id")
        if not doc_id or doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        grade = grades.get(doc_id, 0)
        if grade > 0:
            dcg += (2.0**grade - 1.0) / math.log2(rank + 1)
    expected_grades = [2 for doc_id in expected_sources if doc_id]
    ideal_grades = expected_grades[:k] or sorted(grades.values(), reverse=True)[:k]
    idcg = sum(
        (2.0**grade - 1.0) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
        if grade > 0
    )
    return dcg / idcg if idcg > 0 else 0.0


def required_point_coverage(required_points: list[str], answer: str) -> float | None:
    """Deterministic v1: fraction of required points covered by normalized token overlap.

    A required point is "covered" if every content token in the point appears in the
    answer (bag-of-words containment). Returns None when no required points are
    declared so callers can skip aggregation.
    """
    if not required_points:
        return None
    answer_tokens = set(_normalize_text(answer))
    covered = 0
    for point in required_points:
        point_tokens = _normalize_text(point)
        if not point_tokens:
            continue
        if all(token in answer_tokens for token in point_tokens):
            covered += 1
    declared = sum(1 for point in required_points if _normalize_text(point))
    return covered / declared if declared else 0.0


def forbidden_claims_hit(forbidden_claims: list[str], answer: str) -> float | None:
    """Fraction of forbidden claims present in the answer (0 = clean, 1 = all present)."""
    if not forbidden_claims:
        return None
    answer_tokens = set(_normalize_text(answer))
    hits = 0
    declared = 0
    for claim in forbidden_claims:
        claim_tokens = _normalize_text(claim)
        if not claim_tokens:
            continue
        declared += 1
        if all(token in answer_tokens for token in claim_tokens):
            hits += 1
    return hits / declared if declared else 0.0


def has_rag_eval_fields(example: dict) -> bool:
    """True when an example declares any RAG-triad eval field."""
    return any(
        example.get(field)
        for field in (
            "expected_sources",
            "acceptable_sources",
            "required_points",
            "forbidden_claims",
            "answer_type",
        )
    )


class RAGTriadJudge:
    """Optional LLM-as-a-judge adapter producing RAG-triad verdicts.

    Disabled by default. When constructed with an inner ``RAGJudge`` (or any
    object exposing the same async ``faithfulness``/``answer_relevance``/
    ``context_relevance`` methods), it returns a verdict dict with
    ``faithful``, ``answer_relevant``, ``required_points_covered``,
    ``unsupported_claims``, and ``rationale`` fields. When the inner judge is
    ``None``, :meth:`evaluate` returns ``None`` so callers fall back to the
    deterministic v1 metrics.
    """

    def __init__(self, inner: Any | None = None):
        self._inner = inner

    @classmethod
    def disabled(cls) -> RAGTriadJudge:
        return cls(inner=None)

    @property
    def enabled(self) -> bool:
        return self._inner is not None

    async def evaluate(
        self,
        *,
        query: str,
        answer: str,
        context: str,
        required_points: list[str],
        forbidden_claims: list[str],
    ) -> dict | None:
        if not self.enabled:
            return None
        inner = self._inner
        faithfulness_result = await inner.faithfulness(context, answer)
        answer_relevance_result = await inner.answer_relevance(query, answer)
        context_relevance_result = await inner.context_relevance(query, context)
        return {
            "faithful": float(faithfulness_result.score),
            "answer_relevant": float(answer_relevance_result.score),
            "context_relevant": float(context_relevance_result.score),
            "required_points_covered": None,
            "unsupported_claims": [],
            "rationale": " | ".join(
                filter(
                    None,
                    [
                        f"faithfulness: {faithfulness_result.reason}",
                        f"answer_relevance: {answer_relevance_result.reason}",
                        f"context_relevance: {context_relevance_result.reason}",
                    ],
                )
            ),
        }


def _join_chunk_text(chunks: list[dict], *, limit: int = 1500) -> str:
    parts: list[str] = []
    total = 0
    for chunk in chunks:
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        if total + len(text) > limit:
            text = text[: max(0, limit - total)]
        parts.append(text)
        total += len(text)
        if total >= limit:
            break
    return "\n\n".join(parts)


def citation_coverage(
    gold_span_ids: list[str], citations: list[Citation]
) -> float | None:
    gold = {span_id for span_id in gold_span_ids if span_id}
    if not gold:
        return None
    cited_spans: set[str] = set()
    for citation in citations:
        cited_spans.update(str(span_id) for span_id in citation.get("span_ids", []))
    return len(gold & cited_spans) / len(gold)


def citations_map_to_retrieved(
    citations: list[Citation], retrieved_chunks: list[NormalizedRetrievalHit]
) -> float:
    retrieved_chunk_ids = {
        chunk.get("chunk_id") for chunk in retrieved_chunks if chunk.get("chunk_id")
    }
    citation_chunk_ids = [
        citation.get("chunk_id") for citation in citations if citation.get("chunk_id")
    ]
    if not citation_chunk_ids:
        return False
    return all(chunk_id in retrieved_chunk_ids for chunk_id in citation_chunk_ids)


def _retrieval_metric_available(retrieval_top_k: int | None, *, k: int) -> bool:
    return retrieval_top_k is None or int(retrieval_top_k) >= k


def _failure_label(example: dict, prediction: dict, metrics: dict) -> str | None:
    target_mode = example.get("target_mode")
    decision = prediction.get("decision")
    if has_rag_eval_fields(example):
        return _rag_failure_label(example, prediction, metrics)
    match target_mode:
        case "answer":
            if decision == "clarify":
                return "bad_clarification"
            if decision == "abstain":
                if ((metrics.get("doc_recall_at_3") or 0.0) > 0) or (
                    (metrics.get("span_recall_at_5") or 0.0) > 0
                ):
                    return "abstained_with_evidence"
                return "wrong_doc"
            if (metrics.get("doc_recall_at_3") or 0.0) == 0.0:
                if len(example.get("turns_before_target", [])) >= 3:
                    return "missed_history"
                return "wrong_doc"
            if (metrics.get("doc_recall_at_3") or 0.0) > 0.0 and (
                metrics.get("span_recall_at_5") or 0.0
            ) == 0.0:
                return "right_doc_wrong_section"
            if (metrics.get("citations_valid") or 0.0) == 0.0 or (
                metrics.get("citation_coverage") or 0.0
            ) < 1.0:
                return "weak_citations"
            if (metrics.get("end_to_end_success") or 0.0) == 0.0:
                return "unsupported_answer"
            return None
        case "follow_up":
            if (metrics.get("doc_recall_at_3") or 0.0) == 0.0:
                return "wrong_doc"
            return None
    raise ValueError(f"Unknown target_mode: {target_mode}")


def _rag_failure_label(example: dict, prediction: dict, metrics: dict) -> str | None:
    """Failure labels for RAG-triad examples.

    Splits the legacy ``unsupported_answer`` bucket into clearer labels:
    ``retrieval_miss``, ``right_source_wrong_section``, ``unfaithful_answer``,
    ``incomplete_answer``, ``irrelevant_answer``, and ``weak_citations``.
    """
    target_mode = example.get("target_mode")
    decision = prediction.get("decision")
    if target_mode != "answer":
        if (metrics.get("hit_at_k") or 0.0) == 0.0 and (
            metrics.get("doc_recall_at_3") or 0.0
        ) == 0.0:
            return "retrieval_miss"
        return None
    if decision == "clarify":
        return "irrelevant_answer"
    if decision == "abstain":
        if (metrics.get("hit_at_k") or 0.0) > 0 or (
            metrics.get("doc_recall_at_3") or 0.0
        ) > 0:
            return "incomplete_answer"
        return "retrieval_miss"
    # decision == "answer"
    hit = metrics.get("hit_at_k")
    if hit is not None:
        retrieval_ok = hit > 0.0
    else:
        retrieval_ok = (metrics.get("doc_recall_at_3") or 0.0) > 0.0
    if not retrieval_ok:
        if len(example.get("turns_before_target", [])) >= 3:
            return "retrieval_miss"
        return "retrieval_miss"
    # Retrieved an on-target source but maybe wrong section.
    expected_hits = (metrics.get("doc_recall_at_3") or 0.0) > 0.0
    span_miss = (metrics.get("span_recall_at_5") or 0.0) == 0.0
    if expected_hits and span_miss and (metrics.get("precision_at_k") or 0.0) < 1.0:
        return "right_source_wrong_section"
    if (metrics.get("citations_valid") or 0.0) == 0.0 or (
        metrics.get("citation_coverage") or 0.0
    ) < 1.0:
        return "weak_citations"
    faithfulness = metrics.get("faithfulness")
    if faithfulness is not None and faithfulness < 1.0:
        return "unfaithful_answer"
    if (metrics.get("required_points_covered") or 1.0) < 1.0:
        return "incomplete_answer"
    answer_relevance = metrics.get("answer_relevance")
    if answer_relevance is not None and answer_relevance < 1.0:
        return "irrelevant_answer"
    return None


def _load_or_build_examples(
    settings: Any,
    domain: DomainLike,
    split: DatasetSplitLike,
) -> list[Example]:
    resolved_domain = parse_domain(domain)
    if resolved_domain.value == "kubernetes":
        raise ValueError("Kubernetes uses curated eval subsets, not dialogue files.")
    resolved_split = parse_dataset_split(split)
    path = settings.paths.examples_dir / f"{resolved_domain}_{resolved_split}.jsonl"
    if path.exists():
        return load_examples_jsonl(path)
    dialogues = load_dialogues(
        settings.dataset.root,
        split=resolved_split,
        domains=[resolved_domain],
    )
    examples = build_turn_examples(dialogues)
    write_examples_jsonl(examples, path)
    return examples


def _eval_subset_path(settings: Any, domain: DomainLike, subset: EvalSubset) -> Path:
    domain_path = (
        settings.paths.project_root
        / "data/eval_subsets"
        / str(parse_domain(domain))
        / f"{subset}.jsonl"
    )
    if domain_path.exists():
        return domain_path
    legacy_path = settings.paths.project_root / "data/eval_subsets" / f"{subset}.jsonl"
    if legacy_path.exists():
        return legacy_path
    return domain_path


def load_eval_examples(
    settings: Any,
    domain: DomainLike,
    split: DatasetSplitLike,
    subset: EvalSubsetLike,
) -> tuple[list[Example], str]:
    resolved_subset = parse_eval_subset(subset)
    if resolved_subset in {EvalSubset.SMOKE, EvalSubset.FROZEN_EXPERIMENT}:
        path = _eval_subset_path(settings, domain, resolved_subset)
        return load_subset_jsonl(path), str(resolved_subset)
    examples = _load_or_build_examples(settings, domain, split)
    return examples, str(EvalSubset.FULL_VALIDATION)


def build_eval_config(settings: Settings, domain: DomainLike) -> RuntimeConfig:
    return settings.runtime_for(domain)


def with_config_overrides(
    config: RuntimeConfig,
    experiment: RuntimeExperimentOverrides,
) -> RuntimeConfig:
    return apply_runtime_experiment_overrides(config, experiment)


def build_run_id(
    domain: DomainLike,
    subset: str,
    now: datetime | None = None,
    *,
    slug: str | None = None,
) -> str:
    current = now or datetime.now().astimezone()
    parts = [current.strftime("%Y%m%d-%H%M%S"), str(parse_domain(domain)), subset]
    if slug:
        parts.append(slug)
    return "-".join(parts)


def _prediction_retrieval_ranked_chunks(prediction: dict) -> list[dict]:
    if "retrieval_ranked_chunks" in prediction:
        return prediction["retrieval_ranked_chunks"]
    return prediction.get("retrieved_chunks", [])


def _prediction_retrieved_chunks(prediction: dict) -> list[dict]:
    return prediction.get(
        "retrieved_chunks", _prediction_retrieval_ranked_chunks(prediction)
    )


def _prediction_record(
    example: dict,
    prediction: dict,
    metrics: dict,
    *,
    failure_label: str | None = None,
    runtime_error: dict | None = None,
) -> dict:
    record = {
        "example_id": example.get("example_id"),
        "target_mode": example.get("target_mode"),
        "target_turn_id": example.get("target_turn_id"),
        "latest_user_utterance": prediction.get("latest_user_utterance")
        or example.get("latest_user_utterance"),
        "gold_doc_ids": example.get("gold_doc_ids", []),
        "gold_span_ids": example.get("gold_span_ids", []),
        "target_text": example.get("target_turn", {}).get("utterance", ""),
        "decision": prediction.get("decision"),
        "response_text": prediction.get("response_text"),
        "citations": prediction.get("citations", []),
        "retrieval_ranked_chunks": _prediction_retrieval_ranked_chunks(prediction),
        "retrieved_chunks": _prediction_retrieved_chunks(prediction),
        "trace_summary": prediction.get("trace_summary", {}),
        "metrics": metrics,
        "failure_label": failure_label,
        **({"runtime_error": runtime_error} if runtime_error is not None else {}),
    }
    if has_rag_eval_fields(example):
        record.update(
            {
                "expected_sources": example.get("expected_sources", []),
                "acceptable_sources": example.get("acceptable_sources", []),
                "required_points": example.get("required_points", []),
                "forbidden_claims": example.get("forbidden_claims", []),
                "answer_type": example.get("answer_type"),
            }
        )
    return record


def _prediction_metrics(
    example: dict,
    prediction: dict,
    *,
    retrieval_top_k: int | None = None,
    judge_verdict: dict | None = None,
) -> dict:
    target_text = str(example.get("target_turn", {}).get("utterance", ""))
    retrieval_ranked_chunks = _prediction_retrieval_ranked_chunks(prediction)
    retrieved_chunks = _prediction_retrieved_chunks(prediction)
    citations = prediction.get("citations", [])
    response_text = str(prediction.get("response_text", ""))
    doc_recalls = {
        k: (
            doc_recall_at_k(
                example.get("gold_doc_ids", []),
                retrieval_ranked_chunks,
                k=k,
            )
            if _retrieval_metric_available(retrieval_top_k, k=k)
            else None
        )
        for k in (1, 3, 5, 10)
    }
    span_recall = (
        span_recall_at_k(example.get("gold_span_ids", []), retrieval_ranked_chunks, k=5)
        if _retrieval_metric_available(retrieval_top_k, k=5)
        else None
    )
    mrr = (
        mrr_at_k(example.get("gold_doc_ids", []), retrieval_ranked_chunks, k=5)
        if _retrieval_metric_available(retrieval_top_k, k=5)
        else None
    )
    rouge = rouge_l_f1(response_text, target_text)
    f1 = token_f1(response_text, target_text)
    em = exact_match(response_text, target_text)
    bleu = sacrebleu_score(response_text, target_text)
    citation_cov = citation_coverage(example.get("gold_span_ids", []), citations)
    citations_valid = (
        1.0 if citations_map_to_retrieved(citations, retrieved_chunks) else 0.0
    )
    retrieved_doc_success = (doc_recalls[3] or 0.0) > 0.0 or (span_recall or 0.0) > 0.0
    text_success = max(rouge, f1) >= END_TO_END_TEXT_THRESHOLD
    end_to_end_success = (
        example.get("target_mode") == "answer"
        and prediction.get("decision") == "answer"
        and retrieved_doc_success
        and citations_valid == 1.0
        and text_success
    )
    metrics: dict = {
        "doc_recall_at_1": doc_recalls[1],
        "doc_recall_at_3": doc_recalls[3],
        "doc_recall_at_5": doc_recalls[5],
        "doc_recall_at_10": doc_recalls[10],
        "span_recall_at_5": span_recall,
        "mrr_at_5": mrr,
        "rouge_l": rouge,
        "token_f1": f1,
        "exact_match": em,
        "sacrebleu": bleu,
        "citation_coverage": citation_cov,
        "citations_valid": citations_valid,
        "end_to_end_success": 1.0 if end_to_end_success else 0.0,
    }

    if has_rag_eval_fields(example):
        expected_sources = example.get("expected_sources", [])
        acceptable_sources = example.get("acceptable_sources", [])
        required_points = example.get("required_points", [])
        forbidden_claims = example.get("forbidden_claims", [])
        k = 5
        graded_available = _retrieval_metric_available(retrieval_top_k, k=k)
        hit = (
            hit_at_k(expected_sources, acceptable_sources, retrieval_ranked_chunks, k=k)
            if graded_available
            else None
        )
        precision = (
            precision_at_k(
                expected_sources, acceptable_sources, retrieval_ranked_chunks, k=k
            )
            if graded_available
            else None
        )
        graded_mrr = (
            graded_mrr_at_k(
                expected_sources, acceptable_sources, retrieval_ranked_chunks, k=k
            )
            if graded_available
            else None
        )
        ndcg = (
            ndcg_at_k(
                expected_sources, acceptable_sources, retrieval_ranked_chunks, k=k
            )
            if graded_available
            else None
        )
        metrics.update(
            {
                "hit_at_k": hit,
                "precision_at_k": precision,
                "graded_mrr_at_k": graded_mrr,
                "ndcg_at_k": ndcg,
            }
        )

        # Deterministic v1 RAG triad proxies.
        context_relevance = precision if precision is not None else 0.0
        forbidden_hit = forbidden_claims_hit(forbidden_claims, response_text)
        forbidden_score = forbidden_hit if forbidden_hit is not None else 0.0
        faithfulness = 0.0 if forbidden_score > 0.0 else None
        answer_relevance = None
        points_covered = required_point_coverage(required_points, response_text)
        points_score = points_covered if points_covered is not None else 1.0
        answer_correctness = {
            "required_points_covered": points_score,
            "forbidden_claims_present": forbidden_score,
            "reference_similarity": {
                "rouge_l": rouge,
                "token_f1": f1,
                "exact_match": em,
                "sacrebleu": bleu,
            },
        }
        metrics.update(
            {
                "context_relevance": context_relevance,
                "faithfulness": faithfulness,
                "answer_relevance": answer_relevance,
                "answer_correctness": answer_correctness,
                "required_points_covered": points_score,
                "forbidden_claims_present": forbidden_score,
            }
        )

        if judge_verdict is not None:
            metrics["judge"] = judge_verdict
            if judge_verdict.get("faithful") is not None:
                metrics["faithfulness"] = float(judge_verdict["faithful"])
            if judge_verdict.get("answer_relevant") is not None:
                metrics["answer_relevance"] = float(judge_verdict["answer_relevant"])
            if judge_verdict.get("context_relevant") is not None:
                metrics["context_relevance"] = float(judge_verdict["context_relevant"])

    return metrics


def _safe_mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return mean(present)


def _metric_display(
    value: float | None,
    *,
    retrieval_top_k: int | None = None,
    metric_k: int | None = None,
) -> str:
    if value is not None:
        return f"{value:.3f}"
    if (
        metric_k is not None
        and retrieval_top_k is not None
        and int(retrieval_top_k) < metric_k
    ):
        return f"n/a (retrieval_top_k={int(retrieval_top_k)})"
    return "n/a"


def _rate_map(records: list[dict], key: str) -> dict[str, float]:
    counts = Counter(record.get(key, "unknown") for record in records)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {name: count / total for name, count in sorted(counts.items())}


def _metric_mean(records: list[dict], metric: str) -> float | None:
    return _safe_mean([record["metrics"].get(metric) for record in records])


def _rag_records(records: list[dict]) -> list[dict]:
    return [record for record in records if has_rag_eval_fields(record)]


def _rag_triad_metrics(records: list[dict]) -> dict:
    rag = _rag_records(records)
    if not rag:
        return {}
    return {
        "context_relevance": _metric_mean(rag, "context_relevance"),
        "faithfulness": _metric_mean(rag, "faithfulness"),
        "answer_relevance": _metric_mean(rag, "answer_relevance"),
        "required_points_covered": _metric_mean(rag, "required_points_covered"),
        "forbidden_claims_present": _metric_mean(rag, "forbidden_claims_present"),
        "hit_at_k": _metric_mean(rag, "hit_at_k"),
        "precision_at_k": _metric_mean(rag, "precision_at_k"),
        "graded_mrr_at_k": _metric_mean(rag, "graded_mrr_at_k"),
        "ndcg_at_k": _metric_mean(rag, "ndcg_at_k"),
        "examples": len(rag),
    }


def _aggregate_metrics(predictions: list[dict]) -> dict:
    answer_predictions = [
        record for record in predictions if record.get("target_mode") == "answer"
    ]
    follow_up_predictions = [
        record for record in predictions if record.get("target_mode") == "follow_up"
    ]

    def retrieval_metrics(records: list[dict]) -> dict:
        return {
            "doc_recall_at_1": _metric_mean(records, "doc_recall_at_1"),
            "doc_recall_at_3": _metric_mean(records, "doc_recall_at_3"),
            "doc_recall_at_5": _metric_mean(records, "doc_recall_at_5"),
            "doc_recall_at_10": _metric_mean(records, "doc_recall_at_10"),
            "span_recall_at_5": _metric_mean(records, "span_recall_at_5"),
            "mrr_at_5": _metric_mean(records, "mrr_at_5"),
        }

    answer_latencies = [
        record.get("trace_summary", {}).get("latency_ms")
        for record in predictions
        if record.get("trace_summary", {}).get("latency_ms") is not None
    ]
    ordered_latencies = sorted(float(value) for value in answer_latencies)
    p95_latency = None
    if ordered_latencies:
        p95_index = max(
            0,
            min(
                len(ordered_latencies) - 1,
                int(round(0.95 * (len(ordered_latencies) - 1))),
            ),
        )
        p95_latency = ordered_latencies[p95_index]

    rag_answer = _rag_triad_metrics(answer_predictions)
    rag_overall = _rag_triad_metrics(predictions)
    return {
        "counts": {
            "examples": len(predictions),
            "answer_examples": len(answer_predictions),
            "follow_up_examples": len(follow_up_predictions),
            "rag_examples": len(_rag_records(predictions)),
        },
        "retrieval": {
            "answer": retrieval_metrics(answer_predictions),
            "follow_up": retrieval_metrics(follow_up_predictions),
            "overall": retrieval_metrics(predictions),
        },
        "generation": {
            "answer": {
                "rouge_l": _metric_mean(answer_predictions, "rouge_l"),
                "token_f1": _metric_mean(answer_predictions, "token_f1"),
                "exact_match": _metric_mean(answer_predictions, "exact_match"),
                "sacrebleu": _metric_mean(answer_predictions, "sacrebleu"),
                "citation_coverage": _metric_mean(
                    answer_predictions,
                    "citation_coverage",
                ),
                "end_to_end_success_rate": _metric_mean(
                    answer_predictions,
                    "end_to_end_success",
                ),
            }
        },
        "rag": {
            "answer": rag_answer,
            "overall": rag_overall,
        },
        "decision_distribution": {
            "overall": _rate_map(predictions, "decision"),
            "answer": _rate_map(answer_predictions, "decision"),
            "follow_up": _rate_map(follow_up_predictions, "decision"),
        },
        "latency_ms": {
            "average": _safe_mean(ordered_latencies),
            "p95": p95_latency,
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _compact_join(values: list[str]) -> str:
    return " | ".join(str(value) for value in values if str(value).strip())


def _chunk_ids(chunks: list[dict]) -> list[str]:
    return [
        str(chunk.get("chunk_id"))
        for chunk in chunks
        if chunk.get("chunk_id") is not None and str(chunk.get("chunk_id")).strip()
    ]


def _citation_chunk_ids(citations: list[dict]) -> list[str]:
    return [
        str(citation.get("chunk_id"))
        for citation in citations
        if citation.get("chunk_id") is not None
        and str(citation.get("chunk_id")).strip()
    ]


def _manual_review_rows(run_id: str, predictions: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for record in predictions:
        is_follow_up = record.get("target_mode") == "follow_up"
        is_answer_failure = record.get("target_mode") == "answer" and bool(
            record.get("failure_label")
        )
        if not (is_follow_up or is_answer_failure):
            continue
        trace_summary = record.get("trace_summary", {})
        metrics = record.get("metrics", {})
        judge = metrics.get("judge") or {}
        rows.append(
            {
                "run_id": run_id,
                "example_id": record.get("example_id", ""),
                "target_mode": record.get("target_mode", ""),
                "decision": record.get("decision", ""),
                "failure_label": record.get("failure_label", "") or "",
                "latest_user_utterance": record.get("latest_user_utterance", ""),
                "response_text": record.get("response_text", ""),
                "gold_doc_ids": _compact_join(record.get("gold_doc_ids", [])),
                "gold_span_ids": _compact_join(record.get("gold_span_ids", [])),
                "expected_sources": _compact_join(record.get("expected_sources", [])),
                "acceptable_sources": _compact_join(
                    record.get("acceptable_sources", [])
                ),
                "required_points": _compact_join(record.get("required_points", [])),
                "forbidden_claims": _compact_join(record.get("forbidden_claims", [])),
                "answer_type": record.get("answer_type", "") or "",
                "final_query": trace_summary.get("final_query", ""),
                "retrieval_attempts": trace_summary.get("retrieval_attempts", 0),
                "retrieval_ranked_chunk_ids": _compact_join(
                    _chunk_ids(record.get("retrieval_ranked_chunks", []))
                ),
                "retrieved_chunk_ids": _compact_join(
                    _chunk_ids(record.get("retrieved_chunks", []))
                ),
                "citation_chunk_ids": _compact_join(
                    _citation_chunk_ids(record.get("citations", []))
                ),
                "context_relevance": _metric_display(metrics.get("context_relevance")),
                "faithfulness": _metric_display(metrics.get("faithfulness")),
                "answer_relevance": _metric_display(metrics.get("answer_relevance")),
                "required_points_covered": _metric_display(
                    metrics.get("required_points_covered")
                ),
                "forbidden_claims_present": _metric_display(
                    metrics.get("forbidden_claims_present")
                ),
                "judge_rationale": judge.get("rationale", ""),
                "decision_correct": "",
                "evidence_relevant": "",
                "no_unsupported_claims": "",
                "clear": "",
                "citations_useful": "",
                "reviewer_notes": "",
            }
        )
    return rows


def _retrieval_example_records(predictions: list[dict]) -> list[dict]:
    records: list[dict] = []
    for record in predictions:
        trace_summary = record.get("trace_summary", {})
        metrics = record.get("metrics", {})
        records.append(
            {
                "example_id": record.get("example_id"),
                "target_mode": record.get("target_mode"),
                "latest_user_utterance": record.get("latest_user_utterance"),
                "target_text": record.get("target_text"),
                "gold_doc_ids": record.get("gold_doc_ids", []),
                "gold_span_ids": record.get("gold_span_ids", []),
                "final_query": trace_summary.get("final_query"),
                "retrieval_attempts": trace_summary.get("retrieval_attempts"),
                "retrieval_ranked_chunks": record.get("retrieval_ranked_chunks", []),
                "retrieved_chunks": record.get("retrieved_chunks", []),
                "doc_recall_at_3": metrics.get("doc_recall_at_3"),
                "span_recall_at_5": metrics.get("span_recall_at_5"),
                "mrr_at_5": metrics.get("mrr_at_5"),
                "failure_label": record.get("failure_label"),
                "decision": record.get("decision"),
            }
        )
    return records


def _trace_index_records(predictions: list[dict], *, project_root: Path) -> list[dict]:
    records: list[dict] = []
    for record in predictions:
        trace_summary = record["trace_summary"]
        trace_path = resolve_project_path(project_root, trace_summary["trace_path"])
        summarized = summarize_trace_events(
            load_trace_events(trace_path),
            trace_path=project_relative_path(trace_path, project_root),
        )
        records.append(
            {
                "example_id": record["example_id"],
                "trace_file": trace_path.name,
                "graph_path": summarized.get("graph_path", []),
                "retrieval_attempts": summarized.get("retrieval_attempts", 0),
                "final_query": summarized.get("final_query", ""),
                "decision": record.get("decision") or summarized.get("decision", ""),
                "failure_label": record.get("failure_label"),
                "total_latency_ms": summarized.get("total_latency_ms", 0.0),
                "node_latency_ms": summarized.get("node_latency_ms", {}),
                "retrieval_ranked_count": summarized.get("retrieval_ranked_count", 0),
                "retrieved_count": summarized.get("retrieved_count", 0),
                "fallback_count": summarized.get("fallback_count", 0),
                "fallback_nodes": summarized.get("fallback_nodes", []),
            }
        )
    return records


def _summary_lines(
    run_id: str,
    subset_label: str,
    metrics: dict,
    failure_counts: dict[str, int],
    notes: str | None,
    output_dir: Path,
    *,
    retrieval_top_k: int | None = None,
) -> list[str]:
    retrieval = metrics["retrieval"]["answer"]
    generation = metrics["generation"]["answer"]

    sorted_failures = (
        sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
        if failure_counts
        else []
    )

    recommendation = "keep"
    if failure_counts or generation.get("end_to_end_success_rate") == 0.0:
        recommendation = "investigate"

    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=False,
    )
    template = env.get_template("summary.md.j2")

    rendered = template.render(
        run_id=run_id,
        subset_label=subset_label,
        notes=notes,
        metrics=metrics,
        retrieval=retrieval,
        generation=generation,
        retrieval_top_k=retrieval_top_k,
        failure_counts=sorted_failures,
        recommendation=recommendation,
        output_dir=output_dir,
        metric_display=_metric_display,
    )

    return rendered.splitlines()


async def _maybe_await_result(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _runtime_error_record(
    example: dict,
    exc: Exception,
    *,
    trace_path: str,
    retrieval_top_k: int | None = None,
) -> dict:
    prediction = {
        "decision": "abstain",
        "response_text": "",
        "citations": [],
        "retrieval_ranked_chunks": [],
        "retrieved_chunks": [],
        "trace_summary": {
            "retrieval_attempts": 0,
            "final_query": "",
            "graph_path": [],
            "latency_ms": None,
            "trace_path": trace_path,
            "fallback_count": 0,
            "fallback_nodes": [],
            "fallbacks": [],
        },
        "latest_user_utterance": example.get("latest_user_utterance"),
    }
    runtime_error = {
        "exception_type": type(exc).__name__,
        "error": str(exc),
    }
    return _prediction_record(
        example,
        prediction,
        _prediction_metrics(
            example,
            prediction,
            retrieval_top_k=retrieval_top_k,
        ),
        failure_label="runtime_error",
        runtime_error=runtime_error,
    )


async def evaluate_examples_async(
    examples: list[dict],
    *,
    settings: Any,
    domain: DomainLike,
    split: DatasetSplitLike,
    subset_name: str,
    limit: int | None = None,
    notes: str | None = None,
    run_graph_func: Any = run_graph_async,
    now: datetime | None = None,
    config: RuntimeConfig | None = None,
    run_id_slug: str | None = None,
    subset_label: str | None = None,
    manifest_overrides: dict[str, Any] | None = None,
    max_concurrency: int = 1,
    judge: RAGTriadJudge | None = None,
) -> dict:
    resolved_domain = parse_domain(domain)
    resolved_split = parse_dataset_split(split)
    selected_examples = examples[:limit] if limit is not None else examples
    if not selected_examples:
        raise ValueError("No examples available for evaluation.")
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive.")

    run_id = build_run_id(resolved_domain, subset_name, now=now, slug=run_id_slug)
    artifacts = eval_run_artifacts(settings.paths.project_root, run_id)
    artifacts.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Starting eval run %s domain=%s split=%s subset=%s examples=%s max_concurrency=%s",
        run_id,
        resolved_domain,
        resolved_split,
        subset_name,
        len(selected_examples),
        max_concurrency,
    )
    resolved_config = (
        config if config is not None else build_eval_config(settings, resolved_domain)
    )
    shared_runtime_resources = None
    if run_graph_func is run_graph_async:
        logger.info("Resolving shared runtime resources for eval run %s", run_id)
        shared_runtime_resources = await resolve_runtime_resources_async(
            resolved_config
        )
    if shared_runtime_resources is not None:
        logger.info("Shared runtime resources ready for eval run %s", run_id)
    progress_lock = asyncio.Lock()
    completed_count = 0

    async def _log_prediction_progress(example: dict, record: dict) -> None:
        nonlocal completed_count
        async with progress_lock:
            completed_count += 1
            logger.info(
                "Eval progress %s/%s example=%s target=%s decision=%s failure=%s",
                completed_count,
                len(selected_examples),
                example.get("example_id"),
                example.get("target_mode"),
                record.get("decision"),
                record.get("failure_label") or "none",
            )

    semaphore = asyncio.Semaphore(max_concurrency)

    def trace_path_for_example(example: dict) -> Path:
        example_id = str(example["example_id"])
        return artifacts.trace_path(build_trace_file(example_id))

    async def run_prediction(example: dict, *, trace_path: Path) -> dict:
        call_kwargs = {
            "example": example,
            "config": resolved_config,
            "run_id": run_id,
            "trace_path": trace_path,
            "max_attempts": resolved_config.max_retrieval_attempts,
        }
        if shared_runtime_resources is not None:
            call_kwargs["_runtime_resources"] = shared_runtime_resources
        async with semaphore:
            prediction = await _maybe_await_result(run_graph_func(**call_kwargs))
        trace_summary = dict(prediction["trace_summary"])
        trace_summary["trace_path"] = project_relative_path(
            trace_path,
            settings.paths.project_root,
        )
        return {**prediction, "trace_summary": trace_summary}

    async def evaluate_one(index: int, example: dict) -> tuple[int, dict]:
        trace_path = trace_path_for_example(example)
        relative_trace_path = project_relative_path(
            trace_path, settings.paths.project_root
        )
        try:
            prediction = await run_prediction(example, trace_path=trace_path)
        except Exception as exc:
            await asyncio.to_thread(
                write_trace_event,
                trace_path,
                {
                    "kind": "error",
                    "node": "run_graph_async",
                    "run_id": run_id,
                    "example_id": example.get("example_id"),
                    "exception_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            record = _runtime_error_record(
                example,
                exc,
                trace_path=relative_trace_path,
                retrieval_top_k=resolved_config.retrieval_top_k,
            )
            await _log_prediction_progress(example, record)
            return (
                index,
                record,
            )
        metrics = _prediction_metrics(
            example,
            prediction,
            retrieval_top_k=resolved_config.retrieval_top_k,
        )
        if judge is not None and judge.enabled and has_rag_eval_fields(example):
            judge_verdict = await judge.evaluate(
                query=str(
                    example.get("latest_user_utterance")
                    or example.get("target_turn", {}).get("utterance", "")
                ),
                answer=str(prediction.get("response_text", "")),
                context=_join_chunk_text(_prediction_retrieved_chunks(prediction)),
                required_points=example.get("required_points", []),
                forbidden_claims=example.get("forbidden_claims", []),
            )
            if judge_verdict is not None:
                metrics = _prediction_metrics(
                    example,
                    prediction,
                    retrieval_top_k=resolved_config.retrieval_top_k,
                    judge_verdict=judge_verdict,
                )
        record = _prediction_record(
            example,
            prediction,
            metrics,
            failure_label=_failure_label(example, prediction, metrics),
        )
        await _log_prediction_progress(example, record)
        return (
            index,
            record,
        )

    completed = await asyncio.gather(
        *[
            evaluate_one(index, example)
            for index, example in enumerate(selected_examples)
        ]
    )
    ordered_predictions = [
        record for _, record in sorted(completed, key=lambda item: item[0])
    ]
    predictions = list(ordered_predictions)
    failures = [
        record for record in predictions if record.get("failure_label") is not None
    ]

    metrics = _aggregate_metrics(predictions)
    failure_counts = dict(
        sorted(
            Counter(
                record["failure_label"]
                for record in failures
                if record.get("failure_label")
            ).items()
        )
    )
    logger.info(
        "Finished model execution for eval run %s failures=%s",
        run_id,
        len(failures),
    )
    resolved_subset_label = (
        subset_label or f"{resolved_domain} {resolved_split} / {subset_name}"
    )
    manifest = {
        "run_id": run_id,
        "created_at": (now or datetime.now().astimezone()).isoformat(),
        "dataset_root": str(settings.dataset.root),
        "domains": [str(resolved_domain)],
        "split": str(resolved_split),
        "eval_subset": subset_name,
        "subset_label": resolved_subset_label,
        "target_modes": sorted(
            {example.get("target_mode", "answer") for example in selected_examples}
        ),
        "provider": {
            "type": str(resolved_chat_provider(settings.runtime)),
            "chat_base_url": chat_provider_base_url(settings.runtime),
            "embedding_type": str(resolved_embedding_provider(settings.runtime)),
            "embedding_base_url": embedding_provider_base_url(settings.runtime)
            if settings.runtime.embedding_model
            else None,
            "chat_model": settings.runtime.chat_model,
            "embedding_model": settings.runtime.embedding_model,
        },
        "chunking": {
            "strategy": "section_with_deterministic_subchunks",
            "max_tokens_per_chunk": 512,
        },
        "retrieval": {
            "top_k": resolved_config.retrieval_top_k,
            "candidate_k": resolved_config.retrieval_candidate_k,
            "max_attempts": resolved_config.max_retrieval_attempts,
            "use_history": True,
            "content_only_reasoning": bool(resolved_config.content_only_reasoning),
            "neighbor_expansion": bool(resolved_config.neighbor_expansion),
        },
        "graph": {
            "enable_retry": resolved_config.llm_max_retries > 1,
            "decision_policy_version": "v1",
        },
        "rag_eval": {
            "enabled": any(has_rag_eval_fields(ex) for ex in selected_examples),
            "judge_enabled": bool(judge is not None and judge.enabled),
            "answer_types": sorted(
                {
                    str(ex.get("answer_type"))
                    for ex in selected_examples
                    if ex.get("answer_type")
                }
            ),
        },
        "prompt_version": resolved_config.prompt_version,
        "notes": notes or "Phase 4 MVP eval harness run.",
    }
    if manifest_overrides:
        manifest.update(manifest_overrides)

    logger.info(
        "Writing eval artifacts for run %s to %s",
        run_id,
        artifacts.output_dir,
    )
    await asyncio.to_thread(_write_json, artifacts.manifest, manifest)
    await asyncio.to_thread(
        _write_json,
        artifacts.metrics,
        {**metrics, "failure_counts": failure_counts},
    )
    await asyncio.to_thread(_write_jsonl, artifacts.predictions, predictions)
    await asyncio.to_thread(_write_jsonl, artifacts.failures, failures)
    await asyncio.to_thread(
        _write_csv,
        artifacts.manual_review,
        MANUAL_REVIEW_COLUMNS,
        _manual_review_rows(run_id, predictions),
    )
    await asyncio.to_thread(
        _write_jsonl,
        artifacts.retrieval_examples,
        _retrieval_example_records(predictions),
    )
    trace_index_records = await asyncio.to_thread(
        _trace_index_records,
        predictions,
        project_root=settings.paths.project_root,
    )
    await asyncio.to_thread(
        _write_json,
        artifacts.trace_index,
        {"entries": trace_index_records},
    )
    await asyncio.to_thread(
        artifacts.summary.write_text,
        "\n".join(
            _summary_lines(
                run_id,
                resolved_subset_label,
                metrics,
                failure_counts,
                notes,
                artifacts.output_dir,
                retrieval_top_k=resolved_config.retrieval_top_k,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Eval run %s complete. Summary written to %s",
        run_id,
        artifacts.summary,
    )

    return {
        "run_id": run_id,
        "subset_label": resolved_subset_label,
        "output_dir": artifacts.output_dir,
        "artifact_paths": {
            "manifest": artifacts.manifest,
            "metrics": artifacts.metrics,
            "predictions": artifacts.predictions,
            "failures": artifacts.failures,
            "manual_review": artifacts.manual_review,
            "retrieval_examples": artifacts.retrieval_examples,
            "trace_index": artifacts.trace_index,
            "summary": artifacts.summary,
        },
        "metrics": metrics,
        "retrieval_top_k": resolved_config.retrieval_top_k,
        "failure_counts": failure_counts,
        "predictions": predictions,
        "failures": failures,
    }


async def evaluate_split_async(
    *,
    settings: Any,
    domain: DomainLike,
    split: DatasetSplitLike = DatasetSplit.VALIDATION,
    subset: EvalSubsetLike = EvalSubset.SMOKE,
    limit: int | None = None,
    notes: str | None = None,
    run_graph_func: Any = run_graph_async,
    now: datetime | None = None,
    max_concurrency: int = 1,
) -> dict:
    resolved_domain = parse_domain(domain)
    resolved_split = parse_dataset_split(split)
    resolved_subset = parse_eval_subset(subset)
    examples, subset_name = load_eval_examples(
        settings,
        resolved_domain,
        resolved_split,
        resolved_subset,
    )
    return await evaluate_examples_async(
        examples,
        settings=settings,
        domain=resolved_domain,
        split=resolved_split,
        subset_name=subset_name,
        limit=limit,
        notes=notes,
        run_graph_func=run_graph_func,
        now=now,
        max_concurrency=max_concurrency,
    )
