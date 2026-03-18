"""Phase 4 evaluation harness."""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

from support_graph.data.dataset import load_dialogues
from support_graph.data.eval_subsets import load_subset_jsonl
from support_graph.data.examples import (
    build_turn_examples,
    load_examples_jsonl,
    write_examples_jsonl,
)
from support_graph.runtime.graph import run_graph


END_TO_END_TEXT_THRESHOLD = 0.35


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


def doc_recall_at_k(
    gold_doc_ids: list[str], retrieved_chunks: list[dict], k: int = 3
) -> float | None:
    gold = {doc_id for doc_id in gold_doc_ids if doc_id}
    if not gold:
        return None
    retrieved = {
        chunk.get("doc_id") for chunk in retrieved_chunks[:k] if chunk.get("doc_id")
    }
    return 1.0 if gold & retrieved else 0.0


def span_recall_at_k(
    gold_span_ids: list[str], retrieved_chunks: list[dict], k: int = 5
) -> float | None:
    gold = {span_id for span_id in gold_span_ids if span_id}
    if not gold:
        return None
    retrieved_spans: set[str] = set()
    for chunk in retrieved_chunks[:k]:
        retrieved_spans.update(str(span_id) for span_id in chunk.get("span_ids", []))
    return len(gold & retrieved_spans) / len(gold)


def mrr_at_k(
    gold_doc_ids: list[str], retrieved_chunks: list[dict], k: int = 5
) -> float | None:
    gold = {doc_id for doc_id in gold_doc_ids if doc_id}
    if not gold:
        return None
    for rank, chunk in enumerate(retrieved_chunks[:k], start=1):
        if chunk.get("doc_id") in gold:
            return 1.0 / rank
    return 0.0


def citation_coverage(gold_span_ids: list[str], citations: list[dict]) -> float | None:
    gold = {span_id for span_id in gold_span_ids if span_id}
    if not gold:
        return None
    cited_spans: set[str] = set()
    for citation in citations:
        cited_spans.update(str(span_id) for span_id in citation.get("span_ids", []))
    return len(gold & cited_spans) / len(gold)


def citations_map_to_retrieved(
    citations: list[dict], retrieved_chunks: list[dict]
) -> bool:
    retrieved_chunk_ids = {
        chunk.get("chunk_id") for chunk in retrieved_chunks if chunk.get("chunk_id")
    }
    citation_chunk_ids = [
        citation.get("chunk_id") for citation in citations if citation.get("chunk_id")
    ]
    if not citation_chunk_ids:
        return False
    return all(chunk_id in retrieved_chunk_ids for chunk_id in citation_chunk_ids)


def _failure_label(example: dict, prediction: dict, metrics: dict) -> str | None:
    target_mode = example.get("target_mode")
    decision = prediction.get("decision")
    if target_mode == "answer":
        if decision == "clarify":
            return "bad_clarification"
        if decision == "abstain":
            if (
                metrics.get("doc_recall_at_3", 0.0) > 0
                or metrics.get("span_recall_at_5", 0.0) > 0
            ):
                return "abstained_with_evidence"
            return "wrong_doc"
        if metrics.get("doc_recall_at_3", 0.0) == 0.0:
            if len(example.get("turns_before_target", [])) >= 3:
                return "missed_history"
            return "wrong_doc"
        if (
            metrics.get("doc_recall_at_3", 0.0) > 0.0
            and metrics.get("span_recall_at_5", 0.0) == 0.0
        ):
            return "right_doc_wrong_section"
        if (
            metrics.get("citations_valid", 0.0) == 0.0
            or metrics.get("citation_coverage", 0.0) < 1.0
        ):
            return "weak_citations"
        if metrics.get("end_to_end_success", 0.0) == 0.0:
            return "unsupported_answer"
        return None
    if metrics.get("doc_recall_at_3", 0.0) == 0.0:
        return "wrong_doc"
    return None


def _load_or_build_examples(settings: Any, domain: str, split: str) -> list[dict]:
    path = settings.examples_dir / f"{domain}_{split}.jsonl"
    if path.exists():
        return load_examples_jsonl(path)
    dialogues = load_dialogues(settings.dataset_root, split=split, domains=[domain])
    examples = build_turn_examples(dialogues)
    write_examples_jsonl(examples, path)
    return examples


def load_eval_examples(
    settings: Any, domain: str, split: str, subset: str
) -> tuple[list[dict], str]:
    if subset in {"smoke", "frozen_ablation"}:
        path = settings.project_root / "data/eval_subsets" / f"{subset}.jsonl"
        return load_subset_jsonl(path), subset
    examples = _load_or_build_examples(settings, domain, split)
    return examples, "full_validation"


def build_eval_config(settings: Any, domain: str) -> SimpleNamespace:
    if hasattr(settings, "chunk_artifact_path"):
        chunk_artifact_path = settings.chunk_artifact_path(domain)
    else:
        chunk_artifact_path = (
            settings.project_root / "data/derived/chunks" / f"{domain}.jsonl"
        )
    return SimpleNamespace(
        postgres_dsn=getattr(settings, "postgres_dsn", None),
        provider_type=getattr(settings, "provider_type", "ollama"),
        ollama_base_url=getattr(settings, "ollama_base_url", None),
        embedding_model=getattr(settings, "embedding_model", None),
        chat_model=getattr(settings, "chat_model", None),
        domain=domain,
        collection_name=settings.collection_name(domain),
        retrieval_top_k=getattr(settings, "retrieval_top_k", 5),
        retrieval_candidate_k=getattr(settings, "retrieval_candidate_k", 12),
        retrieval_rerank=True,
        content_only_reasoning=True,
        neighbor_expansion=True,
        max_retrieval_attempts=getattr(settings, "max_retrieval_attempts", 2),
        trace_dir=settings.trace_dir,
        chunk_artifact_path=chunk_artifact_path,
    )


def with_config_overrides(config: Any, **overrides: Any) -> SimpleNamespace:
    payload = dict(vars(config))
    payload.update(overrides)
    return SimpleNamespace(**payload)


def build_run_id(
    domain: str,
    subset: str,
    now: datetime | None = None,
    *,
    slug: str | None = None,
) -> str:
    current = now or datetime.now().astimezone()
    parts = [current.strftime("%Y%m%d-%H%M%S"), domain, subset]
    if slug:
        parts.append(slug)
    return "-".join(parts)


def _git_metadata(project_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    try:
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                capture_output=True,
                check=True,
                text=True,
            ).stdout.strip()
        )
    except Exception:
        dirty = True
    return commit, dirty


def _prediction_metrics(example: dict, prediction: dict) -> dict:
    target_text = str(example.get("target_turn", {}).get("utterance", ""))
    retrieval_ranked_chunks = prediction.get(
        "retrieval_ranked_chunks", prediction.get("retrieved_chunks", [])
    )
    retrieved_chunks = prediction.get("retrieved_chunks", retrieval_ranked_chunks)
    citations = prediction.get("citations", [])
    doc_recall = doc_recall_at_k(
        example.get("gold_doc_ids", []), retrieval_ranked_chunks, k=3
    )
    span_recall = span_recall_at_k(
        example.get("gold_span_ids", []), retrieval_ranked_chunks, k=5
    )
    mrr = mrr_at_k(example.get("gold_doc_ids", []), retrieval_ranked_chunks, k=5)
    rouge = rouge_l_f1(str(prediction.get("response_text", "")), target_text)
    f1 = token_f1(str(prediction.get("response_text", "")), target_text)
    citation_cov = citation_coverage(example.get("gold_span_ids", []), citations)
    citations_valid = (
        1.0 if citations_map_to_retrieved(citations, retrieved_chunks) else 0.0
    )
    retrieved_doc_success = (doc_recall or 0.0) > 0.0 or (span_recall or 0.0) > 0.0
    text_success = max(rouge, f1) >= END_TO_END_TEXT_THRESHOLD
    end_to_end_success = (
        example.get("target_mode") == "answer"
        and prediction.get("decision") == "answer"
        and retrieved_doc_success
        and citations_valid == 1.0
        and text_success
    )
    return {
        "doc_recall_at_3": doc_recall,
        "span_recall_at_5": span_recall,
        "mrr_at_5": mrr,
        "rouge_l": rouge,
        "token_f1": f1,
        "citation_coverage": citation_cov,
        "citations_valid": citations_valid,
        "end_to_end_success": 1.0 if end_to_end_success else 0.0,
    }


def _safe_mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return mean(present)


def _rate_map(records: list[dict], key: str) -> dict[str, float]:
    counts = Counter(record.get(key, "unknown") for record in records)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {name: count / total for name, count in sorted(counts.items())}


def _aggregate_metrics(predictions: list[dict]) -> dict:
    answer_predictions = [
        record for record in predictions if record.get("target_mode") == "answer"
    ]
    follow_up_predictions = [
        record for record in predictions if record.get("target_mode") == "follow_up"
    ]

    def retrieval_metrics(records: list[dict]) -> dict:
        return {
            "doc_recall_at_3": _safe_mean(
                [record["metrics"].get("doc_recall_at_3") for record in records]
            ),
            "span_recall_at_5": _safe_mean(
                [record["metrics"].get("span_recall_at_5") for record in records]
            ),
            "mrr_at_5": _safe_mean(
                [record["metrics"].get("mrr_at_5") for record in records]
            ),
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

    return {
        "counts": {
            "examples": len(predictions),
            "answer_examples": len(answer_predictions),
            "follow_up_examples": len(follow_up_predictions),
        },
        "retrieval": {
            "answer": retrieval_metrics(answer_predictions),
            "follow_up": retrieval_metrics(follow_up_predictions),
            "overall": retrieval_metrics(predictions),
        },
        "generation": {
            "answer": {
                "rouge_l": _safe_mean(
                    [record["metrics"].get("rouge_l") for record in answer_predictions]
                ),
                "token_f1": _safe_mean(
                    [record["metrics"].get("token_f1") for record in answer_predictions]
                ),
                "citation_coverage": _safe_mean(
                    [
                        record["metrics"].get("citation_coverage")
                        for record in answer_predictions
                    ]
                ),
                "end_to_end_success_rate": _safe_mean(
                    [
                        record["metrics"].get("end_to_end_success")
                        for record in answer_predictions
                    ]
                ),
            }
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


def _summary_lines(
    run_id: str,
    subset_label: str,
    metrics: dict,
    failure_counts: dict[str, int],
    notes: str | None,
    output_dir: Path,
) -> list[str]:
    retrieval = metrics["retrieval"]["answer"]
    generation = metrics["generation"]["answer"]
    lines = [
        "# SupportGraph Eval",
        "",
        f"Run: {run_id}",
        f"Subset: {subset_label}",
        "",
        "## What Changed",
        f"- {notes or 'Phase 4 MVP eval harness run.'}",
        "",
        "## Headline Metrics",
        f"- Doc Recall@3: {retrieval.get('doc_recall_at_3', 0.0) or 0.0:.3f}",
        f"- Span Recall@5: {retrieval.get('span_recall_at_5', 0.0) or 0.0:.3f}",
        f"- MRR@5: {retrieval.get('mrr_at_5', 0.0) or 0.0:.3f}",
        f"- ROUGE-L: {generation.get('rouge_l', 0.0) or 0.0:.3f}",
        f"- F1: {generation.get('token_f1', 0.0) or 0.0:.3f}",
        f"- Citation coverage: {generation.get('citation_coverage', 0.0) or 0.0:.3f}",
        "",
        "## Biggest Wins",
        f"- Retrieval is working over {metrics['counts']['examples']} evaluated examples.",
        f"- Average latency per example: {(metrics['latency_ms'].get('average') or 0.0):.1f} ms.",
        "",
        "## Biggest Regressions",
    ]
    if failure_counts:
        for label, count in sorted(
            failure_counts.items(), key=lambda item: (-item[1], item[0])
        )[:5]:
            lines.append(f"- {label}: {count}")
    else:
        lines.append("- No failures recorded in this run.")

    recommendation = "keep"
    if failure_counts:
        recommendation = "investigate"
    if generation.get("end_to_end_success_rate") == 0.0:
        recommendation = "investigate"

    lines.extend(
        [
            "",
            "## Recommendation",
            f"- {recommendation}",
            "",
            "## Artifacts",
            f"- {output_dir / 'manifest.json'}",
            f"- {output_dir / 'metrics.json'}",
            f"- {output_dir / 'predictions.jsonl'}",
            f"- {output_dir / 'failures.jsonl'}",
        ]
    )
    return lines


def evaluate_examples(
    examples: list[dict],
    *,
    settings: Any,
    domain: str,
    split: str,
    subset_name: str,
    limit: int | None = None,
    notes: str | None = None,
    run_graph_func: Any = run_graph,
    now: datetime | None = None,
    config: Any | None = None,
    run_id_slug: str | None = None,
    subset_label: str | None = None,
    manifest_overrides: dict[str, Any] | None = None,
) -> dict:
    selected_examples = examples[:limit] if limit is not None else examples
    if not selected_examples:
        raise ValueError("No examples available for evaluation.")

    run_id = build_run_id(domain, subset_name, now=now, slug=run_id_slug)
    output_dir = settings.eval_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = config or build_eval_config(settings, domain)

    predictions: list[dict] = []
    failures: list[dict] = []
    for example in selected_examples:
        prediction = run_graph_func(
            example=example,
            config=resolved_config,
            max_attempts=getattr(settings, "max_retrieval_attempts", 2),
            trace_dir=settings.trace_dir,
        )
        metrics = _prediction_metrics(example, prediction)
        record = {
            "example_id": example.get("example_id"),
            "target_mode": example.get("target_mode"),
            "target_turn_id": example.get("target_turn_id"),
            "gold_doc_ids": example.get("gold_doc_ids", []),
            "gold_span_ids": example.get("gold_span_ids", []),
            "target_text": example.get("target_turn", {}).get("utterance", ""),
            "decision": prediction.get("decision"),
            "response_text": prediction.get("response_text"),
            "citations": prediction.get("citations", []),
            "retrieval_ranked_chunks": prediction.get(
                "retrieval_ranked_chunks", prediction.get("retrieved_chunks", [])
            ),
            "retrieved_chunks": prediction.get("retrieved_chunks", []),
            "trace_summary": prediction.get("trace_summary", {}),
            "metrics": metrics,
        }
        failure_label = _failure_label(example, prediction, metrics)
        record["failure_label"] = failure_label
        predictions.append(record)
        if failure_label is not None:
            failures.append(record)

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
    resolved_subset_label = subset_label or f"{domain} {split} / {subset_name}"
    commit, dirty = _git_metadata(settings.project_root)

    manifest = {
        "run_id": run_id,
        "created_at": (now or datetime.now().astimezone()).isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "dataset_root": str(settings.dataset_root),
        "domains": [domain],
        "split": split,
        "eval_subset": subset_name,
        "target_modes": sorted(
            {example.get("target_mode", "answer") for example in selected_examples}
        ),
        "provider": {
            "type": getattr(settings, "provider_type", "ollama"),
            "base_url": getattr(settings, "ollama_base_url", None),
            "chat_model": getattr(settings, "chat_model", None),
            "embedding_model": getattr(settings, "embedding_model", None),
        },
        "chunking": {
            "strategy": "section_with_deterministic_subchunks",
            "max_tokens_per_chunk": 512,
        },
        "retrieval": {
            "top_k": getattr(
                resolved_config,
                "retrieval_top_k",
                getattr(settings, "retrieval_top_k", 5),
            ),
            "candidate_k": getattr(
                resolved_config,
                "retrieval_candidate_k",
                getattr(settings, "retrieval_candidate_k", 12),
            ),
            "max_attempts": getattr(
                resolved_config,
                "max_retrieval_attempts",
                getattr(settings, "max_retrieval_attempts", 2),
            ),
            "use_history": True,
            "content_only_reasoning": bool(
                getattr(resolved_config, "content_only_reasoning", True)
            ),
            "neighbor_expansion": bool(
                getattr(resolved_config, "neighbor_expansion", True)
            ),
        },
        "graph": {
            "enable_retry": True,
            "decision_policy_version": "v1",
        },
        "prompt": {
            "answer_prompt_version": "v1",
            "query_prompt_version": "v1",
            "evidence_prompt_version": "v1",
        },
        "notes": notes or "Phase 4 MVP eval harness run.",
    }
    if manifest_overrides:
        manifest.update(manifest_overrides)

    _write_json(output_dir / "manifest.json", manifest)
    _write_json(
        output_dir / "metrics.json", {**metrics, "failure_counts": failure_counts}
    )
    _write_jsonl(output_dir / "predictions.jsonl", predictions)
    _write_jsonl(output_dir / "failures.jsonl", failures)
    summary_path = output_dir / "summary.md"
    summary_path.write_text(
        "\n".join(
            _summary_lines(
                run_id,
                resolved_subset_label,
                metrics,
                failure_counts,
                notes,
                output_dir,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "run_id": run_id,
        "subset_label": resolved_subset_label,
        "output_dir": output_dir,
        "artifact_paths": {
            "manifest": output_dir / "manifest.json",
            "metrics": output_dir / "metrics.json",
            "predictions": output_dir / "predictions.jsonl",
            "failures": output_dir / "failures.jsonl",
            "summary": summary_path,
        },
        "metrics": metrics,
        "failure_counts": failure_counts,
        "predictions": predictions,
        "failures": failures,
    }


def evaluate_split(
    *,
    settings: Any,
    domain: str,
    split: str = "validation",
    subset: str = "smoke",
    limit: int | None = None,
    notes: str | None = None,
    run_graph_func: Any = run_graph,
    now: datetime | None = None,
) -> dict:
    examples, subset_name = load_eval_examples(settings, domain, split, subset)
    return evaluate_examples(
        examples,
        settings=settings,
        domain=domain,
        split=split,
        subset_name=subset_name,
        limit=limit,
        notes=notes,
        run_graph_func=run_graph_func,
        now=now,
    )
