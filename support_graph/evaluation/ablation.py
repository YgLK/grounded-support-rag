"""Targeted ablation runner for DMV Smoke-10."""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Any

from support_graph.evaluation.evaluate import (
    build_eval_config,
    evaluate_examples_async,
    load_eval_examples,
    with_config_overrides,
)
from support_graph.runtime.graph import run_graph_async


PRIMARY_METRICS = (
    ("Span Recall@5", ("retrieval", "answer", "span_recall_at_5")),
    ("Citation coverage", ("generation", "answer", "citation_coverage")),
    ("End-to-end success", ("generation", "answer", "end_to_end_success_rate")),
)

DOC_RECALL_FLOOR = 0.45
LATENCY_LIMIT_MS = 39_400.0


VARIANTS = (
    {
        "id": "control",
        "title": "Control",
        "summary": "Content-only reasoning baseline with the legacy transcript query and direct top-5 retrieval.",
        "notes": "Control: content-only reasoning promoted, but keep the legacy transcript query and no candidate-pool reranking or neighbor expansion.",
        "ablation_options": {
            "query_mode": "legacy_transcript",
            "content_only_reasoning": True,
            "retrieval_rerank": False,
            "retrieval_candidate_k": 5,
            "neighbor_expansion": False,
        },
        "config_overrides": {
            "retrieval_rerank": False,
            "retrieval_candidate_k": 5,
            "content_only_reasoning": True,
            "neighbor_expansion": False,
        },
    },
    {
        "id": "structured-query",
        "title": "Structured Query",
        "summary": "Swap the transcript dump for a compact history-aware query builder while keeping direct top-5 retrieval.",
        "notes": "Variant: compact deterministic query context with content-only reasoning, but still no candidate-pool reranking or neighbor expansion.",
        "ablation_options": {
            "query_mode": "structured",
            "content_only_reasoning": True,
            "retrieval_rerank": False,
            "retrieval_candidate_k": 5,
            "neighbor_expansion": False,
        },
        "config_overrides": {
            "retrieval_rerank": False,
            "retrieval_candidate_k": 5,
            "content_only_reasoning": True,
            "neighbor_expansion": False,
        },
    },
    {
        "id": "structured-query-rerank",
        "title": "Structured Query + Rerank",
        "summary": "Increase the candidate pool and apply deterministic reranking before keeping the best five chunks.",
        "notes": "Variant: compact query builder plus candidate-k retrieval and deterministic reranking, without neighbor expansion.",
        "ablation_options": {
            "query_mode": "structured",
            "content_only_reasoning": True,
            "retrieval_rerank": True,
            "neighbor_expansion": False,
        },
        "config_overrides": {
            "retrieval_rerank": True,
            "retrieval_candidate_k": 12,
            "content_only_reasoning": True,
            "neighbor_expansion": False,
        },
    },
    {
        "id": "structured-query-rerank-neighbors",
        "title": "Structured Query + Rerank + Neighbors",
        "summary": "Add same-doc one-hop neighboring-section expansion on top of the reranked evidence set.",
        "notes": "Variant: compact query builder, candidate-k retrieval, deterministic reranking, and same-doc neighboring-section expansion.",
        "ablation_options": {
            "query_mode": "structured",
            "content_only_reasoning": True,
            "retrieval_rerank": True,
            "neighbor_expansion": True,
        },
        "config_overrides": {
            "retrieval_rerank": True,
            "retrieval_candidate_k": 12,
            "content_only_reasoning": True,
            "neighbor_expansion": True,
        },
    },
)


def _metric_value(result: dict, *path: str) -> float:
    current: Any = result.get("metrics", {})
    for key in path:
        current = current.get(key, {})
    if current in (None, {}):
        return 0.0
    return float(current)


def _failure_count(result: dict, label: str) -> int:
    return int(result.get("failure_counts", {}).get(label, 0))


def _primary_deltas(control: dict, candidate: dict) -> dict[str, float]:
    return {
        name: _metric_value(candidate, *path) - _metric_value(control, *path)
        for name, path in PRIMARY_METRICS
    }


def _material_primary_gain(deltas: dict[str, float]) -> bool:
    return max(deltas.values(), default=0.0) >= 0.10


def classify_variant(control: dict, candidate: dict) -> tuple[str, str]:
    if candidate["variant"]["id"] == "control":
        return "control", "Baseline anchor for comparison."

    deltas = _primary_deltas(control, candidate)
    improved = [name for name, delta in deltas.items() if delta > 1e-9]
    candidate_doc = _metric_value(candidate, "retrieval", "answer", "doc_recall_at_3")
    candidate_latency = _metric_value(candidate, "latency_ms", "average")

    doc_guardrail = candidate_doc < DOC_RECALL_FLOOR
    latency_guardrail = (
        candidate_latency > LATENCY_LIMIT_MS and not _material_primary_gain(deltas)
    )

    if improved and not doc_guardrail and not latency_guardrail:
        return "worked", _why_it_moved(candidate, deltas)
    return "didn't work", _why_it_moved(candidate, deltas)


def _why_it_moved(candidate: dict, deltas: dict[str, float]) -> str:
    variant_id = candidate["variant"]["id"]
    positive = max(deltas.values(), default=0.0) > 0.0

    if variant_id == "structured-query":
        if positive:
            return "The compact history-aware query preserved the user’s actual need while reducing transcript noise in retrieval."
        return "The compact query alone did not move the first-stage retrieval enough to improve the primary metrics."
    if variant_id == "structured-query-rerank":
        if positive:
            return "The larger candidate pool and deterministic reranker improved section precision without changing the model stack."
        return "The reranker did not recover enough better sections from the larger candidate pool to beat the control."
    if variant_id == "structured-query-rerank-neighbors":
        if positive:
            return "Neighbor expansion added nearby evidence from the same document, which improved grounding after the direct retrieval step."
        return "Neighbor expansion increased evidence breadth, but it did not materially improve citations or end-to-end success on Smoke-10."
    return "Content-only reasoning remains the baseline anchor for comparison."


def _variant_score(result: dict) -> tuple[float, float, float, float, float]:
    return (
        _metric_value(result, "generation", "answer", "end_to_end_success_rate"),
        _metric_value(result, "generation", "answer", "citation_coverage"),
        _metric_value(result, "retrieval", "answer", "span_recall_at_5"),
        _metric_value(result, "retrieval", "answer", "doc_recall_at_3"),
        -_metric_value(result, "latency_ms", "average"),
    )


def best_smoke_step(results: list[dict]) -> dict:
    comparisons = [
        item
        for item in results
        if item["variant"]["id"] != "control" and item.get("status") == "worked"
    ]
    if comparisons:
        return max(comparisons, key=_variant_score)
    return results[0]


def smoke_step_clears_frozen_gate(result: dict) -> bool:
    span = _metric_value(result, "retrieval", "answer", "span_recall_at_5")
    citation = _metric_value(result, "generation", "answer", "citation_coverage")
    e2e = _metric_value(result, "generation", "answer", "end_to_end_success_rate")
    return span >= 0.25 or citation >= 0.15 or e2e >= 0.40


def _final_recommendation(results: list[dict]) -> tuple[str, str]:
    best = best_smoke_step(results)
    if best["variant"]["id"] == "control":
        return (
            "revert",
            "None of the retrieval-side increments cleared the Smoke-10 guardrails, so keep the content-only control.",
        )
    if smoke_step_clears_frozen_gate(best):
        return (
            "keep",
            f"Promote {best['variant']['title']} to the Frozen-200 check next.",
        )
    return (
        "investigate",
        f"{best['variant']['title']} is the strongest Smoke-10 step so far, but it does not clear the Frozen-200 gate.",
    )


def _summary_table_rows(results: list[dict]) -> list[str]:
    lines = [
        "| Variant | Status | Doc R@3 | Span R@5 | ROUGE-L | F1 | Citation | E2E | Avg Latency (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            "| {title} | {status} | {doc:.3f} | {span:.3f} | {rouge:.3f} | {f1:.3f} | {citation:.3f} | {e2e:.3f} | {latency:.1f} |".format(
                title=result["variant"]["title"],
                status=result.get("status", "control"),
                doc=_metric_value(result, "retrieval", "answer", "doc_recall_at_3"),
                span=_metric_value(result, "retrieval", "answer", "span_recall_at_5"),
                rouge=_metric_value(result, "generation", "answer", "rouge_l"),
                f1=_metric_value(result, "generation", "answer", "token_f1"),
                citation=_metric_value(
                    result, "generation", "answer", "citation_coverage"
                ),
                e2e=_metric_value(
                    result, "generation", "answer", "end_to_end_success_rate"
                ),
                latency=_metric_value(result, "latency_ms", "average") / 1000.0,
            )
        )
    return lines


def write_ablation_summary(
    *,
    settings: Any,
    domain: str,
    limit: int,
    results: list[dict],
    summary_timestamp: datetime,
    frozen_result: dict | None = None,
) -> Path:
    timestamp_slug = summary_timestamp.strftime("%Y%m%d-%H%M%S")
    path = settings.eval_dir / f"{timestamp_slug}-{domain}-smoke10-ablation-summary.md"
    recommendation, recommendation_line = _final_recommendation(results)
    control = results[0]

    lines = [
        f"# DMV Smoke-{limit} Ablation Summary",
        "",
        "Scope",
        f"- Domain: {domain}",
        f"- Subset: committed smoke / first {limit} examples",
        "- Purpose: compare retrieval-side improvements against the content-only control before widening the benchmark.",
        "",
        "Metric Table",
        *_summary_table_rows(results),
        "",
        "Variant Notes",
    ]
    for result in results:
        deltas = (
            _primary_deltas(control, result)
            if result["variant"]["id"] != "control"
            else {}
        )
        lines.extend(
            [
                f"- {result['variant']['title']}: {result.get('status', 'control')}",
                f"  {result['variant']['summary']}",
                f"  {result.get('status_reason', 'Baseline anchor for comparison.')}",
            ]
        )
        if deltas:
            lines.append(
                "  Deltas vs control: Span Recall@5 {span:+.3f}, citation coverage {citation:+.3f}, end-to-end success {e2e:+.3f}".format(
                    span=deltas["Span Recall@5"],
                    citation=deltas["Citation coverage"],
                    e2e=deltas["End-to-end success"],
                )
            )

    lines.extend(["", "Failure Shifts"])
    for result in results[1:]:
        lines.append(
            "- {title}: right_doc_wrong_section {right_doc}, weak_citations {weak_citations}, abstained_with_evidence {abstained}".format(
                title=result["variant"]["title"],
                right_doc=_failure_count(result, "right_doc_wrong_section"),
                weak_citations=_failure_count(result, "weak_citations"),
                abstained=_failure_count(result, "abstained_with_evidence"),
            )
        )

    lines.extend(["", "Frozen-200"])
    if frozen_result is None:
        best = best_smoke_step(results)
        lines.append(f"- Skipped. Best Smoke-10 step: {best['variant']['title']}.")
        lines.append(
            "- Gate not met: need Span Recall@5 >= 0.25, citation coverage >= 0.15, or end-to-end success >= 0.40."
        )
    else:
        lines.append(f"- Ran with {frozen_result['variant']['title']}.")
        lines.append(f"- Artifacts: {frozen_result['artifact_paths']['summary']}")

    lines.extend(
        [
            "",
            "Recommendation",
            f"- {recommendation}",
            f"- {recommendation_line}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


async def run_smoke10_ablation_async(
    *,
    settings: Any,
    domain: str = "dmv",
    split: str = "validation",
    limit: int = 10,
    run_graph_func: Any | None = None,
    now: datetime | None = None,
) -> dict:
    examples, _ = load_eval_examples(settings, domain, split, "smoke")
    selected_examples = examples[:limit]
    started_at = now or datetime.now().astimezone()
    base_config = build_eval_config(settings, domain)
    results: list[dict] = []

    for variant in VARIANTS:
        variant_config = with_config_overrides(
            base_config,
            ablation_variant=variant["id"],
            ablation_options=variant["ablation_options"],
            **variant["config_overrides"],
        )
        result = await evaluate_examples_async(
            selected_examples,
            settings=settings,
            domain=domain,
            split=split,
            subset_name=f"smoke{limit}",
            limit=None,
            notes=variant["notes"],
            now=started_at,
            config=variant_config,
            run_id_slug=variant["id"],
            subset_label=f"{domain} {split} / smoke first {limit} / {variant['title']}",
            run_graph_func=run_graph_func
            if run_graph_func is not None
            else run_graph_async,
            manifest_overrides={
                "ablation": {
                    "variant_id": variant["id"],
                    "variant_name": variant["title"],
                    "variant_summary": variant["summary"],
                    "scope": f"first {limit} examples from committed smoke subset",
                }
            },
        )
        result["variant"] = variant
        results.append(result)

    control = results[0]
    for result in results:
        status, reason = classify_variant(control, result)
        result["status"] = status
        result["status_reason"] = reason

    frozen_result: dict | None = None
    best = best_smoke_step(results)
    if smoke_step_clears_frozen_gate(best):
        frozen_examples, frozen_subset = load_eval_examples(
            settings, domain, split, "frozen_ablation"
        )
        variant = best["variant"]
        variant_config = with_config_overrides(
            base_config,
            ablation_variant=f"{variant['id']}-frozen",
            ablation_options=variant["ablation_options"],
            **variant["config_overrides"],
        )
        frozen_result = await evaluate_examples_async(
            frozen_examples,
            settings=settings,
            domain=domain,
            split=split,
            subset_name=frozen_subset,
            limit=None,
            notes=f"Frozen-200 follow-through for {variant['title']}.",
            now=started_at,
            config=variant_config,
            run_id_slug=f"{variant['id']}-frozen200",
            subset_label=f"{domain} {split} / frozen 200 / {variant['title']}",
            run_graph_func=run_graph_func
            if run_graph_func is not None
            else run_graph_async,
            manifest_overrides={
                "ablation": {
                    "variant_id": variant["id"],
                    "variant_name": variant["title"],
                    "variant_summary": variant["summary"],
                    "scope": "Frozen-200 follow-through after Smoke-10 gate",
                }
            },
        )
        frozen_result["variant"] = variant

    summary_path = write_ablation_summary(
        settings=settings,
        domain=domain,
        limit=limit,
        results=results,
        summary_timestamp=started_at,
        frozen_result=frozen_result,
    )
    recommendation, recommendation_line = _final_recommendation(results)
    return {
        "domain": domain,
        "limit": limit,
        "results": results,
        "best_result": best,
        "frozen_result": frozen_result,
        "summary_path": summary_path,
        "recommendation": recommendation,
        "recommendation_line": recommendation_line,
    }
