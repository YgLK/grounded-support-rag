"""Controlled model A/B comparison helpers.

Supports a possible migration to a DeepSeek model as a *controlled comparison*,
not a blind swap. The compatibility gate is the blocking prerequisite: a
candidate chat model must support
``with_structured_output(..., method="json_schema")`` across the runtime nodes
(``route_query``, ``grade_evidence``, ``generate_response``,
``resolve_without_answer``). When structured output fails, the runtime silently
drops to heuristic fallbacks; the gate detects this by asserting the
fallback-event count is ~0 on a small smoke run.

The A/B itself is run via the ``eval-variance`` harness with a per-arm
``chat_model`` override (already supported through ``replace(base_config,
chat_model=...)``), so the two arms differ only by model. No reindex is needed
(embeddings unchanged).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from support_graph.evaluation.evaluate import evaluate_examples_async
from support_graph.logging_utils import get_logger
from support_graph.types import DatasetSplitLike, DomainLike

__all__ = [
    "CompatibilityGateResult",
    "run_compatibility_gate_async",
    "fallback_count_for_result",
    "runtime_error_count_for_result",
]

logger = get_logger(__name__)


@dataclass(slots=True)
class CompatibilityGateResult:
    chat_model: str
    passed: bool
    total_fallbacks: int
    fallback_nodes: list[str]
    runtime_error_count: int
    run_id: str
    reason: str


def fallback_count_for_result(result: dict) -> tuple[int, list[str]]:
    """Sum fallback_count across all predictions and collect fallback nodes."""
    total = 0
    nodes: set[str] = set()
    for prediction in result.get("predictions", []) or []:
        trace_summary = prediction.get("trace_summary", {}) or {}
        total += int(trace_summary.get("fallback_count", 0) or 0)
        for node in trace_summary.get("fallback_nodes", []) or []:
            nodes.add(str(node))
    return total, sorted(nodes)


def runtime_error_count_for_result(result: dict) -> int:
    """Count predictions that have a runtime_error or a runtime_error failure label.

    ``evaluate_examples_async`` records per-example runtime errors as
    predictions with ``failure_label="runtime_error"`` and a ``runtime_error``
    dict, but with ``fallback_count=0``. A gate that only checks fallback count
    would mark an unusable model (bad slug, provider/API error, structured-
    output call raising) as passed.
    """
    count = 0
    for prediction in result.get("predictions", []) or []:
        if prediction.get("runtime_error") is not None:
            count += 1
            continue
        if prediction.get("failure_label") == "runtime_error":
            count += 1
    return count


async def run_compatibility_gate_async(
    *,
    settings: Any,
    domain: DomainLike,
    split: DatasetSplitLike,
    examples: list[dict[str, Any]],
    candidate_chat_model: str,
    base_config: Any,
    limit: int = 3,
    max_concurrency: int = 1,
    fallback_tolerance: int = 0,
    now: Any = None,
) -> CompatibilityGateResult:
    """Run a small smoke eval with a candidate chat model and gate on fallbacks.

    Builds a config that differs from ``base_config`` only by ``chat_model``,
    runs a ``limit``-example smoke eval, and asserts the total fallback-event
    count is within ``fallback_tolerance``. A high fallback count means
    ``with_structured_output(method="json_schema")`` is unsupported and the
    runtime is silently dropping to heuristic fallbacks — the A/B must not
    proceed until this passes.
    """
    from dataclasses import replace

    if not candidate_chat_model:
        raise ValueError("candidate_chat_model must be a non-empty model slug.")
    candidate_config = replace(base_config, chat_model=candidate_chat_model)
    result = await evaluate_examples_async(
        examples[:limit],
        settings=settings,
        domain=domain,
        split=split,
        subset_name="compatibility-gate",
        limit=None,
        notes=(
            f"compatibility gate for chat_model={candidate_chat_model} "
            "(structured-output json_schema support check)"
        ),
        config=candidate_config,
        run_id_slug="compat-gate",
        subset_label=f"{domain} {split} / compatibility gate / {candidate_chat_model}",
        max_concurrency=max_concurrency,
        now=now,
    )
    total_fallbacks, fallback_nodes = fallback_count_for_result(result)
    runtime_errors = runtime_error_count_for_result(result)
    passed = total_fallbacks <= fallback_tolerance and runtime_errors == 0
    if runtime_errors > 0:
        reason = (
            f"candidate model could not run: {runtime_errors} prediction(s) "
            "recorded runtime_error (bad slug, provider/API error, or "
            "structured-output call raising). The gate cannot confirm "
            "json_schema support; do not A/B until the model runs cleanly."
        )
    elif total_fallbacks > fallback_tolerance:
        reason = (
            f"structured output likely unsupported: {total_fallbacks} fallback "
            f"events in nodes {fallback_nodes}. The runtime silently dropped to "
            "heuristic fallbacks; do not A/B until json_schema works."
        )
    else:
        reason = (
            "structured output supported; no runtime errors and fallback count "
            "within tolerance."
        )
    return CompatibilityGateResult(
        chat_model=candidate_chat_model,
        passed=passed,
        total_fallbacks=total_fallbacks,
        fallback_nodes=fallback_nodes,
        runtime_error_count=runtime_errors,
        run_id=str(result.get("run_id", "")),
        reason=reason,
    )
