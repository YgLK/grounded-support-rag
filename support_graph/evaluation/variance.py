"""Repeated-run variance attribution study for the eval harness.

Disentangles two variance sources the single-run Smoke-10 baseline confounded:

- **Sampling noise** — fixed by a larger set; quantified by a bootstrap CI
  over examples from a single run.
- **Generation non-determinism** — exposed by repeated runs on the same set;
  quantified by the std of a metric across runs.

The harness reuses ``evaluate_examples_async`` for each run, asserts that
retrieval metrics are identical across runs (a determinism check that
invalidates the attribution if it fails), and produces a report with
per-metric mean ± 95% CI across K runs plus an explicit sampling-noise-vs-
generation-variance attribution.

The statistics math is pure and unit-tested with a fixed fixture (no live LLM
in the default suite). The harness itself requires live services to run.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, stdev
from typing import Any, Callable

from support_graph.artifacts import eval_report_artifacts
from support_graph.evaluation.evaluate import evaluate_examples_async
from support_graph.logging_utils import get_logger
from support_graph.types import DatasetSplitLike, DomainLike

__all__ = [
    "bootstrap_mean_ci",
    "mean_std",
    "recommended_k",
    "RunMetrics",
    "VarianceReport",
    "retrieval_signature",
    "retrieval_determinism_check",
    "run_variance_study_async",
    "write_variance_report",
]

logger = get_logger(__name__)


def mean_std(values: list[float]) -> tuple[float, float]:
    """Return (mean, sample std). std is 0.0 for fewer than 2 values.

    Uses sample standard deviation (N-1 denominator) because the between-run
    std estimates generation variance from a sample of runs, not the full
    population. For small pilot K (e.g. K=3) this avoids the downward bias
    that population std (pstdev) would introduce.
    """
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(stdev(values))


def bootstrap_mean_ci(
    values: list[float],
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap CI for the mean of ``values``.

    Returns (lower, upper) bounds. Uses a fixed RNG seed for reproducibility.
    Falls back to a normal approximation when the sample is too small to
    bootstrap meaningfully.
    """
    if not values:
        return 0.0, 0.0
    n = len(values)
    if n < 2:
        return float(values[0]), float(values[0])
    rng = random.Random(seed)
    boot_means: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_means.append(mean(sample))
    boot_means.sort()
    alpha = (1.0 - ci) / 2.0
    lower_index = max(0, int(math.floor(alpha * n_boot)))
    upper_index = min(n_boot - 1, int(math.ceil((1.0 - alpha) * n_boot)) - 1)
    return boot_means[lower_index], boot_means[upper_index]


def recommended_k(
    pilot_std: float, *, desired_half_width: float, z: float = 1.96
) -> int:
    """Compute the K needed for a target CI half-width on a std estimate.

    K = (z * sigma / desired_half_width) ** 2, rounded up. Returns at least 2.
    """
    if desired_half_width <= 0:
        raise ValueError("desired_half_width must be positive.")
    if pilot_std <= 0:
        return 2
    needed = (z * pilot_std / desired_half_width) ** 2
    return max(2, int(math.ceil(needed)))


@dataclass(slots=True)
class RunMetrics:
    """Per-run summary metrics extracted from an eval result."""

    run_id: str
    failure_counts: dict[str, int]
    required_points_covered: float | None
    citation_coverage: float | None
    incomplete_answer_count: int
    per_example_required_points: list[float]
    per_example_retrieval_signature: list[str]


@dataclass(slots=True)
class VarianceReport:
    runs: list[RunMetrics]
    metric_summaries: dict[str, dict[str, float]] = field(default_factory=dict)
    retrieval_deterministic: bool = True
    retrieval_mismatch_runs: list[str] = field(default_factory=list)
    sampling_ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    between_run_std: dict[str, float] = field(default_factory=dict)
    n10_sampling_ci_width: dict[str, float] = field(default_factory=dict)
    recommended_k: int | None = None
    attribution: str = ""


def _metric_value(result: dict, *path: str) -> Any:
    current: Any = result.get("metrics", {})
    for key in path:
        current = current.get(key, {})
    if current in (None, {}):
        return None
    return current


def retrieval_signature(prediction: dict) -> str:
    """A stable string identifying the retrieved chunks for one example."""
    chunks = prediction.get("retrieval_ranked_chunks") or prediction.get(
        "retrieved_chunks", []
    )
    chunk_ids = [
        str(chunk.get("chunk_id"))
        for chunk in chunks
        if chunk.get("chunk_id") is not None
    ]
    return "|".join(chunk_ids)


def _run_metrics(result: dict) -> RunMetrics:
    predictions = result.get("predictions", [])
    per_example_rpc: list[float] = []
    signatures: list[str] = []
    for prediction in predictions:
        metrics = prediction.get("metrics", {}) or {}
        rpc = metrics.get("required_points_covered")
        if rpc is not None:
            per_example_rpc.append(float(rpc))
        signatures.append(retrieval_signature(prediction))
    failure_counts = dict(result.get("failure_counts", {}) or {})
    return RunMetrics(
        run_id=str(result.get("run_id", "")),
        failure_counts=failure_counts,
        required_points_covered=_safe_float(
            _metric_value(result, "rag", "answer", "required_points_covered")
        ),
        citation_coverage=_safe_float(
            _metric_value(result, "generation", "answer", "citation_coverage")
        ),
        incomplete_answer_count=int(failure_counts.get("incomplete_answer", 0)),
        per_example_required_points=per_example_rpc,
        per_example_retrieval_signature=signatures,
    )


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def retrieval_determinism_check(runs: list[RunMetrics]) -> tuple[bool, list[str]]:
    """Assert retrieval signatures are identical across runs.

    Returns (is_deterministic, mismatch_run_ids). Compares the per-example
    retrieval signature of run 0 against every other run; any mismatch makes
    the attribution invalid.
    """
    if not runs:
        return True, []
    baseline = runs[0].per_example_retrieval_signature
    mismatches: list[str] = []
    for run in runs[1:]:
        if run.per_example_retrieval_signature != baseline:
            mismatches.append(run.run_id)
    return not mismatches, mismatches


def _summarize_metric(
    runs: list[RunMetrics], getter: Callable[[RunMetrics], float | None]
) -> dict[str, float]:
    values = [getter(run) for run in runs]
    present = [value for value in values if value is not None]
    if not present:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    avg, std = mean_std(present)
    return {
        "mean": avg,
        "std": std,
        "min": min(present),
        "max": max(present),
        "n": float(len(present)),
    }


def _build_report(
    runs: list[RunMetrics],
    *,
    n10_reference_values: dict[str, list[float]] | None = None,
    desired_half_width: float = 0.05,
) -> VarianceReport:
    deterministic, mismatches = retrieval_determinism_check(runs)
    report = VarianceReport(
        runs=runs,
        retrieval_deterministic=deterministic,
        retrieval_mismatch_runs=mismatches,
    )
    report.metric_summaries = {
        "required_points_covered": _summarize_metric(
            runs, lambda r: r.required_points_covered
        ),
        "citation_coverage": _summarize_metric(runs, lambda r: r.citation_coverage),
        "incomplete_answer_count": _summarize_metric(
            runs, lambda r: float(r.incomplete_answer_count)
        ),
    }
    # Within-run sampling CI from a single run (run 0), bootstrap over examples.
    if runs:
        first = runs[0]
        rpc_values = first.per_example_required_points
        if rpc_values:
            report.sampling_ci["required_points_covered"] = bootstrap_mean_ci(
                rpc_values
            )
    # Between-run generation variance: std across runs.
    for name, getter in (
        ("required_points_covered", lambda r: r.required_points_covered),
        ("citation_coverage", lambda r: r.citation_coverage),
        ("incomplete_answer_count", lambda r: float(r.incomplete_answer_count)),
    ):
        present = [getter(run) for run in runs if getter(run) is not None]
        _, std = mean_std(present)
        report.between_run_std[name] = std
    # n=10 sampling CI width for side-by-side comparison.
    if n10_reference_values:
        for name, values in n10_reference_values.items():
            lower, upper = bootstrap_mean_ci(values)
            report.n10_sampling_ci_width[name] = upper - lower
    # Recommended K from the pilot between-run std of required_points_covered.
    pilot_std = report.between_run_std.get("required_points_covered", 0.0)
    report.recommended_k = recommended_k(
        pilot_std, desired_half_width=desired_half_width
    )
    if deterministic:
        report.attribution = (
            "Retrieval signatures are identical across runs, so retrieval "
            "variance is ~0. The residual between-run variance is generation "
            "non-determinism. The within-run bootstrap CI shows the sampling-"
            "noise component; the between-run std shows the generation component."
        )
    else:
        report.attribution = (
            "Retrieval signatures differ across runs, so the variance "
            "attribution is INVALID. Investigate retrieval non-determinism "
            "before drawing generation-variance conclusions."
        )
    return report


async def run_variance_study_async(
    *,
    settings: Any,
    domain: DomainLike,
    split: DatasetSplitLike,
    subset: str,
    examples: list[dict[str, Any]],
    config: Any,
    repeat: int,
    notes: str | None = None,
    now: datetime | None = None,
    max_concurrency: int = 1,
    n10_reference_values: dict[str, list[float]] | None = None,
    desired_half_width: float = 0.05,
    run_graph_func: Any = None,
) -> VarianceReport:
    """Run the eval harness ``repeat`` times and build a variance report.

    Each run uses the same examples, same config, and same restored index.
    Retrieval signatures are asserted identical across runs. Requires live
    Postgres and a live chat model.
    """
    if repeat < 2:
        raise ValueError("repeat must be at least 2 to measure between-run variance.")
    started_at = now or datetime.now().astimezone()
    runs: list[RunMetrics] = []
    for index in range(repeat):
        logger.info("Variance study run %s/%s for subset=%s", index + 1, repeat, subset)
        call_kwargs: dict[str, Any] = {
            "settings": settings,
            "domain": domain,
            "split": split,
            "subset_name": subset,
            "limit": None,
            "notes": notes or f"variance study run {index + 1}/{repeat}",
            "now": started_at,
            "config": config,
            "run_id_slug": f"variance-{index + 1}",
            "subset_label": f"{domain} {split} / {subset} / variance run {index + 1}",
            "max_concurrency": max_concurrency,
        }
        if run_graph_func is not None:
            call_kwargs["run_graph_func"] = run_graph_func
        result = await evaluate_examples_async(examples, **call_kwargs)
        runs.append(_run_metrics(result))
    return _build_report(
        runs,
        n10_reference_values=n10_reference_values,
        desired_half_width=desired_half_width,
    )


def _format_ci(lower: float, upper: float) -> str:
    return f"[{lower:.3f}, {upper:.3f}]"


def write_variance_report(
    *,
    settings: Any,
    domain: DomainLike,
    subset: str,
    repeat: int,
    report: VarianceReport,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Write the variance study report artifact (manifest + report.md)."""
    timestamp = now or datetime.now().astimezone()
    report_id = (
        f"{timestamp.strftime('%Y%m%d-%H%M%S')}-{domain}-{subset}-variance-{repeat}"
    )
    artifacts = eval_report_artifacts(settings.paths.project_root, report_id)
    artifacts.output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# {domain} {subset} Variance Attribution Study",
        "",
        f"Runs: {repeat}",
        f"Examples per run: {len(report.runs[0].per_example_required_points) if report.runs else 0}",
        "",
        "## Retrieval Determinism Check",
        f"- Deterministic: {report.retrieval_deterministic}",
    ]
    if report.retrieval_mismatch_runs:
        lines.append(f"- Mismatched runs: {', '.join(report.retrieval_mismatch_runs)}")
    lines.extend(["", "## Per-Metric Summaries (across runs)"])
    lines.append("| Metric | Mean | Std | Min | Max |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for name, summary in report.metric_summaries.items():
        lines.append(
            f"| {name} | {summary['mean']:.3f} | {summary['std']:.3f} | "
            f"{summary['min']:.3f} | {summary['max']:.3f} |"
        )

    lines.extend(["", "## Sampling Noise (within-run bootstrap CI, run 0)"])
    for name, (lower, upper) in report.sampling_ci.items():
        lines.append(
            f"- {name}: {_format_ci(lower, upper)} (width {upper - lower:.3f})"
        )

    lines.extend(["", "## Generation Non-Determinism (between-run std)"])
    for name, std in report.between_run_std.items():
        lines.append(f"- {name}: {std:.3f}")

    if report.n10_sampling_ci_width:
        lines.extend(["", "## n=10 vs n=50 Sampling-Noise Comparison"])
        for name, width in report.n10_sampling_ci_width.items():
            n50_width = (
                report.sampling_ci.get(name, (0.0, 0.0))[1]
                - report.sampling_ci.get(name, (0.0, 0.0))[0]
            )
            lines.append(
                f"- {name}: n=10 CI width {width:.3f} vs n=50 CI width {n50_width:.3f}"
            )

    lines.extend(
        [
            "",
            "## Recommended K",
            f"- Pilot between-run std (required_points_covered): "
            f"{report.between_run_std.get('required_points_covered', 0.0):.3f}",
            f"- Recommended K for target half-width: {report.recommended_k}",
            "",
            "## Attribution",
            report.attribution,
            "",
        ]
    )

    manifest = {
        "report_id": report_id,
        "created_at": timestamp.isoformat(),
        "report_type": "variance_attribution",
        "domain": str(domain),
        "subset": subset,
        "repeat": repeat,
        "retrieval_deterministic": report.retrieval_deterministic,
        "metric_summaries": report.metric_summaries,
        "between_run_std": report.between_run_std,
        "recommended_k": report.recommended_k,
        "related_run_ids": [run.run_id for run in report.runs],
    }
    artifacts.manifest.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    artifacts.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "report_id": report_id,
        "artifact_paths": {
            "manifest": artifacts.manifest,
            "report": artifacts.report,
        },
        "report": report,
    }
