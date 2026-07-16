"""Versioned regression policies for hosted evaluation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import tomllib

from support_graph.evaluation.contracts import (
    ExampleResult,
    ExperimentSnapshot,
    GateResult,
    MetricDelta,
)


@dataclass(frozen=True, slots=True)
class MetricPolicy:
    key: str
    direction: Literal["higher", "lower"]
    required: bool
    max_absolute_regression: float | None
    max_relative_regression: float | None
    protect_examples: bool


@dataclass(frozen=True, slots=True)
class RegressionPolicy:
    schema_version: int
    metric_schema_version: str
    required_feedback: tuple[str, ...]
    metrics: tuple[MetricPolicy, ...]


def load_policy(path: Path) -> RegressionPolicy:
    """Load a validated, versioned regression policy from TOML."""
    with path.open("rb") as source:
        raw = tomllib.load(source)

    required_feedback = _string_tuple(
        raw.get("required_feedback", []), "required_feedback"
    )
    raw_metrics = raw.get("metrics", {})
    if not isinstance(raw_metrics, dict):
        raise ValueError("metrics must be a TOML table")

    metrics = tuple(
        _metric_policy(key, config)
        for key, config in raw_metrics.items()
        if isinstance(key, str)
    )
    if len(metrics) != len(raw_metrics):
        raise ValueError("metric keys must be strings")

    return RegressionPolicy(
        schema_version=_integer(raw.get("schema_version"), "schema_version"),
        metric_schema_version=_string(
            raw.get("metric_schema_version"), "metric_schema_version"
        ),
        required_feedback=required_feedback,
        metrics=metrics,
    )


def compare_experiments(
    candidate: ExperimentSnapshot,
    baseline: ExperimentSnapshot,
    policy: RegressionPolicy,
) -> GateResult:
    """Classify a candidate against a compatible completed baseline."""
    reasons = _compatibility_reasons(candidate, baseline, policy)
    if reasons:
        return _invalid(candidate, baseline, reasons)

    deltas: list[MetricDelta] = []
    for metric in policy.metrics:
        baseline_scores, baseline_reason = _scores(baseline.results, metric.key)
        candidate_scores, candidate_reason = _scores(candidate.results, metric.key)
        if baseline_reason or candidate_reason:
            reasons = tuple(
                reason for reason in (baseline_reason, candidate_reason) if reason
            )
            return _invalid(candidate, baseline, reasons)

        baseline_average = _average(baseline_scores)
        candidate_average = _average(candidate_scores)
        regression = _regression(metric.direction, baseline_average, candidate_average)
        allowed = _allowed_regression(metric, baseline_average)
        regressed_examples = _regressed_example_ids(
            candidate.results,
            baseline.results,
            metric,
        )
        if regressed_examples is None:
            return _invalid(
                candidate,
                baseline,
                (f"missing examples for protected metric: {metric.key}",),
            )
        deltas.append(
            MetricDelta(
                key=metric.key,
                baseline=baseline_average,
                candidate=candidate_average,
                regression=regression,
                allowed_regression=allowed,
                regressed_example_ids=regressed_examples,
            )
        )

    status = (
        "regressed"
        if any(
            delta.regression > delta.allowed_regression or delta.regressed_example_ids
            for delta in deltas
        )
        else "passed"
    )
    return GateResult(
        status=status,
        baseline_experiment_id=baseline.id,
        candidate_experiment_id=candidate.id,
        deltas=tuple(deltas),
    )


def _metric_policy(key: str, raw: object) -> MetricPolicy:
    if not isinstance(raw, dict):
        raise ValueError(f"metric {key} must be a TOML table")

    direction = _string(raw.get("direction"), f"metric {key} direction")
    if direction not in {"higher", "lower"}:
        raise ValueError(f"metric {key} direction must be higher or lower")
    absolute = _optional_number(raw.get("max_absolute_regression"), key)
    relative = _optional_number(raw.get("max_relative_regression"), key)
    if absolute is None and relative is None:
        raise ValueError(f"metric {key} needs absolute or relative tolerance")

    return MetricPolicy(
        key=key,
        direction=cast(Literal["higher", "lower"], direction),
        required=_boolean(raw.get("required", False), f"metric {key} required"),
        max_absolute_regression=absolute,
        max_relative_regression=relative,
        protect_examples=_boolean(
            raw.get("protect_examples", False),
            f"metric {key} protect_examples",
        ),
    )


def _compatibility_reasons(
    candidate: ExperimentSnapshot,
    baseline: ExperimentSnapshot,
    policy: RegressionPolicy,
) -> tuple[str, ...]:
    reasons: list[str] = []
    for label, experiment in (("candidate", candidate), ("baseline", baseline)):
        if experiment.metadata.get("status") != "completed":
            reasons.append(f"{label} execution status is not completed")
        if any(result.error for result in experiment.results):
            reasons.append(f"{label} has execution errors")
        reasons.extend(_missing_feedback_reasons(experiment, policy.required_feedback))

    if candidate.dataset.name != baseline.dataset.name:
        reasons.append("dataset name mismatch")
    if candidate.dataset.sha256 != baseline.dataset.sha256:
        reasons.append("dataset hash mismatch")
    if candidate.metadata.get("metric_schema_version") != baseline.metadata.get(
        "metric_schema_version"
    ):
        reasons.append("metric schema version mismatch")
    if candidate.metadata.get("metric_schema_version") != policy.metric_schema_version:
        reasons.append("candidate metric schema version does not match policy")
    if candidate.metadata.get("evaluator_versions") != baseline.metadata.get(
        "evaluator_versions"
    ):
        reasons.append("evaluator versions mismatch")
    for key in ("judge_provider", "judge_model"):
        if candidate.metadata.get(key) != baseline.metadata.get(key):
            reasons.append(f"{key} mismatch")
    return tuple(reasons)


def _missing_feedback_reasons(
    experiment: ExperimentSnapshot,
    required_feedback: tuple[str, ...],
) -> list[str]:
    available = {
        feedback.key
        for result in experiment.results
        for feedback in result.feedback
        if feedback.error is None
        and (feedback.score is not None or feedback.value is not None)
    }
    return [
        f"missing required feedback: {key}"
        for key in required_feedback
        if key not in available
    ]


def _scores(
    results: tuple[ExampleResult, ...], key: str
) -> tuple[tuple[float, ...], str | None]:
    scores: list[float] = []
    for result in results:
        feedback = next((item for item in result.feedback if item.key == key), None)
        if feedback is None or feedback.score is None or feedback.error is not None:
            return (), f"missing numeric feedback: {key}"
        scores.append(feedback.score)
    if not scores:
        return (), f"missing numeric feedback: {key}"
    return tuple(scores), None


def _regressed_example_ids(
    candidate: tuple[ExampleResult, ...],
    baseline: tuple[ExampleResult, ...],
    metric: MetricPolicy,
) -> tuple[str, ...] | None:
    if not metric.protect_examples:
        return ()
    candidate_by_id = {result.example_id: result for result in candidate}
    baseline_by_id = {result.example_id: result for result in baseline}
    if candidate_by_id.keys() != baseline_by_id.keys():
        return None

    regressed: list[str] = []
    for example_id, baseline_result in baseline_by_id.items():
        baseline_score = _score_for(baseline_result, metric.key)
        candidate_score = _score_for(candidate_by_id[example_id], metric.key)
        if baseline_score is None or candidate_score is None:
            return None
        regression = _regression(metric.direction, baseline_score, candidate_score)
        if regression > _allowed_regression(metric, baseline_score):
            regressed.append(example_id)
    return tuple(regressed)


def _score_for(result: ExampleResult, key: str) -> float | None:
    feedback = next((item for item in result.feedback if item.key == key), None)
    if feedback is None or feedback.error is not None:
        return None
    return feedback.score


def _regression(
    direction: Literal["higher", "lower"], baseline: float, candidate: float
) -> float:
    return baseline - candidate if direction == "higher" else candidate - baseline


def _allowed_regression(metric: MetricPolicy, baseline: float) -> float:
    tolerances = [
        tolerance
        for tolerance in (
            metric.max_absolute_regression,
            (
                metric.max_relative_regression * abs(baseline)
                if metric.max_relative_regression is not None and baseline != 0
                else None
            ),
        )
        if tolerance is not None
    ]
    return max(tolerances)


def _average(scores: tuple[float, ...]) -> float:
    return sum(scores) / len(scores)


def _invalid(
    candidate: ExperimentSnapshot,
    baseline: ExperimentSnapshot,
    reasons: tuple[str, ...],
) -> GateResult:
    return GateResult(
        status="invalid",
        baseline_experiment_id=baseline.id,
        candidate_experiment_id=candidate.id,
        reasons=reasons,
    )


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be an array of strings")
    return tuple(cast(list[str], value))


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _optional_number(value: object, key: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metric {key} tolerance must be numeric")
    if value < 0:
        raise ValueError(f"metric {key} tolerance must not be negative")
    return float(value)


__all__ = [
    "MetricPolicy",
    "RegressionPolicy",
    "compare_experiments",
    "load_policy",
]
