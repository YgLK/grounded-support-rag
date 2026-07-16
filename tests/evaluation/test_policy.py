from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from support_graph.evaluation.contracts import (
    DatasetRef,
    ExampleResult,
    ExperimentSnapshot,
    FeedbackValue,
)
from support_graph.evaluation.policy import (
    MetricPolicy,
    RegressionPolicy,
    compare_experiments,
    load_policy,
)


def test_load_policy_rejects_metric_without_tolerance(tmp_path: Path) -> None:
    path = tmp_path / "policy.toml"
    path.write_text(
        'schema_version = 1\nmetric_schema_version = "1"\n'
        '[metrics.doc_recall_at_3]\ndirection = "higher"\nrequired = true\n'
        "protect_examples = true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="needs absolute or relative tolerance"):
        load_policy(path)


def test_load_policy_rejects_unknown_direction(tmp_path: Path) -> None:
    path = write_policy(tmp_path, direction="sideways")

    with pytest.raises(ValueError, match="direction must be higher or lower"):
        load_policy(path)


def test_compare_experiments_reports_aggregate_regression() -> None:
    baseline = snapshot(metric="doc_recall_at_3", scores=[1.0, 1.0])
    candidate = snapshot(metric="doc_recall_at_3", scores=[1.0, 0.5], id="candidate")
    policy = regression_policy(
        MetricPolicy(
            key="doc_recall_at_3",
            direction="higher",
            required=True,
            max_absolute_regression=0.1,
            max_relative_regression=None,
            protect_examples=False,
        )
    )

    result = compare_experiments(candidate, baseline, policy)

    assert result.status == "regressed"
    assert result.deltas[0].regression == pytest.approx(0.25)
    assert result.deltas[0].allowed_regression == pytest.approx(0.1)


def test_compare_experiments_accepts_higher_metric_within_tolerance() -> None:
    result = compare_experiments(
        snapshot(metric="metric", scores=[1.0], id="candidate"),
        snapshot(metric="metric", scores=[1.05]),
        regression_policy(metric_policy("metric", maximum=0.1)),
    )

    assert result.status == "passed"


def test_compare_experiments_detects_lower_metric_regression() -> None:
    result = compare_experiments(
        snapshot(metric="latency", scores=[2.0], id="candidate"),
        snapshot(metric="latency", scores=[1.0]),
        regression_policy(metric_policy("latency", direction="lower", maximum=0.5)),
    )

    assert result.status == "regressed"


def test_compare_experiments_detects_protected_example_regression() -> None:
    result = compare_experiments(
        snapshot(metric="metric", scores=[1.0, 0.0], id="candidate"),
        snapshot(metric="metric", scores=[1.0, 1.0]),
        regression_policy(metric_policy("metric", maximum=0.0, protect_examples=True)),
    )

    assert result.status == "regressed"
    assert result.deltas[0].regressed_example_ids == ("example-2",)


def test_compare_experiments_rejects_missing_required_feedback() -> None:
    candidate = snapshot(metric="metric", scores=[1.0], id="candidate")
    result = compare_experiments(
        candidate,
        snapshot(metric="metric", scores=[1.0]),
        regression_policy(metric_policy("metric", maximum=0.0), required=("missing",)),
    )

    assert result.status == "invalid"
    assert "missing required feedback: missing" in result.reasons


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_sha256", "different"),
        ("metric_schema_version", "2"),
        ("evaluator_versions", {"deterministic": "2"}),
    ],
)
def test_compare_experiments_rejects_incompatible_contract(
    field: str,
    value: object,
) -> None:
    candidate = snapshot(metric="metric", scores=[1.0], id="candidate")
    metadata = dict(candidate.metadata)
    dataset = candidate.dataset
    if field == "dataset_sha256":
        dataset = DatasetRef(dataset.id, dataset.name, str(value))
    else:
        metadata[field] = value
    candidate = ExperimentSnapshot(
        id=candidate.id,
        name=candidate.name,
        dataset=dataset,
        metadata=metadata,
        results=candidate.results,
    )

    result = compare_experiments(
        candidate,
        snapshot(metric="metric", scores=[1.0]),
        regression_policy(metric_policy("metric", maximum=0.0)),
    )

    assert result.status == "invalid"


def test_compare_experiments_rejects_incomplete_baseline() -> None:
    baseline = snapshot(metric="metric", scores=[1.0], status="running")

    result = compare_experiments(
        snapshot(metric="metric", scores=[1.0], id="candidate"),
        baseline,
        regression_policy(metric_policy("metric", maximum=0.0)),
    )

    assert result.status == "invalid"


def test_compare_experiments_rejects_candidate_execution_error() -> None:
    candidate = snapshot(metric="metric", scores=[1.0], id="candidate", error="timeout")

    result = compare_experiments(
        candidate,
        snapshot(metric="metric", scores=[1.0]),
        regression_policy(metric_policy("metric", maximum=0.0)),
    )

    assert result.status == "invalid"


def write_policy(tmp_path: Path, *, direction: str) -> Path:
    path = tmp_path / "policy.toml"
    path.write_text(
        'schema_version = 1\nmetric_schema_version = "1"\n'
        'required_feedback = ["metric", "decision", "failure_label"]\n'
        "[metrics.metric]\n"
        f'direction = "{direction}"\n'
        "required = true\n"
        "max_absolute_regression = 0.0\n"
        "protect_examples = false\n",
        encoding="utf-8",
    )
    return path


def metric_policy(
    key: str,
    *,
    direction: Literal["higher", "lower"] = "higher",
    maximum: float,
    protect_examples: bool = False,
) -> MetricPolicy:
    return MetricPolicy(
        key=key,
        direction=direction,
        required=True,
        max_absolute_regression=maximum,
        max_relative_regression=None,
        protect_examples=protect_examples,
    )


def regression_policy(
    *metrics: MetricPolicy,
    required: tuple[str, ...] | None = None,
) -> RegressionPolicy:
    required_feedback = required or tuple(metric.key for metric in metrics)
    return RegressionPolicy(
        schema_version=1,
        metric_schema_version="1",
        required_feedback=(*required_feedback, "decision", "failure_label"),
        metrics=metrics,
    )


def snapshot(
    *,
    metric: str,
    scores: list[float],
    id: str = "baseline",
    status: str = "completed",
    error: str | None = None,
) -> ExperimentSnapshot:
    results = tuple(
        ExampleResult(
            example_id=f"example-{index}",
            run_id=f"run-{index}",
            inputs={},
            reference_outputs={},
            outputs={},
            feedback=(
                FeedbackValue(key=metric, score=score),
                FeedbackValue(key="decision", value="pass"),
                FeedbackValue(key="failure_label", value="none"),
            ),
            error=error,
        )
        for index, score in enumerate(scores, start=1)
    )
    return ExperimentSnapshot(
        id=id,
        name=id,
        dataset=DatasetRef("dataset", "support-graph/kubernetes/smoke/hash", "hash"),
        metadata={
            "status": status,
            "metric_schema_version": "1",
            "evaluator_versions": {"deterministic": "1"},
            "judge_provider": None,
            "judge_model": None,
        },
        results=results,
    )
