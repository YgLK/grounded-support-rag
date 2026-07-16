from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from support_graph.evaluation.variance import (
    RunMetrics,
    bootstrap_mean_ci,
    mean_std,
    recommended_k,
    retrieval_determinism_check,
    retrieval_signature,
    run_variance_study_async,
)
from support_graph.evaluation.contracts import (
    DatasetRef,
    ExampleResult,
    ExperimentSnapshot,
    FeedbackValue,
)


def test_variance_study_tags_each_hosted_repetition(monkeypatch, make_settings) -> None:
    calls: list[dict] = []
    settings = make_settings()
    snapshot = ExperimentSnapshot(
        id="exp-1",
        name="variance",
        dataset=DatasetRef("dataset", "dataset", "hash"),
        metadata={"status": "completed"},
        results=(
            ExampleResult(
                example_id="ex-1",
                run_id="run-1",
                inputs={},
                reference_outputs={},
                outputs={"retrieval_ranked_chunks": [{"chunk_id": "chunk-1"}]},
                feedback=(
                    FeedbackValue("required_points_covered", score=1.0),
                    FeedbackValue("citation_coverage", score=1.0),
                    FeedbackValue("failure_label", value="none"),
                ),
            ),
        ),
        url="https://smith/exp-1",
    )

    async def fake_hosted(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(experiment=snapshot)

    monkeypatch.setattr(
        "support_graph.evaluation.variance.run_hosted_evaluation", fake_hosted
    )

    report = asyncio.run(
        run_variance_study_async(
            settings=settings,
            domain="kubernetes",
            split="validation",
            subset="smoke",
            examples=[{"example_id": "ex-1"}],
            config=SimpleNamespace(),
            repeat=2,
            now=datetime(2026, 7, 16, tzinfo=timezone.utc),
            gateway=object(),
            policy=object(),
            corpus_manifest_path=settings.dataset.root / "manifest.json",
            git_state=SimpleNamespace(),
        )
    )

    assert len(calls) == 2
    assert [call["experiment_metadata"] for call in calls] == [
        {
            "study_type": "variance",
            "study_id": "20260716-kubernetes-smoke-variance",
            "variant": "control",
            "repetition": 1,
        },
        {
            "study_type": "variance",
            "study_id": "20260716-kubernetes-smoke-variance",
            "variant": "control",
            "repetition": 2,
        },
    ]
    assert report.experiment_ids == ["exp-1", "exp-1"]


def test_mean_std_handles_empty_and_single_value() -> None:
    assert mean_std([]) == (0.0, 0.0)
    assert mean_std([0.5]) == (0.5, 0.0)
    avg, std = mean_std([0.0, 1.0])
    assert avg == 0.5
    # sample std (N-1): sqrt((0.25+0.25)/1) = sqrt(0.5)
    assert std == pytest.approx(0.5**0.5)


def test_mean_std_uses_sample_std_not_population_std() -> None:
    """For n>=3, sample std (N-1) is larger than population std (N).

    With values [0.0, 0.5, 1.0]: mean=0.5, sum of squared deviations = 0.5.
    Population std = sqrt(0.5/3) = sqrt(1/6) ≈ 0.408.
    Sample std = sqrt(0.5/2) = sqrt(0.25) = 0.5.
    This confirms we use stdev, not pstdev, so pilot-K estimates from small K
    are not biased low.
    """
    avg, std = mean_std([0.0, 0.5, 1.0])
    assert avg == 0.5
    assert std == pytest.approx(0.25**0.5)  # sample std = 0.5
    # sample std (0.5) > population std sqrt(1/6) ≈ 0.408 for n=3
    assert std > (1 / 6) ** 0.5


def test_bootstrap_mean_ci_is_reproducible_and_brackets_mean() -> None:
    values = [0.0, 0.5, 1.0, 0.5, 0.0, 1.0, 0.5, 0.0]
    lower_a, upper_a = bootstrap_mean_ci(values, seed=42)
    lower_b, upper_b = bootstrap_mean_ci(values, seed=42)
    assert (lower_a, upper_a) == (lower_b, upper_b)
    avg = sum(values) / len(values)
    assert lower_a <= avg <= upper_a
    assert lower_a < upper_a


def test_bootstrap_mean_ci_returns_point_for_single_value() -> None:
    lower, upper = bootstrap_mean_ci([0.667])
    assert lower == upper == 0.667


def test_recommended_k_returns_minimum_two_for_zero_std() -> None:
    assert recommended_k(0.0, desired_half_width=0.05) == 2


def test_recommended_k_grows_with_std_and_shrinks_with_half_width() -> None:
    small = recommended_k(0.05, desired_half_width=0.05)
    large = recommended_k(0.20, desired_half_width=0.05)
    assert large > small
    wider = recommended_k(0.20, desired_half_width=0.10)
    assert wider < large


def test_recommended_k_rejects_non_positive_half_width() -> None:
    with pytest.raises(ValueError):
        recommended_k(0.1, desired_half_width=0.0)


def test_retrieval_signature_is_stable_for_same_chunks() -> None:
    prediction = {
        "retrieval_ranked_chunks": [
            {"chunk_id": "a::1"},
            {"chunk_id": "a::2"},
        ]
    }
    assert retrieval_signature(prediction) == "a::1|a::2"
    assert retrieval_signature({}) == ""


def test_retrieval_determinism_check_passes_for_identical_signatures() -> None:
    runs = [
        RunMetrics(
            run_id="r1",
            failure_counts={},
            required_points_covered=1.0,
            citation_coverage=1.0,
            incomplete_answer_count=0,
            per_example_required_points=[1.0, 1.0],
            per_example_retrieval_signature=["a|b", "c|d"],
        ),
        RunMetrics(
            run_id="r2",
            failure_counts={},
            required_points_covered=0.9,
            citation_coverage=1.0,
            incomplete_answer_count=1,
            per_example_required_points=[1.0, 0.667],
            per_example_retrieval_signature=["a|b", "c|d"],
        ),
    ]
    deterministic, mismatches = retrieval_determinism_check(runs)
    assert deterministic
    assert mismatches == []


def test_retrieval_determinism_check_flags_mismatched_signatures() -> None:
    runs = [
        RunMetrics(
            run_id="r1",
            failure_counts={},
            required_points_covered=1.0,
            citation_coverage=1.0,
            incomplete_answer_count=0,
            per_example_required_points=[1.0],
            per_example_retrieval_signature=["a|b"],
        ),
        RunMetrics(
            run_id="r2",
            failure_counts={},
            required_points_covered=1.0,
            citation_coverage=1.0,
            incomplete_answer_count=0,
            per_example_required_points=[1.0],
            per_example_retrieval_signature=["x|y"],
        ),
    ]
    deterministic, mismatches = retrieval_determinism_check(runs)
    assert not deterministic
    assert mismatches == ["r2"]
