from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from support_graph.evaluation.model_ab import (
    fallback_count_for_result,
    run_compatibility_gate_async,
    runtime_error_count_for_result,
)
from support_graph.evaluation.contracts import DatasetRef, ExperimentSnapshot


def test_fallback_count_for_result_sums_across_predictions() -> None:
    result = {
        "predictions": [
            {"trace_summary": {"fallback_count": 1, "fallback_nodes": ["route_query"]}},
            {
                "trace_summary": {
                    "fallback_count": 2,
                    "fallback_nodes": ["answer", "route_query"],
                }
            },
            {"trace_summary": {}},
        ]
    }
    total, nodes = fallback_count_for_result(result)
    assert total == 3
    assert nodes == ["answer", "route_query"]


def test_fallback_count_for_result_handles_empty_predictions() -> None:
    total, nodes = fallback_count_for_result({})
    assert total == 0
    assert nodes == []


def test_runtime_error_count_detects_runtime_error_dict() -> None:
    result = {
        "predictions": [
            {"runtime_error": {"exception_type": "ValueError", "error": "bad slug"}},
            {"trace_summary": {"fallback_count": 0}},
        ]
    }
    assert runtime_error_count_for_result(result) == 1


def test_runtime_error_count_detects_runtime_error_failure_label() -> None:
    result = {
        "predictions": [
            {"failure_label": "runtime_error", "trace_summary": {"fallback_count": 0}},
            {
                "failure_label": "incomplete_answer",
                "trace_summary": {"fallback_count": 0},
            },
        ]
    }
    assert runtime_error_count_for_result(result) == 1


def test_runtime_error_count_handles_empty_predictions() -> None:
    assert runtime_error_count_for_result({}) == 0


def test_runtime_error_count_counts_both_forms() -> None:
    result = {
        "predictions": [
            {
                "runtime_error": {
                    "exception_type": "ConnectionError",
                    "error": "timeout",
                }
            },
            {"failure_label": "runtime_error"},
            {"failure_label": "weak_citations"},
        ]
    }
    assert runtime_error_count_for_result(result) == 2


def test_compatibility_gate_uses_tagged_hosted_experiment(
    monkeypatch,
    make_runtime_config,
):
    calls: list[dict] = []
    snapshot = ExperimentSnapshot(
        id="exp-model",
        name="model",
        dataset=DatasetRef("dataset", "dataset", "hash"),
        metadata={},
        results=(),
        url="https://smith/exp-model",
    )

    async def fake_hosted(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(experiment=snapshot)

    monkeypatch.setattr(
        "support_graph.evaluation.model_ab.run_hosted_evaluation", fake_hosted
    )
    result = asyncio.run(
        run_compatibility_gate_async(
            settings=SimpleNamespace(),
            domain="kubernetes",
            split="validation",
            examples=[{"example_id": "ex-1"}],
            candidate_chat_model="candidate-model",
            base_config=make_runtime_config(
                postgres_dsn="dsn",
                chat_model="chat",
                embedding_model="embed",
            ),
            gateway=object(),
            policy=object(),
            corpus_manifest_path=Path("manifest.json"),
            git_state=SimpleNamespace(),
        )
    )

    assert calls[0]["experiment_metadata"]["study_type"] == "model_compatibility"
    assert calls[0]["experiment_metadata"]["variant"] == "candidate-model"
    assert result.experiment_id == "exp-model"
    assert result.experiment_url == "https://smith/exp-model"
