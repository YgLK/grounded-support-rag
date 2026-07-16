from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from support_graph.evaluation.contracts import (
    DatasetRef,
    ExampleResult,
    ExperimentSnapshot,
    FeedbackValue,
)
from support_graph.evaluation.hosted import (
    build_feedback_evaluator,
    build_hosted_target,
    feedback_from_record,
    run_hosted_evaluation,
)
from support_graph.evaluation.metadata import GitState
from support_graph.evaluation.policy import MetricPolicy, RegressionPolicy


def test_hosted_target_uses_and_removes_temporary_trace(make_runtime_config) -> None:
    seen_trace_paths: list[Path] = []

    async def fake_run_graph(**kwargs: Any) -> dict[str, Any]:
        trace_path = Path(kwargs["trace_path"])
        seen_trace_paths.append(trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text('{"kind":"start"}\n', encoding="utf-8")
        return prediction(trace_path=str(trace_path))

    target = build_hosted_target(
        config=make_runtime_config(),
        run_graph_func=fake_run_graph,
        runtime_resources=None,
    )

    output = asyncio.run(target({"example": example("k8s-001")}))

    assert output["trace_summary"].get("trace_path") is None
    assert seen_trace_paths
    assert not seen_trace_paths[0].exists()


def test_hosted_target_propagates_graph_errors(make_runtime_config) -> None:
    async def fake_run_graph(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("graph unavailable")

    target = build_hosted_target(
        config=make_runtime_config(),
        run_graph_func=fake_run_graph,
        runtime_resources=None,
    )

    with pytest.raises(RuntimeError, match="graph unavailable"):
        asyncio.run(target({"example": example("k8s-001")}))


def test_feedback_from_record_keeps_numeric_and_categorical_feedback() -> None:
    feedback = feedback_from_record(
        {
            "metrics": {"missing": None, "ok": True, "score": 0.75},
            "decision": "answer",
            "failure_label": None,
        }
    )

    assert feedback == [
        {"key": "ok", "score": 1.0},
        {"key": "score", "score": 0.75},
        {"key": "decision", "value": "answer"},
        {"key": "failure_label", "value": "none"},
    ]


def test_feedback_evaluator_adds_judge_feedback(make_runtime_config) -> None:
    class Judge:
        enabled = True

        async def evaluate(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "faithful": 1.0,
                "answer_relevant": 1.0,
                "context_relevant": 1.0,
            }

    evaluator = build_feedback_evaluator(
        config=make_runtime_config(),
        policy=policy(),
        judge=Judge(),
    )
    result = asyncio.run(
        evaluator(
            SimpleNamespace(outputs=prediction()),
            SimpleNamespace(inputs={"example": rag_example("k8s-001")}),
        )
    )

    assert {item["key"] for item in result} >= {
        "doc_recall_at_3",
        "decision",
        "failure_label",
    }
    assert (
        next(item for item in result if item["key"] == "faithfulness")["score"] == 1.0
    )


def test_feedback_evaluator_reports_scorer_errors_without_zero_scores(
    make_runtime_config,
) -> None:
    class Judge:
        enabled = True

        async def evaluate(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("judge unavailable")

    evaluator = build_feedback_evaluator(
        config=make_runtime_config(),
        policy=policy(),
        judge=Judge(),
    )
    result = asyncio.run(
        evaluator(
            SimpleNamespace(outputs=prediction()),
            SimpleNamespace(inputs={"example": rag_example("k8s-001")}),
        )
    )

    assert result == [
        {"key": key, "error": "RuntimeError: judge unavailable"}
        for key in policy().required_feedback
    ]


def test_run_hosted_evaluation_marks_partial_execution_invalid(
    make_runtime_config,
    tmp_path: Path,
) -> None:
    config = hosted_config(make_runtime_config, tmp_path)
    gateway = FakeGateway(
        results=(
            result("k8s-001"),
            result("k8s-002", error="graph timeout"),
        )
    )

    hosted = asyncio.run(
        run_hosted_evaluation(
            gateway=gateway,
            examples=[example("k8s-001"), example("k8s-002")],
            config=config,
            domain="kubernetes",
            subset="smoke",
            policy=policy(),
            corpus_ref="main",
            corpus_manifest_path=tmp_path / "manifest.json",
            git_state=GitState(sha="abc", dirty=False),
            baseline=snapshot("baseline"),
            run_graph_func=fake_run_graph,
        )
    )

    assert hosted.gate.status == "invalid"
    assert "candidate execution error: k8s-002" in hosted.gate.reasons
    assert gateway.updated_metadata["status"] == "invalid"


def test_run_hosted_evaluation_requires_promoted_baseline(
    make_runtime_config,
    tmp_path: Path,
) -> None:
    config = hosted_config(make_runtime_config, tmp_path)
    gateway = FakeGateway(results=(result("k8s-001"),))

    hosted = asyncio.run(
        run_hosted_evaluation(
            gateway=gateway,
            examples=[example("k8s-001")],
            config=config,
            domain="kubernetes",
            subset="smoke",
            policy=policy(),
            corpus_ref="main",
            corpus_manifest_path=tmp_path / "manifest.json",
            git_state=GitState(sha="abc", dirty=False),
            run_graph_func=fake_run_graph,
        )
    )

    assert hosted.gate.status == "invalid"
    assert hosted.gate.reasons == ("No promoted baseline for dataset",)
    assert gateway.updated_metadata["status"] == "completed"


class FakeGateway:
    def __init__(self, *, results: tuple[ExampleResult, ...]) -> None:
        self.results = results
        self.dataset: DatasetRef | None = None
        self.snapshot: ExperimentSnapshot | None = None
        self.updated_metadata: dict[str, Any] = {}

    async def ping(self) -> None:
        return None

    async def get_dataset(self, name: str) -> DatasetRef | None:
        return self.dataset if self.dataset and self.dataset.name == name else None

    async def create_dataset(self, **kwargs: Any) -> DatasetRef:
        self.dataset = DatasetRef(
            id="dataset-1",
            name=kwargs["name"],
            sha256=kwargs["metadata"]["dataset_sha256"],
        )
        return self.dataset

    async def evaluate(self, **kwargs: Any) -> ExperimentSnapshot:
        assert self.dataset is not None
        self.snapshot = ExperimentSnapshot(
            id="candidate",
            name="candidate",
            dataset=self.dataset,
            metadata=dict(kwargs["metadata"]),
            results=self.results,
        )
        return self.snapshot

    async def update_experiment_metadata(
        self,
        experiment_id: str,
        metadata: dict[str, Any],
    ) -> ExperimentSnapshot:
        assert self.snapshot is not None
        assert experiment_id == self.snapshot.id
        self.updated_metadata = metadata
        self.snapshot = replace(self.snapshot, metadata=dict(metadata))
        return self.snapshot

    async def read_experiment(self, experiment_id: str) -> ExperimentSnapshot:
        assert self.snapshot is not None
        assert experiment_id == self.snapshot.id
        return self.snapshot

    async def list_experiments(
        self,
        dataset_id: str,
    ) -> tuple[ExperimentSnapshot, ...]:
        assert self.snapshot is not None
        assert dataset_id == self.snapshot.dataset.id
        return (self.snapshot,)

    async def list_trace_records(
        self,
        experiment_id: str,
    ) -> tuple[dict[str, Any], ...]:
        assert self.snapshot is not None
        assert experiment_id == self.snapshot.id
        return ()


def policy() -> RegressionPolicy:
    return RegressionPolicy(
        schema_version=1,
        metric_schema_version="1",
        required_feedback=("doc_recall_at_3", "decision", "failure_label"),
        metrics=(
            MetricPolicy(
                key="doc_recall_at_3",
                direction="higher",
                required=True,
                max_absolute_regression=0.0,
                max_relative_regression=None,
                protect_examples=True,
            ),
        ),
    )


def example(example_id: str) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "target_mode": "answer",
        "target_turn": {"utterance": "A Pod is a deployable unit."},
        "gold_doc_ids": ["pods"],
        "gold_span_ids": ["pods#overview"],
    }


def rag_example(example_id: str) -> dict[str, Any]:
    return {
        **example(example_id),
        "expected_sources": ["pods"],
        "acceptable_sources": [],
        "required_points": ["deployable unit"],
        "forbidden_claims": [],
        "answer_type": "definition",
    }


def prediction(*, trace_path: str | None = None) -> dict[str, Any]:
    trace_summary: dict[str, Any] = {"latency_ms": 1.0}
    if trace_path is not None:
        trace_summary["trace_path"] = trace_path
    return {
        "decision": "answer",
        "response_text": "A Pod is a deployable unit.",
        "citations": [{"chunk_id": "chunk-1", "span_ids": ["pods#overview"]}],
        "retrieval_ranked_chunks": [
            {
                "doc_id": "pods",
                "chunk_id": "chunk-1",
                "span_ids": ["pods#overview"],
                "text": "A Pod is a deployable unit.",
            }
        ],
        "retrieved_chunks": [
            {
                "doc_id": "pods",
                "chunk_id": "chunk-1",
                "span_ids": ["pods#overview"],
                "text": "A Pod is a deployable unit.",
            }
        ],
        "trace_summary": trace_summary,
    }


async def fake_run_graph(**kwargs: Any) -> dict[str, Any]:
    return prediction()


def result(example_id: str, *, error: str | None = None) -> ExampleResult:
    return ExampleResult(
        example_id=example_id,
        run_id=f"run-{example_id}",
        inputs={},
        reference_outputs={},
        outputs={},
        feedback=(
            FeedbackValue(key="doc_recall_at_3", score=1.0),
            FeedbackValue(key="decision", value="answer"),
            FeedbackValue(key="failure_label", value="none"),
        ),
        error=error,
    )


def snapshot(experiment_id: str) -> ExperimentSnapshot:
    return ExperimentSnapshot(
        id=experiment_id,
        name=experiment_id,
        dataset=DatasetRef("dataset-1", "dataset", "hash"),
        metadata={
            "status": "completed",
            "metric_schema_version": "1",
            "evaluator_versions": {"deterministic": "1", "rag_triad": "1"},
            "judge_provider": None,
            "judge_model": None,
        },
        results=(result("k8s-001"), result("k8s-002")),
    )


def hosted_config(make_runtime_config: Any, tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    chunk_artifact = tmp_path / "chunks.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    chunk_artifact.write_text("{}\n", encoding="utf-8")
    return make_runtime_config(
        postgres_dsn="postgresql://localhost/support_graph",
        chat_model="chat-model",
        embedding_model="embed-model",
        openrouter_api_key="or-key",
        langsmith_tracing_enabled=True,
        langsmith_api_key="ls-key",
        langsmith_project="support-graph",
        chunk_artifact_path=chunk_artifact,
    )
