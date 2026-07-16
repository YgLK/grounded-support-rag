from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock
from uuid import UUID

from langsmith import Client

from support_graph.evaluation.contracts import DatasetRef, FeedbackValue
from support_graph.evaluation.langsmith_gateway import SdkLangSmithGateway


class FakeSdkClient:
    def __init__(self) -> None:
        self.dataset = SimpleNamespace(
            id=UUID("00000000-0000-0000-0000-000000000001"),
            name="support-graph/kubernetes/smoke/abc123def456",
            metadata={"dataset_sha256": "abc123def456"},
            url="https://smith.example/datasets/1",
        )
        self.events: list[str] = []

    def read_dataset(self, *, dataset_name: str):
        assert dataset_name == self.dataset.name
        return self.dataset

    def read_project(self, *, project_name: str):
        self.events.append("read_project")
        assert project_name == "support-graph-kubernetes-smoke-1"
        return SimpleNamespace(
            id=UUID("00000000-0000-0000-0000-000000000002"),
            name=project_name,
            extra={"metadata": {"metric_schema_version": "1"}},
            url="https://smith.example/projects/1",
        )


class FakeEvaluationResults:
    experiment_name = "support-graph-kubernetes-smoke-1"

    def __init__(self, sdk: FakeSdkClient) -> None:
        self.sdk = sdk

    def __aiter__(self):
        return self

    async def __anext__(self):
        if getattr(self, "_yielded", False):
            raise StopAsyncIteration
        self._yielded = True
        self.sdk.events.append("row")
        return {
            "run": SimpleNamespace(
                id=UUID("00000000-0000-0000-0000-000000000003"),
                outputs={"answer": "A Kubernetes Pod"},
                error=None,
            ),
            "example": SimpleNamespace(
                id="k8s-001",
                inputs={"question": "What is a Pod?"},
                outputs={"reference_answer": "A Pod"},
            ),
            "evaluation_results": {
                "results": [{"key": "doc_recall_at_3", "score": 1.0}],
            },
        }


def test_gateway_normalizes_evaluation_rows(monkeypatch) -> None:
    sdk = FakeSdkClient()

    async def fake_aevaluate(target, **kwargs):
        assert target is not None
        assert kwargs["data"] == sdk.dataset.name
        assert kwargs["error_handling"] == "log"
        assert kwargs["upload_results"] is True
        return FakeEvaluationResults(sdk)

    monkeypatch.setattr(
        "support_graph.evaluation.langsmith_gateway.aevaluate",
        fake_aevaluate,
    )
    gateway = SdkLangSmithGateway(cast(Client, sdk))

    snapshot = asyncio.run(
        gateway.evaluate(
            target=AsyncMock(),
            dataset_name=sdk.dataset.name,
            evaluators=[AsyncMock()],
            metadata={"metric_schema_version": "1"},
            experiment_prefix="support-graph-kubernetes-smoke",
            max_concurrency=2,
        )
    )

    assert sdk.events == ["row", "read_project"]
    assert snapshot.dataset.name == sdk.dataset.name
    assert snapshot.results[0].example_id == "k8s-001"
    assert snapshot.results[0].feedback == (
        FeedbackValue(key="doc_recall_at_3", score=1.0),
    )


def test_gateway_lists_experiments_without_async_generator_mismatch(
    monkeypatch,
) -> None:
    sdk = FakeSdkClient()
    gateway = SdkLangSmithGateway(cast(Client, sdk))

    monkeypatch.setattr(
        sdk,
        "list_projects",
        lambda **kwargs: iter([SimpleNamespace(id="project-1")]),
        raising=False,
    )

    async def fake_read_experiment(experiment_id: str):
        return SimpleNamespace(id=experiment_id)

    monkeypatch.setattr(gateway, "read_experiment", fake_read_experiment)

    snapshots = asyncio.run(gateway.list_experiments("dataset-1"))

    assert [snapshot.id for snapshot in snapshots] == ["project-1"]


def test_gateway_does_not_recreate_existing_dataset_examples(monkeypatch) -> None:
    sdk = FakeSdkClient()
    gateway = SdkLangSmithGateway(cast(Client, sdk))
    example_id = UUID("00000000-0000-0000-0000-000000000004")
    create_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        sdk,
        "list_examples",
        lambda **kwargs: iter([SimpleNamespace(id=example_id)]),
        raising=False,
    )
    monkeypatch.setattr(
        sdk,
        "create_examples",
        lambda **kwargs: create_calls.append(kwargs),
        raising=False,
    )

    asyncio.run(
        gateway.upsert_dataset_examples(
            DatasetRef("dataset-1", "dataset", "hash"),
            [{"id": str(example_id)}],
        )
    )

    assert create_calls == []
