from __future__ import annotations

import asyncio
from typing import Any

from support_graph.evaluation.contracts import DatasetRef
from support_graph.evaluation.datasets import (
    dataset_name,
    dataset_sha256,
    publish_dataset,
)


class FakeGateway:
    def __init__(self) -> None:
        self.datasets: dict[str, DatasetRef] = {}
        self.create_dataset_calls: list[dict[str, Any]] = []
        self.upsert_dataset_examples_calls: list[dict[str, Any]] = []

    async def get_dataset(self, name: str) -> DatasetRef | None:
        return self.datasets.get(name)

    async def create_dataset(self, **kwargs: Any) -> DatasetRef:
        self.create_dataset_calls.append(kwargs)
        dataset = DatasetRef(
            id="dataset-2",
            name=kwargs["name"],
            sha256=kwargs["metadata"]["dataset_sha256"],
        )
        self.datasets[dataset.name] = dataset
        return dataset

    async def upsert_dataset_examples(
        self,
        dataset: DatasetRef,
        examples: list[dict[str, Any]],
    ) -> None:
        self.upsert_dataset_examples_calls.append(
            {"dataset": dataset, "examples": examples}
        )


def test_dataset_hash_is_stable_across_mapping_key_order() -> None:
    left = [{"example_id": "k8s-001", "question": "Q", "gold": {"b": 2, "a": 1}}]
    right = [{"gold": {"a": 1, "b": 2}, "question": "Q", "example_id": "k8s-001"}]

    assert dataset_sha256(left) == dataset_sha256(right)


def test_dataset_hash_changes_when_example_order_changes() -> None:
    left = [{"example_id": "a"}, {"example_id": "b"}]
    right = [{"example_id": "b"}, {"example_id": "a"}]

    assert dataset_sha256(left) != dataset_sha256(right)


def test_publish_dataset_reuses_matching_immutable_dataset() -> None:
    gateway = FakeGateway()
    examples = [{"example_id": "k8s-001", "latest_user_utterance": "Q"}]
    digest = dataset_sha256(examples)
    gateway.datasets[dataset_name("kubernetes", "smoke", digest)] = DatasetRef(
        id="dataset-1",
        name=dataset_name("kubernetes", "smoke", digest),
        sha256=digest,
    )

    published = asyncio.run(
        publish_dataset(
            gateway,
            domain="kubernetes",
            subset="smoke",
            examples=examples,
        )
    )

    assert published.id == "dataset-1"
    assert gateway.create_dataset_calls == []
    assert gateway.upsert_dataset_examples_calls[0]["dataset"] == published
    assert gateway.upsert_dataset_examples_calls[0]["examples"][0]["metadata"] == {
        "example_id": "k8s-001"
    }
