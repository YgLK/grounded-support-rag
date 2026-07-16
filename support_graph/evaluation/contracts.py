"""Stable evaluation domain contracts and LangSmith boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

GateStatus = Literal["passed", "regressed", "invalid"]


@dataclass(frozen=True, slots=True)
class DatasetRef:
    id: str
    name: str
    sha256: str
    url: str | None = None


@dataclass(frozen=True, slots=True)
class FeedbackValue:
    key: str
    score: float | None = None
    value: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ExampleResult:
    example_id: str
    run_id: str
    inputs: dict[str, Any]
    reference_outputs: dict[str, Any]
    outputs: dict[str, Any]
    feedback: tuple[FeedbackValue, ...]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ExperimentSnapshot:
    id: str
    name: str
    dataset: DatasetRef
    metadata: dict[str, Any]
    results: tuple[ExampleResult, ...]
    url: str | None = None


@dataclass(frozen=True, slots=True)
class MetricDelta:
    key: str
    baseline: float
    candidate: float
    regression: float
    allowed_regression: float
    regressed_example_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GateResult:
    status: GateStatus
    baseline_experiment_id: str | None
    candidate_experiment_id: str
    deltas: tuple[MetricDelta, ...] = ()
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HostedEvaluation:
    experiment: ExperimentSnapshot
    aggregate_metrics: dict[str, float]
    gate: GateResult


class LangSmithGateway(Protocol):
    async def ping(self) -> None: ...

    async def get_dataset(self, name: str) -> DatasetRef | None: ...

    async def create_dataset(
        self,
        *,
        name: str,
        description: str,
        metadata: dict[str, Any],
        examples: list[dict[str, Any]],
    ) -> DatasetRef: ...

    async def evaluate(
        self,
        *,
        target: Any,
        dataset_name: str,
        evaluators: list[Any],
        metadata: dict[str, Any],
        experiment_prefix: str,
        max_concurrency: int,
    ) -> ExperimentSnapshot: ...

    async def read_experiment(self, experiment_id: str) -> ExperimentSnapshot: ...

    async def list_experiments(
        self, dataset_id: str
    ) -> tuple[ExperimentSnapshot, ...]: ...

    async def update_experiment_metadata(
        self, experiment_id: str, metadata: dict[str, Any]
    ) -> ExperimentSnapshot: ...

    async def list_trace_records(
        self, experiment_id: str
    ) -> tuple[dict[str, Any], ...]: ...


__all__ = [
    "DatasetRef",
    "ExampleResult",
    "ExperimentSnapshot",
    "FeedbackValue",
    "GateResult",
    "GateStatus",
    "HostedEvaluation",
    "LangSmithGateway",
    "MetricDelta",
]
