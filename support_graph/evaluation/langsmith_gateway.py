"""LangSmith SDK adapter for the evaluation domain."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from typing import Any

from langsmith import Client
from langsmith.evaluation import aevaluate
from langsmith.utils import LangSmithNotFoundError

from support_graph.config.runtime import RuntimeConfig
from support_graph.evaluation.contracts import (
    DatasetRef,
    ExampleResult,
    ExperimentSnapshot,
    FeedbackValue,
)
from support_graph.runtime.observability import build_langsmith_client


class SdkLangSmithGateway:
    """Translate LangSmith's public SDK types into evaluation contracts."""

    def __init__(self, client: Client) -> None:
        self._client = client

    async def ping(self) -> None:
        await asyncio.to_thread(self._consume_dataset_probe)

    def _consume_dataset_probe(self) -> None:
        next(iter(self._client.list_datasets(limit=1)), None)

    async def get_dataset(self, name: str) -> DatasetRef | None:
        try:
            dataset = await asyncio.to_thread(
                self._client.read_dataset,
                dataset_name=name,
            )
        except LangSmithNotFoundError:
            return None
        return _dataset_ref(dataset)

    async def create_dataset(
        self,
        *,
        name: str,
        description: str,
        metadata: dict[str, Any],
        examples: list[dict[str, Any]],
    ) -> DatasetRef:
        dataset = await asyncio.to_thread(
            self._client.create_dataset,
            name,
            description=description,
            metadata=metadata,
        )
        dataset_ref = _dataset_ref(dataset)
        await self.upsert_dataset_examples(dataset_ref, examples)
        return dataset_ref

    async def upsert_dataset_examples(
        self,
        dataset: DatasetRef,
        examples: list[dict[str, Any]],
    ) -> None:
        """Idempotently reconcile deterministic example IDs through the public SDK."""
        await asyncio.to_thread(
            self._client.create_examples,
            dataset_id=dataset.id,
            examples=examples,
        )

    async def evaluate(
        self,
        *,
        target: Any,
        dataset_name: str,
        evaluators: list[Any],
        metadata: dict[str, Any],
        experiment_prefix: str,
        max_concurrency: int,
    ) -> ExperimentSnapshot:
        dataset = await asyncio.to_thread(
            self._client.read_dataset,
            dataset_name=dataset_name,
        )
        evaluation = await aevaluate(
            target,
            data=dataset_name,
            evaluators=evaluators,
            metadata=metadata,
            experiment_prefix=experiment_prefix,
            max_concurrency=max_concurrency,
            client=self._client,
            error_handling="log",
            upload_results=True,
        )
        results = tuple([_example_result(row) async for row in evaluation])
        project = await asyncio.to_thread(
            self._client.read_project,
            project_name=evaluation.experiment_name,
        )
        return _experiment_snapshot(project, _dataset_ref(dataset), results)

    async def read_experiment(self, experiment_id: str) -> ExperimentSnapshot:
        project = await asyncio.to_thread(
            self._client.read_project,
            project_id=experiment_id,
        )
        dataset = await asyncio.to_thread(
            self._client.read_dataset,
            dataset_id=project.reference_dataset_id,
        )
        runs = await asyncio.to_thread(
            lambda: tuple(self._client.list_runs(project_id=project.id))
        )
        results_list: list[ExampleResult] = []
        for run in runs:
            if getattr(run, "reference_example_id", None) is not None:
                results_list.append(await self._result_from_run(run))
        results = tuple(results_list)
        return _experiment_snapshot(project, _dataset_ref(dataset), results)

    async def list_experiments(self, dataset_id: str) -> tuple[ExperimentSnapshot, ...]:
        projects = await asyncio.to_thread(
            lambda: tuple(self._client.list_projects(reference_dataset_id=dataset_id))
        )
        experiments: list[ExperimentSnapshot] = []
        for project in projects:
            experiments.append(await self.read_experiment(str(project.id)))
        return tuple(experiments)

    async def update_experiment_metadata(
        self, experiment_id: str, metadata: dict[str, Any]
    ) -> ExperimentSnapshot:
        await asyncio.to_thread(
            self._client.update_project,
            experiment_id,
            metadata=metadata,
        )
        return await self.read_experiment(experiment_id)

    async def list_trace_records(
        self, experiment_id: str
    ) -> tuple[dict[str, Any], ...]:
        runs = await asyncio.to_thread(
            lambda: tuple(self._client.list_runs(project_id=experiment_id))
        )
        return tuple(_trace_record(run) for run in runs)

    async def _result_from_run(self, run: Any) -> ExampleResult:
        example = await asyncio.to_thread(
            self._client.read_example,
            run.reference_example_id,
        )
        feedback = await asyncio.to_thread(
            lambda: tuple(self._client.list_feedback(run_ids=[run.id]))
        )
        return ExampleResult(
            example_id=_example_id(example),
            run_id=str(run.id),
            inputs=_dict_value(example, "inputs"),
            reference_outputs=_dict_value(example, "outputs"),
            outputs=_dict_value(run, "outputs"),
            feedback=tuple(_feedback_value(item) for item in feedback),
            error=_optional_text(_value(run, "error")),
        )


def build_langsmith_gateway(config: RuntimeConfig) -> SdkLangSmithGateway:
    config.validate_for_hosted_eval()
    return SdkLangSmithGateway(build_langsmith_client(config))


def _dataset_ref(dataset: Any) -> DatasetRef:
    metadata = _metadata(dataset)
    sha256 = metadata.get("dataset_sha256")
    if not isinstance(sha256, str) or not sha256:
        raise ValueError(f"LangSmith dataset {dataset.name} is missing dataset_sha256")
    return DatasetRef(
        id=str(dataset.id),
        name=str(dataset.name),
        sha256=sha256,
        url=_optional_text(_value(dataset, "url")),
    )


def _experiment_snapshot(
    project: Any,
    dataset: DatasetRef,
    results: tuple[ExampleResult, ...],
) -> ExperimentSnapshot:
    return ExperimentSnapshot(
        id=str(project.id),
        name=str(project.name),
        dataset=dataset,
        metadata=_metadata(project),
        results=results,
        url=_optional_text(_value(project, "url")),
    )


def _example_result(row: Mapping[str, Any]) -> ExampleResult:
    example = row["example"]
    example_id = _example_id(example)
    if example_id is None:
        raise ValueError("LangSmith result is missing example_id")
    run = row["run"]
    evaluation_results = row.get("evaluation_results", {})
    feedback_results = _value(evaluation_results, "results", [])
    return ExampleResult(
        example_id=str(example_id),
        run_id=str(_value(run, "id")),
        inputs=_dict_value(example, "inputs"),
        reference_outputs=_dict_value(example, "outputs"),
        outputs=_dict_value(run, "outputs"),
        feedback=tuple(_feedback_value(item) for item in feedback_results),
        error=_optional_text(_value(run, "error")),
    )


def _feedback_value(feedback: Any) -> FeedbackValue:
    score = _value(feedback, "score")
    return FeedbackValue(
        key=str(_value(feedback, "key")),
        score=float(score) if isinstance(score, int | float) else None,
        value=_optional_text(_value(feedback, "value")),
        error=_optional_text(_value(feedback, "error")),
    )


def _example_id(example: Any) -> str:
    metadata = _value(example, "metadata", {})
    if isinstance(metadata, Mapping):
        source_id = metadata.get("example_id")
        if source_id:
            return str(source_id)
    identifier = _value(example, "id")
    if identifier is None:
        raise ValueError("LangSmith example is missing example_id")
    return str(identifier)


def _trace_record(run: Any) -> dict[str, Any]:
    return {
        key: _serializable(_value(run, key))
        for key in (
            "id",
            "trace_id",
            "parent_run_id",
            "name",
            "run_type",
            "start_time",
            "end_time",
            "inputs",
            "outputs",
            "error",
        )
        if _value(run, key) is not None
    }


def _metadata(value: Any) -> dict[str, Any]:
    direct_metadata = _value(value, "metadata")
    if isinstance(direct_metadata, Mapping):
        return dict(direct_metadata)
    extra = _value(value, "extra", {})
    metadata = _value(extra, "metadata", {})
    if not isinstance(metadata, Mapping):
        return {}
    return dict(metadata)


def _dict_value(value: Any, key: str) -> dict[str, Any]:
    item = _value(value, key, {})
    return dict(item) if isinstance(item, Mapping) else {}


def _value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _optional_text(value: Any) -> str | None:
    return str(value) if value is not None else None


def _serializable(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        return [_serializable(item) for item in value]
    return str(value) if hasattr(value, "hex") else value


__all__ = ["SdkLangSmithGateway", "build_langsmith_gateway"]
