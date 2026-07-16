from __future__ import annotations

import asyncio
import hashlib
import zipfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from support_graph.evaluation.baselines import (
    export_baseline,
    find_promoted_baseline,
    promote_baseline,
    verify_baseline_export,
)
from support_graph.evaluation.contracts import (
    DatasetRef,
    ExampleResult,
    ExperimentSnapshot,
    FeedbackValue,
)
from support_graph.evaluation.policy import MetricPolicy, RegressionPolicy


def test_find_promoted_baseline_selects_latest_eligible_marker() -> None:
    gateway = FakeGateway(
        experiments=(
            promoted_snapshot("old", promoted_at="2026-07-15T10:00:00+00:00"),
            promoted_snapshot("new", promoted_at="2026-07-16T10:00:00+00:00"),
            snapshot("unpromoted"),
        )
    )

    result = asyncio.run(
        find_promoted_baseline(gateway, gateway.experiments[0].dataset)
    )

    assert result is not None
    assert result.id == "new"


@pytest.mark.parametrize(
    "snapshot_factory",
    [
        lambda: promoted_snapshot("dirty", git_dirty=True),
        lambda: promoted_snapshot("running", status="running"),
        lambda: promoted_snapshot("wrong-hash", dataset_sha256="other"),
        lambda: replace(
            promoted_snapshot("malformed"),
            metadata={"support_graph_baseline": {"promoted_at": "not-a-date"}},
        ),
    ],
)
def test_find_promoted_baseline_rejects_ineligible_markers(
    snapshot_factory: Any,
) -> None:
    rejected = snapshot_factory()
    gateway = FakeGateway(experiments=(rejected,))

    result = asyncio.run(
        find_promoted_baseline(gateway, DatasetRef("dataset", "dataset", "hash"))
    )

    assert result is None


def test_find_promoted_baseline_rejects_ambiguous_latest_marker() -> None:
    gateway = FakeGateway(
        experiments=(
            promoted_snapshot("left", promoted_at="2026-07-16T10:00:00+00:00"),
            promoted_snapshot("right", promoted_at="2026-07-16T10:00:00+00:00"),
        )
    )

    with pytest.raises(ValueError, match="Ambiguous promoted baselines"):
        asyncio.run(find_promoted_baseline(gateway, gateway.experiments[0].dataset))


def test_export_is_byte_identical_and_verifiable(tmp_path: Path) -> None:
    gateway = FakeGateway(experiments=(snapshot("exp-1"),))

    first = asyncio.run(
        export_baseline(gateway, "exp-1", output_dir=tmp_path / "first")
    )
    second = asyncio.run(
        export_baseline(gateway, "exp-1", output_dir=tmp_path / "second")
    )

    assert first.read_bytes() == second.read_bytes()
    manifest = verify_baseline_export(first)
    assert manifest["experiment_id"] == "exp-1"


def test_export_has_exact_members_and_redacts_sensitive_fields(tmp_path: Path) -> None:
    sensitive = snapshot("exp-1")
    sensitive = replace(
        sensitive,
        results=(
            replace(
                sensitive.results[0],
                inputs={"authorization": "Bearer secret", "visible": "ok"},
            ),
        ),
    )
    gateway = FakeGateway(
        experiments=(sensitive,),
        traces=({"headers": {"authorization": "Bearer secret"}, "visible": "ok"},),
    )

    path = asyncio.run(export_baseline(gateway, "exp-1", output_dir=tmp_path))
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == [
            "manifest.json",
            "dataset.jsonl",
            "results.jsonl",
            "feedback.jsonl",
            "traces.jsonl",
            "comparison.json",
            "summary.md",
            "checksums.sha256",
        ]
        dataset = archive.read("dataset.jsonl").decode("utf-8")
        traces = archive.read("traces.jsonl").decode("utf-8")
    assert "authorization" not in dataset
    assert "secret" not in traces


def test_export_rejects_empty_storage_key(tmp_path: Path) -> None:
    gateway = FakeGateway(experiments=(snapshot("@@@"),))

    with pytest.raises(ValueError, match="storage key"):
        asyncio.run(export_baseline(gateway, "@@@", output_dir=tmp_path))


def test_verify_baseline_export_rejects_checksum_mismatch(tmp_path: Path) -> None:
    gateway = FakeGateway(experiments=(snapshot("exp-1"),))
    path = asyncio.run(export_baseline(gateway, "exp-1", output_dir=tmp_path))
    corrupted = tmp_path / "corrupted.zip"
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(corrupted, "w") as target:
        for info in source.infolist():
            content = source.read(info.filename)
            target.writestr(
                info, b"corrupted" if info.filename == "summary.md" else content
            )

    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_baseline_export(corrupted)


def test_promote_baseline_is_idempotent(tmp_path: Path) -> None:
    gateway = FakeGateway(experiments=(snapshot("exp-1"),))
    promoted_at = datetime(2026, 7, 16, 10, tzinfo=timezone.utc)

    first = asyncio.run(
        promote_baseline(
            gateway,
            "exp-1",
            reason="approved smoke run",
            policy=policy(),
            output_dir=tmp_path,
            promoted_at=promoted_at,
        )
    )
    second = asyncio.run(
        promote_baseline(
            gateway,
            "exp-1",
            reason="approved smoke run",
            policy=policy(),
            output_dir=tmp_path,
            promoted_at=promoted_at,
        )
    )

    assert (
        first.metadata["support_graph_baseline"]
        == second.metadata["support_graph_baseline"]
    )
    assert gateway.update_calls == 1
    assert gateway.read_calls >= 2


class FakeGateway:
    def __init__(
        self,
        *,
        experiments: tuple[ExperimentSnapshot, ...],
        traces: tuple[dict[str, Any], ...] = (),
    ) -> None:
        self.experiments = experiments
        self.traces = traces
        self.update_calls = 0
        self.read_calls = 0

    async def ping(self) -> None:
        return None

    async def get_dataset(self, name: str) -> DatasetRef | None:
        return next(
            (item.dataset for item in self.experiments if item.dataset.name == name),
            None,
        )

    async def create_dataset(self, **kwargs: Any) -> DatasetRef:
        return DatasetRef(
            id="created-dataset",
            name=kwargs["name"],
            sha256=kwargs["metadata"]["dataset_sha256"],
        )

    async def evaluate(self, **kwargs: Any) -> ExperimentSnapshot:
        raise AssertionError("evaluate is not used by baseline tests")

    async def read_experiment(self, experiment_id: str) -> ExperimentSnapshot:
        self.read_calls += 1
        return next(item for item in self.experiments if item.id == experiment_id)

    async def list_experiments(self, dataset_id: str) -> tuple[ExperimentSnapshot, ...]:
        return tuple(item for item in self.experiments if item.dataset.id == dataset_id)

    async def list_trace_records(
        self, experiment_id: str
    ) -> tuple[dict[str, Any], ...]:
        assert any(item.id == experiment_id for item in self.experiments)
        return self.traces

    async def update_experiment_metadata(
        self,
        experiment_id: str,
        metadata: dict[str, Any],
    ) -> ExperimentSnapshot:
        self.update_calls += 1
        current = await self.read_experiment(experiment_id)
        updated = replace(current, metadata=metadata)
        self.experiments = tuple(
            updated if item.id == experiment_id else item for item in self.experiments
        )
        return updated


def policy() -> RegressionPolicy:
    return RegressionPolicy(
        schema_version=1,
        metric_schema_version="1",
        required_feedback=("metric", "decision", "failure_label"),
        metrics=(
            MetricPolicy(
                key="metric",
                direction="higher",
                required=True,
                max_absolute_regression=0.0,
                max_relative_regression=None,
                protect_examples=True,
            ),
        ),
    )


def snapshot(
    experiment_id: str,
    *,
    status: str = "completed",
    git_dirty: bool = False,
    dataset_sha256: str = "hash",
) -> ExperimentSnapshot:
    dataset = DatasetRef("dataset", "dataset", dataset_sha256)
    return ExperimentSnapshot(
        id=experiment_id,
        name=experiment_id,
        dataset=dataset,
        metadata={
            "status": status,
            "git_dirty": git_dirty,
            "metric_schema_version": "1",
            "evaluator_versions": {"deterministic": "1"},
            "judge_provider": None,
            "judge_model": None,
            "domain": "kubernetes",
            "subset": "smoke",
        },
        results=(
            ExampleResult(
                example_id="k8s-001",
                run_id="run-1",
                inputs={"visible": "ok"},
                reference_outputs={"answer": "A Pod"},
                outputs={"answer": "A Pod"},
                feedback=(
                    FeedbackValue(key="metric", score=1.0),
                    FeedbackValue(key="decision", value="answer"),
                    FeedbackValue(key="failure_label", value="none"),
                ),
            ),
        ),
    )


def promoted_snapshot(
    experiment_id: str,
    *,
    promoted_at: str = "2026-07-16T10:00:00+00:00",
    status: str = "completed",
    git_dirty: bool = False,
    dataset_sha256: str = "hash",
) -> ExperimentSnapshot:
    result = snapshot(
        experiment_id,
        status=status,
        git_dirty=git_dirty,
        dataset_sha256=dataset_sha256,
    )
    marker = {
        "schema_version": "1",
        "experiment_id": experiment_id,
        "dataset_name": result.dataset.name,
        "dataset_sha256": result.dataset.sha256,
        "promoted_at": promoted_at,
        "reason": "approved",
        "export_sha256": hashlib.sha256(experiment_id.encode()).hexdigest(),
        "retention": {
            "automation": "unsupported_public_sdk",
            "remote_expiry_risk": True,
        },
    }
    return replace(
        result, metadata={**result.metadata, "support_graph_baseline": marker}
    )
