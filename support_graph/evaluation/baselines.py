"""Explicit LangSmith baseline promotion and deterministic evidence exports."""

from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from support_graph.evaluation.contracts import (
    DatasetRef,
    ExperimentSnapshot,
    GateResult,
    LangSmithGateway,
)
from support_graph.evaluation.policy import RegressionPolicy

_MARKER_KEY = "support_graph_baseline"
_MEMBERS = (
    "manifest.json",
    "dataset.jsonl",
    "results.jsonl",
    "feedback.jsonl",
    "traces.jsonl",
    "comparison.json",
    "summary.md",
    "checksums.sha256",
)


async def find_promoted_baseline(
    gateway: LangSmithGateway,
    dataset: DatasetRef,
) -> ExperimentSnapshot | None:
    """Find the latest eligible promoted baseline for an immutable dataset."""
    eligible: list[tuple[datetime, ExperimentSnapshot]] = []
    for experiment in await gateway.list_experiments(dataset.id):
        promoted_at = _eligible_promotion_time(experiment, dataset)
        if promoted_at is not None:
            eligible.append((promoted_at, experiment))
    if not eligible:
        return None

    latest = max(item[0] for item in eligible)
    latest_experiments = [item[1] for item in eligible if item[0] == latest]
    if len({experiment.id for experiment in latest_experiments}) != 1:
        raise ValueError("Ambiguous promoted baselines share the latest timestamp")
    return latest_experiments[0]


async def export_baseline(
    gateway: LangSmithGateway,
    experiment_id: str,
    *,
    output_dir: Path,
    comparison: GateResult | None = None,
) -> Path:
    """Export a deterministic, redacted evidence bundle for one experiment."""
    experiment = await gateway.read_experiment(experiment_id)
    traces = await gateway.list_trace_records(experiment_id)
    output_path = _export_path(experiment, output_dir)
    members = _export_members(experiment, traces, comparison)
    _write_export(output_path, members)
    verify_baseline_export(output_path)
    return output_path


def verify_baseline_export(path: Path) -> dict[str, Any]:
    """Verify a baseline ZIP's fixed members and content checksums."""
    with zipfile.ZipFile(path) as archive:
        if tuple(archive.namelist()) != _MEMBERS:
            raise ValueError("Baseline export has unexpected members")
        checksums = _parse_checksums(archive.read("checksums.sha256"))
        for name in _MEMBERS[:-1]:
            actual = hashlib.sha256(archive.read(name)).hexdigest()
            if checksums.get(name) != actual:
                raise ValueError(f"Checksum mismatch for {name}")
        manifest = json.loads(archive.read("manifest.json"))
    if not isinstance(manifest, dict):
        raise ValueError("Baseline export manifest must be an object")
    return manifest


async def promote_baseline(
    gateway: LangSmithGateway,
    experiment_id: str,
    *,
    reason: str,
    policy: RegressionPolicy,
    output_dir: Path,
    promoted_at: datetime | None = None,
) -> ExperimentSnapshot:
    """Promote a completed clean experiment through public project metadata."""
    experiment = await gateway.read_experiment(experiment_id)
    _validate_promotion(experiment, policy)
    output_path = _export_path(experiment, output_dir)
    existing = experiment.metadata.get(_MARKER_KEY)
    if isinstance(existing, Mapping) and _marker_matches_experiment(
        existing, experiment
    ):
        if output_path.is_file():
            verify_baseline_export(output_path)
            if _file_sha256(output_path) == existing.get("export_sha256"):
                return experiment

    exported = await export_baseline(gateway, experiment_id, output_dir=output_dir)
    export_sha256 = _file_sha256(exported)
    timestamp = (promoted_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    marker = {
        "schema_version": "1",
        "experiment_id": experiment.id,
        "dataset_name": experiment.dataset.name,
        "dataset_sha256": experiment.dataset.sha256,
        "promoted_at": timestamp.isoformat(),
        "reason": reason,
        "export_sha256": export_sha256,
        "retention": {
            "automation": "unsupported_public_sdk",
            "remote_expiry_risk": True,
        },
    }
    metadata = {**experiment.metadata, _MARKER_KEY: marker}
    await gateway.update_experiment_metadata(experiment.id, metadata)
    updated = await gateway.read_experiment(experiment.id)
    if updated.metadata.get(_MARKER_KEY) != marker:
        raise ValueError("LangSmith baseline marker did not persist exactly")
    return updated


def _eligible_promotion_time(
    experiment: ExperimentSnapshot,
    dataset: DatasetRef,
) -> datetime | None:
    marker = experiment.metadata.get(_MARKER_KEY)
    if not isinstance(marker, Mapping):
        return None
    if not _marker_matches_experiment(marker, experiment):
        return None
    if (
        experiment.dataset.name != dataset.name
        or experiment.dataset.sha256 != dataset.sha256
    ):
        return None
    if experiment.metadata.get("status") != "completed":
        return None
    if experiment.metadata.get("git_dirty") is not False:
        return None
    value = marker.get("promoted_at")
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _marker_matches_experiment(
    marker: Mapping[str, Any],
    experiment: ExperimentSnapshot,
) -> bool:
    return (
        marker.get("schema_version") == "1"
        and marker.get("experiment_id") == experiment.id
        and marker.get("dataset_name") == experiment.dataset.name
        and marker.get("dataset_sha256") == experiment.dataset.sha256
        and isinstance(marker.get("reason"), str)
        and isinstance(marker.get("export_sha256"), str)
        and marker.get("retention")
        == {
            "automation": "unsupported_public_sdk",
            "remote_expiry_risk": True,
        }
    )


def _validate_promotion(
    experiment: ExperimentSnapshot,
    policy: RegressionPolicy,
) -> None:
    if experiment.metadata.get("status") != "completed":
        raise ValueError("Baseline promotion requires a completed experiment")
    if experiment.metadata.get("git_dirty") is not False:
        raise ValueError("Baseline promotion requires git_dirty=false")
    if experiment.metadata.get("metric_schema_version") != policy.metric_schema_version:
        raise ValueError("Baseline metric schema does not match the policy")
    for result in experiment.results:
        if result.error:
            raise ValueError(f"Baseline has execution error for {result.example_id}")
        feedback = {item.key: item for item in result.feedback}
        for key in policy.required_feedback:
            item = feedback.get(key)
            if item is None or item.error:
                raise ValueError(f"Baseline missing required feedback: {key}")


def _export_path(experiment: ExperimentSnapshot, output_dir: Path) -> Path:
    storage_key = re.sub(r"[^a-z0-9]+", "-", experiment.id.lower()).strip("-")
    if not storage_key:
        raise ValueError("Baseline export requires a non-empty storage key")
    domain = str(experiment.metadata.get("domain", "dataset"))
    subset = str(experiment.metadata.get("subset", "baseline"))
    return output_dir / f"{domain}-{subset}-{storage_key}.zip"


def _export_members(
    experiment: ExperimentSnapshot,
    traces: tuple[dict[str, Any], ...],
    comparison: GateResult | None,
) -> dict[str, bytes]:
    results = sorted(experiment.results, key=lambda result: result.example_id)
    manifest = {
        "schema_version": "1",
        "experiment_id": experiment.id,
        "experiment_name": experiment.name,
        "dataset": asdict(experiment.dataset),
        "metadata": _redact(experiment.metadata),
    }
    dataset_records = [
        {
            "example_id": result.example_id,
            "inputs": _redact(result.inputs),
            "outputs": _redact(result.reference_outputs),
        }
        for result in results
    ]
    result_records = [
        {
            "example_id": result.example_id,
            "run_id": result.run_id,
            "outputs": _redact(result.outputs),
            "error": result.error,
        }
        for result in results
    ]
    feedback_records = [
        {
            "example_id": result.example_id,
            "run_id": result.run_id,
            "feedback": [_redact(asdict(item)) for item in result.feedback],
        }
        for result in results
    ]
    trace_records = sorted((_redact(record) for record in traces), key=_json_bytes)
    comparison_payload = asdict(comparison) if comparison is not None else None
    members = {
        "manifest.json": _json_bytes(manifest),
        "dataset.jsonl": _jsonl_bytes(dataset_records),
        "results.jsonl": _jsonl_bytes(result_records),
        "feedback.jsonl": _jsonl_bytes(feedback_records),
        "traces.jsonl": _jsonl_bytes(trace_records),
        "comparison.json": _json_bytes(comparison_payload),
        "summary.md": _summary_bytes(experiment, comparison),
    }
    checksums = "".join(
        f"{hashlib.sha256(members[name]).hexdigest()}  {name}\n"
        for name in _MEMBERS[:-1]
    )
    members["checksums.sha256"] = checksums.encode("utf-8")
    return members


def _write_export(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for name in _MEMBERS:
                info = zipfile.ZipInfo(filename=name)
                info.date_time = (1980, 1, 1, 0, 0, 0)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, members[name])
        verify_baseline_export(temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(records: list[Any]) -> bytes:
    return b"".join(_json_bytes(record) for record in records)


def _summary_bytes(
    experiment: ExperimentSnapshot, comparison: GateResult | None
) -> bytes:
    status = comparison.status if comparison is not None else "not_compared"
    return (
        f"# SupportGraph baseline\n\nExperiment: {experiment.id}\nStatus: {status}\n"
    ).encode("utf-8")


def _parse_checksums(value: bytes) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in value.decode("utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid checksums.sha256")
        checksums[name] = digest
    if set(checksums) != set(_MEMBERS[:-1]):
        raise ValueError("checksums.sha256 has unexpected members")
    return checksums


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _redact(item)
            for key, item in value.items()
            if not _sensitive_key(str(key))
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    return value


def _sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return any(
        marker in normalized
        for marker in ("apikey", "authorization", "cookie", "headers", "secret")
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "export_baseline",
    "find_promoted_baseline",
    "promote_baseline",
    "verify_baseline_export",
]
