"""Generated artifact path helpers for the on-disk artifact contract."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path


_PATH_SAFE_PATTERN = re.compile(r"[^a-z0-9]+")
EVAL_RUN_REQUIRED_FILES = (
    "manifest.json",
    "metrics.json",
    "predictions.jsonl",
    "failures.jsonl",
    "manual_review.csv",
    "retrieval_examples.jsonl",
    "trace_index.json",
    "summary.md",
)
EVAL_REPORT_REQUIRED_FILES = (
    "manifest.json",
    "report.md",
)
STANDALONE_RUN_REQUIRED_FILES = (
    "manifest.json",
    "result.json",
    "trace.jsonl",
)


@dataclass(frozen=True, slots=True)
class EvalRunArtifacts:
    run_id: str
    output_dir: Path
    manifest: Path
    metrics: Path
    predictions: Path
    failures: Path
    manual_review: Path
    retrieval_examples: Path
    trace_index: Path
    summary: Path
    traces_dir: Path

    def trace_path(self, trace_file: str) -> Path:
        return self.traces_dir / trace_file


@dataclass(frozen=True, slots=True)
class EvalReportArtifacts:
    report_id: str
    output_dir: Path
    manifest: Path
    report: Path


@dataclass(frozen=True, slots=True)
class StandaloneRunArtifacts:
    run_id: str
    output_dir: Path
    manifest: Path
    result: Path
    trace: Path


def eval_run_artifacts(project_root: Path, run_id: str) -> EvalRunArtifacts:
    output_dir = project_root / "outputs/evals/runs" / run_id
    return EvalRunArtifacts(
        run_id=run_id,
        output_dir=output_dir,
        manifest=output_dir / "manifest.json",
        metrics=output_dir / "metrics.json",
        predictions=output_dir / "predictions.jsonl",
        failures=output_dir / "failures.jsonl",
        manual_review=output_dir / "manual_review.csv",
        retrieval_examples=output_dir / "retrieval_examples.jsonl",
        trace_index=output_dir / "trace_index.json",
        summary=output_dir / "summary.md",
        traces_dir=output_dir / "traces",
    )


def eval_report_artifacts(project_root: Path, report_id: str) -> EvalReportArtifacts:
    output_dir = project_root / "outputs/evals/reports" / report_id
    return EvalReportArtifacts(
        report_id=report_id,
        output_dir=output_dir,
        manifest=output_dir / "manifest.json",
        report=output_dir / "report.md",
    )


def standalone_run_artifacts(
    project_root: Path,
    run_id: str,
) -> StandaloneRunArtifacts:
    output_dir = project_root / "outputs/runs" / run_id
    return StandaloneRunArtifacts(
        run_id=run_id,
        output_dir=output_dir,
        manifest=output_dir / "manifest.json",
        result=output_dir / "result.json",
        trace=output_dir / "trace.jsonl",
    )


def build_trace_file(example_id: str) -> str:
    cleaned = example_id.strip().lower()
    if not cleaned:
        raise ValueError("example_id must not be empty.")
    slug = _PATH_SAFE_PATTERN.sub("-", cleaned).strip("-")
    if not slug:
        raise ValueError(
            f"example_id must contain path-safe characters: {example_id!r}"
        )
    return f"{slug}.jsonl"


def build_standalone_run_id() -> str:
    return f"run-{uuid.uuid4().hex[:12]}"


def project_relative_path(path: str | Path, project_root: Path) -> str:
    resolved = Path(path)
    try:
        return str(resolved.relative_to(project_root))
    except ValueError:
        return str(resolved)


def resolve_project_path(project_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root / candidate
