"""Helpers for the Workbench live-run surface."""

from __future__ import annotations

import json
import shutil
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import urlencode

from support_graph.artifacts import (
    build_standalone_run_id,
    project_relative_path,
    standalone_run_artifacts,
)
from support_graph.config.runtime import RuntimeSettingsLike, build_runtime_config
from support_graph.data.examples import (
    load_example_record as load_example_record_from_paths,
)
from support_graph.runtime.graph import astream_graph_events
from support_graph.runtime.schemas import GraphStreamEvent
from support_graph.types import DomainLike


class LiveRunSettings(RuntimeSettingsLike, Protocol):
    project_root: Path
    dataset_root: Path
    examples_dir: Path
    provider_type: object
    embedding_provider_type: object | None
    chat_model: str | None
    embedding_model: str | None
    prompt_version: str
    max_retrieval_attempts: int

    def runtime_missing_fields(self) -> list[str]: ...

    def selected_domain(
        self, explicit_domain: DomainLike | None = None
    ) -> DomainLike: ...


@dataclass(frozen=True, slots=True)
class LiveRunSession:
    run_id: str
    example: dict[str, object]
    stream_url: str
    run_path: str
    trace_path: str


@dataclass(frozen=True, slots=True)
class LiveRunConfigurationError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class LiveRunExampleNotFoundError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(slots=True)
class LiveRunAccumulator:
    example: dict[str, object]
    retrieval_ranked_chunks: list[dict[str, object]] = field(default_factory=list)
    retrieved_chunks: list[dict[str, object]] = field(default_factory=list)
    evidence_grade: dict[str, object] = field(default_factory=dict)
    completed_event: GraphStreamEvent | None = None

    def apply(self, event: GraphStreamEvent) -> None:
        kind = event.get("kind")
        if kind == "retrieval_complete":
            self.retrieval_ranked_chunks = [
                dict(chunk) for chunk in event.get("retrieval_ranked_chunks", [])
            ]
            self.retrieved_chunks = [
                dict(chunk) for chunk in event.get("retrieved_chunks", [])
            ]
            return
        if kind == "evidence_graded":
            self.evidence_grade = dict(event.get("evidence_grade") or {})
            return
        if kind == "response_completed":
            self.completed_event = dict(event)

    def build_result(self, *, trace_path: str) -> dict[str, object] | None:
        if self.completed_event is None:
            return None
        trace_summary = dict(self.completed_event.get("trace_summary") or {})
        trace_summary["trace_path"] = trace_path
        return {
            "example_id": self.example.get("example_id"),
            "latest_user_utterance": self.example.get("latest_user_utterance"),
            "decision": self.completed_event.get("decision"),
            "response_text": self.completed_event.get("response_text", ""),
            "citations": list(self.completed_event.get("citations") or []),
            "confidence_label": self.completed_event.get("confidence_label", "low"),
            "retrieval_ranked_chunks": self.retrieval_ranked_chunks,
            "retrieved_chunks": self.retrieved_chunks,
            "evidence_grade": self.evidence_grade,
            "trace_summary": trace_summary,
        }


def prepare_live_run_session(
    settings: LiveRunSettings,
    *,
    example_id: str,
) -> LiveRunSession:
    example = _require_live_example(settings, example_id)
    run_id = build_standalone_run_id()
    artifacts = standalone_run_artifacts(settings.project_root, run_id)
    query = urlencode({"example_id": example_id, "run_id": run_id})
    return LiveRunSession(
        run_id=run_id,
        example=example,
        stream_url=f"/live/stream?{query}",
        run_path=f"/runs/{run_id}",
        trace_path=project_relative_path(artifacts.trace, settings.project_root),
    )


def load_example_record(
    settings: LiveRunSettings,
    example_id: str,
) -> dict[str, object]:
    existing_paths = sorted(settings.examples_dir.glob("*.jsonl"))
    if not existing_paths:
        raise LiveRunExampleNotFoundError(
            "Derived examples not found under "
            f"{project_relative_path(settings.examples_dir, settings.project_root)}. "
            "Run `uv run support-graph build-examples --domain dmv --split validation` first."
        )
    try:
        return load_example_record_from_paths(example_id, existing_paths)
    except FileNotFoundError as exc:
        raise LiveRunExampleNotFoundError(
            f"Example not found in derived examples: {example_id}."
        ) from exc


async def iter_live_run_stream(
    settings: LiveRunSettings,
    *,
    example_id: str,
    run_id: str,
) -> AsyncIterator[bytes]:
    artifacts = standalone_run_artifacts(settings.project_root, run_id)
    persisted = False

    try:
        example = _require_live_example(settings, example_id)
        domain = example.get("domain") or settings.selected_domain()
        config = build_runtime_config(settings, domain)
        trace_path = project_relative_path(artifacts.trace, settings.project_root)
        accumulator = LiveRunAccumulator(example=example)
        async for event in astream_graph_events(
            example=example,
            config=config,
            run_id=run_id,
            trace_path=artifacts.trace,
            max_attempts=settings.max_retrieval_attempts,
        ):
            accumulator.apply(event)
            yield _encode_sse(event)

        result_payload = accumulator.build_result(trace_path=trace_path)
        if result_payload is None:
            yield _encode_sse(
                {
                    "kind": "error",
                    "node": "live_stream",
                    "example_id": example_id,
                    "error": "Live run ended before a completed response arrived.",
                    "exception_type": "IncompleteLiveRun",
                }
            )
            return

        artifacts.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            artifacts.manifest,
            _standalone_run_manifest(
                settings,
                run_id=run_id,
                example=example,
                prompt_version=config.prompt_version,
            ),
        )
        _write_json(artifacts.result, result_payload)
        persisted = True
    except LiveRunConfigurationError as exc:
        yield _encode_sse(
            {
                "kind": "error",
                "node": "live_stream",
                "example_id": example_id,
                "error": str(exc),
                "exception_type": type(exc).__name__,
            }
        )
    except LiveRunExampleNotFoundError as exc:
        yield _encode_sse(
            {
                "kind": "error",
                "node": "live_stream",
                "example_id": example_id,
                "error": str(exc),
                "exception_type": type(exc).__name__,
            }
        )
    finally:
        if not persisted:
            _cleanup_incomplete_run(artifacts.output_dir)


def _require_live_example(
    settings: LiveRunSettings,
    example_id: str,
) -> dict[str, object]:
    missing = settings.runtime_missing_fields()
    if missing:
        raise LiveRunConfigurationError(
            "Missing runtime config: " + ", ".join(sorted(missing))
        )
    return load_example_record(settings, example_id)


def _standalone_run_manifest(
    settings: LiveRunSettings,
    *,
    run_id: str,
    example: dict[str, object],
    prompt_version: str,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "example_id": example.get("example_id"),
        "domain": str(example.get("domain") or settings.selected_domain()),
        "provider": {
            "type": settings.provider_type,
            "chat_model": settings.chat_model,
            "embedding_type": settings.embedding_provider_type
            or settings.provider_type,
            "embedding_model": settings.embedding_model,
        },
        "prompt_version": prompt_version,
    }


def _encode_sse(event: GraphStreamEvent | dict[str, object]) -> bytes:
    payload = json.dumps(event, ensure_ascii=True)
    return f"data: {payload}\n\n".encode("utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _cleanup_incomplete_run(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)


__all__ = [
    "LiveRunConfigurationError",
    "LiveRunExampleNotFoundError",
    "LiveRunSession",
    "LiveRunSettings",
    "iter_live_run_stream",
    "prepare_live_run_session",
]
