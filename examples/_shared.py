"""Shared helpers for the numbered walkthrough scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from support_graph.config.runtime import RuntimeConfig
from support_graph.config.settings import Settings
from support_graph.data.chunks import build_chunks, write_chunks_jsonl
from support_graph.data.dataset import load_dialogues, load_documents
from support_graph.data.examples import (
    build_turn_examples,
    load_examples_jsonl,
    write_examples_jsonl,
)
from support_graph.types import DatasetSplit, DatasetSplitLike, Domain, DomainLike


EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent
SAMPLE_DOMAIN = Domain.DMV
SAMPLE_DOC_ID = "Registrations#3_0"
SAMPLE_DIALOGUE_ID = "1409501a35697e0ce68561e29577b90a"
SAMPLE_EXAMPLE_ID = f"{SAMPLE_DOMAIN}::{SAMPLE_DIALOGUE_ID}::turn_2"
SAMPLE_TITLE_CHUNK_ID = (
    "dmv::Top 5 DMV Mistakes and How to Avoid Them#3_0::sec::t_7::sub::0"
)
SAMPLE_CONTENT_CHUNK_ID = (
    "dmv::Top 5 DMV Mistakes and How to Avoid Them#3_0::sec::7::sub::0"
)


def repo_root() -> Path:
    return REPO_ROOT


def walkthrough_sample_ids() -> dict[str, str]:
    return {
        "domain": SAMPLE_DOMAIN,
        "doc_id": SAMPLE_DOC_ID,
        "dialogue_id": SAMPLE_DIALOGUE_ID,
        "example_id": SAMPLE_EXAMPLE_ID,
        "title_chunk_id": SAMPLE_TITLE_CHUNK_ID,
        "content_chunk_id": SAMPLE_CONTENT_CHUNK_ID,
    }


def load_settings(
    config_file: str | Path | None = None,
    secrets_file: str | Path | None = None,
) -> Settings:
    return Settings.load(config_file, secrets_file)


def format_path(path: str | Path) -> str:
    resolved = Path(path)
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def json_lines(value: Any) -> list[str]:
    return json.dumps(value, ensure_ascii=True, indent=2).splitlines()


def print_lines(lines: list[str]) -> None:
    for line in lines:
        print(line)


def next_step_lines(script_name: str) -> list[str]:
    return ["", "Next", f"uv run python examples/{script_name}"]


def status_lines(
    title: str,
    *,
    state: str,
    summary: str,
    details: list[str] | None = None,
    next_commands: list[str] | None = None,
) -> list[str]:
    lines = [title, f"State: {state}", summary]
    if details:
        lines.extend(details)
    if next_commands:
        lines.extend(["Next", *next_commands])
    return lines


def missing_config_lines(
    title: str,
    missing_fields: list[str],
    *,
    next_commands: list[str] | None = None,
) -> list[str]:
    return status_lines(
        title,
        state="missing-config",
        summary="Missing config",
        details=[", ".join(missing_fields)],
        next_commands=next_commands
        or [
            "cp .env.example .env",
            "cp support_graph.toml.example support_graph.toml",
            "docker compose up -d postgres",
            "uv run grounded-support-rag index-docs --domain dmv",
        ],
    )


def unavailable_lines(
    title: str,
    *,
    summary: str,
    error: Exception | str,
    next_commands: list[str],
) -> list[str]:
    return status_lines(
        title,
        state="unavailable",
        summary=summary,
        details=[str(error)],
        next_commands=next_commands,
    )


def load_jsonl_records(path: str | Path) -> list[dict]:
    record_path = Path(path)
    if not record_path.exists():
        return []
    records: list[dict] = []
    for line in record_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def load_or_build_chunks(
    settings: Settings,
    *,
    domain: DomainLike = SAMPLE_DOMAIN,
    max_tokens_per_chunk: int = 512,
) -> list[dict]:
    artifact_path = settings.chunk_artifact_path(domain)
    if artifact_path.exists():
        return load_jsonl_records(artifact_path)
    documents = load_documents(settings.dataset.root, domains=[domain])
    chunks = build_chunks(
        documents,
        max_tokens_per_chunk=max_tokens_per_chunk,
        domains=[domain],
    )
    write_chunks_jsonl(chunks, artifact_path)
    return chunks


def load_or_build_examples(
    settings: Settings,
    *,
    domain: DomainLike = SAMPLE_DOMAIN,
    split: DatasetSplitLike = DatasetSplit.VALIDATION,
) -> list[dict]:
    artifact_path = settings.paths.examples_dir / f"{domain}_{split}.jsonl"
    if artifact_path.exists():
        return load_examples_jsonl(artifact_path)
    dialogues = load_dialogues(settings.dataset.root, split=split, domains=[domain])
    examples = build_turn_examples(dialogues)
    write_examples_jsonl(examples, artifact_path)
    return examples


def load_committed_subset(settings: Settings, subset_name: str) -> list[dict]:
    return load_jsonl_records(
        settings.paths.project_root / "data/eval_subsets" / f"{subset_name}.jsonl"
    )


def find_by_key(records: list[dict], key: str, value: Any) -> dict:
    return next(record for record in records if record.get(key) == value)


def runtime_config(settings: Settings, domain: DomainLike) -> RuntimeConfig:
    return settings.runtime_for(domain)


def latest_eval_run_dir(
    settings: Settings, *, required_files: list[str] | None = None
) -> Path | None:
    if not settings.paths.eval_runs_dir.exists():
        return None
    candidates = [
        path
        for path in settings.paths.eval_runs_dir.iterdir()
        if path.is_dir()
        and all((path / name).exists() for name in (required_files or []))
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.name, reverse=True)[0]
