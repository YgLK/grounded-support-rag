"""Shared helpers for the SupportGraph CLI."""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path

from support_graph.artifacts import EVAL_RUN_REQUIRED_FILES
from support_graph.config.settings import Settings
from support_graph.types import DatasetSplit, Domain, EvalSubset

DOMAIN_CHOICES = [domain.value for domain in Domain]
SPLIT_CHOICES = [split.value for split in DatasetSplit]
EVAL_SUBSET_CHOICES = [subset.value for subset in EvalSubset]


def print_lines(lines: list[str]) -> None:
    for line in lines:
        print(line)


def format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {remaining_minutes}m {remaining_seconds}s"
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"


def relative_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_output_dir(settings: Settings, run_id: str) -> Path:
    return settings.paths.eval_runs_dir / run_id


def eval_run_is_complete(output_dir: Path) -> bool:
    return all((output_dir / name).exists() for name in EVAL_RUN_REQUIRED_FILES)


def shorten(text: str, limit: int = 88) -> str:
    cleaned = " ".join(str(text).split()).strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: max(0, limit - 3)].rstrip()}..."


def run_async_boundary(value: object) -> object:
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


def missing_config_lines(
    *,
    title: str,
    settings: Settings,
    missing: list[str],
) -> list[str]:
    return [
        title,
        "State: missing-config",
        "Missing config",
        ", ".join(missing),
        "Next",
        (
            "Inspect "
            f"{relative_path(settings.paths.config_path, settings.paths.project_root)} and copy "
            f".env.example to {relative_path(settings.paths.secrets_path, settings.paths.project_root)}."
        ),
    ]


def index_missing_lines(
    *,
    title: str,
    collection_name: str,
    domain: str,
) -> list[str]:
    return [
        title,
        "State: index-missing",
        "Index missing",
        f"No indexed rows found for collection {collection_name}.",
        "Next",
        f"Run: uv run support-graph index-docs --domain {domain}",
    ]


def metric_text(
    value: float | None,
    *,
    retrieval_top_k: int | None = None,
    metric_k: int | None = None,
) -> str:
    if value is not None:
        return f"{value:.3f}"
    if (
        metric_k is not None
        and retrieval_top_k is not None
        and retrieval_top_k < metric_k
    ):
        return f"n/a (retrieval_top_k={retrieval_top_k})"
    return "n/a"


def index_unavailable_lines(
    *,
    title: str,
    collection_name: str,
    error: str,
) -> list[str]:
    return [
        title,
        "State: index-unavailable",
        "Index unavailable",
        f"Could not inspect collection {collection_name}.",
        error,
        "Next",
        "Verify Postgres is reachable and the DSN is correct, then retry.",
    ]
