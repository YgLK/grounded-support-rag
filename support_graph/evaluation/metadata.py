"""Pure experiment metadata and reproducibility probes."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from support_graph.config.runtime import RuntimeConfig
from support_graph.evaluation.contracts import DatasetRef
from support_graph.providers import chat_provider, embedding_provider


@dataclass(frozen=True, slots=True)
class GitState:
    sha: str
    dirty: bool


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_git_state(project_root: Path) -> GitState:
    """Read the exact source revision and dirty flag for an experiment."""
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return GitState(sha=sha, dirty=dirty)


def build_experiment_metadata(
    *,
    config: RuntimeConfig,
    dataset: DatasetRef,
    domain: str,
    subset: str,
    corpus_ref: str,
    corpus_manifest_path: Path,
    evaluator_versions: dict[str, str],
    judge_provider: str | None,
    judge_model: str | None,
    git_state: GitState,
    started_at: datetime,
    example_count: int,
) -> dict[str, Any]:
    """Build the full hosted-evaluation contract without side effects."""
    chunk_artifact_path = config.chunk_artifact_path
    if chunk_artifact_path is None:
        raise FileNotFoundError("Evaluation metadata requires a chunk artifact path")
    if not corpus_manifest_path.is_file():
        raise FileNotFoundError(f"Corpus manifest not found: {corpus_manifest_path}")
    if not chunk_artifact_path.is_file():
        raise FileNotFoundError(f"Chunk artifact not found: {chunk_artifact_path}")

    return {
        "experiment_schema_version": "1",
        "metric_schema_version": "1",
        "dataset_name": dataset.name,
        "dataset_sha256": dataset.sha256,
        "domain": domain,
        "subset": subset,
        "git_sha": git_state.sha,
        "git_dirty": git_state.dirty,
        "corpus_ref": corpus_ref,
        "corpus_manifest_sha256": file_sha256(corpus_manifest_path),
        "chunk_artifact_sha256": file_sha256(chunk_artifact_path),
        "prompt_version": config.prompt_version,
        "retrieval_policy_version": "1",
        "retrieval_config": _retrieval_config(config),
        "chat_provider": chat_provider(config).value,
        "chat_model": config.chat_model,
        "embedding_provider": embedding_provider(config).value,
        "embedding_model": config.embedding_model,
        "evaluator_versions": dict(evaluator_versions),
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "started_at": started_at.astimezone(timezone.utc).isoformat(),
        "completed_at": None,
        "example_count": example_count,
        "status": "running",
    }


def _retrieval_config(config: RuntimeConfig) -> dict[str, Any]:
    return {
        "top_k": config.retrieval_top_k,
        "candidate_k": config.retrieval_candidate_k,
        "rerank": config.retrieval_rerank,
        "content_only_reasoning": config.content_only_reasoning,
        "neighbor_expansion": config.neighbor_expansion,
        "experiment_overrides": dict(config.experiment_options),
    }
