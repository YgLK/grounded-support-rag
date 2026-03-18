"""Environment-backed settings for the SupportGraph MVP."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _csv_to_tuple(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not value:
        return default
    items = tuple(part.strip() for part in value.split(",") if part.strip())
    return items or default


def _int_value(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


@dataclass(frozen=True)
class Settings:
    project_root: Path
    dataset_root: Path
    enabled_domains: tuple[str, ...]
    postgres_dsn: str | None
    provider_type: str
    ollama_base_url: str
    chat_model: str | None
    embedding_model: str | None
    retrieval_top_k: int
    retrieval_candidate_k: int
    max_retrieval_attempts: int
    trace_dir: Path
    eval_dir: Path
    derived_dir: Path
    chunks_dir: Path
    examples_dir: Path
    dotenv_path: Path

    @classmethod
    def from_env(cls, dotenv_path: str | Path | None = None) -> "Settings":
        project_root = _repo_root()
        resolved_dotenv = Path(dotenv_path) if dotenv_path is not None else project_root / ".env"
        file_values = _read_dotenv(resolved_dotenv)
        env = {**file_values, **os.environ}

        dataset_root = Path(
            env.get("SUPPORT_GRAPH_DATASET_ROOT", project_root / "multidoc2dial")
        )
        trace_dir = Path(env.get("SUPPORT_GRAPH_TRACE_DIR", project_root / "outputs/traces"))
        eval_dir = Path(env.get("SUPPORT_GRAPH_EVAL_DIR", project_root / "outputs/evals"))
        derived_dir = project_root / "data/derived"
        chunks_dir = derived_dir / "chunks"
        examples_dir = derived_dir / "examples"

        return cls(
            project_root=project_root,
            dataset_root=dataset_root,
            enabled_domains=_csv_to_tuple(env.get("SUPPORT_GRAPH_ENABLED_DOMAINS"), ("dmv",)),
            postgres_dsn=env.get("SUPPORT_GRAPH_POSTGRES_DSN") or None,
            provider_type=env.get("SUPPORT_GRAPH_PROVIDER_TYPE", "ollama"),
            ollama_base_url=env.get("SUPPORT_GRAPH_OLLAMA_BASE_URL", "http://localhost:11434"),
            chat_model=env.get("SUPPORT_GRAPH_CHAT_MODEL") or None,
            embedding_model=env.get("SUPPORT_GRAPH_EMBEDDING_MODEL") or None,
            retrieval_top_k=_int_value(env.get("SUPPORT_GRAPH_RETRIEVAL_TOP_K"), 5),
            retrieval_candidate_k=_int_value(env.get("SUPPORT_GRAPH_RETRIEVAL_CANDIDATE_K"), 12),
            max_retrieval_attempts=_int_value(env.get("SUPPORT_GRAPH_MAX_RETRIEVAL_ATTEMPTS"), 2),
            trace_dir=trace_dir,
            eval_dir=eval_dir,
            derived_dir=derived_dir,
            chunks_dir=chunks_dir,
            examples_dir=examples_dir,
            dotenv_path=resolved_dotenv,
        )

    def selected_domain(self, explicit_domain: str | None = None) -> str:
        if explicit_domain:
            return explicit_domain
        return self.enabled_domains[0] if self.enabled_domains else "dmv"

    def chunk_artifact_path(self, explicit_domain: str | None = None) -> Path:
        return self.chunks_dir / f"{self.selected_domain(explicit_domain)}.jsonl"

    def collection_name(self, explicit_domain: str | None = None) -> str:
        return f"support_graph_{self.selected_domain(explicit_domain)}"

    def index_missing_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.postgres_dsn:
            missing.append("SUPPORT_GRAPH_POSTGRES_DSN")
        if not self.embedding_model:
            missing.append("SUPPORT_GRAPH_EMBEDDING_MODEL")
        return missing

    def runtime_missing_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.postgres_dsn:
            missing.append("SUPPORT_GRAPH_POSTGRES_DSN")
        if not self.chat_model:
            missing.append("SUPPORT_GRAPH_CHAT_MODEL")
        if not self.embedding_model:
            missing.append("SUPPORT_GRAPH_EMBEDDING_MODEL")
        return missing
