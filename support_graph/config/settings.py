"""Environment-backed settings for the SupportGraph MVP."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from support_graph.providers import (
    Provider,
    validate_chat_provider_type,
    validate_embedding_provider_type,
)
from support_graph.types import Domain, DomainLike, parse_domain, parse_domains


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _resolve_path(
    value: str | Path | None,
    *,
    project_root: Path,
    default: Path,
) -> Path:
    if value is None or value == "":
        return default
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path


def _env_value(env: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = env.get(key)
        if value is not None and value != "":
            return value
    return None


def _find_closing_quote(value: str, quote: str) -> int | None:
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == quote:
            return index
    return None


def _parse_dotenv_value(
    lines: list[str],
    *,
    line_index: int,
    raw_value: str,
) -> tuple[str, int]:
    value = raw_value.strip()
    if not value or value[0] not in {'"', "'"}:
        return value, line_index

    quote = value[0]
    parts = [value[1:]]
    next_index = line_index
    while True:
        closing_index = _find_closing_quote(parts[-1], quote)
        if closing_index is not None:
            parts[-1] = parts[-1][:closing_index]
            break
        if next_index >= len(lines):
            break
        parts.append(lines[next_index])
        next_index += 1
    return "\n".join(parts), next_index


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    line_index = 0
    while line_index < len(lines):
        raw_line = lines[line_index]
        line_index += 1
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized_key = key.strip()
        if not normalized_key:
            continue
        parsed_value, line_index = _parse_dotenv_value(
            lines,
            line_index=line_index,
            raw_value=value,
        )
        values[normalized_key] = parsed_value
    return values


def _csv_to_domains(
    value: str | None, default: tuple[Domain, ...]
) -> tuple[Domain, ...]:
    if not value:
        return default
    items = [part.strip() for part in value.split(",") if part.strip()]
    resolved = parse_domains(items)
    return resolved or default


def _int_value(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _float_value(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _bool_value(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    project_root: Path
    dataset_root: Path
    enabled_domains: tuple[Domain, ...]
    postgres_dsn: str | None
    provider_type: Provider
    embedding_provider_type: Provider | None
    ollama_base_url: str
    openai_base_url: str | None
    openai_api_key: str | None
    anthropic_base_url: str | None
    anthropic_api_key: str | None
    chat_model: str | None
    embedding_model: str | None
    prompt_version: str
    retrieval_top_k: int
    retrieval_candidate_k: int
    max_retrieval_attempts: int
    llm_max_concurrency: int
    llm_max_retries: int
    llm_retry_base_delay_seconds: float
    llm_retry_max_delay_seconds: float
    langsmith_tracing_enabled: bool
    langsmith_project: str | None
    langsmith_api_key: str | None
    langsmith_endpoint: str | None
    otel_enabled: bool
    otel_service_name: str
    otel_exporter: str | None
    otel_endpoint: str | None
    otel_headers: str | None
    trace_dir: Path
    eval_dir: Path
    derived_dir: Path
    chunks_dir: Path
    examples_dir: Path
    dotenv_path: Path

    @classmethod
    def from_env(cls, dotenv_path: str | Path | None = None) -> "Settings":
        project_root = _repo_root()
        resolved_dotenv = _resolve_path(
            dotenv_path,
            project_root=project_root,
            default=project_root / ".env",
        )
        file_values = _read_dotenv(resolved_dotenv)
        env = {**file_values, **os.environ}
        provider_type = validate_chat_provider_type(
            _env_value(env, "SUPPORT_GRAPH_PROVIDER_TYPE")
        )
        embedding_provider_raw = _env_value(
            env, "SUPPORT_GRAPH_EMBEDDING_PROVIDER_TYPE"
        )
        embedding_provider_type = (
            validate_embedding_provider_type(embedding_provider_raw)
            if embedding_provider_raw is not None
            else None
        )

        dataset_root = _resolve_path(
            env.get("SUPPORT_GRAPH_DATASET_ROOT"),
            project_root=project_root,
            default=project_root / "multidoc2dial",
        )
        trace_dir = _resolve_path(
            env.get("SUPPORT_GRAPH_TRACE_DIR"),
            project_root=project_root,
            default=project_root / "outputs/traces",
        )
        eval_dir = _resolve_path(
            env.get("SUPPORT_GRAPH_EVAL_DIR"),
            project_root=project_root,
            default=project_root / "outputs/evals",
        )
        derived_dir = project_root / "data/derived"
        chunks_dir = derived_dir / "chunks"
        examples_dir = derived_dir / "examples"

        return cls(
            project_root=project_root,
            dataset_root=dataset_root,
            enabled_domains=_csv_to_domains(
                env.get("SUPPORT_GRAPH_ENABLED_DOMAINS"), (Domain.DMV,)
            ),
            postgres_dsn=env.get("SUPPORT_GRAPH_POSTGRES_DSN") or None,
            provider_type=provider_type,
            embedding_provider_type=embedding_provider_type,
            ollama_base_url=env.get(
                "SUPPORT_GRAPH_OLLAMA_BASE_URL", "http://localhost:11434"
            ),
            openai_base_url=_env_value(
                env, "SUPPORT_GRAPH_OPENAI_BASE_URL", "OPENAI_BASE_URL"
            ),
            openai_api_key=_env_value(
                env, "SUPPORT_GRAPH_OPENAI_API_KEY", "OPENAI_API_KEY"
            ),
            anthropic_base_url=_env_value(
                env, "SUPPORT_GRAPH_ANTHROPIC_BASE_URL", "ANTHROPIC_BASE_URL"
            ),
            anthropic_api_key=_env_value(
                env, "SUPPORT_GRAPH_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"
            ),
            chat_model=env.get("SUPPORT_GRAPH_CHAT_MODEL") or None,
            embedding_model=env.get("SUPPORT_GRAPH_EMBEDDING_MODEL") or None,
            prompt_version=env.get("SUPPORT_GRAPH_PROMPT_VERSION", "v1"),
            retrieval_top_k=_int_value(env.get("SUPPORT_GRAPH_RETRIEVAL_TOP_K"), 5),
            retrieval_candidate_k=_int_value(
                env.get("SUPPORT_GRAPH_RETRIEVAL_CANDIDATE_K"), 12
            ),
            max_retrieval_attempts=_int_value(
                env.get("SUPPORT_GRAPH_MAX_RETRIEVAL_ATTEMPTS"), 2
            ),
            llm_max_concurrency=_int_value(
                env.get("SUPPORT_GRAPH_LLM_MAX_CONCURRENCY"), 4
            ),
            llm_max_retries=_int_value(env.get("SUPPORT_GRAPH_LLM_MAX_RETRIES"), 3),
            llm_retry_base_delay_seconds=_float_value(
                env.get("SUPPORT_GRAPH_LLM_RETRY_BASE_DELAY_SECONDS"), 0.5
            ),
            llm_retry_max_delay_seconds=_float_value(
                env.get("SUPPORT_GRAPH_LLM_RETRY_MAX_DELAY_SECONDS"), 4.0
            ),
            langsmith_tracing_enabled=_bool_value(
                _env_value(
                    env,
                    "SUPPORT_GRAPH_LANGSMITH_TRACING_ENABLED",
                    "LANGSMITH_TRACING",
                    "LANGCHAIN_TRACING_V2",
                ),
                False,
            ),
            langsmith_project=_env_value(
                env, "SUPPORT_GRAPH_LANGSMITH_PROJECT", "LANGSMITH_PROJECT"
            ),
            langsmith_api_key=_env_value(
                env, "SUPPORT_GRAPH_LANGSMITH_API_KEY", "LANGSMITH_API_KEY"
            ),
            langsmith_endpoint=_env_value(
                env, "SUPPORT_GRAPH_LANGSMITH_ENDPOINT", "LANGSMITH_ENDPOINT"
            ),
            otel_enabled=_bool_value(env.get("SUPPORT_GRAPH_OTEL_ENABLED"), False),
            otel_service_name=env.get(
                "SUPPORT_GRAPH_OTEL_SERVICE_NAME", "support-graph"
            ),
            otel_exporter=env.get("SUPPORT_GRAPH_OTEL_EXPORTER") or None,
            otel_endpoint=_env_value(
                env, "SUPPORT_GRAPH_OTEL_ENDPOINT", "OTEL_EXPORTER_OTLP_ENDPOINT"
            ),
            otel_headers=_env_value(
                env, "SUPPORT_GRAPH_OTEL_HEADERS", "OTEL_EXPORTER_OTLP_HEADERS"
            ),
            trace_dir=trace_dir,
            eval_dir=eval_dir,
            derived_dir=derived_dir,
            chunks_dir=chunks_dir,
            examples_dir=examples_dir,
            dotenv_path=resolved_dotenv,
        )

    def selected_domain(self, explicit_domain: DomainLike | None = None) -> Domain:
        if explicit_domain:
            return parse_domain(explicit_domain)
        if not self.enabled_domains:
            raise ValueError("enabled_domains must contain at least one domain.")
        return self.enabled_domains[0]

    def chunk_artifact_path(self, explicit_domain: DomainLike | None = None) -> Path:
        return self.chunks_dir / f"{self.selected_domain(explicit_domain)}.jsonl"

    def collection_name(self, explicit_domain: DomainLike | None = None) -> str:
        return f"support_graph_{self.selected_domain(explicit_domain)}"

    def index_missing_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.postgres_dsn:
            missing.append("SUPPORT_GRAPH_POSTGRES_DSN")
        if not self.embedding_model:
            missing.append("SUPPORT_GRAPH_EMBEDDING_MODEL")
        missing.extend(self._embedding_provider_missing_fields())
        return list(dict.fromkeys(missing))

    def runtime_missing_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.postgres_dsn:
            missing.append("SUPPORT_GRAPH_POSTGRES_DSN")
        if not self.chat_model:
            missing.append("SUPPORT_GRAPH_CHAT_MODEL")
        if not self.embedding_model:
            missing.append("SUPPORT_GRAPH_EMBEDDING_MODEL")
        missing.extend(self._chat_provider_missing_fields())
        missing.extend(self._embedding_provider_missing_fields())
        return list(dict.fromkeys(missing))

    def _chat_provider_missing_fields(self) -> list[str]:
        match self.provider_type:
            case Provider.OLLAMA:
                return []
            case Provider.OPENAI:
                return [] if self.openai_api_key else ["SUPPORT_GRAPH_OPENAI_API_KEY"]
            case Provider.ANTHROPIC:
                return (
                    []
                    if self.anthropic_api_key
                    else ["SUPPORT_GRAPH_ANTHROPIC_API_KEY"]
                )
        raise ValueError(f"Unknown provider_type: {self.provider_type!r}")

    def _embedding_provider_missing_fields(self) -> list[str]:
        if not self.embedding_model:
            return []
        if (
            self.embedding_provider_type is None
            and self.provider_type == Provider.ANTHROPIC
        ):
            return ["SUPPORT_GRAPH_EMBEDDING_PROVIDER_TYPE"]
        provider = self.embedding_provider_type or self.provider_type
        match provider:
            case Provider.OLLAMA:
                return []
            case Provider.OPENAI:
                return [] if self.openai_api_key else ["SUPPORT_GRAPH_OPENAI_API_KEY"]
        raise ValueError(f"Unknown embedding provider_type: {provider}")
