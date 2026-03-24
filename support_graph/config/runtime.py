"""Concrete runtime config and experiment overrides."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from support_graph.providers import (
    DEFAULT_OPENROUTER_BASE_URL,
    Provider,
    chat_provider as resolved_chat_provider,
    embedding_provider as resolved_embedding_provider,
)
from support_graph.types import Domain, DomainLike

if TYPE_CHECKING:
    from support_graph.config.settings import Settings

DEFAULT_RETRIEVAL_RERANK = True
DEFAULT_CONTENT_ONLY_REASONING = True
DEFAULT_NEIGHBOR_EXPANSION = True
MISSING_POSTGRES_DSN = ".env: SUPPORT_GRAPH_POSTGRES_DSN"
MISSING_CHAT_MODEL = "support_graph.toml: runtime.chat_model"
MISSING_EMBEDDING_MODEL = "support_graph.toml: runtime.embedding_model"
MISSING_OPENROUTER_API_KEY = ".env: SUPPORT_GRAPH_OPENROUTER_API_KEY"


class ConfigValidationError(ValueError):
    def __init__(self, *, scope: str, missing_fields: list[str]) -> None:
        unique_fields = tuple(dict.fromkeys(missing_fields))
        self.scope = scope
        self.missing_fields = unique_fields
        joined = ", ".join(unique_fields)
        super().__init__(f"Missing {scope} config: {joined}")


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeExperimentOverrides:
    retrieval_rerank: bool = DEFAULT_RETRIEVAL_RERANK
    content_only_reasoning: bool = DEFAULT_CONTENT_ONLY_REASONING
    neighbor_expansion: bool = DEFAULT_NEIGHBOR_EXPANSION
    experiment_variant: str | None = None
    experiment_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeConfig:
    postgres_dsn: str | None = None
    chat_provider_type: Provider = Provider.OPENROUTER
    embedding_provider_type: Provider = Provider.OPENROUTER
    ollama_base_url: str = "http://localhost:11434"
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL
    openrouter_api_key: str | None = None
    chat_model: str | None = None
    embedding_model: str | None = None
    prompt_version: str = "v2"
    retrieval_top_k: int = 5
    retrieval_candidate_k: int = 12
    max_retrieval_attempts: int = 2
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3
    llm_retry_base_delay_seconds: float = 0.5
    llm_retry_max_delay_seconds: float = 4.0
    langsmith_tracing_enabled: bool = False
    langsmith_project: str | None = None
    langsmith_api_key: str | None = None
    langsmith_endpoint: str | None = None
    domain: Domain = Domain.DMV
    collection_name: str = "support_graph_dmv"
    retrieval_rerank: bool = DEFAULT_RETRIEVAL_RERANK
    content_only_reasoning: bool = DEFAULT_CONTENT_ONLY_REASONING
    neighbor_expansion: bool = DEFAULT_NEIGHBOR_EXPANSION
    chunk_artifact_path: Path | None = None
    chunk_records: list[dict[str, Any]] | None = None
    experiment_variant: str | None = None
    experiment_options: dict[str, Any] = field(default_factory=dict)
    embedding_client: Any | None = None

    def validate_for_index(self) -> None:
        self._validate(scope="index", needs_chat_model=False)

    def validate_for_run(self) -> None:
        self._validate(scope="runtime", needs_chat_model=True)

    def _validate(self, *, scope: str, needs_chat_model: bool) -> None:
        missing = self._missing_fields(needs_chat_model=needs_chat_model)
        if missing:
            raise ConfigValidationError(scope=scope, missing_fields=missing)

    def _missing_fields(self, *, needs_chat_model: bool) -> list[str]:
        missing: list[str] = []
        if not self.postgres_dsn:
            missing.append(MISSING_POSTGRES_DSN)
        if needs_chat_model and not self.chat_model:
            missing.append(MISSING_CHAT_MODEL)
        if not self.embedding_model:
            missing.append(MISSING_EMBEDDING_MODEL)
        missing.extend(self._provider_missing_fields())
        return missing

    def _provider_missing_fields(self) -> list[str]:
        providers = {
            resolved_chat_provider(self),
            resolved_embedding_provider(self),
        }
        if Provider.OPENROUTER in providers and not self.openrouter_api_key:
            return [MISSING_OPENROUTER_API_KEY]
        return []


def apply_runtime_experiment_overrides(
    config: RuntimeConfig,
    experiment: RuntimeExperimentOverrides | None = None,
) -> RuntimeConfig:
    if experiment is None:
        return config
    return replace(
        config,
        retrieval_rerank=experiment.retrieval_rerank,
        content_only_reasoning=experiment.content_only_reasoning,
        neighbor_expansion=experiment.neighbor_expansion,
        experiment_variant=experiment.experiment_variant,
        experiment_options=dict(experiment.experiment_options),
    )


def build_runtime_config(settings: Settings, domain: DomainLike) -> RuntimeConfig:
    return settings.runtime_for(domain)


__all__ = [
    "ConfigValidationError",
    "RuntimeConfig",
    "RuntimeExperimentOverrides",
    "apply_runtime_experiment_overrides",
    "build_runtime_config",
    "DEFAULT_CONTENT_ONLY_REASONING",
    "DEFAULT_NEIGHBOR_EXPANSION",
    "DEFAULT_RETRIEVAL_RERANK",
]
