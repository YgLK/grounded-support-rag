"""Shared runtime configuration contracts and builders."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from support_graph.providers import ChatProviderType, EmbeddingProviderType


class RuntimeSettingsLike(Protocol):
    postgres_dsn: str | None
    provider_type: ChatProviderType
    embedding_provider_type: EmbeddingProviderType | None
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

    def collection_name(self, explicit_domain: str | None = None) -> str: ...

    def chunk_artifact_path(self, explicit_domain: str | None = None) -> Path: ...


class EmbeddingBenchmarkConfigLike(Protocol):
    embedding_model: str | None


class IndexConfigLike(Protocol):
    postgres_dsn: str | None
    provider_type: ChatProviderType | None
    embedding_provider_type: EmbeddingProviderType | None
    ollama_base_url: str | None
    openai_base_url: str | None
    openai_api_key: str | None
    anthropic_base_url: str | None
    anthropic_api_key: str | None
    embedding_model: str | None
    embedding_client: Any | None
    domain: str
    collection_name: str
    chunk_artifact_path: Path | None


class RuntimeConfigLike(IndexConfigLike, Protocol):
    chat_model: str | None
    prompt_version: str
    retrieval_top_k: int
    retrieval_candidate_k: int
    retrieval_rerank: bool
    content_only_reasoning: bool
    neighbor_expansion: bool
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
    chunk_records: list[dict[str, Any]] | None
    ablation_variant: str | None
    ablation_options: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    postgres_dsn: str | None = None
    provider_type: ChatProviderType = "ollama"
    embedding_provider_type: EmbeddingProviderType | None = None
    ollama_base_url: str | None = None
    openai_base_url: str | None = None
    openai_api_key: str | None = None
    anthropic_base_url: str | None = None
    anthropic_api_key: str | None = None
    embedding_model: str | None = None
    chat_model: str | None = None
    prompt_version: str = "v1"
    domain: str = "dmv"
    collection_name: str = "support_graph_dmv"
    retrieval_top_k: int = 5
    retrieval_candidate_k: int = 12
    retrieval_rerank: bool = True
    content_only_reasoning: bool = True
    neighbor_expansion: bool = True
    max_retrieval_attempts: int = 2
    llm_max_concurrency: int = 4
    llm_max_retries: int = 3
    llm_retry_base_delay_seconds: float = 0.5
    llm_retry_max_delay_seconds: float = 4.0
    langsmith_tracing_enabled: bool = False
    langsmith_project: str | None = None
    langsmith_api_key: str | None = None
    langsmith_endpoint: str | None = None
    otel_enabled: bool = False
    otel_service_name: str = "support-graph"
    otel_exporter: str | None = None
    otel_endpoint: str | None = None
    otel_headers: str | None = None
    trace_dir: Path = Path("outputs/traces")
    chunk_artifact_path: Path | None = None
    chunk_records: list[dict[str, Any]] | None = None
    ablation_variant: str | None = None
    ablation_options: dict[str, Any] = field(default_factory=dict)
    embedding_client: Any | None = None


def build_runtime_config(settings: RuntimeSettingsLike, domain: str) -> RuntimeConfig:
    return RuntimeConfig(
        postgres_dsn=settings.postgres_dsn,
        provider_type=settings.provider_type,
        embedding_provider_type=settings.embedding_provider_type,
        ollama_base_url=settings.ollama_base_url,
        openai_base_url=settings.openai_base_url,
        openai_api_key=settings.openai_api_key,
        anthropic_base_url=settings.anthropic_base_url,
        anthropic_api_key=settings.anthropic_api_key,
        embedding_model=settings.embedding_model,
        chat_model=settings.chat_model,
        prompt_version=settings.prompt_version,
        domain=domain,
        collection_name=settings.collection_name(domain),
        retrieval_top_k=settings.retrieval_top_k,
        retrieval_candidate_k=settings.retrieval_candidate_k,
        retrieval_rerank=True,
        content_only_reasoning=True,
        neighbor_expansion=True,
        max_retrieval_attempts=settings.max_retrieval_attempts,
        llm_max_concurrency=settings.llm_max_concurrency,
        llm_max_retries=settings.llm_max_retries,
        llm_retry_base_delay_seconds=settings.llm_retry_base_delay_seconds,
        llm_retry_max_delay_seconds=settings.llm_retry_max_delay_seconds,
        langsmith_tracing_enabled=settings.langsmith_tracing_enabled,
        langsmith_project=settings.langsmith_project,
        langsmith_api_key=settings.langsmith_api_key,
        langsmith_endpoint=settings.langsmith_endpoint,
        otel_enabled=settings.otel_enabled,
        otel_service_name=settings.otel_service_name,
        otel_exporter=settings.otel_exporter,
        otel_endpoint=settings.otel_endpoint,
        otel_headers=settings.otel_headers,
        trace_dir=settings.trace_dir,
        chunk_artifact_path=settings.chunk_artifact_path(domain),
    )


def with_runtime_config_overrides(
    config: RuntimeConfig, **overrides: Any
) -> RuntimeConfig:
    return replace(config, **overrides)


__all__ = [
    "EmbeddingBenchmarkConfigLike",
    "IndexConfigLike",
    "RuntimeConfig",
    "RuntimeConfigLike",
    "RuntimeSettingsLike",
    "build_runtime_config",
    "with_runtime_config_overrides",
]
