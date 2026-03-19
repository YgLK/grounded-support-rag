"""Shared runtime configuration contracts and builders."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol


class RuntimeSettingsLike(Protocol):
    postgres_dsn: str | None
    provider_type: str
    ollama_base_url: str
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
    trace_dir: Path

    def collection_name(self, explicit_domain: str | None = None) -> str: ...

    def chunk_artifact_path(self, explicit_domain: str | None = None) -> Path: ...


class EmbeddingBenchmarkConfigLike(Protocol):
    embedding_model: str | None


class IndexConfigLike(Protocol):
    postgres_dsn: str | None
    provider_type: str
    ollama_base_url: str | None
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
    trace_dir: Path
    chunk_records: list[dict[str, Any]] | None
    ablation_variant: str | None
    ablation_options: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    postgres_dsn: str | None = None
    provider_type: str = "ollama"
    ollama_base_url: str | None = None
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
        ollama_base_url=settings.ollama_base_url,
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
