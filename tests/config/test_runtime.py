from dataclasses import replace
from pathlib import Path

import pytest

from support_graph.config.runtime import (
    ConfigValidationError,
    RuntimeConfig,
    RuntimeExperimentOverrides,
    build_runtime_config,
)
from support_graph.config.settings import (
    DatasetSettings,
    PathSettings,
    Settings,
)
from support_graph.providers import Provider
from support_graph.types import Domain


def _make_settings() -> Settings:
    return Settings(
        dataset=DatasetSettings(
            root=Path("/tmp/project/raw/kubernetes/current"),
            enabled_domains=(Domain.KUBERNETES,),
        ),
        paths=PathSettings(
            project_root=Path("/tmp/project"),
            config_path=Path("/tmp/project/support_graph.toml"),
            secrets_path=Path("/tmp/project/.env"),
            eval_runs_dir=Path("/tmp/project/outputs/evals/runs"),
            eval_reports_dir=Path("/tmp/project/outputs/evals/reports"),
            runs_dir=Path("/tmp/project/outputs/runs"),
            log_dir=Path("/tmp/project/logs"),
            log_level="INFO",
            derived_dir=Path("/tmp/project/data/derived"),
            chunks_dir=Path("/tmp/project/data/derived/chunks"),
            examples_dir=Path("/tmp/project/data/derived/examples"),
        ),
        runtime=RuntimeConfig(
            postgres_dsn="postgresql://localhost/support_graph",
            chat_provider_type=Provider.OPENROUTER,
            embedding_provider_type=Provider.OPENROUTER,
            ollama_base_url="http://localhost:11434",
            openrouter_base_url="https://openrouter.ai/api/v1",
            openrouter_api_key="or-key",
            chat_model="google/gemini-2.5-flash-preview",
            embedding_model="openai/text-embedding-3-small",
            prompt_version="v2",
            retrieval_top_k=7,
            retrieval_candidate_k=15,
            max_retrieval_attempts=3,
            llm_timeout_seconds=30.0,
            llm_max_retries=5,
            llm_retry_base_delay_seconds=0.25,
            llm_retry_max_delay_seconds=2.5,
            langsmith_tracing_enabled=True,
            langsmith_project="grounded-support-rag",
            langsmith_api_key="ls-key",
            langsmith_endpoint="https://api.smith.langchain.com",
            domain=Domain.KUBERNETES,
            collection_name="support_graph_kubernetes",
            chunk_artifact_path=Path(
                "/tmp/project/data/derived/chunks/kubernetes.jsonl"
            ),
        ),
    )


def test_build_runtime_config_reuses_shared_runtime_fields() -> None:
    settings = _make_settings()

    assert settings.runtime.chat_model == "google/gemini-2.5-flash-preview"

    config = build_runtime_config(settings, "kubernetes")

    assert config.domain == Domain.KUBERNETES
    assert config.collection_name == "support_graph_kubernetes"
    assert config.chunk_artifact_path == Path(
        "/tmp/project/data/derived/chunks/kubernetes.jsonl"
    )
    assert config.retrieval_top_k == 7
    assert config.llm_timeout_seconds == 30.0
    assert config.langsmith_tracing_enabled is True


def test_runtime_for_applies_typed_experiment_overrides() -> None:
    settings = _make_settings()

    experiment = RuntimeExperimentOverrides(
        retrieval_rerank=False,
        content_only_reasoning=False,
        neighbor_expansion=False,
        experiment_variant="structured-query",
        experiment_options={"query_mode": "structured"},
    )

    config = settings.runtime_for("kubernetes", experiment)

    assert config.domain == Domain.KUBERNETES
    assert config.collection_name == "support_graph_kubernetes"
    assert config.chunk_artifact_path == Path(
        "/tmp/project/data/derived/chunks/kubernetes.jsonl"
    )
    assert config.retrieval_rerank is False
    assert config.content_only_reasoning is False
    assert config.neighbor_expansion is False
    assert config.experiment_variant == "structured-query"
    assert config.experiment_options == {"query_mode": "structured"}
    assert settings.runtime.retrieval_rerank is True
    assert settings.runtime.domain == Domain.KUBERNETES


def test_validate_for_hosted_eval_requires_langsmith_configuration() -> None:
    config = replace(
        _make_settings().runtime,
        langsmith_tracing_enabled=False,
        langsmith_api_key=None,
        langsmith_project=None,
    )

    with pytest.raises(ConfigValidationError) as exc_info:
        config.validate_for_hosted_eval()

    assert exc_info.value.scope == "hosted evaluation"
    assert list(exc_info.value.missing_fields) == [
        "LANGSMITH_TRACING=true",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
    ]


def test_validate_for_run_does_not_require_langsmith() -> None:
    replace(
        _make_settings().runtime,
        langsmith_tracing_enabled=False,
        langsmith_api_key=None,
        langsmith_project=None,
    ).validate_for_run()
