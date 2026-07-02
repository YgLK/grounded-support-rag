from __future__ import annotations

import pytest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from support_graph.config.runtime import (
    ConfigValidationError,
    RuntimeConfig,
    RuntimeExperimentOverrides,
    apply_runtime_experiment_overrides,
)
from support_graph.providers import Provider
from support_graph.types import Domain, parse_domain


DEFAULT_POSTGRES_DSN = "postgresql://postgres:postgres@localhost:5432/support_graph"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CHAT_MODEL = "chat-model"
DEFAULT_EMBEDDING_MODEL = "embed-model"


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def make_settings(repo_root: Path):
    def factory(
        *,
        project_root: Path | None = None,
        missing_fields: list[str] | None = None,
        has_index: bool = True,
        enabled_domains: tuple[str, ...] = ("kubernetes",),
        overrides: dict | None = None,
    ) -> SimpleNamespace:
        root = Path(project_root) if project_root is not None else repo_root
        domain = enabled_domains[0] if enabled_domains else "kubernetes"
        parsed_domain = parse_domain(domain)
        parsed_domains = tuple(parse_domain(value) for value in enabled_domains) or (
            Domain.KUBERNETES,
        )
        missing = list(missing_fields or [])
        runtime_payload: dict[str, Any] = {
            "chat_provider_type": Provider.OPENROUTER,
            "embedding_provider_type": Provider.OPENROUTER,
            "ollama_base_url": DEFAULT_OLLAMA_BASE_URL,
            "openrouter_base_url": DEFAULT_OPENROUTER_BASE_URL,
            "openrouter_api_key": "or-test",
            "chat_model": DEFAULT_CHAT_MODEL,
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "prompt_version": "v1",
            "postgres_dsn": DEFAULT_POSTGRES_DSN,
            "retrieval_top_k": 5,
            "retrieval_candidate_k": 12,
            "max_retrieval_attempts": 2,
            "llm_timeout_seconds": 60.0,
            "llm_max_retries": 3,
            "llm_retry_base_delay_seconds": 0.5,
            "llm_retry_max_delay_seconds": 4.0,
            "langsmith_tracing_enabled": False,
            "langsmith_project": None,
            "langsmith_api_key": None,
            "langsmith_endpoint": None,
            "domain": parsed_domain,
            "collection_name": f"support_graph_{parsed_domain}",
            "chunk_artifact_path": root
            / "data/derived/chunks"
            / f"{parsed_domain}.jsonl",
        }
        payload = {
            "paths": SimpleNamespace(
                project_root=root,
                config_path=root / "support_graph.toml",
                secrets_path=root / ".env",
                eval_runs_dir=root / "outputs/evals/runs",
                eval_reports_dir=root / "outputs/evals/reports",
                runs_dir=root / "outputs/runs",
                log_dir=root / "logs",
                log_level="INFO",
                derived_dir=root / "data/derived",
                chunks_dir=root / "data/derived/chunks",
                examples_dir=root / "data/derived/examples",
            ),
            "dataset": SimpleNamespace(
                root=root / "raw/kubernetes/current",
                enabled_domains=parsed_domains,
            ),
            "runtime": None,
            "selected_domain": lambda explicit_domain=None: explicit_domain or domain,
            "chunk_artifact_path": lambda explicit_domain=None: (
                root / "data/derived/chunks" / f"{explicit_domain or domain}.jsonl"
            ),
            "collection_name": lambda explicit_domain=None: (
                f"support_graph_{explicit_domain or domain}"
            ),
        }
        if overrides:
            runtime_field_names = set(RuntimeConfig.__dataclass_fields__)
            runtime_overrides = {
                key: value
                for key, value in overrides.items()
                if key in runtime_field_names
            }
            payload.update(
                {
                    key: value
                    for key, value in overrides.items()
                    if key not in runtime_field_names
                }
            )
            if runtime_overrides:
                runtime_payload.update(runtime_overrides)

        def validate_for_index() -> None:
            if not has_index:
                raise ConfigValidationError(
                    scope="index",
                    missing_fields=[".env: SUPPORT_GRAPH_POSTGRES_DSN"],
                )

        def validate_for_run() -> None:
            if missing:
                raise ConfigValidationError(
                    scope="runtime",
                    missing_fields=missing,
                )

        payload["runtime"] = SimpleNamespace(
            **runtime_payload,
            validate_for_index=validate_for_index,
            validate_for_run=validate_for_run,
        )

        def runtime_for(
            explicit_domain=None,
            experiment: RuntimeExperimentOverrides | None = None,
        ) -> RuntimeConfig:
            resolved_domain = parse_domain(explicit_domain or domain)
            config = replace(
                RuntimeConfig(**runtime_payload),
                domain=resolved_domain,
                collection_name=f"support_graph_{resolved_domain}",
                chunk_artifact_path=root
                / "data/derived/chunks"
                / f"{resolved_domain}.jsonl",
            )
            return apply_runtime_experiment_overrides(config, experiment)

        payload["runtime_for"] = runtime_for
        return SimpleNamespace(**payload)

    return factory


@pytest.fixture
def runtime_example() -> dict:
    return {
        "example_id": "kubernetes::pods::turn_2",
        "domain": "kubernetes",
        "conversation": [
            {
                "turn_id": 1,
                "role": "user",
                "utterance": "What is a Kubernetes Pod?",
            },
        ],
        "latest_user_turn_id": 1,
        "latest_user_utterance": "What is a Kubernetes Pod?",
    }


@pytest.fixture
def make_runtime_config():
    def factory(**overrides) -> RuntimeConfig:
        payload: dict[str, Any] = {
            "domain": "kubernetes",
            "collection_name": "support_graph_kubernetes",
            "chat_provider_type": Provider.OPENROUTER,
            "embedding_provider_type": Provider.OPENROUTER,
            "retrieval_top_k": 5,
            "retrieval_candidate_k": 12,
            "retrieval_rerank": True,
            "content_only_reasoning": True,
            "neighbor_expansion": True,
            "max_retrieval_attempts": 2,
            "llm_timeout_seconds": 60.0,
        }
        payload.update(overrides)
        return RuntimeConfig(**payload)

    return factory


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run tests marked integration that require live Postgres/Ollama services",
    )


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: tests that require live Postgres and Ollama services",
    )


def pytest_collection_modifyitems(config, items) -> None:
    if config.getoption("--run-integration"):
        return
    skip_integration = pytest.mark.skip(
        reason="needs --run-integration to exercise live Postgres/Ollama tests"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
