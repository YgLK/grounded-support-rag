"""File-backed settings for the SupportGraph MVP."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field

from support_graph.config.runtime import (
    RuntimeConfig,
    RuntimeExperimentOverrides,
    apply_runtime_experiment_overrides,
)
from support_graph.providers import Provider
from support_graph.types import Domain, DomainLike, parse_domain, parse_domains

DEFAULT_SETTINGS_FILE_NAME = "support_graph.toml"
DEFAULT_SECRETS_FILE_NAME = ".env"


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


def read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {
        key: value for key, value in dotenv_values(path).items() if value is not None
    }


def read_settings_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Settings file must decode to a table: {path}")
    return payload


def _optional_secret(secrets: dict[str, str], key: str) -> str | None:
    value = secrets.get(key)
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetFileConfig(_FrozenModel):
    root: str = "multidoc2dial"
    enabled_domains: list[str] = Field(default_factory=lambda: [Domain.DMV.value])


class PathsFileConfig(_FrozenModel):
    eval_runs_dir: str = "outputs/evals/runs"
    eval_reports_dir: str = "outputs/evals/reports"
    runs_dir: str = "outputs/runs"
    log_dir: str = "logs"
    log_level: str = "INFO"


class RuntimeFileConfig(_FrozenModel):
    chat_provider_type: Provider = Provider.OPENROUTER
    embedding_provider_type: Provider = Provider.OPENROUTER
    ollama_base_url: str = "http://localhost:11434"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    chat_model: str | None = None
    embedding_model: str | None = None
    prompt_version: str = "v1"
    retrieval_top_k: int = 5
    retrieval_candidate_k: int = 12
    max_retrieval_attempts: int = 2
    llm_max_concurrency: int = 4
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3
    llm_retry_base_delay_seconds: float = 0.5
    llm_retry_max_delay_seconds: float = 4.0


class LangSmithFileConfig(_FrozenModel):
    tracing_enabled: bool = False
    project: str | None = "support-graph"
    endpoint: str | None = None


class ObservabilityFileConfig(_FrozenModel):
    langsmith: LangSmithFileConfig = Field(default_factory=LangSmithFileConfig)


class SettingsFile(_FrozenModel):
    dataset: DatasetFileConfig = Field(default_factory=DatasetFileConfig)
    paths: PathsFileConfig = Field(default_factory=PathsFileConfig)
    runtime: RuntimeFileConfig = Field(default_factory=RuntimeFileConfig)
    observability: ObservabilityFileConfig = Field(
        default_factory=ObservabilityFileConfig
    )

    @classmethod
    def from_toml(cls, path: str | Path) -> "SettingsFile":
        return cls.model_validate(read_settings_toml(Path(path)))


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetSettings:
    root: Path
    enabled_domains: tuple[Domain, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class PathSettings:
    project_root: Path
    config_path: Path
    secrets_path: Path
    eval_runs_dir: Path
    eval_reports_dir: Path
    runs_dir: Path
    log_dir: Path
    log_level: str
    derived_dir: Path
    chunks_dir: Path
    examples_dir: Path


@dataclass(frozen=True, slots=True, kw_only=True)
class Settings:
    dataset: DatasetSettings
    paths: PathSettings
    runtime: RuntimeConfig

    @classmethod
    def load(
        cls,
        config_path: str | Path | None = None,
        secrets_path: str | Path | None = None,
    ) -> "Settings":
        project_root = _repo_root()
        resolved_config_path = _resolve_path(
            config_path,
            project_root=project_root,
            default=project_root / DEFAULT_SETTINGS_FILE_NAME,
        )
        resolved_secrets_path = _resolve_path(
            secrets_path,
            project_root=project_root,
            default=project_root / DEFAULT_SECRETS_FILE_NAME,
        )

        file_config = SettingsFile.from_toml(resolved_config_path)
        secrets = read_dotenv(resolved_secrets_path)

        derived_dir = project_root / "data/derived"
        enabled_domains = parse_domains(file_config.dataset.enabled_domains) or (
            Domain.DMV,
        )
        selected_domain = enabled_domains[0]
        runtime_kwargs = file_config.runtime.model_dump(mode="python")

        return cls(
            dataset=DatasetSettings(
                root=_resolve_path(
                    file_config.dataset.root,
                    project_root=project_root,
                    default=project_root / "multidoc2dial",
                ),
                enabled_domains=enabled_domains,
            ),
            paths=PathSettings(
                project_root=project_root,
                config_path=resolved_config_path,
                secrets_path=resolved_secrets_path,
                eval_runs_dir=_resolve_path(
                    file_config.paths.eval_runs_dir,
                    project_root=project_root,
                    default=project_root / "outputs/evals/runs",
                ),
                eval_reports_dir=_resolve_path(
                    file_config.paths.eval_reports_dir,
                    project_root=project_root,
                    default=project_root / "outputs/evals/reports",
                ),
                runs_dir=_resolve_path(
                    file_config.paths.runs_dir,
                    project_root=project_root,
                    default=project_root / "outputs/runs",
                ),
                log_dir=_resolve_path(
                    file_config.paths.log_dir,
                    project_root=project_root,
                    default=project_root / "logs",
                ),
                log_level=file_config.paths.log_level,
                derived_dir=derived_dir,
                chunks_dir=derived_dir / "chunks",
                examples_dir=derived_dir / "examples",
            ),
            runtime=RuntimeConfig(
                langsmith_tracing_enabled=file_config.observability.langsmith.tracing_enabled,
                langsmith_project=file_config.observability.langsmith.project,
                langsmith_endpoint=file_config.observability.langsmith.endpoint,
                postgres_dsn=_optional_secret(secrets, "SUPPORT_GRAPH_POSTGRES_DSN"),
                openrouter_api_key=_optional_secret(
                    secrets, "SUPPORT_GRAPH_OPENROUTER_API_KEY"
                ),
                langsmith_api_key=_optional_secret(
                    secrets, "SUPPORT_GRAPH_LANGSMITH_API_KEY"
                ),
                domain=selected_domain,
                collection_name=f"support_graph_{selected_domain}",
                chunk_artifact_path=derived_dir / "chunks" / f"{selected_domain}.jsonl",
                **runtime_kwargs,
            ),
        )

    def selected_domain(self, explicit_domain: DomainLike | None = None) -> Domain:
        if explicit_domain:
            return parse_domain(explicit_domain)
        if not self.dataset.enabled_domains:
            raise ValueError("enabled_domains must contain at least one domain.")
        return self.dataset.enabled_domains[0]

    def chunk_artifact_path(self, explicit_domain: DomainLike | None = None) -> Path:
        return self.paths.chunks_dir / f"{self.selected_domain(explicit_domain)}.jsonl"

    def collection_name(self, explicit_domain: DomainLike | None = None) -> str:
        return f"support_graph_{self.selected_domain(explicit_domain)}"

    def runtime_for(
        self,
        explicit_domain: DomainLike | None = None,
        experiment: RuntimeExperimentOverrides | None = None,
    ) -> RuntimeConfig:
        resolved_domain = self.selected_domain(explicit_domain)
        config = replace(
            self.runtime,
            domain=resolved_domain,
            collection_name=self.collection_name(resolved_domain),
            chunk_artifact_path=self.chunk_artifact_path(resolved_domain),
        )
        return apply_runtime_experiment_overrides(config, experiment)


__all__ = [
    "DatasetSettings",
    "PathSettings",
    "Settings",
    "SettingsFile",
    "read_dotenv",
    "read_settings_toml",
]
