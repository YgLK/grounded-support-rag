from pathlib import Path

import pytest

from support_graph.config.runtime import ConfigValidationError, RuntimeConfig
from support_graph.config.settings import (
    DatasetSettings,
    PathSettings,
    Settings,
    read_dotenv,
    read_settings_toml,
)
from support_graph.providers import Provider
from support_graph.types import Domain


def _write_settings_toml(
    path: Path,
    *,
    content: str | None = None,
) -> Path:
    payload = (
        content
        or """
[dataset]
root = "raw/kubernetes/current"
enabled_domains = ["kubernetes"]

[paths]
eval_runs_dir = "outputs/evals/runs"
eval_reports_dir = "outputs/evals/reports"
runs_dir = "outputs/runs"
log_dir = "logs"
log_level = "INFO"

[runtime]
chat_provider_type = "openrouter"
embedding_provider_type = "openrouter"
openrouter_base_url = "https://openrouter.ai/api/v1"
chat_model = "openai/gpt-4.1-mini"
embedding_model = "openai/text-embedding-3-small"

[observability.langsmith]
tracing_enabled = false
project = "grounded-support-rag"


""".strip()
    )
    path.write_text(payload + "\n", encoding="utf-8")
    return path


def test_read_dotenv_returns_empty_dict_for_missing_file(tmp_path: Path) -> None:
    assert read_dotenv(tmp_path / "missing.env") == {}


def test_read_dotenv_strips_quotes_supports_multiline_and_ignores_invalid_lines(
    tmp_path: Path,
) -> None:
    dotenv_path = tmp_path / "example.env"
    dotenv_path.write_text(
        "\n".join(
            [
                "# ignored comment",
                'SUPPORT_GRAPH_OPENROUTER_API_KEY="sk-or-v1-test"',
                'SUPPORT_GRAPH_NOTES="first line',
                'second line"',
                "SUPPORT_GRAPH_EMPTY=",
                "NOT_A_SETTING_LINE",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    expected = {
        "SUPPORT_GRAPH_OPENROUTER_API_KEY": "sk-or-v1-test",
        "SUPPORT_GRAPH_NOTES": "first line\nsecond line",
        "SUPPORT_GRAPH_EMPTY": "",
    }
    assert read_dotenv(dotenv_path) == expected


def test_read_settings_toml_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Settings file not found"):
        read_settings_toml(tmp_path / "missing.toml")


def test_settings_load_resolves_repo_root_and_relative_paths_from_repo_root(
    tmp_path: Path,
    repo_root: Path,
) -> None:
    settings_path = _write_settings_toml(tmp_path / "support_graph.toml")

    settings = Settings.load(settings_path)

    assert settings.paths.project_root == repo_root
    assert settings.paths.config_path == settings_path
    assert settings.dataset.root == repo_root / "raw/kubernetes/current"
    assert settings.runtime.chat_provider_type is Provider.OPENROUTER
    assert settings.runtime.embedding_provider_type is Provider.OPENROUTER
    assert settings.runtime.ollama_base_url == "http://localhost:11434"
    assert settings.runtime.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert settings.runtime.retrieval_top_k == 5
    assert settings.runtime.retrieval_candidate_k == 12
    assert settings.runtime.domain is Domain.KUBERNETES
    assert settings.runtime.collection_name == "support_graph_kubernetes"
    assert (
        settings.runtime.chunk_artifact_path
        == repo_root / "data/derived/chunks/kubernetes.jsonl"
    )
    assert settings.paths.runs_dir == repo_root / "outputs/runs"
    assert settings.paths.eval_runs_dir == repo_root / "outputs/evals/runs"
    assert settings.paths.eval_reports_dir == repo_root / "outputs/evals/reports"
    assert settings.paths.log_dir == repo_root / "logs"
    assert settings.paths.log_level == "INFO"
    assert (
        settings.chunk_artifact_path("kubernetes")
        == repo_root / "data/derived/chunks/kubernetes.jsonl"
    )
    assert (
        settings.paths.project_root / "data/eval_subsets/kubernetes/smoke.jsonl"
    ).exists()


def test_settings_load_reads_toml_and_secret_values(tmp_path: Path) -> None:
    settings_path = _write_settings_toml(
        tmp_path / "providers.toml",
        content="""
[dataset]
root = "raw/kubernetes/current"
enabled_domains = ["kubernetes"]

[paths]
eval_runs_dir = "outputs/evals/runs"
eval_reports_dir = "outputs/evals/reports"
runs_dir = "outputs/runs"
log_dir = "logs"
log_level = "INFO"

[runtime]
chat_provider_type = "openrouter"
embedding_provider_type = "openrouter"
openrouter_base_url = "https://openrouter.ai/api/v1"
chat_model = "google/gemini-2.5-flash-preview"
embedding_model = "openai/text-embedding-3-small"
prompt_version = "v2"
retrieval_top_k = 7
retrieval_candidate_k = 18
max_retrieval_attempts = 3
llm_timeout_seconds = 30.0
llm_max_retries = 5
llm_retry_base_delay_seconds = 0.25
llm_retry_max_delay_seconds = 2.5

[observability.langsmith]
tracing_enabled = true
project = "grounded-support-rag-prod"
endpoint = "https://api.smith.langchain.com"


""",
    )
    secrets_path = tmp_path / ".env"
    secrets_path.write_text(
        "\n".join(
            [
                "SUPPORT_GRAPH_POSTGRES_DSN=postgresql://localhost/support_graph",
                "SUPPORT_GRAPH_OPENROUTER_API_KEY=openrouter-key",
                "SUPPORT_GRAPH_LANGSMITH_API_KEY=langsmith-key",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    settings = Settings.load(settings_path, secrets_path)

    assert settings.runtime.postgres_dsn == "postgresql://localhost/support_graph"
    assert settings.runtime.chat_provider_type is Provider.OPENROUTER
    assert settings.runtime.embedding_provider_type is Provider.OPENROUTER
    assert settings.runtime.openrouter_api_key == "openrouter-key"
    assert settings.runtime.chat_model == "google/gemini-2.5-flash-preview"
    assert settings.runtime.embedding_model == "openai/text-embedding-3-small"
    assert settings.runtime.prompt_version == "v2"
    assert settings.runtime.retrieval_top_k == 7
    assert settings.runtime.retrieval_candidate_k == 18
    assert settings.runtime.max_retrieval_attempts == 3
    assert settings.runtime.llm_timeout_seconds == 30.0
    assert settings.runtime.llm_max_retries == 5
    assert settings.runtime.llm_retry_base_delay_seconds == 0.25
    assert settings.runtime.llm_retry_max_delay_seconds == 2.5
    assert settings.runtime.langsmith_tracing_enabled is True
    assert settings.runtime.langsmith_project == "grounded-support-rag-prod"
    assert settings.runtime.langsmith_api_key == "langsmith-key"
    assert settings.runtime.langsmith_endpoint == "https://api.smith.langchain.com"

    assert settings.runtime.domain is Domain.KUBERNETES
    assert settings.runtime.collection_name == "support_graph_kubernetes"


def test_runtime_validation_requires_openrouter_key_when_provider_is_openrouter() -> (
    None
):
    settings = Settings(
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
            openrouter_api_key=None,
            chat_model="google/gemini-2.5-flash-preview",
            embedding_model="openai/text-embedding-3-small",
            prompt_version="v1",
            retrieval_top_k=5,
            retrieval_candidate_k=12,
            max_retrieval_attempts=2,
            llm_timeout_seconds=60.0,
            llm_max_retries=3,
            llm_retry_base_delay_seconds=0.5,
            llm_retry_max_delay_seconds=4.0,
            langsmith_tracing_enabled=False,
            langsmith_project=None,
            langsmith_api_key=None,
            langsmith_endpoint=None,
            domain=Domain.KUBERNETES,
            collection_name="support_graph_kubernetes",
            chunk_artifact_path=Path(
                "/tmp/project/data/derived/chunks/kubernetes.jsonl"
            ),
        ),
    )

    with pytest.raises(ConfigValidationError) as exc_info:
        settings.runtime.validate_for_run()

    assert list(exc_info.value.missing_fields) == [
        ".env: SUPPORT_GRAPH_OPENROUTER_API_KEY"
    ]


def test_settings_parse_enabled_domains_into_shared_enum_values(
    tmp_path: Path,
) -> None:
    settings_path = _write_settings_toml(
        tmp_path / "domains.toml",
        content="""
[dataset]
root = "raw/kubernetes/current"
enabled_domains = ["kubernetes", "kubernetes"]

[paths]
eval_runs_dir = "outputs/evals/runs"
eval_reports_dir = "outputs/evals/reports"
runs_dir = "outputs/runs"
log_dir = "logs"
log_level = "INFO"

[runtime]
chat_provider_type = "openrouter"
embedding_provider_type = "openrouter"
openrouter_base_url = "https://openrouter.ai/api/v1"
chat_model = "openai/gpt-4.1-mini"
embedding_model = "openai/text-embedding-3-small"

[observability.langsmith]
tracing_enabled = false
project = "grounded-support-rag"


""",
    )

    settings = Settings.load(settings_path)

    assert settings.dataset.enabled_domains == (Domain.KUBERNETES,)
    assert settings.runtime.domain is Domain.KUBERNETES


def test_settings_reject_unknown_enabled_domain(tmp_path: Path) -> None:
    settings_path = _write_settings_toml(
        tmp_path / "domains.toml",
        content="""
[dataset]
root = "raw/kubernetes/current"
enabled_domains = ["kubernetes", "unknown"]

[paths]
eval_runs_dir = "outputs/evals/runs"
eval_reports_dir = "outputs/evals/reports"
runs_dir = "outputs/runs"
log_dir = "logs"
log_level = "INFO"

[runtime]
chat_provider_type = "openrouter"
embedding_provider_type = "openrouter"
openrouter_base_url = "https://openrouter.ai/api/v1"
chat_model = "openai/gpt-4.1-mini"
embedding_model = "openai/text-embedding-3-small"

[observability.langsmith]
tracing_enabled = false
project = "grounded-support-rag"


""",
    )

    with pytest.raises(ValueError, match="Unsupported domain 'unknown'"):
        Settings.load(settings_path)


def test_settings_resolve_log_dir_from_toml(tmp_path: Path, repo_root: Path) -> None:
    settings_path = _write_settings_toml(
        tmp_path / "logging.toml",
        content="""
[dataset]
root = "raw/kubernetes/current"
enabled_domains = ["kubernetes"]

[paths]
eval_runs_dir = "outputs/evals/runs"
eval_reports_dir = "outputs/evals/reports"
runs_dir = "outputs/runs"
log_dir = "runtime_logs"
log_level = "DEBUG"

[runtime]
chat_provider_type = "openrouter"
embedding_provider_type = "openrouter"
openrouter_base_url = "https://openrouter.ai/api/v1"
chat_model = "openai/gpt-4.1-mini"
embedding_model = "openai/text-embedding-3-small"

[observability.langsmith]
tracing_enabled = false
project = "grounded-support-rag"


""",
    )

    settings = Settings.load(settings_path)

    assert settings.paths.log_dir == repo_root / "runtime_logs"
    assert settings.paths.log_level == "DEBUG"
