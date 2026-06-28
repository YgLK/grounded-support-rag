from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples._shared import (
    SAMPLE_DOMAIN,
    format_path,
    load_or_build_chunks,
    load_settings,
    missing_config_lines,
    next_step_lines,
    print_lines,
    unavailable_lines,
)
from support_graph.config.runtime import ConfigValidationError
from support_graph.retrieval.index import collection_row_count, index_documents


def _index_config(settings, domain: str) -> SimpleNamespace:
    return SimpleNamespace(
        postgres_dsn=settings.runtime.postgres_dsn,
        provider_type=settings.runtime.provider_type,
        ollama_base_url=settings.runtime.ollama_base_url,
        openrouter_base_url=settings.runtime.openrouter_base_url,
        openrouter_api_key=settings.runtime.openrouter_api_key,
        embedding_model=settings.runtime.embedding_model,
        embedding_client=None,
        chat_model=settings.runtime.chat_model,
        domain=domain,
        collection_name=settings.collection_name(domain),
        chunk_artifact_path=settings.chunk_artifact_path(domain),
    )


def build_lines(
    *,
    settings=None,
    collection_row_count_func=collection_row_count,
    index_documents_func=index_documents,
) -> list[str]:
    settings = settings or load_settings()
    try:
        settings.runtime.validate_for_index()
    except ConfigValidationError as exc:
        return missing_config_lines(
            "SupportGraph Example 07",
            list(exc.missing_fields),
            next_commands=[
                "cp .env.example .env",
                "cp support_graph.toml.example support_graph.toml",
                "docker compose up -d postgres",
                "uv run grounded-support-rag index-docs --domain dmv",
            ],
        )

    domain = SAMPLE_DOMAIN
    chunk_records = load_or_build_chunks(settings, domain=domain)
    config = _index_config(settings, domain)
    try:
        row_count = collection_row_count_func(
            settings.runtime.postgres_dsn, settings.collection_name(domain)
        )
    except Exception as exc:
        return unavailable_lines(
            "SupportGraph Example 07",
            summary="Postgres or pgvector is not reachable yet",
            error=exc,
            next_commands=[
                "docker compose up -d postgres",
                "uv run grounded-support-rag index-docs --domain dmv",
            ],
        )

    action_line = "Collection already has indexed rows."
    if row_count == 0:
        try:
            index_documents_func(config, chunk_records=chunk_records, batch_size=1)
            row_count = collection_row_count_func(
                settings.runtime.postgres_dsn, settings.collection_name(domain)
            )
            action_line = f"Indexed {len(chunk_records)} chunk rows into pgvector."
        except Exception as exc:
            return unavailable_lines(
                "SupportGraph Example 07",
                summary="Indexing could not complete",
                error=exc,
                next_commands=[
                    "docker compose up -d postgres",
                    "uv run grounded-support-rag index-docs --domain dmv",
                ],
            )

    lines = [
        "SupportGraph Example 07",
        "Index and vectorstore walkthrough",
        "",
        "Index inputs",
        f"- Chunk artifact: {format_path(settings.chunk_artifact_path(domain))}",
        f"- Collection name: {settings.collection_name(domain)}",
        f"- Embedding model: {settings.runtime.embedding_model}",
        "- The embedding model is used for chunk text and query vectors.",
        "",
        "Index state",
        f"- {action_line}",
        f"- Indexed row count: {row_count}",
    ]
    lines.extend(next_step_lines("08_run_graph_example.py"))
    return lines


def main() -> None:
    print_lines(build_lines())


if __name__ == "__main__":
    main()
