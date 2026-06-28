from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples._shared import (
    SAMPLE_EXAMPLE_ID,
    find_by_key,
    format_path,
    load_or_build_examples,
    load_settings,
    missing_config_lines,
    next_step_lines,
    print_lines,
    runtime_config,
    unavailable_lines,
)
from support_graph.config.runtime import ConfigValidationError
from support_graph.artifacts import build_standalone_run_id, standalone_run_artifacts
from support_graph.retrieval.index import collection_row_count
from support_graph.runtime.graph import run_graph_async


def build_lines(
    *,
    settings=None,
    collection_row_count_func=collection_row_count,
    run_graph_func=run_graph_async,
) -> list[str]:
    settings = settings or load_settings()
    try:
        settings.runtime.validate_for_run()
    except ConfigValidationError as exc:
        return missing_config_lines(
            "SupportGraph Example 08",
            list(exc.missing_fields),
            next_commands=[
                "cp .env.example .env",
                "cp support_graph.toml.example support_graph.toml",
                "docker compose up -d postgres",
                "uv run grounded-support-rag index-docs --domain dmv",
            ],
        )

    examples = load_or_build_examples(settings)
    example = find_by_key(examples, "example_id", SAMPLE_EXAMPLE_ID)
    domain = example.get("domain", "dmv")
    config = runtime_config(settings, domain)

    try:
        row_count = collection_row_count_func(
            settings.runtime.postgres_dsn, settings.collection_name(domain)
        )
    except Exception as exc:
        return unavailable_lines(
            "SupportGraph Example 08",
            summary="Postgres or pgvector is not reachable yet",
            error=exc,
            next_commands=[
                "docker compose up -d postgres",
                "uv run grounded-support-rag index-docs --domain dmv",
            ],
        )
    if row_count == 0:
        return unavailable_lines(
            "SupportGraph Example 08",
            summary="The DMV collection is empty",
            error="No indexed rows found for support_graph_dmv.",
            next_commands=["uv run grounded-support-rag index-docs --domain dmv"],
        )

    try:
        run_id = build_standalone_run_id()
        artifacts = standalone_run_artifacts(settings.paths.project_root, run_id)
        artifacts.output_dir.mkdir(parents=True, exist_ok=True)
        result = asyncio.run(
            run_graph_func(
                example=example,
                config=config,
                run_id=run_id,
                trace_path=artifacts.trace,
                max_attempts=settings.runtime.max_retrieval_attempts,
            )
        )
    except Exception as exc:
        return unavailable_lines(
            "SupportGraph Example 08",
            summary="run_graph could not complete",
            error=exc,
            next_commands=[
                "uv run grounded-support-rag run --example-id 'dmv::1409501a35697e0ce68561e29577b90a::turn_2' --verbose",
            ],
        )

    trace_summary = result.get("trace_summary", {})
    lines = [
        "SupportGraph Example 08",
        "One real graph run",
        "",
        "Runtime input",
        f"- Example ID: {example['example_id']}",
        f"- Target mode: {example['target_mode']}",
        f"- Latest user utterance: {example['latest_user_utterance']}",
        f"- Gold doc IDs: {example['gold_doc_ids']}",
        f"- Gold span IDs: {example['gold_span_ids']}",
        "",
        "Runtime output",
        f"- Decision: {result.get('decision')}",
        f"- Response text: {result.get('response_text')}",
        f"- Citation chunk IDs: {[citation.get('chunk_id') for citation in result.get('citations', [])]}",
        f"- Ranked chunks: {len(result.get('retrieval_ranked_chunks', []))}",
        f"- Expanded evidence chunks: {len(result.get('retrieved_chunks', []))}",
        f"- Trace path: {format_path(artifacts.trace)}",
        f"- Graph path: {' -> '.join(trace_summary.get('graph_path', []))}",
    ]
    lines.extend(next_step_lines("09_eval_smoke_walkthrough.py"))
    return lines


def main() -> None:
    print_lines(build_lines())


if __name__ == "__main__":
    main()
