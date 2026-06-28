from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples._shared import (
    SAMPLE_DOMAIN,
    format_path,
    load_settings,
    missing_config_lines,
    next_step_lines,
    print_lines,
    unavailable_lines,
)
from support_graph.config.runtime import ConfigValidationError
from support_graph.evaluation.evaluate import evaluate_split_async
from support_graph.retrieval.index import collection_row_count


def build_lines(
    *,
    settings=None,
    collection_row_count_func=collection_row_count,
    evaluate_split_func=evaluate_split_async,
) -> list[str]:
    settings = settings or load_settings()
    try:
        settings.runtime.validate_for_run()
    except ConfigValidationError as exc:
        return missing_config_lines(
            "SupportGraph Example 09",
            list(exc.missing_fields),
            next_commands=[
                "cp .env.example .env",
                "cp support_graph.toml.example support_graph.toml",
                "docker compose up -d postgres",
                "uv run grounded-support-rag index-docs --domain dmv",
            ],
        )

    try:
        row_count = collection_row_count_func(
            settings.runtime.postgres_dsn, settings.collection_name(SAMPLE_DOMAIN)
        )
    except Exception as exc:
        return unavailable_lines(
            "SupportGraph Example 09",
            summary="Postgres or pgvector is not reachable yet",
            error=exc,
            next_commands=[
                "docker compose up -d postgres",
                "uv run grounded-support-rag index-docs --domain dmv",
            ],
        )
    if row_count == 0:
        return unavailable_lines(
            "SupportGraph Example 09",
            summary="The DMV collection is empty",
            error="No indexed rows found for support_graph_dmv.",
            next_commands=["uv run grounded-support-rag index-docs --domain dmv"],
        )

    try:
        result = asyncio.run(
            evaluate_split_func(
                settings=settings,
                domain=SAMPLE_DOMAIN,
                split="validation",
                subset="smoke",
                limit=3,
                notes="Examples walkthrough smoke sample.",
            )
        )
    except Exception as exc:
        return unavailable_lines(
            "SupportGraph Example 09",
            summary="The smoke eval walkthrough could not complete",
            error=exc,
            next_commands=[
                "uv run grounded-support-rag eval --domain dmv --subset smoke --limit 3",
            ],
        )

    metrics = result["metrics"]
    retrieval = metrics["retrieval"]["answer"]
    generation = metrics["generation"]["answer"]
    artifact_paths = result["artifact_paths"]
    lines = [
        "SupportGraph Example 09",
        "Small real smoke eval walkthrough",
        "",
        "Run",
        f"- Run ID: {result['run_id']}",
        f"- Subset label: {result['subset_label']}",
        f"- Evaluated examples: {metrics['counts']['examples']}",
        "",
        "Headline metrics",
        f"- Doc Recall@3: {(retrieval.get('doc_recall_at_3') or 0.0):.3f}",
        f"- Span Recall@5: {(retrieval.get('span_recall_at_5') or 0.0):.3f}",
        f"- ROUGE-L: {(generation.get('rouge_l') or 0.0):.3f}",
        f"- Token F1: {(generation.get('token_f1') or 0.0):.3f}",
        "",
        "Artifacts",
        f"- manifest.json -> {format_path(artifact_paths['manifest'])}",
        f"- metrics.json -> {format_path(artifact_paths['metrics'])}",
        f"- predictions.jsonl -> {format_path(artifact_paths['predictions'])}",
        f"- summary.md -> {format_path(artifact_paths['summary'])}",
    ]
    lines.extend(next_step_lines("10_trace_and_failure_review.py"))
    return lines


def main() -> None:
    print_lines(build_lines())


if __name__ == "__main__":
    main()
