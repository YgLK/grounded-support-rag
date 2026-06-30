"""CLI entrypoint for the SupportGraph MVP."""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import cast

from support_graph.cli.formatting import (
    format_experiment_output,
    format_eval_output,
    format_review_failures_output,
    format_run_output,
    format_trace_show_output,
)
from support_graph.cli.parser import CliHandlers, build_parser as build_cli_parser
from support_graph.cli.utils import (
    format_duration,
    index_missing_lines,
    index_unavailable_lines,
    load_json,
    load_jsonl,
    missing_config_lines,
    print_lines,
    relative_path,
    run_async_boundary,
    run_output_dir,
    shorten,
    write_json,
    eval_run_is_complete,
)
from support_graph.artifacts import build_standalone_run_id, standalone_run_artifacts
from support_graph.config.runtime import ConfigValidationError, RuntimeConfig
from support_graph.config.settings import Settings
from support_graph.data.chunks import build_chunks, write_chunks_jsonl
from support_graph.data.documents import load_documents
from support_graph.data.eval_subsets import build_subset, write_subset_jsonl
from support_graph.data.eval_subsets import load_subset_jsonl
from support_graph.data.kubernetes import fetch_kubernetes_docs
from support_graph.evaluation.experiment import run_smoke10_experiment_async
from support_graph.evaluation.benchmark import (
    benchmark_embeddings,
    load_benchmark_chunk_records,
)
from support_graph.evaluation.authoring import (
    build_default_chat_draft,
    draft_examples,
    load_seed_topics,
    append_candidate_rows,
)
from support_graph.evaluation.eval_examples import (
    build_chunk_index,
    load_examples,
    promote_examples,
    validate_examples,
)
from support_graph.evaluation.evaluate import (
    build_eval_config,
    evaluate_split_async,
    load_eval_examples,
)
from support_graph.evaluation.variance import (
    run_variance_study_async,
    write_variance_report,
)
from support_graph.evaluation.model_ab import run_compatibility_gate_async
from support_graph.logging_utils import configure_logging, get_logger
from support_graph.providers import (
    chat_provider as resolved_chat_provider,
    embedding_provider as resolved_embedding_provider,
)
from support_graph.retrieval.index import (
    collection_row_count,
    index_documents,
    load_chunk_records,
)
from support_graph.runtime.graph import run_graph_async
from support_graph.runtime.traces import load_trace_events, summarize_trace_events
from support_graph.types import DomainLike


logger = get_logger(__name__)


def _load_settings(args: argparse.Namespace) -> Settings:
    """Load settings from TOML files specified in CLI arguments."""
    return Settings.load(args.config_file, args.secrets_file)


def _run_config(settings: Settings, domain: DomainLike) -> RuntimeConfig:
    """Get the runtime configuration for a specific domain."""
    return settings.runtime_for(domain)


def _print_missing_config(
    *,
    title: str,
    settings: Settings,
    missing: list[str],
) -> int:
    """Print a standardized error message for missing configuration fields."""
    print_lines(
        missing_config_lines(
            title=title,
            settings=settings,
            missing=missing,
        )
    )
    return 1


def _print_config_validation_error(
    *,
    title: str,
    settings: Settings,
    error: ConfigValidationError,
) -> int:
    """Print a standardized error message for a configuration validation error."""
    return _print_missing_config(
        title=title,
        settings=settings,
        missing=list(error.missing_fields),
    )


def _ensure_chunk_artifact(
    *,
    settings: Settings,
    domain: DomainLike,
    chunk_file: str | None,
    max_tokens_per_chunk: int,
    reason: str,
) -> Path:
    """Ensure a chunk artifact exists, creating it on-demand if missing.

    A chunk artifact is a JSONL file where each line represents a piece of a
    source document, optimized for retrieval. It serves as the canonical source
    for indexing, benchmarking, and retrieval operations.

    - Checks for a pre-existing chunk file at the conventional path.
    - If found, returns the path immediately.
    - If missing, it triggers a full build pipeline: loads documents, builds chunks,
      and writes the new artifact to disk.

    Args:
        settings: The application settings.
        domain: The domain for which to ensure chunks.
        chunk_file: An optional explicit path to the chunk file.
        max_tokens_per_chunk: The token limit for chunking.
        reason: The reason for needing the chunks (for logging).

    Returns:
        The path to the existing or newly created chunk artifact.
    """
    chunk_artifact_path = (
        Path(chunk_file) if chunk_file else settings.chunk_artifact_path(domain)
    )
    if chunk_artifact_path.exists():
        return chunk_artifact_path

    logger.info(
        "Chunk artifact missing for %s domain=%s, building fresh chunks at %s",
        reason,
        domain,
        chunk_artifact_path,
    )
    documents = load_documents(settings.dataset.root, domains=[domain])
    chunks = build_chunks(
        documents,
        max_tokens_per_chunk=max_tokens_per_chunk,
        domains=[domain],
    )
    write_chunks_jsonl(chunks, chunk_artifact_path)
    return chunk_artifact_path


def _checked_collection_row_count(
    postgres_dsn: str,
    collection_name: str,
) -> tuple[int | None, str | None]:
    """Safely get the row count of a vector collection, handling exceptions."""
    try:
        return collection_row_count(postgres_dsn, collection_name), None
    except Exception as exc:
        logger.warning(
            "Failed to inspect pgvector row count for %s: %s",
            collection_name,
            exc,
            exc_info=exc,
        )
        return None, str(exc)


def _index_preflight_error_lines(
    *,
    title: str,
    postgres_dsn: str,
    collection_name: str,
    domain: DomainLike,
) -> list[str] | None:
    """Check if the vector index is available and populated, returning error lines if not."""
    row_count, row_count_error = _checked_collection_row_count(
        postgres_dsn,
        collection_name,
    )
    if row_count is None:
        return index_unavailable_lines(
            title=title,
            collection_name=collection_name,
            error=row_count_error or "Unknown pgvector inspection error.",
        )
    if row_count == 0:
        return index_missing_lines(
            title=title,
            collection_name=collection_name,
            domain=str(domain),
        )
    return None


def _ensure_index_ready(
    *,
    title: str,
    postgres_dsn: str,
    collection_name: str,
    domain: DomainLike,
) -> bool:
    """Run index pre-flight checks and print error messages if checks fail."""
    error_lines = _index_preflight_error_lines(
        title=title,
        postgres_dsn=postgres_dsn,
        collection_name=collection_name,
        domain=domain,
    )
    if error_lines is None:
        return True
    print_lines(error_lines)
    return False


def _required_eval_run_dir(
    *,
    title: str,
    settings: Settings,
    run_id: str,
) -> Path | None:
    """Check for a complete eval run directory, printing an error if missing."""
    output_dir = run_output_dir(settings, run_id)
    if not output_dir.exists():
        print_lines(
            [
                title,
                "State: run-missing",
                "Run missing",
                f"No eval run found at {relative_path(output_dir, settings.paths.project_root)}.",
            ]
        )
        return None
    if not eval_run_is_complete(output_dir):
        print_lines(
            [
                title,
                "State: artifacts-missing",
                "Artifacts missing",
                f"Run {run_id} is incomplete under {relative_path(output_dir, settings.paths.project_root)}.",
                "Next",
                "Run eval again to regenerate the full eval artifact set.",
            ]
        )
        return None
    return output_dir


def _answer_example_count(examples: list[dict]) -> int:
    """Count the number of examples with a target mode of 'answer'."""
    return sum(1 for example in examples if example.get("target_mode") == "answer")


def load_example_record(
    example_id: str,
    settings: Settings,
    domain: DomainLike | None = None,
) -> dict:
    """Load a single example record by ID from committed/derived artifacts."""
    candidate_paths = []
    if domain is not None:
        candidate_paths.append(
            settings.paths.examples_dir / f"{domain}_validation.jsonl"
        )
        candidate_paths.extend(
            sorted(
                (settings.paths.project_root / "data/eval_subsets" / str(domain)).glob(
                    "*.jsonl"
                )
            )
        )
    candidate_paths.extend(sorted(settings.paths.examples_dir.glob("*.jsonl")))
    candidate_paths.extend(
        sorted((settings.paths.project_root / "data/eval_subsets").glob("*.jsonl"))
    )
    candidate_paths.extend(
        sorted((settings.paths.project_root / "data/eval_subsets").glob("*/*.jsonl"))
    )
    existing_paths = [path for path in candidate_paths if path.exists()]
    for path in existing_paths:
        for record in load_subset_jsonl(path):
            if record["example_id"] == example_id:
                return record
    raise FileNotFoundError(f"Example not found: {example_id}")


def _log_graph_event(event: dict) -> None:
    """Log a graph execution event to the console with structured formatting."""
    kind = str(event.get("kind") or "")
    run_id = str(event.get("run_id") or "unknown-run")
    example_id = str(event.get("example_id") or "unknown-example")
    match kind:
        case "query_ready":
            logger.info(
                "Run %s prepared query for %s: %s",
                run_id,
                example_id,
                shorten(str(event.get("query") or ""), limit=96),
            )
        case "query_refined":
            logger.info(
                "Run %s refined query for %s: %s",
                run_id,
                example_id,
                shorten(str(event.get("refined_query") or ""), limit=96),
            )
        case "retrieval_complete":
            logger.info(
                "Run %s retrieval attempt %s for %s returned %s ranked / %s expanded chunks",
                run_id,
                int(event.get("retrieval_attempts") or 0),
                example_id,
                len(event.get("retrieval_ranked_chunks") or []),
                len(event.get("retrieved_chunks") or []),
            )
        case "evidence_graded":
            grade = event.get("evidence_grade") or {}
            logger.info(
                "Run %s evidence verdict for %s: %s",
                run_id,
                example_id,
                grade.get("verdict", "unknown"),
            )
        case "fallback":
            fallback = event.get("fallback") or {}
            logger.warning(
                "Run %s fallback in %s for %s: %s: %s",
                run_id,
                fallback.get("node") or event.get("node") or "unknown-node",
                example_id,
                fallback.get("exception_type", "UnknownError"),
                fallback.get("error", "unknown fallback"),
            )
        case "response_completed":
            logger.info(
                "Run %s completed for %s with decision=%s citations=%s",
                run_id,
                example_id,
                event.get("decision"),
                len(event.get("citations") or []),
            )
        case "error":
            logger.error(
                "Run %s failed for %s: %s: %s",
                run_id,
                example_id,
                event.get("exception_type", "UnknownError"),
                event.get("error", "unknown error"),
            )


def _build_chunks(args: argparse.Namespace) -> int:
    """CLI handler for the 'build-chunks' command."""
    settings = _load_settings(args)
    domain = settings.selected_domain(args.domain)
    logger.info("Building chunks for domain=%s", domain)
    documents = load_documents(settings.dataset.root, domains=[domain])
    logger.info("Loaded %s documents for domain=%s", len(documents), domain)
    chunks = build_chunks(
        documents,
        max_tokens_per_chunk=args.max_tokens_per_chunk,
        domains=[domain],
    )
    output_path = (
        Path(args.output)
        if args.output
        else settings.paths.chunks_dir / f"{domain}.jsonl"
    )
    logger.info("Writing %s chunks to %s", len(chunks), output_path)
    write_chunks_jsonl(chunks, output_path)
    print_lines(
        [
            "SupportGraph Build Chunks",
            f"Domain: {domain}",
            f"Documents: {len(documents)}",
            f"Chunks: {len(chunks)}",
            f"Artifact: {output_path}",
        ]
    )
    return 0


def _fetch_kubernetes_docs(args: argparse.Namespace) -> int:
    settings = _load_settings(args)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = settings.paths.project_root / output_path
    manifest = fetch_kubernetes_docs(
        output_path,
        ref=args.ref,
        replace=args.replace,
    )
    print_lines(
        [
            "SupportGraph Fetch Kubernetes Docs",
            f"Ref: {manifest['requested_ref']}",
            f"Resolved SHA: {manifest['resolved_sha']}",
            f"Docs Path: {manifest['docs_path']}",
            f"Artifact: {output_path}",
        ]
    )
    return 0


def _build_subsets(args: argparse.Namespace) -> int:
    """CLI handler for building deterministic subsets from an examples JSONL."""
    examples = load_jsonl(Path(args.examples_file))
    smoke_examples = build_subset(
        examples,
        size=args.smoke_size,
        target_mode="answer",
        salt="smoke",
    )
    frozen_examples = build_subset(
        examples,
        size=args.frozen_size,
        target_mode="answer",
        salt="frozen_experiment",
    )
    output_dir = Path(args.output_dir) if args.output_dir else Path("data/eval_subsets")
    smoke_path = output_dir / "smoke.jsonl"
    frozen_path = output_dir / "frozen_experiment.jsonl"
    logger.info(
        "Writing subset artifacts to %s (smoke=%s, frozen_experiment=%s)",
        output_dir,
        len(smoke_examples),
        len(frozen_examples),
    )
    write_subset_jsonl(smoke_examples, smoke_path)
    write_subset_jsonl(frozen_examples, frozen_path)
    print_lines(
        [
            "SupportGraph Build Subsets",
            f"Examples File: {args.examples_file}",
            f"Answer Examples: {_answer_example_count(examples)}",
            f"Smoke: {len(smoke_examples)} -> {smoke_path}",
            f"Frozen Experiment: {len(frozen_examples)} -> {frozen_path}",
        ]
    )
    return 0


def _index_docs(args: argparse.Namespace) -> int:
    """CLI handler for the 'index' command."""
    settings = _load_settings(args)
    domain = settings.selected_domain(args.domain)
    try:
        settings.runtime.validate_for_index()
    except ConfigValidationError as exc:
        return _print_config_validation_error(
            title="SupportGraph Index Docs",
            settings=settings,
            error=exc,
        )

    chunk_artifact_path = _ensure_chunk_artifact(
        settings=settings,
        domain=domain,
        chunk_file=args.chunk_file,
        max_tokens_per_chunk=args.max_tokens_per_chunk,
        reason="indexing",
    )
    logger.info("Loading chunk records from %s", chunk_artifact_path)
    chunk_records = load_chunk_records(chunk_artifact_path)
    config = replace(
        settings.runtime_for(domain),
        chunk_artifact_path=chunk_artifact_path,
    )
    logger.info(
        "Indexing %s chunks into collection=%s batch_size=%s recreate=%s",
        len(chunk_records),
        config.collection_name,
        args.batch_size,
        bool(args.recreate),
    )
    index_documents(
        config,
        chunk_records=chunk_records,
        pre_delete_collection=args.recreate,
        batch_size=args.batch_size,
    )

    print_lines(
        [
            "SupportGraph Index Docs",
            f"Domain: {domain}",
            f"Collection: {config.collection_name}",
            f"Chunks Indexed: {len(chunk_records)}",
            f"Artifact: {chunk_artifact_path}",
        ]
    )
    return 0


def _benchmark_embeddings(args: argparse.Namespace) -> int:
    """CLI handler for the 'benchmark-embeddings' command."""
    settings = _load_settings(args)
    domain = settings.selected_domain(args.domain)
    try:
        settings.runtime.validate_for_index()
    except ConfigValidationError as exc:
        return _print_config_validation_error(
            title="SupportGraph Benchmark Embeddings",
            settings=settings,
            error=exc,
        )

    chunk_artifact_path = _ensure_chunk_artifact(
        settings=settings,
        domain=domain,
        chunk_file=args.chunk_file,
        max_tokens_per_chunk=args.max_tokens_per_chunk,
        reason="benchmarking",
    )
    chunk_records = load_benchmark_chunk_records(str(chunk_artifact_path))
    config = settings.runtime_for(domain)
    logger.info(
        "Benchmarking embeddings for domain=%s sample_size=%s batch_size=%s",
        domain,
        args.sample_size,
        args.batch_size,
    )
    result = cast(
        dict,
        run_async_boundary(
            benchmark_embeddings(
                config,
                chunk_records=chunk_records,
                sample_size=args.sample_size,
                batch_size=args.batch_size,
                warmup=not args.skip_warmup,
            )
        ),
    )
    estimated_minutes = (
        result["estimated_total_seconds"] / 60
        if math.isfinite(result["estimated_total_seconds"])
        else float("inf")
    )
    print_lines(
        [
            "SupportGraph Benchmark Embeddings",
            f"Domain: {domain}",
            f"Model: {result['model']}",
            f"Sample: {result['sample_size']} / {result['total_chunks']} chunks",
            f"Batch Size: {result['batch_size']}",
            f"Warmup: {format_duration(result['warmup_seconds'])}",
            f"Measured Time: {format_duration(result['elapsed_seconds'])}",
            f"Throughput: {result['chunks_per_second']:.2f} chunks/s",
            f"Estimated Full Corpus: {format_duration(result['estimated_total_seconds'])} ({estimated_minutes:.1f} min)",
        ]
    )
    return 0


def _run_example(args: argparse.Namespace) -> int:
    """CLI handler for the 'run' command."""
    settings = _load_settings(args)
    try:
        settings.runtime.validate_for_run()
    except ConfigValidationError as exc:
        return _print_config_validation_error(
            title="SupportGraph Run",
            settings=settings,
            error=exc,
        )

    logger.info("Loading example %s", args.example_id)
    example = load_example_record(args.example_id, settings)
    domain = example.get("domain") or settings.selected_domain()
    run_config = _run_config(settings, domain)
    postgres_dsn = run_config.postgres_dsn
    if postgres_dsn is None:
        raise ValueError("Missing postgres_dsn for run.")
    if not _ensure_index_ready(
        title="SupportGraph Run",
        postgres_dsn=postgres_dsn,
        collection_name=run_config.collection_name,
        domain=domain,
    ):
        return 1

    logger.info(
        "Running graph for example=%s domain=%s collection=%s",
        example.get("example_id"),
        domain,
        run_config.collection_name,
    )
    created_at = datetime.now().astimezone().isoformat()
    run_id = build_standalone_run_id()
    artifacts = standalone_run_artifacts(settings.paths.project_root, run_id)
    artifacts.output_dir.mkdir(parents=True, exist_ok=True)
    result_payload = cast(
        dict,
        run_async_boundary(
            run_graph_async(
                example=example,
                config=run_config,
                run_id=run_id,
                trace_path=artifacts.trace,
                max_attempts=settings.runtime.max_retrieval_attempts,
                _event_sink=_log_graph_event,
            )
        ),
    )
    trace_summary = dict(result_payload.get("trace_summary", {}))
    trace_summary["trace_path"] = relative_path(
        artifacts.trace, settings.paths.project_root
    )
    result_payload = {**result_payload, "trace_summary": trace_summary}
    write_json(
        artifacts.manifest,
        {
            "run_id": run_id,
            "created_at": created_at,
            "example_id": example.get("example_id"),
            "domain": str(domain),
            "provider": {
                "type": str(resolved_chat_provider(settings.runtime)),
                "chat_model": settings.runtime.chat_model,
                "embedding_type": str(resolved_embedding_provider(settings.runtime)),
                "embedding_model": settings.runtime.embedding_model,
            },
            "prompt_version": run_config.prompt_version,
        },
    )
    write_json(artifacts.result, result_payload)
    print_lines(
        format_run_output(
            {
                **result_payload,
                "run_id": run_id,
                "artifact_paths": {
                    "manifest": artifacts.manifest,
                    "result": artifacts.result,
                    "trace": artifacts.trace,
                },
            },
            settings,
            verbose=args.verbose,
        )
    )
    return 0


def _eval_split(args: argparse.Namespace) -> int:
    """CLI handler for the 'eval' command."""
    settings = _load_settings(args)
    try:
        settings.runtime.validate_for_run()
    except ConfigValidationError as exc:
        return _print_config_validation_error(
            title="SupportGraph Eval",
            settings=settings,
            error=exc,
        )

    domain = settings.selected_domain(args.domain)
    postgres_dsn = settings.runtime.postgres_dsn
    if postgres_dsn is None:
        raise ValueError("Missing postgres_dsn for evaluation.")
    if not _ensure_index_ready(
        title="SupportGraph Eval",
        postgres_dsn=postgres_dsn,
        collection_name=settings.collection_name(domain),
        domain=domain,
    ):
        return 1

    logger.info(
        "Running eval for domain=%s split=%s subset=%s limit=%s max_concurrency=%s",
        domain,
        args.split,
        args.subset,
        args.limit if args.limit is not None else "all",
        args.max_concurrency,
    )
    result = cast(
        dict,
        run_async_boundary(
            evaluate_split_async(
                settings=settings,
                domain=domain,
                split=args.split,
                subset=args.subset,
                limit=args.limit,
                notes=args.notes,
                max_concurrency=args.max_concurrency,
            )
        ),
    )
    print_lines(format_eval_output(result, settings))
    return 0


def _experiment_smoke10(args: argparse.Namespace) -> int:
    """CLI handler for the 'experiment-smoke10' command."""
    settings = _load_settings(args)
    try:
        settings.runtime.validate_for_run()
    except ConfigValidationError as exc:
        return _print_config_validation_error(
            title="SupportGraph Experiment",
            settings=settings,
            error=exc,
        )

    domain = settings.selected_domain(args.domain)
    postgres_dsn = settings.runtime.postgres_dsn
    if postgres_dsn is None:
        raise ValueError("Missing postgres_dsn for experiment.")
    if not _ensure_index_ready(
        title="SupportGraph Experiment",
        postgres_dsn=postgres_dsn,
        collection_name=settings.collection_name(domain),
        domain=domain,
    ):
        return 1

    result = cast(
        dict,
        run_async_boundary(
            run_smoke10_experiment_async(
                settings=settings,
                domain=domain,
                split=args.split,
                limit=args.limit,
            )
        ),
    )
    print_lines(format_experiment_output(result, settings))
    return 0


def _review_failures(args: argparse.Namespace) -> int:
    """CLI handler for the 'review-failures' command."""
    settings = _load_settings(args)
    output_dir = _required_eval_run_dir(
        title="SupportGraph Review Failures",
        settings=settings,
        run_id=args.run_id,
    )
    if output_dir is None:
        return 1

    failures = load_jsonl(output_dir / "failures.jsonl")
    filtered = failures
    if args.label is not None:
        filtered = [
            record for record in filtered if record.get("failure_label") == args.label
        ]
    if args.target_mode is not None:
        filtered = [
            record
            for record in filtered
            if record.get("target_mode") == args.target_mode
        ]

    print_lines(
        format_review_failures_output(
            run_id=args.run_id,
            output_dir=output_dir,
            failures=failures,
            filtered=filtered,
            limit=args.limit,
            settings=settings,
        )
    )
    return 0


def _trace_show(args: argparse.Namespace) -> int:
    """CLI handler for the 'trace-show' command."""
    settings = _load_settings(args)
    output_dir = _required_eval_run_dir(
        title="SupportGraph Trace Show",
        settings=settings,
        run_id=args.run_id,
    )
    if output_dir is None:
        return 1

    trace_index = load_json(output_dir / "trace_index.json")
    entry = next(
        (
            candidate
            for candidate in list(trace_index.get("entries", []))
            if candidate.get("example_id") == args.example_id
        ),
        None,
    )
    if entry is None:
        print_lines(
            [
                "SupportGraph Trace Show",
                "State: example-missing",
                "Example missing",
                f"No trace entry found for {args.example_id}.",
            ]
        )
        return 1

    trace_file = str(entry.get("trace_file", "")).strip()
    trace_path = output_dir / "traces" / trace_file if trace_file else Path()
    if not trace_file or not trace_path.exists() or not trace_path.is_file():
        print_lines(
            [
                "SupportGraph Trace Show",
                "State: trace-missing",
                "Trace missing",
                f"Trace file not found: {relative_path(trace_path, settings.paths.project_root)}.",
            ]
        )
        return 1

    trace_summary = summarize_trace_events(
        load_trace_events(trace_path),
        trace_path=relative_path(trace_path, settings.paths.project_root),
    )
    print_lines(
        format_trace_show_output(
            run_id=args.run_id,
            example_id=args.example_id,
            trace_summary=trace_summary,
            settings=settings,
        )
    )
    return 0


def _serve_ui(args: argparse.Namespace) -> int:
    """CLI handler for the 'ui' command."""
    from support_graph.ui import create_app, build_loader
    import uvicorn

    settings = _load_settings(args)
    uvicorn.run(
        create_app(loader=build_loader(settings)),
        host=args.host,
        port=args.port,
    )
    return 0


def _resolve_chunk_file(
    settings: Settings, domain: DomainLike, chunk_file: str | None
) -> Path:
    if chunk_file:
        return Path(chunk_file)
    return settings.paths.chunks_dir / f"{settings.selected_domain(domain)}.jsonl"


def _draft_eval_examples(args: argparse.Namespace) -> int:
    """CLI handler for the 'draft-eval-examples' command."""
    settings = _load_settings(args)
    try:
        settings.runtime.validate_for_run()
    except ConfigValidationError as exc:
        return _print_config_validation_error(
            title="SupportGraph Draft Eval Examples",
            settings=settings,
            error=exc,
        )
    domain = settings.selected_domain(args.domain)
    postgres_dsn = settings.runtime.postgres_dsn
    if postgres_dsn is None:
        raise ValueError("Missing postgres_dsn for drafting.")
    if not _ensure_index_ready(
        title="SupportGraph Draft Eval Examples",
        postgres_dsn=postgres_dsn,
        collection_name=settings.collection_name(domain),
        domain=domain,
    ):
        return 1

    seeds = load_seed_topics(args.seed_file)
    if not seeds:
        print_lines(
            [
                "SupportGraph Draft Eval Examples",
                "State: empty-seed-file",
                "No seed topics found.",
                f"Seed File: {args.seed_file}",
            ]
        )
        return 1
    output_path = (
        Path(args.output)
        if args.output
        else settings.paths.project_root
        / "data/eval_subsets"
        / str(domain)
        / "_candidates"
        / "expanded.candidates.jsonl"
    )
    run_config = _run_config(settings, domain)
    logger.info(
        "Drafting %s eval candidates for domain=%s output=%s",
        len(seeds),
        domain,
        output_path,
    )
    from support_graph.runtime.nodes import build_chat_model

    chat_model = build_chat_model(run_config)
    chat_draft = build_default_chat_draft(chat_model)
    results = cast(
        list,
        run_async_boundary(
            draft_examples(
                seeds,
                config=run_config,
                chat_draft=chat_draft,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
            )
        ),
    )
    appended = append_candidate_rows(results, output_path)
    print_lines(
        [
            "SupportGraph Draft Eval Examples",
            f"Domain: {domain}",
            f"Seeds: {len(seeds)}",
            f"Drafted: {len(results)}",
            f"Candidates File: {output_path}",
            f"Total Candidates (after append): {len(appended)}",
            "Next",
            f"Review and edit {relative_path(output_path, settings.paths.project_root)}, "
            "then run validate-eval-examples and promote-eval-examples.",
        ]
    )
    return 0


def _validate_eval_examples(args: argparse.Namespace) -> int:
    """CLI handler for the 'validate-eval-examples' command."""
    settings = _load_settings(args)
    domain = settings.selected_domain(args.domain)
    chunk_path = _resolve_chunk_file(settings, domain, args.chunk_file)
    if not chunk_path.exists():
        print_lines(
            [
                "SupportGraph Validate Eval Examples",
                "State: chunk-file-missing",
                "Chunk file missing",
                f"No chunk corpus found at {relative_path(chunk_path, settings.paths.project_root)}.",
                "Next",
                f"Run: uv run grounded-support-rag build-chunks --domain {domain}",
            ]
        )
        return 1
    examples = load_examples(args.examples_file)
    chunk_records = load_subset_jsonl(chunk_path)
    chunk_index = build_chunk_index(chunk_records)
    report = validate_examples(
        examples, chunk_index, require_rag_fields=args.require_rag_fields
    )
    lines = [
        "SupportGraph Validate Eval Examples",
        f"Examples: {report.example_count}",
        f"Errors: {len(report.errors)}",
        f"Warnings: {len(report.warnings)}",
        f"Duplicate IDs: {', '.join(report.duplicate_ids) or 'none'}",
        "Answer Type Distribution",
    ]
    for answer_type, count in report.answer_type_distribution.items():
        lines.append(f"- {answer_type}: {count}")
    if report.issues:
        lines.append("Issues")
        lines.extend(report.issue_lines())
    lines.append("State: valid" if report.ok else "State: invalid")
    print_lines(lines)
    return 0 if report.ok else 1


def _promote_eval_examples(args: argparse.Namespace) -> int:
    """CLI handler for the 'promote-eval-examples' command."""
    settings = _load_settings(args)
    domain = settings.selected_domain(args.domain)
    chunk_path = _resolve_chunk_file(settings, domain, args.chunk_file)
    if not chunk_path.exists():
        print_lines(
            [
                "SupportGraph Promote Eval Examples",
                "State: chunk-file-missing",
                "Chunk file missing",
                f"No chunk corpus found at {relative_path(chunk_path, settings.paths.project_root)}.",
            ]
        )
        return 1
    candidates = load_examples(args.candidates_file)
    chunk_records = load_subset_jsonl(chunk_path)
    chunk_index = build_chunk_index(chunk_records)
    report = validate_examples(candidates, chunk_index)
    if not report.ok:
        print_lines(
            [
                "SupportGraph Promote Eval Examples",
                "State: validation-failed",
                "Promotion refused: candidate file has validation errors.",
                *report.issue_lines(),
                "Next",
                "Fix the errors in the candidates file, then re-run promote-eval-examples.",
            ]
        )
        return 1
    target_path = (
        Path(args.target_file)
        if args.target_file
        else settings.paths.project_root
        / "data/eval_subsets"
        / str(domain)
        / "expanded.jsonl"
    )
    existing = load_examples(target_path) if target_path.exists() else []
    merged = promote_examples(candidates, target_path=target_path, existing=existing)
    print_lines(
        [
            "SupportGraph Promote Eval Examples",
            f"Domain: {domain}",
            f"Candidates: {len(candidates)}",
            f"Existing: {len(existing)}",
            f"Merged: {len(merged)}",
            f"Target File: {relative_path(target_path, settings.paths.project_root)}",
            "Next",
            f"Run: uv run grounded-support-rag eval --subset expanded --domain {domain}",
        ]
    )
    return 0


def _eval_variance(args: argparse.Namespace) -> int:
    """CLI handler for the 'eval-variance' command."""
    settings = _load_settings(args)
    try:
        settings.runtime.validate_for_run()
    except ConfigValidationError as exc:
        return _print_config_validation_error(
            title="SupportGraph Eval Variance",
            settings=settings,
            error=exc,
        )
    domain = settings.selected_domain(args.domain)
    postgres_dsn = settings.runtime.postgres_dsn
    if postgres_dsn is None:
        raise ValueError("Missing postgres_dsn for variance study.")
    if not _ensure_index_ready(
        title="SupportGraph Eval Variance",
        postgres_dsn=postgres_dsn,
        collection_name=settings.collection_name(domain),
        domain=domain,
    ):
        return 1

    examples, subset_name = load_eval_examples(
        settings, domain, args.split, args.subset
    )
    if not examples:
        print_lines(
            [
                "SupportGraph Eval Variance",
                "State: empty-subset",
                "No examples found for subset.",
                f"Subset: {args.subset}",
            ]
        )
        return 1
    config = build_eval_config(settings, domain)
    logger.info(
        "Running variance study domain=%s subset=%s examples=%s repeat=%s",
        domain,
        args.subset,
        len(examples),
        args.repeat,
    )
    report = cast(
        object,
        run_async_boundary(
            run_variance_study_async(
                settings=settings,
                domain=domain,
                split=args.split,
                subset=subset_name,
                examples=examples,
                config=config,
                repeat=args.repeat,
                notes=args.notes,
                max_concurrency=args.max_concurrency,
                desired_half_width=args.desired_half_width,
            )
        ),
    )
    result = write_variance_report(
        settings=settings,
        domain=domain,
        subset=subset_name,
        repeat=args.repeat,
        report=report,  # type: ignore[arg-type]
    )
    variance_report = result["report"]  # type: ignore[index]
    lines = [
        "SupportGraph Eval Variance",
        f"Domain: {domain}",
        f"Subset: {subset_name}",
        f"Examples: {len(examples)}",
        f"Repeat (K): {args.repeat}",
        f"Retrieval Deterministic: {variance_report.retrieval_deterministic}",  # type: ignore[attr-defined]
        f"Recommended K: {variance_report.recommended_k}",  # type: ignore[attr-defined]
        f"Report: {relative_path(result['artifact_paths']['report'], settings.paths.project_root)}",  # type: ignore[index]
    ]
    if not variance_report.retrieval_deterministic:  # type: ignore[attr-defined]
        lines.append(
            "WARNING: retrieval was non-deterministic; attribution is invalid."
        )
    print_lines(lines)
    return 0 if variance_report.retrieval_deterministic else 1  # type: ignore[attr-defined]


def _model_ab_compatibility(args: argparse.Namespace) -> int:
    """CLI handler for the 'model-ab-compatibility' command."""
    settings = _load_settings(args)
    try:
        settings.runtime.validate_for_run()
    except ConfigValidationError as exc:
        return _print_config_validation_error(
            title="SupportGraph Model A/B Compatibility",
            settings=settings,
            error=exc,
        )
    domain = settings.selected_domain(args.domain)
    postgres_dsn = settings.runtime.postgres_dsn
    if postgres_dsn is None:
        raise ValueError("Missing postgres_dsn for compatibility gate.")
    if not _ensure_index_ready(
        title="SupportGraph Model A/B Compatibility",
        postgres_dsn=postgres_dsn,
        collection_name=settings.collection_name(domain),
        domain=domain,
    ):
        return 1

    examples, _ = load_eval_examples(settings, domain, args.split, args.subset)
    if not examples:
        print_lines(
            [
                "SupportGraph Model A/B Compatibility",
                "State: empty-subset",
                "No examples found for subset.",
                f"Subset: {args.subset}",
            ]
        )
        return 1
    base_config = build_eval_config(settings, domain)
    logger.info(
        "Running compatibility gate domain=%s candidate_chat_model=%s limit=%s",
        domain,
        args.candidate_chat_model,
        args.limit,
    )
    gate = cast(
        object,
        run_async_boundary(
            run_compatibility_gate_async(
                settings=settings,
                domain=domain,
                split=args.split,
                examples=examples,
                candidate_chat_model=args.candidate_chat_model,
                base_config=base_config,
                limit=args.limit,
                max_concurrency=args.max_concurrency,
                fallback_tolerance=args.fallback_tolerance,
            )
        ),
    )
    lines = [
        "SupportGraph Model A/B Compatibility",
        f"Domain: {domain}",
        f"Candidate Chat Model: {gate.chat_model}",  # type: ignore[attr-defined]
        f"Passed: {gate.passed}",  # type: ignore[attr-defined]
        f"Total Fallbacks: {gate.total_fallbacks}",  # type: ignore[attr-defined]
        f"Runtime Errors: {gate.runtime_error_count}",  # type: ignore[attr-defined]
        f"Fallback Nodes: {', '.join(gate.fallback_nodes) or 'none'}",  # type: ignore[attr-defined]
        f"Gate Run: {gate.run_id}",  # type: ignore[attr-defined]
        gate.reason,  # type: ignore[attr-defined]
    ]
    if not gate.passed:  # type: ignore[attr-defined]
        lines.extend(
            [
                "Next",
                (
                    "Do not A/B until json_schema structured output works. "
                    "Consider a tool_calling/json_object fallback method."
                ),
            ]
        )
    else:
        lines.extend(
            [
                "Next",
                (
                    "Run eval-variance per arm with --config-file overrides setting "
                    "runtime.chat_model for each arm; compare quality, variance, "
                    "latency, and cost."
                ),
            ]
        )
    print_lines(lines)
    return 0 if gate.passed else 1  # type: ignore[attr-defined]


def build_parser() -> argparse.ArgumentParser:
    return build_cli_parser(
        CliHandlers(
            fetch_kubernetes_docs=_fetch_kubernetes_docs,
            build_chunks=_build_chunks,
            build_subsets=_build_subsets,
            benchmark_embeddings=_benchmark_embeddings,
            index_docs=_index_docs,
            run_example=_run_example,
            eval_split=_eval_split,
            experiment_smoke10=_experiment_smoke10,
            review_failures=_review_failures,
            trace_show=_trace_show,
            serve_ui=_serve_ui,
            draft_eval_examples=_draft_eval_examples,
            validate_eval_examples=_validate_eval_examples,
            promote_eval_examples=_promote_eval_examples,
            eval_variance=_eval_variance,
            model_ab_compatibility=_model_ab_compatibility,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = _load_settings(args)
    log_path = configure_logging(
        level=settings.paths.log_level,
        log_dir=settings.paths.log_dir,
        command_name=args.command,
    )
    if log_path is not None:
        logger.info("Writing command logs to %s", log_path)
    return args.func(args)
