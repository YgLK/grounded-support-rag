"""CLI entrypoint for the SupportGraph MVP."""

from __future__ import annotations

import asyncio
import argparse
import inspect
import json
import math
from datetime import datetime
from pathlib import Path

from support_graph.artifacts import (
    EVAL_RUN_REQUIRED_FILES,
    build_standalone_run_id,
    standalone_run_artifacts,
)
from support_graph.config.runtime import (
    RuntimeConfig,
    build_runtime_config,
    with_runtime_config_overrides,
)
from support_graph.config.settings import Settings
from support_graph.data.chunks import build_chunks, write_chunks_jsonl
from support_graph.data.dataset import load_dialogues, load_documents
from support_graph.data.eval_subsets import build_subset, write_subset_jsonl
from support_graph.data.examples import build_turn_examples, write_examples_jsonl
from support_graph.data.examples import (
    load_example_record as load_example_record_from_paths,
)
from support_graph.evaluation.ablation import run_smoke10_ablation_async
from support_graph.evaluation.benchmark import (
    benchmark_embeddings,
    load_benchmark_chunk_records,
)
from support_graph.evaluation.evaluate import evaluate_split_async
from support_graph.retrieval.index import (
    collection_row_count,
    index_documents,
    load_chunk_records,
)
from support_graph.logging_utils import configure_logging, get_logger
from support_graph.runtime.graph import run_graph_async
from support_graph.runtime.traces import load_trace_events, summarize_trace_events
from support_graph.types import DatasetSplit, Domain, DomainLike, EvalSubset


logger = get_logger(__name__)
DOMAIN_CHOICES = [domain.value for domain in Domain]
SPLIT_CHOICES = [split.value for split in DatasetSplit]
EVAL_SUBSET_CHOICES = [subset.value for subset in EvalSubset]


def _print_lines(lines: list[str]) -> None:
    for line in lines:
        print(line)


def _format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {remaining_minutes}m {remaining_seconds}s"
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"


def _run_config(settings: Settings, domain: DomainLike) -> RuntimeConfig:
    return build_runtime_config(settings, domain)


def _relative_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run_output_dir(settings: Settings, run_id: str) -> Path:
    return settings.eval_runs_dir / run_id


def _eval_run_is_complete(output_dir: Path) -> bool:
    return all((output_dir / name).exists() for name in EVAL_RUN_REQUIRED_FILES)


def _shorten(text: str, limit: int = 88) -> str:
    cleaned = " ".join(str(text).split()).strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: max(0, limit - 3)].rstrip()}..."


def _run_async_boundary(value: object) -> object:
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


def _missing_config_lines(
    *,
    title: str,
    settings: Settings,
    missing: list[str],
) -> list[str]:
    return [
        title,
        "State: missing-config",
        "Missing config",
        ", ".join(missing),
        "Next",
        f"Inspect {settings.dotenv_path} or copy .env.example to .env.",
    ]


def _index_missing_lines(
    *,
    title: str,
    collection_name: str,
    domain: DomainLike,
) -> list[str]:
    return [
        title,
        "State: index-missing",
        "Index missing",
        f"No indexed rows found for collection {collection_name}.",
        "Next",
        f"Run: uv run support-graph index-docs --domain {domain}",
    ]


def _metric_text(
    value: float | None,
    *,
    retrieval_top_k: int | None = None,
    metric_k: int | None = None,
) -> str:
    if value is not None:
        return f"{value:.3f}"
    if (
        metric_k is not None
        and retrieval_top_k is not None
        and retrieval_top_k < metric_k
    ):
        return f"n/a (retrieval_top_k={retrieval_top_k})"
    return "n/a"


def _index_preflight_error_lines(
    *,
    title: str,
    postgres_dsn: str,
    collection_name: str,
    domain: DomainLike,
) -> list[str] | None:
    row_count, row_count_error = _checked_collection_row_count(
        postgres_dsn,
        collection_name,
    )
    if row_count is None:
        return _index_unavailable_lines(
            title=title,
            collection_name=collection_name,
            error=row_count_error or "Unknown pgvector inspection error.",
        )
    if row_count == 0:
        return _index_missing_lines(
            title=title,
            collection_name=collection_name,
            domain=domain,
        )
    return None


def load_example_record(
    example_id: str,
    settings: Settings,
    domain: DomainLike | None = None,
) -> dict:
    candidate_paths = []
    if domain is not None:
        candidate_paths.append(settings.examples_dir / f"{domain}_validation.jsonl")
    candidate_paths.extend(sorted(settings.examples_dir.glob("*.jsonl")))
    existing_paths = [path for path in candidate_paths if path.exists()]
    if not existing_paths:
        selected_domain = domain or settings.selected_domain()
        dialogues = load_dialogues(
            settings.dataset_root,
            split=DatasetSplit.VALIDATION,
            domains=[selected_domain],
        )
        examples = build_turn_examples(dialogues)
        output_path = settings.examples_dir / f"{selected_domain}_validation.jsonl"
        write_examples_jsonl(examples, output_path)
        existing_paths = [output_path]
    return load_example_record_from_paths(example_id, existing_paths)


def _run_next_lines(result: dict) -> list[str]:
    decision = result.get("decision")
    match decision:
        case "answer":
            return ["Grounded answer produced from retrieved DMV documentation."]
        case "clarify":
            return [
                "Ask one concrete missing-condition question.",
                "If needed, retry with --verbose to inspect the evidence path.",
            ]
        case "abstain":
            return [
                "No sufficient support was found for a safe answer.",
                "Retry with --verbose or inspect the saved trace under outputs/runs/.",
            ]
        case _:
            raise ValueError(f"Unknown decision: {decision}")


def _format_run_output(
    result: dict,
    settings: Settings,
    *,
    verbose: bool = False,
) -> list[str]:
    trace_summary = result.get("trace_summary", {})
    artifact_paths = result["artifact_paths"]
    lines = [
        "SupportGraph Run",
        f"Run: {result.get('run_id')}",
        f"Example: {result.get('example_id')}",
        "Context",
        f"User: {result.get('latest_user_utterance') or 'No recent user turn found.'}",
        "",
        f"Decision: {result.get('decision')}",
        "",
        "Response",
        result.get("response_text", ""),
        "",
        "Citations",
    ]
    citations = result.get("citations", [])
    if citations:
        for index, citation in enumerate(citations, start=1):
            span_ids = ",".join(citation.get("span_ids", []))
            lines.append(f"[{index}] {citation.get('doc_id')} :: spans {span_ids}")
    else:
        lines.append("None")

    lines.extend(
        [
            "",
            "Next",
            *_run_next_lines(result),
            "",
            "Trace",
            f"Attempts: {trace_summary.get('retrieval_attempts', 0)}",
            f"Path: {' -> '.join(trace_summary.get('graph_path', []))}",
            f"Fallbacks: {trace_summary.get('fallback_count', 0)}",
        ]
    )
    if verbose:
        lines.extend(
            [
                "",
                "Verbose",
                f"Final Query: {trace_summary.get('final_query', '')}",
                f"Evidence Grade: {result.get('evidence_grade', {})}",
                "Retrieved Chunks",
            ]
        )
        for chunk in result.get("retrieved_chunks", [])[:5]:
            lines.append(f"- {chunk.get('chunk_id')} :: {chunk.get('text', '')[:160]}")
    lines.extend(
        [
            "",
            "Artifacts",
            _relative_path(Path(artifact_paths["manifest"]), settings.project_root),
            _relative_path(Path(artifact_paths["result"]), settings.project_root),
            _relative_path(Path(artifact_paths["trace"]), settings.project_root),
        ]
    )
    return lines


def _format_eval_output(result: dict, settings: Settings) -> list[str]:
    retrieval = result.get("metrics", {}).get("retrieval", {}).get("answer", {})
    generation = result.get("metrics", {}).get("generation", {}).get("answer", {})
    failure_counts = result.get("failure_counts", {})
    output_dir = Path(result.get("output_dir"))
    retrieval_top_k = result.get("retrieval_top_k", settings.retrieval_top_k)

    lines = [
        "SupportGraph Eval",
        f"Run: {result.get('run_id')}",
        f"Subset: {result.get('subset_label')}",
        "",
        "Headline Metrics",
        f"Doc Recall@3: {_metric_text(retrieval.get('doc_recall_at_3'), retrieval_top_k=retrieval_top_k, metric_k=3)}",
        f"Span Recall@5: {_metric_text(retrieval.get('span_recall_at_5'), retrieval_top_k=retrieval_top_k, metric_k=5)}",
        f"ROUGE-L: {_metric_text(generation.get('rouge_l'))}",
        f"F1: {_metric_text(generation.get('token_f1'))}",
        "",
        "Paper Reference",
        f"Recall@1: {_metric_text(retrieval.get('doc_recall_at_1'), retrieval_top_k=retrieval_top_k, metric_k=1)}",
        f"Recall@5: {_metric_text(retrieval.get('doc_recall_at_5'), retrieval_top_k=retrieval_top_k, metric_k=5)}",
        f"Recall@10: {_metric_text(retrieval.get('doc_recall_at_10'), retrieval_top_k=retrieval_top_k, metric_k=10)}",
        f"Exact Match: {_metric_text(generation.get('exact_match'))}",
        f"SacreBLEU: {_metric_text(generation.get('sacrebleu'))}",
        "",
        "Failure Snapshot",
    ]
    if failure_counts:
        for label, count in sorted(
            failure_counts.items(), key=lambda item: (-item[1], item[0])
        )[:3]:
            lines.append(f"{label}: {count}")
    else:
        lines.append("none: 0")

    lines.extend(
        [
            "",
            "Artifacts",
            _relative_path(output_dir / "summary.md", settings.project_root),
            _relative_path(output_dir / "failures.jsonl", settings.project_root),
            _relative_path(output_dir / "manual_review.csv", settings.project_root),
            _relative_path(
                output_dir / "retrieval_examples.jsonl", settings.project_root
            ),
            _relative_path(output_dir / "trace_index.json", settings.project_root),
        ]
    )
    return lines


def _format_ablation_output(result: dict, settings: Settings) -> list[str]:
    lines = [
        "SupportGraph Ablation",
        f"Scope: {result.get('domain')} smoke / first {result.get('limit')}",
        "",
        "Variants",
    ]
    for item in result.get("results", []):
        metrics = item.get("metrics", {})
        retrieval = metrics.get("retrieval", {}).get("answer", {})
        generation = metrics.get("generation", {}).get("answer", {})
        lines.append(
            "- {title}: {status} | Span Recall@5 {span:.3f} | Citation {citation:.3f} | E2E {e2e:.3f}".format(
                title=item.get("variant", {}).get("title"),
                status=item.get("status"),
                span=(retrieval.get("span_recall_at_5") or 0.0),
                citation=(generation.get("citation_coverage") or 0.0),
                e2e=(generation.get("end_to_end_success_rate") or 0.0),
            )
        )

    lines.extend(
        [
            "",
            "Recommendation",
            f"{result.get('recommendation')}: {result.get('recommendation_line')}",
            "",
            "Frozen-200",
        ]
    )
    if result.get("frozen_result") is not None:
        lines.append(f"ran: {result['frozen_result'].get('variant', {}).get('title')}")
    else:
        lines.append("skipped: Smoke-10 gate not met")

    lines.extend(
        [
            "",
            "Artifacts",
            _relative_path(
                Path(result["report_artifact_paths"]["report"]), settings.project_root
            ),
        ]
    )
    return lines


def _format_review_failures_output(
    *,
    run_id: str,
    output_dir: Path,
    failures: list[dict],
    filtered: list[dict],
    limit: int,
    settings: Settings,
) -> list[str]:
    failure_counts: dict[str, int] = {}
    for record in failures:
        label = str(record.get("failure_label") or "unknown")
        failure_counts[label] = failure_counts.get(label, 0) + 1

    lines = [
        "SupportGraph Review Failures",
        f"Run: {run_id}",
        "",
        "Failure Counts",
    ]
    if failure_counts:
        for label, count in sorted(
            failure_counts.items(), key=lambda item: (-item[1], item[0])
        )[:5]:
            lines.append(f"{label}: {count}")
    else:
        lines.append("none: 0")

    lines.extend(["", "Examples"])
    if filtered:
        for record in filtered[:limit]:
            lines.append(
                "{example_id} | {label} | {decision} | {need}".format(
                    example_id=record.get("example_id"),
                    label=record.get("failure_label") or "unknown",
                    decision=record.get("decision") or "unknown",
                    need=_shorten(record.get("latest_user_utterance", "")),
                )
            )
    else:
        lines.append("No matching failures.")

    lines.extend(
        [
            "",
            "Artifacts",
            _relative_path(output_dir / "failures.jsonl", settings.project_root),
            _relative_path(output_dir / "manual_review.csv", settings.project_root),
            _relative_path(
                output_dir / "retrieval_examples.jsonl", settings.project_root
            ),
        ]
    )
    return lines


def _format_trace_show_output(
    *,
    run_id: str,
    example_id: str,
    trace_summary: dict,
    settings: Settings,
) -> list[str]:
    node_latency_ms = trace_summary.get("node_latency_ms", {})
    fallbacks = trace_summary.get("fallbacks", [])
    observability = trace_summary.get("observability", {})
    observability_lines: list[str] = []
    if observability:
        otel = observability.get("opentelemetry", {})
        langsmith = observability.get("langsmith", {})
        if otel.get("enabled"):
            observability_lines.append(
                "OpenTelemetry: {service} via {exporter}".format(
                    service=otel.get("service_name", "support-graph"),
                    exporter=otel.get("exporter", "console"),
                )
            )
        if langsmith.get("enabled"):
            observability_lines.append(
                f"LangSmith: {langsmith.get('project', 'support-graph')}"
            )
    lines = [
        "SupportGraph Trace Show",
        f"Run: {run_id}",
        f"Example: {example_id}",
        "",
        "Trace",
        f"Final Query: {trace_summary.get('final_query', '')}",
        f"Attempts: {trace_summary.get('retrieval_attempts', 0)}",
        f"Path: {' -> '.join(trace_summary.get('graph_path', []))}",
        "",
        "Node Latency",
    ]
    if node_latency_ms:
        for node, latencies in node_latency_ms.items():
            formatted = ", ".join(f"{float(value):.2f} ms" for value in latencies)
            lines.append(f"{node}: {formatted}")
    else:
        lines.append("none")

    lines.extend(["", "Fallbacks"])
    if fallbacks:
        for fallback in fallbacks:
            lines.append(
                "{node}: {exception_type} :: {error}".format(
                    node=fallback.get("node", "unknown"),
                    exception_type=fallback.get("exception_type", "unknown"),
                    error=fallback.get("error", "unknown"),
                )
            )
    else:
        lines.append("none")

    lines.extend(["", "Observability"])
    lines.extend(observability_lines or ["none"])

    lines.extend(
        [
            "",
            "Evidence",
            f"Grade: {trace_summary.get('evidence_grade', {})}",
            "Counts: ranked {ranked} / expanded {expanded}".format(
                ranked=trace_summary.get("retrieval_ranked_count", 0),
                expanded=trace_summary.get("retrieved_count", 0),
            ),
            "",
            "Decision",
            str(trace_summary.get("decision", "")),
            "",
            "Artifact",
            _relative_path(
                Path(str(trace_summary.get("trace_path", ""))), settings.project_root
            ),
        ]
    )
    return lines


def _checked_collection_row_count(
    postgres_dsn: str,
    collection_name: str,
) -> tuple[int | None, str | None]:
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


def _index_unavailable_lines(
    *, title: str, collection_name: str, error: str
) -> list[str]:
    return [
        title,
        "State: index-unavailable",
        "Index unavailable",
        f"Could not inspect collection {collection_name}.",
        error,
        "Next",
        "Verify Postgres is reachable and the DSN is correct, then retry.",
    ]


def _log_graph_event(event: dict) -> None:
    kind = str(event.get("kind") or "")
    run_id = str(event.get("run_id") or "unknown-run")
    example_id = str(event.get("example_id") or "unknown-example")
    match kind:
        case "query_ready":
            logger.info(
                "Run %s prepared query for %s: %s",
                run_id,
                example_id,
                _shorten(str(event.get("query") or ""), limit=96),
            )
        case "query_refined":
            logger.info(
                "Run %s refined query for %s: %s",
                run_id,
                example_id,
                _shorten(str(event.get("refined_query") or ""), limit=96),
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
    settings = Settings.from_env(args.env_file)
    domain = settings.selected_domain(args.domain)
    logger.info("Building chunks for domain=%s", domain)
    documents = load_documents(settings.dataset_root, domains=[domain])
    logger.info("Loaded %s documents for domain=%s", len(documents), domain)
    chunks = build_chunks(
        documents, max_tokens_per_chunk=args.max_tokens_per_chunk, domains=[domain]
    )
    output_path = (
        Path(args.output) if args.output else settings.chunks_dir / f"{domain}.jsonl"
    )
    logger.info("Writing %s chunks to %s", len(chunks), output_path)
    write_chunks_jsonl(chunks, output_path)
    _print_lines(
        [
            "SupportGraph Build Chunks",
            f"Domain: {domain}",
            f"Documents: {len(documents)}",
            f"Chunks: {len(chunks)}",
            f"Artifact: {output_path}",
        ]
    )
    return 0


def _build_examples(args: argparse.Namespace) -> int:
    settings = Settings.from_env(args.env_file)
    domain = settings.selected_domain(args.domain)
    logger.info("Building examples for domain=%s split=%s", domain, args.split)
    dialogues = load_dialogues(
        settings.dataset_root, split=args.split, domains=[domain]
    )
    logger.info(
        "Loaded %s dialogues for domain=%s split=%s", len(dialogues), domain, args.split
    )
    examples = build_turn_examples(dialogues)
    output_path = (
        Path(args.output)
        if args.output
        else settings.examples_dir / f"{domain}_{args.split}.jsonl"
    )
    write_examples_jsonl(examples, output_path)
    logger.info("Wrote %s examples to %s", len(examples), output_path)
    _print_lines(
        [
            "SupportGraph Build Examples",
            f"Domain: {domain}",
            f"Split: {args.split}",
            f"Dialogues: {len(dialogues)}",
            f"Examples: {len(examples)}",
            f"Artifact: {output_path}",
        ]
    )
    return 0


def _build_subsets(args: argparse.Namespace) -> int:
    settings = Settings.from_env(args.env_file)
    domain = settings.selected_domain(args.domain)
    logger.info("Building eval subsets for domain=%s split=%s", domain, args.split)
    dialogues = load_dialogues(
        settings.dataset_root, split=args.split, domains=[domain]
    )
    examples = build_turn_examples(dialogues)
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
        salt="frozen_ablation",
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else settings.project_root / "data/eval_subsets"
    )
    smoke_path = output_dir / "smoke.jsonl"
    frozen_path = output_dir / "frozen_ablation.jsonl"
    logger.info(
        "Writing subset artifacts to %s (smoke=%s, frozen_ablation=%s)",
        output_dir,
        len(smoke_examples),
        len(frozen_examples),
    )
    write_subset_jsonl(smoke_examples, smoke_path)
    write_subset_jsonl(frozen_examples, frozen_path)
    _print_lines(
        [
            "SupportGraph Build Subsets",
            f"Domain: {domain}",
            f"Split: {args.split}",
            f"Answer Examples: {len([example for example in examples if example.get('target_mode') == 'answer'])}",
            f"Smoke: {len(smoke_examples)} -> {smoke_path}",
            f"Frozen Ablation: {len(frozen_examples)} -> {frozen_path}",
        ]
    )
    return 0


def _index_docs(args: argparse.Namespace) -> int:
    settings = Settings.from_env(args.env_file)
    domain = settings.selected_domain(args.domain)
    missing = settings.index_missing_fields()
    if missing:
        _print_lines(
            _missing_config_lines(
                title="SupportGraph Index Docs",
                settings=settings,
                missing=missing,
            )
        )
        return 1

    chunk_artifact_path = (
        Path(args.chunk_file)
        if args.chunk_file
        else settings.chunk_artifact_path(domain)
    )
    if not chunk_artifact_path.exists():
        logger.info(
            "Chunk artifact missing for domain=%s, building fresh chunks at %s",
            domain,
            chunk_artifact_path,
        )
        documents = load_documents(settings.dataset_root, domains=[domain])
        chunks = build_chunks(
            documents, max_tokens_per_chunk=args.max_tokens_per_chunk, domains=[domain]
        )
        write_chunks_jsonl(chunks, chunk_artifact_path)

    logger.info("Loading chunk records from %s", chunk_artifact_path)
    chunk_records = load_chunk_records(chunk_artifact_path)
    config = build_runtime_config(settings, domain)
    config = with_runtime_config_overrides(
        config, chunk_artifact_path=chunk_artifact_path
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

    _print_lines(
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
    settings = Settings.from_env(args.env_file)
    domain = settings.selected_domain(args.domain)
    missing = settings.index_missing_fields()
    if missing:
        _print_lines(
            _missing_config_lines(
                title="SupportGraph Benchmark Embeddings",
                settings=settings,
                missing=missing,
            )
        )
        return 1

    chunk_artifact_path = (
        Path(args.chunk_file)
        if args.chunk_file
        else settings.chunk_artifact_path(domain)
    )
    if not chunk_artifact_path.exists():
        logger.info(
            "Chunk artifact missing for benchmark, building fresh chunks at %s",
            chunk_artifact_path,
        )
        documents = load_documents(settings.dataset_root, domains=[domain])
        chunks = build_chunks(
            documents, max_tokens_per_chunk=args.max_tokens_per_chunk, domains=[domain]
        )
        write_chunks_jsonl(chunks, chunk_artifact_path)

    chunk_records = load_benchmark_chunk_records(str(chunk_artifact_path))
    config = build_runtime_config(settings, domain)
    logger.info(
        "Benchmarking embeddings for domain=%s sample_size=%s batch_size=%s",
        domain,
        args.sample_size,
        args.batch_size,
    )
    result = benchmark_embeddings(
        config,
        chunk_records=chunk_records,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        warmup=not args.skip_warmup,
    )
    estimated_minutes = (
        result["estimated_total_seconds"] / 60
        if math.isfinite(result["estimated_total_seconds"])
        else float("inf")
    )
    _print_lines(
        [
            "SupportGraph Benchmark Embeddings",
            f"Domain: {domain}",
            f"Model: {result['model']}",
            f"Sample: {result['sample_size']} / {result['total_chunks']} chunks",
            f"Batch Size: {result['batch_size']}",
            f"Warmup: {_format_duration(result['warmup_seconds'])}",
            f"Measured Time: {_format_duration(result['elapsed_seconds'])}",
            f"Throughput: {result['chunks_per_second']:.2f} chunks/s",
            f"Estimated Full Corpus: {_format_duration(result['estimated_total_seconds'])} ({estimated_minutes:.1f} min)",
        ]
    )
    return 0


def _run_example(args: argparse.Namespace) -> int:
    settings = Settings.from_env(args.env_file)
    missing = settings.runtime_missing_fields()
    if missing:
        _print_lines(
            _missing_config_lines(
                title="SupportGraph Run",
                settings=settings,
                missing=missing,
            )
        )
        return 1

    logger.info("Loading example %s", args.example_id)
    example = load_example_record(args.example_id, settings)
    domain = example.get("domain") or settings.selected_domain()
    run_config = _run_config(settings, domain)
    postgres_dsn = run_config.postgres_dsn
    if postgres_dsn is None:
        raise ValueError("Missing postgres_dsn for run.")
    error_lines = _index_preflight_error_lines(
        title="SupportGraph Run",
        postgres_dsn=postgres_dsn,
        collection_name=run_config.collection_name,
        domain=domain,
    )
    if error_lines is not None:
        _print_lines(error_lines)
        return 1

    logger.info(
        "Running graph for example=%s domain=%s collection=%s",
        example.get("example_id"),
        domain,
        run_config.collection_name,
    )
    created_at = datetime.now().astimezone().isoformat()
    run_id = build_standalone_run_id()
    artifacts = standalone_run_artifacts(settings.project_root, run_id)
    artifacts.output_dir.mkdir(parents=True, exist_ok=True)
    result_payload = _run_async_boundary(
        run_graph_async(
            example=example,
            config=run_config,
            run_id=run_id,
            trace_path=artifacts.trace,
            max_attempts=settings.max_retrieval_attempts,
            _event_sink=_log_graph_event,
        )
    )
    trace_summary = dict(result_payload.get("trace_summary", {}))
    trace_summary["trace_path"] = _relative_path(artifacts.trace, settings.project_root)
    result_payload = {**result_payload, "trace_summary": trace_summary}
    manifest = {
        "run_id": run_id,
        "created_at": created_at,
        "example_id": example.get("example_id"),
        "domain": str(domain),
        "provider": {
            "type": settings.provider_type,
            "chat_model": settings.chat_model,
            "embedding_type": settings.embedding_provider_type
            or settings.provider_type,
            "embedding_model": settings.embedding_model,
        },
        "prompt_version": run_config.prompt_version,
    }
    _write_json(artifacts.manifest, manifest)
    _write_json(artifacts.result, result_payload)
    result = {
        **result_payload,
        "run_id": run_id,
        "artifact_paths": {
            "manifest": artifacts.manifest,
            "result": artifacts.result,
            "trace": artifacts.trace,
        },
    }
    _print_lines(_format_run_output(result, settings, verbose=args.verbose))
    return 0


def _eval_split(args: argparse.Namespace) -> int:
    settings = Settings.from_env(args.env_file)
    missing = settings.runtime_missing_fields()
    if missing:
        _print_lines(
            _missing_config_lines(
                title="SupportGraph Eval",
                settings=settings,
                missing=missing,
            )
        )
        return 1

    domain = settings.selected_domain(args.domain)
    postgres_dsn = settings.postgres_dsn
    if postgres_dsn is None:
        raise ValueError("Missing postgres_dsn for evaluation.")
    error_lines = _index_preflight_error_lines(
        title="SupportGraph Eval",
        postgres_dsn=postgres_dsn,
        collection_name=settings.collection_name(domain),
        domain=domain,
    )
    if error_lines is not None:
        _print_lines(error_lines)
        return 1

    logger.info(
        "Running eval for domain=%s split=%s subset=%s limit=%s max_concurrency=%s",
        domain,
        args.split,
        args.subset,
        args.limit if args.limit is not None else "all",
        args.max_concurrency,
    )
    result = _run_async_boundary(
        evaluate_split_async(
            settings=settings,
            domain=domain,
            split=args.split,
            subset=args.subset,
            limit=args.limit,
            notes=args.notes,
            max_concurrency=args.max_concurrency,
        )
    )
    _print_lines(_format_eval_output(result, settings))
    return 0


def _ablate_smoke10(args: argparse.Namespace) -> int:
    settings = Settings.from_env(args.env_file)
    missing = settings.runtime_missing_fields()
    if missing:
        _print_lines(
            _missing_config_lines(
                title="SupportGraph Ablation",
                settings=settings,
                missing=missing,
            )
        )
        return 1

    domain = settings.selected_domain(args.domain)
    postgres_dsn = settings.postgres_dsn
    if postgres_dsn is None:
        raise ValueError("Missing postgres_dsn for ablation.")
    error_lines = _index_preflight_error_lines(
        title="SupportGraph Ablation",
        postgres_dsn=postgres_dsn,
        collection_name=settings.collection_name(domain),
        domain=domain,
    )
    if error_lines is not None:
        _print_lines(error_lines)
        return 1

    result = _run_async_boundary(
        run_smoke10_ablation_async(
            settings=settings,
            domain=domain,
            split=args.split,
            limit=args.limit,
        )
    )
    _print_lines(_format_ablation_output(result, settings))
    return 0


def _review_failures(args: argparse.Namespace) -> int:
    settings = Settings.from_env(args.env_file)
    output_dir = _run_output_dir(settings, args.run_id)
    failures_path = output_dir / "failures.jsonl"
    if not output_dir.exists():
        _print_lines(
            [
                "SupportGraph Review Failures",
                "State: run-missing",
                "Run missing",
                f"No eval run found at {_relative_path(output_dir, settings.project_root)}.",
            ]
        )
        return 1
    if not _eval_run_is_complete(output_dir):
        _print_lines(
            [
                "SupportGraph Review Failures",
                "State: artifacts-missing",
                "Artifacts missing",
                f"Run {args.run_id} is incomplete under {_relative_path(output_dir, settings.project_root)}.",
                "Next",
                "Run eval again to regenerate the full eval artifact set.",
            ]
        )
        return 1

    failures = _load_jsonl(failures_path)
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

    _print_lines(
        _format_review_failures_output(
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
    settings = Settings.from_env(args.env_file)
    output_dir = _run_output_dir(settings, args.run_id)
    trace_index_path = output_dir / "trace_index.json"
    if not output_dir.exists():
        _print_lines(
            [
                "SupportGraph Trace Show",
                "State: run-missing",
                "Run missing",
                f"No eval run found at {_relative_path(output_dir, settings.project_root)}.",
            ]
        )
        return 1
    if not _eval_run_is_complete(output_dir):
        _print_lines(
            [
                "SupportGraph Trace Show",
                "State: artifacts-missing",
                "Artifacts missing",
                f"Run {args.run_id} is incomplete under {_relative_path(output_dir, settings.project_root)}.",
                "Next",
                "Run eval again to regenerate the full eval artifact set.",
            ]
        )
        return 1

    trace_index = _load_json(trace_index_path)
    entries = list(trace_index.get("entries", []))
    entry = next(
        (
            candidate
            for candidate in entries
            if candidate.get("example_id") == args.example_id
        ),
        None,
    )
    if entry is None:
        _print_lines(
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
        _print_lines(
            [
                "SupportGraph Trace Show",
                "State: trace-missing",
                "Trace missing",
                f"Trace file not found: {_relative_path(trace_path, settings.project_root)}.",
            ]
        )
        return 1

    trace_summary = summarize_trace_events(
        load_trace_events(trace_path),
        trace_path=_relative_path(trace_path, settings.project_root),
    )
    _print_lines(
        _format_trace_show_output(
            run_id=args.run_id,
            example_id=args.example_id,
            trace_summary=trace_summary,
            settings=settings,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="support-graph")
    parser.add_argument(
        "--env-file", default=None, help="Optional path to a .env file."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_chunks_parser = subparsers.add_parser(
        "build-chunks", help="Build section-aware chunks."
    )
    build_chunks_parser.add_argument(
        "--domain",
        default=None,
        choices=DOMAIN_CHOICES,
        help="Domain to build. Defaults to the configured MVP domain.",
    )
    build_chunks_parser.add_argument("--max-tokens-per-chunk", type=int, default=512)
    build_chunks_parser.add_argument(
        "--output", default=None, help="Optional JSONL output path."
    )
    build_chunks_parser.set_defaults(func=_build_chunks)

    build_examples_parser = subparsers.add_parser(
        "build-examples", help="Build turn-level examples."
    )
    build_examples_parser.add_argument(
        "--domain",
        default=None,
        choices=DOMAIN_CHOICES,
        help="Domain to build. Defaults to the configured MVP domain.",
    )
    build_examples_parser.add_argument(
        "--split", default=DatasetSplit.VALIDATION, choices=SPLIT_CHOICES
    )
    build_examples_parser.add_argument(
        "--output", default=None, help="Optional JSONL output path."
    )
    build_examples_parser.set_defaults(func=_build_examples)

    build_subsets_parser = subparsers.add_parser(
        "build-subsets", help="Build deterministic eval subsets."
    )
    build_subsets_parser.add_argument(
        "--domain",
        default=None,
        choices=DOMAIN_CHOICES,
        help="Domain to build. Defaults to the configured MVP domain.",
    )
    build_subsets_parser.add_argument(
        "--split", default=DatasetSplit.VALIDATION, choices=SPLIT_CHOICES
    )
    build_subsets_parser.add_argument("--smoke-size", type=int, default=25)
    build_subsets_parser.add_argument("--frozen-size", type=int, default=200)
    build_subsets_parser.add_argument(
        "--output-dir", default=None, help="Optional output directory."
    )
    build_subsets_parser.set_defaults(func=_build_subsets)

    benchmark_embeddings_parser = subparsers.add_parser(
        "benchmark-embeddings", help="Benchmark local embedding throughput."
    )
    benchmark_embeddings_parser.add_argument(
        "--domain",
        default=None,
        choices=DOMAIN_CHOICES,
        help="Domain to benchmark. Defaults to the configured MVP domain.",
    )
    benchmark_embeddings_parser.add_argument(
        "--chunk-file", default=None, help="Optional chunk artifact path."
    )
    benchmark_embeddings_parser.add_argument(
        "--sample-size", type=int, default=100, help="Number of chunks to benchmark."
    )
    benchmark_embeddings_parser.add_argument(
        "--batch-size", type=int, default=1, help="Embedding batch size."
    )
    benchmark_embeddings_parser.add_argument(
        "--skip-warmup", action="store_true", help="Skip the one-chunk warmup request."
    )
    benchmark_embeddings_parser.add_argument(
        "--max-tokens-per-chunk", type=int, default=512
    )
    benchmark_embeddings_parser.set_defaults(func=_benchmark_embeddings)

    index_docs_parser = subparsers.add_parser(
        "index-docs", help="Index section-aware chunks into pgvector."
    )
    index_docs_parser.add_argument(
        "--domain",
        default=None,
        choices=DOMAIN_CHOICES,
        help="Domain to index. Defaults to the configured MVP domain.",
    )
    index_docs_parser.add_argument(
        "--chunk-file", default=None, help="Optional chunk artifact path."
    )
    index_docs_parser.add_argument("--max-tokens-per-chunk", type=int, default=512)
    index_docs_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Embedding/index batch size. Default 1 for local Ollama compatibility.",
    )
    index_docs_parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection before indexing.",
    )
    index_docs_parser.set_defaults(func=_index_docs)

    run_parser = subparsers.add_parser(
        "run", help="Run the retrieval-backed graph for one example."
    )
    run_parser.add_argument("--example-id", required=True)
    run_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Append query, evidence, and chunk details after the main answer view.",
    )
    run_parser.set_defaults(func=_run_example)

    eval_parser = subparsers.add_parser(
        "eval", help="Run the Phase 4 evaluation harness."
    )
    eval_parser.add_argument(
        "--split", default=DatasetSplit.VALIDATION, choices=SPLIT_CHOICES
    )
    eval_parser.add_argument("--domain", default=None, choices=DOMAIN_CHOICES)
    eval_parser.add_argument(
        "--subset",
        default=EvalSubset.SMOKE,
        choices=EVAL_SUBSET_CHOICES,
        help="Eval subset. Defaults to the developer-friendly smoke subset.",
    )
    eval_parser.add_argument(
        "--limit", type=int, default=None, help="Optional cap on evaluated examples."
    )
    eval_parser.add_argument(
        "--notes", default=None, help="Optional run note stored in the manifest."
    )
    eval_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum number of examples to evaluate concurrently.",
    )
    eval_parser.set_defaults(func=_eval_split)

    ablation_parser = subparsers.add_parser(
        "ablate-smoke10",
        help="Run the DMV Smoke-10 ablation variants and write a comparison note.",
    )
    ablation_parser.add_argument(
        "--split", default=DatasetSplit.VALIDATION, choices=SPLIT_CHOICES
    )
    ablation_parser.add_argument("--domain", default=None, choices=DOMAIN_CHOICES)
    ablation_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of leading smoke examples to compare. Default 10.",
    )
    ablation_parser.set_defaults(func=_ablate_smoke10)

    review_failures_parser = subparsers.add_parser(
        "review-failures",
        help="Inspect failure examples and review artifacts for one eval run.",
    )
    review_failures_parser.add_argument("--run-id", required=True)
    review_failures_parser.add_argument("--label", default=None)
    review_failures_parser.add_argument(
        "--target-mode",
        default=None,
        choices=["answer", "follow_up"],
    )
    review_failures_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of filtered examples to print.",
    )
    review_failures_parser.set_defaults(func=_review_failures)

    trace_show_parser = subparsers.add_parser(
        "trace-show",
        help="Inspect the raw trace for one evaluated example.",
    )
    trace_show_parser.add_argument("--run-id", required=True)
    trace_show_parser.add_argument("--example-id", required=True)
    trace_show_parser.set_defaults(func=_trace_show)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings.from_env(args.env_file)
    log_path = configure_logging(log_dir=settings.log_dir, command_name=args.command)
    if log_path is not None:
        logger.info("Writing command logs to %s", log_path)
    return args.func(args)
