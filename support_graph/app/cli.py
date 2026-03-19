"""CLI entrypoint for the SupportGraph MVP."""

from __future__ import annotations

import asyncio
import argparse
import inspect
import json
import math
from pathlib import Path

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


logger = get_logger(__name__)


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


def _run_config(settings: Settings, domain: str) -> RuntimeConfig:
    return build_runtime_config(settings, domain)


def _relative_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run_output_dir(settings: Settings, run_id: str) -> Path:
    return settings.eval_dir / run_id


def _shorten(text: str, limit: int = 88) -> str:
    cleaned = " ".join(str(text).split()).strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: max(0, limit - 3)].rstrip()}..."


def _run_async_boundary(value: object) -> object:
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


def load_example_record(
    example_id: str, settings: Settings, domain: str | None = None
) -> dict:
    candidate_paths = []
    if domain is not None:
        candidate_paths.append(settings.examples_dir / f"{domain}_validation.jsonl")
    candidate_paths.extend(sorted(settings.examples_dir.glob("*.jsonl")))
    existing_paths = [path for path in candidate_paths if path.exists()]
    if not existing_paths:
        selected_domain = domain or settings.selected_domain()
        dialogues = load_dialogues(
            settings.dataset_root, split="validation", domains=[selected_domain]
        )
        examples = build_turn_examples(dialogues)
        output_path = settings.examples_dir / f"{selected_domain}_validation.jsonl"
        write_examples_jsonl(examples, output_path)
        existing_paths = [output_path]
    return load_example_record_from_paths(example_id, existing_paths)


def _run_next_lines(result: dict) -> list[str]:
    decision = result.get("decision")
    if decision == "answer":
        return ["Grounded answer produced from retrieved DMV documentation."]
    if decision == "clarify":
        return [
            "Ask one concrete missing-condition question.",
            "If needed, retry with --verbose to inspect the evidence path.",
        ]
    return [
        "No sufficient support was found for a safe answer.",
        "Retry with --verbose or inspect outputs/traces/ for the retrieval path.",
    ]


def _format_run_output(result: dict, verbose: bool = False) -> list[str]:
    trace_summary = result.get("trace_summary", {})
    lines = [
        "SupportGraph Run",
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
    return lines


def _format_eval_output(result: dict, settings: Settings) -> list[str]:
    retrieval = result.get("metrics", {}).get("retrieval", {}).get("answer", {})
    generation = result.get("metrics", {}).get("generation", {}).get("answer", {})
    failure_counts = result.get("failure_counts", {})
    output_dir = Path(result.get("output_dir"))
    lines = [
        "SupportGraph Eval",
        f"Run: {result.get('run_id')}",
        f"Subset: {result.get('subset_label')}",
        "",
        "Headline Metrics",
        f"Doc Recall@3: {(retrieval.get('doc_recall_at_3') or 0.0):.3f}",
        f"Span Recall@5: {(retrieval.get('span_recall_at_5') or 0.0):.3f}",
        f"ROUGE-L: {(generation.get('rouge_l') or 0.0):.3f}",
        f"F1: {(generation.get('token_f1') or 0.0):.3f}",
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
            _relative_path(Path(result.get("summary_path")), settings.project_root),
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


def _print_pending_surface(name: str, settings: Settings, phase_label: str) -> int:
    missing = settings.runtime_missing_fields()
    lines = [f"SupportGraph {name}", f"State: {phase_label}"]
    if missing:
        lines.extend(
            [
                "Missing config",
                ", ".join(missing),
                "Next",
                f"Inspect {settings.dotenv_path} or copy .env.example to .env.",
            ]
        )
    else:
        lines.extend(
            [
                "Next",
                "Phase 1 data preparation is implemented.",
                "Phase 2+ runtime surfaces are scaffolded but not wired yet.",
            ]
        )
    _print_lines(lines)
    return 1


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


def _build_chunks(args: argparse.Namespace) -> int:
    settings = Settings.from_env(args.env_file)
    domain = settings.selected_domain(args.domain)
    documents = load_documents(settings.dataset_root, domains=[domain])
    chunks = build_chunks(
        documents, max_tokens_per_chunk=args.max_tokens_per_chunk, domains=[domain]
    )
    output_path = (
        Path(args.output) if args.output else settings.chunks_dir / f"{domain}.jsonl"
    )
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
    dialogues = load_dialogues(
        settings.dataset_root, split=args.split, domains=[domain]
    )
    examples = build_turn_examples(dialogues)
    output_path = (
        Path(args.output)
        if args.output
        else settings.examples_dir / f"{domain}_{args.split}.jsonl"
    )
    write_examples_jsonl(examples, output_path)
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
            [
                "SupportGraph Index Docs",
                "State: missing-config",
                "Missing config",
                ", ".join(missing),
                "Next",
                f"Inspect {settings.dotenv_path} or copy .env.example to .env.",
            ]
        )
        return 1

    chunk_artifact_path = (
        Path(args.chunk_file)
        if args.chunk_file
        else settings.chunk_artifact_path(domain)
    )
    if not chunk_artifact_path.exists():
        documents = load_documents(settings.dataset_root, domains=[domain])
        chunks = build_chunks(
            documents, max_tokens_per_chunk=args.max_tokens_per_chunk, domains=[domain]
        )
        write_chunks_jsonl(chunks, chunk_artifact_path)

    chunk_records = load_chunk_records(chunk_artifact_path)
    config = build_runtime_config(settings, domain)
    config = with_runtime_config_overrides(
        config, chunk_artifact_path=chunk_artifact_path
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
            [
                "SupportGraph Benchmark Embeddings",
                "State: missing-config",
                "Missing config",
                ", ".join(missing),
                "Next",
                f"Inspect {settings.dotenv_path} or copy .env.example to .env.",
            ]
        )
        return 1

    chunk_artifact_path = (
        Path(args.chunk_file)
        if args.chunk_file
        else settings.chunk_artifact_path(domain)
    )
    if not chunk_artifact_path.exists():
        documents = load_documents(settings.dataset_root, domains=[domain])
        chunks = build_chunks(
            documents, max_tokens_per_chunk=args.max_tokens_per_chunk, domains=[domain]
        )
        write_chunks_jsonl(chunks, chunk_artifact_path)

    chunk_records = load_benchmark_chunk_records(str(chunk_artifact_path))
    config = build_runtime_config(settings, domain)
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
            [
                "SupportGraph Run",
                "State: missing-config",
                "Missing config",
                ", ".join(missing),
                "Next",
                f"Inspect {settings.dotenv_path} or copy .env.example to .env.",
            ]
        )
        return 1

    example = load_example_record(args.example_id, settings)
    domain = example.get("domain") or settings.selected_domain()
    run_config = _run_config(settings, domain)
    row_count, row_count_error = _checked_collection_row_count(
        run_config.postgres_dsn,
        run_config.collection_name,
    )
    if row_count is None:
        _print_lines(
            _index_unavailable_lines(
                title="SupportGraph Run",
                collection_name=run_config.collection_name,
                error=row_count_error or "Unknown pgvector inspection error.",
            )
        )
        return 1
    if row_count == 0:
        _print_lines(
            [
                "SupportGraph Run",
                "State: index-missing",
                "Index missing",
                f"No indexed rows found for collection {run_config.collection_name}.",
                "Next",
                f"Run: uv run support-graph index-docs --domain {domain}",
            ]
        )
        return 1

    result = _run_async_boundary(
        run_graph_async(
            example=example,
            config=run_config,
            max_attempts=settings.max_retrieval_attempts,
            trace_dir=settings.trace_dir,
        )
    )
    _print_lines(_format_run_output(result, verbose=args.verbose))
    return 0


def _eval_split(args: argparse.Namespace) -> int:
    settings = Settings.from_env(args.env_file)
    missing = settings.runtime_missing_fields()
    if missing:
        _print_lines(
            [
                "SupportGraph Eval",
                "State: missing-config",
                "Missing config",
                ", ".join(missing),
                "Next",
                f"Inspect {settings.dotenv_path} or copy .env.example to .env.",
            ]
        )
        return 1

    domain = settings.selected_domain(args.domain)
    row_count, row_count_error = _checked_collection_row_count(
        settings.postgres_dsn, settings.collection_name(domain)
    )
    if row_count is None:
        _print_lines(
            _index_unavailable_lines(
                title="SupportGraph Eval",
                collection_name=settings.collection_name(domain),
                error=row_count_error or "Unknown pgvector inspection error.",
            )
        )
        return 1
    if row_count == 0:
        _print_lines(
            [
                "SupportGraph Eval",
                "State: index-missing",
                "Index missing",
                f"No indexed rows found for collection {settings.collection_name(domain)}.",
                "Next",
                f"Run: uv run support-graph index-docs --domain {domain}",
            ]
        )
        return 1

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
            [
                "SupportGraph Ablation",
                "State: missing-config",
                "Missing config",
                ", ".join(missing),
                "Next",
                f"Inspect {settings.dotenv_path} or copy .env.example to .env.",
            ]
        )
        return 1

    domain = settings.selected_domain(args.domain)
    row_count, row_count_error = _checked_collection_row_count(
        settings.postgres_dsn, settings.collection_name(domain)
    )
    if row_count is None:
        _print_lines(
            _index_unavailable_lines(
                title="SupportGraph Ablation",
                collection_name=settings.collection_name(domain),
                error=row_count_error or "Unknown pgvector inspection error.",
            )
        )
        return 1
    if row_count == 0:
        _print_lines(
            [
                "SupportGraph Ablation",
                "State: index-missing",
                "Index missing",
                f"No indexed rows found for collection {settings.collection_name(domain)}.",
                "Next",
                f"Run: uv run support-graph index-docs --domain {domain}",
            ]
        )
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
    manual_review_path = output_dir / "manual_review.csv"
    retrieval_examples_path = output_dir / "retrieval_examples.jsonl"
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
    required_paths = [failures_path, manual_review_path, retrieval_examples_path]
    if any(not path.exists() for path in required_paths):
        _print_lines(
            [
                "SupportGraph Review Failures",
                "State: artifacts-missing",
                "Artifacts missing",
                f"Run {args.run_id} predates the analysis-tooling artifacts or is incomplete.",
                "Next",
                "Run eval again to generate failures.jsonl, manual_review.csv, and retrieval_examples.jsonl.",
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
    if not trace_index_path.exists():
        _print_lines(
            [
                "SupportGraph Trace Show",
                "State: artifacts-missing",
                "Artifacts missing",
                f"Run {args.run_id} predates trace_index.json or is incomplete.",
                "Next",
                "Run eval again to generate trace_index.json.",
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

    raw_trace_path = str(entry.get("trace_path", "")).strip()
    trace_path = Path(raw_trace_path) if raw_trace_path else Path()
    if not raw_trace_path or not trace_path.exists() or not trace_path.is_file():
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
        trace_path=trace_path,
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
        help="Domain to build. Defaults to the configured MVP domain.",
    )
    build_examples_parser.add_argument(
        "--split", default="validation", choices=["train", "validation", "test"]
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
        help="Domain to build. Defaults to the configured MVP domain.",
    )
    build_subsets_parser.add_argument(
        "--split", default="validation", choices=["train", "validation", "test"]
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
        "--split", default="validation", choices=["train", "validation", "test"]
    )
    eval_parser.add_argument("--domain", default=None)
    eval_parser.add_argument(
        "--subset",
        default="smoke",
        choices=["smoke", "frozen_ablation", "full_validation"],
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
        "--split", default="validation", choices=["train", "validation", "test"]
    )
    ablation_parser.add_argument("--domain", default=None)
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
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
