"""Argument parser construction for the SupportGraph CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

from support_graph.cli.utils import (
    DOMAIN_CHOICES,
    EVAL_SUBSET_CHOICES,
    SPLIT_CHOICES,
)
from support_graph.types import DatasetSplit, EvalSubset

CommandHandler = Callable[[argparse.Namespace], int]
Subparsers = argparse._SubParsersAction


@dataclass(frozen=True)
class CliHandlers:
    fetch_kubernetes_docs: CommandHandler
    build_chunks: CommandHandler
    build_subsets: CommandHandler
    benchmark_embeddings: CommandHandler
    index_docs: CommandHandler
    run_example: CommandHandler
    eval_split: CommandHandler
    experiment_smoke10: CommandHandler
    review_failures: CommandHandler
    trace_show: CommandHandler
    serve_ui: CommandHandler
    validate_eval_examples: CommandHandler
    promote_eval_examples: CommandHandler
    eval_variance: CommandHandler
    model_ab_compatibility: CommandHandler
    baseline_promote: CommandHandler
    baseline_export: CommandHandler
    doctor: CommandHandler


def _add_global_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config-file",
        default=None,
        help="Optional path to a settings TOML file.",
    )
    parser.add_argument(
        "--secrets-file",
        default=None,
        help="Optional path to a .env secrets file.",
    )


def _register_build_chunks(subparsers: Subparsers, handlers: CliHandlers) -> None:
    """Register the 'build-chunks' command.

    This command processes raw documents into smaller, section-aware chunks
    suitable for vector embedding and retrieval.
    """
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
    build_chunks_parser.set_defaults(func=handlers.build_chunks)


def _register_fetch_kubernetes_docs(
    subparsers: Subparsers, handlers: CliHandlers
) -> None:
    fetch_parser = subparsers.add_parser(
        "fetch-kubernetes-docs",
        help="Fetch a pinned Kubernetes website docs snapshot.",
    )
    fetch_parser.add_argument(
        "--ref",
        default="main",
        help="Git ref or SHA from kubernetes/website. Defaults to main.",
    )
    fetch_parser.add_argument(
        "--output",
        default="raw/kubernetes/current",
        help="Output corpus directory.",
    )
    fetch_parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace the output directory if it already exists.",
    )
    fetch_parser.set_defaults(func=handlers.fetch_kubernetes_docs)


def _register_build_subsets(subparsers: Subparsers, handlers: CliHandlers) -> None:
    """Register the 'build-subsets' command."""
    build_subsets_parser = subparsers.add_parser(
        "build-subsets", help="Build deterministic eval subsets."
    )
    build_subsets_parser.add_argument(
        "--examples-file",
        required=True,
        help="Input examples JSONL file.",
    )
    build_subsets_parser.add_argument("--smoke-size", type=int, default=25)
    build_subsets_parser.add_argument("--frozen-size", type=int, default=200)
    build_subsets_parser.add_argument(
        "--output-dir", default=None, help="Optional output directory."
    )
    build_subsets_parser.set_defaults(func=handlers.build_subsets)


def _register_benchmark_embeddings(
    subparsers: Subparsers, handlers: CliHandlers
) -> None:
    """Register the 'benchmark-embeddings' command.

    This command measures the throughput (chunks/sec) of the configured
    embedding model to help estimate the time required for a full indexing run.
    """
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
    benchmark_embeddings_parser.set_defaults(func=handlers.benchmark_embeddings)


def _register_index_docs(subparsers: Subparsers, handlers: CliHandlers) -> None:
    """Register the 'index-docs' command.

    This command populates the pgvector database with embeddings from a chunk
    artifact, making the documents searchable.
    """
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
    index_docs_parser.set_defaults(func=handlers.index_docs)


def _register_run(subparsers: Subparsers, handlers: CliHandlers) -> None:
    """Register the 'run' command.

    This command executes the full retrieval and generation graph for a single
    example ID, producing a grounded answer, clarification, or abstention.
    """
    run_parser = subparsers.add_parser(
        "run", help="Run the retrieval-backed graph for one example."
    )
    run_parser.add_argument("--example-id", required=True)
    run_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Append query, evidence, and chunk details after the main answer view.",
    )
    run_parser.set_defaults(func=handlers.run_example)


def _register_eval(subparsers: Subparsers, handlers: CliHandlers) -> None:
    """Register the 'eval' command.

    This command runs the full evaluation harness on a specified dataset split
    and subset, generating metrics for retrieval and generation.
    """
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
    eval_parser.set_defaults(func=handlers.eval_split)


def _register_baseline_promote(subparsers: Subparsers, handlers: CliHandlers) -> None:
    parser = subparsers.add_parser(
        "baseline-promote",
        help="Promote a completed LangSmith experiment as the dataset baseline.",
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--reason", required=True)
    parser.set_defaults(func=handlers.baseline_promote)


def _register_baseline_export(subparsers: Subparsers, handlers: CliHandlers) -> None:
    parser = subparsers.add_parser(
        "baseline-export",
        help="Export and verify one LangSmith experiment.",
    )
    parser.add_argument("--experiment-id", required=True)
    parser.set_defaults(func=handlers.baseline_export)


def _register_experiment(subparsers: Subparsers, handlers: CliHandlers) -> None:
    """Register the 'experiment-smoke10' command.

    This command runs a predefined experiment comparing multiple graph variants
    on the 'smoke' subset and recommends the best performer.
    """
    experiment_parser = subparsers.add_parser(
        "experiment-smoke10",
        help="Run Smoke-10 experiment variants and write a comparison note.",
    )
    experiment_parser.add_argument(
        "--split", default=DatasetSplit.VALIDATION, choices=SPLIT_CHOICES
    )
    experiment_parser.add_argument("--domain", default=None, choices=DOMAIN_CHOICES)
    experiment_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of leading smoke examples to compare. Default 10.",
    )
    experiment_parser.set_defaults(func=handlers.experiment_smoke10)


def _register_review_failures(subparsers: Subparsers, handlers: CliHandlers) -> None:
    """Register the 'review-failures' command.

    This command provides a summary of failure cases from a specific evaluation
    run, allowing for filtering by failure label and target mode.
    """
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
    review_failures_parser.set_defaults(func=handlers.review_failures)


def _register_trace_show(subparsers: Subparsers, handlers: CliHandlers) -> None:
    """Register the 'trace-show' command.

    This command displays a detailed execution trace for a single example from a
    given evaluation run, showing the graph path, latencies, and decision logic.
    """
    trace_show_parser = subparsers.add_parser(
        "trace-show",
        help="Inspect the raw trace for one evaluated example.",
    )
    trace_show_parser.add_argument("--run-id", required=True)
    trace_show_parser.add_argument("--example-id", required=True)
    trace_show_parser.set_defaults(func=handlers.trace_show)


def _register_ui(subparsers: Subparsers, handlers: CliHandlers) -> None:
    """Register the 'ui' command.

    This command serves the local SupportGraph Workbench UI, a web-based tool
    for interactively running and debugging the system.
    """
    ui_parser = subparsers.add_parser(
        "ui",
        help="Serve the local SupportGraph Workbench UI.",
    )
    ui_parser.add_argument("--host", default="127.0.0.1")
    ui_parser.add_argument("--port", type=int, default=8008)
    ui_parser.set_defaults(func=handlers.serve_ui)


def _register_validate_eval_examples(
    subparsers: Subparsers, handlers: CliHandlers
) -> None:
    """Register the 'validate-eval-examples' command.

    Validates a curated eval subset against the pinned chunk corpus: checks
    that declared doc/span IDs exist, required-point alias groups are grounded
    in source span text, no duplicate example_ids, and answer_type is valid.
    """
    parser = subparsers.add_parser(
        "validate-eval-examples",
        help="Validate a curated eval subset against the pinned chunk corpus.",
    )
    parser.add_argument(
        "--examples-file",
        required=True,
        help="JSONL examples file to validate (candidates or final subset).",
    )
    parser.add_argument(
        "--chunk-file",
        default=None,
        help=(
            "Chunk corpus JSONL to validate against. Defaults to "
            "data/derived/chunks/<domain>.jsonl."
        ),
    )
    parser.add_argument(
        "--domain",
        default=None,
        choices=DOMAIN_CHOICES,
        help="Domain. Defaults to the configured MVP domain.",
    )
    parser.add_argument(
        "--require-rag-fields",
        action="store_true",
        default=True,
        help="Require RAG eval fields (required_points, etc.). Default on.",
    )
    parser.set_defaults(func=handlers.validate_eval_examples)


def _register_promote_eval_examples(
    subparsers: Subparsers, handlers: CliHandlers
) -> None:
    """Register the 'promote-eval-examples' command.

    Merges verified candidate rows into the final expanded subset file,
    keeping existing rows and appending new example_ids. Runs the validator
    first and refuses to promote if errors are found.
    """
    parser = subparsers.add_parser(
        "promote-eval-examples",
        help="Promote verified eval candidates into the final expanded subset.",
    )
    parser.add_argument(
        "--candidates-file",
        required=True,
        help="JSONL candidates file (edited/verified by a human).",
    )
    parser.add_argument(
        "--target-file",
        default=None,
        help=(
            "Final subset JSONL to merge into. Defaults to "
            "data/eval_subsets/<domain>/expanded.jsonl."
        ),
    )
    parser.add_argument(
        "--domain",
        default=None,
        choices=DOMAIN_CHOICES,
        help="Domain. Defaults to the configured MVP domain.",
    )
    parser.add_argument(
        "--chunk-file",
        default=None,
        help="Chunk corpus JSONL for the promotion validation gate.",
    )
    parser.set_defaults(func=handlers.promote_eval_examples)


def _register_eval_variance(subparsers: Subparsers, handlers: CliHandlers) -> None:
    """Register the 'eval-variance' command.

    Runs the eval harness repeatedly on the same subset/config/index and
    produces a variance attribution report (sampling noise vs generation
    non-determinism), asserting retrieval is deterministic across runs.
    """
    parser = subparsers.add_parser(
        "eval-variance",
        help="Run a repeated-run variance attribution study.",
    )
    parser.add_argument(
        "--subset",
        default=EvalSubset.EXPANDED,
        choices=EVAL_SUBSET_CHOICES,
        help="Eval subset to repeat over. Defaults to expanded.",
    )
    parser.add_argument("--domain", default=None, choices=DOMAIN_CHOICES)
    parser.add_argument(
        "--split", default=DatasetSplit.VALIDATION, choices=SPLIT_CHOICES
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of repeated runs (K). Default 3 (pilot).",
    )
    parser.add_argument(
        "--desired-half-width",
        type=float,
        default=0.05,
        help="Target CI half-width for the recommended-K computation.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum number of examples to evaluate concurrently per run.",
    )
    parser.add_argument(
        "--notes", default=None, help="Optional run note stored in the manifest."
    )
    parser.set_defaults(func=handlers.eval_variance)


def _register_model_ab_compatibility(
    subparsers: Subparsers, handlers: CliHandlers
) -> None:
    """Register the 'model-ab-compatibility' command.

    Blocking prerequisite for a model A/B comparison: confirms a candidate
    chat model supports structured output (json_schema) by running a small
    smoke eval and asserting the fallback-event count is ~0.
    """
    parser = subparsers.add_parser(
        "model-ab-compatibility",
        help="Gate a candidate chat model on structured-output support.",
    )
    parser.add_argument(
        "--candidate-chat-model",
        required=True,
        help="Candidate OpenRouter chat model slug to gate.",
    )
    parser.add_argument("--domain", default=None, choices=DOMAIN_CHOICES)
    parser.add_argument(
        "--split", default=DatasetSplit.VALIDATION, choices=SPLIT_CHOICES
    )
    parser.add_argument(
        "--subset",
        default=EvalSubset.SMOKE,
        choices=EVAL_SUBSET_CHOICES,
        help="Subset to draw gate examples from. Defaults to smoke.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of examples to run for the gate. Default 3.",
    )
    parser.add_argument(
        "--fallback-tolerance",
        type=int,
        default=0,
        help="Max fallback events allowed to still pass. Default 0.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum number of examples to evaluate concurrently.",
    )
    parser.set_defaults(func=handlers.model_ab_compatibility)


def _register_doctor(subparsers: Subparsers, handlers: CliHandlers) -> None:
    """Register the read-only LangSmith doctor command."""
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check LangSmith connectivity, dataset, and baseline access.",
    )
    doctor_parser.add_argument(
        "--domain",
        default=None,
        choices=DOMAIN_CHOICES,
        help="Domain to check. Defaults to the configured MVP domain.",
    )
    doctor_parser.add_argument(
        "--subset",
        default=EvalSubset.SMOKE,
        choices=EVAL_SUBSET_CHOICES,
    )
    doctor_parser.set_defaults(func=handlers.doctor)


def register_subcommands(subparsers: Subparsers, handlers: CliHandlers) -> None:
    command_registrars: tuple[Callable[[Subparsers, CliHandlers], None], ...] = (
        _register_fetch_kubernetes_docs,
        _register_build_chunks,
        _register_build_subsets,
        _register_benchmark_embeddings,
        _register_index_docs,
        _register_run,
        _register_eval,
        _register_baseline_promote,
        _register_baseline_export,
        _register_experiment,
        _register_review_failures,
        _register_trace_show,
        _register_ui,
        _register_validate_eval_examples,
        _register_promote_eval_examples,
        _register_eval_variance,
        _register_model_ab_compatibility,
        _register_doctor,
    )
    for register_command in command_registrars:
        register_command(subparsers, handlers)


def build_parser(handlers: CliHandlers) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grounded-support-rag")
    _add_global_options(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_subcommands(subparsers, handlers)

    return parser
