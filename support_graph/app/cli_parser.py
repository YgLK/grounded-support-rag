"""Argument parser construction for the SupportGraph CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

from support_graph.app.cli_shared import (
    DOMAIN_CHOICES,
    EVAL_SUBSET_CHOICES,
    SPLIT_CHOICES,
)
from support_graph.types import DatasetSplit, EvalSubset

CommandHandler = Callable[[argparse.Namespace], int]
Subparsers = argparse._SubParsersAction


@dataclass(frozen=True)
class CliHandlers:
    build_chunks: CommandHandler
    build_examples: CommandHandler
    build_subsets: CommandHandler
    benchmark_embeddings: CommandHandler
    index_docs: CommandHandler
    run_example: CommandHandler
    eval_split: CommandHandler
    ablate_smoke10: CommandHandler
    review_failures: CommandHandler
    trace_show: CommandHandler
    serve_ui: CommandHandler


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


def _register_build_examples(subparsers: Subparsers, handlers: CliHandlers) -> None:
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
    build_examples_parser.set_defaults(func=handlers.build_examples)


def _register_build_subsets(subparsers: Subparsers, handlers: CliHandlers) -> None:
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
    build_subsets_parser.set_defaults(func=handlers.build_subsets)


def _register_benchmark_embeddings(
    subparsers: Subparsers, handlers: CliHandlers
) -> None:
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


def _register_ablation(subparsers: Subparsers, handlers: CliHandlers) -> None:
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
    ablation_parser.set_defaults(func=handlers.ablate_smoke10)


def _register_review_failures(subparsers: Subparsers, handlers: CliHandlers) -> None:
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
    trace_show_parser = subparsers.add_parser(
        "trace-show",
        help="Inspect the raw trace for one evaluated example.",
    )
    trace_show_parser.add_argument("--run-id", required=True)
    trace_show_parser.add_argument("--example-id", required=True)
    trace_show_parser.set_defaults(func=handlers.trace_show)


def _register_ui(subparsers: Subparsers, handlers: CliHandlers) -> None:
    ui_parser = subparsers.add_parser(
        "ui",
        help="Serve the local SupportGraph Workbench UI.",
    )
    ui_parser.add_argument("--host", default="127.0.0.1")
    ui_parser.add_argument("--port", type=int, default=8008)
    ui_parser.set_defaults(func=handlers.serve_ui)


def register_subcommands(subparsers: Subparsers, handlers: CliHandlers) -> None:
    command_registrars: tuple[Callable[[Subparsers, CliHandlers], None], ...] = (
        _register_build_chunks,
        _register_build_examples,
        _register_build_subsets,
        _register_benchmark_embeddings,
        _register_index_docs,
        _register_run,
        _register_eval,
        _register_ablation,
        _register_review_failures,
        _register_trace_show,
        _register_ui,
    )
    for register_command in command_registrars:
        register_command(subparsers, handlers)


def build_parser(handlers: CliHandlers) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="support-graph")
    _add_global_options(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_subcommands(subparsers, handlers)

    return parser
