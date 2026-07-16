"""Dense, keyword, hybrid, and rerank retrieval ablation diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from support_graph.config.runtime import RuntimeConfig
from support_graph.config.settings import Settings
from support_graph.evaluation.evaluate import (
    doc_recall_at_k,
    graded_mrr_at_k,
    has_rag_eval_fields,
    hit_at_k,
    load_eval_examples,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    span_recall_at_k,
)
from support_graph.retrieval.index import load_chunk_records
from support_graph.retrieval.retrieve import (
    RetrievalMode,
    build_keyword_retriever,
    build_query,
    build_query_context,
    get_vectorstore,
    retrieve_chunks,
)
from support_graph.types import (
    DatasetSplit,
    EvalSubset,
    Example,
    NormalizedRetrievalHit,
    parse_dataset_split,
    parse_domain,
    parse_eval_subset,
)

MODE_ORDER = ("dense_only", "keyword_only", "hybrid_no_rerank", "hybrid_rerank")
METRIC_KEYS = (
    "doc_recall_at_1",
    "doc_recall_at_3",
    "doc_recall_at_5",
    "doc_recall_at_10",
    "span_recall_at_5",
    "mrr_at_5",
    "hit_at_5",
    "precision_at_5",
    "graded_mrr_at_5",
    "ndcg_at_5",
)


@dataclass(frozen=True, slots=True)
class AblationMode:
    name: str
    retrieval_mode: RetrievalMode
    rerank: bool


ABLATION_MODES = {
    "dense_only": AblationMode("dense_only", "dense", False),
    "keyword_only": AblationMode("keyword_only", "keyword", False),
    "hybrid_no_rerank": AblationMode("hybrid_no_rerank", "hybrid", False),
    "hybrid_rerank": AblationMode("hybrid_rerank", "hybrid", True),
}


def metric_value(
    example: Mapping[str, Any],
    chunks: list[NormalizedRetrievalHit],
    key: str,
    top_k: int,
) -> float | None:
    if key == "doc_recall_at_1":
        return doc_recall_at_k(example.get("gold_doc_ids", []), chunks, k=1)
    if key == "doc_recall_at_3":
        return doc_recall_at_k(example.get("gold_doc_ids", []), chunks, k=3)
    if key == "doc_recall_at_5":
        return doc_recall_at_k(example.get("gold_doc_ids", []), chunks, k=5)
    if key == "doc_recall_at_10":
        if top_k < 10:
            return None
        return doc_recall_at_k(example.get("gold_doc_ids", []), chunks, k=10)
    if key == "span_recall_at_5":
        acceptable = (
            example.get("acceptable_span_ids", [])
            if has_rag_eval_fields(example)
            else None
        )
        return span_recall_at_k(
            example.get("gold_span_ids", []),
            chunks,
            k=5,
            acceptable_span_ids=acceptable,
        )
    if key == "mrr_at_5":
        return mrr_at_k(example.get("gold_doc_ids", []), chunks, k=5)
    if not has_rag_eval_fields(example):
        return None
    expected = example.get("expected_sources", [])
    acceptable_sources = example.get("acceptable_sources", [])
    if key == "hit_at_5":
        return hit_at_k(expected, acceptable_sources, chunks, k=5)
    if key == "precision_at_5":
        return precision_at_k(expected, acceptable_sources, chunks, k=5)
    if key == "graded_mrr_at_5":
        return graded_mrr_at_k(expected, acceptable_sources, chunks, k=5)
    if key == "ndcg_at_5":
        return ndcg_at_k(expected, acceptable_sources, chunks, k=5)
    raise ValueError(f"Unknown metric: {key}")


def retrieval_metrics(
    example: Mapping[str, Any],
    chunks: list[NormalizedRetrievalHit],
    *,
    top_k: int,
) -> dict[str, float | None]:
    return {key: metric_value(example, chunks, key, top_k) for key in METRIC_KEYS}


def _mean(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def aggregate_metrics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | int | None]]:
    by_mode: dict[str, list[Mapping[str, Any]]] = {mode: [] for mode in MODE_ORDER}
    for record in records:
        by_mode[str(record["mode"])].append(record)

    aggregates: dict[str, dict[str, float | int | None]] = {}
    for mode, mode_records in by_mode.items():
        metrics = {
            key: _mean([record["metrics"].get(key) for record in mode_records])
            for key in METRIC_KEYS
        }
        source_counts: Counter[str] = Counter()
        for record in mode_records:
            source_counts.update(record.get("retrieval_source_counts", {}))
        aggregates[mode] = {
            "examples": len(mode_records),
            **metrics,
            "dense_sources": source_counts.get("dense", 0),
            "keyword_sources": source_counts.get("keyword", 0),
        }
    return aggregates


def quality_score(metrics: Mapping[str, float | None]) -> float:
    for key in (
        "ndcg_at_5",
        "hit_at_5",
        "span_recall_at_5",
        "doc_recall_at_5",
        "doc_recall_at_3",
        "mrr_at_5",
    ):
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return 0.0


def comparison_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_example: dict[str, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        by_example.setdefault(str(record["example_id"]), {})[str(record["mode"])] = (
            record
        )

    rows: list[dict[str, Any]] = []
    for example_id, modes in sorted(by_example.items()):
        scores = {
            mode: quality_score(modes.get(mode, {}).get("metrics", {}))
            for mode in MODE_ORDER
        }
        best_score = max(scores.values()) if scores else 0.0
        best_modes = [mode for mode, score in scores.items() if score == best_score]
        row = {
            "example_id": example_id,
            "best_modes": "|".join(best_modes),
            "dense_score": scores["dense_only"],
            "keyword_score": scores["keyword_only"],
            "hybrid_no_rerank_score": scores["hybrid_no_rerank"],
            "hybrid_rerank_score": scores["hybrid_rerank"],
            "dense_beats_keyword": scores["dense_only"] > scores["keyword_only"],
            "keyword_beats_dense": scores["keyword_only"] > scores["dense_only"],
            "hybrid_rescue": scores["hybrid_no_rerank"]
            > max(scores["dense_only"], scores["keyword_only"]),
            "rerank_rescue": scores["hybrid_rerank"] > scores["hybrid_no_rerank"],
            "rerank_regression": scores["hybrid_rerank"] < scores["hybrid_no_rerank"],
            "all_modes_miss": all(score <= 0.0 for score in scores.values()),
        }
        rows.append(row)
    return rows


def _chunk_ids(chunks: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(chunk["chunk_id"]) for chunk in chunks if chunk.get("chunk_id")]


def _source_counts(chunks: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(
        Counter(str(chunk.get("retrieval_source", "unknown")) for chunk in chunks)
    )


def build_run_id(
    now: datetime | None = None, *, domain: str, subset: str, slug: str | None = None
) -> str:
    current = now or datetime.now().astimezone()
    parts = [current.strftime("%Y%m%d-%H%M%S"), domain, subset, "retrieval-ablation"]
    if slug:
        parts.append(slug)
    return "-".join(parts)


def run_ablation(
    examples: Sequence[Example],
    *,
    config: RuntimeConfig,
    vectorstore: Any,
    keyword_retriever: Any,
    top_k: int,
    candidate_k: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for example in examples:
        query = build_query(example, include_labels=False)
        query_context = build_query_context(example)
        for mode_name in MODE_ORDER:
            mode = ABLATION_MODES[mode_name]
            chunks = retrieve_chunks(
                example=example,
                vectorstore=vectorstore,
                keyword_retriever=keyword_retriever,
                config=config,
                top_k=top_k,
                candidate_k=candidate_k,
                query=query,
                query_context=query_context,
                retrieval_mode=mode.retrieval_mode,
                rerank=mode.rerank,
            )
            records.append(
                {
                    "example_id": example["example_id"],
                    "mode": mode_name,
                    "query": query,
                    "metrics": retrieval_metrics(example, chunks, top_k=top_k),
                    "retrieval_ranked_chunks": chunks,
                    "retrieval_ranked_chunk_ids": _chunk_ids(chunks),
                    "retrieval_source_counts": _source_counts(chunks),
                }
            )
    return records


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "example_id",
        "best_modes",
        "dense_score",
        "keyword_score",
        "hybrid_no_rerank_score",
        "hybrid_rerank_score",
        "dense_beats_keyword",
        "keyword_beats_dense",
        "hybrid_rescue",
        "rerank_rescue",
        "rerank_regression",
        "all_modes_miss",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def summary_markdown(
    metrics: Mapping[str, Mapping[str, Any]], comparisons: Sequence[Mapping[str, Any]]
) -> str:
    lines = [
        "# Retrieval Ablation",
        "",
        "Retrieval-only diagnostic. No answer generation, no LLM judging.",
        "",
        "These results measure evidence retrieval, not final answer quality.",
        "",
        "## Metrics",
        "",
    ]
    for mode in MODE_ORDER:
        mode_metrics = metrics[mode]
        lines.append(f"### {mode}")
        lines.append(f"- examples: {mode_metrics['examples']}")
        lines.append(f"- doc_recall_at_3: {_fmt(mode_metrics['doc_recall_at_3'])}")
        lines.append(f"- span_recall_at_5: {_fmt(mode_metrics['span_recall_at_5'])}")
        lines.append(f"- mrr_at_5: {_fmt(mode_metrics['mrr_at_5'])}")
        lines.append(f"- ndcg_at_5: {_fmt(mode_metrics['ndcg_at_5'])}")
        lines.append(
            f"- sources: dense={mode_metrics['dense_sources']} keyword={mode_metrics['keyword_sources']}"
        )
        lines.append("")

    counters = {
        "dense_beats_keyword": sum(
            1 for row in comparisons if row["dense_beats_keyword"]
        ),
        "keyword_beats_dense": sum(
            1 for row in comparisons if row["keyword_beats_dense"]
        ),
        "hybrid_rescue": sum(1 for row in comparisons if row["hybrid_rescue"]),
        "rerank_rescue": sum(1 for row in comparisons if row["rerank_rescue"]),
        "rerank_regression": sum(1 for row in comparisons if row["rerank_regression"]),
        "all_modes_miss": sum(1 for row in comparisons if row["all_modes_miss"]),
    }
    lines.extend(
        [
            "## Diagnostics",
            "",
            f"- dense wins over keyword: {counters['dense_beats_keyword']}",
            f"- keyword wins over dense: {counters['keyword_beats_dense']}",
            f"- hybrid rescue cases: {counters['hybrid_rescue']}",
            f"- rerank rescue cases: {counters['rerank_rescue']}",
            f"- rerank regression cases: {counters['rerank_regression']}",
            f"- all modes miss: {counters['all_modes_miss']}",
            "",
            "Inspect `comparison.csv` and `per_example.jsonl` for the exact examples.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    *,
    output_dir: Path,
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> None:
    metrics = aggregate_metrics(records)
    comparisons = comparison_rows(records)
    _write_json(output_dir / "manifest.json", manifest)
    _write_jsonl(output_dir / "per_example.jsonl", records)
    _write_json(output_dir / "metrics.json", metrics)
    _write_csv(output_dir / "comparison.csv", comparisons)
    (output_dir / "summary.md").write_text(
        summary_markdown(metrics, comparisons),
        encoding="utf-8",
    )


def run_command(args: argparse.Namespace) -> None:
    settings = Settings.load(args.config_file, args.secrets_file)
    domain = parse_domain(args.domain)
    split = parse_dataset_split(args.split)
    subset = parse_eval_subset(args.subset)
    config = settings.runtime_for(domain)
    top_k = int(args.top_k or config.retrieval_top_k)
    candidate_k = int(args.candidate_k or config.retrieval_candidate_k)
    config = replace(config, retrieval_top_k=top_k, retrieval_candidate_k=candidate_k)
    examples, subset_name = load_eval_examples(settings, domain, split, subset)
    if args.limit is not None:
        examples = examples[: int(args.limit)]

    if config.chunk_artifact_path is None:
        raise ValueError("Missing chunk_artifact_path for retrieval ablation.")
    chunk_records = load_chunk_records(config.chunk_artifact_path)
    vectorstore = get_vectorstore(config)
    keyword_retriever = build_keyword_retriever(config, chunk_records=chunk_records)
    run_id = args.run_id or build_run_id(domain=str(domain), subset=subset_name)
    output_dir = (
        settings.paths.project_root / "outputs/dev_tools/retrieval_ablation" / run_id
    )
    records = run_ablation(
        examples,
        config=config,
        vectorstore=vectorstore,
        keyword_retriever=keyword_retriever,
        top_k=top_k,
        candidate_k=candidate_k,
    )
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "domain": str(domain),
        "split": str(split),
        "subset": subset_name,
        "examples": len(examples),
        "top_k": top_k,
        "candidate_k": candidate_k,
        "modes": list(MODE_ORDER),
        "collection_name": config.collection_name,
        "embedding_model": config.embedding_model,
        "output_dir": str(output_dir),
    }
    write_outputs(output_dir=output_dir, manifest=manifest, records=records)
    print(f"Wrote retrieval ablation artifacts to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run retrieval ablation diagnostics.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run retrieval ablation.")
    run_parser.add_argument("--config-file", default="support_graph.kubernetes.toml")
    run_parser.add_argument("--secrets-file", default=None)
    run_parser.add_argument("--domain", default="kubernetes")
    run_parser.add_argument("--split", default=DatasetSplit.VALIDATION)
    run_parser.add_argument("--subset", default=EvalSubset.SMOKE)
    run_parser.add_argument("--top-k", type=int, default=None)
    run_parser.add_argument("--candidate-k", type=int, default=None)
    run_parser.add_argument("--limit", type=int, default=None)
    run_parser.add_argument("--run-id", default=None)
    run_parser.set_defaults(func=run_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
