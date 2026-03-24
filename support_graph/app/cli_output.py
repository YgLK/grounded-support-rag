"""Terminal output formatting for the SupportGraph CLI."""

from __future__ import annotations

from pathlib import Path

from support_graph.app.cli_shared import metric_text, relative_path, shorten
from support_graph.config.settings import Settings


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


def format_run_output(
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
            relative_path(
                Path(artifact_paths["manifest"]), settings.paths.project_root
            ),
            relative_path(Path(artifact_paths["result"]), settings.paths.project_root),
            relative_path(Path(artifact_paths["trace"]), settings.paths.project_root),
        ]
    )
    return lines


def format_eval_output(result: dict, settings: Settings) -> list[str]:
    retrieval = result.get("metrics", {}).get("retrieval", {}).get("answer", {})
    generation = result.get("metrics", {}).get("generation", {}).get("answer", {})
    failure_counts = result.get("failure_counts", {})
    output_dir = Path(result.get("output_dir"))
    retrieval_top_k = result.get("retrieval_top_k", settings.runtime.retrieval_top_k)

    lines = [
        "SupportGraph Eval",
        f"Run: {result.get('run_id')}",
        f"Subset: {result.get('subset_label')}",
        "",
        "Headline Metrics",
        f"Doc Recall@3: {metric_text(retrieval.get('doc_recall_at_3'), retrieval_top_k=retrieval_top_k, metric_k=3)}",
        f"Span Recall@5: {metric_text(retrieval.get('span_recall_at_5'), retrieval_top_k=retrieval_top_k, metric_k=5)}",
        f"ROUGE-L: {metric_text(generation.get('rouge_l'))}",
        f"F1: {metric_text(generation.get('token_f1'))}",
        "",
        "Paper Reference",
        f"Recall@1: {metric_text(retrieval.get('doc_recall_at_1'), retrieval_top_k=retrieval_top_k, metric_k=1)}",
        f"Recall@5: {metric_text(retrieval.get('doc_recall_at_5'), retrieval_top_k=retrieval_top_k, metric_k=5)}",
        f"Recall@10: {metric_text(retrieval.get('doc_recall_at_10'), retrieval_top_k=retrieval_top_k, metric_k=10)}",
        f"Exact Match: {metric_text(generation.get('exact_match'))}",
        f"SacreBLEU: {metric_text(generation.get('sacrebleu'))}",
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
            relative_path(output_dir / "summary.md", settings.paths.project_root),
            relative_path(output_dir / "failures.jsonl", settings.paths.project_root),
            relative_path(
                output_dir / "manual_review.csv", settings.paths.project_root
            ),
            relative_path(
                output_dir / "retrieval_examples.jsonl", settings.paths.project_root
            ),
            relative_path(output_dir / "trace_index.json", settings.paths.project_root),
        ]
    )
    return lines


def format_experiment_output(result: dict, settings: Settings) -> list[str]:
    lines = [
        "SupportGraph Experiment",
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
            relative_path(
                Path(result["report_artifact_paths"]["report"]),
                settings.paths.project_root,
            ),
        ]
    )
    return lines


def format_review_failures_output(
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
                    need=shorten(record.get("latest_user_utterance", "")),
                )
            )
    else:
        lines.append("No matching failures.")

    lines.extend(
        [
            "",
            "Artifacts",
            relative_path(output_dir / "failures.jsonl", settings.paths.project_root),
            relative_path(
                output_dir / "manual_review.csv", settings.paths.project_root
            ),
            relative_path(
                output_dir / "retrieval_examples.jsonl", settings.paths.project_root
            ),
        ]
    )
    return lines


def format_trace_show_output(
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
        langsmith = observability.get("langsmith", {})
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
            relative_path(
                Path(str(trace_summary.get("trace_path", ""))),
                settings.paths.project_root,
            ),
        ]
    )
    return lines
