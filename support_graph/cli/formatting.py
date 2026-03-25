"""Terminal output formatting for the SupportGraph CLI."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from support_graph.cli.utils import metric_text, relative_path, shorten
from support_graph.config.settings import Settings


def _format_next_steps_guidance(result: dict) -> list[str]:
    """Generate user guidance on what to do next based on the run decision."""
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
    """Format a single run result into a CLI-friendly text summary.

    - Renders a Jinja2 template for consistent output.
    - Displays the run context, decision, response, citations, and trace summary.
    - Lists key artifact paths for further inspection.

    Args:
        result: A dictionary containing the run result and metadata.
        settings: The application settings.
        verbose: If True, include detailed query and retrieval info.

    Returns:
        A list of strings representing the formatted output lines.
    """
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=False,
    )
    template = env.get_template("run_output.txt.j2")

    raw_artifact_paths = result["artifact_paths"]
    template_context = {
        "result": result,
        "trace_summary": result.get("trace_summary", {}),
        "next_lines": _format_next_steps_guidance(result),
        "verbose": verbose,
        "artifact_paths": {
            "manifest": relative_path(
                Path(raw_artifact_paths["manifest"]), settings.paths.project_root
            ),
            "result": relative_path(
                Path(raw_artifact_paths["result"]), settings.paths.project_root
            ),
            "trace": relative_path(
                Path(raw_artifact_paths["trace"]), settings.paths.project_root
            ),
        },
    }

    return template.render(template_context).splitlines()


def format_eval_output(result: dict, settings: Settings) -> list[str]:
    """Format evaluation results into a CLI-friendly text summary.

    - Renders a Jinja2 template for consistent output.
    - Displays headline metrics, paper-reference metrics, and a failure snapshot.
    - Lists key artifact paths for further inspection.

    Args:
        result: A dictionary containing the evaluation metrics and metadata.
        settings: The application settings.

    Returns:
        A list of strings representing the formatted output lines.
    """
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=False,
    )
    template = env.get_template("eval_output.txt.j2")

    retrieval = result.get("metrics", {}).get("retrieval", {}).get("answer", {})
    generation = result.get("metrics", {}).get("generation", {}).get("answer", {})
    failure_counts = result.get("failure_counts", {})
    output_dir = Path(result.get("output_dir"))
    retrieval_top_k = result.get("retrieval_top_k", settings.runtime.retrieval_top_k)

    headline_metrics = {
        "doc_recall_at_3": metric_text(
            retrieval.get("doc_recall_at_3"),
            retrieval_top_k=retrieval_top_k,
            metric_k=3,
        ),
        "span_recall_at_5": metric_text(
            retrieval.get("span_recall_at_5"),
            retrieval_top_k=retrieval_top_k,
            metric_k=5,
        ),
        "rouge_l": metric_text(generation.get("rouge_l")),
        "token_f1": metric_text(generation.get("token_f1")),
    }

    paper_metrics = {
        "recall_at_1": metric_text(
            retrieval.get("doc_recall_at_1"),
            retrieval_top_k=retrieval_top_k,
            metric_k=1,
        ),
        "recall_at_5": metric_text(
            retrieval.get("doc_recall_at_5"),
            retrieval_top_k=retrieval_top_k,
            metric_k=5,
        ),
        "recall_at_10": metric_text(
            retrieval.get("doc_recall_at_10"),
            retrieval_top_k=retrieval_top_k,
            metric_k=10,
        ),
        "exact_match": metric_text(generation.get("exact_match")),
        "sacrebleu": metric_text(generation.get("sacrebleu")),
    }

    failure_snapshot = (
        sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))[:3]
        if failure_counts
        else []
    )

    artifact_paths = [
        relative_path(output_dir / "summary.md", settings.paths.project_root),
        relative_path(output_dir / "failures.jsonl", settings.paths.project_root),
        relative_path(output_dir / "manual_review.csv", settings.paths.project_root),
        relative_path(
            output_dir / "retrieval_examples.jsonl", settings.paths.project_root
        ),
        relative_path(output_dir / "trace_index.json", settings.paths.project_root),
    ]

    template_context = {
        "run_id": result.get("run_id"),
        "subset_label": result.get("subset_label"),
        "headline_metrics": headline_metrics,
        "paper_metrics": paper_metrics,
        "failure_snapshot": failure_snapshot,
        "artifact_paths": artifact_paths,
    }

    return template.render(template_context).splitlines()


def format_experiment_output(result: dict, settings: Settings) -> list[str]:
    """Format experiment results into a CLI-friendly text summary.

    - Renders a Jinja2 template for consistent output.
    - Displays the experiment scope, variant results, and recommendation.
    - Lists key artifact paths for further inspection.

    Args:
        result: A dictionary containing the experiment results and metadata.
        settings: The application settings.

    Returns:
        A list of strings representing the formatted output lines.
    """
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=False,
    )
    template = env.get_template("experiment_output.txt.j2")

    results_context = []
    for item in result.get("results", []):
        metrics = item.get("metrics", {})
        retrieval = metrics.get("retrieval", {}).get("answer", {})
        generation = metrics.get("generation", {}).get("answer", {})
        results_context.append(
            {
                "title": item.get("variant", {}).get("title"),
                "status": item.get("status"),
                "span": (retrieval.get("span_recall_at_5") or 0.0),
                "citation": (generation.get("citation_coverage") or 0.0),
                "e2e": (generation.get("end_to_end_success_rate") or 0.0),
            }
        )

    template_context = {
        "scope": f"{result.get('domain')} smoke / first {result.get('limit')}",
        "results": results_context,
        "recommendation": result.get("recommendation"),
        "recommendation_line": result.get("recommendation_line"),
        "frozen_result": result.get("frozen_result", {})
        .get("variant", {})
        .get("title"),
        "report_path": relative_path(
            Path(result["report_artifact_paths"]["report"]),
            settings.paths.project_root,
        ),
    }

    return template.render(template_context).splitlines()


def format_review_failures_output(
    *,
    run_id: str,
    output_dir: Path,
    failures: list[dict],
    filtered: list[dict],
    limit: int,
    settings: Settings,
) -> list[str]:
    """Format failure review data into a CLI-friendly text summary.

    - Renders a Jinja2 template for consistent output.
    - Displays failure counts and a sample of failure examples.
    - Lists key artifact paths for further inspection.

    Args:
        run_id: The ID of the evaluation run.
        output_dir: The output directory for the run artifacts.
        failures: The full list of failure records.
        filtered: The filtered list of failure records to display.
        limit: The maximum number of examples to show.
        settings: The application settings.

    Returns:
        A list of strings representing the formatted output lines.
    """
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=False,
    )
    template = env.get_template("review_failures_output.txt.j2")

    failure_counts: dict[str, int] = {}
    for record in failures:
        label = str(record.get("failure_label") or "unknown")
        failure_counts[label] = failure_counts.get(label, 0) + 1

    examples_context = [
        {
            "example_id": record.get("example_id"),
            "label": record.get("failure_label") or "unknown",
            "decision": record.get("decision") or "unknown",
            "need": shorten(record.get("latest_user_utterance", "")),
        }
        for record in filtered[:limit]
    ]

    template_context = {
        "run_id": run_id,
        "failure_counts": sorted(
            failure_counts.items(), key=lambda item: (-item[1], item[0])
        )[:5],
        "examples": examples_context,
        "artifact_paths": [
            relative_path(output_dir / "failures.jsonl", settings.paths.project_root),
            relative_path(
                output_dir / "manual_review.csv", settings.paths.project_root
            ),
            relative_path(
                output_dir / "retrieval_examples.jsonl", settings.paths.project_root
            ),
        ],
    }

    return template.render(template_context).splitlines()


def format_trace_show_output(
    *,
    run_id: str,
    example_id: str,
    trace_summary: dict,
    settings: Settings,
) -> list[str]:
    """Format a trace summary into a CLI-friendly text view.

    - Renders a Jinja2 template for consistent output.
    - Displays trace details, node latencies, fallbacks, and evidence.
    - Lists key artifact paths for further inspection.

    Args:
        run_id: The ID of the run.
        example_id: The ID of the example.
        trace_summary: The trace summary dictionary.
        settings: The application settings.

    Returns:
        A list of strings representing the formatted output lines.
    """
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=False,
    )
    template = env.get_template("trace_show_output.txt.j2")

    node_latency_ms = trace_summary.get("node_latency_ms", {})
    formatted_latencies = {
        node: ", ".join(f"{float(value):.2f} ms" for value in latencies)
        for node, latencies in node_latency_ms.items()
    }

    fallbacks_context = [
        {
            "node": fallback.get("node", "unknown"),
            "exception_type": fallback.get("exception_type", "unknown"),
            "error": fallback.get("error", "unknown"),
        }
        for fallback in trace_summary.get("fallbacks", [])
    ]

    observability = trace_summary.get("observability", {})
    observability_lines: list[str] = []
    if observability:
        langsmith = observability.get("langsmith", {})
        if langsmith.get("enabled"):
            observability_lines.append(
                f"LangSmith: {langsmith.get('project', 'grounded-support-rag')}"
            )

    template_context = {
        "run_id": run_id,
        "example_id": example_id,
        "trace_summary": trace_summary,
        "node_latency_ms": formatted_latencies,
        "fallbacks": fallbacks_context,
        "observability_lines": observability_lines,
        "trace_path": relative_path(
            Path(str(trace_summary.get("trace_path", ""))),
            settings.paths.project_root,
        ),
    }

    return template.render(template_context).splitlines()
