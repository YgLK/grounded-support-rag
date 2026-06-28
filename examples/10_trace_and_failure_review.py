from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples._shared import (
    format_path,
    latest_eval_run_dir,
    load_jsonl_records,
    load_settings,
    print_lines,
    status_lines,
)


def build_lines(*, settings=None) -> list[str]:
    settings = settings or load_settings()
    run_dir = latest_eval_run_dir(
        settings,
        required_files=[
            "manual_review.csv",
            "retrieval_examples.jsonl",
            "trace_index.json",
            "summary.md",
        ],
    )
    if run_dir is None:
        return status_lines(
            "SupportGraph Example 10",
            state="artifacts-missing",
            summary="No completed eval run with analysis artifacts was found",
            next_commands=[
                "uv run python examples/09_eval_smoke_walkthrough.py",
                "uv run grounded-support-rag review-failures --run-id <run_id>",
                "uv run grounded-support-rag trace-show --run-id <run_id> --example-id '<example_id>'",
            ],
        )

    manual_review_path = run_dir / "manual_review.csv"
    retrieval_examples_path = run_dir / "retrieval_examples.jsonl"
    trace_index_path = run_dir / "trace_index.json"
    failures_path = run_dir / "failures.jsonl"
    trace_index = json.loads(trace_index_path.read_text(encoding="utf-8"))
    retrieval_examples = load_jsonl_records(retrieval_examples_path)
    failures = load_jsonl_records(failures_path)
    sample_entry = next(iter(trace_index.get("entries", [])), {})
    sample_example_id = sample_entry.get("example_id", "<example_id>")

    lines = [
        "SupportGraph Example 10",
        "Trace and failure review walkthrough",
        "",
        "Latest run",
        f"- Run ID: {run_dir.name}",
        f"- manual_review.csv -> {format_path(manual_review_path)}",
        f"- retrieval_examples.jsonl -> {format_path(retrieval_examples_path)}",
        f"- trace_index.json -> {format_path(trace_index_path)}",
        "",
        "Artifact counts",
        f"- Failure records: {len(failures)}",
        f"- Retrieval example records: {len(retrieval_examples)}",
        f"- Trace index entries: {len(trace_index.get('entries', []))}",
        "",
        "Sample trace entry",
        f"- Example ID: {sample_example_id}",
        f"- Graph path: {' -> '.join(sample_entry.get('graph_path', []))}",
        f"- Retrieval attempts: {sample_entry.get('retrieval_attempts', 0)}",
        "",
        "CLI inspection commands",
        f"- uv run grounded-support-rag review-failures --run-id {run_dir.name}",
        f"- uv run grounded-support-rag trace-show --run-id {run_dir.name} --example-id '{sample_example_id}'",
        "",
        "Next",
        "Use the walkthrough outputs as a map, then run the real CLI commands on your own examples or eval runs.",
    ]
    return lines


def main() -> None:
    print_lines(build_lines())


if __name__ == "__main__":
    main()
