from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples._shared import load_settings, next_step_lines, print_lines


def build_lines() -> list[str]:
    settings = load_settings()
    lines = [
        "SupportGraph Example 00",
        "Project overview",
        "",
        "Package layout",
        "- support_graph/data: dataset loading, chunking, example building, subsets",
        "- support_graph/retrieval: indexing, query construction, retrieval, reranking",
        "- support_graph/runtime: LangGraph runtime and local trace writing",
        "- support_graph/evaluation: eval harness, metrics, ablations, benchmark helpers",
        "- support_graph/app: CLI entrypoints and terminal formatting",
        "",
        "Key files",
        "- Runtime graph: support_graph/runtime/graph.py",
        "- Retrieval logic: support_graph/retrieval/retrieve.py",
        "- Eval harness: support_graph/evaluation/evaluate.py",
        "- CLI: support_graph/app/cli.py",
        "",
        "CLI surfaces",
        "- build-chunks, build-examples, build-subsets",
        "- index-docs, run, eval, ablate-smoke10",
        "- review-failures, trace-show",
        "",
        "Artifact directories",
        f"- data/derived/chunks -> {settings.paths.chunks_dir}",
        f"- data/derived/examples -> {settings.paths.examples_dir}",
        f"- outputs/evals/runs -> {settings.paths.eval_runs_dir}",
        f"- outputs/evals/reports -> {settings.paths.eval_reports_dir}",
        f"- outputs/runs -> {settings.paths.runs_dir}",
        "",
        "Deterministic steps",
        "- raw dataset loading",
        "- chunk building and deterministic subchunk IDs",
        "- example building and subset selection",
        "- query-context extraction, reranking, neighbor expansion",
        "- eval metrics and citation validation",
        "",
        "Model-backed steps",
        "- embedding model: index-docs and pgvector similarity search",
        "- chat model: grade_evidence, generate_response, resolve_without_answer",
    ]
    lines.extend(next_step_lines("01_dataset_eda.py"))
    return lines


def main() -> None:
    print_lines(build_lines())


if __name__ == "__main__":
    main()
