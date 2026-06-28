from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from support_graph.cli import handlers as cli
from support_graph.evaluation import experiment


def _smoke10_examples() -> list[dict]:
    examples: list[dict] = []
    for index in range(10):
        examples.append(
            {
                "example_id": f"dmv::smoke::{index}",
                "domain": "dmv",
                "target_mode": "answer",
                "target_turn_id": index + 1,
                "turns_before_target": [
                    {"turn_id": 1, "role": "user", "utterance": f"Question {index}?"}
                ],
                "target_turn": {"utterance": f"Gold answer {index}."},
                "gold_doc_ids": [f"doc-{index}"],
                "gold_span_ids": [f"span-{index}"],
            }
        )
    return examples


def _run_graph_variant(*, example: dict, config: Any, **kwargs) -> dict:
    gold_doc_id = example["gold_doc_ids"][0]
    gold_span_id = example["gold_span_ids"][0]
    variant = getattr(config, "experiment_variant", "control").replace("-frozen", "")
    ranked_doc_id = gold_doc_id
    ranked_span_ids = [gold_span_id]
    retrieved_doc_id = gold_doc_id
    retrieved_span_ids = [gold_span_id]
    citations = [
        {
            "doc_id": gold_doc_id,
            "chunk_id": f"{variant}::{example['example_id']}::expanded",
            "span_ids": [gold_span_id],
        }
    ]
    response_text = example["target_turn"]["utterance"]
    latency_ms = 1000.0

    if variant == "structured-query":
        ranked_doc_id = "wrong-doc"
        ranked_span_ids = ["wrong-span"]
        citations = []
        response_text = "Off-topic answer."
    elif variant == "structured-query-rerank":
        latency_ms = 1200.0
    elif variant == "structured-query-rerank-neighbors":
        latency_ms = 1300.0

    return {
        "example_id": example["example_id"],
        "decision": "answer",
        "response_text": response_text,
        "citations": citations,
        "retrieval_ranked_chunks": [
            {
                "rank": 1,
                "chunk_id": f"{variant}::{example['example_id']}::ranked",
                "domain": "dmv",
                "doc_id": ranked_doc_id,
                "doc_title": ranked_doc_id,
                "section_id": "1",
                "section_title": "Body",
                "parent_titles": [],
                "span_ids": ranked_span_ids,
                "token_count": 12,
                "text": response_text,
                "score": 0.01,
            }
        ],
        "retrieved_chunks": [
            {
                "rank": 1,
                "chunk_id": f"{variant}::{example['example_id']}::expanded",
                "domain": "dmv",
                "doc_id": retrieved_doc_id,
                "doc_title": retrieved_doc_id,
                "section_id": "1",
                "section_title": "Body",
                "parent_titles": [],
                "span_ids": retrieved_span_ids,
                "token_count": 12,
                "text": response_text,
                "score": 0.01,
            }
        ],
        "trace_summary": {
            "retrieval_attempts": 1,
            "final_query": "smoke10 query",
            "graph_path": [
                "prepare_query",
                "retrieve_docs",
                "grade_evidence",
                "generate_response",
                "finalize",
            ],
            "latency_ms": latency_ms,
        },
    }


def test_smoke10_variant_run_ids_and_artifact_paths(
    monkeypatch,
    tmp_path: Path,
    make_settings,
) -> None:
    settings = make_settings(project_root=tmp_path)
    fixed_now = datetime(2026, 3, 18, 14, 30, tzinfo=timezone.utc)

    monkeypatch.setattr(
        experiment,
        "load_eval_examples",
        lambda *args, **kwargs: (_smoke10_examples(), "smoke"),
        raising=False,
    )
    result = asyncio.run(
        experiment.run_smoke10_experiment_async(
            settings=settings,
            domain="dmv",
            limit=10,
            run_graph_func=_run_graph_variant,
            now=fixed_now,
        )
    )

    assert result["report_id"] == "20260318-143000-dmv-smoke10-experiment-summary"
    assert (
        result["report_artifact_paths"]["manifest"]
        == tmp_path
        / "outputs/evals/reports/20260318-143000-dmv-smoke10-experiment-summary/manifest.json"
    )
    assert (
        result["report_artifact_paths"]["report"]
        == tmp_path
        / "outputs/evals/reports/20260318-143000-dmv-smoke10-experiment-summary/report.md"
    )
    assert len(result["results"]) == 4
    for item in result["results"]:
        expected_run_id = f"20260318-143000-dmv-smoke10-{item['variant']['id']}"
        expected_dir = tmp_path / "outputs/evals/runs" / expected_run_id
        assert item["run_id"] == expected_run_id
        assert item["output_dir"] == expected_dir
        assert item["artifact_paths"]["manifest"] == expected_dir / "manifest.json"
        assert item["artifact_paths"]["metrics"] == expected_dir / "metrics.json"
        assert (
            item["artifact_paths"]["predictions"] == expected_dir / "predictions.jsonl"
        )
        assert item["artifact_paths"]["failures"] == expected_dir / "failures.jsonl"
        assert item["artifact_paths"]["summary"] == expected_dir / "summary.md"


def test_smoke10_comparison_summary_classification_and_guardrails() -> None:
    control = {
        "variant": next(
            item for item in experiment.VARIANTS if item["id"] == "control"
        ),
        "metrics": {
            "retrieval": {
                "answer": {"doc_recall_at_3": 0.60, "span_recall_at_5": 0.20}
            },
            "generation": {
                "answer": {"citation_coverage": 0.20, "end_to_end_success_rate": 0.20}
            },
            "latency_ms": {"average": 100.0},
        },
        "failure_counts": {},
    }
    structured_query = {
        "variant": next(
            item for item in experiment.VARIANTS if item["id"] == "structured-query"
        ),
        "metrics": {
            "retrieval": {
                "answer": {"doc_recall_at_3": 0.44, "span_recall_at_5": 0.25}
            },
            "generation": {
                "answer": {"citation_coverage": 0.20, "end_to_end_success_rate": 0.20}
            },
            "latency_ms": {"average": 105.0},
        },
        "failure_counts": {},
    }
    rerank = {
        "variant": next(
            item
            for item in experiment.VARIANTS
            if item["id"] == "structured-query-rerank"
        ),
        "metrics": {
            "retrieval": {
                "answer": {"doc_recall_at_3": 0.60, "span_recall_at_5": 0.35}
            },
            "generation": {
                "answer": {"citation_coverage": 0.25, "end_to_end_success_rate": 0.30}
            },
            "latency_ms": {"average": 110.0},
        },
        "failure_counts": {},
    }
    neighbors = {
        "variant": next(
            item
            for item in experiment.VARIANTS
            if item["id"] == "structured-query-rerank-neighbors"
        ),
        "metrics": {
            "retrieval": {
                "answer": {"doc_recall_at_3": 0.60, "span_recall_at_5": 0.21}
            },
            "generation": {
                "answer": {"citation_coverage": 0.21, "end_to_end_success_rate": 0.20}
            },
            "latency_ms": {"average": 40_000.0},
        },
        "failure_counts": {},
    }

    assert experiment.classify_variant(control, structured_query)[0] == "didn't work"
    assert experiment.classify_variant(control, rerank)[0] == "worked"
    assert experiment.classify_variant(control, neighbors)[0] == "didn't work"


def test_experiment_cli_output_hierarchy(
    monkeypatch,
    capsys,
    tmp_path: Path,
    make_settings,
) -> None:
    settings = make_settings(project_root=tmp_path)
    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(lambda cls, config_file=None, secrets_file=None: settings),
        raising=False,
    )
    monkeypatch.setattr(
        cli, "collection_row_count", lambda *args, **kwargs: 4316, raising=False
    )
    monkeypatch.setattr(
        cli,
        "run_smoke10_experiment_async",
        lambda **kwargs: {
            "domain": "dmv",
            "limit": 10,
            "results": [
                {
                    "variant": {"title": "Control"},
                    "status": "control",
                    "metrics": {
                        "retrieval": {"answer": {"span_recall_at_5": 0.2}},
                        "generation": {
                            "answer": {
                                "citation_coverage": 0.2,
                                "end_to_end_success_rate": 0.2,
                            }
                        },
                    },
                },
                {
                    "variant": {"title": "Structured Query + Rerank"},
                    "status": "worked",
                    "metrics": {
                        "retrieval": {"answer": {"span_recall_at_5": 0.3}},
                        "generation": {
                            "answer": {
                                "citation_coverage": 0.3,
                                "end_to_end_success_rate": 0.3,
                            }
                        },
                    },
                },
            ],
            "recommendation": "keep",
            "recommendation_line": "Promote Structured Query + Rerank to the Frozen-200 check next.",
            "frozen_result": None,
            "report_artifact_paths": {
                "manifest": tmp_path
                / "outputs/evals/reports/20260318-143000-dmv-smoke10-experiment-summary/manifest.json",
                "report": tmp_path
                / "outputs/evals/reports/20260318-143000-dmv-smoke10-experiment-summary/report.md",
            },
        },
        raising=False,
    )

    exit_code = cli.main(["experiment-smoke10", "--domain", "dmv"])
    output = capsys.readouterr().out.splitlines()

    assert exit_code == 0
    assert output[0] == "SupportGraph Experiment"
    assert output[1].startswith("Scope:")
    assert "Variants" in output
    assert any("Structured Query + Rerank: worked" in line for line in output)
    assert "Recommendation" in output
    assert any(line.startswith("keep:") for line in output)
    assert "Frozen-200" in output
    assert any(line.startswith("skipped:") for line in output)
    assert "Artifacts" in output
    assert any(line.endswith("report.md") for line in output)
