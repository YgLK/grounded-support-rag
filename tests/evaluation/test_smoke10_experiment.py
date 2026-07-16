from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from support_graph.cli import handlers as cli
from support_graph.evaluation import experiment
from support_graph.evaluation.contracts import DatasetRef, ExperimentSnapshot


def _smoke10_examples() -> list[dict]:
    examples: list[dict] = []
    for index in range(10):
        examples.append(
            {
                "example_id": f"kubernetes::smoke::{index}",
                "domain": "kubernetes",
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
                "domain": "kubernetes",
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
                "domain": "kubernetes",
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


def test_smoke10_variants_are_tagged_hosted_experiments(
    monkeypatch,
    tmp_path: Path,
    make_settings,
) -> None:
    settings = make_settings(project_root=tmp_path)
    settings.runtime.validate_for_hosted_eval = lambda: None
    fixed_now = datetime(2026, 3, 18, 14, 30, tzinfo=timezone.utc)

    monkeypatch.setattr(
        experiment,
        "load_eval_examples",
        lambda *args, **kwargs: (_smoke10_examples(), "smoke"),
        raising=False,
    )
    calls: list[dict] = []

    async def fake_hosted(**kwargs):
        calls.append(kwargs)
        variant = kwargs["experiment_metadata"]["variant"]
        snapshot = ExperimentSnapshot(
            id=f"exp-{variant}",
            name=variant,
            dataset=DatasetRef("dataset", "dataset", "hash"),
            metadata=kwargs["experiment_metadata"],
            results=(),
            url=f"https://smith/{variant}",
        )
        return SimpleNamespace(
            experiment=snapshot,
            aggregate_metrics={
                "doc_recall_at_3": 0.6,
                "span_recall_at_5": 0.2,
                "citation_coverage": 0.1,
                "end_to_end_success": 0.2,
            },
        )

    monkeypatch.setattr(experiment, "run_hosted_evaluation", fake_hosted)
    result = asyncio.run(
        experiment.run_smoke10_experiment_async(
            settings=settings,
            domain="kubernetes",
            limit=10,
            now=fixed_now,
            gateway=object(),
            policy=object(),
            corpus_manifest_path=tmp_path / "manifest.json",
            git_state=SimpleNamespace(),
        )
    )

    assert len(result["results"]) == 4
    assert result["study_id"] == "20260318-kubernetes-smoke-retrieval-ablation"
    assert result["experiment_ids"] == [
        "exp-control",
        "exp-structured-query",
        "exp-structured-query-rerank",
        "exp-structured-query-rerank-neighbors",
    ]
    assert [call["experiment_metadata"] for call in calls] == [
        {
            "study_type": "retrieval_ablation",
            "study_id": "20260318-kubernetes-smoke-retrieval-ablation",
            "variant": item["variant"]["id"],
            "repetition": 1,
        }
        for item in result["results"]
    ]
    assert not (tmp_path / "outputs").exists()


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
    settings.runtime.validate_for_hosted_eval = lambda: None
    monkeypatch.setattr(
        cli.Settings,
        "load",
        classmethod(lambda cls, config_file=None, secrets_file=None: settings),
        raising=False,
    )
    monkeypatch.setattr(cli, "build_langsmith_gateway", lambda config: object())
    monkeypatch.setattr(cli, "_baseline_policy", lambda *args: object())
    monkeypatch.setattr(cli, "read_git_state", lambda root: SimpleNamespace())
    monkeypatch.setattr(
        cli,
        "run_smoke10_experiment_async",
        lambda **kwargs: {
            "domain": "kubernetes",
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
            "study_id": "20260318-kubernetes-smoke-retrieval-ablation",
            "experiment_ids": ["exp-control", "exp-rerank"],
            "experiment_urls": [
                "https://smith/exp-control",
                "https://smith/exp-rerank",
            ],
        },
        raising=False,
    )

    exit_code = cli.main(["experiment-smoke10", "--domain", "kubernetes"])
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
    assert "Artifacts" not in output
    assert "https://smith/exp-control" in output
