from __future__ import annotations

import asyncio
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import get_type_hints

from support_graph.evaluation import evaluate
from support_graph.data.eval_subsets import write_subset_jsonl
from support_graph.runtime.traces import write_trace_event


def test_evaluate_examples_writes_required_artifacts_and_uses_ranked_vs_expanded_lists(
    monkeypatch,
    tmp_path: Path,
    make_settings,
) -> None:
    settings = make_settings(
        project_root=tmp_path,
        overrides={"prompt_version": "v-test"},
    )
    now = datetime(2026, 3, 18, 14, 30, tzinfo=timezone.utc)

    examples = [
        {
            "example_id": "kubernetes::one::turn_2",
            "target_mode": "answer",
            "target_turn_id": 2,
            "turns_before_target": [
                {"turn_id": 1, "role": "user", "utterance": "What should I bring?"}
            ],
            "target_turn": {"utterance": "Bring your insurance card tomorrow."},
            "gold_doc_ids": ["doc-a"],
            "gold_span_ids": ["1", "2"],
        }
    ]

    def fake_run_graph(*, trace_path: Path, **kwargs):
        write_trace_event(
            trace_path,
            {
                "node": "prepare_query",
                "query": "bring insurance card",
                "latency_ms": 1.0,
            },
        )
        write_trace_event(
            trace_path,
            {
                "node": "retrieve_docs",
                "retrieval_attempts": 1,
                "retrieval_ranked_count": 1,
                "retrieved_count": 1,
                "latency_ms": 2.0,
            },
        )
        write_trace_event(
            trace_path,
            {
                "node": "grade_evidence",
                "evidence_grade": {"verdict": "sufficient"},
                "latency_ms": 3.0,
            },
        )
        write_trace_event(
            trace_path,
            {"node": "generate_response", "decision": "answer", "latency_ms": 4.0},
        )
        write_trace_event(
            trace_path,
            {
                "node": "finalize",
                "decision": "answer",
                "total_latency_ms": 120.0,
                "latency_ms": 0.5,
            },
        )
        return {
            "decision": "answer",
            "response_text": "Bring your insurance card tomorrow.",
            "citations": [
                {
                    "doc_id": "doc-a",
                    "chunk_id": "chunk-expanded",
                    "span_ids": ["1", "2"],
                }
            ],
            "retrieval_ranked_chunks": [
                {"doc_id": "wrong-doc", "chunk_id": "chunk-ranked", "span_ids": ["9"]}
            ],
            "retrieved_chunks": [
                {
                    "doc_id": "doc-a",
                    "chunk_id": "chunk-expanded",
                    "span_ids": ["1", "2"],
                }
            ],
            "trace_summary": {
                "retrieval_attempts": 1,
                "graph_path": ["prepare_query"],
                "latency_ms": 120.0,
                "trace_path": str(trace_path),
                "final_query": "bring insurance card",
            },
            "latest_user_utterance": "What should I bring?",
        }

    result = asyncio.run(
        evaluate.evaluate_examples_async(
            examples,
            settings=settings,
            domain="kubernetes",
            split="validation",
            subset_name="smoke",
            notes="Test eval run",
            run_graph_func=fake_run_graph,
            now=now,
        )
    )

    output_dir = result["output_dir"]
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    prediction_records = [
        json.loads(line)
        for line in (output_dir / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    retrieval_records = [
        json.loads(line)
        for line in (output_dir / "retrieval_examples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    trace_index = json.loads(
        (output_dir / "trace_index.json").read_text(encoding="utf-8")
    )
    manual_review_rows = list(
        csv.DictReader(
            (output_dir / "manual_review.csv").read_text(encoding="utf-8").splitlines()
        )
    )
    summary = (output_dir / "summary.md").read_text(encoding="utf-8")

    assert manifest["run_id"] == "20260318-143000-kubernetes-smoke"
    assert manifest["prompt_version"] == "v-test"
    assert manifest["retrieval"]["candidate_k"] == 12
    assert manifest["retrieval"]["content_only_reasoning"] is True
    assert manifest["retrieval"]["neighbor_expansion"] is True
    assert metrics["retrieval"]["answer"]["doc_recall_at_3"] == 0.0
    assert metrics["retrieval"]["answer"]["doc_recall_at_1"] == 0.0
    assert metrics["retrieval"]["answer"]["doc_recall_at_5"] == 0.0
    assert metrics["retrieval"]["answer"]["doc_recall_at_10"] is None
    assert metrics["retrieval"]["answer"]["span_recall_at_5"] == 0.0
    assert metrics["retrieval"]["answer"]["mrr_at_5"] == 0.0
    assert metrics["generation"]["answer"]["exact_match"] == 1.0
    assert metrics["generation"]["answer"]["sacrebleu"] > 0.99
    assert metrics["generation"]["answer"]["citation_coverage"] == 1.0
    assert prediction_records[0]["metrics"]["citations_valid"] == 1.0
    assert (
        prediction_records[0]["retrieval_ranked_chunks"][0]["chunk_id"]
        == "chunk-ranked"
    )
    assert prediction_records[0]["retrieved_chunks"][0]["chunk_id"] == "chunk-expanded"
    assert (
        retrieval_records[0]["retrieval_ranked_chunks"][0]["chunk_id"] == "chunk-ranked"
    )
    assert retrieval_records[0]["retrieved_chunks"][0]["chunk_id"] == "chunk-expanded"
    assert trace_index["entries"][0]["example_id"] == "kubernetes::one::turn_2"
    assert trace_index["entries"][0]["trace_file"] == "kubernetes-one-turn-2.jsonl"
    assert trace_index["entries"][0]["graph_path"] == [
        "prepare_query",
        "retrieve_docs",
        "grade_evidence",
        "generate_response",
        "finalize",
    ]
    assert trace_index["entries"][0]["retrieval_ranked_count"] == 1
    assert trace_index["entries"][0]["retrieved_count"] == 1
    assert (
        prediction_records[0]["trace_summary"]["trace_path"]
        == "outputs/evals/runs/20260318-143000-kubernetes-smoke/traces/kubernetes-one-turn-2.jsonl"
    )
    assert (output_dir / "traces" / trace_index["entries"][0]["trace_file"]).exists()
    assert len(manual_review_rows) == 1
    assert manual_review_rows[0]["example_id"] == "kubernetes::one::turn_2"
    assert manual_review_rows[0]["failure_label"] == "wrong_doc"
    assert result["artifact_paths"]["manual_review"] == output_dir / "manual_review.csv"
    assert (
        result["artifact_paths"]["retrieval_examples"]
        == output_dir / "retrieval_examples.jsonl"
    )
    assert result["artifact_paths"]["trace_index"] == output_dir / "trace_index.json"
    assert "Headline Metrics" in summary
    assert "Recall@10: n/a (retrieval_top_k=5)" in summary
    assert "Recommendation" in summary


def test_load_eval_examples_prefers_domain_specific_subset(
    tmp_path: Path,
    make_settings,
) -> None:
    settings = make_settings(project_root=tmp_path, enabled_domains=("kubernetes",))
    subset_dir = tmp_path / "data/eval_subsets/kubernetes"
    expected = [
        {
            "example_id": "kubernetes::pods::turn_2",
            "domain": "kubernetes",
            "target_mode": "answer",
            "target_turn": {"utterance": "Pods are deployable units."},
            "gold_doc_ids": ["concepts/workloads/pods"],
            "gold_span_ids": ["concepts/workloads/pods#overview"],
        }
    ]
    write_subset_jsonl(expected, subset_dir / "smoke.jsonl")

    examples, subset_name = evaluate.load_eval_examples(
        settings,
        domain="kubernetes",
        split="validation",
        subset="smoke",
    )

    assert examples == expected
    assert subset_name == "smoke"


def test_manual_review_csv_includes_follow_up_predictions_and_answer_failures(
    monkeypatch,
    tmp_path: Path,
    make_settings,
) -> None:
    settings = make_settings(project_root=tmp_path)
    now = datetime(2026, 3, 18, 14, 30, tzinfo=timezone.utc)

    examples = [
        {
            "example_id": "kubernetes::followup::turn_2",
            "target_mode": "follow_up",
            "target_turn_id": 2,
            "latest_user_utterance": "Do you need my plate number too?",
            "target_turn": {"utterance": "Is your license still current?"},
            "gold_doc_ids": ["doc-followup"],
            "gold_span_ids": ["10"],
        },
        {
            "example_id": "kubernetes::answer::turn_2",
            "target_mode": "answer",
            "target_turn_id": 2,
            "latest_user_utterance": "What title documents do I need?",
            "target_turn": {"utterance": "Bring your title application."},
            "gold_doc_ids": ["doc-answer"],
            "gold_span_ids": ["20"],
        },
    ]

    def fake_run_graph(*, example, trace_path: Path, **kwargs):
        if example["example_id"] == "kubernetes::followup::turn_2":
            write_trace_event(
                trace_path,
                {
                    "node": "prepare_query",
                    "query": "need renewal status",
                    "latency_ms": 1.0,
                },
            )
            write_trace_event(
                trace_path,
                {
                    "node": "retrieve_docs",
                    "retrieval_attempts": 1,
                    "retrieval_ranked_count": 1,
                    "retrieved_count": 1,
                    "latency_ms": 2.0,
                },
            )
            write_trace_event(
                trace_path,
                {
                    "node": "finalize",
                    "decision": "clarify",
                    "total_latency_ms": 10.0,
                    "latency_ms": 0.5,
                },
            )
            return {
                "decision": "clarify",
                "response_text": "Is your license still current?",
                "citations": [
                    {
                        "doc_id": "doc-followup",
                        "chunk_id": "chunk-followup",
                        "span_ids": ["10"],
                    }
                ],
                "retrieval_ranked_chunks": [
                    {
                        "doc_id": "doc-followup",
                        "chunk_id": "chunk-followup",
                        "span_ids": ["10"],
                    }
                ],
                "retrieved_chunks": [
                    {
                        "doc_id": "doc-followup",
                        "chunk_id": "chunk-followup",
                        "span_ids": ["10"],
                    }
                ],
                "trace_summary": {
                    "retrieval_attempts": 1,
                    "graph_path": ["prepare_query", "retrieve_docs", "finalize"],
                    "latency_ms": 10.0,
                    "trace_path": str(trace_path),
                    "final_query": "need renewal status",
                },
                "latest_user_utterance": "Do you need my plate number too?",
            }

        write_trace_event(
            trace_path,
            {"node": "prepare_query", "query": "bring title docs", "latency_ms": 1.0},
        )
        write_trace_event(
            trace_path,
            {
                "node": "retrieve_docs",
                "retrieval_attempts": 1,
                "retrieval_ranked_count": 1,
                "retrieved_count": 1,
                "latency_ms": 2.0,
            },
        )
        write_trace_event(
            trace_path,
            {
                "node": "finalize",
                "decision": "answer",
                "total_latency_ms": 12.0,
                "latency_ms": 0.5,
            },
        )
        return {
            "decision": "answer",
            "response_text": "Bring your title application.",
            "citations": [],
            "retrieval_ranked_chunks": [
                {"doc_id": "wrong-doc", "chunk_id": "chunk-answer", "span_ids": ["999"]}
            ],
            "retrieved_chunks": [
                {"doc_id": "wrong-doc", "chunk_id": "chunk-answer", "span_ids": ["999"]}
            ],
            "trace_summary": {
                "retrieval_attempts": 1,
                "graph_path": ["prepare_query", "retrieve_docs", "finalize"],
                "latency_ms": 12.0,
                "trace_path": str(trace_path),
                "final_query": "bring title docs",
            },
            "latest_user_utterance": "What title documents do I need?",
        }

    result = asyncio.run(
        evaluate.evaluate_examples_async(
            examples,
            settings=settings,
            domain="kubernetes",
            split="validation",
            subset_name="smoke",
            notes="Test review export run",
            run_graph_func=fake_run_graph,
            now=now,
        )
    )

    rows = list(
        csv.DictReader(
            (result["output_dir"] / "manual_review.csv")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    )

    assert len(rows) == 2
    assert {row["example_id"] for row in rows} == {
        "kubernetes::followup::turn_2",
        "kubernetes::answer::turn_2",
    }
    assert (
        next(
            row for row in rows if row["example_id"] == "kubernetes::followup::turn_2"
        )["failure_label"]
        == ""
    )
    assert (
        next(row for row in rows if row["example_id"] == "kubernetes::answer::turn_2")[
            "failure_label"
        ]
        == "wrong_doc"
    )


def test_evaluate_examples_logs_progress_and_artifact_writes(
    monkeypatch,
    tmp_path: Path,
    make_settings,
) -> None:
    settings = make_settings(project_root=tmp_path)
    now = datetime(2026, 3, 18, 14, 30, tzinfo=timezone.utc)
    logged: list[str] = []

    monkeypatch.setattr(
        evaluate.logger,
        "info",
        lambda message, *args, **kwargs: logged.append(
            message % args if args else str(message)
        ),
    )

    examples = [
        {
            "example_id": "kubernetes::one::turn_2",
            "target_mode": "answer",
            "target_turn_id": 2,
            "latest_user_utterance": "What should I bring?",
            "target_turn": {"utterance": "Bring your insurance card tomorrow."},
            "gold_doc_ids": ["doc-a"],
            "gold_span_ids": ["1", "2"],
        },
        {
            "example_id": "kubernetes::two::turn_2",
            "target_mode": "answer",
            "target_turn_id": 2,
            "latest_user_utterance": "Do I need the title too?",
            "target_turn": {"utterance": "Bring your title application too."},
            "gold_doc_ids": ["doc-b"],
            "gold_span_ids": ["3"],
        },
    ]

    def fake_run_graph(*, example, **kwargs):
        chunk_id = f"chunk::{example['example_id']}"
        return {
            "decision": "answer",
            "response_text": example["target_turn"]["utterance"],
            "citations": [
                {
                    "doc_id": example["gold_doc_ids"][0],
                    "chunk_id": chunk_id,
                    "span_ids": example["gold_span_ids"],
                }
            ],
            "retrieval_ranked_chunks": [
                {
                    "doc_id": example["gold_doc_ids"][0],
                    "chunk_id": chunk_id,
                    "span_ids": example["gold_span_ids"],
                }
            ],
            "retrieved_chunks": [
                {
                    "doc_id": example["gold_doc_ids"][0],
                    "chunk_id": chunk_id,
                    "span_ids": example["gold_span_ids"],
                }
            ],
            "trace_summary": {
                "retrieval_attempts": 1,
                "graph_path": ["prepare_query", "retrieve_docs", "finalize"],
                "latency_ms": 10.0,
                "trace_path": "",
                "final_query": example["latest_user_utterance"],
            },
            "latest_user_utterance": example["latest_user_utterance"],
        }

    result = asyncio.run(
        evaluate.evaluate_examples_async(
            examples,
            settings=settings,
            domain="kubernetes",
            split="validation",
            subset_name="smoke",
            notes="Log progress test",
            run_graph_func=fake_run_graph,
            now=now,
            max_concurrency=2,
        )
    )

    assert result["run_id"] == "20260318-143000-kubernetes-smoke"
    assert any(
        "Starting eval run 20260318-143000-kubernetes-smoke" in line for line in logged
    )
    assert any(
        "Eval progress 1/2 example=kubernetes::one::turn_2" in line
        or "Eval progress 1/2 example=kubernetes::two::turn_2" in line
        for line in logged
    )
    assert any("Eval progress 2/2" in line for line in logged)
    assert any(
        "Writing eval artifacts for run 20260318-143000-kubernetes-smoke" in line
        for line in logged
    )
    assert any(
        "Eval run 20260318-143000-kubernetes-smoke complete." in line for line in logged
    )


def test_text_and_retrieval_metrics_are_deterministic() -> None:
    assert (
        evaluate.rouge_l_f1("bring proof of insurance", "bring proof of insurance")
        == 1.0
    )
    assert (
        evaluate.token_f1("bring proof of insurance", "bring proof of insurance") == 1.0
    )
    assert (
        evaluate.exact_match("Bring proof of insurance.", "bring proof of insurance")
        == 1.0
    )
    assert (
        evaluate.sacrebleu_score("bring proof of insurance", "bring proof of insurance")
        > 0.99
    )
    assert evaluate.doc_recall_at_k(["doc-a"], [{"doc_id": "doc-a"}], k=3) == 1.0
    assert evaluate.span_recall_at_k(["1", "2"], [{"span_ids": ["2"]}], k=5) == 0.5
    assert (
        evaluate.mrr_at_k(["doc-b"], [{"doc_id": "doc-x"}, {"doc_id": "doc-b"}], k=5)
        == 0.5
    )
    assert evaluate.citations_map_to_retrieved(
        [{"chunk_id": "chunk-expanded", "span_ids": ["1"]}],
        [{"chunk_id": "chunk-expanded", "span_ids": ["1"]}],
    )


def test_prediction_metrics_mark_unavailable_ranks_above_retrieval_top_k() -> None:
    metrics = evaluate._prediction_metrics(
        {
            "target_mode": "answer",
            "target_turn": {"utterance": "Bring proof of insurance."},
            "gold_doc_ids": ["doc-a"],
            "gold_span_ids": ["1"],
        },
        {
            "decision": "answer",
            "response_text": "Bring proof of insurance.",
            "citations": [{"chunk_id": "chunk-a", "span_ids": ["1"]}],
            "retrieval_ranked_chunks": [
                {"doc_id": "doc-a", "chunk_id": "chunk-a", "span_ids": ["1"]}
            ],
            "retrieved_chunks": [
                {"doc_id": "doc-a", "chunk_id": "chunk-a", "span_ids": ["1"]}
            ],
        },
        retrieval_top_k=5,
    )

    assert metrics["doc_recall_at_5"] == 1.0
    assert metrics["doc_recall_at_10"] is None


def test_evaluate_examples_async_preserves_input_order_under_concurrency(
    tmp_path: Path,
    make_settings,
) -> None:
    settings = make_settings(project_root=tmp_path)
    examples = [
        {
            "example_id": "kubernetes::first::turn_1",
            "target_mode": "answer",
            "target_turn_id": 1,
            "latest_user_utterance": "first",
            "target_turn": {"utterance": "First answer."},
            "gold_doc_ids": ["doc-1"],
            "gold_span_ids": ["span-1"],
        },
        {
            "example_id": "kubernetes::second::turn_1",
            "target_mode": "answer",
            "target_turn_id": 1,
            "latest_user_utterance": "second",
            "target_turn": {"utterance": "Second answer."},
            "gold_doc_ids": ["doc-2"],
            "gold_span_ids": ["span-2"],
        },
    ]

    async def fake_run_graph(*, example, **kwargs):
        if example["example_id"] == "kubernetes::first::turn_1":
            await asyncio.sleep(0.02)
        else:
            await asyncio.sleep(0.001)
        return {
            "example_id": example["example_id"],
            "decision": "answer",
            "response_text": example["target_turn"]["utterance"],
            "citations": [],
            "retrieval_ranked_chunks": [],
            "retrieved_chunks": [],
            "trace_summary": {
                "retrieval_attempts": 1,
                "graph_path": [],
                "latency_ms": 1.0,
            },
            "latest_user_utterance": example["latest_user_utterance"],
        }

    result = asyncio.run(
        evaluate.evaluate_examples_async(
            examples,
            settings=settings,
            domain="kubernetes",
            split="validation",
            subset_name="smoke",
            run_graph_func=fake_run_graph,
            max_concurrency=2,
        )
    )

    assert [record["example_id"] for record in result["predictions"]] == [
        "kubernetes::first::turn_1",
        "kubernetes::second::turn_1",
    ]


def test_evaluate_examples_async_records_runtime_errors_without_aborting(
    tmp_path: Path,
    make_settings,
) -> None:
    settings = make_settings(project_root=tmp_path)
    examples = [
        {
            "example_id": "kubernetes::ok::turn_1",
            "target_mode": "answer",
            "target_turn_id": 1,
            "latest_user_utterance": "ok",
            "target_turn": {"utterance": "Bring proof of insurance."},
            "gold_doc_ids": ["doc-ok"],
            "gold_span_ids": ["span-ok"],
        },
        {
            "example_id": "kubernetes::boom::turn_1",
            "target_mode": "answer",
            "target_turn_id": 1,
            "latest_user_utterance": "boom",
            "target_turn": {"utterance": "Bring your title."},
            "gold_doc_ids": ["doc-boom"],
            "gold_span_ids": ["span-boom"],
        },
    ]

    async def flaky_run_graph(*, example, **kwargs):
        if example["example_id"] == "kubernetes::boom::turn_1":
            raise RuntimeError("vectorstore offline")
        return {
            "example_id": example["example_id"],
            "decision": "answer",
            "response_text": example["target_turn"]["utterance"],
            "citations": [],
            "retrieval_ranked_chunks": [],
            "retrieved_chunks": [],
            "trace_summary": {
                "retrieval_attempts": 1,
                "graph_path": [],
                "latency_ms": 1.0,
            },
            "latest_user_utterance": example["latest_user_utterance"],
        }

    result = asyncio.run(
        evaluate.evaluate_examples_async(
            examples,
            settings=settings,
            domain="kubernetes",
            split="validation",
            subset_name="smoke",
            run_graph_func=flaky_run_graph,
            max_concurrency=2,
        )
    )

    assert [record["example_id"] for record in result["predictions"]] == [
        "kubernetes::ok::turn_1",
        "kubernetes::boom::turn_1",
    ]
    error_record = next(
        record
        for record in result["predictions"]
        if record["example_id"] == "kubernetes::boom::turn_1"
    )
    assert error_record["failure_label"] == "runtime_error"
    assert error_record["runtime_error"]["exception_type"] == "RuntimeError"
    assert error_record["runtime_error"]["error"] == "vectorstore offline"
    assert result["failure_counts"]["runtime_error"] == 1

    failures = [
        json.loads(line)
        for line in result["artifact_paths"]["failures"]
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert any(record["failure_label"] == "runtime_error" for record in failures)


def test_evaluate_examples_async_reuses_shared_runtime_resources_for_default_graph(
    monkeypatch,
    tmp_path: Path,
    make_settings,
) -> None:
    settings = make_settings(project_root=tmp_path)
    examples = [
        {
            "example_id": "kubernetes::first::turn_1",
            "target_mode": "answer",
            "target_turn_id": 1,
            "latest_user_utterance": "first",
            "target_turn": {"utterance": "First answer."},
            "gold_doc_ids": ["doc-1"],
            "gold_span_ids": ["span-1"],
        },
        {
            "example_id": "kubernetes::second::turn_1",
            "target_mode": "answer",
            "target_turn_id": 1,
            "latest_user_utterance": "second",
            "target_turn": {"utterance": "Second answer."},
            "gold_doc_ids": ["doc-2"],
            "gold_span_ids": ["span-2"],
        },
    ]

    shared_resources = object()
    resolved_configs: list[object] = []
    seen_resources: list[object] = []

    async def fake_resolve_runtime_resources(config):
        resolved_configs.append(config)
        return shared_resources

    async def fake_run_graph(*, example, _runtime_resources=None, **kwargs):
        seen_resources.append(_runtime_resources)
        return {
            "example_id": example["example_id"],
            "decision": "answer",
            "response_text": example["target_turn"]["utterance"],
            "citations": [],
            "retrieval_ranked_chunks": [],
            "retrieved_chunks": [],
            "trace_summary": {
                "retrieval_attempts": 1,
                "graph_path": [],
                "latency_ms": 1.0,
            },
            "latest_user_utterance": example["latest_user_utterance"],
        }

    monkeypatch.setattr(
        evaluate,
        "resolve_runtime_resources_async",
        fake_resolve_runtime_resources,
        raising=False,
    )
    monkeypatch.setattr(evaluate, "run_graph_async", fake_run_graph, raising=False)

    asyncio.run(
        evaluate.evaluate_examples_async(
            examples,
            settings=settings,
            domain="kubernetes",
            split="validation",
            subset_name="smoke",
            run_graph_func=evaluate.run_graph_async,
            max_concurrency=2,
        )
    )

    assert len(resolved_configs) == 1
    assert seen_resources == [shared_resources, shared_resources]


# ---------------------------------------------------------------------------
# RAG-triad eval upgrade: graded retrieval, required-point coverage, failure
# labels, backward compatibility, and the optional LLM judge path.
# ---------------------------------------------------------------------------


def test_hit_at_k_with_expected_and_acceptable_sources() -> None:
    chunks = [
        {"doc_id": "noise"},
        {"doc_id": "reference/glossary/pod"},
        {"doc_id": "concepts/workloads/pods"},
    ]
    assert (
        evaluate.hit_at_k(
            ["concepts/workloads/pods"],
            ["reference/glossary/pod"],
            chunks,
            k=5,
        )
        == 1.0
    )
    assert (
        evaluate.hit_at_k(
            ["concepts/workloads/pods"],
            ["reference/glossary/pod"],
            [{"doc_id": "noise"}],
            k=5,
        )
        == 0.0
    )
    assert evaluate.hit_at_k([], [], [{"doc_id": "noise"}], k=5) is None


def test_precision_at_k_counts_expected_and_acceptable() -> None:
    chunks = [
        {"doc_id": "concepts/workloads/pods"},
        {"doc_id": "noise"},
        {"doc_id": "reference/glossary/pod"},
    ]
    assert (
        evaluate.precision_at_k(
            ["concepts/workloads/pods"],
            ["reference/glossary/pod"],
            chunks,
            k=3,
        )
        == 2 / 3
    )


def test_graded_mrr_at_k_weights_expected_above_acceptable() -> None:
    chunks = [
        {"doc_id": "reference/glossary/pod"},
        {"doc_id": "concepts/workloads/pods"},
    ]
    # Acceptable (grade 1) at rank 1 -> 1/1 / 2 = 0.5
    assert (
        evaluate.graded_mrr_at_k(
            ["concepts/workloads/pods"],
            ["reference/glossary/pod"],
            chunks,
            k=5,
        )
        == 0.5
    )
    # Expected (grade 2) at rank 1 -> 2/1 / 2 = 1.0
    assert (
        evaluate.graded_mrr_at_k(
            ["concepts/workloads/pods"],
            [],
            [{"doc_id": "concepts/workloads/pods"}],
            k=5,
        )
        == 1.0
    )


def test_ndcg_at_k_with_graded_relevance() -> None:
    # Non-ideal order: acceptable (grade 1) before expected (grade 2).
    chunks = [
        {"doc_id": "reference/glossary/pod"},
        {"doc_id": "concepts/workloads/pods"},
        {"doc_id": "noise"},
    ]
    ndcg = evaluate.ndcg_at_k(
        ["concepts/workloads/pods"],
        ["reference/glossary/pod"],
        chunks,
        k=5,
    )
    # Ideal order only requires the expected source; acceptable sources are alternates.
    assert ndcg is not None
    assert 0.5 < ndcg < 1.0
    # Perfect ranking: expected source at rank 1. Acceptable source is not required.
    assert (
        evaluate.ndcg_at_k(
            ["concepts/workloads/pods"],
            ["reference/glossary/pod"],
            [{"doc_id": "concepts/workloads/pods"}],
            k=5,
        )
        == 1.0
    )
    # Repeated chunks from the same expected doc must not push NDCG above 1.0.
    assert (
        evaluate.ndcg_at_k(
            ["concepts/services-networking/service"],
            [],
            [
                {"doc_id": "concepts/services-networking/service"},
                {"doc_id": "concepts/services-networking/service"},
            ],
            k=5,
        )
        == 1.0
    )


def test_ndcg_at_k_caps_acceptable_and_repeated_sources() -> None:
    """Smoke-10 regression: acceptable docs and repeats cannot add extra gain."""
    ndcg = evaluate.ndcg_at_k(
        ["concepts/workloads/controllers/deployment"],
        ["tasks/run-application/update-deployment-rolling"],
        [
            {"doc_id": "concepts/workloads/controllers/deployment"},
            {"doc_id": "concepts/workloads/controllers/deployment"},
            {"doc_id": "tasks/run-application/update-deployment-rolling"},
            {"doc_id": "concepts/workloads/controllers/deployment"},
        ],
        k=5,
    )
    assert ndcg == 1.0

    duplicate_expected_ndcg = evaluate.ndcg_at_k(
        [
            "concepts/workloads/controllers/deployment",
            "concepts/workloads/controllers/deployment",
        ],
        ["tasks/run-application/update-deployment-rolling"],
        [
            {"doc_id": "concepts/workloads/controllers/deployment"},
            {"doc_id": "tasks/run-application/update-deployment-rolling"},
        ],
        k=5,
    )
    assert duplicate_expected_ndcg == 1.0


def test_metric_invariants_stay_in_unit_interval() -> None:
    retrieval_chunks = [
        {"doc_id": "expected", "chunk_id": "chunk-expected", "span_ids": ["gold"]},
        {"doc_id": "expected", "chunk_id": "chunk-expected-2", "span_ids": ["gold"]},
        {
            "doc_id": "acceptable",
            "chunk_id": "chunk-acceptable",
            "span_ids": ["alternate"],
        },
    ]
    metrics = [
        evaluate.hit_at_k(["expected"], ["acceptable"], retrieval_chunks, k=5),
        evaluate.precision_at_k(["expected"], ["acceptable"], retrieval_chunks, k=5),
        evaluate.graded_mrr_at_k(["expected"], ["acceptable"], retrieval_chunks, k=5),
        evaluate.ndcg_at_k(["expected"], ["acceptable"], retrieval_chunks, k=5),
        evaluate.citation_coverage(
            ["gold"],
            [{"chunk_id": "chunk-acceptable", "span_ids": ["alternate"]}],
            acceptable_span_ids=["alternate"],
        ),
        evaluate.required_point_coverage(
            [
                [
                    "scheduler cannot find a node",
                    "scheduler couldn't find a node",
                    "scheduler find a suitable node",
                ]
            ],
            "The scheduler couldn’t find a node that satisfies the pod requirements.",
        ),
    ]

    for metric in metrics:
        assert metric is not None
        assert 0.0 <= metric <= 1.0


def test_required_point_coverage_full_partial_and_zero() -> None:
    answer = (
        "A Pod is the smallest deployable compute object with shared storage "
        "and shared network resources."
    )
    full = [
        "smallest deployable compute object",
        "shared storage",
        "shared network resources",
    ]
    assert evaluate.required_point_coverage(full, answer) == 1.0

    partial = [
        "smallest deployable compute object",
        "one or more containers",
    ]
    assert evaluate.required_point_coverage(partial, answer) == 0.5

    assert evaluate.required_point_coverage(["missing claim entirely"], answer) == 0.0
    assert evaluate.required_point_coverage([], answer) is None


def test_forbidden_claims_hit_detects_present_claims() -> None:
    answer = "A Pod is a virtual machine that runs containers."
    assert evaluate.forbidden_claims_hit(["Pod is a virtual machine"], answer) == 1.0
    assert (
        evaluate.forbidden_claims_hit(["Pod is a virtual machine"], "A Pod is a unit.")
        == 0.0
    )
    assert evaluate.forbidden_claims_hit([], answer) is None


def test_prediction_metrics_legacy_examples_unchanged_without_rag_fields() -> None:
    metrics = evaluate._prediction_metrics(
        {
            "target_mode": "answer",
            "target_turn": {"utterance": "Bring proof of insurance."},
            "gold_doc_ids": ["doc-a"],
            "gold_span_ids": ["1"],
        },
        {
            "decision": "answer",
            "response_text": "Bring proof of insurance.",
            "citations": [{"chunk_id": "chunk-a", "span_ids": ["1"]}],
            "retrieval_ranked_chunks": [
                {"doc_id": "doc-a", "chunk_id": "chunk-a", "span_ids": ["1"]}
            ],
            "retrieved_chunks": [
                {"doc_id": "doc-a", "chunk_id": "chunk-a", "span_ids": ["1"]}
            ],
        },
        retrieval_top_k=5,
    )

    # Legacy metrics present, RAG triad absent.
    assert metrics["doc_recall_at_5"] == 1.0
    assert metrics["end_to_end_success"] == 1.0
    assert "context_relevance" not in metrics
    assert "faithfulness" not in metrics
    assert "answer_correctness" not in metrics
    assert "hit_at_k" not in metrics


def test_prediction_metrics_rag_example_emits_triad_and_graded_metrics() -> None:
    metrics = evaluate._prediction_metrics(
        {
            "target_mode": "answer",
            "target_turn": {
                "utterance": "A Pod is the smallest deployable compute object."
            },
            "gold_doc_ids": ["concepts/workloads/pods"],
            "gold_span_ids": ["concepts/workloads/pods#overview"],
            "expected_sources": ["concepts/workloads/pods"],
            "acceptable_sources": ["reference/glossary/pod"],
            "required_points": ["smallest deployable compute object"],
            "forbidden_claims": [],
            "answer_type": "definition",
        },
        {
            "decision": "answer",
            "response_text": "A Pod is the smallest deployable compute object.",
            "citations": [
                {
                    "doc_id": "concepts/workloads/pods",
                    "chunk_id": "chunk-pod",
                    "span_ids": ["concepts/workloads/pods#overview"],
                }
            ],
            "retrieval_ranked_chunks": [
                {
                    "doc_id": "concepts/workloads/pods",
                    "chunk_id": "chunk-pod",
                    "span_ids": ["concepts/workloads/pods#overview"],
                }
            ],
            "retrieved_chunks": [
                {
                    "doc_id": "concepts/workloads/pods",
                    "chunk_id": "chunk-pod",
                    "span_ids": ["concepts/workloads/pods#overview"],
                }
            ],
        },
        retrieval_top_k=5,
    )

    assert metrics["hit_at_k"] == 1.0
    assert metrics["precision_at_k"] == 1.0
    assert metrics["ndcg_at_k"] == 1.0
    assert metrics["context_relevance"] == 1.0
    assert metrics["faithfulness"] is None
    assert metrics["answer_relevance"] is None
    assert metrics["required_points_covered"] == 1.0
    assert metrics["answer_correctness"]["required_points_covered"] == 1.0
    assert metrics["answer_correctness"]["reference_similarity"]["rouge_l"] == 1.0
    # Legacy metrics still present for continuity.
    assert metrics["doc_recall_at_3"] == 1.0
    assert metrics["rouge_l"] == 1.0


def test_rag_failure_label_retrieval_miss() -> None:
    example = {
        "target_mode": "answer",
        "expected_sources": ["concepts/workloads/pods"],
        "acceptable_sources": [],
        "required_points": ["smallest deployable compute object"],
    }
    metrics = {
        "hit_at_k": 0.0,
        "doc_recall_at_3": 0.0,
        "span_recall_at_5": 0.0,
        "precision_at_k": 0.0,
        "faithfulness": 1.0,
        "citations_valid": 1.0,
        "citation_coverage": 1.0,
        "required_points_covered": 1.0,
        "answer_relevance": 1.0,
    }
    assert (
        evaluate._failure_label(example, {"decision": "answer"}, metrics)
        == "retrieval_miss"
    )


def test_rag_failure_label_right_source_wrong_section() -> None:
    example = {
        "target_mode": "answer",
        "expected_sources": ["concepts/workloads/pods"],
        "acceptable_sources": ["reference/glossary/pod"],
        "required_points": ["smallest deployable compute object"],
    }
    metrics = {
        "hit_at_k": 1.0,
        "doc_recall_at_3": 1.0,
        "span_recall_at_5": 0.0,
        "precision_at_k": 0.5,
        "faithfulness": 1.0,
        "citations_valid": 1.0,
        "citation_coverage": 1.0,
        "required_points_covered": 1.0,
        "answer_relevance": 1.0,
    }
    assert (
        evaluate._failure_label(example, {"decision": "answer"}, metrics)
        == "right_source_wrong_section"
    )


def test_rag_failure_label_weak_citations_and_unfaithful() -> None:
    example = {
        "target_mode": "answer",
        "expected_sources": ["concepts/workloads/pods"],
        "acceptable_sources": [],
        "required_points": ["smallest deployable compute object"],
    }
    base = {
        "hit_at_k": 1.0,
        "doc_recall_at_3": 1.0,
        "span_recall_at_5": 1.0,
        "precision_at_k": 1.0,
        "required_points_covered": 1.0,
        "answer_relevance": 1.0,
    }
    weak = {
        **base,
        "faithfulness": 1.0,
        "citations_valid": 0.0,
        "citation_coverage": 0.0,
    }
    assert (
        evaluate._failure_label(example, {"decision": "answer"}, weak)
        == "weak_citations"
    )
    unfaithful = {
        **base,
        "faithfulness": 0.0,
        "citations_valid": 1.0,
        "citation_coverage": 1.0,
    }
    assert (
        evaluate._failure_label(example, {"decision": "answer"}, unfaithful)
        == "unfaithful_answer"
    )


def test_rag_failure_label_incomplete_and_irrelevant() -> None:
    example = {
        "target_mode": "answer",
        "expected_sources": ["concepts/workloads/pods"],
        "acceptable_sources": [],
        "required_points": ["smallest deployable compute object"],
    }
    base = {
        "hit_at_k": 1.0,
        "doc_recall_at_3": 1.0,
        "span_recall_at_5": 1.0,
        "precision_at_k": 1.0,
        "faithfulness": 1.0,
        "citations_valid": 1.0,
        "citation_coverage": 1.0,
        "answer_relevance": 1.0,
    }
    incomplete = {**base, "required_points_covered": 0.5}
    assert (
        evaluate._failure_label(example, {"decision": "answer"}, incomplete)
        == "incomplete_answer"
    )
    irrelevant = {**base, "answer_relevance": 0.5, "required_points_covered": 1.0}
    assert (
        evaluate._failure_label(example, {"decision": "answer"}, irrelevant)
        == "irrelevant_answer"
    )


def test_legacy_failure_labels_preserved_when_no_rag_fields() -> None:
    example = {
        "target_mode": "answer",
        "gold_doc_ids": ["doc-a"],
        "gold_span_ids": ["1"],
        "turns_before_target": [],
    }
    metrics = {
        "doc_recall_at_3": 0.0,
        "span_recall_at_5": 0.0,
        "citations_valid": 1.0,
        "citation_coverage": 1.0,
        "end_to_end_success": 0.0,
    }
    assert (
        evaluate._failure_label(example, {"decision": "answer"}, metrics) == "wrong_doc"
    )


def test_rag_triad_judge_disabled_returns_none() -> None:
    judge = evaluate.RAGTriadJudge.disabled()
    assert judge.enabled is False
    import asyncio

    verdict = asyncio.run(
        judge.evaluate(
            query="q",
            answer="a",
            context="c",
            required_points=[],
            forbidden_claims=[],
        )
    )
    assert verdict is None


def test_rag_triad_judge_enabled_overrides_deterministic_metrics() -> None:
    class FakeResult:
        def __init__(self, score: float, reason: str):
            self.score = score
            self.reason = reason

    class FakeInner:
        async def faithfulness(self, context, answer):
            return FakeResult(0.8, "mostly faithful")

        async def answer_relevance(self, query, answer):
            return FakeResult(0.9, "on topic")

        async def context_relevance(self, query, context):
            return FakeResult(0.7, "useful")

    judge = evaluate.RAGTriadJudge(inner=FakeInner())
    assert judge.enabled is True
    metrics = evaluate._prediction_metrics(
        {
            "target_mode": "answer",
            "target_turn": {"utterance": "A Pod is the smallest deployable object."},
            "gold_doc_ids": ["concepts/workloads/pods"],
            "gold_span_ids": ["concepts/workloads/pods#overview"],
            "expected_sources": ["concepts/workloads/pods"],
            "acceptable_sources": [],
            "required_points": ["smallest deployable object"],
            "forbidden_claims": [],
            "answer_type": "definition",
        },
        {
            "decision": "answer",
            "response_text": "A Pod is the smallest deployable object.",
            "citations": [
                {
                    "doc_id": "concepts/workloads/pods",
                    "chunk_id": "chunk-pod",
                    "span_ids": ["concepts/workloads/pods#overview"],
                }
            ],
            "retrieval_ranked_chunks": [
                {"doc_id": "concepts/workloads/pods", "chunk_id": "chunk-pod"}
            ],
            "retrieved_chunks": [
                {"doc_id": "concepts/workloads/pods", "chunk_id": "chunk-pod"}
            ],
        },
        retrieval_top_k=5,
        judge_verdict={
            "faithful": 0.8,
            "answer_relevant": 0.9,
            "context_relevant": 0.7,
            "required_points_covered": None,
            "unsupported_claims": [],
            "rationale": "judge rationale",
        },
    )
    assert metrics["faithfulness"] == 0.8
    assert metrics["answer_relevance"] == 0.9
    assert metrics["context_relevance"] == 0.7
    assert metrics["judge"]["rationale"] == "judge rationale"


def test_evaluate_kubernetes_smoke_emits_rag_triad_buckets_and_review_columns(
    tmp_path: Path,
    make_settings,
) -> None:
    """Integration: mocked kubernetes smoke eval produces all four RAG buckets,
    a summary that explains retrieval/grounding/relevance/correctness failures,
    and a manual review CSV with the new RAG review columns.
    """
    settings = make_settings(project_root=tmp_path, enabled_domains=("kubernetes",))
    now = datetime(2026, 6, 28, 12, 0, tzinfo=timezone.utc)

    examples = [
        {
            "example_id": "kubernetes::pods::turn_2",
            "domain": "kubernetes",
            "target_mode": "answer",
            "target_turn_id": 2,
            "latest_user_utterance": "What is a Kubernetes Pod?",
            "target_turn": {
                "utterance": (
                    "A Pod is the smallest deployable compute object in Kubernetes "
                    "and represents one or more containers with shared storage and "
                    "network resources."
                )
            },
            "gold_doc_ids": ["concepts/workloads/pods"],
            "gold_span_ids": ["concepts/workloads/pods#overview"],
            "expected_sources": ["concepts/workloads/pods"],
            "acceptable_sources": ["reference/glossary/pod"],
            "required_points": [
                "smallest deployable compute object",
                "one or more containers",
                "shared storage",
                "shared network resources",
            ],
            "forbidden_claims": [],
            "answer_type": "definition",
        },
        {
            "example_id": "kubernetes::deployments::turn_2",
            "domain": "kubernetes",
            "target_mode": "answer",
            "target_turn_id": 2,
            "latest_user_utterance": "What does a Kubernetes Deployment manage?",
            "target_turn": {
                "utterance": (
                    "A Deployment manages Pods and ReplicaSets and lets you "
                    "declaratively roll out application updates."
                )
            },
            "gold_doc_ids": ["concepts/workloads/controllers/deployment"],
            "gold_span_ids": ["concepts/workloads/controllers/deployment#overview"],
            "expected_sources": ["concepts/workloads/controllers/deployment"],
            "acceptable_sources": [
                "tasks/run-application/run-stateless-application-deployment"
            ],
            "required_points": [
                "Deployment manages Pods",
                "Deployment manages ReplicaSets",
                "declaratively roll out application updates",
            ],
            "forbidden_claims": [],
            "answer_type": "definition",
        },
    ]

    def fake_run_graph(*, example, trace_path: Path, **kwargs):
        write_trace_event(
            trace_path,
            {
                "node": "prepare_query",
                "query": example["latest_user_utterance"],
                "latency_ms": 1.0,
            },
        )
        write_trace_event(
            trace_path,
            {
                "node": "retrieve_docs",
                "retrieval_attempts": 1,
                "retrieval_ranked_count": 1,
                "retrieved_count": 1,
                "latency_ms": 2.0,
            },
        )
        write_trace_event(
            trace_path,
            {
                "node": "finalize",
                "decision": "answer",
                "total_latency_ms": 50.0,
                "latency_ms": 0.5,
            },
        )
        if example["example_id"] == "kubernetes::pods::turn_2":
            doc_id = "concepts/workloads/pods"
            chunk_id = "chunk-pod"
            span_ids = ["concepts/workloads/pods#overview"]
            response = (
                "A Pod is the smallest deployable compute object in Kubernetes "
                "and represents one or more containers with shared storage and "
                "network resources."
            )
        else:
            # Deployment retrieval miss: wrong doc retrieved.
            doc_id = "concepts/workloads/pods"
            chunk_id = "chunk-wrong"
            span_ids = ["concepts/workloads/pods#overview"]
            response = "A Deployment manages Pods and ReplicaSets."
        return {
            "decision": "answer",
            "response_text": response,
            "citations": [
                {"doc_id": doc_id, "chunk_id": chunk_id, "span_ids": span_ids}
            ],
            "retrieval_ranked_chunks": [
                {"doc_id": doc_id, "chunk_id": chunk_id, "span_ids": span_ids}
            ],
            "retrieved_chunks": [
                {"doc_id": doc_id, "chunk_id": chunk_id, "span_ids": span_ids}
            ],
            "trace_summary": {
                "retrieval_attempts": 1,
                "graph_path": ["prepare_query", "retrieve_docs", "finalize"],
                "latency_ms": 50.0,
                "trace_path": str(trace_path),
                "final_query": example["latest_user_utterance"],
            },
            "latest_user_utterance": example["latest_user_utterance"],
        }

    result = asyncio.run(
        evaluate.evaluate_examples_async(
            examples,
            settings=settings,
            domain="kubernetes",
            split="validation",
            subset_name="smoke",
            notes="RAG triad smoke test",
            run_graph_func=fake_run_graph,
            now=now,
        )
    )

    output_dir = result["output_dir"]
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    predictions = [
        json.loads(line)
        for line in (output_dir / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    manual_review_rows = list(
        csv.DictReader(
            (output_dir / "manual_review.csv").read_text(encoding="utf-8").splitlines()
        )
    )
    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    # All four RAG eval buckets present in metrics JSON.
    rag = metrics["rag"]["answer"]
    assert "context_relevance" in rag
    assert "faithfulness" in rag
    assert "answer_relevance" in rag
    assert "required_points_covered" in rag
    assert rag["examples"] == 2
    assert metrics["counts"]["rag_examples"] == 2

    # Per-prediction RAG fields and graded retrieval metrics.
    pods_pred = next(p for p in predictions if p["example_id"].endswith("pods::turn_2"))
    dep_pred = next(
        p for p in predictions if p["example_id"].endswith("deployments::turn_2")
    )
    assert pods_pred["expected_sources"] == ["concepts/workloads/pods"]
    assert pods_pred["answer_type"] == "definition"
    assert pods_pred["metrics"]["hit_at_k"] == 1.0
    assert pods_pred["metrics"]["required_points_covered"] == 1.0
    assert pods_pred["metrics"]["answer_correctness"]["required_points_covered"] == 1.0
    # Pods: full success -> no failure label.
    assert pods_pred["failure_label"] is None

    # Deployment retrieval miss surfaces as retrieval_miss with graded hit@k = 0.
    assert dep_pred["metrics"]["hit_at_k"] == 0.0
    assert dep_pred["failure_label"] == "retrieval_miss"

    # Summary renders both headline RAG metrics and legacy paper-reference metrics,
    # and explains the failure as a retrieval failure.
    assert "RAG Triad" in summary
    assert "Graded Retrieval" in summary
    assert "Paper-Reference Metrics" in summary
    assert "retrieval_miss" in summary

    # Manual review CSV includes the new RAG review columns.
    review_columns = set(manual_review_rows[0].keys())
    assert "expected_sources" in review_columns
    assert "acceptable_sources" in review_columns
    assert "acceptable_span_ids" in review_columns
    assert "required_points" in review_columns
    assert "context_relevance" in review_columns
    assert "faithfulness" in review_columns
    assert "answer_relevance" in review_columns
    assert "required_points_covered" in review_columns
    assert "judge_rationale" in review_columns
    # Only the failing deployment example lands in manual review.
    assert len(manual_review_rows) == 1
    assert manual_review_rows[0]["example_id"] == "kubernetes::deployments::turn_2"
    assert manual_review_rows[0]["failure_label"] == "retrieval_miss"

    # Manifest records RAG eval enablement.
    assert manifest["rag_eval"]["enabled"] is True
    assert manifest["rag_eval"]["judge_enabled"] is False
    assert manifest["rag_eval"]["answer_types"] == ["definition"]


# ---------------------------------------------------------------------------
# Alias groups for required points and acceptable_span_ids alternates.
# ---------------------------------------------------------------------------


def test_required_point_coverage_legacy_strings_unchanged() -> None:
    """Legacy list[str] required_points still work exactly as before."""
    answer = (
        "A Pod is the smallest deployable compute object with shared storage "
        "and shared network resources."
    )
    full = [
        "smallest deployable compute object",
        "shared storage",
        "shared network resources",
    ]
    assert evaluate.required_point_coverage(full, answer) == 1.0

    partial = [
        "smallest deployable compute object",
        "one or more containers",
    ]
    assert evaluate.required_point_coverage(partial, answer) == 0.5

    assert evaluate.required_point_coverage(["missing claim entirely"], answer) == 0.0
    assert evaluate.required_point_coverage([], answer) is None


def test_required_point_coverage_alias_group_covered_by_any_phrase() -> None:
    """An alias group is covered when any single phrase matches."""
    answer = "A Pod is the smallest deployable object in Kubernetes."
    points = [
        ["smallest deployable compute object", "smallest deployable object"],
    ]
    assert evaluate.required_point_coverage(points, answer) == 1.0

    # First phrase matches, second doesn't — still covered.
    answer2 = "A Pod is the smallest deployable compute object."
    assert evaluate.required_point_coverage(points, answer2) == 1.0


def test_required_point_coverage_alias_group_counts_as_one_point() -> None:
    """An alias group counts as a single required point, not multiple."""
    answer = "A Pod is the smallest deployable object with shared volumes."
    points = [
        ["smallest deployable compute object", "smallest deployable object"],
        ["shared storage", "shared volumes"],
        "one or more containers",  # not covered
    ]
    # 2 of 3 points covered (alias groups each count as one).
    assert evaluate.required_point_coverage(points, answer) == 2 / 3


def test_required_point_coverage_alias_group_no_match_returns_zero() -> None:
    """When no alias phrase matches, the point is uncovered."""
    answer = "A Pod is a unit that runs one or more containers."
    points = [
        ["smallest deployable compute object", "smallest deployable object"],
        "one or more containers",
    ]
    # "one or more containers" is covered; alias group is not.
    assert evaluate.required_point_coverage(points, answer) == 0.5

    # No points covered at all.
    answer_bare = "A Pod is a unit."
    points_all_miss = [
        ["smallest deployable compute object", "smallest deployable object"],
        ["shared storage", "shared volumes"],
    ]
    assert evaluate.required_point_coverage(points_all_miss, answer_bare) == 0.0


def test_required_point_coverage_pods_unique_ip_address_alias() -> None:
    answer = (
        "A Pod is the smallest deployable compute object with one or more "
        "containers, shared storage, and a unique IP address."
    )
    points = [
        ["smallest deployable compute object", "smallest deployable object"],
        "one or more containers",
        ["shared storage", "shared volumes"],
        [
            "shared network resources",
            "unique IP address",
            "unique network IP address",
            "shared IP address and port space",
        ],
    ]
    assert evaluate.required_point_coverage(points, answer) == 1.0


def test_rag_eval_fields_public_shape_supports_alias_groups_and_spans() -> None:
    from support_graph.types import RAGEvalFields, RequiredPoint

    hints = get_type_hints(RAGEvalFields)
    assert RequiredPoint == str | list[str]
    assert hints["required_points"] == list[RequiredPoint]
    assert hints["acceptable_span_ids"] == list[str]

    fields: RAGEvalFields = {
        "expected_sources": ["concepts/workloads/pods"],
        "acceptable_sources": ["reference/glossary/pod"],
        "acceptable_span_ids": ["concepts/workloads/pods#what-is-a-pod"],
        "required_points": [
            "one or more containers",
            ["shared network resources", "unique IP address"],
        ],
        "forbidden_claims": [],
        "answer_type": "definition",
    }
    assert fields["required_points"][1] == [
        "shared network resources",
        "unique IP address",
    ]
    assert fields["acceptable_span_ids"] == ["concepts/workloads/pods#what-is-a-pod"]


def test_manual_review_rows_include_acceptable_span_ids() -> None:
    rows = evaluate._manual_review_rows(
        "run-1",
        [
            {
                "example_id": "kubernetes::pods::turn_2",
                "target_mode": "answer",
                "decision": "answer",
                "failure_label": "incomplete_answer",
                "latest_user_utterance": "What is a Kubernetes Pod?",
                "response_text": "A Pod has one or more containers.",
                "gold_doc_ids": ["concepts/workloads/pods"],
                "gold_span_ids": ["concepts/workloads/pods#overview"],
                "acceptable_span_ids": ["concepts/workloads/pods#what-is-a-pod"],
                "expected_sources": ["concepts/workloads/pods"],
                "acceptable_sources": ["reference/glossary/pod"],
                "required_points": [["shared network resources", "unique IP address"]],
                "forbidden_claims": [],
                "answer_type": "definition",
                "retrieval_ranked_chunks": [],
                "retrieved_chunks": [],
                "citations": [],
                "trace_summary": {},
                "metrics": {},
            }
        ],
    )
    assert rows[0]["acceptable_span_ids"] == "concepts/workloads/pods#what-is-a-pod"


def test_span_recall_accepts_acceptable_span_ids() -> None:
    """RAG span recall counts acceptable spans as equivalent substitutes."""
    gold = ["concepts/workloads/pods#overview"]
    acceptable = ["concepts/workloads/pods#what-is-a-pod"]
    chunks = [{"span_ids": ["concepts/workloads/pods#what-is-a-pod"]}]
    # Retrieving the acceptable span gives full recall (capped at 1.0).
    assert (
        evaluate.span_recall_at_k(gold, chunks, k=5, acceptable_span_ids=acceptable)
        == 1.0
    )
    # Retrieving the gold span still gives full recall.
    chunks_gold = [{"span_ids": ["concepts/workloads/pods#overview"]}]
    assert (
        evaluate.span_recall_at_k(
            gold, chunks_gold, k=5, acceptable_span_ids=acceptable
        )
        == 1.0
    )
    # Retrieving neither gives zero.
    chunks_miss = [{"span_ids": ["concepts/workloads/pods#other"]}]
    assert (
        evaluate.span_recall_at_k(
            gold, chunks_miss, k=5, acceptable_span_ids=acceptable
        )
        == 0.0
    )


def test_span_recall_legacy_remains_exact_without_acceptable_spans() -> None:
    """Legacy span recall (no acceptable_span_ids) is unchanged."""
    gold = ["concepts/workloads/pods#overview"]
    chunks = [{"span_ids": ["concepts/workloads/pods#what-is-a-pod"]}]
    # Without acceptable spans, the alternate section does not count.
    assert evaluate.span_recall_at_k(gold, chunks, k=5) == 0.0
    assert evaluate.span_recall_at_k(gold, chunks, k=5, acceptable_span_ids=None) == 0.0
    assert evaluate.span_recall_at_k(gold, chunks, k=5, acceptable_span_ids=[]) == 0.0


def test_citation_coverage_accepts_acceptable_span_ids() -> None:
    """RAG citation coverage counts acceptable spans as equivalent."""
    gold = ["concepts/workloads/pods#overview"]
    acceptable = ["concepts/workloads/pods#what-is-a-pod"]
    citations = [{"span_ids": ["concepts/workloads/pods#what-is-a-pod"]}]
    assert (
        evaluate.citation_coverage(gold, citations, acceptable_span_ids=acceptable)
        == 1.0
    )
    # Legacy: without acceptable spans, alternate section does not count.
    assert evaluate.citation_coverage(gold, citations) == 0.0


def test_prediction_metrics_passes_acceptable_spans_for_rag_examples() -> None:
    """_prediction_metrics uses acceptable_span_ids for RAG examples only."""
    # RAG example: acceptable span should boost span_recall and citation_coverage.
    rag_metrics = evaluate._prediction_metrics(
        {
            "target_mode": "answer",
            "target_turn": {"utterance": "A Pod is the smallest deployable object."},
            "gold_doc_ids": ["concepts/workloads/pods"],
            "gold_span_ids": ["concepts/workloads/pods#overview"],
            "acceptable_span_ids": ["concepts/workloads/pods#what-is-a-pod"],
            "expected_sources": ["concepts/workloads/pods"],
            "acceptable_sources": [],
            "required_points": ["smallest deployable object"],
            "forbidden_claims": [],
            "answer_type": "definition",
        },
        {
            "decision": "answer",
            "response_text": "A Pod is the smallest deployable object.",
            "citations": [
                {
                    "doc_id": "concepts/workloads/pods",
                    "chunk_id": "chunk-pod",
                    "span_ids": ["concepts/workloads/pods#what-is-a-pod"],
                }
            ],
            "retrieval_ranked_chunks": [
                {
                    "doc_id": "concepts/workloads/pods",
                    "chunk_id": "chunk-pod",
                    "span_ids": ["concepts/workloads/pods#what-is-a-pod"],
                }
            ],
            "retrieved_chunks": [
                {
                    "doc_id": "concepts/workloads/pods",
                    "chunk_id": "chunk-pod",
                    "span_ids": ["concepts/workloads/pods#what-is-a-pod"],
                }
            ],
        },
        retrieval_top_k=5,
    )
    assert rag_metrics["span_recall_at_5"] == 1.0
    assert rag_metrics["citation_coverage"] == 1.0

    # Legacy example: acceptable_span_ids is ignored even if present.
    legacy_metrics = evaluate._prediction_metrics(
        {
            "target_mode": "answer",
            "target_turn": {"utterance": "A Pod is the smallest deployable object."},
            "gold_doc_ids": ["concepts/workloads/pods"],
            "gold_span_ids": ["concepts/workloads/pods#overview"],
            "acceptable_span_ids": ["concepts/workloads/pods#what-is-a-pod"],
        },
        {
            "decision": "answer",
            "response_text": "A Pod is the smallest deployable object.",
            "citations": [
                {
                    "doc_id": "concepts/workloads/pods",
                    "chunk_id": "chunk-pod",
                    "span_ids": ["concepts/workloads/pods#what-is-a-pod"],
                }
            ],
            "retrieval_ranked_chunks": [
                {
                    "doc_id": "concepts/workloads/pods",
                    "chunk_id": "chunk-pod",
                    "span_ids": ["concepts/workloads/pods#what-is-a-pod"],
                }
            ],
            "retrieved_chunks": [
                {
                    "doc_id": "concepts/workloads/pods",
                    "chunk_id": "chunk-pod",
                    "span_ids": ["concepts/workloads/pods#what-is-a-pod"],
                }
            ],
        },
        retrieval_top_k=5,
    )
    # Legacy: acceptable span does NOT count; span_recall and citation_coverage are 0.
    assert legacy_metrics["span_recall_at_5"] == 0.0
    assert legacy_metrics["citation_coverage"] == 0.0


# ---------------------------------------------------------------------------
# Integration-style eval tests for the Kubernetes smoke rubric cleanup.
# ---------------------------------------------------------------------------


def _kube_smoke_run(
    examples: list[dict],
    *,
    make_settings,
    tmp_path: Path,
    run_graph_func,
) -> dict:
    settings = make_settings(project_root=tmp_path, enabled_domains=("kubernetes",))
    now = datetime(2026, 6, 28, 12, 0, tzinfo=timezone.utc)
    return asyncio.run(
        evaluate.evaluate_examples_async(
            examples,
            settings=settings,
            domain="kubernetes",
            split="validation",
            subset_name="smoke",
            notes="Smoke rubric cleanup test",
            run_graph_func=run_graph_func,
            now=now,
        )
    )


def _kube_trace_and_result(
    example, *, doc_id, chunk_id, span_ids, response, trace_path
):
    write_trace_event(
        trace_path,
        {
            "node": "prepare_query",
            "query": example["latest_user_utterance"],
            "latency_ms": 1.0,
        },
    )
    write_trace_event(
        trace_path,
        {
            "node": "retrieve_docs",
            "retrieval_attempts": 1,
            "retrieval_ranked_count": 1,
            "retrieved_count": 1,
            "latency_ms": 2.0,
        },
    )
    write_trace_event(
        trace_path,
        {
            "node": "finalize",
            "decision": "answer",
            "total_latency_ms": 50.0,
            "latency_ms": 0.5,
        },
    )
    return {
        "decision": "answer",
        "response_text": response,
        "citations": [{"doc_id": doc_id, "chunk_id": chunk_id, "span_ids": span_ids}],
        "retrieval_ranked_chunks": [
            {"doc_id": doc_id, "chunk_id": chunk_id, "span_ids": span_ids}
        ],
        "retrieved_chunks": [
            {"doc_id": doc_id, "chunk_id": chunk_id, "span_ids": span_ids}
        ],
        "trace_summary": {
            "retrieval_attempts": 1,
            "graph_path": ["prepare_query", "retrieve_docs", "finalize"],
            "latency_ms": 50.0,
            "trace_path": str(trace_path),
            "final_query": example["latest_user_utterance"],
        },
        "latest_user_utterance": example["latest_user_utterance"],
    }


def _load_smoke_examples() -> list[dict]:
    """Load the real Kubernetes smoke rubric from the data directory."""
    from support_graph.data.eval_subsets import load_subset_jsonl

    repo_root = Path(__file__).resolve().parent.parent.parent
    path = repo_root / "data/eval_subsets/kubernetes/smoke.jsonl"
    return load_subset_jsonl(path)


def test_smoke_pods_what_is_a_pod_span_not_right_source_wrong_section(
    tmp_path: Path,
    make_settings,
) -> None:
    """Pods retrieval hitting #what-is-a-pod (acceptable span) is not flagged
    as right_source_wrong_section."""
    examples = _load_smoke_examples()
    pods_example = next(
        ex for ex in examples if ex["example_id"] == "kubernetes::pods::turn_2"
    )

    def fake_run_graph(*, example, trace_path: Path, **kwargs):
        return _kube_trace_and_result(
            example,
            doc_id="concepts/workloads/pods",
            chunk_id="chunk-pod-what-is",
            span_ids=["concepts/workloads/pods#what-is-a-pod"],
            response=(
                "A Pod is the smallest deployable compute object in Kubernetes "
                "and represents one or more containers with shared storage and "
                "network resources."
            ),
            trace_path=trace_path,
        )

    result = _kube_smoke_run(
        [pods_example],
        make_settings=make_settings,
        tmp_path=tmp_path,
        run_graph_func=fake_run_graph,
    )
    pods_pred = result["predictions"][0]
    # Acceptable span gives full span recall and citation coverage.
    assert pods_pred["metrics"]["span_recall_at_5"] == 1.0
    assert pods_pred["metrics"]["citation_coverage"] == 1.0
    # Must NOT be right_source_wrong_section.
    assert pods_pred["failure_label"] != "right_source_wrong_section"
    # With full coverage and required points satisfied, no failure label.
    assert pods_pred["failure_label"] is None


def test_smoke_services_alias_answer_reaches_full_required_point_coverage(
    tmp_path: Path,
    make_settings,
) -> None:
    """Services answer using alternate wording (stable network identity / DNS or
    IP / changing pod IPs) reaches full required-point coverage via aliases."""
    examples = _load_smoke_examples()
    services_example = next(
        ex for ex in examples if ex["example_id"] == "kubernetes::services::turn_2"
    )

    alternate_answer = (
        "A Service exposes an application running on Pods as a network service, "
        "providing a stable network identity (DNS or IP) so clients can reach "
        "changing pod IPs."
    )

    def fake_run_graph(*, example, trace_path: Path, **kwargs):
        return _kube_trace_and_result(
            example,
            doc_id="concepts/services-networking/service",
            chunk_id="chunk-service",
            span_ids=["concepts/services-networking/service#overview"],
            response=alternate_answer,
            trace_path=trace_path,
        )

    result = _kube_smoke_run(
        [services_example],
        make_settings=make_settings,
        tmp_path=tmp_path,
        run_graph_func=fake_run_graph,
    )
    services_pred = result["predictions"][0]
    # All three required points (with aliases) are covered.
    assert services_pred["metrics"]["required_points_covered"] == 1.0
    # No incomplete_answer failure.
    assert services_pred["failure_label"] != "incomplete_answer"
    # Full success -> no failure label.
    assert services_pred["failure_label"] is None


def test_smoke_failedscheduling_suitable_node_alias_not_incomplete(
    tmp_path: Path,
    make_settings,
) -> None:
    examples = _load_smoke_examples()
    failedscheduling_example = next(
        ex
        for ex in examples
        if ex["example_id"] == "kubernetes::troubleshooting-failedscheduling::turn_2"
    )
    answer = (
        "When a Pod stays Pending with a FailedScheduling event, check resource "
        "availability and Pod resource requests, then add more nodes if resources "
        "are exhausted. Addressing these points should help the scheduler find a "
        "suitable node for the Pod."
    )

    def fake_run_graph(*, example, trace_path: Path, **kwargs):
        return _kube_trace_and_result(
            example,
            doc_id="concepts/configuration/manage-resources-containers",
            chunk_id="chunk-failedscheduling",
            span_ids=[
                "concepts/configuration/manage-resources-containers#my-pods-are-pending-with-event-message-failedscheduling"
            ],
            response=answer,
            trace_path=trace_path,
        )

    result = _kube_smoke_run(
        [failedscheduling_example],
        make_settings=make_settings,
        tmp_path=tmp_path,
        run_graph_func=fake_run_graph,
    )
    pred = result["predictions"][0]
    assert pred["metrics"]["required_points_covered"] == 1.0
    assert pred["failure_label"] != "incomplete_answer"
    assert pred["failure_label"] is None


def test_smoke10_style_metric_invariants_on_mocked_predictions(
    tmp_path: Path,
    make_settings,
) -> None:
    examples = _load_smoke_examples()

    def fake_run_graph(*, example, trace_path: Path, **kwargs):
        expected_doc = example["expected_sources"][0]
        acceptable_doc = next(iter(example.get("acceptable_sources", [])), expected_doc)
        gold_span = example["gold_span_ids"][0]
        acceptable_span = next(iter(example.get("acceptable_span_ids", [])), gold_span)
        write_trace_event(
            trace_path,
            {"node": "prepare_query", "query": "smoke", "latency_ms": 1.0},
        )
        write_trace_event(
            trace_path,
            {
                "node": "retrieve_docs",
                "retrieval_attempts": 1,
                "retrieval_ranked_count": 3,
                "retrieved_count": 3,
                "latency_ms": 2.0,
            },
        )
        write_trace_event(
            trace_path,
            {
                "node": "finalize",
                "decision": "answer",
                "total_latency_ms": 10.0,
                "latency_ms": 0.5,
            },
        )
        chunks = [
            {
                "doc_id": expected_doc,
                "chunk_id": "chunk-expected",
                "span_ids": [gold_span],
            },
            {
                "doc_id": expected_doc,
                "chunk_id": "chunk-expected-repeat",
                "span_ids": [gold_span],
            },
            {
                "doc_id": acceptable_doc,
                "chunk_id": "chunk-acceptable",
                "span_ids": [acceptable_span],
            },
        ]
        return {
            "decision": "answer",
            "response_text": example["target_turn"]["utterance"],
            "citations": [
                {
                    "doc_id": expected_doc,
                    "chunk_id": "chunk-expected",
                    "span_ids": [gold_span],
                }
            ],
            "retrieval_ranked_chunks": chunks,
            "retrieved_chunks": chunks,
            "trace_summary": {
                "retrieval_attempts": 1,
                "graph_path": ["prepare_query", "retrieve_docs", "finalize"],
                "latency_ms": 10.0,
                "trace_path": str(trace_path),
                "final_query": "smoke",
            },
            "latest_user_utterance": example["latest_user_utterance"],
        }

    result = _kube_smoke_run(
        examples,
        make_settings=make_settings,
        tmp_path=tmp_path,
        run_graph_func=fake_run_graph,
    )
    assert len(result["predictions"]) == 10
    metric_names = (
        "hit_at_k",
        "precision_at_k",
        "graded_mrr_at_k",
        "ndcg_at_k",
        "citation_coverage",
        "required_points_covered",
    )
    for prediction in result["predictions"]:
        for metric_name in metric_names:
            value = prediction["metrics"][metric_name]
            assert value is not None
            assert 0.0 <= value <= 1.0
    assert result["metrics"]["rag"]["answer"]["ndcg_at_k"] == 1.0


def test_smoke_deployments_wrong_retrieval_still_retrieval_miss(
    tmp_path: Path,
    make_settings,
) -> None:
    """Deployments retrieval miss (wrong doc retrieved) still produces
    retrieval_miss — the strict rubric exposes the real retrieval problem."""
    examples = _load_smoke_examples()
    dep_example = next(
        ex for ex in examples if ex["example_id"] == "kubernetes::deployments::turn_2"
    )

    def fake_run_graph(*, example, trace_path: Path, **kwargs):
        # Wrong doc retrieved (pods instead of deployment).
        return _kube_trace_and_result(
            example,
            doc_id="concepts/workloads/pods",
            chunk_id="chunk-wrong",
            span_ids=["concepts/workloads/pods#overview"],
            response="A Deployment manages Pods and ReplicaSets.",
            trace_path=trace_path,
        )

    result = _kube_smoke_run(
        [dep_example],
        make_settings=make_settings,
        tmp_path=tmp_path,
        run_graph_func=fake_run_graph,
    )
    dep_pred = result["predictions"][0]
    assert dep_pred["metrics"]["hit_at_k"] == 0.0
    assert dep_pred["failure_label"] == "retrieval_miss"
