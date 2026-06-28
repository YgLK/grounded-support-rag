from __future__ import annotations

import asyncio
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

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
            "example_id": "dmv::one::turn_2",
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
            domain="dmv",
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

    assert manifest["run_id"] == "20260318-143000-dmv-smoke"
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
    assert trace_index["entries"][0]["example_id"] == "dmv::one::turn_2"
    assert trace_index["entries"][0]["trace_file"] == "dmv-one-turn-2.jsonl"
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
        == "outputs/evals/runs/20260318-143000-dmv-smoke/traces/dmv-one-turn-2.jsonl"
    )
    assert (output_dir / "traces" / trace_index["entries"][0]["trace_file"]).exists()
    assert len(manual_review_rows) == 1
    assert manual_review_rows[0]["example_id"] == "dmv::one::turn_2"
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
            "example_id": "dmv::followup::turn_2",
            "target_mode": "follow_up",
            "target_turn_id": 2,
            "latest_user_utterance": "Do you need my plate number too?",
            "target_turn": {"utterance": "Is your license still current?"},
            "gold_doc_ids": ["doc-followup"],
            "gold_span_ids": ["10"],
        },
        {
            "example_id": "dmv::answer::turn_2",
            "target_mode": "answer",
            "target_turn_id": 2,
            "latest_user_utterance": "What title documents do I need?",
            "target_turn": {"utterance": "Bring your title application."},
            "gold_doc_ids": ["doc-answer"],
            "gold_span_ids": ["20"],
        },
    ]

    def fake_run_graph(*, example, trace_path: Path, **kwargs):
        if example["example_id"] == "dmv::followup::turn_2":
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
            domain="dmv",
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
        "dmv::followup::turn_2",
        "dmv::answer::turn_2",
    }
    assert (
        next(row for row in rows if row["example_id"] == "dmv::followup::turn_2")[
            "failure_label"
        ]
        == ""
    )
    assert (
        next(row for row in rows if row["example_id"] == "dmv::answer::turn_2")[
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
            "example_id": "dmv::one::turn_2",
            "target_mode": "answer",
            "target_turn_id": 2,
            "latest_user_utterance": "What should I bring?",
            "target_turn": {"utterance": "Bring your insurance card tomorrow."},
            "gold_doc_ids": ["doc-a"],
            "gold_span_ids": ["1", "2"],
        },
        {
            "example_id": "dmv::two::turn_2",
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
            domain="dmv",
            split="validation",
            subset_name="smoke",
            notes="Log progress test",
            run_graph_func=fake_run_graph,
            now=now,
            max_concurrency=2,
        )
    )

    assert result["run_id"] == "20260318-143000-dmv-smoke"
    assert any("Starting eval run 20260318-143000-dmv-smoke" in line for line in logged)
    assert any(
        "Eval progress 1/2 example=dmv::one::turn_2" in line
        or "Eval progress 1/2 example=dmv::two::turn_2" in line
        for line in logged
    )
    assert any("Eval progress 2/2" in line for line in logged)
    assert any(
        "Writing eval artifacts for run 20260318-143000-dmv-smoke" in line
        for line in logged
    )
    assert any(
        "Eval run 20260318-143000-dmv-smoke complete." in line for line in logged
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
            "example_id": "dmv::first::turn_1",
            "target_mode": "answer",
            "target_turn_id": 1,
            "latest_user_utterance": "first",
            "target_turn": {"utterance": "First answer."},
            "gold_doc_ids": ["doc-1"],
            "gold_span_ids": ["span-1"],
        },
        {
            "example_id": "dmv::second::turn_1",
            "target_mode": "answer",
            "target_turn_id": 1,
            "latest_user_utterance": "second",
            "target_turn": {"utterance": "Second answer."},
            "gold_doc_ids": ["doc-2"],
            "gold_span_ids": ["span-2"],
        },
    ]

    async def fake_run_graph(*, example, **kwargs):
        if example["example_id"] == "dmv::first::turn_1":
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
            domain="dmv",
            split="validation",
            subset_name="smoke",
            run_graph_func=fake_run_graph,
            max_concurrency=2,
        )
    )

    assert [record["example_id"] for record in result["predictions"]] == [
        "dmv::first::turn_1",
        "dmv::second::turn_1",
    ]


def test_evaluate_examples_async_records_runtime_errors_without_aborting(
    tmp_path: Path,
    make_settings,
) -> None:
    settings = make_settings(project_root=tmp_path)
    examples = [
        {
            "example_id": "dmv::ok::turn_1",
            "target_mode": "answer",
            "target_turn_id": 1,
            "latest_user_utterance": "ok",
            "target_turn": {"utterance": "Bring proof of insurance."},
            "gold_doc_ids": ["doc-ok"],
            "gold_span_ids": ["span-ok"],
        },
        {
            "example_id": "dmv::boom::turn_1",
            "target_mode": "answer",
            "target_turn_id": 1,
            "latest_user_utterance": "boom",
            "target_turn": {"utterance": "Bring your title."},
            "gold_doc_ids": ["doc-boom"],
            "gold_span_ids": ["span-boom"],
        },
    ]

    async def flaky_run_graph(*, example, **kwargs):
        if example["example_id"] == "dmv::boom::turn_1":
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
            domain="dmv",
            split="validation",
            subset_name="smoke",
            run_graph_func=flaky_run_graph,
            max_concurrency=2,
        )
    )

    assert [record["example_id"] for record in result["predictions"]] == [
        "dmv::ok::turn_1",
        "dmv::boom::turn_1",
    ]
    error_record = next(
        record
        for record in result["predictions"]
        if record["example_id"] == "dmv::boom::turn_1"
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
            "example_id": "dmv::first::turn_1",
            "target_mode": "answer",
            "target_turn_id": 1,
            "latest_user_utterance": "first",
            "target_turn": {"utterance": "First answer."},
            "gold_doc_ids": ["doc-1"],
            "gold_span_ids": ["span-1"],
        },
        {
            "example_id": "dmv::second::turn_1",
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
            domain="dmv",
            split="validation",
            subset_name="smoke",
            run_graph_func=evaluate.run_graph_async,
            max_concurrency=2,
        )
    )

    assert len(resolved_configs) == 1
    assert seen_resources == [shared_resources, shared_resources]
