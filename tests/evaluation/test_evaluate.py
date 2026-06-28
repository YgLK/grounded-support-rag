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
