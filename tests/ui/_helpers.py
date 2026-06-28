from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from support_graph.artifacts import (
    build_trace_file,
    eval_report_artifacts,
    eval_run_artifacts,
    standalone_run_artifacts,
)
from support_graph.runtime.traces import write_trace_event


RUN_ID = "20260318-143000-dmv-smoke"
SECOND_RUN_ID = "20260317-090000-medicaid-frozen_experiment"
EXAMPLE_ID = "dmv::one::turn_2"
SECOND_EXAMPLE_ID = "dmv::two::turn_4"
REPORT_ID = "20260318-143000-dmv-smoke10-experiment-summary"
STANDALONE_RUN_ID = "run-7df0f6c3fb11"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
        encoding="utf-8",
    )


def build_example_artifact(
    settings,
    *,
    example_id: str = EXAMPLE_ID,
    latest_user_utterance: str = "What title form do I need?",
) -> Path:
    path = settings.paths.examples_dir / "dmv_validation.jsonl"
    write_jsonl(
        path,
        [
            {
                "example_id": example_id,
                "domain": "dmv",
                "conversation": [
                    {
                        "turn_id": 1,
                        "role": "user",
                        "utterance": latest_user_utterance,
                    }
                ],
                "latest_user_turn_id": 1,
                "latest_user_utterance": latest_user_utterance,
                "target_mode": "answer",
                "target_turn_id": 2,
                "gold_doc_ids": ["doc-a"],
                "gold_span_ids": ["1", "2"],
            }
        ],
    )
    return path


def _chunk(
    chunk_id: str,
    *,
    doc_id: str,
    rank: int | None = None,
    text: str = "Document text",
) -> dict:
    payload = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "doc_title": None,
        "section_id": None,
        "section_title": None,
        "parent_titles": [],
        "span_ids": ["1", "2"],
        "text": text,
        "original_rank": rank,
        "vector_distance": 0.267063 if rank is not None else None,
        "text_overlap_count": 2,
        "title_overlap_count": 1,
        "rerank_score": 0.167063 if rank is not None else None,
    }
    if rank is not None:
        payload["rank"] = rank
    return payload


def _trace_summary(trace_path: str, *, final_query: str) -> dict:
    return {
        "retrieval_attempts": 1,
        "final_query": final_query,
        "graph_path": [
            "prepare_query",
            "retrieve_docs",
            "grade_evidence",
            "generate_response",
            "finalize",
        ],
        "latency_ms": 120.0,
        "trace_path": trace_path,
        "fallback_count": 0,
        "fallback_nodes": [],
        "fallbacks": [],
    }


def _per_example_metrics(
    *, citations_valid: float, citation_coverage: float | None
) -> dict:
    return {
        "doc_recall_at_1": 1.0,
        "doc_recall_at_3": 1.0,
        "doc_recall_at_5": 1.0,
        "doc_recall_at_10": 1.0,
        "span_recall_at_5": 1.0,
        "mrr_at_5": 1.0,
        "rouge_l": 0.8,
        "token_f1": 0.75,
        "exact_match": 1.0,
        "sacrebleu": 0.7,
        "citation_coverage": citation_coverage,
        "citations_valid": citations_valid,
        "end_to_end_success": 0.0 if citations_valid < 1.0 else 1.0,
    }


def _prediction_record(
    example_id: str,
    trace_path: str,
    *,
    latest_user_utterance: str,
    target_text: str,
    final_query: str,
    failure_label: str | None,
) -> dict:
    return {
        "example_id": example_id,
        "target_mode": "answer",
        "target_turn_id": 2,
        "latest_user_utterance": latest_user_utterance,
        "gold_doc_ids": ["doc-a"],
        "gold_span_ids": ["1", "2"],
        "target_text": target_text,
        "decision": "answer",
        "response_text": target_text,
        "citations": [
            {
                "doc_id": "doc-a",
                "chunk_id": "chunk-expanded",
                "span_ids": ["1", "2"],
            }
        ],
        "retrieval_ranked_chunks": [
            _chunk("chunk-ranked", doc_id="doc-a", rank=1, text="Ranked text")
        ],
        "retrieved_chunks": [
            _chunk("chunk-expanded", doc_id="doc-a", text="Expanded text")
        ],
        "trace_summary": _trace_summary(trace_path, final_query=final_query),
        "metrics": _per_example_metrics(
            citations_valid=0.0 if failure_label else 1.0,
            citation_coverage=0.5 if failure_label else 1.0,
        ),
        "failure_label": failure_label,
    }


def _retrieval_record(
    example_id: str,
    *,
    latest_user_utterance: str,
    target_text: str,
    final_query: str,
    failure_label: str | None,
) -> dict:
    return {
        "example_id": example_id,
        "target_mode": "answer",
        "latest_user_utterance": latest_user_utterance,
        "target_text": target_text,
        "gold_doc_ids": ["doc-a"],
        "gold_span_ids": ["1", "2"],
        "final_query": final_query,
        "retrieval_attempts": 1,
        "retrieval_ranked_chunks": [
            _chunk("chunk-ranked", doc_id="doc-a", rank=1, text="Ranked text")
        ],
        "retrieved_chunks": [
            _chunk("chunk-expanded", doc_id="doc-a", text="Expanded text")
        ],
        "doc_recall_at_3": 1.0,
        "span_recall_at_5": 1.0,
        "mrr_at_5": 1.0,
        "failure_label": failure_label,
        "decision": "answer",
    }


def _write_trace(path: Path, *, final_query: str, decision: str) -> None:
    write_trace_event(
        path,
        {"node": "prepare_query", "query": final_query, "latency_ms": 1.0},
    )
    write_trace_event(
        path,
        {
            "node": "retrieve_docs",
            "retrieval_attempts": 1,
            "retrieval_ranked_count": 1,
            "retrieved_count": 1,
            "latency_ms": 2.0,
        },
    )
    write_trace_event(
        path,
        {
            "node": "grade_evidence",
            "evidence_grade": {"verdict": "sufficient"},
            "latency_ms": 3.0,
        },
    )
    write_trace_event(
        path,
        {"node": "generate_response", "decision": decision, "latency_ms": 4.0},
    )
    write_trace_event(
        path,
        {
            "node": "finalize",
            "decision": decision,
            "total_latency_ms": 120.0,
            "latency_ms": 0.5,
        },
    )


def build_eval_run_artifact(
    settings,
    *,
    run_id: str = RUN_ID,
    created_at: str = "2026-03-18T14:30:00+00:00",
    domain: str = "dmv",
    split: str = "validation",
    eval_subset: str = "smoke",
    subset_label: str = "Smoke 25",
    provider_type: str = "ollama",
) -> None:
    artifacts = eval_run_artifacts(settings.paths.project_root, run_id)
    artifacts.traces_dir.mkdir(parents=True, exist_ok=True)

    trace_file = build_trace_file(EXAMPLE_ID)
    second_trace_file = build_trace_file(SECOND_EXAMPLE_ID)
    first_trace_path = f"outputs/evals/runs/{run_id}/traces/{trace_file}"
    second_trace_path = f"outputs/evals/runs/{run_id}/traces/{second_trace_file}"

    write_json(
        artifacts.manifest,
        {
            "run_id": run_id,
            "created_at": created_at,
            "dataset_root": "multidoc2dial",
            "domains": [domain],
            "split": split,
            "eval_subset": eval_subset,
            "subset_label": subset_label,
            "target_modes": ["answer", "follow_up"],
            "provider": {
                "type": provider_type,
                "chat_base_url": None,
                "embedding_type": provider_type,
                "embedding_base_url": None,
                "chat_model": f"{provider_type}-chat",
                "embedding_model": f"{provider_type}-embed",
            },
            "chunking": {
                "strategy": "section_aware",
                "max_tokens_per_chunk": 512,
            },
            "retrieval": {
                "top_k": 5,
                "candidate_k": 12,
                "max_attempts": 2,
                "use_history": True,
                "content_only_reasoning": True,
                "neighbor_expansion": True,
            },
            "graph": {
                "enable_retry": True,
                "decision_policy_version": "v1",
            },
            "prompt_version": "v1",
            "notes": "UI test run",
        },
    )
    write_json(
        artifacts.metrics,
        {
            "counts": {
                "examples": 2,
                "answer_examples": 2,
                "follow_up_examples": 0,
            },
            "retrieval": {
                "answer": {
                    "doc_recall_at_1": 1.0,
                    "doc_recall_at_3": 1.0,
                    "doc_recall_at_5": 1.0,
                    "doc_recall_at_10": 1.0,
                    "span_recall_at_5": 1.0,
                    "mrr_at_5": 1.0,
                },
                "follow_up": {
                    "doc_recall_at_1": None,
                    "doc_recall_at_3": None,
                    "doc_recall_at_5": None,
                    "doc_recall_at_10": None,
                    "span_recall_at_5": None,
                    "mrr_at_5": None,
                },
                "overall": {
                    "doc_recall_at_1": 1.0,
                    "doc_recall_at_3": 1.0,
                    "doc_recall_at_5": 1.0,
                    "doc_recall_at_10": 1.0,
                    "span_recall_at_5": 1.0,
                    "mrr_at_5": 1.0,
                },
            },
            "generation": {
                "answer": {
                    "rouge_l": 0.8,
                    "token_f1": 0.75,
                    "exact_match": 1.0,
                    "sacrebleu": 0.7,
                    "citation_coverage": 0.75,
                    "end_to_end_success_rate": 0.5,
                }
            },
            "decision_distribution": {
                "overall": {"answer": 1.0},
                "answer": {"answer": 1.0},
                "follow_up": {},
            },
            "latency_ms": {
                "average": 120.0,
                "p95": 120.0,
            },
            "failure_counts": {"wrong_doc": 1},
        },
    )
    write_jsonl(
        artifacts.predictions,
        [
            _prediction_record(
                EXAMPLE_ID,
                first_trace_path,
                latest_user_utterance="What title form do I need?",
                target_text="Bring your title form.",
                final_query="title form dmv",
                failure_label="wrong_doc",
            ),
            _prediction_record(
                SECOND_EXAMPLE_ID,
                second_trace_path,
                latest_user_utterance="How do I replace my registration?",
                target_text="Complete the replacement registration request.",
                final_query="replace registration dmv",
                failure_label=None,
            ),
        ],
    )
    write_jsonl(
        artifacts.failures,
        [
            _prediction_record(
                EXAMPLE_ID,
                first_trace_path,
                latest_user_utterance="What title form do I need?",
                target_text="Bring your title form.",
                final_query="title form dmv",
                failure_label="wrong_doc",
            )
        ],
    )
    write_jsonl(
        artifacts.retrieval_examples,
        [
            _retrieval_record(
                EXAMPLE_ID,
                latest_user_utterance="What title form do I need?",
                target_text="Bring your title form.",
                final_query="title form dmv",
                failure_label="wrong_doc",
            ),
            _retrieval_record(
                SECOND_EXAMPLE_ID,
                latest_user_utterance="How do I replace my registration?",
                target_text="Complete the replacement registration request.",
                final_query="replace registration dmv",
                failure_label=None,
            ),
        ],
    )
    write_json(
        artifacts.trace_index,
        {
            "entries": [
                {
                    "example_id": EXAMPLE_ID,
                    "trace_file": trace_file,
                    "graph_path": [
                        "prepare_query",
                        "retrieve_docs",
                        "grade_evidence",
                        "generate_response",
                        "finalize",
                    ],
                    "retrieval_attempts": 1,
                    "final_query": "title form dmv",
                    "decision": "answer",
                    "failure_label": "wrong_doc",
                    "total_latency_ms": 120.0,
                    "node_latency_ms": {"retrieve_docs": [2.0]},
                    "retrieval_ranked_count": 1,
                    "retrieved_count": 1,
                    "fallback_count": 0,
                    "fallback_nodes": [],
                },
                {
                    "example_id": SECOND_EXAMPLE_ID,
                    "trace_file": second_trace_file,
                    "graph_path": [
                        "prepare_query",
                        "retrieve_docs",
                        "grade_evidence",
                        "generate_response",
                        "finalize",
                    ],
                    "retrieval_attempts": 1,
                    "final_query": "replace registration dmv",
                    "decision": "answer",
                    "failure_label": None,
                    "total_latency_ms": 120.0,
                    "node_latency_ms": {"retrieve_docs": [2.0]},
                    "retrieval_ranked_count": 1,
                    "retrieved_count": 1,
                    "fallback_count": 0,
                    "fallback_nodes": [],
                },
            ]
        },
    )

    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "run_id",
            "example_id",
            "target_mode",
            "decision",
            "failure_label",
            "latest_user_utterance",
            "response_text",
            "gold_doc_ids",
            "gold_span_ids",
            "final_query",
            "retrieval_attempts",
            "retrieval_ranked_chunk_ids",
            "retrieved_chunk_ids",
            "citation_chunk_ids",
            "decision_correct",
            "evidence_relevant",
            "no_unsupported_claims",
            "clear",
            "citations_useful",
            "reviewer_notes",
        ],
    )
    writer.writeheader()
    writer.writerow(
        {
            "run_id": run_id,
            "example_id": EXAMPLE_ID,
            "target_mode": "answer",
            "decision": "answer",
            "failure_label": "wrong_doc",
            "latest_user_utterance": "What title form do I need?",
            "response_text": "Bring your title form.",
            "gold_doc_ids": "doc-a",
            "gold_span_ids": "1,2",
            "final_query": "title form dmv",
            "retrieval_attempts": "1",
            "retrieval_ranked_chunk_ids": "chunk-ranked",
            "retrieved_chunk_ids": "chunk-expanded",
            "citation_chunk_ids": "chunk-expanded",
            "decision_correct": "",
            "evidence_relevant": "",
            "no_unsupported_claims": "",
            "clear": "",
            "citations_useful": "",
            "reviewer_notes": "",
        }
    )
    artifacts.manual_review.write_text(buffer.getvalue(), encoding="utf-8")
    artifacts.summary.write_text(
        "# Eval Summary\n\nThis run is easy to inspect.\n", encoding="utf-8"
    )

    _write_trace(
        artifacts.trace_path(trace_file),
        final_query="title form dmv",
        decision="answer",
    )
    _write_trace(
        artifacts.trace_path(second_trace_file),
        final_query="replace registration dmv",
        decision="answer",
    )


def build_eval_report_artifact(
    settings,
    *,
    report_id: str = REPORT_ID,
    created_at: str = "2026-03-18T14:30:00+00:00",
) -> None:
    artifacts = eval_report_artifacts(settings.paths.project_root, report_id)
    write_json(
        artifacts.manifest,
        {
            "report_id": report_id,
            "created_at": created_at,
            "report_type": "experiment_summary",
            "title": "Smoke-10 Experiment Summary",
            "related_run_ids": [RUN_ID],
            "domain": "dmv",
            "split": "validation",
            "subset_label": "Smoke 25",
            "notes": "Compare control vs experiment variants.",
        },
    )
    artifacts.report.write_text(
        "# Smoke-10 Experiment Summary\n\nControl recovered the gold document more often.\n",
        encoding="utf-8",
    )


def build_standalone_run_artifact(
    settings,
    *,
    run_id: str = STANDALONE_RUN_ID,
    created_at: str = "2026-03-19T10:00:00+00:00",
) -> None:
    artifacts = standalone_run_artifacts(settings.paths.project_root, run_id)
    write_json(
        artifacts.manifest,
        {
            "run_id": run_id,
            "created_at": created_at,
            "example_id": EXAMPLE_ID,
            "domain": "dmv",
            "provider": {
                "type": "ollama",
                "chat_model": "qwen3:8b-q4_K_M",
                "embedding_type": "ollama",
                "embedding_model": "qwen3-embedding:4b-q4_K_M",
            },
            "prompt_version": "v1",
            "notes": "UI test standalone run",
        },
    )
    write_json(
        artifacts.result,
        {
            "example_id": EXAMPLE_ID,
            "latest_user_utterance": "What title form do I need?",
            "decision": "answer",
            "response_text": "Bring your title form.",
            "citations": [
                {
                    "doc_id": "doc-a",
                    "chunk_id": "chunk-expanded",
                    "span_ids": ["1", "2"],
                }
            ],
            "confidence_label": "high",
            "retrieval_ranked_chunks": [
                _chunk("chunk-ranked", doc_id="doc-a", rank=1, text="Ranked text")
            ],
            "retrieved_chunks": [
                _chunk("chunk-expanded", doc_id="doc-a", text="Expanded text")
            ],
            "evidence_grade": {
                "verdict": "sufficient",
                "reason": "The retrieved evidence covers the requested form.",
                "missing_information": [],
            },
            "trace_summary": {
                "retrieval_attempts": 1,
                "final_query": "title form dmv",
                "graph_path": [
                    "prepare_query",
                    "retrieve_docs",
                    "grade_evidence",
                    "generate_response",
                    "finalize",
                ],
                "latency_ms": 87.0,
                "trace_path": f"outputs/runs/{run_id}/trace.jsonl",
                "fallback_count": 0,
                "fallback_nodes": [],
                "fallbacks": [],
            },
        },
    )
    _write_trace(artifacts.trace, final_query="title form dmv", decision="answer")
