from __future__ import annotations

from pathlib import Path

from support_graph.runtime import traces


def test_load_trace_events_missing_returns_empty_list(tmp_path: Path) -> None:
    path = tmp_path / "missing.jsonl"

    assert traces.load_trace_events(path) == []


def test_summarize_trace_events_collects_queries_latencies_and_retry_counts() -> None:
    events = [
        {
            "node": "prepare_query",
            "query": "Domain: dmv",
            "latency_ms": 1.0,
        },
        {
            "node": "retrieve_docs",
            "retrieval_attempts": 1,
            "retrieval_ranked_count": 5,
            "retrieved_count": 4,
            "latency_ms": 2.0,
        },
        {
            "node": "grade_evidence",
            "evidence_grade": {"verdict": "partial"},
            "fallback": {
                "node": "grade_evidence",
                "exception_type": "ValueError",
                "error": "schema parse failed",
            },
            "latency_ms": 3.0,
        },
        {
            "node": "refine_query",
            "refined_query": "Domain: dmv\nMissing condition: insurance status",
            "latency_ms": 4.0,
        },
        {
            "node": "retrieve_docs",
            "retrieval_attempts": 2,
            "retrieval_ranked_count": 5,
            "retrieved_count": 5,
            "latency_ms": 2.5,
        },
        {"node": "generate_response", "decision": "answer", "latency_ms": 5.0},
        {
            "node": "finalize",
            "decision": "answer",
            "total_latency_ms": 20.0,
            "latency_ms": 0.5,
        },
    ]

    summary = traces.summarize_trace_events(
        events,
        trace_path="/tmp/run-abc.jsonl",
    )

    assert summary["trace_path"] == "/tmp/run-abc.jsonl"
    assert summary["graph_path"] == [
        "prepare_query",
        "retrieve_docs",
        "grade_evidence",
        "refine_query",
        "retrieve_docs",
        "generate_response",
        "finalize",
    ]
    assert summary["retrieval_attempts"] == 2
    assert summary["final_query"].endswith("Missing condition: insurance status")
    assert summary["decision"] == "answer"
    assert summary["evidence_grade"] == {"verdict": "partial"}
    assert summary["total_latency_ms"] == 20.0
    assert summary["node_latency_ms"]["retrieve_docs"] == [2.0, 2.5]
    assert summary["retrieval_ranked_count"] == 5
    assert summary["retrieved_count"] == 5
    assert summary["fallback_count"] == 1
    assert summary["fallback_nodes"] == ["grade_evidence"]


def test_write_and_load_trace_events_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "run-123.jsonl"
    traces.write_trace_event(path, {"node": "prepare_query", "query": "q"})
    traces.write_trace_event(path, {"node": "finalize", "decision": "answer"})

    events = traces.load_trace_events(path)

    assert path.exists()
    assert [event["node"] for event in events] == ["prepare_query", "finalize"]
