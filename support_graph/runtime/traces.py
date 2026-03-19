"""Local structured trace helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def trace_file_path(trace_dir: str | Path, run_id: str) -> Path:
    return Path(trace_dir) / f"{run_id}.jsonl"


def write_trace_event(trace_dir: str | Path, run_id: str, event: dict) -> Path:
    path = trace_file_path(trace_dir, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True))
        handle.write("\n")
    return path


def load_trace_events(path: str | Path) -> list[dict]:
    trace_path = Path(path)
    if not trace_path.exists():
        return []
    events: list[dict] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def summarize_graph_path(events: Iterable[dict]) -> str:
    nodes = [str(event["node"]) for event in events if event.get("node")]
    return " -> ".join(nodes)


def summarize_trace_events(
    events: Iterable[dict],
    *,
    trace_path: str | Path | None = None,
) -> dict:
    ordered_events = list(events)
    graph_path = [str(event["node"]) for event in ordered_events if event.get("node")]
    node_latency_ms: dict[str, list[float]] = {}
    retrieval_attempts = 0
    final_query = ""
    decision = ""
    evidence_grade = {}
    total_latency_ms = 0.0
    retrieval_ranked_count = 0
    retrieved_count = 0
    fallback_events: list[dict] = []
    observability: dict = {}

    for event in ordered_events:
        node = ""
        if event.get("node") is not None:
            node = str(event["node"]).strip()
        latency_ms = event.get("latency_ms")
        if node and latency_ms is not None:
            node_latency_ms.setdefault(node, []).append(float(latency_ms))

        raw_observability = event.get("observability")
        if isinstance(raw_observability, dict):
            observability = dict(raw_observability)

        raw_fallback = event.get("fallback")
        if isinstance(raw_fallback, dict):
            fallback = dict(raw_fallback)
            if "node" not in fallback and node:
                fallback["node"] = node
            fallback_events.append(fallback)

        if node == "prepare_query":
            query = event.get("query")
            if query is not None:
                final_query = str(query)
        elif node == "refine_query":
            refined_query = event.get("refined_query")
            if refined_query is not None:
                final_query = str(refined_query)
        elif node == "retrieve_docs":
            retrieval_attempts = max(
                retrieval_attempts, int(event.get("retrieval_attempts") or 0)
            )
            retrieval_ranked_count = int(
                event.get("retrieval_ranked_count") or retrieval_ranked_count
            )
            retrieved_count = int(event.get("retrieved_count") or retrieved_count)
        elif node == "grade_evidence":
            raw_evidence_grade = event.get("evidence_grade")
            if raw_evidence_grade is not None:
                evidence_grade = dict(raw_evidence_grade)
        elif node in {"generate_response", "resolve_without_answer", "finalize"}:
            raw_decision = event.get("decision")
            if raw_decision is not None:
                decision = str(raw_decision)

        total_latency = event.get("total_latency_ms")
        if total_latency is not None:
            total_latency_ms = float(total_latency)

    summary = {
        "graph_path": graph_path,
        "retrieval_attempts": retrieval_attempts,
        "final_query": final_query,
        "decision": decision,
        "evidence_grade": evidence_grade,
        "total_latency_ms": total_latency_ms,
        "node_latency_ms": node_latency_ms,
        "retrieval_ranked_count": retrieval_ranked_count,
        "retrieved_count": retrieved_count,
        "fallback_count": len(fallback_events),
        "fallback_nodes": [
            str(event["node"])
            for event in fallback_events
            if event.get("node") is not None
        ],
        "fallbacks": fallback_events,
    }
    if observability:
        summary["observability"] = observability
    if trace_path is not None:
        summary["trace_path"] = str(trace_path)
    return summary
