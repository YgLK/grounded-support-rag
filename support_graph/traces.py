"""Local structured trace helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def write_trace_event(trace_dir: str | Path, run_id: str, event: dict) -> Path:
    path = Path(trace_dir) / f"{run_id}.jsonl"
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
    nodes = [event.get("node") for event in events if event.get("node")]
    return " -> ".join(nodes)
