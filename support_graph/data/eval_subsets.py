"""Deterministic subset generation for SupportGraph eval tiers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _subset_sort_key(example: dict, salt: str) -> tuple[str, str]:
    example_id = str(example.get("example_id", ""))
    digest = hashlib.sha256(f"{salt}:{example_id}".encode("utf-8")).hexdigest()
    return digest, example_id


def build_subset(
    examples: list[dict],
    size: int,
    target_mode: str = "answer",
    salt: str | None = None,
) -> list[dict]:
    filtered = [
        example for example in examples if example.get("target_mode") == target_mode
    ]
    effective_salt = salt or f"support-graph:{target_mode}:{size}"
    selected = sorted(
        filtered, key=lambda example: _subset_sort_key(example, effective_salt)
    )[:size]
    return sorted(selected, key=lambda example: example.get("example_id", ""))


def write_subset_jsonl(examples: list[dict], path: str | Path) -> None:
    subset_path = Path(path)
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    with subset_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=True))
            handle.write("\n")


def load_subset_jsonl(path: str | Path) -> list[dict]:
    subset_path = Path(path)
    if not subset_path.exists():
        raise FileNotFoundError(f"Subset file not found: {subset_path}")
    records: list[dict] = []
    for line in subset_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records
