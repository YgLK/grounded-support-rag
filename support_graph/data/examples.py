"""Turn-level example builder for MultiDoc2Dial dialogues."""

from __future__ import annotations

import json
from pathlib import Path

from support_graph.data._utils import normalize_turn
from support_graph.types import Dialogue, Example, TargetMode

__all__ = [
    "build_turn_examples",
    "write_examples_jsonl",
    "load_examples_jsonl",
    "load_example_record",
]


def _target_mode(turn: dict) -> TargetMode:
    if str(turn["da"]).startswith("respond_"):
        return "answer"
    return "follow_up"


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def build_turn_examples(dialogues: list[Dialogue]) -> list[Example]:
    """Build one example per agent turn."""

    examples: list[Example] = []
    for dialogue in dialogues:
        domain = str(dialogue["domain"])
        dial_id = str(dialogue["dial_id"])
        turns = [normalize_turn(turn) for turn in dialogue["turns"]]

        for index, target_turn in enumerate(turns):
            if target_turn["role"] != "agent":
                continue

            turns_before_target = turns[:index]
            latest_user_turn = None
            for turn in reversed(turns_before_target):
                if turn["role"] == "user":
                    latest_user_turn = turn
                    break
            references = target_turn["references"]
            gold_doc_ids = _dedupe_preserve_order(
                [reference["doc_id"] for reference in references]
            )
            gold_span_ids = _dedupe_preserve_order(
                [reference["id_sp"] for reference in references]
            )
            latest_user_turn_id = None
            latest_user_utterance = None
            if latest_user_turn is not None:
                latest_user_turn_id = latest_user_turn["turn_id"]
                latest_user_utterance = latest_user_turn["utterance"]
            examples.append(
                {
                    "example_id": f"{domain}::{dial_id}::turn_{target_turn['turn_id']}",
                    "domain": domain,
                    "dial_id": dial_id,
                    "target_turn_id": target_turn["turn_id"],
                    "turns_before_target": turns_before_target,
                    "latest_user_turn_id": latest_user_turn_id,
                    "latest_user_utterance": latest_user_utterance,
                    "target_turn": target_turn,
                    "target_mode": _target_mode(target_turn),
                    "gold_doc_ids": gold_doc_ids,
                    "gold_span_ids": gold_span_ids,
                }
            )

    return examples


def write_examples_jsonl(examples: list[Example], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=True))
            handle.write("\n")


def load_examples_jsonl(path: str | Path) -> list[Example]:
    example_path = Path(path)
    if not example_path.exists():
        raise FileNotFoundError(f"Example artifact not found: {example_path}")

    records: list[Example] = []
    for line in example_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def load_example_record(example_id: str, paths: list[str | Path]) -> Example:
    for path in paths:
        for record in load_examples_jsonl(path):
            if record["example_id"] == example_id:
                return record
    raise FileNotFoundError(f"Example not found: {example_id}")
