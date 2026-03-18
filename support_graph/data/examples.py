"""Turn-level example builder for MultiDoc2Dial dialogues."""

from __future__ import annotations

import json
from pathlib import Path


def _normalize_reference(reference: dict) -> dict:
    return {
        "label": reference.get("label", ""),
        "id_sp": str(reference.get("id_sp", "")),
        "doc_id": str(reference.get("doc_id", "")),
    }


def _normalize_turn(turn: dict) -> dict:
    return {
        "turn_id": turn.get("turn_id"),
        "role": turn.get("role", ""),
        "da": turn.get("da", ""),
        "utterance": turn.get("utterance", ""),
        "references": [
            _normalize_reference(reference)
            for reference in turn.get("references", [])
            if reference is not None
        ],
    }


def _target_mode(turn: dict) -> str:
    if str(turn.get("da", "")).startswith("respond_"):
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


def build_turn_examples(dialogues: list[dict]) -> list[dict]:
    """Build one example per agent turn."""

    examples: list[dict] = []
    for dialogue in dialogues:
        domain = dialogue.get("domain", "")
        dial_id = dialogue.get("dial_id", "")
        turns = [_normalize_turn(turn) for turn in dialogue.get("turns", []) or []]

        for index, target_turn in enumerate(turns):
            if target_turn.get("role") != "agent":
                continue

            turns_before_target = turns[:index]
            latest_user_turn = next(
                (
                    turn
                    for turn in reversed(turns_before_target)
                    if turn.get("role") == "user"
                ),
                None,
            )
            references = target_turn.get("references", [])
            gold_doc_ids = _dedupe_preserve_order(
                [reference.get("doc_id", "") for reference in references]
            )
            gold_span_ids = _dedupe_preserve_order(
                [reference.get("id_sp", "") for reference in references]
            )
            examples.append(
                {
                    "example_id": f"{domain}::{dial_id}::turn_{target_turn.get('turn_id')}",
                    "domain": domain,
                    "dial_id": dial_id,
                    "target_turn_id": target_turn.get("turn_id"),
                    "turns_before_target": turns_before_target,
                    "latest_user_turn_id": None
                    if latest_user_turn is None
                    else latest_user_turn.get("turn_id"),
                    "latest_user_utterance": None
                    if latest_user_turn is None
                    else latest_user_turn.get("utterance", ""),
                    "target_turn": target_turn,
                    "target_mode": _target_mode(target_turn),
                    "gold_doc_ids": gold_doc_ids,
                    "gold_span_ids": gold_span_ids,
                }
            )

    return examples


def write_examples_jsonl(examples: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=True))
            handle.write("\n")


def load_examples_jsonl(path: str | Path) -> list[dict]:
    example_path = Path(path)
    if not example_path.exists():
        raise FileNotFoundError(f"Example artifact not found: {example_path}")

    records: list[dict] = []
    for line in example_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def load_example_record(example_id: str, paths: list[str | Path]) -> dict:
    for path in paths:
        for record in load_examples_jsonl(path):
            if record.get("example_id") == example_id:
                return record
    raise FileNotFoundError(f"Example not found: {example_id}")
