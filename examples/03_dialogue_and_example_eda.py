from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples._shared import (
    SAMPLE_DIALOGUE_ID,
    SAMPLE_EXAMPLE_ID,
    find_by_key,
    json_lines,
    load_settings,
    next_step_lines,
    print_lines,
)
from support_graph.data.dataset import load_dialogues
from support_graph.data.examples import build_turn_examples


def build_lines() -> list[str]:
    settings = load_settings()
    dialogues = load_dialogues(
        settings.dataset.root, split="validation", domains=["dmv"]
    )
    dialogue = find_by_key(dialogues, "dial_id", SAMPLE_DIALOGUE_ID)
    examples = build_turn_examples([dialogue])
    example = find_by_key(examples, "example_id", SAMPLE_EXAMPLE_ID)

    lines = [
        "SupportGraph Example 03",
        "Dialogue and derived example EDA",
        "",
        "Raw dialogue",
        f"- Dialogue ID: {dialogue['dial_id']}",
        f"- Turn count: {len(dialogue['turns'])}",
        "",
        "Turns before target",
    ]
    for turn in example["turns_before_target"]:
        lines.append(
            f"- turn_{turn['turn_id']} | {turn['role']} | {turn['da']} | {turn['utterance']}"
        )
    lines.extend(
        [
            "",
            "Derived example",
            f"- Example ID: {example['example_id']}",
            f"- latest_user_turn_id: {example['latest_user_turn_id']}",
            f"- latest_user_utterance: {example['latest_user_utterance']}",
            f"- target_mode: {example['target_mode']}",
            f"- gold_doc_ids: {example['gold_doc_ids']}",
            f"- gold_span_ids: {example['gold_span_ids']}",
            "",
            "Target turn",
        ]
    )
    lines.extend(json_lines(example["target_turn"]))
    lines.extend(next_step_lines("04_chunking_eda.py"))
    return lines


def main() -> None:
    print_lines(build_lines())


if __name__ == "__main__":
    main()
