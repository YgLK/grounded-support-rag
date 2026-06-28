from pathlib import Path

from support_graph.data.dataset import load_dialogues
from support_graph.data.examples import build_turn_examples


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT = REPO_ROOT / "multidoc2dial"


def test_build_turn_examples_for_dmv_validation() -> None:
    dialogues = load_dialogues(DATASET_ROOT, split="validation", domains=["dmv"])
    examples = build_turn_examples(dialogues)

    assert len(examples) == 1218
    assert all(example["domain"] == "dmv" for example in examples)
    assert all(example["target_turn"]["role"] == "agent" for example in examples)


def test_build_turn_examples_flattens_gold_refs_for_known_turn() -> None:
    dialogues = load_dialogues(DATASET_ROOT, split="validation", domains=["dmv"])
    examples = build_turn_examples(dialogues)
    example = next(
        item
        for item in examples
        if item["example_id"] == "dmv::1409501a35697e0ce68561e29577b90a::turn_2"
    )

    assert example["latest_user_turn_id"] == 1
    assert example["latest_user_utterance"] == "My insurance ended so what should i do"
    assert example["target_mode"] == "answer"
    assert example["gold_doc_ids"] == ["Top 5 DMV Mistakes and How to Avoid Them#3_0"]
    assert example["gold_span_ids"] == ["24", "25", "26"]


def test_build_turn_examples_supports_agent_after_agent_sequences() -> None:
    dialogues = [
        {
            "domain": "dmv",
            "dial_id": "dial-1",
            "turns": [
                {
                    "turn_id": 1,
                    "role": "user",
                    "da": "query_condition",
                    "utterance": "I moved.",
                    "references": [],
                },
                {
                    "turn_id": 2,
                    "role": "agent",
                    "da": "respond_solution",
                    "utterance": "Update your address.",
                    "references": [],
                },
                {
                    "turn_id": 3,
                    "role": "agent",
                    "da": "query_condition",
                    "utterance": "Was the license current?",
                    "references": [],
                },
            ],
        }
    ]

    examples = build_turn_examples(dialogues)

    assert len(examples) == 2
    assert examples[1]["latest_user_turn_id"] == 1
    assert examples[1]["latest_user_utterance"] == "I moved."
    assert examples[1]["target_mode"] == "follow_up"
    assert len(examples[1]["turns_before_target"]) == 2
