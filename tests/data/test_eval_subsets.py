from pathlib import Path

from support_graph.data.dataset import load_dialogues
from support_graph.data.eval_subsets import (
    build_subset,
    load_subset_jsonl,
    write_subset_jsonl,
)
from support_graph.data.examples import build_turn_examples


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT = REPO_ROOT / "multidoc2dial"


def test_build_subset_sizes_and_target_mode_filter() -> None:
    dialogues = load_dialogues(DATASET_ROOT, split="validation", domains=["dmv"])
    examples = build_turn_examples(dialogues)

    smoke = build_subset(examples, size=25, target_mode="answer", salt="smoke")
    frozen = build_subset(
        examples, size=200, target_mode="answer", salt="frozen_experiment"
    )

    assert len(smoke) == 25
    assert len(frozen) == 200
    assert all(example["target_mode"] == "answer" for example in smoke)
    assert all(example["target_mode"] == "answer" for example in frozen)


def test_build_subset_is_deterministic() -> None:
    dialogues = load_dialogues(DATASET_ROOT, split="validation", domains=["dmv"])
    examples = build_turn_examples(dialogues)

    first = build_subset(examples, size=25, target_mode="answer", salt="smoke")
    second = build_subset(examples, size=25, target_mode="answer", salt="smoke")

    assert first == second


def test_committed_subset_files_match_builder_output() -> None:
    dialogues = load_dialogues(DATASET_ROOT, split="validation", domains=["dmv"])
    examples = build_turn_examples(dialogues)

    expected_smoke = build_subset(examples, size=25, target_mode="answer", salt="smoke")
    expected_frozen = build_subset(
        examples, size=200, target_mode="answer", salt="frozen_experiment"
    )

    assert (
        load_subset_jsonl(REPO_ROOT / "data/eval_subsets/smoke.jsonl") == expected_smoke
    )
    assert (
        load_subset_jsonl(REPO_ROOT / "data/eval_subsets/frozen_experiment.jsonl")
        == expected_frozen
    )


def test_subset_jsonl_round_trip(tmp_path: Path) -> None:
    examples = [
        {"example_id": "dmv::one::turn_2", "target_mode": "answer"},
        {"example_id": "dmv::two::turn_4", "target_mode": "answer"},
    ]
    path = tmp_path / "smoke.jsonl"

    write_subset_jsonl(examples, path)

    assert load_subset_jsonl(path) == examples
