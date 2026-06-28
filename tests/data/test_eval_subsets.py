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


def test_kubernetes_smoke10_uses_existing_chunk_spans() -> None:
    subset = load_subset_jsonl(REPO_ROOT / "data/eval_subsets/kubernetes/smoke.jsonl")
    chunk_records = load_subset_jsonl(
        REPO_ROOT / "data/derived/chunks/kubernetes.jsonl"
    )
    chunk_doc_ids = {record["doc_id"] for record in chunk_records}
    chunk_span_ids = {
        span_id for record in chunk_records for span_id in record.get("span_ids", [])
    }

    assert len(subset) == 10
    assert all(example["domain"] == "kubernetes" for example in subset)
    assert all(example["target_mode"] == "answer" for example in subset)
    assert sum(1 for example in subset if example["answer_type"] == "diagnosis") == 7

    for example in subset:
        declared_doc_ids = [
            *example.get("gold_doc_ids", []),
            *example.get("expected_sources", []),
            *example.get("acceptable_sources", []),
        ]
        declared_span_ids = [
            *example.get("gold_span_ids", []),
            *example.get("acceptable_span_ids", []),
        ]
        assert set(declared_doc_ids) <= chunk_doc_ids
        assert set(declared_span_ids) <= chunk_span_ids


def test_subset_jsonl_round_trip(tmp_path: Path) -> None:
    examples = [
        {"example_id": "dmv::one::turn_2", "target_mode": "answer"},
        {"example_id": "dmv::two::turn_4", "target_mode": "answer"},
    ]
    path = tmp_path / "smoke.jsonl"

    write_subset_jsonl(examples, path)

    assert load_subset_jsonl(path) == examples
