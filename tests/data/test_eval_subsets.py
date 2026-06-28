from pathlib import Path

from support_graph.data.eval_subsets import (
    build_subset,
    load_subset_jsonl,
    write_subset_jsonl,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_build_subset_sizes_and_target_mode_filter() -> None:
    examples = [
        {"example_id": f"kubernetes::{index}", "target_mode": "answer"}
        for index in range(30)
    ] + [{"example_id": "kubernetes::follow-up", "target_mode": "follow_up"}]

    smoke = build_subset(examples, size=25, target_mode="answer", salt="smoke")

    assert len(smoke) == 25
    assert all(example["target_mode"] == "answer" for example in smoke)


def test_build_subset_is_deterministic() -> None:
    examples = [
        {"example_id": f"kubernetes::{index}", "target_mode": "answer"}
        for index in range(30)
    ]

    first = build_subset(examples, size=25, target_mode="answer", salt="smoke")
    second = build_subset(examples, size=25, target_mode="answer", salt="smoke")

    assert first == second


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
        {"example_id": "kubernetes::one::turn_2", "target_mode": "answer"},
        {"example_id": "kubernetes::two::turn_4", "target_mode": "answer"},
    ]
    path = tmp_path / "smoke.jsonl"

    write_subset_jsonl(examples, path)

    assert load_subset_jsonl(path) == examples
