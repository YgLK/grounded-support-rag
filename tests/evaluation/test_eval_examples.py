from __future__ import annotations

from pathlib import Path

import pytest

from support_graph.evaluation.eval_examples import (
    VALID_ANSWER_TYPES,
    alias_group_phrases,
    alias_tokens_in_text,
    answer_type_distribution,
    append_candidates,
    build_chunk_index,
    load_examples,
    promote_examples,
    span_text_for_example,
    validate_examples,
    write_examples,
)


def _chunk_records() -> list[dict]:
    return [
        {
            "doc_id": "concepts/configmap",
            "chunk_id": "concepts/configmap::intro",
            "span_ids": ["span-configmap-what", "span-configmap-use"],
            "text": "A ConfigMap is an API object used to store non-confidential data in key-value pairs. You can mount a ConfigMap as a volume or use it as environment variables.",
        },
        {
            "doc_id": "concepts/secret",
            "chunk_id": "concepts/secret::intro",
            "span_ids": ["span-secret-what"],
            "text": "A Secret stores sensitive data such as passwords, OAuth tokens, and SSH keys. Use a Secret instead of a ConfigMap for confidential data.",
        },
        {
            "doc_id": "concepts/pod",
            "chunk_id": "concepts/pod::intro",
            "span_ids": ["span-pod-what"],
            "text": "A Pod is the smallest deployable unit in Kubernetes and holds one or more containers.",
        },
    ]


def _valid_example() -> dict:
    return {
        "example_id": "kubernetes::configmap::turn_2",
        "domain": "kubernetes",
        "answer_type": "definition",
        "gold_doc_ids": ["concepts/configmap"],
        "gold_span_ids": ["span-configmap-what"],
        "acceptable_span_ids": ["span-configmap-use"],
        "required_points": [
            ["ConfigMap stores non-confidential data", "key-value pairs"],
            ["mounted as a volume", "environment variables"],
        ],
    }


def test_build_chunk_index_collects_doc_and_span_ids() -> None:
    index = build_chunk_index(_chunk_records())
    assert index.doc_ids == frozenset(
        {"concepts/configmap", "concepts/secret", "concepts/pod"}
    )
    assert "span-configmap-what" in index.span_ids
    assert "span-configmap-use" in index.span_ids
    assert "ConfigMap" in index.span_text["span-configmap-what"]


def test_alias_tokens_in_text_is_bag_of_words() -> None:
    text = "A ConfigMap stores non-confidential data in key-value pairs."
    assert alias_tokens_in_text("ConfigMap stores data", text)
    assert alias_tokens_in_text("key value pairs", text)
    assert not alias_tokens_in_text("Secret confidential", text)


def test_alias_group_phrases_handles_str_and_list() -> None:
    assert alias_group_phrases("single phrase") == ["single phrase"]
    assert alias_group_phrases(["alpha", "beta"]) == ["alpha", "beta"]


def test_validate_examples_passes_for_grounded_example() -> None:
    index = build_chunk_index(_chunk_records())
    report = validate_examples([_valid_example()], index)
    assert report.ok, report.issue_lines()
    assert report.errors == []
    assert report.example_count == 1
    assert report.answer_type_distribution == {"definition": 1}


def test_validate_examples_flags_unknown_doc_id() -> None:
    index = build_chunk_index(_chunk_records())
    example = _valid_example()
    example["gold_doc_ids"] = ["concepts/nonexistent"]
    report = validate_examples([example], index)
    assert not report.ok
    assert any(issue.code == "unknown_doc_id" for issue in report.errors)


def test_validate_examples_flags_unknown_span_id() -> None:
    index = build_chunk_index(_chunk_records())
    example = _valid_example()
    example["gold_span_ids"] = ["span-does-not-exist"]
    report = validate_examples([example], index)
    assert not report.ok
    assert any(issue.code == "unknown_span_id" for issue in report.errors)


def test_validate_examples_flags_invalid_answer_type() -> None:
    index = build_chunk_index(_chunk_records())
    example = _valid_example()
    example["answer_type"] = "comparison"
    report = validate_examples([example], index)
    assert not report.ok
    assert any(issue.code == "invalid_answer_type" for issue in report.errors)
    assert "comparison" not in VALID_ANSWER_TYPES


def test_validate_examples_flags_duplicate_example_ids() -> None:
    index = build_chunk_index(_chunk_records())
    example = _valid_example()
    report = validate_examples([example, example], index)
    assert not report.ok
    assert report.duplicate_ids == ["kubernetes::configmap::turn_2"]
    assert any(issue.code == "duplicate_example_id" for issue in report.errors)


def test_validate_examples_flags_missing_example_id() -> None:
    index = build_chunk_index(_chunk_records())
    example = _valid_example()
    del example["example_id"]
    report = validate_examples([example], index)
    assert not report.ok
    assert any(issue.code == "missing_example_id" for issue in report.errors)


def test_validate_examples_flags_empty_example_id() -> None:
    index = build_chunk_index(_chunk_records())
    example = _valid_example()
    example["example_id"] = ""
    report = validate_examples([example], index)
    assert not report.ok
    assert any(issue.code == "missing_example_id" for issue in report.errors)


def test_validate_examples_flags_ungrounded_alias_group() -> None:
    index = build_chunk_index(_chunk_records())
    example = _valid_example()
    example["required_points"] = [
        ["Secret stores OAuth tokens", "passwords and SSH keys"],
    ]
    report = validate_examples([example], index)
    assert not report.ok
    assert any(issue.code == "ungrounded_alias_group" for issue in report.errors)


def test_validate_examples_warns_on_missing_required_points() -> None:
    index = build_chunk_index(_chunk_records())
    example = _valid_example()
    example["required_points"] = []
    report = validate_examples([example], index)
    assert report.ok
    assert any(issue.code == "missing_required_points" for issue in report.warnings)


def test_answer_type_distribution_counts_known_types() -> None:
    examples = [
        {"answer_type": "definition"},
        {"answer_type": "definition"},
        {"answer_type": "diagnosis"},
        {"answer_type": ""},
    ]
    assert answer_type_distribution(examples) == {"definition": 2, "diagnosis": 1}


def test_span_text_for_example_concatenates_gold_and_acceptable() -> None:
    index = build_chunk_index(_chunk_records())
    example = _valid_example()
    text = span_text_for_example(example, index)
    assert "ConfigMap" in text
    assert "environment variables" in text


def test_append_candidates_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    rows = [
        {"example_id": "kubernetes::a::turn_2", "answer_type": "definition"},
        {"example_id": "kubernetes::b::turn_2", "answer_type": "diagnosis"},
    ]
    first = append_candidates(rows, path)
    assert len(first) == 2
    second = append_candidates(rows, path)
    assert len(second) == 2
    third = append_candidates(
        [{"example_id": "kubernetes::c::turn_2", "answer_type": "procedure"}], path
    )
    assert len(third) == 3


def test_promote_examples_appends_only_new_ids_and_sorts(tmp_path: Path) -> None:
    target = tmp_path / "expanded.jsonl"
    existing = [
        {"example_id": "kubernetes::a::turn_2", "answer_type": "definition"},
    ]
    write_examples(existing, target)
    candidates = [
        {"example_id": "kubernetes::a::turn_2", "answer_type": "definition"},
        {"example_id": "kubernetes::c::turn_2", "answer_type": "procedure"},
        {"example_id": "kubernetes::b::turn_2", "answer_type": "diagnosis"},
    ]
    merged = promote_examples(candidates, target_path=target, existing=existing)
    ids = [row["example_id"] for row in merged]
    assert ids == [
        "kubernetes::a::turn_2",
        "kubernetes::b::turn_2",
        "kubernetes::c::turn_2",
    ]
    reloaded = load_examples(target)
    assert [row["example_id"] for row in reloaded] == ids


def test_load_examples_returns_empty_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_examples(tmp_path / "missing.jsonl")
