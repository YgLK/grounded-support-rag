from __future__ import annotations

from pathlib import Path

from support_graph.evaluation.authoring import (
    DraftedExampleFields,
    DraftResult,
    SeedTopic,
    append_candidate_rows,
    build_seed_example,
    draft_example,
    draft_examples,
    load_seed_topics,
    select_gold_chunk,
    write_seed_topics,
)


def _chunks() -> list[dict]:
    return [
        {
            "doc_id": "concepts/configmap",
            "chunk_id": "concepts/configmap::intro",
            "section_id": "concepts/configmap::intro",
            "span_ids": ["span-configmap-what"],
            "text": "A ConfigMap stores non-confidential data in key-value pairs.",
        },
        {
            "doc_id": "concepts/configmap",
            "chunk_id": "concepts/configmap::use",
            "section_id": "concepts/configmap::use",
            "span_ids": ["span-configmap-use"],
            "text": "You can mount a ConfigMap as a volume or use it as env vars.",
        },
    ]


def test_seed_topic_round_trip(tmp_path: Path) -> None:
    topics = [
        SeedTopic(
            seed_id="configmap",
            question="What is a ConfigMap?",
            answer_type="definition",
        ),
        SeedTopic(
            seed_id="secret",
            question="What is a Secret?",
            answer_type="definition",
            expected_doc_id="concepts/secret",
            notes="sensitive data",
        ),
    ]
    path = tmp_path / "seeds.jsonl"
    write_seed_topics(topics, path)
    reloaded = load_seed_topics(path)
    assert reloaded == topics


def test_build_seed_example_uses_seed_question() -> None:
    seed = SeedTopic("configmap", "What is a ConfigMap?", "definition")
    example = build_seed_example(seed)
    assert example["latest_user_utterance"] == "What is a ConfigMap?"
    assert example["target_mode"] == "answer"
    assert example["turns_before_target"][0]["utterance"] == "What is a ConfigMap?"


def test_select_gold_chunk_prefers_expected_doc_id() -> None:
    chunks = _chunks() + [
        {
            "doc_id": "concepts/other",
            "chunk_id": "concepts/other::intro",
            "span_ids": ["span-other"],
            "text": "Other doc.",
        }
    ]
    gold = select_gold_chunk(chunks, expected_doc_id="concepts/configmap")
    assert gold is not None
    assert gold["doc_id"] == "concepts/configmap"


def test_select_gold_chunk_returns_none_for_empty() -> None:
    assert select_gold_chunk([]) is None


async def _fake_chat_draft(*, seed, chunks, config) -> DraftedExampleFields:
    return DraftedExampleFields(
        reference_answer="A ConfigMap stores non-confidential data in key-value pairs.",
        required_points=[
            ["ConfigMap stores non-confidential data", "key-value pairs"],
            ["mount as a volume", "env vars"],
        ],
        expected_sources=["concepts/configmap"],
        acceptable_sources=[],
        forbidden_claims=["ConfigMaps store secrets"],
    )


def _fake_retriever(*, example, config, top_k, candidate_k) -> list[dict]:
    return _chunks()


def test_draft_example_builds_grounded_candidate() -> None:
    seed = SeedTopic("configmap", "What is a ConfigMap?", "definition")
    result = _run(
        draft_example(
            seed,
            config=None,
            retriever=_fake_retriever,
            chat_draft=_fake_chat_draft,
        )
    )
    assert isinstance(result, DraftResult)
    candidate = result.candidate
    assert candidate["example_id"] == "kubernetes::draft::configmap"
    assert candidate["gold_doc_ids"] == ["concepts/configmap"]
    assert candidate["gold_span_ids"] == ["span-configmap-what"]
    assert "span-configmap-use" in candidate["acceptable_span_ids"]
    assert candidate["answer_type"] == "definition"
    prov = result.provenance
    assert prov["gold_doc_id"] == "concepts/configmap"
    assert prov["gold_span_id"] == "span-configmap-what"
    assert "concepts/configmap::intro" in prov["retrieved_chunk_ids"]


def test_draft_examples_skips_seeds_with_no_chunks() -> None:
    seed_a = SeedTopic("configmap", "What is a ConfigMap?", "definition")
    seed_b = SeedTopic("missing", "What is nothing?", "definition")

    def selective_retriever(*, example, config, top_k, candidate_k) -> list[dict]:
        if "nothing" in example["latest_user_utterance"]:
            return []
        return _chunks()

    results = _run(
        draft_examples(
            [seed_a, seed_b],
            config=None,
            retriever=selective_retriever,
            chat_draft=_fake_chat_draft,
        )
    )
    assert len(results) == 1
    assert results[0].candidate["example_id"] == "kubernetes::draft::configmap"


def test_append_candidate_rows_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    seed = SeedTopic("configmap", "What is a ConfigMap?", "definition")
    result = _run(
        draft_example(
            seed,
            config=None,
            retriever=_fake_retriever,
            chat_draft=_fake_chat_draft,
        )
    )
    first = append_candidate_rows([result], path)
    assert len(first) == 1
    second = append_candidate_rows([result], path)
    assert len(second) == 1


def _run(coro):
    import asyncio

    return (
        asyncio.get_event_loop().run_until_complete(coro)
        if False
        else asyncio.run(coro)
    )
