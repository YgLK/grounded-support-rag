from __future__ import annotations

from types import SimpleNamespace

from support_graph.retrieval import retrieve


def test_build_query_context_includes_latest_user_need_agent_question_and_two_carry_forward_turns() -> (
    None
):
    example = {
        "domain": "dmv",
        "conversation": [
            {
                "turn_id": 1,
                "role": "user",
                "utterance": "I moved to a new address last week and need to update my records.",
            },
            {
                "turn_id": 2,
                "role": "agent",
                "utterance": "Did your vehicle registration move with you?",
            },
            {
                "turn_id": 3,
                "role": "user",
                "utterance": "Yes, I need the registration steps too and I also changed insurers.",
            },
            {
                "turn_id": 4,
                "role": "agent",
                "utterance": "Do you need the DMV address-change form as well?",
            },
            {
                "turn_id": 5,
                "role": "user",
                "utterance": "I need to update my license and registration after moving.",
            },
        ],
        "latest_user_turn_id": 5,
        "latest_user_utterance": "I need to update my license and registration after moving.",
    }

    context = retrieve.build_query_context(example)

    assert context["domain"] == "dmv"
    assert (
        context["latest_user_need"]
        == "I need to update my license and registration after moving."
    )
    assert (
        context["last_agent_question"]
        == "Do you need the DMV address-change form as well?"
    )
    assert len(context["carry_forward_context"]) == 2
    assert [turn["role"] for turn in context["carry_forward_context"]] == [
        "user",
        "user",
    ]


def test_build_query_context_excludes_short_and_duplicate_turns() -> None:
    example = {
        "domain": "dmv",
        "conversation": [
            {
                "turn_id": 1,
                "role": "user",
                "utterance": "Need title transfer steps for a car from my parent.",
            },
            {"turn_id": 2, "role": "agent", "utterance": "Okay."},
            {
                "turn_id": 3,
                "role": "user",
                "utterance": "title transfer steps for a car from my parent",
            },
            {
                "turn_id": 4,
                "role": "agent",
                "utterance": "Did the car come from out of state?",
            },
            {
                "turn_id": 5,
                "role": "user",
                "utterance": "What title transfer steps do I need for a car from my parent?",
            },
        ],
        "latest_user_turn_id": 5,
        "latest_user_utterance": "What title transfer steps do I need for a car from my parent?",
    }

    context = retrieve.build_query_context(example)

    assert context["last_agent_question"] == "Did the car come from out of state?"
    assert context["carry_forward_context"] == []


def test_build_query_renders_compact_labeled_blocks() -> None:
    example = {
        "domain": "dmv",
        "conversation": [
            {
                "turn_id": 1,
                "role": "user",
                "utterance": "My insurance lapsed last month and I got a letter.",
            },
            {
                "turn_id": 2,
                "role": "agent",
                "utterance": "Did you already restore coverage?",
            },
            {
                "turn_id": 3,
                "role": "user",
                "utterance": "What do I need to do to avoid suspension now?",
            },
        ],
        "latest_user_turn_id": 3,
        "latest_user_utterance": "What do I need to do to avoid suspension now?",
    }

    query = retrieve.build_query(example)

    assert "Domain: dmv" in query
    assert "Latest user need: What do I need to do to avoid suspension now?" in query
    assert "Last agent question: Did you already restore coverage?" in query
    assert "Recent conversation:" not in query


def test_build_metadata_filter_limits_domain_and_doc_ids() -> None:
    flt = retrieve.build_metadata_filter(domain="dmv", doc_ids=["doc-a", "doc-b"])

    assert flt["domain"] == "dmv"
    assert flt["doc_id"] == {"$in": ["doc-a", "doc-b"]}


def test_normalize_retrieval_hits_preserves_rank_and_scores() -> None:
    fake_hits = [
        SimpleNamespace(
            page_content="alpha beta",
            metadata={
                "chunk_id": "dmv::doc::sec::1::sub::0",
                "doc_id": "doc",
                "span_ids": ["1"],
            },
            score=0.12,
        ),
        {
            "page_content": "gamma delta",
            "metadata": {
                "chunk_id": "dmv::doc::sec::2::sub::0",
                "doc_id": "doc",
                "span_ids": ["2"],
            },
            "score": 0.34,
        },
    ]

    normalized = retrieve.normalize_retrieval_hits(fake_hits)

    assert normalized[0]["chunk_id"] == "dmv::doc::sec::1::sub::0"
    assert normalized[0]["score"] == 0.12
    assert normalized[0]["original_rank"] == 1
    assert normalized[1]["chunk_id"] == "dmv::doc::sec::2::sub::0"
    assert normalized[1]["span_ids"] == ["2"]


def test_retrieve_chunks_uses_candidate_k_before_returning_top_k() -> None:
    captured = {}

    class FakeVectorStore:
        def similarity_search_with_score(self, query, k=5, filter=None):
            captured["query"] = query
            captured["k"] = k
            captured["filter"] = filter
            return [
                (
                    SimpleNamespace(
                        page_content=f"text {index}",
                        metadata={
                            "chunk_id": f"dmv::doc::sec::{index}::sub::0",
                            "doc_id": "doc",
                            "section_id": str(index),
                            "section_title": f"Section {index}",
                            "span_ids": [str(index)],
                            "token_count": 20,
                        },
                    ),
                    0.10 + (index * 0.01),
                )
                for index in range(1, 8)
            ]

    example = {
        "domain": "dmv",
        "conversation": [
            {
                "turn_id": 1,
                "role": "user",
                "utterance": "I moved and need address update steps.",
            }
        ],
        "latest_user_turn_id": 1,
        "latest_user_utterance": "I moved and need address update steps.",
    }

    results = retrieve.retrieve_chunks(
        example=example,
        vectorstore=FakeVectorStore(),
        top_k=5,
        candidate_k=7,
        rerank=False,
    )

    assert captured["k"] == 7
    assert len(results) == 5
    assert results[0]["chunk_id"] == "dmv::doc::sec::1::sub::0"


def test_retrieve_chunks_merges_filtered_keyword_hits_without_duplicates() -> None:
    captured = {}

    class FakeVectorStore:
        def similarity_search_with_score(self, query, k=5, filter=None):
            captured["query"] = query
            captured["k"] = k
            captured["filter"] = filter
            return [
                (
                    SimpleNamespace(
                        page_content="vector hit",
                        metadata={
                            "chunk_id": "dmv::doc-a::sec::1::sub::0",
                            "domain": "dmv",
                            "doc_id": "doc-a",
                            "section_id": "1",
                            "section_title": "Vector",
                            "span_ids": ["1"],
                            "token_count": 20,
                        },
                    ),
                    0.1,
                )
            ]

    class FakeKeywordRetriever:
        def invoke(self, query):
            captured["keyword_query"] = query
            return [
                SimpleNamespace(
                    page_content="duplicate hit",
                    metadata={
                        "chunk_id": "dmv::doc-a::sec::1::sub::0",
                        "domain": "dmv",
                        "doc_id": "doc-a",
                    },
                ),
                SimpleNamespace(
                    page_content="keyword hit",
                    metadata={
                        "chunk_id": "dmv::doc-a::sec::2::sub::0",
                        "domain": "dmv",
                        "doc_id": "doc-a",
                        "section_id": "2",
                        "section_title": "Keyword",
                        "span_ids": ["2"],
                        "token_count": 18,
                    },
                ),
                SimpleNamespace(
                    page_content="wrong doc",
                    metadata={
                        "chunk_id": "dmv::doc-b::sec::3::sub::0",
                        "domain": "dmv",
                        "doc_id": "doc-b",
                    },
                ),
            ]

    example = {
        "domain": "dmv",
        "conversation": [
            {
                "turn_id": 1,
                "role": "user",
                "utterance": "I moved and need address update steps.",
            }
        ],
        "latest_user_turn_id": 1,
        "latest_user_utterance": "I moved and need address update steps.",
    }

    results = retrieve.retrieve_chunks(
        example=example,
        vectorstore=FakeVectorStore(),
        keyword_retriever=FakeKeywordRetriever(),
        top_k=5,
        candidate_k=7,
        doc_ids="doc-a",
        rerank=False,
    )

    assert captured["k"] == 7
    assert captured["filter"] == {"domain": "dmv", "doc_id": "doc-a"}
    assert "Domain:" not in captured["query"]
    assert captured["keyword_query"] == captured["query"]
    assert [hit["chunk_id"] for hit in results] == [
        "dmv::doc-a::sec::1::sub::0",
        "dmv::doc-a::sec::2::sub::0",
    ]


def test_rerank_retrieval_hits_penalizes_title_and_short_single_span_chunks() -> None:
    context = {
        "domain": "dmv",
        "latest_user_need": "insurance lapse registration suspension",
        "last_agent_question": "",
        "carry_forward_context": [],
    }
    hits = [
        {
            "rank": 1,
            "original_rank": 1,
            "chunk_id": "dmv::doc::sec::t_1::sub::0",
            "doc_id": "doc",
            "section_id": "t_1",
            "section_title": "What happens if my insurance ends?",
            "parent_titles": [],
            "span_ids": ["10"],
            "token_count": 8,
            "text": "What happens if my insurance ends?",
            "score": 0.20,
            "vector_distance": 0.20,
        },
        {
            "rank": 2,
            "original_rank": 2,
            "chunk_id": "dmv::doc::sec::2::sub::0",
            "doc_id": "doc",
            "section_id": "2",
            "section_title": "Insurance lapse guidance",
            "parent_titles": [],
            "span_ids": ["11", "12"],
            "token_count": 32,
            "text": "If your insurance lapses, restore coverage before driving the vehicle again.",
            "score": 0.21,
            "vector_distance": 0.21,
        },
    ]

    reranked = retrieve.rerank_retrieval_hits(hits, query_context=context)

    assert reranked[0]["chunk_id"] == "dmv::doc::sec::2::sub::0"


def test_query_context_tokens_uses_library_stopwords() -> None:
    context = {
        "domain": "dmv",
        "latest_user_need": "What do I need to do for a title transfer?",
        "last_agent_question": "How do I complete this?",
        "carry_forward_context": [],
    }

    tokens = retrieve.query_context_tokens(context)

    assert "what" not in tokens
    assert "how" not in tokens
    assert "for" not in tokens
    assert "title" in tokens
    assert "transfer" in tokens


def test_rerank_retrieval_hits_uses_text_and_title_overlap_to_lift_relevant_chunk() -> (
    None
):
    context = {
        "domain": "dmv",
        "latest_user_need": "insurance lapse registration suspension",
        "last_agent_question": "",
        "carry_forward_context": [],
    }
    hits = [
        {
            "rank": 1,
            "original_rank": 1,
            "chunk_id": "dmv::doc::sec::1::sub::0",
            "doc_id": "doc",
            "section_id": "1",
            "section_title": "Fees",
            "parent_titles": ["Payments"],
            "span_ids": ["1", "2"],
            "token_count": 30,
            "text": "A replacement card costs twenty dollars.",
            "score": 0.20,
            "vector_distance": 0.20,
        },
        {
            "rank": 2,
            "original_rank": 2,
            "chunk_id": "dmv::doc::sec::2::sub::0",
            "doc_id": "doc",
            "section_id": "2",
            "section_title": "Insurance suspension",
            "parent_titles": ["Registration"],
            "span_ids": ["3", "4"],
            "token_count": 30,
            "text": "If insurance lapses, your registration can be suspended until coverage is restored.",
            "score": 0.22,
            "vector_distance": 0.22,
        },
    ]

    reranked = retrieve.rerank_retrieval_hits(hits, query_context=context)

    assert reranked[0]["chunk_id"] == "dmv::doc::sec::2::sub::0"
