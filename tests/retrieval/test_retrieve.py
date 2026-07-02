from __future__ import annotations

from types import SimpleNamespace

from tests._fixtures import _hit, _query_context
from support_graph.retrieval import retrieve


def test_build_query_context_includes_latest_user_need_agent_question_and_two_carry_forward_turns() -> (
    None
):
    example = {
        "domain": "kubernetes",
        "conversation": [
            {
                "turn_id": 1,
                "role": "user",
                "utterance": "I renamed my app last week and need to update my workloads.",
            },
            {
                "turn_id": 2,
                "role": "agent",
                "utterance": "Did your Service selector change too?",
            },
            {
                "turn_id": 3,
                "role": "user",
                "utterance": "Yes, I need the Service steps too and I also changed labels.",
            },
            {
                "turn_id": 4,
                "role": "agent",
                "utterance": "Do you need the Deployment label selector updated as well?",
            },
            {
                "turn_id": 5,
                "role": "user",
                "utterance": "I need to update my Deployment and Service after relabeling.",
            },
        ],
        "latest_user_turn_id": 5,
        "latest_user_utterance": "I need to update my Deployment and Service after relabeling.",
    }

    context = retrieve.build_query_context(example)

    assert context["domain"] == "kubernetes"
    assert (
        context["latest_user_need"]
        == "I need to update my Deployment and Service after relabeling."
    )
    assert (
        context["last_agent_question"]
        == "Do you need the Deployment label selector updated as well?"
    )
    assert len(context["carry_forward_context"]) == 2
    assert [turn["role"] for turn in context["carry_forward_context"]] == [
        "user",
        "user",
    ]


def test_build_query_context_excludes_short_and_duplicate_turns() -> None:
    example = {
        "domain": "kubernetes",
        "conversation": [
            {
                "turn_id": 1,
                "role": "user",
                "utterance": "Need rollout restart steps for a Deployment.",
            },
            {"turn_id": 2, "role": "agent", "utterance": "Okay."},
            {
                "turn_id": 3,
                "role": "user",
                "utterance": "rollout restart steps for a Deployment",
            },
            {
                "turn_id": 4,
                "role": "agent",
                "utterance": "Did the rollout already start?",
            },
            {
                "turn_id": 5,
                "role": "user",
                "utterance": "What rollout restart steps do I need for a Deployment?",
            },
        ],
        "latest_user_turn_id": 5,
        "latest_user_utterance": "What rollout restart steps do I need for a Deployment?",
    }

    context = retrieve.build_query_context(example)

    assert context["last_agent_question"] == "Did the rollout already start?"
    assert context["carry_forward_context"] == []


def test_build_query_renders_compact_labeled_blocks() -> None:
    example = {
        "domain": "kubernetes",
        "conversation": [
            {
                "turn_id": 1,
                "role": "user",
                "utterance": "My rollout stalled last night and I got an alert.",
            },
            {
                "turn_id": 2,
                "role": "agent",
                "utterance": "Did you already inspect the Deployment status?",
            },
            {
                "turn_id": 3,
                "role": "user",
                "utterance": "What do I need to do to diagnose it now?",
            },
        ],
        "latest_user_turn_id": 3,
        "latest_user_utterance": "What do I need to do to diagnose it now?",
    }

    query = retrieve.build_query(example)

    assert "Domain: kubernetes" in query
    assert "Latest user need: What do I need to do to diagnose it now?" in query
    assert (
        "Last agent question: Did you already inspect the Deployment status?" in query
    )
    assert "Recent conversation:" not in query


def test_build_metadata_filter_limits_domain_and_doc_ids() -> None:
    flt = retrieve.build_metadata_filter(
        domain="kubernetes", doc_ids=["doc-a", "doc-b"]
    )

    assert flt is not None
    assert flt["domain"] == "kubernetes"
    assert flt["doc_id"] == {"$in": ["doc-a", "doc-b"]}


def test_normalize_retrieval_hits_preserves_rank_and_scores() -> None:
    fake_hits = [
        SimpleNamespace(
            page_content="alpha beta",
            metadata={
                "chunk_id": "kubernetes::doc::sec::1::sub::0",
                "doc_id": "doc",
                "span_ids": ["1"],
            },
            score=0.12,
        ),
        {
            "page_content": "gamma delta",
            "metadata": {
                "chunk_id": "kubernetes::doc::sec::2::sub::0",
                "doc_id": "doc",
                "span_ids": ["2"],
            },
            "score": 0.34,
        },
    ]

    normalized = retrieve.normalize_retrieval_hits(fake_hits)

    assert normalized[0]["chunk_id"] == "kubernetes::doc::sec::1::sub::0"
    assert normalized[0]["score"] == 0.12
    assert normalized[0]["original_rank"] == 1
    assert normalized[1]["chunk_id"] == "kubernetes::doc::sec::2::sub::0"
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
                            "chunk_id": f"kubernetes::doc::sec::{index}::sub::0",
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
        "domain": "kubernetes",
        "conversation": [
            {
                "turn_id": 1,
                "role": "user",
                "utterance": "I changed labels and need Service update steps.",
            }
        ],
        "latest_user_turn_id": 1,
        "latest_user_utterance": "I changed labels and need Service update steps.",
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
    assert results[0]["chunk_id"] == "kubernetes::doc::sec::1::sub::0"


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
                            "chunk_id": "kubernetes::doc-a::sec::1::sub::0",
                            "domain": "kubernetes",
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
        k = 5

        def invoke(self, query, **kwargs):
            captured["keyword_query"] = query
            return [
                SimpleNamespace(
                    page_content="duplicate hit",
                    metadata={
                        "chunk_id": "kubernetes::doc-a::sec::1::sub::0",
                        "domain": "kubernetes",
                        "doc_id": "doc-a",
                    },
                ),
                SimpleNamespace(
                    page_content="keyword hit",
                    metadata={
                        "chunk_id": "kubernetes::doc-a::sec::2::sub::0",
                        "domain": "kubernetes",
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
                        "chunk_id": "kubernetes::doc-b::sec::3::sub::0",
                        "domain": "kubernetes",
                        "doc_id": "doc-b",
                    },
                ),
            ]

    example = {
        "domain": "kubernetes",
        "conversation": [
            {
                "turn_id": 1,
                "role": "user",
                "utterance": "I changed labels and need Service update steps.",
            }
        ],
        "latest_user_turn_id": 1,
        "latest_user_utterance": "I changed labels and need Service update steps.",
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
    assert captured["filter"] == {"domain": "kubernetes", "doc_id": "doc-a"}
    assert "Domain:" not in captured["query"]
    assert captured["keyword_query"] == captured["query"]
    assert [hit["chunk_id"] for hit in results] == [
        "kubernetes::doc-a::sec::1::sub::0",
        "kubernetes::doc-a::sec::2::sub::0",
    ]


def test_retrieve_chunks_scores_keyword_hits_on_dense_scale_before_rerank() -> None:
    class FakeVectorStore:
        def similarity_search_with_score(self, query, k=5, filter=None):
            return [
                (
                    SimpleNamespace(
                        page_content="generic deployment tutorial",
                        metadata={
                            "chunk_id": "kubernetes::tutorial::sec::1::sub::0",
                            "domain": "kubernetes",
                            "doc_id": "tutorials/hello-minikube",
                            "section_id": "1",
                            "section_title": "Create a Deployment",
                            "span_ids": [
                                "tutorials/hello-minikube#create-a-deployment"
                            ],
                            "token_count": 40,
                        },
                    ),
                    0.36,
                )
            ]

    class FakeKeywordRetriever:
        k = 5

        def invoke(self, query, **kwargs):
            return [
                SimpleNamespace(
                    page_content="Complete Deployment manage updated replicas.",
                    metadata={
                        "chunk_id": (
                            "kubernetes::concepts/workloads/controllers/deployment"
                            "::sec::complete-deployment::sub::0"
                        ),
                        "domain": "kubernetes",
                        "doc_id": "concepts/workloads/controllers/deployment",
                        "section_id": "complete-deployment",
                        "section_title": "Complete Deployment",
                        "span_ids": [
                            "concepts/workloads/controllers/deployment"
                            "#complete-deployment"
                        ],
                        "token_count": 32,
                    },
                )
            ]

    example = {
        "domain": "kubernetes",
        "conversation": [
            {
                "turn_id": 1,
                "role": "user",
                "utterance": "What does a Kubernetes Deployment manage?",
            }
        ],
        "latest_user_turn_id": 1,
        "latest_user_utterance": "What does a Kubernetes Deployment manage?",
    }

    results = retrieve.retrieve_chunks(
        example=example,
        vectorstore=FakeVectorStore(),
        keyword_retriever=FakeKeywordRetriever(),
        top_k=1,
        candidate_k=1,
        rerank=True,
    )

    assert results[0]["doc_id"] == "concepts/workloads/controllers/deployment"
    assert results[0]["retrieval_source"] == "keyword"
    assert results[0]["score"] == 0.362


def test_rerank_retrieval_hits_penalizes_title_and_short_single_span_chunks() -> None:
    context = _query_context(
        latest_user_need="deployment rollout stalled unavailable replicas",
    )
    hits = [
        _hit(
            "kubernetes::doc::sec::t_1::sub::0",
            rank=1,
            doc_id="doc",
            section_id="t_1",
            section_title="What happens if my rollout stalls?",
            span_ids=["10"],
            token_count=8,
            text="What happens if my rollout stalls?",
            score=0.20,
            vector_distance=0.20,
        ),
        _hit(
            "kubernetes::doc::sec::2::sub::0",
            rank=2,
            doc_id="doc",
            section_id="2",
            section_title="Deployment rollout guidance",
            span_ids=["11", "12"],
            token_count=32,
            text="If a rollout stalls, inspect Deployment conditions and unavailable replicas.",
            score=0.21,
            vector_distance=0.21,
        ),
    ]

    reranked = retrieve.rerank_retrieval_hits(hits, query_context=context)

    assert reranked[0]["chunk_id"] == "kubernetes::doc::sec::2::sub::0"


def test_query_context_tokens_uses_library_stopwords() -> None:
    context = _query_context(
        latest_user_need="What do I need to do for a rollout restart?",
        last_agent_question="How do I complete this?",
    )

    tokens = retrieve.query_context_tokens(context)

    assert "what" not in tokens
    assert "how" not in tokens
    assert "for" not in tokens
    assert "rollout" in tokens
    assert "restart" in tokens


def test_rerank_retrieval_hits_uses_text_and_title_overlap_to_lift_relevant_chunk() -> (
    None
):
    context = _query_context(
        latest_user_need="deployment rollout stalled unavailable replicas",
    )
    hits = [
        _hit(
            "kubernetes::doc::sec::1::sub::0",
            rank=1,
            doc_id="doc",
            section_id="1",
            section_title="Images",
            parent_titles=["Containers"],
            span_ids=["1", "2"],
            token_count=30,
            text="A container image should use an immutable tag.",
            score=0.20,
            vector_distance=0.20,
        ),
        _hit(
            "kubernetes::doc::sec::2::sub::0",
            rank=2,
            doc_id="doc",
            section_id="2",
            section_title="Deployment rollout",
            parent_titles=["Workloads"],
            span_ids=["3", "4"],
            token_count=30,
            text="If a Deployment rollout stalls, check unavailable replicas and rollout status.",
            score=0.22,
            vector_distance=0.22,
        ),
    ]

    reranked = retrieve.rerank_retrieval_hits(hits, query_context=context)

    assert reranked[0]["chunk_id"] == "kubernetes::doc::sec::2::sub::0"


def test_rerank_retrieval_hits_uses_doc_path_overlap_to_lift_canonical_page() -> None:
    context = _query_context(
        latest_user_need="What does a Kubernetes Deployment manage?",
    )
    hits = [
        _hit(
            "kubernetes::tutorials/hello-minikube::sec::create::sub::0",
            rank=1,
            doc_id="tutorials/hello-minikube",
            section_id="create",
            section_title="Create a Deployment",
            span_ids=["tutorials/hello-minikube#create-a-deployment"],
            token_count=40,
            text="A Kubernetes Deployment checks the health of your Pod.",
            score=0.37,
            vector_distance=0.37,
        ),
        _hit(
            "kubernetes::concepts/workloads/controllers/deployment"
            "::sec::complete-deployment::sub::0",
            rank=2,
            doc_id="concepts/workloads/controllers/deployment",
            section_id="complete-deployment",
            section_title="Complete Deployment",
            span_ids=["concepts/workloads/controllers/deployment#complete-deployment"],
            token_count=32,
            text="A complete Deployment has updated replicas available.",
            score=0.372,
            vector_distance=0.372,
        ),
    ]

    reranked = retrieve.rerank_retrieval_hits(hits, query_context=context)

    assert reranked[0]["doc_id"] == "concepts/workloads/controllers/deployment"
    assert reranked[0]["path_overlap_count"] == 1
