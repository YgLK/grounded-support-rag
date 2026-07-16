from __future__ import annotations

from types import SimpleNamespace

from tests._fixtures import _example, _hit
from dev_tools.retrieval_ablation import retrieval_ablation
from support_graph.config.runtime import RuntimeConfig


def test_ablation_metric_aggregation_handles_rag_and_missing_fields() -> None:
    example = _example(
        "ex-1",
        gold_doc_ids=["doc-a"],
        gold_span_ids=["span-a"],
        expected_sources=["doc-a"],
        acceptable_sources=["doc-b"],
        acceptable_span_ids=["span-b"],
    )
    chunks = [_hit("chunk-a", doc_id="doc-a", span_ids=["span-a"])]

    metrics = retrieval_ablation.retrieval_metrics(example, chunks, top_k=5)
    aggregate = retrieval_ablation.aggregate_metrics(
        [{"mode": "dense_only", "metrics": metrics, "retrieval_source_counts": {}}]
    )

    assert metrics["doc_recall_at_1"] == 1.0
    assert metrics["span_recall_at_5"] == 1.0
    assert metrics["hit_at_5"] == 1.0
    assert aggregate["dense_only"]["examples"] == 1
    assert aggregate["dense_only"]["ndcg_at_5"] == 1.0


def test_comparison_rows_classify_component_wins() -> None:
    records = [
        {"example_id": "ex", "mode": "dense_only", "metrics": {"ndcg_at_5": 0.0}},
        {"example_id": "ex", "mode": "keyword_only", "metrics": {"ndcg_at_5": 0.5}},
        {
            "example_id": "ex",
            "mode": "hybrid_no_rerank",
            "metrics": {"ndcg_at_5": 0.6},
        },
        {
            "example_id": "ex",
            "mode": "hybrid_rerank",
            "metrics": {"ndcg_at_5": 1.0},
        },
    ]

    rows = retrieval_ablation.comparison_rows(records)

    assert rows[0]["keyword_beats_dense"] is True
    assert rows[0]["hybrid_rescue"] is True
    assert rows[0]["rerank_rescue"] is True
    assert rows[0]["all_modes_miss"] is False


def test_run_ablation_uses_all_modes_without_live_services() -> None:
    class FakeVectorStore:
        def similarity_search_with_score(self, query, k=5, filter=None):
            return [
                (
                    SimpleNamespace(
                        page_content="dense pod text",
                        metadata={
                            "chunk_id": "dense",
                            "domain": "kubernetes",
                            "doc_id": "doc-a",
                            "doc_title": "Doc A",
                            "section_id": "1",
                            "section_title": "Pods",
                            "parent_titles": [],
                            "span_ids": ["span-a"],
                            "token_count": 20,
                        },
                    ),
                    0.1,
                )
            ]

    class FakeKeywordRetriever:
        def invoke(self, query):
            return [
                SimpleNamespace(
                    page_content="keyword service text",
                    metadata={
                        "chunk_id": "keyword",
                        "domain": "kubernetes",
                        "doc_id": "doc-b",
                        "doc_title": "Doc B",
                        "section_id": "2",
                        "section_title": "Services",
                        "parent_titles": [],
                        "span_ids": ["span-b"],
                        "token_count": 20,
                    },
                )
            ]

    records = retrieval_ablation.run_ablation(
        [_example("ex-1", gold_doc_ids=["doc-a"], gold_span_ids=["span-a"])],
        config=RuntimeConfig(),
        vectorstore=FakeVectorStore(),
        keyword_retriever=FakeKeywordRetriever(),
        top_k=5,
        candidate_k=5,
    )

    assert [record["mode"] for record in records] == list(retrieval_ablation.MODE_ORDER)
    assert records[0]["retrieval_ranked_chunk_ids"] == ["dense"]
    assert records[1]["retrieval_ranked_chunk_ids"] == ["keyword"]
