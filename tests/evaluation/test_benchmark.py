from support_graph.evaluation.benchmark import (
    benchmark_embeddings,
    select_benchmark_records,
)


def test_select_benchmark_records_is_deterministic_and_sized() -> None:
    records = [
        {"chunk_id": f"chunk-{index}", "text": f"text {index}"} for index in range(20)
    ]

    first = select_benchmark_records(records, 5)
    second = select_benchmark_records(records, 5)

    assert first == second
    assert len(first) == 5
    assert first[0]["chunk_id"] == "chunk-0"


def test_benchmark_embeddings_reports_throughput_and_estimate() -> None:
    class FakeEmbeddings:
        def __init__(self) -> None:
            self.calls = []

        def embed_documents(self, texts):
            self.calls.append(list(texts))
            return [[0.1, 0.2] for _ in texts]

    records = [
        {"chunk_id": f"chunk-{index}", "text": f"text {index}"} for index in range(10)
    ]
    embeddings = FakeEmbeddings()

    result = benchmark_embeddings(
        embedding_model="fake-model",
        chunk_records=records,
        embeddings=embeddings,
        sample_size=4,
        batch_size=2,
        warmup=True,
    )

    assert result["model"] == "fake-model"
    assert result["sample_size"] == 4
    assert result["total_chunks"] == 10
    assert len(embeddings.calls) == 3
    assert len(embeddings.calls[0]) == 1
    assert len(embeddings.calls[1]) == 2
    assert len(embeddings.calls[2]) == 2
    assert result["chunks_per_second"] > 0
    assert result["estimated_total_seconds"] > 0
