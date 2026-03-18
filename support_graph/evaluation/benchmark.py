"""Local embedding benchmark helpers."""

from __future__ import annotations

import math
import time
from typing import Any

from support_graph.retrieval.index import build_embeddings, load_chunk_records


def select_benchmark_records(chunk_records: list[dict], sample_size: int) -> list[dict]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if sample_size >= len(chunk_records):
        return list(chunk_records)

    step = len(chunk_records) / sample_size
    indices = []
    seen = set()
    for index in range(sample_size):
        candidate = min(len(chunk_records) - 1, math.floor(index * step))
        if candidate not in seen:
            indices.append(candidate)
            seen.add(candidate)

    while len(indices) < sample_size:
        candidate = len(indices)
        if candidate not in seen:
            indices.append(candidate)
            seen.add(candidate)

    return [chunk_records[index] for index in indices[:sample_size]]


def chunk_records_to_texts(chunk_records: list[dict]) -> list[str]:
    return [str(chunk_record.get("text", "")) for chunk_record in chunk_records]


def benchmark_embeddings(
    config: Any,
    *,
    chunk_records: list[dict],
    sample_size: int = 100,
    batch_size: int = 1,
    warmup: bool = True,
    embeddings: Any = None,
) -> dict:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    sample_records = select_benchmark_records(chunk_records, sample_size)
    sample_texts = chunk_records_to_texts(sample_records)
    embedding_client = embeddings if embeddings is not None else build_embeddings(config)

    warmup_seconds = 0.0
    if warmup and sample_texts:
        start = time.perf_counter()
        embedding_client.embed_documents(sample_texts[:1])
        warmup_seconds = time.perf_counter() - start

    start = time.perf_counter()
    for offset in range(0, len(sample_texts), batch_size):
        batch = sample_texts[offset : offset + batch_size]
        embedding_client.embed_documents(batch)
    elapsed_seconds = time.perf_counter() - start

    measured_chunks = len(sample_texts)
    chunks_per_second = measured_chunks / elapsed_seconds if elapsed_seconds else 0.0
    total_chunks = len(chunk_records)
    estimated_total_seconds = (
        total_chunks / chunks_per_second if chunks_per_second else float("inf")
    )

    return {
        "model": getattr(config, "embedding_model", None),
        "sample_size": measured_chunks,
        "total_chunks": total_chunks,
        "batch_size": batch_size,
        "warmup_seconds": warmup_seconds,
        "elapsed_seconds": elapsed_seconds,
        "chunks_per_second": chunks_per_second,
        "estimated_total_seconds": estimated_total_seconds,
    }


def load_benchmark_chunk_records(path: str) -> list[dict]:
    return load_chunk_records(path)
