from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples._shared import (
    SAMPLE_CONTENT_CHUNK_ID,
    SAMPLE_EXAMPLE_ID,
    SAMPLE_TITLE_CHUNK_ID,
    find_by_key,
    json_lines,
    load_or_build_chunks,
    load_or_build_examples,
    load_settings,
    next_step_lines,
    print_lines,
)
from support_graph.retrieval.retrieve import (
    build_query,
    build_query_context,
    rerank_retrieval_hits,
)


def build_lines() -> list[str]:
    settings = load_settings()
    examples = load_or_build_examples(settings)
    example = find_by_key(examples, "example_id", SAMPLE_EXAMPLE_ID)
    chunks = load_or_build_chunks(settings)
    title_chunk = find_by_key(chunks, "chunk_id", SAMPLE_TITLE_CHUNK_ID)
    content_chunk = find_by_key(chunks, "chunk_id", SAMPLE_CONTENT_CHUNK_ID)
    query_context = build_query_context(example)
    query = build_query(example)

    reranked = rerank_retrieval_hits(
        [
            {
                **title_chunk,
                "rank": 1,
                "original_rank": 1,
                "vector_distance": 0.12,
                "score": 0.12,
            },
            {
                **content_chunk,
                "rank": 2,
                "original_rank": 2,
                "vector_distance": 0.15,
                "score": 0.15,
            },
        ],
        query_context=query_context,
    )

    lines = [
        "SupportGraph Example 06",
        "Query construction and deterministic reranking",
        "",
        "Query context",
    ]
    lines.extend(json_lines(query_context))
    lines.extend(
        [
            "",
            "Rendered query",
            query,
            "",
            "Why the content chunk wins",
            "- the title chunk is penalized for section_id starting with t_",
            "- the content chunk gets text-overlap lift on insurance and lapses",
            "- reranking keeps the best 5 direct chunks before neighbor expansion",
            "",
            "Reranked candidates",
        ]
    )
    for chunk in reranked:
        lines.append(
            "- rank {rank} | original {original_rank} | chunk_id {chunk_id} | rerank_score {score} | text_overlap {text_overlap} | title_overlap {title_overlap}".format(
                rank=chunk["rank"],
                original_rank=chunk["original_rank"],
                chunk_id=chunk["chunk_id"],
                score=chunk["rerank_score"],
                text_overlap=chunk["text_overlap_count"],
                title_overlap=chunk["title_overlap_count"],
            )
        )
    lines.extend(next_step_lines("07_index_and_vectorstore_walkthrough.py"))
    return lines


def main() -> None:
    print_lines(build_lines())


if __name__ == "__main__":
    main()
