from tests._fixtures import _document, _span
from support_graph.data.chunks import build_chunks


def test_build_chunks_is_deterministic_and_unique() -> None:
    documents = [
        _document(
            "concepts/workloads/pods",
            title="Pods",
            spans=[
                _span(
                    "concepts/workloads/pods#overview",
                    title="Overview",
                    id_sec="overview",
                    text_sp="Pod overview.",
                    text_sec="Pod overview.",
                )
            ],
        )
    ]
    first = build_chunks(documents, domains=["kubernetes"])
    second = build_chunks(documents, domains=["kubernetes"])

    assert first == second
    assert len(first) == len({chunk["chunk_id"] for chunk in first})
    assert all(chunk["domain"] == "kubernetes" for chunk in first)


def test_build_chunks_preserves_section_metadata() -> None:
    documents = [
        _document(
            "tasks/debug/debug-application/debug-service",
            title="Debug Services",
            spans=[
                _span(
                    "tasks/debug/debug-application/debug-service#dns",
                    title="Does the Service work by DNS name?",
                    parent_titles=["Debug Services"],
                    id_sec="dns",
                    text_sp="Check DNS.",
                    text_sec="Check DNS.",
                )
            ],
        )
    ]
    chunks = build_chunks(documents, domains=["kubernetes"])
    chunk = chunks[0]

    assert chunk["section_title"] == "Does the Service work by DNS name?"
    assert chunk["parent_titles"] == ["Debug Services"]
    assert chunk["span_ids"] == ["tasks/debug/debug-application/debug-service#dns"]
    assert chunk["token_count"] > 0


def test_build_chunks_splits_oversized_sections_stably() -> None:
    document = _document(
        "doc-1",
        title="Doc",
        spans=[
            _span(
                "1",
                tag="u",
                id_sec="10",
                text_sp="one two three four ",
                text_sec="one two three four ",
            ),
            _span(
                "2",
                tag="u",
                id_sec="10",
                text_sp="five six seven eight ",
                text_sec="five six seven eight ",
            ),
        ],
    )

    chunks = build_chunks([document], max_tokens_per_chunk=4)

    assert [chunk["chunk_id"] for chunk in chunks] == [
        "kubernetes::doc-1::sec::10::sub::0",
        "kubernetes::doc-1::sec::10::sub::1",
    ]
    assert chunks[0]["span_ids"] == ["1"]
    assert chunks[1]["span_ids"] == ["2"]
