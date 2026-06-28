from support_graph.data.chunks import build_chunks


def test_build_chunks_is_deterministic_and_unique() -> None:
    documents = [
        {
            "domain": "kubernetes",
            "doc_id": "concepts/workloads/pods",
            "title": "Pods",
            "doc_text": "",
            "spans": [
                {
                    "id_sp": "concepts/workloads/pods#overview",
                    "tag": "section",
                    "start_sp": 0,
                    "end_sp": 10,
                    "text_sp": "Pod overview.",
                    "title": "Overview",
                    "parent_titles": [],
                    "id_sec": "overview",
                    "start_sec": 0,
                    "end_sec": 10,
                    "text_sec": "Pod overview.",
                }
            ],
        }
    ]
    first = build_chunks(documents, domains=["kubernetes"])
    second = build_chunks(documents, domains=["kubernetes"])

    assert first == second
    assert len(first) == len({chunk["chunk_id"] for chunk in first})
    assert all(chunk["domain"] == "kubernetes" for chunk in first)


def test_build_chunks_preserves_section_metadata() -> None:
    documents = [
        {
            "domain": "kubernetes",
            "doc_id": "tasks/debug/debug-application/debug-service",
            "title": "Debug Services",
            "doc_text": "",
            "spans": [
                {
                    "id_sp": "tasks/debug/debug-application/debug-service#dns",
                    "tag": "section",
                    "start_sp": 0,
                    "end_sp": 12,
                    "text_sp": "Check DNS.",
                    "title": "Does the Service work by DNS name?",
                    "parent_titles": ["Debug Services"],
                    "id_sec": "dns",
                    "start_sec": 0,
                    "end_sec": 12,
                    "text_sec": "Check DNS.",
                }
            ],
        }
    ]
    chunks = build_chunks(documents, domains=["kubernetes"])
    chunk = chunks[0]

    assert chunk["section_title"] == "Does the Service work by DNS name?"
    assert chunk["parent_titles"] == ["Debug Services"]
    assert chunk["span_ids"] == ["tasks/debug/debug-application/debug-service#dns"]
    assert chunk["token_count"] > 0


def test_build_chunks_splits_oversized_sections_stably() -> None:
    document = {
        "domain": "kubernetes",
        "doc_id": "doc-1",
        "title": "Doc",
        "doc_text": "",
        "spans": [
            {
                "id_sp": "1",
                "tag": "u",
                "start_sp": 0,
                "end_sp": 10,
                "text_sp": "one two three four ",
                "title": "Section",
                "parent_titles": ["Parent"],
                "id_sec": "10",
                "start_sec": 0,
                "end_sec": 20,
                "text_sec": "one two three four ",
            },
            {
                "id_sp": "2",
                "tag": "u",
                "start_sp": 10,
                "end_sp": 20,
                "text_sp": "five six seven eight ",
                "title": "Section",
                "parent_titles": ["Parent"],
                "id_sec": "10",
                "start_sec": 0,
                "end_sec": 20,
                "text_sec": "five six seven eight ",
            },
        ],
    }

    chunks = build_chunks([document], max_tokens_per_chunk=4)

    assert [chunk["chunk_id"] for chunk in chunks] == [
        "kubernetes::doc-1::sec::10::sub::0",
        "kubernetes::doc-1::sec::10::sub::1",
    ]
    assert chunks[0]["span_ids"] == ["1"]
    assert chunks[1]["span_ids"] == ["2"]
