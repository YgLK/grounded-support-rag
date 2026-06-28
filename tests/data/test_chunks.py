from pathlib import Path

from support_graph.data.chunks import build_chunks
from support_graph.data.dataset import load_documents


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT = REPO_ROOT / "multidoc2dial"


def test_build_chunks_for_dmv_is_deterministic_and_unique() -> None:
    documents = load_documents(DATASET_ROOT, domains=["dmv"])
    first = build_chunks(documents, domains=["dmv"])
    second = build_chunks(documents, domains=["dmv"])

    assert first == second
    assert len(first) == len({chunk["chunk_id"] for chunk in first})
    assert all(chunk["domain"] == "dmv" for chunk in first)


def test_build_chunks_preserves_section_metadata() -> None:
    documents = load_documents(DATASET_ROOT, domains=["dmv"])
    chunks = build_chunks(documents, domains=["dmv"])
    renew_chunk = next(
        chunk
        for chunk in chunks
        if chunk["chunk_id"] == "dmv::Registrations#3_0::sec::4::sub::0"
    )

    assert renew_chunk["section_title"] == "Renew"
    assert renew_chunk["parent_titles"] == ["Vehicles already registered in New York"]
    assert renew_chunk["span_ids"] == ["8", "9", "10", "11", "12"]
    assert renew_chunk["token_count"] > 0


def test_build_chunks_splits_oversized_sections_stably() -> None:
    document = {
        "domain": "dmv",
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
        "dmv::doc-1::sec::10::sub::0",
        "dmv::doc-1::sec::10::sub::1",
    ]
    assert chunks[0]["span_ids"] == ["1"]
    assert chunks[1]["span_ids"] == ["2"]
