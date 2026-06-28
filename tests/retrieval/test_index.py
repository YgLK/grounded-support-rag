from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from support_graph.retrieval import index


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHUNK_ARTIFACT = REPO_ROOT / "data" / "derived" / "chunks" / "dmv.jsonl"


def _resolve_callable(*names: str):
    for name in names:
        candidate = getattr(index, name, None)
        if callable(candidate):
            return candidate
    pytest.xfail(
        f"Phase 2 index API not implemented yet; expected one of: {', '.join(names)}"
    )


def _xfail_if_placeholder(func) -> None:
    try:
        source = inspect.getsource(func)
    except OSError:
        return
    if "NotImplementedError" in source:
        pytest.xfail("Phase 2 index implementation is still a placeholder.")


def test_load_chunk_records_reads_jsonl_round_trip(tmp_path: Path) -> None:
    loader = _resolve_callable(
        "load_chunk_records", "load_chunk_artifact", "load_chunks_jsonl"
    )
    _xfail_if_placeholder(loader)

    records = [
        {
            "chunk_id": "dmv::doc::sec::1::sub::0",
            "domain": "dmv",
            "doc_id": "doc",
            "doc_title": "Doc",
            "section_id": "1",
            "section_title": "Section",
            "parent_titles": ["Parent"],
            "subchunk_index": 0,
            "text": "alpha beta",
            "span_ids": ["1"],
            "token_count": 2,
        }
    ]
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )

    loaded = loader(path)

    assert loaded == records


def test_validate_index_config_requires_postgres_and_models() -> None:
    validator = _resolve_callable(
        "validate_index_config", "validate_config", "validate_settings"
    )
    _xfail_if_placeholder(validator)

    invalid_config = SimpleNamespace(
        postgres_dsn=None,
        chat_provider_type="ollama",
        embedding_provider_type="ollama",
        ollama_base_url=None,
        openrouter_base_url=None,
        openrouter_api_key=None,
        chat_model=None,
        embedding_model=None,
        embedding_client=None,
        domain="dmv",
        collection_name="support_graph_dmv",
        chunk_artifact_path=None,
    )

    with pytest.raises((ValueError, RuntimeError)) as excinfo:
        validator(invalid_config)

    message = str(excinfo.value)
    assert "postgres" in message.lower() or "dsn" in message.lower()
    assert "chat" in message.lower() or "embedding" in message.lower()


def test_index_documents_converts_chunks_to_documents_and_forwards_to_vectorstore(
    monkeypatch,
) -> None:
    index_documents = _resolve_callable("index_documents")
    _xfail_if_placeholder(index_documents)

    captured = {}

    class FakeDocument:
        def __init__(self, page_content: str, metadata: dict):
            self.page_content = page_content
            self.metadata = metadata

    class FakeVectorStore:
        @classmethod
        def from_documents(cls, documents, embedding, *, ids=None, **kwargs):
            captured["documents"] = documents
            captured["embedding"] = embedding
            captured["ids"] = ids
            captured["kwargs"] = kwargs
            return {"documents": documents, "embedding": embedding}

    def fake_load_chunk_records(path):
        assert Path(path) == CHUNK_ARTIFACT
        return [
            {
                "chunk_id": "dmv::doc::sec::1::sub::0",
                "domain": "dmv",
                "doc_id": "doc",
                "doc_title": "Doc",
                "section_id": "1",
                "section_title": "Section",
                "parent_titles": ["Parent"],
                "subchunk_index": 0,
                "text": "alpha beta",
                "span_ids": ["1"],
                "token_count": 2,
            }
        ]

    monkeypatch.setattr(index, "Document", FakeDocument, raising=False)
    monkeypatch.setattr(index, "PGVector", FakeVectorStore, raising=False)
    monkeypatch.setattr(
        index, "load_chunk_records", fake_load_chunk_records, raising=False
    )

    config = SimpleNamespace(
        postgres_dsn="postgresql://localhost:5432/support_graph",
        chat_provider_type="ollama",
        embedding_provider_type="ollama",
        ollama_base_url="http://localhost:11434",
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        chat_model="ignored-for-indexing",
        embedding_model="fake-embedding-model",
        embedding_client=None,
        collection_name="support_graph_dmv",
        domain="dmv",
        chunk_artifact_path=CHUNK_ARTIFACT,
    )

    try:
        result = index_documents(config, embeddings="fake-embedding-model")
    except NotImplementedError:
        pytest.xfail("Phase 2 index_documents is still a placeholder.")

    assert captured["documents"][0].page_content == "alpha beta"
    assert captured["documents"][0].metadata["chunk_id"] == "dmv::doc::sec::1::sub::0"
    assert captured["documents"][0].metadata["doc_id"] == "doc"
    assert captured["embedding"] == "fake-embedding-model"
    assert result == {
        "documents": captured["documents"],
        "embedding": "fake-embedding-model",
    }
