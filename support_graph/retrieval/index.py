"""Phase 2 pgvector indexing helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import psycopg
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

from support_graph.config.runtime import IndexConfigLike


def build_collection_name(domain: str) -> str:
    return f"support_graph_{domain}"


def build_vector_id(collection_name: str, chunk_id: str) -> str:
    return f"{collection_name}::{chunk_id}"


def normalize_postgres_connection(connection: str) -> str:
    if connection.startswith("postgresql+"):
        return connection
    if connection.startswith("postgresql://"):
        return connection.replace("postgresql://", "postgresql+psycopg://", 1)
    if connection.startswith("postgres://"):
        return connection.replace("postgres://", "postgresql+psycopg://", 1)
    return connection


def psycopg_connection_string(connection: str) -> str:
    return connection.replace("postgresql+psycopg://", "postgresql://", 1)


def load_chunk_records(path: str | Path) -> list[dict]:
    chunk_path = Path(path)
    if not chunk_path.exists():
        raise FileNotFoundError(f"Chunk artifact not found: {chunk_path}")

    records: list[dict] = []
    for line in chunk_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def validate_index_config(config: IndexConfigLike) -> None:
    missing: list[str] = []
    if not config.postgres_dsn:
        missing.append("postgres_dsn")
    if not config.embedding_model:
        missing.append("embedding_model")

    provider_type = config.provider_type or "ollama"
    if provider_type != "ollama":
        raise ValueError(f"Unsupported provider_type for Phase 2: {provider_type}")

    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required index config: {joined}")


def build_embeddings(
    config: IndexConfigLike, embeddings_cls: type[OllamaEmbeddings] = OllamaEmbeddings
) -> Any:
    validate_index_config(config)

    if config.embedding_client is not None:
        return config.embedding_client

    provider_type = config.provider_type
    base_url = config.ollama_base_url

    # Tests can inject a plain value without full provider metadata.
    if provider_type is None and base_url is None:
        return config.embedding_model

    return embeddings_cls(
        model=config.embedding_model,
        base_url=base_url,
    )


def chunk_record_to_document(
    chunk_record: dict, document_cls: type[Document] | None = None
) -> Document:
    resolved_document_cls = document_cls or Document
    metadata = {
        "chunk_id": chunk_record.get("chunk_id"),
        "domain": chunk_record.get("domain"),
        "doc_id": chunk_record.get("doc_id"),
        "doc_title": chunk_record.get("doc_title"),
        "section_id": chunk_record.get("section_id"),
        "section_title": chunk_record.get("section_title"),
        "parent_titles": chunk_record.get("parent_titles", []),
        "span_ids": chunk_record.get("span_ids", []),
        "subchunk_index": chunk_record.get("subchunk_index"),
        "token_count": chunk_record.get("token_count"),
    }
    return resolved_document_cls(
        page_content=chunk_record.get("text", ""), metadata=metadata
    )


def chunk_records_to_documents(
    chunk_records: list[dict],
    document_cls: type[Document] | None = None,
) -> tuple[list[Document], list[str]]:
    documents = [
        chunk_record_to_document(chunk_record, document_cls=document_cls)
        for chunk_record in chunk_records
    ]
    ids = [str(chunk_record.get("chunk_id", "")) for chunk_record in chunk_records]
    return documents, ids


def load_indexed_chunk_ids(connection: str, collection_name: str) -> set[str]:
    query = """
        select e.cmetadata->>'chunk_id'
        from langchain_pg_embedding e
        join langchain_pg_collection c on e.collection_id = c.uuid
        where c.name = %s
    """
    with psycopg.connect(
        psycopg_connection_string(connection), connect_timeout=5
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (collection_name,))
            return {row[0] for row in cur.fetchall() if row[0]}


def collection_row_count(connection: str, collection_name: str) -> int:
    query = """
        select count(*)
        from langchain_pg_embedding e
        join langchain_pg_collection c on e.collection_id = c.uuid
        where c.name = %s
    """
    with psycopg.connect(
        psycopg_connection_string(connection), connect_timeout=5
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (collection_name,))
            return int(cur.fetchone()[0])


def index_documents(
    config: IndexConfigLike,
    *,
    chunk_records: list[dict] | None = None,
    vectorstore_cls: type[PGVector] | None = None,
    document_cls: type[Document] | None = None,
    embeddings: Any = None,
    create_extension: bool = True,
    pre_delete_collection: bool = False,
    batch_size: int = 1,
) -> Any:
    validate_index_config(config)

    resolved_chunk_records = chunk_records
    if resolved_chunk_records is None:
        chunk_artifact_path = config.chunk_artifact_path
        if not chunk_artifact_path:
            raise ValueError("Missing chunk_artifact_path for indexing.")
        resolved_chunk_records = load_chunk_records(chunk_artifact_path)

    documents, ids = chunk_records_to_documents(
        resolved_chunk_records, document_cls=document_cls
    )
    embedding_client = (
        embeddings if embeddings is not None else build_embeddings(config)
    )
    collection_name = config.collection_name or build_collection_name(config.domain)
    vector_ids = [build_vector_id(collection_name, chunk_id) for chunk_id in ids]
    resolved_vectorstore_cls = vectorstore_cls or PGVector

    kwargs = {
        "connection": normalize_postgres_connection(config.postgres_dsn),
        "collection_name": collection_name,
        "use_jsonb": True,
        "pre_delete_collection": pre_delete_collection,
        "create_extension": create_extension,
    }

    if hasattr(resolved_vectorstore_cls, "add_documents"):
        store = resolved_vectorstore_cls(
            embeddings=embedding_client,
            **kwargs,
        )
        for start in range(0, len(documents), batch_size):
            end = start + batch_size
            store.add_documents(documents[start:end], ids=vector_ids[start:end])
        stored_ids = load_indexed_chunk_ids(kwargs["connection"], collection_name)
        missing_ids = [chunk_id for chunk_id in ids if chunk_id not in stored_ids]
        if missing_ids:
            by_chunk_id = {
                chunk_record["chunk_id"]: chunk_record
                for chunk_record in resolved_chunk_records
            }
            missing_records = [by_chunk_id[chunk_id] for chunk_id in missing_ids]
            missing_documents, missing_document_ids = chunk_records_to_documents(
                missing_records,
                document_cls=document_cls,
            )
            missing_vector_ids = [
                build_vector_id(collection_name, chunk_id)
                for chunk_id in missing_document_ids
            ]
            for start in range(0, len(missing_documents), batch_size):
                end = start + batch_size
                store.add_documents(
                    missing_documents[start:end],
                    ids=missing_vector_ids[start:end],
                )
        return store

    kwargs["ids"] = vector_ids
    return resolved_vectorstore_cls.from_documents(
        documents,
        embedding_client,
        **kwargs,
    )
