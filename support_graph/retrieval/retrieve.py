"""Phase 2 retrieval helpers."""

from __future__ import annotations

import re
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_postgres import PGVector
from stop_words import get_stop_words

from support_graph.config.runtime import RuntimeConfig
from support_graph.retrieval.index import (
    build_collection_name,
    build_embeddings,
    normalize_postgres_connection,
    validate_index_config,
)
from support_graph.types import (
    Example,
    NormalizedRetrievalHit,
    QueryContext,
    QueryContextTurn,
    RetrieverLike,
    VectorStoreLike,
)

__all__ = [
    "build_query_context",
    "build_query",
    "build_legacy_query",
    "build_metadata_filter",
    "normalize_retrieval_hits",
    "rerank_retrieval_hits",
    "retrieve_chunks",
    "get_vectorstore",
    "build_keyword_retriever",
]

QUERY_TOKEN_STOPWORDS = frozenset(get_stop_words("en"))


def _conversation_from_example(example: dict[str, Any]) -> list[dict[str, Any]]:
    if example.get("conversation"):
        return list(example.get("conversation", []))
    if example.get("turns_before_target"):
        return list(example.get("turns_before_target", []))
    return []


def _latest_user_utterance(
    example: dict[str, Any], conversation: list[dict[str, Any]]
) -> str:
    latest = example.get("latest_user_utterance")
    if latest:
        return str(latest)
    for turn in reversed(conversation):
        if turn.get("role") == "user" and turn.get("utterance"):
            return str(turn.get("utterance"))
    return ""


def _domain_from_example(example: dict[str, Any]) -> str:
    return str(example.get("domain") or "").strip()


def _normalize_tokens(
    text: str, *, minimum_length: int = 1, drop_stopwords: bool = False
) -> list[str]:
    normalized: list[str] = []
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if len(token) < minimum_length:
            continue
        if drop_stopwords and token in QUERY_TOKEN_STOPWORDS:
            continue
        normalized.append(token)
    return normalized


def _normalized_text(text: str) -> str:
    return " ".join(_normalize_tokens(text))


def _find_latest_user_index(example: dict, conversation: list[dict]) -> int | None:
    latest_turn_id = example.get("latest_user_turn_id")
    if latest_turn_id is not None:
        for index in range(len(conversation) - 1, -1, -1):
            turn = conversation[index]
            if turn.get("role") == "user" and turn.get("turn_id") == latest_turn_id:
                return index

    latest_user = str(example.get("latest_user_utterance") or "").strip()
    if latest_user:
        for index in range(len(conversation) - 1, -1, -1):
            turn = conversation[index]
            if (
                turn.get("role") == "user"
                and str(turn.get("utterance", "")).strip() == latest_user
            ):
                return index

    for index in range(len(conversation) - 1, -1, -1):
        if conversation[index].get("role") == "user" and conversation[index].get(
            "utterance"
        ):
            return index
    return None


def _is_duplicate_or_substring_variant(candidate: str, latest_user: str) -> bool:
    candidate_normalized = _normalized_text(candidate)
    latest_normalized = _normalized_text(latest_user)
    if not candidate_normalized or not latest_normalized:
        return False
    candidate_tokens = set(_normalize_tokens(candidate))
    latest_tokens = set(_normalize_tokens(latest_user))
    return (
        candidate_normalized == latest_normalized
        or candidate_normalized in latest_normalized
        or latest_normalized in candidate_normalized
        or candidate_tokens.issubset(latest_tokens)
        or latest_tokens.issubset(candidate_tokens)
    )


def build_query_context(example: Example | dict[str, Any]) -> QueryContext:
    """Extract and format essential context from a conversation example.

    This function extracts key dialogue components for document retrieval:
    - Latest user utterance (the current need)
    - Last agent question (if it was a clarifying question)
    - Carry-forward context (up to 2 meaningful prior turns)

    Filters out duplicate phrases and short/meaningless turns to keep context focused.

    Args:
        example: A dialogue record containing the domain, user utterance, and conversation history.

    Returns:
        A QueryContext dict containing the domain, latest user need, last agent question,
        and up to 2 preceding user/agent turns as carry-forward context.
    """
    conversation = _conversation_from_example(example)
    latest_user = _latest_user_utterance(example, conversation).strip()
    latest_user_index = _find_latest_user_index(example, conversation)
    domain = _domain_from_example(example)

    last_agent_question = ""
    if latest_user_index is not None and latest_user_index > 0:
        preceding_turn = conversation[latest_user_index - 1]
        preceding_text = str(preceding_turn.get("utterance", "")).strip()
        if preceding_turn.get("role") == "agent" and preceding_text.endswith("?"):
            last_agent_question = preceding_text

    user_context: list[QueryContextTurn] = []
    agent_context: list[QueryContextTurn] = []
    prior_turns = (
        conversation[:latest_user_index]
        if latest_user_index is not None
        else conversation[:-1]
    )
    for turn in reversed(prior_turns):
        utterance = str(turn.get("utterance", "")).strip()
        if not utterance or utterance == last_agent_question:
            continue
        if len(_normalize_tokens(utterance)) < 4:
            continue
        if _is_duplicate_or_substring_variant(utterance, latest_user):
            continue
        candidate: QueryContextTurn = {
            "role": str(turn.get("role", "")).strip() or "unknown",
            "utterance": utterance,
        }
        if candidate["role"] == "user":
            user_context.append(candidate)
        else:
            agent_context.append(candidate)

    carry_forward_context = (user_context[:2] + agent_context[:2])[:2]
    return {
        "domain": domain,
        "latest_user_need": latest_user,
        "last_agent_question": last_agent_question,
        "carry_forward_context": carry_forward_context,
    }


def _render_query_context(
    context: QueryContext,
    *,
    include_history: bool = True,
    include_labels: bool = True,
) -> str:
    parts: list[str] = []
    if context.get("domain"):
        parts.append(
            f"Domain: {context['domain']}" if include_labels else str(context["domain"])
        )
    if context.get("latest_user_need"):
        parts.append(
            f"Latest user need: {context['latest_user_need']}"
            if include_labels
            else str(context["latest_user_need"])
        )
    if include_history and context.get("last_agent_question"):
        parts.append(
            f"Last agent question: {context['last_agent_question']}"
            if include_labels
            else str(context["last_agent_question"])
        )
    carry_forward = list(context["carry_forward_context"]) if include_history else []
    if carry_forward:
        if include_labels:
            parts.append("Carry-forward context:")
        for turn in carry_forward:
            role = turn["role"].strip().title() or "Unknown"
            utterance = turn["utterance"].strip()
            parts.append(f"- {role}: {utterance}" if include_labels else utterance)
    return "\n".join(parts).strip()


def build_query(
    example: Example | dict[str, Any],
    history_turn_limit: int = 4,
    *,
    include_history: bool = True,
    include_labels: bool = True,
) -> str:
    del history_turn_limit
    context = build_query_context(example)
    if not include_history:
        context = {
            **context,
            "last_agent_question": "",
            "carry_forward_context": [],
        }
    return _render_query_context(
        context, include_history=include_history, include_labels=include_labels
    )


def build_legacy_query(
    example: Example | dict[str, Any],
    history_turn_limit: int = 4,
    *,
    include_history: bool = True,
) -> str:
    conversation = _conversation_from_example(example)
    latest_user = _latest_user_utterance(example, conversation)
    domain = _domain_from_example(example)

    history_lines = [
        f"{str(turn.get('role', '')).title()}: {turn.get('utterance', '')}"
        for turn in conversation[-history_turn_limit:]
        if turn.get("utterance")
    ]

    parts: list[str] = []
    if domain:
        parts.append(f"Domain: {domain}")
    if latest_user:
        parts.append(f"Latest user need: {latest_user}")
    if include_history and history_lines:
        parts.append("Recent conversation:")
        parts.extend(history_lines)

    return "\n".join(parts).strip()


def query_context_tokens(context: QueryContext) -> set[str]:
    values: list[str] = []
    if context["domain"]:
        values.append(context["domain"])
    if context["latest_user_need"]:
        values.append(context["latest_user_need"])
    if context["last_agent_question"]:
        values.append(context["last_agent_question"])
    for turn in context["carry_forward_context"]:
        values.append(turn["utterance"])
    return set(
        _normalize_tokens(" ".join(values), minimum_length=3, drop_stopwords=True)
    )


def build_metadata_filter(
    *,
    domain: str | None = None,
    doc_ids: list[str] | str | None = None,
) -> dict | None:
    """Construct a metadata filter dictionary for vector search.

    Translates basic criteria into vector store-compatible filters:
    - Sets exact `domain` matching
    - Deduplicates and handles single vs. multiple (`$in`) `doc_id` filters

    Args:
        domain: Domain string to filter by (e.g., 'dmv').
        doc_ids: Optional document ID or list of permissible document IDs.

    Returns:
        Filter criteria dictionary, or None if empty.
    """
    filters: dict[str, Any] = {}
    if domain:
        filters["domain"] = domain

    values = []
    if isinstance(doc_ids, str):
        values = [doc_ids]
    elif doc_ids:
        values = [value for value in doc_ids if value]
    deduped_values = list(dict.fromkeys(values))

    if len(deduped_values) == 1:
        filters["doc_id"] = deduped_values[0]
    elif len(deduped_values) > 1:
        filters["doc_id"] = {"$in": deduped_values}

    return filters or None


def _hit_document_and_score(hit: Any) -> tuple[Any, float | None]:
    if isinstance(hit, tuple):
        document, score = hit
        return document, score
    if isinstance(hit, dict):
        return hit, hit.get("score")
    return hit, getattr(hit, "score", None)


def _document_page_content_and_metadata(
    document: Any,
) -> tuple[str, dict[str, Any]]:
    if isinstance(document, dict):
        return str(document.get("page_content", "")), document.get("metadata", {}) or {}
    return document.page_content, document.metadata or {}


def normalize_retrieval_hits(hits: list[Any]) -> list[NormalizedRetrievalHit]:
    """Standardize hit objects from different retrievers into a common format.

    Unwraps diverse formats into a unified `NormalizedRetrievalHit` dictionary:
    - Tuples of `(Document, score)`
    - Dictionary representations
    - Objects with `page_content` and `metadata`

    Args:
        hits: Raw hit objects from vector or BM25 retrievers.

    Returns:
        List of standardized NormalizedRetrievalHit dictionaries.
    """
    normalized: list[NormalizedRetrievalHit] = []
    for index, hit in enumerate(hits, start=1):
        document, score = _hit_document_and_score(hit)
        page_content, metadata = _document_page_content_and_metadata(document)

        normalized.append(
            {
                "rank": index,
                "original_rank": index,
                "chunk_id": metadata.get("chunk_id"),
                "domain": metadata.get("domain"),
                "doc_id": metadata.get("doc_id"),
                "doc_title": metadata.get("doc_title"),
                "section_id": metadata.get("section_id"),
                "section_title": metadata.get("section_title"),
                "parent_titles": metadata.get("parent_titles", []),
                "span_ids": metadata.get("span_ids", []),
                "token_count": metadata.get("token_count"),
                "text": page_content,
                "score": score,
                "vector_distance": score,
            }
        )
    return normalized


def _overlap_count(query_tokens: set[str], text: str) -> int:
    if not query_tokens or not text:
        return 0
    candidate_tokens = set(
        _normalize_tokens(text, minimum_length=3, drop_stopwords=True)
    )
    return len(query_tokens & candidate_tokens)


def rerank_retrieval_hits(
    hits: list[NormalizedRetrievalHit],
    *,
    query_context: QueryContext | None = None,
) -> list[NormalizedRetrievalHit]:
    """Rerank retrieval results using a heuristic scoring algorithm.

    Applies custom heuristics to sort chunks from multiple retrievers:
    - Penalizes chunks with low text/title overlap with query context
    - Awards bonuses to title chunks, short chunks, and single-span chunks

    Args:
        hits: List of NormalizedRetrievalHit dicts.
        query_context: Optional context for measuring token overlap.

    Returns:
        Hits sorted by `rerank_score` with updated `rank` attributes.
    """
    reranked = [dict(hit) for hit in hits]
    query_tokens = query_context_tokens(query_context or {})

    def sort_key(hit: dict) -> tuple[float, int]:
        score = hit.get("vector_distance", hit.get("score"))
        base_score = (
            float(score)
            if score is not None
            else float(hit.get("original_rank", hit.get("rank", 0)))
        )
        section_id = str(hit.get("section_id", "")).strip()
        token_count = int(hit.get("token_count") or 0)
        span_count = len(hit.get("span_ids", []))
        text_overlap_count = _overlap_count(query_tokens, str(hit.get("text", "")))
        title_text = " ".join(
            [
                str(hit.get("section_title", "")),
                *[str(title) for title in hit.get("parent_titles", [])],
            ]
        )
        title_overlap_count = _overlap_count(query_tokens, title_text)

        rerank_score = (
            base_score
            + (0.08 if section_id.startswith("t_") else 0.0)
            + (0.04 if token_count < 12 else 0.0)
            + (0.02 if span_count == 1 else 0.0)
            - (0.03 * min(3, text_overlap_count))
            - (0.04 * min(2, title_overlap_count))
        )
        hit["text_overlap_count"] = text_overlap_count
        hit["title_overlap_count"] = title_overlap_count
        hit["rerank_score"] = round(rerank_score, 6)
        return (rerank_score, int(hit.get("original_rank", hit.get("rank", 0))))

    reranked.sort(key=sort_key)
    for index, hit in enumerate(reranked, start=1):
        hit["rank"] = index
    return reranked


def get_vectorstore(
    config: RuntimeConfig,
    *,
    vectorstore_cls: type[PGVector] = PGVector,
    embeddings: Any = None,
    create_extension: bool = True,
) -> Any:
    validate_index_config(config)
    embedding_client = (
        embeddings if embeddings is not None else build_embeddings(config)
    )
    collection_name = config.collection_name or build_collection_name(config.domain)
    return vectorstore_cls(
        embeddings=embedding_client,
        connection=normalize_postgres_connection(config.postgres_dsn),
        collection_name=collection_name,
        use_jsonb=True,
        create_extension=create_extension,
    )


def build_keyword_retriever(
    config: RuntimeConfig,
    chunk_records: list[dict] | None = None,
) -> BM25Retriever | None:
    if not chunk_records:
        return None

    from langchain_core.documents import Document

    documents = [
        Document(
            page_content=record.get("text", ""),
            metadata={key: value for key, value in record.items() if key != "text"},
        )
        for record in chunk_records
    ]
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = config.retrieval_candidate_k
    return retriever


def _normalized_bm25_hits(
    *,
    retriever: BM25Retriever | None,
    query: str,
    domain: str,
    doc_ids: list[str] | str | None,
) -> list[NormalizedRetrievalHit]:
    if retriever is None:
        return []

    allowed_doc_ids = set()
    if isinstance(doc_ids, str):
        allowed_doc_ids.add(doc_ids)
    elif doc_ids:
        allowed_doc_ids.update(value for value in doc_ids if value)

    hits: list[Any] = []
    for hit in retriever.invoke(query):
        metadata = hit.metadata
        if domain and metadata.get("domain") != domain:
            continue
        if allowed_doc_ids and metadata.get("doc_id") not in allowed_doc_ids:
            continue
        hits.append(hit)
    return normalize_retrieval_hits(hits)


def retrieve_chunks(
    *,
    example: Example | dict[str, Any],
    vectorstore: VectorStoreLike | None = None,
    keyword_retriever: RetrieverLike | None = None,
    config: RuntimeConfig | None = None,
    top_k: int = 5,
    candidate_k: int | None = None,
    query: str | None = None,
    query_context: QueryContext | None = None,
    domain: str | None = None,
    doc_ids: list[str] | str | None = None,
    rerank: bool = True,
) -> list[NormalizedRetrievalHit]:
    """Execute the end-to-end document retrieval pipeline.

    Pipeline stages:
    - Dense vector search (primary)
    - Optional BM25 keyword search (supplementary)
    - Hit normalization and deduplication by chunk ID
    - Optional heuristic reranking

    Args:
        example: Dialogue example used for query/context generation.
        vectorstore: Vector store instance to query.
        keyword_retriever: Optional BM25 retriever.
        config: Runtime configuration for defaults.
        top_k: Final number of chunks to return.
        candidate_k: Number of pre-rerank candidates.
        query: Explicit query string (overrides example-based generation).
        query_context: Explicit context for reranking.
        domain: Domain filter (overrides example-based generation).
        doc_ids: Optional exact document ID or list of allowed document IDs.
        rerank: Whether to apply heuristic reranking to candidates.

    Returns:
        List of top_k normalized chunk dicts, sorted by relevance.
    """
    resolved_query_context = query_context or build_query_context(example)
    resolved_query = query or build_query(example, include_labels=False)
    resolved_domain = domain or _domain_from_example(example)
    resolved_top_k = max(1, int(top_k))
    resolved_candidate_k = candidate_k
    if resolved_candidate_k is None:
        resolved_candidate_k = (
            config.retrieval_candidate_k if config is not None else resolved_top_k
        )
    resolved_candidate_k = max(resolved_top_k, int(resolved_candidate_k))
    metadata_filter = build_metadata_filter(
        domain=resolved_domain,
        doc_ids=doc_ids,
    )

    if vectorstore is None:
        if config is None:
            raise ValueError("retrieve_chunks requires either vectorstore or config.")
        vectorstore = get_vectorstore(config)

    hits = vectorstore.similarity_search_with_score(
        resolved_query,
        k=resolved_candidate_k,
        filter=metadata_filter,
    )
    normalized_hits = normalize_retrieval_hits(hits)

    bm25_hits = _normalized_bm25_hits(
        retriever=keyword_retriever,
        query=resolved_query,
        domain=resolved_domain,
        doc_ids=doc_ids,
    )
    if bm25_hits:
        seen_chunk_ids = {hit["chunk_id"] for hit in normalized_hits}
        for hit in bm25_hits:
            if hit["chunk_id"] not in seen_chunk_ids:
                normalized_hits.append(hit)
                seen_chunk_ids.add(hit["chunk_id"])

    if not rerank:
        return normalized_hits[:resolved_top_k]
    reranked_hits = rerank_retrieval_hits(
        normalized_hits, query_context=resolved_query_context
    )
    return reranked_hits[:resolved_top_k]
