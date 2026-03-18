"""Phase 2 retrieval helpers."""

from __future__ import annotations

import re
from typing import Any

from langchain_postgres import PGVector

from support_graph.index import (
    build_collection_name,
    build_embeddings,
    normalize_postgres_connection,
    validate_index_config,
)


QUERY_TOKEN_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "what",
    "when",
    "how",
    "from",
    "have",
    "this",
    "that",
    "your",
}


def _conversation_from_example(example: dict) -> list[dict]:
    if example.get("conversation"):
        return list(example.get("conversation", []))
    if example.get("turns_before_target"):
        return list(example.get("turns_before_target", []))
    return []


def _latest_user_utterance(example: dict, conversation: list[dict]) -> str:
    latest = example.get("latest_user_utterance")
    if latest:
        return str(latest)
    for turn in reversed(conversation):
        if turn.get("role") == "user" and turn.get("utterance"):
            return str(turn.get("utterance"))
    return ""


def _normalize_tokens(text: str, *, minimum_length: int = 1, drop_stopwords: bool = False) -> list[str]:
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
            if turn.get("role") == "user" and str(turn.get("utterance", "")).strip() == latest_user:
                return index

    for index in range(len(conversation) - 1, -1, -1):
        if conversation[index].get("role") == "user" and conversation[index].get("utterance"):
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


def build_query_context(example: dict) -> dict:
    conversation = _conversation_from_example(example)
    latest_user = _latest_user_utterance(example, conversation).strip()
    latest_user_index = _find_latest_user_index(example, conversation)
    domain = str(example.get("domain_hint") or example.get("domain") or "").strip()

    last_agent_question = ""
    if latest_user_index is not None and latest_user_index > 0:
        preceding_turn = conversation[latest_user_index - 1]
        preceding_text = str(preceding_turn.get("utterance", "")).strip()
        if preceding_turn.get("role") == "agent" and preceding_text.endswith("?"):
            last_agent_question = preceding_text

    user_context: list[dict] = []
    agent_context: list[dict] = []
    prior_turns = conversation[:latest_user_index] if latest_user_index is not None else conversation[:-1]
    for turn in reversed(prior_turns):
        utterance = str(turn.get("utterance", "")).strip()
        if not utterance or utterance == last_agent_question:
            continue
        if len(_normalize_tokens(utterance)) < 4:
            continue
        if _is_duplicate_or_substring_variant(utterance, latest_user):
            continue
        candidate = {"role": str(turn.get("role", "")).strip() or "unknown", "utterance": utterance}
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


def _render_query_context(context: dict, *, include_history: bool = True) -> str:
    parts: list[str] = []
    if context.get("domain"):
        parts.append(f"Domain: {context['domain']}")
    if context.get("latest_user_need"):
        parts.append(f"Latest user need: {context['latest_user_need']}")
    if include_history and context.get("last_agent_question"):
        parts.append(f"Last agent question: {context['last_agent_question']}")
    carry_forward = list(context.get("carry_forward_context", [])) if include_history else []
    if carry_forward:
        parts.append("Carry-forward context:")
        for turn in carry_forward:
            role = str(turn.get("role", "")).strip().title() or "Unknown"
            parts.append(f"- {role}: {str(turn.get('utterance', '')).strip()}")
    return "\n".join(parts).strip()


def build_query(
    example: dict,
    history_turn_limit: int = 4,
    *,
    include_history: bool = True,
) -> str:
    del history_turn_limit
    context = build_query_context(example)
    if not include_history:
        context = {**context, "last_agent_question": "", "carry_forward_context": []}
    return _render_query_context(context, include_history=include_history)


def build_legacy_query(
    example: dict,
    history_turn_limit: int = 4,
    *,
    include_history: bool = True,
) -> str:
    conversation = _conversation_from_example(example)
    latest_user = _latest_user_utterance(example, conversation)
    domain = example.get("domain_hint") or example.get("domain") or ""

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


def query_context_tokens(context: dict) -> set[str]:
    values: list[str] = []
    if context.get("domain"):
        values.append(str(context["domain"]))
    if context.get("latest_user_need"):
        values.append(str(context["latest_user_need"]))
    if context.get("last_agent_question"):
        values.append(str(context["last_agent_question"]))
    for turn in context.get("carry_forward_context", []):
        values.append(str(turn.get("utterance", "")))
    return set(_normalize_tokens(" ".join(values), minimum_length=3, drop_stopwords=True))


def build_metadata_filter(
    *,
    domain: str | None = None,
    doc_id: str | None = None,
    doc_ids: list[str] | None = None,
) -> dict | None:
    filters: dict[str, Any] = {}
    if domain:
        filters["domain"] = domain

    values = [value for value in (doc_ids or []) if value]
    if doc_id:
        values.append(doc_id)
    deduped_values = list(dict.fromkeys(values))

    if len(deduped_values) == 1:
        filters["doc_id"] = deduped_values[0]
    elif len(deduped_values) > 1:
        filters["doc_id"] = {"$in": deduped_values}

    return filters or None


def normalize_retrieval_hits(hits: list[Any]) -> list[dict]:
    normalized: list[dict] = []
    for index, hit in enumerate(hits, start=1):
        if isinstance(hit, tuple):
            document, score = hit
        elif isinstance(hit, dict):
            document = hit
            score = hit.get("score")
        else:
            document = hit
            score = getattr(hit, "score", None)

        if isinstance(document, dict):
            page_content = document.get("page_content", "")
            metadata = document.get("metadata", {}) or {}
        else:
            page_content = getattr(document, "page_content", "")
            metadata = getattr(document, "metadata", {}) or {}

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


def _is_heading_like(hit: dict) -> bool:
    text = " ".join(str(hit.get("text", "")).split()).strip()
    section_title = " ".join(str(hit.get("section_title", "")).split()).strip()
    section_id = str(hit.get("section_id", "")).strip()
    span_ids = hit.get("span_ids", [])
    token_count = hit.get("token_count")

    normalized_text = text.lower().rstrip(".")
    normalized_title = section_title.lower().rstrip(".")
    if section_id.startswith("t_"):
        return True
    if normalized_text and normalized_text == normalized_title and (token_count is None or token_count <= 16):
        return True
    if len(span_ids) <= 1 and len(text.split()) <= 16 and text.endswith("?"):
        return True
    if len(span_ids) <= 1 and len(text.split()) <= 10:
        return True
    return False


def _overlap_count(query_tokens: set[str], text: str) -> int:
    if not query_tokens or not text:
        return 0
    candidate_tokens = set(_normalize_tokens(text, minimum_length=3, drop_stopwords=True))
    return len(query_tokens & candidate_tokens)


def rerank_retrieval_hits(hits: list[dict], *, query_context: dict | None = None) -> list[dict]:
    reranked = [dict(hit) for hit in hits]
    query_tokens = query_context_tokens(query_context or {})

    def sort_key(hit: dict) -> tuple[float, int]:
        score = hit.get("vector_distance", hit.get("score"))
        base_score = float(score) if score is not None else float(hit.get("original_rank", hit.get("rank", 0)))
        section_id = str(hit.get("section_id", "")).strip()
        token_count = int(hit.get("token_count") or 0)
        span_count = len(hit.get("span_ids", []))
        text_overlap_count = _overlap_count(query_tokens, str(hit.get("text", "")))
        title_text = " ".join(
            [str(hit.get("section_title", "")), *[str(title) for title in hit.get("parent_titles", [])]]
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
    config: Any,
    *,
    vectorstore_cls: type[PGVector] = PGVector,
    embeddings: Any = None,
    create_extension: bool = True,
) -> Any:
    validate_index_config(config)
    embedding_client = embeddings if embeddings is not None else build_embeddings(config)
    collection_name = getattr(config, "collection_name", None) or build_collection_name(
        getattr(config, "domain", "dmv")
    )
    return vectorstore_cls(
        embeddings=embedding_client,
        connection=normalize_postgres_connection(getattr(config, "postgres_dsn")),
        collection_name=collection_name,
        use_jsonb=True,
        create_extension=create_extension,
    )


def retrieve_chunks(
    *,
    example: dict,
    vectorstore: Any = None,
    config: Any = None,
    top_k: int = 5,
    candidate_k: int | None = None,
    query: str | None = None,
    query_context: dict | None = None,
    domain: str | None = None,
    doc_id: str | None = None,
    doc_ids: list[str] | None = None,
    rerank: bool = True,
) -> list[dict]:
    resolved_query_context = query_context or build_query_context(example)
    resolved_query = query or build_query(example)
    resolved_domain = domain or example.get("domain_hint") or example.get("domain")
    metadata_filter = build_metadata_filter(
        domain=resolved_domain,
        doc_id=doc_id,
        doc_ids=doc_ids,
    )

    resolved_vectorstore = vectorstore
    if resolved_vectorstore is None:
        if config is None:
            raise ValueError("retrieve_chunks requires either vectorstore or config.")
        resolved_vectorstore = get_vectorstore(config)

    resolved_top_k = max(1, int(top_k))
    resolved_candidate_k = max(
        resolved_top_k,
        int(
            candidate_k
            if candidate_k is not None
            else getattr(config, "retrieval_candidate_k", resolved_top_k)
        ),
    )
    hits = resolved_vectorstore.similarity_search_with_score(
        resolved_query,
        k=resolved_candidate_k,
        filter=metadata_filter,
    )
    normalized_hits = normalize_retrieval_hits(hits)
    if not rerank:
        return normalized_hits[:resolved_top_k]
    reranked_hits = rerank_retrieval_hits(normalized_hits, query_context=resolved_query_context)
    return reranked_hits[:resolved_top_k]
