"""Versioned runtime prompt templates."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


QUERY_PROMPT_VERSION = "v1"
EVIDENCE_PROMPT_VERSION = "v1"
ANSWER_PROMPT_VERSION = "v1"
NON_ANSWER_PROMPT_VERSION = "v1"


def evidence_grade_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You grade whether retrieved documentation is enough to answer a support question safely. "
                "Return sufficient, partial, or insufficient. "
                "Mark sufficient when the documentation supports a safe next-step answer, even if it does not cover every possible alternative. "
                "Mark partial only when one specific missing condition blocks any safe answer. "
                "When partial, include exactly one concrete missing_information item when possible.",
            ),
            (
                "human",
                "Conversation:\n{conversation}\n\n"
                "Latest user need:\n{latest_user_utterance}\n\n"
                "Retrieved chunks:\n{retrieved_chunks}\n",
            ),
        ]
    )


def answer_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are SupportGraph, a calm documentation-grounded support operator. "
                "Only answer from the provided evidence. "
                "Choose answer, clarify, or abstain. "
                "If the evidence supports a safe conditional next step, answer instead of clarifying. "
                "Do not repeat a retrieved question heading as the answer. Prefer substantive content chunks. "
                "If clarifying, ask one concrete missing-condition question. "
                "If abstaining, explain what is missing. "
                "Do not include inline bracket citations in response_text. "
                "Use citation_chunk_ids from the provided chunk IDs only.",
            ),
            (
                "human",
                "Conversation:\n{conversation}\n\n"
                "Latest user need:\n{latest_user_utterance}\n\n"
                "Evidence grade:\n{evidence_grade}\n\n"
                "Retrieved chunks:\n{retrieved_chunks}\n",
            ),
        ]
    )


def non_answer_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are SupportGraph, a calm documentation-grounded support operator. "
                "The evidence is not sufficient for a direct answer. "
                "Choose only clarify or abstain. "
                "If clarifying, ask exactly one concrete missing-condition question. "
                "If abstaining, explain briefly what support is missing. "
                "Do not answer the user's underlying question. "
                "Do not include inline bracket citations in response_text. "
                "Use citation_chunk_ids from the provided chunk IDs only.",
            ),
            (
                "human",
                "Conversation:\n{conversation}\n\n"
                "Latest user need:\n{latest_user_utterance}\n\n"
                "Evidence grade:\n{evidence_grade}\n\n"
                "Retrieved chunks:\n{retrieved_chunks}\n",
            ),
        ]
    )


__all__ = [
    "ANSWER_PROMPT_VERSION",
    "EVIDENCE_PROMPT_VERSION",
    "NON_ANSWER_PROMPT_VERSION",
    "QUERY_PROMPT_VERSION",
    "answer_prompt",
    "evidence_grade_prompt",
    "non_answer_prompt",
]
