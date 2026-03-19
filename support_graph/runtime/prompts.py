"""Versioned runtime prompt registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from langchain_core.prompts import ChatPromptTemplate


DEFAULT_PROMPT_VERSION = "v1"


@dataclass(frozen=True, slots=True)
class PromptSet:
    version: str
    evidence_grade: ChatPromptTemplate
    answer: ChatPromptTemplate
    non_answer: ChatPromptTemplate
    streaming_answer: ChatPromptTemplate


@dataclass(frozen=True, slots=True)
class PromptRegistry:
    prompt_sets: Mapping[str, PromptSet]
    default_version: str = DEFAULT_PROMPT_VERSION

    def resolve(self, version: str | None = None) -> PromptSet:
        resolved_version = self.default_version if version is None else version
        if resolved_version not in self.prompt_sets:
            available = ", ".join(sorted(self.prompt_sets))
            raise ValueError(
                f"Unsupported prompt version {resolved_version!r}. Available: {available}."
            )
        return self.prompt_sets[resolved_version]


def _v1_prompt_set() -> PromptSet:
    return PromptSet(
        version=DEFAULT_PROMPT_VERSION,
        evidence_grade=ChatPromptTemplate.from_messages(
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
        ),
        answer=ChatPromptTemplate.from_messages(
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
        ),
        non_answer=ChatPromptTemplate.from_messages(
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
        ),
        streaming_answer=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are SupportGraph, a calm documentation-grounded support operator. "
                    "Respond using only the provided evidence. "
                    "Produce only the user-facing answer text with no citations, labels, or JSON.",
                ),
                (
                    "human",
                    "Conversation:\n{conversation}\n\n"
                    "Latest user need:\n{latest_user_utterance}\n\n"
                    "Evidence grade:\n{evidence_grade}\n\n"
                    "Retrieved chunks:\n{retrieved_chunks}\n",
                ),
            ]
        ),
    )


PROMPT_REGISTRY = PromptRegistry({DEFAULT_PROMPT_VERSION: _v1_prompt_set()})


def resolve_prompt_set(version: str | None = None) -> PromptSet:
    return PROMPT_REGISTRY.resolve(version)


__all__ = [
    "DEFAULT_PROMPT_VERSION",
    "PROMPT_REGISTRY",
    "PromptRegistry",
    "PromptSet",
    "resolve_prompt_set",
]
