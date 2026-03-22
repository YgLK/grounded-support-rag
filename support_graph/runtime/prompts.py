"""Versioned runtime prompt registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from langchain_core.prompts import ChatPromptTemplate


V1_PROMPT_VERSION = "v1"
V2_PROMPT_VERSION = "v2"
DEFAULT_PROMPT_VERSION = "v2"


@dataclass(frozen=True, slots=True)
class PromptSet:
    version: str
    route_query: ChatPromptTemplate
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
        version=V1_PROMPT_VERSION,
        route_query=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Classify if the user query is 'chitchat' or a 'document_query'.",
                ),
                (
                    "human",
                    "Query: {latest_user_utterance}",
                ),
            ]
        ),
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


def _v2_prompt_set() -> PromptSet:
    return PromptSet(
        version=V2_PROMPT_VERSION,
        route_query=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a router. Classify user intent:\n"
                    "- document_query: Factual questions about DMV, licenses, rules, or fees.\n"
                    "- chitchat: Greetings, thanks, generic feedback, or non-factual statements.\n\n"
                    "Reason before you decide.",
                ),
                (
                    "human",
                    "Conversation History:\n{conversation}\n\n"
                    "Current User Need: {latest_user_utterance}",
                ),
            ]
        ),
        evidence_grade=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert documentation grader. Your task is to determine if the retrieved context is sufficient to answer a user's question SAFELY and ACCURATELY.\n\n"
                    "Verdicts:\n"
                    "- sufficient: The context contains the direct answer or a clear conditional next step.\n"
                    "- partial: The context is highly relevant but is missing ONE specific detail to be certain.\n"
                    "- insufficient: The context is irrelevant or lacks any substantive facts to help the user.\n\n"
                    "Strict Rule: Do not use your own knowledge. Only grade based on the provided <context> tags.",
                ),
                (
                    "human",
                    "Conversation History:\n{conversation}\n\n"
                    "Current User Need: {latest_user_utterance}\n\n"
                    "Retrieved Context:\n<context>\n{retrieved_chunks}\n</context>",
                ),
            ]
        ),
        answer=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are SupportGraph, a strictly grounded technical support assistant.\n\n"
                    "CRITICAL RULES:\n"
                    "1. ONLY use information from the provided <context> tags.\n"
                    "2. If the context does not contain the answer, you MUST abstain or clarify.\n"
                    "3. NEVER use your internal knowledge about the world (e.g., general DMV rules) if they are not in the context.\n"
                    "4. If the context is 'sufficient', provide a direct, helpful answer.\n"
                    "5. If the context is 'partial', ask for the specific missing piece of information.\n\n"
                    "Output Requirements:\n"
                    "- Do not use inline citations like [1] or [Chunk ID].\n"
                    "- Populate 'citation_chunk_ids' with the specific IDs of chunks you used.\n"
                    "- Keep the tone professional and concise.",
                ),
                (
                    "human",
                    "Conversation History:\n{conversation}\n\n"
                    "Current User Need: {latest_user_utterance}\n\n"
                    "Context Evidence:\n<context>\n{retrieved_chunks}\n</context>\n\n"
                    "Evidence Grade: {evidence_grade}",
                ),
            ]
        ),
        non_answer=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are SupportGraph. You cannot answer the user's question because the documentation is insufficient.\n\n"
                    "Your goal is to either:\n"
                    "1. CLARIFY: Ask for exactly one missing detail if the context was 'partial'.\n"
                    "2. ABSTAIN: Politely explain that you do not have information on this topic in the manuals if the context was 'insufficient'.\n\n"
                    "Never hallucinate or guess.",
                ),
                (
                    "human",
                    "Conversation History:\n{conversation}\n\n"
                    "Current User Need: {latest_user_utterance}\n\n"
                    "Context Evidence:\n<context>\n{retrieved_chunks}\n</context>\n\n"
                    "Evidence Grade: {evidence_grade}",
                ),
            ]
        ),
        streaming_answer=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are SupportGraph. Respond ONLY using the provided <context>.\n"
                    "Produce only the final user-facing text. No JSON, no labels.",
                ),
                (
                    "human",
                    "Conversation History:\n{conversation}\n\n"
                    "Current User Need: {latest_user_utterance}\n\n"
                    "Context Evidence:\n<context>\n{retrieved_chunks}\n</context>",
                ),
            ]
        ),
    )


PROMPT_REGISTRY = PromptRegistry(
    {
        V1_PROMPT_VERSION: _v1_prompt_set(),
        V2_PROMPT_VERSION: _v2_prompt_set(),
    }
)


def resolve_prompt_set(version: str | None = None) -> PromptSet:
    return PROMPT_REGISTRY.resolve(version)


__all__ = [
    "DEFAULT_PROMPT_VERSION",
    "PROMPT_REGISTRY",
    "PromptRegistry",
    "PromptSet",
    "resolve_prompt_set",
]
