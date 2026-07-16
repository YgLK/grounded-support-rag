"""Immutable LangSmith dataset publication."""

from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from support_graph.evaluation.contracts import DatasetRef, LangSmithGateway


def canonical_dataset_payload(examples: list[dict[str, Any]]) -> bytes:
    """Serialize examples deterministically without changing their order."""
    return json.dumps(
        examples,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def dataset_sha256(examples: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_dataset_payload(examples)).hexdigest()


def dataset_name(domain: str, subset: str, sha256: str) -> str:
    return f"support-graph/{domain}/{subset}/{sha256[:12]}"


async def publish_dataset(
    gateway: LangSmithGateway,
    *,
    domain: str,
    subset: str,
    examples: list[dict[str, Any]],
) -> DatasetRef:
    """Reuse or publish an immutable dataset addressed by its content hash."""
    _validate_examples(examples)
    digest = dataset_sha256(examples)
    name = dataset_name(domain, subset, digest)
    existing = await gateway.get_dataset(name)
    if existing is not None:
        if existing.sha256 != digest:
            raise ValueError(f"LangSmith dataset hash mismatch for {name}")
        return existing

    return await gateway.create_dataset(
        name=name,
        description=f"SupportGraph {domain} {subset} evaluation dataset.",
        metadata={
            "dataset_sha256": digest,
            "domain": domain,
            "subset": subset,
            "schema_version": "1",
        },
        examples=[_dataset_example(example) for example in examples],
    )


def _validate_examples(examples: list[dict[str, Any]]) -> None:
    if not examples:
        raise ValueError("Evaluation dataset must contain at least one example")
    example_ids = [str(example.get("example_id", "")) for example in examples]
    if any(not example_id for example_id in example_ids):
        raise ValueError(
            "Evaluation dataset examples require non-empty example_id values"
        )
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("Evaluation dataset example_id values must be unique")


def _dataset_example(example: dict[str, Any]) -> dict[str, Any]:
    example_id = str(example["example_id"])
    return {
        "id": str(uuid5(NAMESPACE_URL, f"support-graph/example/{example_id}")),
        "inputs": {"example": example},
        "outputs": {
            "reference_answer": example.get("reference_answer"),
            "gold_doc_ids": example.get("gold_doc_ids", []),
            "gold_span_ids": example.get("gold_span_ids", []),
            "acceptable_span_ids": example.get("acceptable_span_ids", []),
            "required_points": example.get("required_points", []),
            "forbidden_claims": example.get("forbidden_claims", []),
            "target_mode": example.get("target_mode", "answer"),
        },
        "metadata": {"example_id": example_id},
    }
