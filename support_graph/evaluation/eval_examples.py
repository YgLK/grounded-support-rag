"""Authoring, validation, and promotion helpers for curated eval subsets.

The curated Kubernetes eval set is hand-authored but LLM-assisted: a draft
tool produces corpus-grounded candidates, a validator gates promotion, and a
promote step moves verified rows into the final subset file. Labels are
provenance-anchored to the pinned chunk corpus so the validator can catch
hallucinated doc/span IDs and ungrounded required-point aliases.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from support_graph.data.eval_subsets import load_subset_jsonl, write_subset_jsonl
from support_graph.types import RequiredPoint

__all__ = [
    "VALID_ANSWER_TYPES",
    "ValidationIssue",
    "ValidationReport",
    "ChunkIndex",
    "build_chunk_index",
    "load_examples",
    "write_examples",
    "validate_examples",
    "answer_type_distribution",
    "alias_group_phrases",
    "alias_tokens_in_text",
    "span_text_for_example",
    "promote_examples",
    "Provenance",
]


VALID_ANSWER_TYPES: frozenset[str] = frozenset(
    {"definition", "procedure", "diagnosis", "clarification", "abstain"}
)


@dataclass(slots=True)
class ChunkIndex:
    """Lookup tables over a pinned chunk corpus."""

    doc_ids: frozenset[str]
    span_ids: frozenset[str]
    span_text: dict[str, str]
    chunk_records: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ValidationIssue:
    severity: str  # "error" or "warning"
    example_id: str
    code: str
    message: str


@dataclass(slots=True)
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    example_count: int = 0
    answer_type_distribution: dict[str, int] = field(default_factory=dict)
    duplicate_ids: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def issue_lines(self) -> list[str]:
        lines: list[str] = []
        for issue in self.issues:
            lines.append(
                f"[{issue.severity}] {issue.example_id} | {issue.code}: {issue.message}"
            )
        return lines


def _normalize_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_chunk_index(chunk_records: Iterable[dict[str, Any]]) -> ChunkIndex:
    """Build doc/span ID sets and a span_id -> concatenated-text map."""
    records = list(chunk_records)
    doc_ids: set[str] = set()
    span_ids: set[str] = set()
    span_text: dict[str, list[str]] = {}
    for record in records:
        doc_id = record.get("doc_id")
        if doc_id:
            doc_ids.add(str(doc_id))
        for span_id in record.get("span_ids", []) or []:
            if span_id:
                span_ids.add(str(span_id))
                span_text.setdefault(str(span_id), []).append(
                    str(record.get("text", ""))
                )
    return ChunkIndex(
        doc_ids=frozenset(doc_ids),
        span_ids=frozenset(span_ids),
        span_text={key: " ".join(parts) for key, parts in span_text.items()},
        chunk_records=records,
    )


def load_examples(path: str | Path) -> list[dict[str, Any]]:
    return load_subset_jsonl(path)


def write_examples(examples: list[dict[str, Any]], path: str | Path) -> None:
    write_subset_jsonl(examples, path)


def alias_group_phrases(point: RequiredPoint) -> list[str]:
    if isinstance(point, str):
        return [point]
    return [phrase for phrase in point if isinstance(phrase, str)]


def alias_tokens_in_text(phrase: str, text: str) -> bool:
    """True when all content tokens of ``phrase`` appear in ``text`` (bag-of-words)."""
    phrase_tokens = _normalize_tokens(phrase)
    if not phrase_tokens:
        return False
    text_tokens = set(_normalize_tokens(text))
    return all(token in text_tokens for token in phrase_tokens)


def _example_span_ids(example: dict[str, Any]) -> list[str]:
    return [
        *example.get("gold_span_ids", []),
        *example.get("acceptable_span_ids", []),
    ]


def span_text_for_example(example: dict[str, Any], chunk_index: ChunkIndex) -> str:
    """Concatenated source text for the example's gold + acceptable spans."""
    parts: list[str] = []
    for span_id in _example_span_ids(example):
        text = chunk_index.span_text.get(str(span_id))
        if text:
            parts.append(text)
    return " ".join(parts)


def answer_type_distribution(examples: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for example in examples:
        answer_type = str(example.get("answer_type") or "")
        if answer_type:
            counts[answer_type] += 1
    return dict(sorted(counts.items()))


def validate_examples(
    examples: list[dict[str, Any]],
    chunk_index: ChunkIndex,
    *,
    require_rag_fields: bool = True,
) -> ValidationReport:
    """Validate a curated eval subset against a pinned chunk corpus.

    Errors gate promotion; warnings are surfaced for review. Checks:
    - declared doc/span IDs exist in the chunk corpus
    - each required-point alias group has at least one alias whose tokens
      appear in the source span text (catches hallucinated/ungrounded aliases)
    - no duplicate example_id
    - no missing or empty example_id
    - answer_type is a known value
    """
    report = ValidationReport(
        example_count=len(examples),
        answer_type_distribution=answer_type_distribution(examples),
    )

    seen_ids: dict[str, int] = {}
    for example in examples:
        example_id = str(example.get("example_id") or "")
        seen_ids[example_id] = seen_ids.get(example_id, 0) + 1
    report.duplicate_ids = sorted(
        example_id for example_id, count in seen_ids.items() if count > 1
    )
    for example_id in report.duplicate_ids:
        report.issues.append(
            ValidationIssue(
                severity="error",
                example_id=example_id,
                code="duplicate_example_id",
                message=f"example_id '{example_id}' appears more than once.",
            )
        )
    # A missing or empty example_id is an error: later eval code uses it for
    # trace paths and per-example metrics, and promote_examples uses it as the
    # merge key. A row without a usable ID would be silently dropped or written
    # with an empty key.
    if "" in seen_ids:
        report.issues.append(
            ValidationIssue(
                severity="error",
                example_id="<missing>",
                code="missing_example_id",
                message=(
                    f"{seen_ids['']} example(s) have a missing or empty example_id. "
                    "Every example must have a non-empty example_id."
                ),
            )
        )

    for example in examples:
        example_id = str(example.get("example_id") or "<missing>")

        declared_doc_ids = [
            *example.get("gold_doc_ids", []),
            *example.get("expected_sources", []),
            *example.get("acceptable_sources", []),
        ]
        for doc_id in declared_doc_ids:
            if doc_id and str(doc_id) not in chunk_index.doc_ids:
                report.issues.append(
                    ValidationIssue(
                        severity="error",
                        example_id=example_id,
                        code="unknown_doc_id",
                        message=f"doc_id '{doc_id}' not found in chunk corpus.",
                    )
                )

        declared_span_ids = _example_span_ids(example)
        for span_id in declared_span_ids:
            if span_id and str(span_id) not in chunk_index.span_ids:
                report.issues.append(
                    ValidationIssue(
                        severity="error",
                        example_id=example_id,
                        code="unknown_span_id",
                        message=f"span_id '{span_id}' not found in chunk corpus.",
                    )
                )

        answer_type = str(example.get("answer_type") or "")
        if answer_type and answer_type not in VALID_ANSWER_TYPES:
            report.issues.append(
                ValidationIssue(
                    severity="error",
                    example_id=example_id,
                    code="invalid_answer_type",
                    message=(
                        f"answer_type '{answer_type}' is not one of "
                        f"{sorted(VALID_ANSWER_TYPES)}."
                    ),
                )
            )

        if require_rag_fields:
            required_points = example.get("required_points", []) or []
            if not required_points:
                report.issues.append(
                    ValidationIssue(
                        severity="warning",
                        example_id=example_id,
                        code="missing_required_points",
                        message="No required_points declared; required-point grading will be skipped.",
                    )
                )
            else:
                source_text = span_text_for_example(example, chunk_index)
                for index, point in enumerate(required_points, start=1):
                    phrases = alias_group_phrases(point)
                    if not phrases:
                        report.issues.append(
                            ValidationIssue(
                                severity="error",
                                example_id=example_id,
                                code="empty_alias_group",
                                message=f"required_points[{index}] has no alias phrases.",
                            )
                        )
                        continue
                    grounded = any(
                        alias_tokens_in_text(phrase, source_text) for phrase in phrases
                    )
                    if not grounded:
                        report.issues.append(
                            ValidationIssue(
                                severity="error",
                                example_id=example_id,
                                code="ungrounded_alias_group",
                                message=(
                                    f"required_points[{index}] has no alias whose "
                                    "tokens appear in the source span text."
                                ),
                            )
                        )

    return report


def promote_examples(
    candidates: list[dict[str, Any]],
    *,
    target_path: str | Path,
    existing: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Merge verified candidate rows into the final subset file.

    Existing rows are kept; candidates with a new ``example_id`` are appended.
    Candidates whose ``example_id`` already exists in ``existing`` are skipped
    (verified rows are never overwritten). Returns the merged example list in
    ``example_id`` order and writes it to ``target_path``.
    """
    merged_by_id: dict[str, dict[str, Any]] = {}
    for record in existing or []:
        merged_by_id[str(record.get("example_id"))] = record
    for record in candidates:
        example_id = str(record.get("example_id"))
        if not example_id or example_id in merged_by_id:
            continue
        merged_by_id[example_id] = record
    merged = sorted(merged_by_id.values(), key=lambda r: str(r.get("example_id")))
    write_examples(merged, target_path)
    return merged


# Provenance is a free-form dict carried on each candidate row for review.
Provenance = dict[str, Any]


def load_provenance(example: dict[str, Any]) -> Provenance:
    return dict(example.get("provenance") or {})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_candidates(path: str | Path) -> list[dict[str, Any]]:
    return _read_jsonl(Path(path))


def append_candidates(
    candidates: list[dict[str, Any]], path: str | Path
) -> list[dict[str, Any]]:
    """Append only candidate rows whose example_id is not already present.

    Idempotent: re-running appends only missing seeds; never overwrites
    verified rows. Returns the full candidate list (existing + appended).
    """
    path = Path(path)
    existing = {str(r.get("example_id")) for r in _read_jsonl(path)}
    new_rows = [
        candidate
        for candidate in candidates
        if str(candidate.get("example_id")) not in existing
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in new_rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")
    return _read_jsonl(path)
