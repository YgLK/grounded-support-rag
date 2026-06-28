from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples._shared import (  # noqa: E402
    SAMPLE_DIALOGUE_ID,
    load_settings,
    next_step_lines,
    print_lines,
)
from support_graph.data.dataset import load_dialogues, load_documents


def build_lines() -> list[str]:
    settings = load_settings()
    documents = load_documents(settings.dataset.root)
    domain_counts = Counter(document["domain"] for document in documents)
    validation_dialogues = load_dialogues(
        settings.dataset.root, split="validation", domains=["dmv"]
    )

    lines = [
        "SupportGraph Example 01",
        "Raw dataset EDA",
        "",
        "Documents",
        f"- Dataset root: {settings.dataset.root}",
        f"- Document count: {len(documents)}",
        f"- Domain counts: {dict(sorted(domain_counts.items()))}",
        f"- First document ID: {documents[0]['doc_id']}",
        "",
        "Dialogue splits",
    ]
    for split in ("train", "validation", "test"):
        dialogues = load_dialogues(settings.dataset.root, split=split)
        lines.append(f"- {split}: {len(dialogues)} dialogues")
    lines.extend(
        [
            "",
            "DMV sample",
            f"- Validation DMV dialogues: {len(validation_dialogues)}",
            f"- Sample dialogue ID: {SAMPLE_DIALOGUE_ID}",
        ]
    )
    lines.extend(next_step_lines("02_document_anatomy.py"))
    return lines


def main() -> None:
    print_lines(build_lines())


if __name__ == "__main__":
    main()
