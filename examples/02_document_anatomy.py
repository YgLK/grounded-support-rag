from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples._shared import (
    SAMPLE_DOC_ID,
    find_by_key,
    json_lines,
    load_settings,
    next_step_lines,
    print_lines,
)
from support_graph.data.dataset import load_documents


def build_lines() -> list[str]:
    settings = load_settings()
    documents = load_documents(settings.dataset.root, domains=["dmv"])
    document = find_by_key(documents, "doc_id", SAMPLE_DOC_ID)
    spans = document["spans"]
    unique_section_ids = []
    for span in spans:
        section_id = span.get("id_sec")
        if section_id not in unique_section_ids:
            unique_section_ids.append(section_id)

    lines = [
        "SupportGraph Example 02",
        "DMV document anatomy",
        "",
        "Document",
        f"- Doc ID: {document['doc_id']}",
        f"- Title: {document['title']}",
        f"- Span count: {len(spans)}",
        f"- Unique section IDs: {len(unique_section_ids)}",
        f"- First section IDs: {unique_section_ids[:8]}",
        "",
        "Why this matters",
        "- spans are already normalized and sorted by section offsets",
        "- each span carries title and parent_titles metadata",
        "- chunk building groups spans by id_sec, then preserves doc-local order",
        "",
        "First span",
    ]
    lines.extend(json_lines(spans[0]))
    lines.extend(["", "Sixth span"])
    lines.extend(json_lines(spans[5]))
    lines.extend(next_step_lines("03_dialogue_and_example_eda.py"))
    return lines


def main() -> None:
    print_lines(build_lines())


if __name__ == "__main__":
    main()
