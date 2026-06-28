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
    load_or_build_chunks,
    load_settings,
    next_step_lines,
    print_lines,
)
from support_graph.data.chunks import build_chunks


def _oversized_section_demo() -> list[dict]:
    document = {
        "domain": "dmv",
        "doc_id": "doc-1",
        "title": "Doc",
        "doc_text": "",
        "spans": [
            {
                "id_sp": "1",
                "tag": "u",
                "start_sp": 0,
                "end_sp": 10,
                "text_sp": "one two three four ",
                "title": "Section",
                "parent_titles": ["Parent"],
                "id_sec": "10",
                "start_sec": 0,
                "end_sec": 20,
                "text_sec": "one two three four ",
            },
            {
                "id_sp": "2",
                "tag": "u",
                "start_sp": 10,
                "end_sp": 20,
                "text_sp": "five six seven eight ",
                "title": "Section",
                "parent_titles": ["Parent"],
                "id_sec": "10",
                "start_sec": 0,
                "end_sec": 20,
                "text_sec": "five six seven eight ",
            },
        ],
    }
    return build_chunks([document], max_tokens_per_chunk=4)


def build_lines() -> list[str]:
    settings = load_settings()
    chunks = load_or_build_chunks(settings)
    doc_chunks = [chunk for chunk in chunks if chunk["doc_id"] == SAMPLE_DOC_ID]
    renew_chunk = find_by_key(
        doc_chunks, "chunk_id", "dmv::Registrations#3_0::sec::4::sub::0"
    )
    oversized = _oversized_section_demo()

    lines = [
        "SupportGraph Example 04",
        "Chunking EDA",
        "",
        "Committed DMV chunk artifact",
        f"- Chunk count: {len(chunks)}",
        f"- Chunks for {SAMPLE_DOC_ID}: {len(doc_chunks)}",
        f"- First doc chunk IDs: {[chunk['chunk_id'] for chunk in doc_chunks[:5]]}",
        "",
        "One real chunk",
    ]
    lines.extend(json_lines(renew_chunk))
    lines.extend(
        [
            "",
            "Oversized section demo",
            "- same section id, but deterministic subchunk IDs",
        ]
    )
    for chunk in oversized:
        lines.append(f"- {chunk['chunk_id']} -> span_ids={chunk['span_ids']}")
    lines.extend(next_step_lines("05_subset_eda.py"))
    return lines


def main() -> None:
    print_lines(build_lines())


if __name__ == "__main__":
    main()
