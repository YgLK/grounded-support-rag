from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples._shared import (
    SAMPLE_EXAMPLE_ID,
    load_committed_subset,
    load_or_build_examples,
    load_settings,
    next_step_lines,
    print_lines,
)
from support_graph.data.eval_subsets import build_subset


def build_lines() -> list[str]:
    settings = load_settings()
    examples = load_or_build_examples(settings)
    smoke = build_subset(examples, size=25, target_mode="answer", salt="smoke")
    frozen = build_subset(
        examples, size=200, target_mode="answer", salt="frozen_ablation"
    )
    committed_smoke = load_committed_subset(settings, "smoke")
    committed_frozen = load_committed_subset(settings, "frozen_ablation")

    lines = [
        "SupportGraph Example 05",
        "Subset EDA",
        "",
        "How subsets are derived",
        "- filter to target_mode=answer",
        "- sort by sha256(salt:example_id)",
        "- take the requested size, then sort by example_id for stable files",
        "",
        "Smoke",
        f"- Size: {len(smoke)}",
        f"- Matches committed file: {'yes' if smoke == committed_smoke else 'no'}",
        f"- First 3 example IDs: {[example['example_id'] for example in smoke[:3]]}",
        "",
        "Frozen ablation",
        f"- Size: {len(frozen)}",
        f"- Matches committed file: {'yes' if frozen == committed_frozen else 'no'}",
        f"- First 3 example IDs: {[example['example_id'] for example in frozen[:3]]}",
        "",
        "Sample example membership",
        f"- {SAMPLE_EXAMPLE_ID} in smoke: {SAMPLE_EXAMPLE_ID in {example['example_id'] for example in committed_smoke}}",
        f"- {SAMPLE_EXAMPLE_ID} in frozen_ablation: {SAMPLE_EXAMPLE_ID in {example['example_id'] for example in committed_frozen}}",
    ]
    lines.extend(next_step_lines("06_query_and_retrieval_eda.py"))
    return lines


def main() -> None:
    print_lines(build_lines())


if __name__ == "__main__":
    main()
