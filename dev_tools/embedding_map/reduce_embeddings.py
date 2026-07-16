"""Project exported embeddings with PCA and UMAP."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _main() -> int:
    from dev_tools.embedding_map.embedding_map import main

    return main(["project", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(_main())
