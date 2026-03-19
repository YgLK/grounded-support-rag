# `support_graph/data/`

This package owns deterministic data preparation. It reads the raw MultiDoc2Dial files and produces the stable artifacts that the rest of the system depends on.

## File Map

| File | Responsibility | Connects To |
| --- | --- | --- |
| `__init__.py` | Package marker for dataset and derived-data helpers. | No runtime logic. |
| `_utils.py` | Normalizes common record fragments such as domains, dialogue turns, and reference objects. | Shared by `dataset.py` and `examples.py` so raw data becomes consistent before later stages use it. |
| `dataset.py` | Loads raw document and dialogue JSON files from `multidoc2dial/`, preserves source metadata, and returns predictable in-memory records. | Feeds `chunks.py` and `examples.py`. |
| `chunks.py` | Groups spans into sections, splits long sections deterministically, and writes retrieval chunk JSONL artifacts. | Produces chunk records for `retrieval/index.py`, `runtime/nodes.py`, and embedding benchmarks. |
| `examples.py` | Builds one turn-level example per agent response, infers the target mode, and reads or writes example JSONL files. | Produces runtime and evaluation inputs used by the CLI, `runtime/graph.py`, and `evaluation/evaluate.py`. |
| `eval_subsets.py` | Selects deterministic eval subsets from examples by salted hashing and reads or writes committed subset JSONL files. | Feeds `evaluation/evaluate.py` and `evaluation/ablation.py`. |

## How It Connects Later On

1. `dataset.py` is the raw ingestion layer. It stays close to the source dataset so later stages can be deterministic.
2. `chunks.py` converts raw documents into the retrieval corpus. Those chunk artifacts are what `index-docs` pushes into pgvector.
3. `examples.py` converts dialogues into runtime-ready examples. Those examples are what `run` consumes and what `eval` iterates over.
4. `eval_subsets.py` narrows full example sets into stable benchmark slices, which keeps smoke and frozen runs reproducible.
5. The entire retrieval, runtime, and evaluation stack assumes these artifacts stay stable, so this package is the reproducibility foundation of the project.
