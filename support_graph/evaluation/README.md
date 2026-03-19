# `support_graph/evaluation/`

This package owns offline measurement and analysis. It runs the shared runtime graph over fixed example sets, scores the outputs, and writes artifacts for review.

## File Map

| File | Responsibility | Connects To |
| --- | --- | --- |
| `__init__.py` | Package marker for evaluation helpers. | No runtime logic. |
| `evaluate.py` | Loads eval examples or subsets, builds runtime configs, repeatedly calls the shared graph, computes retrieval and generation metrics, labels failures, and writes eval artifacts such as `metrics.json`, `predictions.jsonl`, `failures.jsonl`, `manual_review.csv`, and `summary.md`. | Reuses `data/`, `config/`, `runtime/graph.py`, and `runtime/traces.py`. |
| `ablation.py` | Defines retrieval-side experiment variants, runs the Smoke-10 ladder and optional Frozen-200 follow-through, classifies results, and writes a markdown ablation summary. | Builds on top of `evaluate.py` and still uses the shared runtime graph. |
| `benchmark.py` | Samples chunk records and measures embedding throughput for a configured embedding provider. | Uses `retrieval/index.py` for chunk loading and embedding construction. |

## How It Connects Later On

1. `evaluate.py` is the main offline scoring loop. It is the bridge between deterministic examples and the shared runtime graph.
2. It explicitly keeps retrieval scoring tied to `retrieval_ranked_chunks`, while citation validation and answer grading use `retrieved_chunks`.
3. `ablation.py` layers experiment management on top of the same eval harness so variant comparisons stay apples to apples.
4. `benchmark.py` is adjacent to evaluation rather than runtime because it measures retrieval infrastructure quality, not answer generation behavior.
