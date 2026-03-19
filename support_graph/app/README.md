# `support_graph/app/`

This package is the CLI surface. It is the only layer that speaks directly to the terminal and turns commands into calls into the rest of the system.

## File Map

| File | Responsibility | Connects To |
| --- | --- | --- |
| `__init__.py` | Marks the package and keeps the namespace clean. | No runtime logic. |
| `cli.py` | Defines all CLI commands, formats terminal output, performs config and index preflight checks, and dispatches into the data, retrieval, runtime, and evaluation modules. | Calls `config/` to load settings, `data/` for deterministic prep, `retrieval/index.py` for indexing, `runtime/graph.py` for single-example runs, `runtime/traces.py` for trace inspection, and `evaluation/` for evals, ablations, and benchmarks. |

## How It Connects Later On

1. Every command starts here, so this package is the boundary between user intent and internal modules.
2. Build commands call `support_graph.data.*` and write reproducible JSONL artifacts under `data/derived/` or `data/eval_subsets/`.
3. Runtime commands build a `RuntimeConfig`, then call `support_graph.runtime.graph.run_graph_async`.
4. Eval and ablation commands call `support_graph.evaluation.evaluate` and `support_graph.evaluation.ablation`, which themselves reuse the same runtime graph used by `run`.
5. Trace review commands read the JSONL trace files produced by `support_graph.runtime.traces` and summarize them for terminal inspection.
