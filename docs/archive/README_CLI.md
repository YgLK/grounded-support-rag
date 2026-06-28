# `support_graph/cli/`

This package is the CLI surface. It is the only layer that speaks directly to the terminal and turns commands into calls into the rest of the system.

## File Map

| File | Responsibility | Connects To |
| --- | --- | --- |
| `__init__.py` | Marks the package and keeps the namespace clean. | No runtime logic. |
| `handlers.py` | Thin entrypoint and stable import surface for command handlers. Keeps the names that tests patch while delegating parser and formatting details to smaller modules. | Calls `config/` to load settings, `data/` for deterministic prep, `retrieval/index.py` for indexing, `runtime/graph.py` for single-example runs, `runtime/traces.py` for trace inspection, `evaluation/` for evals and ablations, and `ui/` for the Workbench server. |
| `parser.py` | Builds the top-level `argparse` tree from a set of command handlers. | Depends on CLI constants and command callbacks. |
| `formatting.py` | Formats terminal output blocks for run, eval, ablation, and artifact-inspection commands. | Depends on shared path and metric helpers. |
| `utils.py` | Holds small pure helpers for paths, JSON/JSONL IO, duration formatting, and standard preflight status blocks. | Shared by `handlers.py`, `parser.py`, and `formatting.py`. |

## How It Connects Later On

1. Every command starts here, so this package is the boundary between user intent and internal modules.
2. Build commands call `support_graph.data.*` and write reproducible JSONL artifacts under `data/derived/` or `data/eval_subsets/`.
3. Runtime commands resolve a domain-scoped `RuntimeConfig` from `Settings`, then call `support_graph.runtime.graph.run_graph_async`.
4. Eval and ablation commands call `support_graph.evaluation.evaluate` and `support_graph.evaluation.ablation`, which themselves reuse the same runtime graph used by `run`.
5. Trace review commands read the JSONL trace files produced by `support_graph.runtime.traces` and summarize them for terminal inspection.
6. The `ui` command serves `support_graph.ui.create_app`, which layers HTML, HTMX partials, and live SSE streaming on top of the existing artifact contracts.
