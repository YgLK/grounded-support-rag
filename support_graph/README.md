# `support_graph/`

This package is the application core. The CLI, data pipeline, retrieval stack, runtime graph, and evaluation harness all live under this package root.

## File Map

| File | Responsibility | Connects To |
| --- | --- | --- |
| `__init__.py` | Exposes the package version. | Used by packaging and any version-aware tooling. |
| `__main__.py` | Makes `python -m support_graph` behave like the CLI entrypoint. | Delegates directly to `support_graph.app.cli.main`. |
| `types.py` | Defines shared enums, literals, and parser helpers for domains, dataset splits, eval subsets, and turn roles. | Imported by `config/`, `data/`, `app/`, and `evaluation/` so the whole app speaks the same vocabulary. |
| `providers.py` | Centralizes provider validation and model or embedding construction for Ollama, OpenAI, and Anthropic. | Used by `config/settings.py`, `retrieval/index.py`, `runtime/nodes.py`, and `evaluation/benchmark.py`. |
| `logging_utils.py` | Configures project logging and per-command log files. | Used by the CLI and by runtime or evaluation modules that emit structured logs. |

## Subpackages

| Directory | Responsibility | Guide |
| --- | --- | --- |
| `app/` | CLI command parsing and terminal output. | [app/README.md](app/README.md) |
| `config/` | Environment loading and runtime config assembly. | [config/README.md](config/README.md) |
| `data/` | Dataset loading and deterministic artifact generation. | [data/README.md](data/README.md) |
| `retrieval/` | pgvector indexing, query building, retrieval, and reranking. | [retrieval/README.md](retrieval/README.md) |
| `runtime/` | LangGraph workflow, node logic, prompts, traces, and observability. | [runtime/README.md](runtime/README.md) |
| `evaluation/` | Offline metrics, eval runs, ablations, and embedding benchmarks. | [evaluation/README.md](evaluation/README.md) |
| `ui/` | Workbench artifact explorer, trace inspector, and live run UI. | [ui/README.md](ui/README.md) |

## How The Package Connects End To End

1. `support_graph.app.cli` is the outer shell. It parses user commands and decides which pipeline path to run.
2. `support_graph.config.settings` loads `.env` values and project paths, then `support_graph.config.runtime` converts them into a `RuntimeConfig`.
3. `support_graph.data` turns raw MultiDoc2Dial files into deterministic chunk, example, and subset artifacts on disk.
4. `support_graph.retrieval.index` reads chunk artifacts and pushes them into pgvector. Later, `support_graph.retrieval.retrieve` uses the same collection for query-time search.
5. `support_graph.runtime.graph` and `support_graph.runtime.nodes` execute the shared `run` or `eval` workflow, using `providers.py` for model clients, `prompts.py` for prompt templates, and `traces.py` for local execution traces.
6. `support_graph.evaluation.evaluate` repeatedly calls the same runtime graph, scores outputs, and writes eval artifacts. `ablation.py` and `benchmark.py` sit on top of that layer for deeper analysis.
7. `support_graph.ui` reads those on-disk artifacts directly, renders server-driven Workbench pages, and streams the live run surface over SSE without redefining the backend contracts.
