# `support_graph/ui/`

This package is the local Workbench surface. It does not replace the CLI and it does not invent a second storage model. Instead, it reads the existing eval, report, standalone run, and trace artifacts directly, then renders operator-focused pages on top.

## File Map

| File | Responsibility | Connects To |
| --- | --- | --- |
| `__init__.py` | Re-exports the app factory and loader entrypoints. | Used by the CLI and tests. |
| `app.py` | Defines JSON API routes, server-rendered HTML pages, HTMX partials, exception handling, and the live SSE endpoint. | Calls `loaders.py` for artifact views and `live.py` for live execution wiring. |
| `live.py` | Resolves example ids, streams `astream_graph_events(...)`, and persists completed standalone run artifacts for the live screen. | Calls `config/runtime.py`, `data/examples.py`, `data/dataset.py`, and `runtime/graph.py`. |
| `loaders.py` | Normalizes on-disk eval, report, run, and trace artifacts into strict Workbench view models. | Reads paths from `artifacts.py`, parses models from `models.py`, and loads raw trace JSONL. |
| `models.py` | Defines strict Pydantic response and page models for the Workbench backend. | Shared by the JSON API, HTML routes, and tests. |
| `templates/` | Jinja templates for full pages and HTMX partials. HTMX, Alpine.js, and Tabler are loaded from CDN inside the base template. | Rendered by `app.py`. |

## How It Connects Later On

1. `support_graph.app.cli` starts this package through `uv run support-graph ui`.
2. `loaders.py` reads the exact artifact namespaces defined in `ARTIFACT_LAYOUT.md`, keeping `retrieval_ranked_chunks` and `retrieved_chunks` distinct.
3. The HTML layer stays server-driven for artifact exploration, failure filtering, and detail pages; HTMX only swaps fragments and Alpine.js stays limited to small local state.
4. The live run screen calls the same shared runtime graph used by CLI `run` and evals, streams milestones over SSE, and writes completed runs under `outputs/runs/`.
