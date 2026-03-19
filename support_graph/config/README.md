# `support_graph/config/`

This package turns environment and path state into typed configuration objects that the rest of the app can trust.

## File Map

| File | Responsibility | Connects To |
| --- | --- | --- |
| `__init__.py` | Package marker for configuration helpers. | No runtime logic. |
| `settings.py` | Loads `.env` and process environment values, resolves project-relative paths, validates provider requirements, exposes derived directories, and reports missing config for indexing or runtime. | Used first by the CLI. Feeds `RuntimeConfig` assembly and preflight checks. |
| `runtime.py` | Defines `RuntimeConfig` plus small config protocols, builds a runtime config from `Settings`, and supports additive overrides for evals and ablations. | Consumed by `retrieval/`, `runtime/`, `evaluation/`, and CLI command handlers. |

## How It Connects Later On

1. `Settings.from_env()` is the first typed object created by most CLI commands.
2. `build_runtime_config()` turns those raw settings into the smaller config contract used by indexing, retrieval, runtime execution, and evaluation.
3. `with_runtime_config_overrides()` lets evaluation and ablation code tweak retrieval behavior without changing the base environment contract.
4. Because this package is the shared source of truth for paths, provider selection, and concurrency settings, it keeps downstream modules from re-parsing env vars independently.
