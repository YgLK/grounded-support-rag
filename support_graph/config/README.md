# `support_graph/config/`

This package turns repo-local config files into typed configuration objects that the rest of the app can trust.

## File Map

| File | Responsibility | Connects To |
| --- | --- | --- |
| `__init__.py` | Package marker for configuration helpers. | No runtime logic. |
| `settings.py` | Loads `support_graph.toml` into typed Pydantic section models, merges `.env` secrets, exposes the resolved `Settings` object, and reports missing config for indexing or runtime. | Used first by the CLI. Owns default-domain runtime resolution and preflight checks. |
| `runtime.py` | Defines the resolved `RuntimeConfig`, the typed `RuntimeExperimentOverrides`, and the small compatibility shim for older call sites. | Consumed by `retrieval/`, `runtime/`, `evaluation/`, and CLI command handlers. |

## How It Connects Later On

1. `Settings.load()` is the first typed object created by most CLI commands.
2. `SettingsFile.from_toml()` validates the checked-in non-secret config shape before `Settings` resolves repo-relative paths and `.env` secrets.
3. `Settings.load()` also builds the default-domain `RuntimeConfig`, so `settings.runtime` is already ready for preflight checks and provider metadata.
4. `Settings.runtime_for(domain, experiment=...)` derives domain-scoped runtime configs and applies typed experiment overrides for evals and ablations.
