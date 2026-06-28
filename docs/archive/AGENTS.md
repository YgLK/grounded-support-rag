# AGENTS.md

## Project Summary

SupportGraph is a CLI-first modular monolith for grounded support assistance over the local MultiDoc2Dial dataset. The current MVP centers on the `dmv` domain, uses `validation` as the headline eval split, writes local traces and eval artifacts to disk, and optionally integrates with LangSmith or OpenTelemetry.

The system has three main modes:

- deterministic data prep: load raw dataset files, build chunks, build turn-level examples, build fixed eval subsets
- retrieval/runtime: index chunks into Postgres + pgvector, run the LangGraph workflow, produce grounded responses with citations
- offline evaluation: score retrieval and generation, write machine-readable artifacts plus a short markdown summary

## Repository Structure

- `support_graph/data/`: raw dataset loading, chunk building, turn example building, eval subsets
- `support_graph/retrieval/`: pgvector indexing, query handling, retrieval, reranking
- `support_graph/runtime/`: LangGraph runtime, nodes, schemas, prompts, tracing, observability
- `support_graph/evaluation/`: metrics, eval runs, experiments, embedding benchmark
- `support_graph/config/`: env-backed settings and runtime config assembly
- `support_graph/app/cli.py`: CLI entrypoint and terminal output formatting
- `examples/`: guided scripts that explain the pipeline end to end
- `multidoc2dial/`: committed raw dataset copy
- `data/eval_subsets/`: committed `smoke` and `frozen_experiment` subsets
- `outputs/evals/runs/`: eval run artifacts
- `outputs/evals/reports/`: eval reports such as experiment summaries
- `outputs/runs/`: standalone `run` artifacts
- `ARTIFACT_LAYOUT.md`: source of truth for generated artifact paths, filenames, and schemas
- `KNOWLEDGE_VAULT.md`: retained architectural notes and evaluation heuristics
- `TODO.md`: deferred work list with priorities
- `docs/archive/`: historical planning and spec docs that are no longer treated as active source-of-truth documents

## Setup

Base setup:

```bash
uv sync --dev
cp .env.example .env
cp support_graph.toml.example support_graph.toml
```

Safe local-only work:

- `build-chunks`
- `build-examples`
- `build-subsets`
- most unit and slice tests under `tests/`
- example scripts `00` through `06`

Commands that require live services:

- `index-docs`
- `run`
- `eval`
- `experiment-smoke10`
- `benchmark-embeddings`
- example scripts `07` through `10`

Default local service path:

```bash
cp .env.example .env
cp support_graph.toml.example support_graph.toml
docker compose up -d postgres
```

## Common Commands

```bash
uv run grounded-support-rag build-chunks --domain dmv
uv run grounded-support-rag build-examples --domain dmv --split validation
uv run grounded-support-rag build-subsets --domain dmv --split validation
uv run grounded-support-rag index-docs --domain dmv
uv run grounded-support-rag run --example-id 'dmv::1409501a35697e0ce68561e29577b90a::turn_2' --verbose
uv run grounded-support-rag eval --domain dmv --subset smoke
uv run grounded-support-rag experiment-smoke10 --domain dmv --limit 10
uv run grounded-support-rag review-failures --run-id <run_id>
uv run grounded-support-rag trace-show --run-id <run_id> --example-id <example_id>
uv run pytest
uv run pytest --run-integration
```

## Project Invariants

Keep these true unless the change explicitly redefines the contract:

- `dmv` is the required end-to-end MVP domain.
- `validation` is the headline eval split. Do not treat `test` as the default benchmark.
- Data-prep paths should remain deterministic and reproducible.
- Chunk IDs, example IDs, subset contents, and output file names should stay stable unless the schema change is deliberate and documented.
- `run` and `eval` share the same runtime graph in `support_graph/runtime/graph.py`.
- Default CLI output should stay concise and human-readable, with verbose detail behind flags or artifact inspection.
- Generated artifact layout, filenames, and schemas should follow `ARTIFACT_LAYOUT.md`.

One easy-to-break distinction matters in runtime and evaluation:

- `retrieval_ranked_chunks` is the direct ranked retrieval list used by retrieval metrics.
- `retrieved_chunks` is the expanded evidence set used for grading, generation, verbose output, and citation validation.

Do not collapse these concepts unless you are intentionally redesigning the evaluation contract.

## Coding Expectations

- Match the existing style: typed functions, small helpers, `Path`-based filesystem handling, and concise module docstrings.
- **Docstrings**: Provide concise, bulleted docstrings for classes, types, and key methods handling complex logic (especially in `retrieval/`, `runtime/`, and `data/` modules). Avoid overly wordy explanations; optimize for skimmability.
- Prefer mature, widely used libraries over custom utility code when they solve the problem well and make the code more readable. Do not reimplement common library functionality unless the dependency is a poor fit for the project or the custom behavior is deliberately required.
- Use explicit typings. Do not use `Any` unless the change explicitly requires it and there is no narrower correct type.
- Do not use `getattr`; prefer explicit attributes, typed protocols, or well-scoped conditionals.
- Prefer straightforward Python over framework-heavy abstractions. This repo leans on deterministic code paths outside the actual provider/model calls.
- Keep provider-specific branching inside `support_graph/providers.py` and config/runtime assembly instead of scattering it across feature code.
- Preserve the modular-monolith layout. Put code by concern, not by temporary feature.
- Favor additive changes over broad refactors. The tests exercise exact fields, file names, and CLI output structure.
- If you change CLI behavior, eval artifacts, or walkthrough scripts, update the matching docs and tests in the same change.
- When defining or changing artifact contracts, API payloads, manifests, indexes, or report formats, specify concrete schemas instead of loose prose.
- For spec documents, list required keys, optional keys, types, nullability, exact filenames and paths, and the producer/consumer relationship for each artifact.
- When a string field has a bounded value set, define the allowed values explicitly as part of the contract instead of leaving the field typed only as `string`.
- When a filesystem artifact uses a logical identifier, distinguish the canonical logical id from any path-safe storage key instead of assuming raw ids should become filenames.
- When a storage key is not intended to be derived by consumers, say so explicitly in the contract and require consumers to resolve it from the owning manifest or index.
- Treat `ARTIFACT_LAYOUT.md` as the canonical generated-output contract. Do not invent parallel artifact layouts in implementation or docs.
- If something should be done later rather than in the current change, record it in `TODO.md` with a priority instead of leaving it only in discussion or review notes.

## Testing Guidance

- Run `uv run pytest` for the default deterministic suite.
- Use `uv run pytest --run-integration` only when local Postgres and Ollama are intentionally available.
- When changing chunking, example building, or subset logic, update the corresponding tests under `tests/data/`.
- When changing retrieval or graph behavior, check `tests/retrieval/` and `tests/runtime/`.
- When changing metrics, artifact writing, or experiments, check `tests/evaluation/`.
- When changing user-facing terminal output, check `tests/app/`.

## Practical Notes for Agents

- The raw dataset is already committed locally. Do not add network-dependent dataset bootstrap steps.
- Prefer `uv run ...` over ad hoc environment activation.
- Use the artifact namespaces from `ARTIFACT_LAYOUT.md` instead of inventing new output locations.
- Live execution still depends on the existing derived examples artifacts for example lookup; the Workbench artifact namespaces do not replace `data/derived/examples/`.
- Use `TODO.md` for deferred follow-up work. Add concise items with a priority label when something is intentionally postponed.
- The walkthrough scripts under `examples/` are part of the onboarding story; keep them consistent with the code.
- There is no repo-local formatter/linter config yet. Follow the existing formatting and keep changes surgical.
