# Examples Walkthrough

The `examples/` directory is a guided tour of the current SupportGraph codebase. It starts with raw MultiDoc2Dial EDA, then moves through chunking, retrieval, runtime execution, eval artifacts, and trace review.

## Prerequisites

Base repo setup:

```bash
uv sync --dev
```

Local-only scripts:

- `00` through `06` use the committed dataset and derived artifacts only.
- They do not require Postgres or a model provider.

Full-pipeline scripts:

- `07` through `10` use the real indexing, runtime, eval, and analysis surfaces.
- They require a configured `.env`, local Postgres with `pgvector`, and an OpenRouter API key.

Recommended local setup:

```bash
cp .env.example .env
cp support_graph.toml.example support_graph.toml
docker compose up -d postgres
```

## Script Order

```bash
uv run python examples/00_project_overview.py
uv run python examples/01_dataset_eda.py
uv run python examples/02_document_anatomy.py
uv run python examples/03_dialogue_and_example_eda.py
uv run python examples/04_chunking_eda.py
uv run python examples/05_subset_eda.py
uv run python examples/06_query_and_retrieval_eda.py
uv run python examples/07_index_and_vectorstore_walkthrough.py
uv run python examples/08_run_graph_example.py
uv run python examples/09_eval_smoke_walkthrough.py
uv run python examples/10_trace_and_failure_review.py
```

What each script covers:

- `00_project_overview.py`: package layout, CLI surfaces, artifact directories, deterministic vs model-driven steps
- `01_dataset_eda.py`: raw dataset counts, domain distribution, split counts, sample document and dialogue IDs
- `02_document_anatomy.py`: one DMV document in detail, including spans, titles, parent titles, and ordering
- `03_dialogue_and_example_eda.py`: one raw dialogue and the derived agent-turn example record
- `04_chunking_eda.py`: section-aware DMV chunking plus deterministic oversized-section splitting
- `05_subset_eda.py`: how `smoke` and `frozen_ablation` are derived and checked against committed subset files
- `06_query_and_retrieval_eda.py`: query-context extraction, rendered query text, and deterministic reranking without a live vectorstore
- `07_index_and_vectorstore_walkthrough.py`: real pgvector indexing walkthrough when local services are configured
- `08_run_graph_example.py`: one real `run_graph_async(...)` example, including decision, citations, and trace output
- `09_eval_smoke_walkthrough.py`: a small real eval over the committed smoke subset, plus output artifact locations
- `10_trace_and_failure_review.py`: analysis artifacts and the `review-failures` / `trace-show` CLI commands

The scripts are meant to be read as well as executed. Each one points at the next step so a new contributor can move through the system in order.
