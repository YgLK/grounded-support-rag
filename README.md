# SupportGraph

SupportGraph is a CLI-first modular monolith for grounded support assistance over MultiDoc2Dial. The current implementation covers Phases 1 through 4 from the planning docs: local config/bootstrap, raw dataset loading, section-aware chunk building, agent-turn example building, deterministic DMV eval subsets, pgvector indexing, retrieval-backed `run`, and the offline eval harness.

## MVP Scope

- Architecture: modular monolith
- Required domain: `dmv`
- Config/provider: LangChain provider abstraction with `ollama`, `openai`, and `anthropic` chat support; embeddings support `ollama` and `openai`
- Local traces: JSON or JSONL under `outputs/traces/`
- Hosted observability: optional LangSmith and OpenTelemetry alongside the local trace files
- Headline eval split: `validation`

## Bootstrap

```bash
uv sync --dev
cp .env.example .env
```

The data-prep commands do not require Postgres or a model endpoint. `index-docs`, `run`, and `eval` require Postgres plus whichever provider config you choose below. The default documented path is still local Postgres + Ollama.

## Walkthrough Examples

For a gradual, code-first onboarding path, start with the numbered scripts under [examples/README.md](/Users/yglk/coding/support-graph/examples/README.md). The series goes from raw MultiDoc2Dial EDA through chunking, retrieval, the runtime graph, eval artifacts, and trace review.

## Commands

```bash
uv run support-graph build-chunks --domain dmv
uv run support-graph build-examples --domain dmv --split validation
uv run support-graph build-subsets --domain dmv --split validation
uv run support-graph index-docs --domain dmv
uv run support-graph run --example-id 'dmv::1409501a35697e0ce68561e29577b90a::turn_2'
uv run support-graph eval --split validation --domain dmv
uv run support-graph ablate-smoke10 --domain dmv --limit 10
uv run support-graph review-failures --run-id 20260318-143000-dmv-smoke
uv run support-graph trace-show --run-id 20260318-143000-dmv-smoke --example-id 'dmv::1409501a35697e0ce68561e29577b90a::turn_2'
```

Artifacts are written to:

- `data/derived/chunks/`
- `data/derived/examples/`
- `data/eval_subsets/`
- `logs/`
- `outputs/traces/`
- `outputs/evals/`

Eval run directories under `outputs/evals/<run_id>/` currently include:

- `manifest.json`
- `metrics.json`
- `predictions.jsonl`
- `failures.jsonl`
- `manual_review.csv`
- `retrieval_examples.jsonl`
- `trace_index.json`
- `summary.md`

## Current Package Layout

The code is organized by concern:

- `support_graph/`: package-level wiring, shared types, provider helpers, and subsystem guides in [support_graph/README.md](support_graph/README.md)
- `support_graph/data/`: raw dataset loading, chunk building, example building, eval subsets in [support_graph/data/README.md](support_graph/data/README.md)
- `support_graph/retrieval/`: pgvector indexing, query construction, retrieval, reranking in [support_graph/retrieval/README.md](support_graph/retrieval/README.md)
- `support_graph/runtime/`: the LangGraph runtime and local trace writing in [support_graph/runtime/README.md](support_graph/runtime/README.md)
- `support_graph/evaluation/`: offline metrics, eval runs, Smoke-10 ablations, embedding benchmark in [support_graph/evaluation/README.md](support_graph/evaluation/README.md)
- `support_graph/config/`: environment loading and runtime config assembly in [support_graph/config/README.md](support_graph/config/README.md)
- `support_graph/app/`: CLI entrypoints and terminal output formatting in [support_graph/app/README.md](support_graph/app/README.md)

The runtime graph itself lives in `support_graph/runtime/graph.py`.

## Sequential Flow

The end-to-end path is easier to read when split by mode.

### Data Prep

```mermaid
sequenceDiagram
    actor User
    participant CLI as support_graph.app.cli
    participant Settings as support_graph.config.settings
    participant Data as support_graph.data

    User->>CLI: build-chunks / build-examples / build-subsets
    CLI->>Settings: load .env and paths
    CLI->>Data: load dataset and build derived records
    Data-->>CLI: chunks, examples, subsets
    CLI-->>User: write JSONL artifacts under data/
```

### Indexing

```mermaid
sequenceDiagram
    actor User
    participant CLI as support_graph.app.cli
    participant Settings as support_graph.config.settings
    participant Retrieval as support_graph.retrieval
    participant PG as Postgres + pgvector

    User->>CLI: index-docs
    CLI->>Settings: validate indexing config
    CLI->>Retrieval: load chunk artifact and build embeddings
    Retrieval->>PG: upsert vectors into collection support_graph_domain
    PG-->>CLI: indexed row count
    CLI-->>User: indexing summary
```

### Single Run

```mermaid
sequenceDiagram
    actor User
    participant CLI as support_graph.app.cli
    participant Settings as support_graph.config.settings
    participant Runtime as support_graph.runtime.graph
    participant Retrieval as support_graph.retrieval
    participant PG as Postgres + pgvector
    participant Traces as outputs/traces

    User->>CLI: run --example-id ...
    CLI->>Settings: build RuntimeConfig
    CLI->>Runtime: run_graph_async(example, config)
    Runtime->>Retrieval: build query and retrieve ranked chunks
    Retrieval->>PG: similarity search with metadata filters
    PG-->>Runtime: candidate chunks
    Runtime->>Runtime: grade evidence, refine if needed, generate or abstain
    Runtime->>Traces: append node events to JSONL trace
    Runtime-->>CLI: final response payload
    CLI-->>User: concise terminal output
```

### Evaluation

```mermaid
sequenceDiagram
    actor User
    participant CLI as support_graph.app.cli
    participant Settings as support_graph.config.settings
    participant Eval as support_graph.evaluation.evaluate
    participant Runtime as support_graph.runtime.graph
    participant Traces as outputs/traces
    participant Artifacts as outputs/evals

    User->>CLI: eval --subset ...
    CLI->>Settings: build RuntimeConfig
    CLI->>Eval: evaluate_split_async(...)
    Eval->>Runtime: run shared graph for each example
    Runtime->>Traces: write per-example traces
    Eval->>Artifacts: write metrics, predictions, failures, summary
    Eval-->>CLI: eval result summary
    CLI-->>User: headline metrics + artifact paths
```

## System Architecture

The diagram below shows the main runtime pieces and the durable artifacts they exchange.

```mermaid
flowchart LR
    User["Operator / CLI user"] --> CLI["CLI layer<br/>support_graph.app.cli"]

    subgraph Config["Configuration"]
        Settings["Settings<br/>support_graph.config.settings"]
        RuntimeConfig["RuntimeConfig<br/>support_graph.config.runtime"]
    end

    subgraph DataPrep["Deterministic Data Prep"]
        Dataset["dataset.py<br/>raw MultiDoc2Dial loaders"]
        Chunks["chunks.py<br/>section-aware chunk builder"]
        Examples["examples.py<br/>turn-level examples"]
        Subsets["eval_subsets.py<br/>deterministic slices"]
    end

    subgraph Storage["Artifacts And Services"]
        Raw["multidoc2dial/"]
        Derived["data/derived/*.jsonl"]
        EvalSubsets["data/eval_subsets/*.jsonl"]
        PG["Postgres + pgvector"]
        TraceFiles["outputs/traces/*.jsonl"]
        EvalFiles["outputs/evals/run_id/"]
    end

    subgraph Retrieval["Retrieval"]
        Index["retrieval/index.py"]
        Retrieve["retrieval/retrieve.py"]
    end

    subgraph Runtime["Shared Runtime"]
        Graph["runtime/graph.py"]
        Nodes["runtime/nodes.py"]
        Prompts["runtime/prompts.py"]
        Policy["runtime/llm_policy.py"]
        Traces["runtime/traces.py"]
        Observability["runtime/observability.py"]
    end

    subgraph Eval["Offline Evaluation"]
        Evaluate["evaluation/evaluate.py"]
        Ablation["evaluation/ablation.py"]
        Benchmark["evaluation/benchmark.py"]
    end

    subgraph Providers["Provider Layer"]
        ProviderHelpers["providers.py"]
        Chat["Chat model"]
        Embeddings["Embedding model"]
    end

    CLI --> Settings --> RuntimeConfig
    Raw --> Dataset
    Dataset --> Chunks --> Derived
    Dataset --> Examples --> Derived
    Examples --> Subsets --> EvalSubsets

    CLI --> Index
    RuntimeConfig --> Index
    Derived --> Index --> PG
    Index --> Embeddings
    ProviderHelpers --> Embeddings

    CLI --> Graph
    RuntimeConfig --> Graph
    Graph --> Nodes
    Nodes --> Retrieve --> PG
    Nodes --> Prompts
    Nodes --> Policy
    Nodes --> ProviderHelpers --> Chat
    Graph --> Traces --> TraceFiles
    Graph --> Observability

    CLI --> Evaluate
    RuntimeConfig --> Evaluate
    Derived --> Evaluate
    EvalSubsets --> Evaluate
    Evaluate --> Graph
    Evaluate --> EvalFiles
    Ablation --> Evaluate
    Benchmark --> Derived
    Benchmark --> ProviderHelpers
```

## End-to-End Pipeline

```mermaid
flowchart TD
    A["MultiDoc2Dial raw files<br/>multidoc2dial_doc.json<br/>multidoc2dial_dial_*.json"] --> B["support_graph.data.dataset<br/>load_documents / load_dialogues"]
    B --> C["support_graph.data.chunks<br/>build_chunks"]
    B --> D["support_graph.data.examples<br/>build_turn_examples"]
    D --> E["support_graph.data.eval_subsets<br/>build_subset"]
    C --> F["data/derived/chunks/dmv.jsonl"]
    D --> G["data/derived/examples/dmv_validation.jsonl"]
    E --> H["data/eval_subsets/smoke.jsonl<br/>data/eval_subsets/frozen_ablation.jsonl"]
    F --> I["support_graph.retrieval.index<br/>index_documents"]
    I --> J["pgvector collection<br/>support_graph_dmv"]
    G --> K["support_graph.runtime.graph<br/>run_graph"]
    J --> K
    H --> L["support_graph.evaluation.evaluate<br/>evaluate_examples"]
    G --> L
    J --> L
    K --> M["outputs/traces/run-*.jsonl"]
    L --> N["outputs/evals/<run_id>/manifest.json"]
    L --> O["outputs/evals/<run_id>/metrics.json"]
    L --> P["outputs/evals/<run_id>/predictions.jsonl"]
    L --> Q["outputs/evals/<run_id>/failures.jsonl"]
    L --> R["outputs/evals/<run_id>/summary.md"]
```

## Runtime Graph

`run` and `eval` both execute the same graph from `support_graph/runtime/graph.py`.

```mermaid
flowchart TD
    START["START"] --> PQ["prepare_query"]
    PQ --> RD["retrieve_docs"]
    RD --> GE["grade_evidence"]
    GE -->|sufficient| GR["generate_response"]
    GE -->|partial and attempts < max_attempts| RQ["refine_query"]
    GE -->|insufficient or retries exhausted| RNA["resolve_without_answer"]
    RQ --> RD
    GR --> F["finalize"]
    RNA --> F
    F --> END["END"]
```

What each node does today:

- `prepare_query`: builds a compact deterministic query with `domain`, `latest_user_need`, optional last agent question, and up to two carry-forward turns
- `retrieve_docs`: fetches `candidate_k` chunks from pgvector, normalizes them, deterministically reranks them, keeps the top `5`, then expands evidence with same-doc neighbors when enabled
- `grade_evidence`: uses the chat model to classify evidence as `sufficient`, `partial`, or `insufficient`
- `refine_query`: appends either `Missing condition: ...` or `Focus sections: ...` for one retry
- `generate_response`: uses the chat model to produce an `answer`, `clarify`, or `abstain` payload with citation chunk IDs
- `resolve_without_answer`: uses the chat model only for `clarify` or `abstain` when direct answer generation is not allowed
- `finalize`: normalizes citations, enforces decision consistency against the evidence grade, and returns the final runtime payload

## Models And What They Do

There are two model surfaces, and they are used for different jobs.

```mermaid
flowchart LR
    A["Embedding model<br/>SUPPORT_GRAPH_EMBEDDING_MODEL"] --> B["index-docs"]
    B --> C["pgvector embeddings for chunk text"]
    C --> D["similarity_search_with_score"]
    E["Chat model<br/>SUPPORT_GRAPH_CHAT_MODEL"] --> F["grade_evidence"]
    E --> G["generate_response"]
    E --> H["resolve_without_answer"]
    I["No model call"] --> J["build_chunks / build_examples / build_subsets"]
    I --> K["query builder / reranker / neighbor expansion"]
    I --> L["eval metrics / failure labels / ablation guardrails"]
```

- The embedding model is used to index retrieval chunks and to embed retrieval queries for pgvector search.
- The chat model is used only in the graph runtime, not in chunking, example building, reranking, or metric computation.
- Query building, reranking, neighbor expansion, citation validation, and metrics are deterministic Python code.

## Inputs And Outputs

### Runtime Input

The runtime consumes one example record, either loaded from `data/derived/examples/*.jsonl` or passed directly:

```json
{
  "example_id": "dmv::1409501a35697e0ce68561e29577b90a::turn_2",
  "domain": "dmv",
  "turns_before_target": [
    {
      "turn_id": 1,
      "role": "user",
      "utterance": "My insurance ended so what should i do"
    }
  ],
  "latest_user_turn_id": 1,
  "latest_user_utterance": "My insurance ended so what should i do",
  "target_turn": {
    "turn_id": 2,
    "role": "agent",
    "da": "respond_solution"
  },
  "target_mode": "answer",
  "gold_doc_ids": ["Top 5 DMV Mistakes and How to Avoid Them#3_0"],
  "gold_span_ids": ["24", "25", "26"]
}
```

### Runtime Output

`run_graph(...)` returns a normalized payload shaped like:

```json
{
  "example_id": "dmv::1409501a35697e0ce68561e29577b90a::turn_2",
  "decision": "answer",
  "response_text": "Restore coverage immediately to avoid a registration suspension.",
  "citations": [
    {
      "doc_id": "doc",
      "chunk_id": "dmv::doc::sec::2::sub::0",
      "span_ids": ["2", "3"]
    }
  ],
  "confidence_label": "high",
  "latest_user_utterance": "My insurance ended so what should i do",
  "retrieval_ranked_chunks": [],
  "retrieved_chunks": [],
  "evidence_grade": {
    "verdict": "sufficient",
    "reason": "direct support",
    "missing_information": []
  },
  "trace_summary": {
    "retrieval_attempts": 1,
    "final_query": "Domain: dmv\nLatest user need: ...",
    "graph_path": [
      "prepare_query",
      "retrieve_docs",
      "grade_evidence",
      "generate_response",
      "finalize"
    ],
    "latency_ms": 1234.56
  }
}
```

Two chunk lists matter:

- `retrieval_ranked_chunks`: the direct top-5 retrieval list used for `Doc Recall@3`, `Span Recall@5`, and `MRR@5`
- `retrieved_chunks`: the expanded evidence set used for grading, generation, verbose output, and citation validation

### Eval Output

`eval` writes one run directory under `outputs/evals/<run_id>/`:

- `manifest.json`: config and run metadata
- `metrics.json`: aggregate retrieval, generation, decision, and latency metrics
- `predictions.jsonl`: one record per evaluated example
- `failures.jsonl`: the subset with non-null failure labels
- `manual_review.csv`: review sheet for follow-up predictions and answer failures
- `retrieval_examples.jsonl`: retrieval-centric per-example records
- `trace_index.json`: per-example trace summary pointing at raw trace JSONL files
- `summary.md`: short human-readable run summary

## What The Tests Actually Cover

The test suite is mostly deterministic unit and slice tests. It is not a full live Postgres + Ollama integration suite.

By default, `uv run pytest` skips any test marked `integration` and prints coverage for `support_graph`. Use `uv run pytest --run-integration` only when local Postgres + Ollama are intentionally available.

```mermaid
flowchart LR
    A["tests/data/*"] --> B["Real bundled dataset files"]
    B --> C["loaders, chunking, example building, committed subsets"]
    D["tests/retrieval/*"] --> E["Deterministic query builder, metadata filter, hit normalization, reranker, index adapter"]
    F["tests/runtime/*"] --> G["Graph control flow, retry loop, neighbor expansion, ranked vs expanded evidence"]
    H["tests/evaluation/*"] --> I["Metric math, artifact writing, ablation guardrails, benchmark helpers"]
    J["tests/app/*"] --> K["CLI output hierarchy and missing-config handling"]
```

More concretely:

- `tests/data/*` reads the committed `multidoc2dial/` files and checks document counts, dialogue counts, chunk determinism, example shape, and that `smoke` and `frozen_ablation` subsets match the committed JSONL files.
- `tests/retrieval/*` checks the compact history-aware query builder, candidate-pool retrieval, deterministic reranking, and index conversion logic.
- `tests/runtime/*` monkeypatches graph nodes to verify the path through the graph, the one-retry loop, and same-doc neighbor expansion behavior.
- `tests/evaluation/*` uses fake predictions to verify metric computation, that retrieval metrics use `retrieval_ranked_chunks`, and that citation validation uses `retrieved_chunks`.
- `tests/app/*` checks terminal UX: missing config, missing index, output hierarchy for `run`, `eval`, and `ablate-smoke10`, plus the new `review-failures` and `trace-show` analysis commands.

If you want a true end-to-end run against local services, use the CLI commands rather than relying on the unit tests:

```bash
uv run support-graph build-chunks --domain dmv
uv run support-graph build-examples --domain dmv --split validation
uv run support-graph index-docs --domain dmv
uv run support-graph run --example-id 'dmv::1409501a35697e0ce68561e29577b90a::turn_2' --verbose
uv run support-graph eval --domain dmv --subset smoke
```

## Phase 2 Runtime Prerequisites

`index-docs` now uses Postgres + `pgvector` plus LangChain-managed embeddings. The default documented path is still local-first with Ollama:

```bash
docker compose up -d postgres
ollama pull qwen3:8b-q4_K_M
ollama pull qwen3-embedding:4b-q4_K_M
```

Then set `.env` values for:

- `SUPPORT_GRAPH_POSTGRES_DSN`
- `SUPPORT_GRAPH_PROVIDER_TYPE`
- `SUPPORT_GRAPH_EMBEDDING_PROVIDER_TYPE` when chat and embedding providers differ
- `SUPPORT_GRAPH_OLLAMA_BASE_URL`
- `SUPPORT_GRAPH_OPENAI_API_KEY` for OpenAI-backed chat or embeddings
- `SUPPORT_GRAPH_ANTHROPIC_API_KEY` for Anthropic-backed chat
- `SUPPORT_GRAPH_EMBEDDING_MODEL`
- `SUPPORT_GRAPH_CHAT_MODEL`
- `SUPPORT_GRAPH_RETRIEVAL_CANDIDATE_K`

The included `compose.yml` starts a local `pgvector/pgvector:pg16` Postgres with:

- database: `support_graph`
- user: `postgres`
- password: `postgres`
- port: `5432`

Recommended DSN:

```dotenv
SUPPORT_GRAPH_POSTGRES_DSN=postgresql+psycopg://postgres:postgres@localhost:5432/support_graph
```

Recommended local Ollama config:

```dotenv
# Provider
SUPPORT_GRAPH_PROVIDER_TYPE=ollama
SUPPORT_GRAPH_OLLAMA_BASE_URL=http://localhost:11434

# Models
SUPPORT_GRAPH_CHAT_MODEL=qwen3:8b-q4_K_M
SUPPORT_GRAPH_EMBEDDING_MODEL=qwen3-embedding:4b-q4_K_M

# Retrieval
SUPPORT_GRAPH_RETRIEVAL_TOP_K=5
SUPPORT_GRAPH_RETRIEVAL_CANDIDATE_K=12
SUPPORT_GRAPH_MAX_RETRIEVAL_ATTEMPTS=2
```

OpenAI chat + embeddings:

```dotenv
SUPPORT_GRAPH_PROVIDER_TYPE=openai
SUPPORT_GRAPH_OPENAI_API_KEY=sk-...
SUPPORT_GRAPH_CHAT_MODEL=gpt-4o-mini
SUPPORT_GRAPH_EMBEDDING_MODEL=text-embedding-3-small
```

Anthropic chat + OpenAI embeddings:

```dotenv
SUPPORT_GRAPH_PROVIDER_TYPE=anthropic
SUPPORT_GRAPH_ANTHROPIC_API_KEY=sk-ant-...
SUPPORT_GRAPH_CHAT_MODEL=claude-3-5-haiku-latest
SUPPORT_GRAPH_EMBEDDING_PROVIDER_TYPE=openai
SUPPORT_GRAPH_OPENAI_API_KEY=sk-...
SUPPORT_GRAPH_EMBEDDING_MODEL=text-embedding-3-small
```

Use exact Ollama model tags from `ollama list`. If you change `SUPPORT_GRAPH_EMBEDDING_MODEL`, re-run `index-docs` because the pgvector index depends on the embedding space. Changing only `SUPPORT_GRAPH_CHAT_MODEL` does not require re-indexing.

## Observability

Local JSONL traces remain the default and still power `trace-show`, eval artifacts, and failure review. Two hosted backends are now optional:

- LangSmith: set `SUPPORT_GRAPH_LANGSMITH_TRACING_ENABLED=true` plus `SUPPORT_GRAPH_LANGSMITH_API_KEY`, and optionally `SUPPORT_GRAPH_LANGSMITH_PROJECT`
- OpenTelemetry: set `SUPPORT_GRAPH_OTEL_ENABLED=true`; use `SUPPORT_GRAPH_OTEL_EXPORTER=otlp` with `SUPPORT_GRAPH_OTEL_ENDPOINT` for a collector, or omit the endpoint for local console spans

When enabled, the runtime keeps writing local traces and also emits standard tracing context for each graph run and node.

CLI commands also write timestamped execution logs under `logs/` by default. Override the location with `SUPPORT_GRAPH_LOG_DIR`, and adjust verbosity with `SUPPORT_GRAPH_LOG_LEVEL`.

## Eval Notes

- `eval` defaults to the committed `smoke` subset so the default command stays practical on a local machine.
- Use `--subset frozen_ablation` for a fairer ablation pass.
- Use `--subset full_validation` once the DMV benchmark is stable and you want the full validation run.
- `ablate-smoke10` runs the control plus three targeted variants on the first 10 committed smoke examples and writes a cross-run markdown note under `outputs/evals/`.
- `review-failures` inspects one eval run’s failure list and review artifacts from the terminal.
- `trace-show` resolves an evaluated example through `trace_index.json` and prints the raw graph trace summary.

## Next Phases

- Improve retrieval quality and query refinement on DMV.
- Add richer failure analysis and optional trace indexing for eval runs.
- Expand beyond `dmv` only after the DMV benchmark is stable.
