# Grounded-support-RAG

Grounded-support-RAG is a retrieval-augmented support assistant for grounded Kubernetes troubleshooting and multi-document support knowledge bases. It turns source content into indexed evidence, runs a LangGraph pipeline that retrieves and grades relevant context for each user turn, and produces grounded answers with citations, traces, and offline evaluation artifacts for debugging and model comparison.

## Scope

- Deterministic data prep: raw dataset loading, section-aware chunking, turn-level example building, stable eval subsets
- Retrieval stack: Postgres + pgvector indexing with provider-swappable embeddings
- Runtime graph: retrieve evidence, decide `answer` / `clarify` / `abstain`, return cited responses, persist traces
- Evaluation: retrieval, generation, grounding, and end-to-end metrics with failure review artifacts
- Local workbench: FastAPI + Jinja2 + HTMX UI over runs, evals, reports, and traces

## Stack

- Runtime: `Python 3.11`, `LangGraph`, `LangChain`
- Retrieval: `Postgres`, `pgvector`
- Providers: `OpenRouter`, `Ollama`
- Workbench: `FastAPI`, `Jinja2`, `HTMX`

## LangGraph Runtime
Runtime behavior:

- Routes the user turn before retrieval
- Builds a search query from conversation context
- Retrieves and grades evidence before generation
- Retries retrieval with query refinement when evidence is partial
- Returns grounded `answer`, `clarify`, or `abstain` outputs with citations and trace artifacts

```mermaid
flowchart TD
    A([START]) --> B[route_query]
    B -->|document_query| C[prepare_query]
    B -->|chitchat / unsupported| G[resolve_without_answer]
    C --> D[retrieve_docs]
    D --> E[grade_evidence]
    E -->|sufficient| F[generate_response]
    E -->|partial and attempts remain| H[refine_query]
    H --> D
    E -->|insufficient or attempts exhausted| G
    F --> I[finalize]
    G --> I
    I --> J([END])
```


## Evaluation Snapshot

Current flagship run: Kubernetes Smoke-10 troubleshooting baseline (`10` examples):

- Chat: `openrouter / openai/gpt-oss-120b:nitro`
- Embeddings: `openrouter / openai/text-embedding-3-small`
- `Doc Recall@1 0.700`, `Doc Recall@3 1.000`, `Span Recall@5 1.000`, `MRR@5 0.833`
- `ROUGE-L 0.172`, `F1 0.225`, `Citation Coverage 0.900`
- Failure bucket: `incomplete_answer` only (`5` examples); retrieval misses cleared

The Kubernetes baseline is intentionally artifact-heavy: each eval writes traces, prediction records, retrieval examples, manual-review CSVs, and a summary that separates retrieval, grounding, citation, and answer-completeness failures.

## Run

Setup the local environment:

```bash
uv sync --dev
cp .env.example .env
cp support_graph.toml.example support_graph.toml
docker compose up -d postgres
```

Required local config:

- `.env`: set `SUPPORT_GRAPH_POSTGRES_DSN`
- `.env`: if using OpenRouter for chat or embeddings, set `SUPPORT_GRAPH_OPENROUTER_API_KEY`
- `support_graph.toml`: choose provider combination under `[runtime]`

Then run the pipeline:

```bash
uv run grounded-support-rag fetch-kubernetes-docs --ref main --output raw/kubernetes/current
uv run grounded-support-rag build-chunks --domain kubernetes
uv run grounded-support-rag index-docs --domain kubernetes
uv run grounded-support-rag --config-file support_graph.kubernetes.toml eval --domain kubernetes --subset smoke
uv run grounded-support-rag ui --host 127.0.0.1 --port 8008
```
