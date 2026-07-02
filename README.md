# SupportGraph

SupportGraph is a local-first, evaluation-first RAG assistant for grounded Kubernetes troubleshooting. It turns a pinned `kubernetes/website` docs snapshot into indexed evidence, runs a LangGraph pipeline that retrieves and grades relevant context for each user turn, and returns grounded answers with citations, traces, and reproducible offline evaluation artifacts for debugging and model comparison. The CLI is `grounded-support-rag`.

The system is a production-shaped modular monolith, not a production-ready SaaS: it is built to demonstrate a defensible LLM systems story (retrieval, grounding, evaluation, observability) without the auth, tenancy, retention, and SLO machinery real production traffic would require. Deferred production gaps are listed in `ROADMAP.md` (P4).

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

## Why This Stack

- **LangGraph** models retrieval as a stateful workflow: route the turn, prepare a query, retrieve, grade evidence, retry with query refinement, and fall back to `clarify` / `abstain` when evidence is insufficient. One graph makes routing, grading, retries, and fallback decisions inspectable in traces instead of hiding them in ad hoc prompt chains.
- **Postgres + pgvector** keeps retrieval grounded in a pinned Kubernetes docs corpus. Answers cite retrieved chunks only, so quality is bounded by what the index returns rather than parametric memory, and the index is reproducible from a fetched docs ref.
- **Offline evaluation** catches regressions before claims: retrieval metrics (Doc Recall@1/3/5, Span Recall@5, MRR@5), generation metrics (ROUGE-L, F1), grounding/citation metrics (Citation Coverage, required-points coverage), and a manual failure-review CSV separate retrieval, grounding, citation, and answer-completeness failures.
- **What the current failure teaches:** the one remaining `weak_citations` case is a PVC Pending answer that is well-grounded and cites acceptable storage docs (`storage-classes`, `csi-storage-capacity`, `persistent-volume-claim`), but legacy citation coverage still expects the gold `concepts/storage/dynamic-provisioning` doc/span that was not retrieved. It shows that strict gold-span citation metrics can undercount acceptable grounded answers, and motivates a citation-repair follow-up rather than a retrieval change.

## Evaluation Snapshot

Current flagship run: Kubernetes Smoke-10 troubleshooting baseline (`10` examples):

- Chat: `openrouter / openai/gpt-oss-120b:nitro`
- Embeddings: `openrouter / openai/text-embedding-3-small`
- Latest run: `20260629-002315-kubernetes-smoke`
- `Doc Recall@1 0.700`, `Doc Recall@3 0.900`, `Span Recall@5 1.000`, `MRR@5 0.800`
- `ROUGE-L 0.161`, `F1 0.204`, `Citation Coverage 0.900`
- Failure bucket: `weak_citations` only (`1` PVC example); `incomplete_answer` cleared

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

Fast path with the committed DB dump (skips corpus fetch + indexing):

```bash
docker compose up -d postgres
cat support_graph.pg.dump | docker compose exec -T postgres pg_restore -U postgres -d support_graph --clean --if-exists
uv run grounded-support-rag --config-file support_graph.kubernetes.toml doctor --domain kubernetes
uv run grounded-support-rag --config-file support_graph.kubernetes.toml eval --domain kubernetes --subset smoke
uv run grounded-support-rag --config-file support_graph.kubernetes.toml ui --host 127.0.0.1 --port 8008
```

Full rebuild path (fetch + chunk + index from a fresh docs snapshot):

```bash
uv run grounded-support-rag fetch-kubernetes-docs --ref main --output raw/kubernetes/current
uv run grounded-support-rag build-chunks --domain kubernetes
uv run grounded-support-rag index-docs --domain kubernetes
uv run grounded-support-rag --config-file support_graph.kubernetes.toml eval --domain kubernetes --subset smoke
uv run grounded-support-rag --config-file support_graph.kubernetes.toml ui --host 127.0.0.1 --port 8008
```

Inspect a run from the command line:

```bash
uv run grounded-support-rag --config-file support_graph.kubernetes.toml review-failures --run-id <run-id>
uv run grounded-support-rag --config-file support_graph.kubernetes.toml trace-show --run-id <run-id> --example-id <example-id>
```

For command dispatch and handler responsibilities, see `docs/CLI_HANDLERS.md`.
