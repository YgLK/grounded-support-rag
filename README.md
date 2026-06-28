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
- `Doc Recall@1 0.700`, `Doc Recall@3 1.000`, `Span Recall@5 1.000`, `MRR@5 0.800`
- `ROUGE-L 0.178`, `F1 0.237`, `Citation Coverage 1.000`
- Failure bucket: `incomplete_answer` only; retrieval misses cleared

The Kubernetes baseline is intentionally artifact-heavy: each eval writes traces, prediction records, retrieval examples, manual-review CSVs, and a summary that separates retrieval, grounding, citation, and answer-completeness failures.

## DMV Model Comparison

Recent smoke runs on the same `dmv validation / smoke` subset (`25` examples):

<!-- 
| Chat | Embeddings | Doc R@3 | Span R@5 | MRR@5 | F1 | Citation Cov. | E2E | Avg Latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `openrouter / openai/gpt-oss-120b:nitro` | `ollama / qwen3-embedding:4b-q4_K_M` | 0.760 | 0.307 | 0.573 | 0.211 | 0.240 | 0.160 | 2588 ms |
| `ollama / qwen3:8b-q4_K_M` | `ollama / qwen3-embedding:4b-q4_K_M` | 0.680 | 0.320 | 0.565 | 0.188 | 0.100 | 0.160 | 27430 ms |
| `openrouter / openai/gpt-oss-120b:nitro` | `openrouter / qwen/qwen3-embedding-8b` | 0.160 | 0.120 | 0.110 | 0.099 | 0.000 | 0.040 | 7319 ms |
 -->
 
| Chat | Embeddings | Doc R@3 | Span R@5 | MRR@5 | F1 | Citation Cov. | Avg Latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `openrouter / openai/gpt-oss-120b:nitro` | `ollama / qwen3-embedding:4b-q4_K_M` | 0.760 | 0.307 | 0.573 | 0.211 | 0.240 | 2588 ms |
| `ollama / qwen3:8b-q4_K_M` | `ollama / qwen3-embedding:4b-q4_K_M` | 0.680 | 0.320 | 0.565 | 0.188 | 0.100 | 27430 ms |
| `openrouter / openai/gpt-oss-120b:nitro` | `openrouter / qwen/qwen3-embedding-8b` | 0.160 | 0.120 | 0.110 | 0.099 | 0.000 | 7319 ms |

DMV conclusions:

- Best practical setup so far is `openrouter / openai/gpt-oss-120b:nitro` + `ollama / qwen3-embedding:4b-q4_K_M`
- Swapping embeddings from `qwen3:8b-q4_K_M` to `qwen/qwen3-embedding-8b` caused the main retrieval collapse
- OpenRouter chat improved latency dramatically over the fully local Ollama path while also improving citation quality


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

DMV / MultiDoc2Dial corpus:

```bash
uv run grounded-support-rag build-chunks --domain dmv
uv run grounded-support-rag build-examples --domain dmv --split validation
uv run grounded-support-rag build-subsets --domain dmv --split validation
uv run grounded-support-rag index-docs --domain dmv
uv run grounded-support-rag run --example-id 'dmv::1409501a35697e0ce68561e29577b90a::turn_2'
uv run grounded-support-rag eval --split validation --domain dmv
```
