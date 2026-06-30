# SupportGraph Roadmap

Goal: make SupportGraph interview-ready as an LLM systems project.

Shape: execution checklist. No dates. Each phase should leave a demoable artifact or a defensible engineering story.

## Current State

- Kubernetes-first grounded support assistant.
- CLI pipeline covers corpus fetch, chunk build, indexing, single run, eval, experiment comparison, failure review, trace display, and UI launch.
- Retrieval stack: Postgres + pgvector, with provider-swappable embeddings.
- Runtime stack: LangGraph workflow with routing, query prep, retrieval, evidence grading, bounded refinement, answer/clarify/abstain, citations, traces.
- Evaluation stack: retrieval, generation, grounding, citation, failure labels, manual review CSV, trace index, summary artifacts.
- Workbench: local FastAPI/Jinja/HTMX UI over runs, evals, reports, failures, and traces.
- Headline baseline: Kubernetes Smoke-10, `20260629-002315-kubernetes-smoke`.
- Current metrics: Doc Recall@1 `0.700`, Doc Recall@3 `0.900`, Span Recall@5 `1.000`, MRR@5 `0.800`, ROUGE-L `0.161`, F1 `0.204`, Citation Coverage `0.900`.
- Current known failure bucket: `weak_citations` only, `1` PVC example.

## P0: Interview Narrative Lock

Goal: make the project easy to explain in 2 minutes and deep enough for a 45-minute technical drilldown.

Checklist:

- Align README, ROADMAP, and wiki on one project story: grounded Kubernetes troubleshooting RAG, not generic chatbot.
- Keep the architecture claim precise: production-shaped modular monolith, not production-ready SaaS.
- Add or preserve a clear system diagram in README showing LangGraph node flow.
- Keep latest eval snapshot in README accurate after meaningful eval reruns.
- Keep one concise "why LangGraph" explanation: stateful retrieval workflow with routing, grading, retries, fallback decisions.
- Keep one concise "why evaluation matters" explanation: string metrics, retrieval metrics, citation validity, manual failure review.
- Keep one concise "what failed" explanation: weak citation case and what it teaches.

Acceptance:

- A reviewer can identify the problem, architecture, eval method, and current result from README alone.
- ROADMAP explains what remains without implying missing core functionality.
- Wiki contains meaningful Kubernetes baseline findings after significant eval changes.

Interview signal:

- Clear product framing.
- Honest system boundaries.
- Ability to explain tradeoffs without overselling.

## P1: Reproducible Local Demo

Goal: a fresh local checkout can run the same demo path without guesswork.

Checklist:

- Verify setup path:

  ```bash
  uv sync --dev
  cp .env.example .env
  cp support_graph.toml.example support_graph.toml
  docker compose up -d postgres
  ```

- Verify database bootstrap path from the dump when available:

  ```bash
  cat support_graph.pg.dump | docker compose exec -T postgres pg_restore -U postgres -d support_graph --clean --if-exists
  ```

- Verify corpus path:

  ```bash
  uv run grounded-support-rag fetch-kubernetes-docs --ref main --output raw/kubernetes/current --replace
  uv run grounded-support-rag build-chunks --domain kubernetes
  uv run grounded-support-rag index-docs --domain kubernetes
  ```

- Keep setup docs explicit about required local config:
  - `.env`: `SUPPORT_GRAPH_POSTGRES_DSN`
  - `.env`: `SUPPORT_GRAPH_OPENROUTER_API_KEY` if using OpenRouter
  - `support_graph.toml`: chat and embedding provider/model settings

- Add a smoke command block that uses committed config defaults where possible:

  ```bash
  uv run grounded-support-rag --config-file support_graph.kubernetes.toml eval --domain kubernetes --subset smoke
  uv run grounded-support-rag --config-file support_graph.kubernetes.toml ui --host 127.0.0.1 --port 8008
  ```

Acceptance:

- `docker compose ps postgres` shows Postgres healthy.
- `pg_stat_user_tables` shows `langchain_pg_collection` and `langchain_pg_embedding` after dump restore or indexing.
- Smoke eval writes manifest, metrics, predictions, failures, retrieval examples, manual review CSV, trace index, summary, and traces.
- UI starts and can load the latest eval run.

Interview signal:

- Reproducible engineering.
- Local-first operations.
- Concrete artifact contract, not hidden notebook state.

## P2: RAG Quality + Eval Story

Goal: make quality claims defensible through artifacts, metrics, and failure analysis.

Checklist:

- Keep one canonical Kubernetes Smoke-10 run as the headline baseline.
- For each meaningful change to prompts, retrieval, citation handling, or graph routing, run smoke eval and compare:
  - Doc Recall@1/3/5
  - Span Recall@5
  - MRR@5
  - ROUGE-L
  - F1
  - Citation Coverage
  - citation validity
  - failure labels
- Use `review-failures` for every regression or surprising metric movement.
- Preserve at least one successful trace and one failed/weak trace for demo.
- Keep failure labels actionable: wrong doc, weak citations, unsupported answer, runtime error, incomplete answer if present.
- Avoid tuning only for ROUGE/F1; explain where lexical metrics undercount acceptable grounded answers.
- Keep citation behavior strict: only cite retrieved chunks.
- Keep abstain/clarify behavior visible in traces and eval artifacts.

Acceptance:

- Latest smoke run has all required eval artifacts.
- At least one failure review can be explained from artifacts without reading raw code.
- Metrics section tells a coherent story: retrieval strength, generation limits, grounding/citation quality.
- Any regression has a named suspected cause or a follow-up item.

Current findings:

- Canonical headline baseline: `20260629-002315-kubernetes-smoke` (Smoke-10, all eval artifacts present). Failure bucket: `weak_citations` 1 (PVC Pending); `incomplete_answer` 0.
- A same-config verification re-run (`20260629-131400-kubernetes-smoke`, no code/prompt/retrieval change) produced `incomplete_answer` 3. Retrieval metrics were identical across both runs (Doc Recall@1/3/5, Span Recall@5, MRR@5, Hit@5, Graded MRR@5 all matched), confirming retrieval is deterministic over the index. The `incomplete_answer` failures are generation omissions: the LLM omitted one required point (e.g., the Deployment answer omitted ReplicaSets) despite the evidence being retrieved. Named suspected cause: LLM generation non-determinism (`openai/gpt-oss-120b:nitro`).
- Named follow-up: stabilize answer completeness with a stricter generation prompt or a lightweight post-generation required-points checklist that forces cited docs' concrete checklist items into the final answer. Generation-side fix only; no reindex or retrieval change warranted.
- Lexical metrics move only slightly across runs (ROUGE-L 0.161 vs 0.149, F1 0.204 vs 0.203) while failure labels move more, which is why the project tracks required-points coverage and failure labels instead of tuning only for ROUGE/F1.
- Demo traces from the canonical headline run: success `kubernetes::pods::turn_2`; failure `kubernetes::troubleshooting-pvc-pending::turn_2` (`weak_citations`).
- Detailed finding in `wiki/supportgraph/kubernetes-baseline.md` (Smoke-10 Generation Variance section).

Eval expansion + variance attribution tooling (2026-06-30):

- New `EvalSubset.EXPANDED` tier between `SMOKE` and `FROZEN_EXPERIMENT`; loads from `data/eval_subsets/<domain>/expanded.jsonl`.
- New CLI: `draft-eval-examples` (corpus-grounded authoring with provenance), `validate-eval-examples` (doc/span ID existence, grounded alias groups, dup IDs, answer_type), `promote-eval-examples` (validator-gated merge into `expanded.jsonl`).
- New CLI: `eval-variance` (repeated-run K, retrieval-determinism assert, bootstrap CI, between-run std, n=10 vs n=50 sampling-noise comparison, recommended-K, explicit sampling-vs-generation attribution).
- New CLI: `model-ab-compatibility` (blocking gate for model A/B; asserts structured-output json_schema support via fallback-event count).
- Determinism-diagnostic config knobs: `chat_temperature`, `chat_seed`, `openrouter_provider_order`, `openrouter_allow_fallbacks` (diagnostic-only; defaults preserve today's behavior).
- 50-topic seed file committed at `data/eval_subsets/kubernetes/_candidates/expanded.seeds.jsonl`.
- 39 new tests; 188 passing. Detailed writeup in `wiki/supportgraph/eval-expansion-variance-tooling.md`.
- Live operational steps (require running Postgres + OpenRouter, not executed by the agent): draft candidates, human review/edit, validate + promote, run `eval-variance --subset expanded --repeat K`, optional determinism sweep, optional model-ab-compatibility gate.

Interview signal:

- Evaluation discipline.
- Grounding awareness.
- Debuggable RAG, not prompt-only iteration.

## P3: Workbench Demo Polish

Goal: make the local UI useful for explaining system behavior live.

Checklist:

- Verify homepage lists eval runs, reports, and standalone runs from artifact manifests.
- Verify eval detail page shows headline metrics, failure counts, provider/model snapshot, and artifact paths.
- Verify failure view supports scanning one weak citation case without opening JSONL.
- Verify trace view shows graph path, retrieval attempts, final query, evidence grade, final decision, citations, and raw events.
- Verify error pages are useful for missing or invalid artifacts.
- Keep UI as local operator workbench; do not reposition as customer support chat product.
- If adding UI polish, prioritize density, scanability, and artifact explainability over decorative layout.

Acceptance:

- Demo path:

  ```bash
  uv run grounded-support-rag --config-file support_graph.kubernetes.toml ui --host 127.0.0.1 --port 8008
  ```

- Reviewer can inspect:
  - latest eval run
  - one failure
  - one trace
  - one generated summary/report
- UI labels match CLI/artifact terminology.
- No UI-only source of truth is introduced.

Interview signal:

- Operational debugging surface.
- Full-stack enough to explain the system.
- Artifact-first architecture.

## P4: Production-Shaped Gaps

Goal: be explicit about what would be required before real production traffic.

Checklist:

- Document deferred requirements without implementing them prematurely:
  - auth and authorization
  - tenant isolation
  - conversation persistence
  - request idempotency
  - dependency timeout budgets
  - circuit breakers or degraded modes
  - retention and redaction policies
  - SLOs, dashboards, alerts, runbooks
  - rollout controls for model, prompt, graph, and retrieval changes
- Keep current project described as local-first and evaluation-first.
- Preserve modular monolith unless independent deploy cadence, scaling profile, or ownership boundaries justify a split.
- Keep config validation explicit and actionable.
- Keep provider failures visible; no silent fallbacks.

Acceptance:

- README or ROADMAP can answer "what is not production-ready yet?"
- Tradeoffs are framed as intentional MVP constraints, not omissions.
- No production-only complexity is added unless it improves the interview demo or local reliability.

Interview signal:

- Senior judgment.
- Scope control.
- Production awareness without cargo-cult architecture.

## P5: Stretch Differentiators

Goal: add depth only after demo reproducibility and eval story are solid.

Checklist:

- Add an ablation note comparing graph variants if it produces a clear lesson:
  - rerank on/off
  - content-only reasoning on/off
  - neighbor expansion on/off
  - one retrieval attempt vs two
- Add a small LLM-as-judge sample only if cost and reproducibility are acceptable.
- Add a focused citation repair experiment for the weak PVC case.
- Add a query/retrieval inspection note for technical Kubernetes terms and acronyms.
- Add one "model/provider tradeoff" note comparing local vs hosted latency, cost, and quality.
- Consider a tiny canned demo script only after live commands are reliable.

Acceptance:

- Each stretch item has a written finding, not just a code change.
- Findings connect to a design decision.
- Stretch work does not destabilize the baseline demo path.

Interview signal:

- Experiment design.
- Ability to isolate variables.
- Practical RAG improvement loop.

## Demo Path

Fast path with existing dump:

```bash
docker compose up -d postgres
cat support_graph.pg.dump | docker compose exec -T postgres pg_restore -U postgres -d support_graph --clean --if-exists
uv run grounded-support-rag --config-file support_graph.kubernetes.toml eval --domain kubernetes --subset smoke
uv run grounded-support-rag --config-file support_graph.kubernetes.toml ui --host 127.0.0.1 --port 8008
```

Full rebuild path:

```bash
uv sync --dev
cp .env.example .env
cp support_graph.toml.example support_graph.toml
docker compose up -d postgres
uv run grounded-support-rag fetch-kubernetes-docs --ref main --output raw/kubernetes/current --replace
uv run grounded-support-rag build-chunks --domain kubernetes
uv run grounded-support-rag index-docs --domain kubernetes
uv run grounded-support-rag --config-file support_graph.kubernetes.toml eval --domain kubernetes --subset smoke
uv run grounded-support-rag --config-file support_graph.kubernetes.toml ui --host 127.0.0.1 --port 8008
```

Inspection path:

```bash
uv run grounded-support-rag --config-file support_graph.kubernetes.toml review-failures --run-id <run-id>
uv run grounded-support-rag --config-file support_graph.kubernetes.toml trace-show --run-id <run-id> --example-id <example-id>
```

## Do Not Spend Time On Yet

- Do not build auth, accounts, or tenant isolation for the interview demo.
- Do not split services before the modular monolith has a real scaling or ownership problem.
- Do not expand to broad datasets before Kubernetes demo quality is stable.
- Do not add new dependencies unless they materially improve the demo or eval story.
- Do not chase production deployment polish before local reproducibility is boring.
- Do not optimize only for lexical metrics when grounding/citation failures are more important.

## Definition of Interview-Ready

- Fresh setup path is documented and recently verified.
- Database bootstrap path works from either dump restore or full indexing.
- Latest README eval snapshot matches the latest meaningful baseline.
- One smoke eval can be rerun and inspected.
- At least one success and one failure can be explained from trace artifacts.
- Workbench loads latest artifacts and supports the demo path.
- Failure analysis produces a concrete next step.
- Known production gaps are explicit and defensible.
- The project can be described as: local-first, grounded Kubernetes troubleshooting RAG with LangGraph orchestration, pgvector retrieval, strict citations, traces, and reproducible offline evaluation.
