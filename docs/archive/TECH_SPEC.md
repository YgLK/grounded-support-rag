# Technical Specification

## Project

**Name:** SupportGraph  
**Dataset:** MultiDoc2Dial  
**Goal:** Build a conversation-aware, multi-document support assistant with grounded retrieval and reproducible evaluation

## 1. Scope of This Document

This document specifies:

- how raw MultiDoc2Dial files are structured
- how they should be transformed for retrieval and evaluation
- the MVP runtime contract
- how LangChain and LangGraph fit into the system
- how the project will be evaluated
- how experiments will be tracked and compared
- how the CLI and report surfaces should behave
- what the MVP repository layout should be

## 2. Source Dataset

Local dataset path:

`/Users/yglk/coding/grounded-support-rag/multidoc2dial`

Relevant files:

- `multidoc2dial_doc.json`
- `multidoc2dial_dial_train.json`
- `multidoc2dial_dial_validation.json`
- `multidoc2dial_dial_test.json`

Observed local corpus counts:

- documents:
  - `dmv`: 149
  - `ssa`: 109
  - `va`: 138
  - `studentaid`: 92
  - total: 488
- dialogues:
  - train: 3474
  - validation: 661
  - test: 661

Important notes:

- the bundled dataset README says the `test` file is a dummy file
- the local `multidoc2dial_dial_test.json` actually contains labeled `dial_data`
- MVP headline metrics must therefore use `validation`
- observed local agent turns include references in train, validation, and test, but loaders should still tolerate empty references defensively

## 3. Raw Data Structure

### 3.1 Document file structure

Top-level structure:

```json
{
  "doc_data": {
    "<domain>": {
      "<doc_id>": {
        "title": "...",
        "doc_id": "...",
        "domain": "...",
        "doc_text": "...",
        "spans": {
          "<id_sp>": {
            "id_sp": "4",
            "start_sp": 123,
            "end_sp": 180,
            "text_sp": "...",
            "id_sec": "...",
            "start_sec": 100,
            "end_sec": 260,
            "text_sec": "...",
            "title": "...",
            "parent_titles": ["..."]
          }
        },
        "doc_html_ts": "...",
        "doc_html_raw": "..."
      }
    }
  }
}
```

Important properties:

- each document belongs to one domain
- `doc_text` contains cleaned full text
- `spans` give gold segment boundaries and section metadata
- span annotations are valuable for retrieval evaluation

### 3.2 Dialogue file structure

Top-level structure:

```json
{
  "dial_data": {
    "<domain>": [
      {
        "dial_id": "...",
        "turns": [
          {
            "turn_id": 1,
            "role": "user",
            "da": "query_condition",
            "references": [
              {
                "label": "precondition",
                "id_sp": "4",
                "doc_id": "..."
              }
            ],
            "utterance": "..."
          }
        ]
      }
    ]
  }
}
```

Important properties:

- dialogues are grouped by domain
- turns include dialogue act and grounded references
- agent turns are not always immediately preceded by a user turn
- clarification-like agent turns appear as `query_condition`
- answer-like agent turns appear as `respond_solution`, `respond_solution_positive`, `respond_solution_negative`, and `respond_no_solution`

## 4. Derived Data Artifacts

The raw dataset is not the final runtime format. The project should generate two derived artifacts:

- retrieval chunks
- turn-level examples

### 4.1 Retrieval chunk schema

Use section/span-aware chunks rather than naive fixed-size slices where possible.

Suggested schema:

```json
{
  "chunk_id": "dmv::Registrations#3_0::sec::3::sub::0",
  "domain": "dmv",
  "doc_id": "Registrations#3_0",
  "doc_title": "Registrations#3",
  "section_id": "3",
  "section_title": "Renew",
  "parent_titles": ["Vehicles already registered in New York"],
  "subchunk_index": 0,
  "text": "...",
  "span_ids": ["10", "11", "12"],
  "token_count": 221
}
```

Recommended chunking strategy:

- group spans by `doc_id + id_sec`
- produce one chunk per section by default
- if a section is too large, split it into deterministic subchunks with inherited metadata
- chunk IDs must be deterministic so indexing and evaluation are reproducible

Why:

- aligns retrieval with the dataset's annotation boundaries
- makes citation and evaluation easier
- reduces arbitrary chunk boundaries

Observed local section-size distribution supports this plan:

- median section length is small
- most sections can remain single chunks
- only large outliers need subdivision

### 4.2 Turn-level example schema

Create one example per target agent turn.

Suggested schema:

```json
{
  "example_id": "dmv::8df07b7a98990db27c395cb1f68a962e::turn_2",
  "domain": "dmv",
  "dial_id": "8df07b7a98990db27c395cb1f68a962e",
  "target_turn_id": 2,
  "turns_before_target": [
    {
      "turn_id": 1,
      "role": "user",
      "da": "query_condition",
      "utterance": "Hello, I forgot to update my address, can you help me with that?"
    }
  ],
  "latest_user_turn_id": 1,
  "latest_user_utterance": "Hello, I forgot to update my address, can you help me with that?",
  "target_turn": {
    "turn_id": 2,
    "role": "agent",
    "da": "respond_solution",
    "utterance": "Hi, you have to report any change of address to DMV within 10 days after moving...",
    "references": [
      {
        "label": "solution",
        "id_sp": "4",
        "doc_id": "Top 5 DMV Mistakes and How to Avoid Them#3_0"
      }
    ]
  },
  "target_mode": "answer",
  "gold_doc_ids": [
    "Top 5 DMV Mistakes and How to Avoid Them#3_0"
  ],
  "gold_span_ids": [
    "4",
    "5"
  ]
}
```

Rules:

- keep only examples where the target turn role is `agent`
- preserve all turns before the target in order
- `latest_user_turn_id` and `latest_user_utterance` may point to an earlier turn, not necessarily `target_turn_id - 1`
- flatten all target-turn references into gold doc and span lists
- set `target_mode = answer` for `respond_*` dialogue acts
- set `target_mode = follow_up` for `query_condition`
- loaders must tolerate empty references even though the observed local agent turns include them

Why this shape:

- it matches real dialogues where agent turns can occur back to back
- it supports both live inference and offline evaluation
- it avoids assuming that every assistant generation is a direct reply to a fresh user turn

## 5. MVP Domain and Split Strategy

### 5.1 Domain selection

Required MVP domain:

- `dmv`

First expansion after the baseline is stable:

- `ssa`

Reason:

- `dmv` gives a complete, understandable smoke path
- `ssa` is the next best expansion without jumping to all four domains

### 5.2 Split usage

- `train`: prompt iteration, retrieval tuning, optional few-shot examples
- `validation`: primary offline benchmark
- `test`: do not use for headline metrics in the MVP

MVP recommendation:

- implement the full pipeline on `dmv`
- report primary metrics on `dmv` validation first
- extend the same benchmark flow to `ssa` only after the `dmv` path is stable

## 6. Runtime Contract

The MVP must be runnable locally with a documented bootstrap path.

### 6.1 Required runtime pieces

- Python 3.11+
- Postgres with `pgvector`
- a local `.env` or equivalent config source
- one chat model endpoint
- one embedding model endpoint

### 6.2 MVP provider baseline

The default documented MVP path should be local-first:

- local Postgres + `pgvector`
- local Ollama endpoint at `http://localhost:11434` for chat and embeddings

Recommended implementation rule:

- model provider and model names are config, not hard-coded business logic
- the first implementation only needs one provider path
- optional hosted providers can be added later, but they are out of MVP scope

### 6.3 Required config surface

At minimum, config must include:

- dataset root
- enabled domains
- Postgres connection string
- Ollama base URL
- embedding model name
- chat model name
- retrieval top-k
- max retrieval attempts
- output directories for traces and evals

### 6.4 Example `.env`

This is an example config shape, not a pinned model choice. Model names remain configurable.

```dotenv
SUPPORT_GRAPH_DATASET_ROOT=/Users/yglk/coding/grounded-support-rag/multidoc2dial
SUPPORT_GRAPH_ENABLED_DOMAINS=dmv
SUPPORT_GRAPH_POSTGRES_DSN=postgresql://localhost:5432/support_graph
SUPPORT_GRAPH_OLLAMA_BASE_URL=http://localhost:11434
SUPPORT_GRAPH_CHAT_MODEL=your-chat-model
SUPPORT_GRAPH_EMBEDDING_MODEL=your-embedding-model
SUPPORT_GRAPH_RETRIEVAL_TOP_K=5
SUPPORT_GRAPH_MAX_RETRIEVAL_ATTEMPTS=2
SUPPORT_GRAPH_RUNS_DIR=outputs/runs
SUPPORT_GRAPH_EVAL_RUNS_DIR=outputs/evals/runs
SUPPORT_GRAPH_EVAL_REPORTS_DIR=outputs/evals/reports
```

Implementation notes:

- config loading should fail fast with a clear error if required values are missing
- chat and embedding model names may point to the same Ollama model only if that model supports both tasks well enough
- later provider support should map into the same logical config fields rather than creating provider-specific business logic paths

## 7. Retrieval System Design

### 7.1 Storage

Use:

- Postgres
- `pgvector`

Rationale:

- straightforward local setup
- sufficient for this corpus size
- simple metadata filtering by domain and doc

### 7.2 Indexed content

Index:

- section-aware chunks derived from documents

Metadata fields:

- `domain`
- `doc_id`
- `doc_title`
- `section_id`
- `section_title`
- `parent_titles`
- `span_ids`

### 7.3 Retrieval inputs

At inference time, the retrieval query should be built from:

- the latest user turn when available
- a compressed form of the recent dialogue history
- optional domain restriction

If no fresh user turn exists immediately before the target assistant turn, the query builder should fall back to the latest unresolved user need inferred from the transcript so far.

### 7.4 Retrieval outputs

Return top-k chunks with:

- chunk text
- document metadata
- span IDs
- similarity score if available

### 7.5 Retrieval baseline

Initial retrieval:

- vector retrieval only
- top-k: 5
- max retrieval attempts: 2 total

Later improvements:

- query rewriting before retrieval
- reranking
- hybrid retrieval

These improvements are not part of the MVP baseline.

## 8. LangChain Responsibilities

LangChain should be used for:

- document loading from local files
- text splitting when section text must be subdivided
- embedding model wrapper
- PGVector integration
- retriever abstraction
- chat model wrapper
- prompt templates
- structured output parsing
- optional tracing hooks

This project should genuinely use LangChain as infrastructure, not just mention it.

## 9. LangGraph Responsibilities

LangGraph should orchestrate the dialogue-aware workflow.

### 9.1 Why a graph is justified

This task is not only `retrieve -> answer`. It requires:

- maintaining conversation state
- deciding if retrieved evidence is enough
- optionally refining retrieval
- deciding when to answer and when to clarify or abstain

### 9.2 Graph state schema

Suggested state:

```json
{
  "example_id": "...",
  "domain": "dmv",
  "conversation": [],
  "latest_user_turn_id": 5,
  "latest_user_utterance": "...",
  "query": "...",
  "retrieved_chunks": [],
  "retrieval_attempts": 0,
  "evidence_grade": {
    "verdict": "partial",
    "reason": "...",
    "missing_information": ["..."]
  },
  "decision": null,
  "response_text": null,
  "citations": []
}
```

### 9.3 MVP graph nodes

#### `prepare_query`

- input: conversation transcript + latest user context
- output: retrieval-oriented query or summary

#### `retrieve_docs`

- input: query, domain
- output: top-k chunks from `pgvector`

#### `grade_evidence`

- input: retrieved chunks, query, conversation
- output: structured verdict: `sufficient`, `partial`, or `insufficient`

#### `refine_query`

- input: prior query and evidence-grade feedback
- output: revised query for one additional retrieval attempt

#### `generate_response`

- input: conversation + retrieved chunks + evidence grade
- output: structured response with decision and citations

#### `finalize`

- input: generated response
- output: final normalized response payload

### 9.4 Conditional routing

Flow:

```text
conversation
    |
    v
prepare_query
    |
    v
retrieve_docs
    |
    v
grade_evidence
    |
    +--> sufficient -----------------> generate_response ----> finalize(decision=answer)
    |
    +--> partial + attempts remain --> refine_query ---------> retrieve_docs
    |
    +--> partial/insufficient + no attempts left -----------> finalize(decision=clarify or abstain)
```

### 9.5 Decision rules

For the MVP:

- `answer` if evidence is sufficient and at least one cited chunk directly supports the response
- `clarify` if evidence is partially relevant but a missing condition blocks a safe answer
- `abstain` if evidence remains insufficient after the allowed retries

### 9.6 Stopping conditions

- evidence sufficient
- max retrieval attempts reached
- response finalized

## 10. Inference Contract

### 10.1 Input

```json
{
  "example_id": "...",
  "domain": "dmv",
  "conversation": [
    {"turn_id": 1, "role": "user", "utterance": "..."},
    {"turn_id": 2, "role": "agent", "utterance": "..."},
    {"turn_id": 3, "role": "user", "utterance": "..."}
  ],
  "latest_user_turn_id": 3,
  "latest_user_utterance": "..."
}
```

Notes:

- `conversation` contains all turns up to the point where the assistant should speak next
- `latest_user_turn_id` and `latest_user_utterance` may be `null` if there is no unresolved fresh user turn immediately before generation

### 10.2 Output

```json
{
  "example_id": "...",
  "decision": "answer",
  "response_text": "Grounded assistant response",
  "citations": [
    {
      "doc_id": "...",
      "chunk_id": "...",
      "span_ids": ["4", "5"]
    }
  ],
  "confidence_label": "high",
  "trace_summary": {
    "retrieval_attempts": 1,
    "final_query": "...",
    "graph_path": [
      "prepare_query",
      "retrieve_docs",
      "grade_evidence",
      "generate_response",
      "finalize"
    ]
  }
}
```

Possible decisions:

- `answer`
- `clarify`
- `abstain`

Use `confidence_label`, not a fake calibrated probability. Valid values:

- `high`
- `medium`
- `low`

## 11. Evaluation Design

The evaluation should separate retrieval from answer generation.

### 11.1 Retrieval evaluation

For each turn example:

- gold labels come from `gold_doc_ids` and `gold_span_ids`
- retrieved labels come from top-k retrieved chunks

Metrics:

- `Doc Recall@3`
- `Span Recall@5`
- `MRR@5`

Report retrieval metrics separately for:

- `target_mode = answer`
- `target_mode = follow_up`

### 11.2 Generation evaluation

Primary automatic generation evaluation should use only `target_mode = answer`.

Compare:

- generated response text
- target agent utterance

Metrics:

- `ROUGE-L`
- token-level `F1`
- citation coverage

Citation coverage:

- fraction of gold span IDs that appear in cited chunks

### 11.3 Follow-up evaluation

For `target_mode = follow_up`:

- keep retrieval metrics
- report the model decision distribution
- store outputs for manual review
- do not fold these turns into the primary automatic text-quality score

This keeps the benchmark honest because clarification wording is more weakly constrained than answer wording.

### 11.4 End-to-end evaluation

For the MVP, a turn is considered an end-to-end automatic success only for `target_mode = answer` if:

- a gold doc or span is retrieved
- the generated response clears a minimum similarity threshold
- citations map to valid retrieved chunks
- the final decision is `answer`

Clarification-like turns should be summarized separately.

### 11.5 Evaluation subset for MVP

To keep the project feasible:

- only score agent turns
- keep loaders tolerant of empty references even if they are not observed in the local agent targets
- report `dmv` first
- add `ssa` only after the `dmv` benchmark is stable

### 11.6 Failure analysis

Store failures in a machine-readable report with:

- example ID
- target mode
- gold docs and spans
- retrieved docs and chunks
- decision
- response text
- metric results

This report should support manual inspection after each run.

### 11.7 Experiment tracking requirements

Every evaluation run must write:

- a run manifest with config and model metadata
- machine-readable metrics
- machine-readable failures
- saved predictions

The experiment process, subset strategy, and experiment matrix are defined in `EXPERIMENTS.md`.

## 12. Observability

### 12.1 Required

- local structured traces
- per-run logs with:
  - graph node sequence
  - final query
  - retrieved chunks
  - answer decision
  - latency

### 12.2 Optional

- hosted tracing behind a config flag

Hosted observability should remain optional, not a hard dependency.

### 12.3 Data-flow diagram

```text
raw docs ------------------> chunk builder ------------------> derived chunks ---------> pgvector index
raw dialogues -------------> example builder ---------------> turn examples ----------> eval harness

turn examples / live input --> graph runtime --> traces + outputs + failure reports
```

## 13. CLI Commands and Interaction Design

The CLI is the MVP interface. It needs explicit interaction design, not just command names.

The detailed design rules live in `DESIGN.md`. This section defines the minimum technical contract those rules depend on.

### 13.1 Command surfaces

Required user-facing commands:

- `grounded-support-rag build-chunks --domain dmv`
- `grounded-support-rag build-examples --domain dmv --split validation`
- `grounded-support-rag index-docs --domain dmv`
- `grounded-support-rag run --example-id <id>`
- `grounded-support-rag eval --split validation --domain dmv`

### 13.2 `grounded-support-rag run` default hierarchy

Default output must present, in this order:

1. example identity
2. last relevant user context
3. decision
4. response text
5. citations
6. short trace summary

Default output must not start with:

- raw JSON
- chunk dumps
- prompt text
- metric blocks

Recommended shape:

```text
SupportGraph Run
Example: dmv::...::turn_6
Context
User: I moved and need to change the address on my license.

Decision: answer

Response
...

Citations
[1] ...

Trace
Attempts: 1
Path: prepare_query -> retrieve_docs -> grade_evidence -> generate_response -> finalize
```

### 13.3 `grounded-support-rag run --verbose`

Verbose mode may add:

- full recent conversation context
- final query
- evidence grade
- retrieved chunk previews
- graph node path

Verbose details should appear after the primary answer view.

### 13.4 `grounded-support-rag eval` default hierarchy

Default eval output must present, in this order:

1. run identity
2. subset identity
3. headline metrics
4. failure-category summary
5. artifact paths

Recommended shape:

```text
SupportGraph Eval
Run: 20260318-143000-dmv-history-retry
Subset: dmv validation / answer

Headline Metrics
Doc Recall@3: ...
Span Recall@5: ...
ROUGE-L: ...
F1: ...

Failure Snapshot
wrong_doc: ...
missed_history: ...

Artifacts
outputs/evals/runs/<run_id>/summary.md
outputs/evals/runs/<run_id>/failures.jsonl
```

### 13.5 State coverage

The CLI contract must explicitly handle:

- missing config
- missing index
- no sufficient evidence found
- `answer`
- `clarify`
- `abstain`
- no eval artifacts yet

For each state, the interface must tell the user what to do next.

### 13.6 Copy rules

User-facing copy should be:

- direct
- restrained
- specific
- source-grounded

User-facing copy should avoid:

- hype
- filler
- anthropomorphic phrasing
- fake numeric confidence

### 13.7 Accessibility rules

- state meaning must not rely on color alone
- output must remain readable in plain text
- citations must use stable labels
- markdown summaries must be understandable without opening JSON artifacts

## 14. Repository Layout

Keep the MVP layout intentionally small.

Suggested layout:

```text
.
├── README.md
├── DESIGN.md
├── EXPERIMENTS.md
├── PRD.md
├── TECH_SPEC.md
├── multidoc2dial/
│   ├── multidoc2dial_doc.json
│   ├── multidoc2dial_dial_train.json
│   ├── multidoc2dial_dial_validation.json
│   └── multidoc2dial_dial_test.json
├── data/
│   ├── derived/
│   │   ├── chunks/
│   │   └── examples/
│   ├── eval_subsets/
│   │   ├── smoke.jsonl
│   │   └── frozen_ablation.jsonl
│   └── indexes/
├── support_graph/
│   ├── cli.py
│   ├── settings.py
│   ├── dataset.py
│   ├── chunks.py
│   ├── examples.py
│   ├── index.py
│   ├── retrieve.py
│   ├── graph.py
│   ├── evaluate.py
│   └── traces.py
├── tests/
│   ├── test_dataset.py
│   ├── test_chunks.py
│   ├── test_examples.py
│   ├── test_retrieve.py
│   ├── test_graph.py
│   └── test_evaluate.py
└── outputs/
    ├── evals/
    │   ├── reports/
    │   └── runs/
    │       └── <run_id>/
    │           ├── manifest.json
    │           ├── metrics.json
    │           ├── failures.jsonl
    │           ├── predictions.jsonl
    │           └── summary.md
    └── runs/
        └── <run_id>/
            ├── manifest.json
            ├── result.json
            └── trace.jsonl
```

### 14.1 Future production structure

This is not the MVP repository shape. It is the intended evolution path if SupportGraph needs to serve high traffic, persist live conversations, and run separate online and offline workloads.

The recommended evolution is:

- keep a modular monolith first
- split into multiple process roles from the same codebase
- only split into microservices later if scaling or team boundaries justify it

Target production structure:

```text
.
├── README.md
├── DESIGN.md
├── EXPERIMENTS.md
├── PRD.md
├── TECH_SPEC.md
├── .env.example
├── pyproject.toml
├── alembic.ini
├── migrations/
├── deploy/
│   ├── docker/
│   ├── k8s/
│   └── systemd/
├── docs/
│   ├── architecture.md
│   ├── runbooks/
│   └── incidents/
├── data/
│   ├── eval_subsets/
│   └── fixtures/
├── scripts/
│   ├── bootstrap_db.py
│   ├── backfill_embeddings.py
│   └── run_eval.py
├── support_graph/
│   ├── config/
│   │   ├── settings.py
│   │   └── logging.py
│   ├── entrypoints/
│   │   ├── cli.py
│   │   ├── api.py
│   │   ├── worker.py
│   │   └── scheduler.py
│   ├── contracts/
│   │   ├── requests.py
│   │   ├── responses.py
│   │   └── events.py
│   ├── dataset/
│   │   ├── loaders.py
│   │   ├── chunks.py
│   │   └── examples.py
│   ├── conversations/
│   │   ├── repository.py
│   │   ├── summarizer.py
│   │   └── state.py
│   ├── retrieval/
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   └── rerank.py
│   ├── graph/
│   │   ├── state.py
│   │   ├── nodes.py
│   │   ├── policies.py
│   │   └── runtime.py
│   ├── inference/
│   │   ├── prompts.py
│   │   ├── models.py
│   │   └── citations.py
│   ├── storage/
│   │   ├── postgres.py
│   │   ├── redis.py
│   │   └── blobs.py
│   ├── jobs/
│   │   ├── indexing.py
│   │   ├── evaluations.py
│   │   └── maintenance.py
│   ├── telemetry/
│   │   ├── traces.py
│   │   ├── metrics.py
│   │   └── audit.py
│   └── ui/
│       └── presenters.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   └── load/
└── outputs/
    └── evals/
```

Architectural meaning:

- `entrypoints/` separates the CLI, API, worker, and scheduler as distinct process roles
- `contracts/` keeps payloads stable so internal boundaries can become service boundaries later
- `conversations/` makes persisted chat state explicit instead of keeping it in process memory
- `storage/` prepares for Postgres, Redis, and blob storage without forcing a distributed architecture on day one
- `jobs/` isolates long-running indexing and evaluation work from online serving

Split triggers for later microservices:

- online inference and offline indexing have materially different scaling profiles
- one subsystem needs independent deploy cadence or uptime guarantees
- queue-backed jobs become a major bottleneck
- different teams need clear ownership boundaries
- a single deployable becomes operationally hard to reason about

Until one of those triggers appears, the default recommendation remains a modular monolith.

### 14.2 Production readiness gaps and deferred requirements

The MVP architecture is intentionally simpler than a production deployment. Before SupportGraph should be described as production-ready for live user traffic, the following gaps must be addressed.

#### A. Auth, authorization, and tenant isolation

Needed for production:

- authenticated API access
- authorization rules for conversation and artifact access
- tenant or user ownership on persisted conversations
- access control for traces, predictions, and failure artifacts

Why this is deferred:

- the MVP is CLI-first over a public benchmark dataset
- there is no live multi-user API in scope yet

#### B. Conversation consistency and idempotency

Needed for production:

- stable `conversation_id` and `turn_id` rules
- idempotency keys for message submission
- duplicate-request protection
- ordering guarantees for concurrent writes
- a policy for stale-context retries

Why this is deferred:

- the MVP inference path primarily consumes offline examples
- live conversation persistence is part of the future production path, not the first build

#### C. Dependency failure policy

Needed for production:

- request timeout budgets
- bounded retries with backoff
- circuit-breaker or degraded-mode behavior for model and database dependencies
- explicit behavior when Postgres, `pgvector`, or Ollama is unavailable

Why this is deferred:

- the MVP is local-first and primarily operator-driven
- the first goal is functional correctness and reproducible evaluation

#### D. SRE observability package

Needed for production:

- named SLIs and SLOs
- dashboards
- alerts
- runbooks
- queue-depth and dependency-health monitoring for background jobs

Minimum suggested SLIs:

- request success rate
- p50 and p95 end-to-end latency
- retrieval latency
- model latency
- error rate by stage
- `clarify` and `abstain` rate
- indexing and eval job duration

Why this is deferred:

- the MVP only requires traces, logs, and artifact outputs
- alerting is unnecessary before a live served system exists

#### E. Data retention, redaction, and privacy

Needed for production:

- retention policies for conversations, traces, predictions, and failures
- redaction rules for sensitive content
- environment separation for local, staging, and production data
- artifact access controls

Why this is deferred:

- the MVP uses a public benchmark corpus
- there is no user-submitted private data in the initial scope

#### F. Safe rollout controls

Needed for production:

- feature flags for graph or prompt variants
- canary or shadow rollout options
- rollback rules for model and prompt changes
- environment-specific configuration controls

Why this is deferred:

- the MVP uses offline evaluation as the primary quality gate
- production rollout workflows only matter once a live API or app exists

Recommended rule:

- do not market or describe the MVP as production-ready
- describe it as a production-shaped modular monolith with explicit deferred operational requirements

## 15. Implementation Order

### Phase 0

- lock config and runtime contract
- add the data-flow and graph-flow diagrams
- add a tiny smoke-test fixture strategy
- define frozen eval subsets and run manifest format
- define default CLI layouts and state copy from `DESIGN.md`

### Phase 1

- parser for raw docs and dialogues
- derived chunk builder
- derived example builder
- `dmv` smoke path

### Phase 2

- `pgvector` schema and indexing
- retrieval baseline

### Phase 3

- LangGraph workflow
- bounded retry path
- grounded response generation with citations

### Phase 4

- offline evaluation harness
- failure report generation
- run manifest and prediction artifact generation
- `dmv` validation benchmark on `target_mode = answer`
- manual review sample for `target_mode = follow_up`
- human-readable eval summary generation

### Phase 5

- traces
- cleanup
- README and results
- expand to `ssa` if the `dmv` path is stable

## 16. Definition of Done

The MVP is done when:

- raw dataset can be transformed into retrieval chunks and turn examples
- documents are indexed in `pgvector`
- one CLI command runs the graph for a validation example
- the output contains a normalized decision, response text, and citations
- validation metrics can be generated automatically for `target_mode = answer`
- every eval run writes a manifest, metrics, failures, and predictions artifact set
- clarification-style turns are reported separately with saved review artifacts
- default CLI output is readable without raw JSON
- setup, empty, clarify, abstain, and failure states all include a clear next step
- the project is understandable without external context
