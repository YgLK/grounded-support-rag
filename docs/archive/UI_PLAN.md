# SupportGraph Workbench Plan

## Purpose

This document defines the plan for a local-first internal UI for SupportGraph.

The goal is not to build a polished end-user support product yet. The goal is to make the existing system easier to understand, debug, demo, and evaluate while the runtime and evaluation contracts are still stabilizing.

## Product Stance

SupportGraph remains CLI-first.

The Workbench is an operator and developer surface layered on top of the existing runtime, trace, and eval artifacts. It should help answer:

1. What did the system see?
2. What query did it build?
3. What did ranked retrieval return?
4. What evidence was actually used to grade and generate?
5. Why did the graph answer, clarify, abstain, or retry?
6. Did a change improve metrics or just shift failure modes?

## Why Now

The repo already has the right raw materials:

- streaming runtime milestones in `support_graph/runtime/graph.py`
- structured local trace files in `support_graph/runtime/traces.py`
- eval artifacts in `support_graph/evaluation/evaluate.py`
- terminal inspection flows in `support_graph/app/cli.py`

That means a thin local UI can be built without redesigning the core system.

## Non-Goals

Not in scope for this plan:

- a customer-facing support chat product
- auth, accounts, or multi-user state
- deployment-first architecture
- persistent chat history beyond the current local run and saved artifacts
- replacing the CLI as the primary development interface
- redesigning the runtime graph or evaluation contract for the sake of the UI

## Users

Primary users:

- the developer improving retrieval, grounding, and runtime behavior
- the operator reviewing single runs and eval failures
- the reviewer trying to understand how the system works

## Core Constraints

The UI must preserve current project invariants:

- `run` and `eval` continue to share the same runtime graph
- `retrieval_ranked_chunks` and `retrieved_chunks` stay distinct
- local traces and eval artifacts remain the source of truth
- default local workflow stays deterministic where the backend is already deterministic
- the UI should read the artifact layout defined in `ARTIFACT_LAYOUT.md` directly instead of inventing extra UI-only storage
- artifact discovery should come from the fixed namespaces in `ARTIFACT_LAYOUT.md`, not from mixed-directory heuristics

## MVP Definition

The first useful version of the Workbench should provide four surfaces.

### 1. Artifact Explorer

Purpose:

- list existing eval runs, eval reports, and standalone runs
- make eval metrics, failure counts, and run summaries legible
- provide entry points into deeper inspection

Must show for eval runs:

- run id
- created-at timestamp
- persisted display subset label
- provider and model snapshot
- headline retrieval and generation metrics
- failure counts
- artifact locations

Must show for standalone runs:

- run id
- created-at timestamp
- example id
- provider and model snapshot
- final decision
- final query when present
- trace file location
- artifact location

Must show for eval reports:

- report id
- created-at timestamp
- report type
- title
- related run ids
- artifact location

Primary sources:

- `outputs/evals/runs/<eval_run_id>/manifest.json`
- `outputs/evals/runs/<eval_run_id>/metrics.json`
- `outputs/evals/runs/<eval_run_id>/summary.md`
- `outputs/evals/reports/<report_id>/manifest.json`
- `outputs/evals/reports/<report_id>/report.md`
- `outputs/runs/<run_id>/manifest.json`
- `outputs/runs/<run_id>/result.json`
- `outputs/runs/<run_id>/trace.jsonl`

Notes:

- the new artifact contract should persist `subset_label` in the eval manifest instead of relying on UI derivation
- eval reports should appear as a separate section from eval runs and standalone runs
- eval report listing should use report manifest metadata instead of filename or file mtime heuristics

### 2. Failure Review

Purpose:

- filter and inspect failures without opening raw JSONL manually

Must show per example:

- example id
- target mode
- failure label
- latest user utterance
- target text
- final decision
- response text
- gold doc ids and span ids
- retrieval metrics
- citation validity signals

Primary sources:

- `outputs/evals/runs/<eval_run_id>/failures.jsonl`
- `outputs/evals/runs/<eval_run_id>/predictions.jsonl`
- `outputs/evals/runs/<eval_run_id>/retrieval_examples.jsonl`
- `outputs/evals/runs/<eval_run_id>/manual_review.csv`

### 3. Trace Inspector

Purpose:

- make one finished run easy to read
- support both eval-backed example traces and standalone runs

Must show:

- graph path
- node latencies
- retrieval attempts
- final query
- evidence grade
- fallback events
- final decision and citations
- raw trace events in a collapsible debug panel

Primary sources:

- `outputs/evals/runs/<eval_run_id>/trace_index.json`
- `outputs/evals/runs/<eval_run_id>/predictions.jsonl`
- `outputs/evals/runs/<eval_run_id>/traces/<trace_file>.jsonl`
- `outputs/runs/<run_id>/result.json`
- `outputs/runs/<run_id>/trace.jsonl`

Notes:

- when the trace is linked to an eval example, enrich the trace view with `predictions.jsonl`
- resolve eval-local trace files through `trace_index.json` using `example_id` -> `trace_file`, not by deriving filenames from raw example ids
- standalone runs should use `result.json` for response text, citations, and summary fields, with `trace.jsonl` as the raw debug source

### 4. Live Run

Purpose:

- watch one example execute in real time

Must show:

- latest user utterance
- node-by-node milestone status as the graph progresses
- live query and query refinement
- ranked retrieval results
- expanded evidence set
- evidence grading verdict
- streamed answer text when available
- final citations and trace summary

Primary source:

- `astream_graph_events(...)` from `support_graph/runtime/graph.py`

## Recommended Page Map

1. `/`
   Artifact explorer for eval runs, reports, and standalone runs
2. `/evals/<eval_run_id>`
   Eval summary plus failure overview
3. `/evals/<eval_run_id>/failures`
   Filterable failure table
4. `/evals/<eval_run_id>/examples/<example_id>`
   Combined failure detail and trace inspection
5. `/runs/<run_id>`
   Standalone run inspection
6. `/reports/<report_id>`
   Eval report detail
7. `/live`
   Run one example and watch graph events

## Local Serving Defaults

Recommended defaults:

- host: `127.0.0.1`
- port: `8008`

Recommended CLI flags:

- `grounded-support-rag ui --host 127.0.0.1 --port 8008`
- optional `--open` later if that improves the local workflow

## Information Hierarchy

The UI should follow the same hierarchy already established for the CLI and markdown artifacts.

For a single example:

1. latest user context
2. decision
3. answer or follow-up text
4. citations
5. query and evidence path
6. raw debug detail

For an eval run:

1. run identity and config snapshot
2. headline metrics
3. biggest failure buckets
4. examples worth inspecting
5. artifact links and raw files

## Key Interaction Patterns

### Ranked vs Expanded Evidence

This is the most important debugging distinction in the current system.

Every example-detail screen should show two side-by-side panels:

- `retrieval_ranked_chunks`
- `retrieved_chunks`

The UI should never flatten these into one list.

### Click Through From Aggregate To Example

Each failure bucket should be clickable so the user can move from:

- run metrics
- to failure label
- to example
- to trace

without leaving the UI.

### Progressive Disclosure

Default panels should stay readable.

Raw JSON, full chunk text, and event payloads should live behind drawers, tabs, or expandable sections.

### One Example, Many Views

The same example-detail route should be able to show:

- eval-derived metrics
- ranked retrieval
- expanded evidence
- final answer
- citations
- raw trace timeline

This avoids fragmenting debugging across too many screens.

## Technical Approach

### Architecture

Build a thin local web app inside the existing Python project.

Recommended shape:

- command: `uv run grounded-support-rag ui`
- backend: FastAPI
- templates: Jinja2
- interaction layer: HTMX loaded from CDN
- light client state: Alpine.js loaded from CDN
- component/style layer: Tabler loaded from CDN
- live updates: Server-Sent Events

Reason:

- simple local setup
- easy access to existing Python loaders and runtime functions
- no need to introduce a separate frontend build pipeline on day one
- no need to vendor or version frontend assets for the first prototype
- HTMX keeps the UI server-driven for filters, drawers, and partial page refreshes
- Alpine.js can stay limited to tiny local state where HTMX alone is awkward
- Tabler provides usable internal-tool components without building a design system from scratch

### Backend Responsibilities

The backend should:

- enumerate eval runs, eval reports, and standalone runs from the namespaces defined in `ARTIFACT_LAYOUT.md`
- load manifest, metrics, failures, predictions, retrieval examples, and trace index
- load report manifests and render report markdown
- load standalone run results and raw trace JSONL files
- allocate artifact ids and final trace destinations before execution so runtime traces are written in place
- resolve example ids from the existing derived examples artifacts before calling the live runtime
- expose a live run endpoint that wraps `astream_graph_events(...)`
- expose small normalized view models for templates

### Frontend Responsibilities

The frontend should:

- render dense operator-oriented screens
- keep typography and spacing calm and legible
- show timelines, status pills, and side-by-side evidence panels
- avoid heavy client-side state unless needed for the live run screen
- prefer HTMX swaps over custom JavaScript for data loading and view transitions
- reserve Alpine.js for lightweight toggles, drawers, and ephemeral panel state

## Data Contracts For V1

The UI should read the defined artifact layout as-is.

Filesystem paths, required files, and writer responsibilities live in `ARTIFACT_LAYOUT.md`.

If a normalization layer is needed, keep it in the backend view models rather than changing the artifact schema immediately.

The canonical per-example source for eval detail pages should be `predictions.jsonl`.

Use the artifact set like this:

- `predictions.jsonl` for `decision`, `response_text`, `citations`, ranked retrieval, expanded evidence, and per-example metrics
- `failures.jsonl` for failure-only filtering, buckets, and failure navigation
- `retrieval_examples.jsonl` for retrieval-focused tables and summaries
- `trace_index.json` plus per-example trace JSONL for eval trace lookup, timelines, and debug detail; `trace_index.json` is the canonical mapping from `example_id` to the path-safe `trace_file`
- report `manifest.json` for report list metadata and report routing
- report `report.md` for rendered report content
- `result.json` plus `trace.jsonl` for standalone run summary and debug detail

Trace ownership should follow the direct-write model from `ARTIFACT_LAYOUT.md`.

Suggested normalized entities:

- `ArtifactListEntry`
- `EvalRunSummary`
- `EvalRunDetail`
- `EvalReportSummary`
- `EvalReportDetail`
- `TraceRunSummary`
- `FailureRecordView`
- `ExampleDetailView`
- `ExampleTraceView`
- `LiveGraphEventView`

## Delivery Phases

### Phase 1: Static Artifact Explorer

Scope:

- list eval runs, eval reports, and standalone runs
- show run summary page
- show standalone run summary page
- show failure buckets and example counts

Dependencies:

- the structured eval and run artifact layout from `ARTIFACT_LAYOUT.md`

Success criteria:

- a user can identify the latest run, its headline metrics, and its dominant failure labels in under 30 seconds
- a user can identify the latest standalone run and open its summary in under 30 seconds
- a user can sort artifact lists by date and filter eval runs by domain, subset, and provider
- a valid run directory under the defined artifact layout renders without any path inference heuristics

### Phase 2: Failure Review And Trace Inspector

Scope:

- failure table with filters
- example-detail page
- trace timeline and raw event drawer

Dependencies:

- existing `failures.jsonl`, `predictions.jsonl`, `retrieval_examples.jsonl`, `trace_index.json`, per-example eval trace JSONL files, and standalone run artifacts under the new layout

Success criteria:

- a user can move from a failure label to the raw trace for one example in under 3 clicks

### Phase 3: Live Run

Scope:

- form to run one example id
- SSE-driven event timeline
- live query, retrieval, grading, and response panels

Dependencies:

- `astream_graph_events(...)`
- existing example lookup over derived examples JSONL

Success criteria:

- a user can watch one graph run in real time and understand where a retry or abstention happened without opening the terminal

### Phase 4: Compare

Scope:

- compare two runs or two examples side by side
- highlight differences in query, ranked retrieval, evidence grade, decision, and citations

Default rule:

- compare runs on the same subset by default
- allow cross-subset comparison only as an explicit secondary mode with a visible warning that results are not apples to apples

Dependencies:

- stable example-detail views from earlier phases

Success criteria:

- a user can explain why run B improved or regressed on a given example using one comparison screen

## Risks

### Risk 1: UI Forces Premature Contract Changes

Mitigation:

- read existing files first
- normalize in view models
- keep artifact-schema changes explicit and minimal

### Risk 2: Live Screen Becomes A JSON Dump

Mitigation:

- keep the default view decision-first
- treat raw payloads as debug drawers

### Risk 3: Too Much Frontend Infrastructure Too Early

Mitigation:

- prefer server-rendered pages for phases 1 and 2
- limit client-side code to filters and live streaming

## Visual Direction

The Workbench should feel like an investigation console, not a generic SaaS dashboard.

Guidelines:

- calm light background
- dense but readable panels
- strong monospace support for ids, paths, and queries
- restrained status colors for answer, clarify, abstain, retry, and failure
- clear visual separation between ranked retrieval and expanded evidence

## Definition Of Success

The Workbench is successful when:

- single-run behavior is legible without opening raw JSONL files
- eval results can be reviewed visually instead of by jumping between terminal commands and artifact files
- retrieval regressions are easier to diagnose
- demos and project walkthroughs become easier without changing the underlying system contract

## Immediate Next Step

Start with Phase 1 and Phase 2.

That means:

1. add a `grounded-support-rag ui` command
2. land the full layout migration described in `ARTIFACT_LAYOUT.md`, including writers, CLI consumers, settings defaults, docs, and tests
3. land the runtime contract changes needed for direct trace writes to final artifact paths
4. delete old generated artifacts and regenerate fresh runs under the new structure
5. build a small FastAPI app that reads the new eval and run artifact namespaces directly
6. ship artifact explorer, eval detail, failure list, standalone run view, and example trace views before attempting live execution
