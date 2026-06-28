# SupportGraph Artifact Layout

## Purpose

This document defines the on-disk generated artifact contract for SupportGraph run, eval, and report outputs that SupportGraph writers and the Workbench UI share.

It covers generated outputs only. It does not change the source-code package layout of the repository.

## Goals

- use fixed namespaces instead of mixed-root crawling
- make artifact discovery deterministic
- give each eval run, eval report, and standalone run one owning location
- let the UI read artifacts directly without UI-only storage
- drop legacy mixed-root artifact layouts instead of supporting them indefinitely

## Namespaces

Generated artifacts should live under three namespaces:

- `outputs/evals/runs/`
- `outputs/evals/reports/`
- `outputs/runs/`

These namespaces cover the generated run, eval, and report outputs the Workbench reads for artifact exploration and inspection.

They do not replace the existing derived example artifacts used as inputs for live execution.

That means:

- artifact explorer, eval detail, report detail, and standalone run inspection should read from these namespaces
- `/live` may still depend on derived examples under `data/derived/examples/` to resolve an input `example_id`

## Schema Conventions

Use concrete schemas for artifact contracts.

Rules:

- every JSON artifact should define required keys explicitly
- optional keys should be listed explicitly
- keys not listed are out of contract for v1
- timestamps should use ISO 8601 strings with timezone information
- nullable fields should be present with `null` rather than omitted when the contract expects the key
- arrays should default to `[]`, not be omitted

When a filesystem artifact needs a safe filename, separate:

- the canonical logical id used by the product, such as `example_id`
- the path-safe storage key used in the filesystem, such as `trace_file`

`trace_file` should be:

- deterministic from `example_id`
- ASCII and path-safe
- stored with the `.jsonl` suffix
- treated as a storage key, not as the canonical identity

`trace_file` is producer-owned opaque storage data.

Rules:

- producers may generate `trace_file` deterministically
- consumers, tests, migration scripts, and repair tools must not derive `trace_file` from `example_id`
- consumers must resolve `trace_file` from the owning manifest or index artifact

## Shared Enums

### `Decision`

Allowed values:

- `answer`
- `clarify`
- `abstain`

### `TargetMode`

Allowed values:

- `answer`
- `follow_up`

### `EvidenceVerdict`

Allowed values:

- `sufficient`
- `partial`
- `insufficient`

### `ResponseConfidence`

Allowed values:

- `high`
- `medium`
- `low`

### `FailureLabel`

Allowed values:

- `bad_clarification`
- `abstained_with_evidence`
- `wrong_doc`
- `missed_history`
- `right_doc_wrong_section`
- `weak_citations`
- `unsupported_answer`
- `runtime_error`

### `ReportType`

Allowed values:

- `experiment_summary`
- `comparison_report`

## Shared Nested Schemas

### `CitationRecord`

Required keys:

- `doc_id`: string
- `chunk_id`: string
- `span_ids`: array of strings

Optional keys:

- none for v1

### `ChunkRecordView`

Required keys:

- `chunk_id`: string
- `doc_id`: string
- `doc_title`: string or null
- `section_id`: string or null
- `section_title`: string or null
- `parent_titles`: array of strings
- `span_ids`: array of strings
- `text`: string

Optional keys:

- `domain`: string
- `rank`: integer or null
- `score`: number or null
- `token_count`: integer or null
- `subchunk_index`: integer or null
- `start_sec`: integer or null
- `end_sec`: integer or null

### `RankedChunkRecordView`

`RankedChunkRecordView` uses the same keys as `ChunkRecordView` with one stricter rule:

- `rank` is required and must be a non-null integer

Rules:

- `retrieval_ranked_chunks` must preserve retrieval rank order
- the order of `retrieval_ranked_chunks` is semantically meaningful and must not be treated as presentation-only
- consumers should not re-sort `retrieval_ranked_chunks` except by the stored `rank`

### `FallbackRecord`

Required keys:

- `used`: boolean
- `node`: string
- `mode`: string
- `exception_type`: string
- `error`: string

Optional keys:

- none for v1

### `TraceSummary`

Required keys:

- `retrieval_attempts`: integer
- `final_query`: string
- `graph_path`: array of strings
- `latency_ms`: number or null
- `trace_path`: string
- `fallback_count`: integer
- `fallback_nodes`: array of strings
- `fallbacks`: array of `FallbackRecord`

Optional keys:

- `observability`: object

Rules:

- `trace_path` must be a project-relative path from the repository root, not an absolute path
- for eval-backed examples, `trace_path` should be `outputs/evals/runs/<eval_run_id>/traces/<trace_file>.jsonl`
- for standalone runs, `trace_path` should be `outputs/runs/<run_id>/trace.jsonl`

### `PerExampleMetrics`

Required keys:

- `doc_recall_at_1`: number or null
- `doc_recall_at_3`: number or null
- `doc_recall_at_5`: number or null
- `doc_recall_at_10`: number or null
- `span_recall_at_5`: number or null
- `mrr_at_5`: number or null
- `rouge_l`: number
- `token_f1`: number
- `exact_match`: number
- `sacrebleu`: number
- `citation_coverage`: number or null
- `citations_valid`: number
- `end_to_end_success`: number

Optional keys:

- none for v1

## Eval Runs

Each eval run owns one directory:

```text
outputs/evals/runs/<eval_run_id>/
  eval_subsets/
    smoke.jsonl
    frozen_experiment.jsonl
  indexes/
  manual_review.csv
  retrieval_examples.jsonl
  trace_index.json
  summary.md
  traces/
    <trace_file>.jsonl
```

Required files:

- `manifest.json`
- `metrics.json`
- `predictions.jsonl`
- `failures.jsonl`
- `manual_review.csv`
- `retrieval_examples.jsonl`
- `trace_index.json`
- `summary.md`

Rules:

- `manifest.json` must persist `subset_label`
- `predictions.jsonl` is the canonical per-example source for eval detail pages
- `trace_index.json` should index only traces owned by this eval run
- per-example traces should live under the run-local `traces/` directory, not in a global trace root
- eval execution should write each trace directly to its final path under `outputs/evals/runs/<eval_run_id>/traces/`
- the UI and CLI review tools should treat an eval run as valid only when the required files are present

### Eval Run `manifest.json` Schema

Required keys:

- `run_id`: string
- `created_at`: string
- `dataset_root`: string
- `domains`: array of strings
- `split`: string
- `eval_subset`: string
- `subset_label`: string
- `target_modes`: array of strings
- `provider`: object
- `chunking`: object
- `retrieval`: object
- `graph`: object
- `prompt_version`: string
- `notes`: string

`provider` required keys:

- `type`: string
- `chat_base_url`: string or null
- `embedding_type`: string or null
- `embedding_base_url`: string or null
- `chat_model`: string or null
- `embedding_model`: string or null

`chunking` required keys:

- `strategy`: string
- `max_tokens_per_chunk`: integer

`retrieval` required keys:

- `top_k`: integer or null
- `candidate_k`: integer
- `max_attempts`: integer
- `use_history`: boolean
- `content_only_reasoning`: boolean
- `neighbor_expansion`: boolean

`graph` required keys:

- `enable_retry`: boolean
- `decision_policy_version`: string

Optional keys:

- `experiment`: object

### Eval Run `metrics.json` Schema

Required keys:

- `counts`: object
- `retrieval`: object
- `generation`: object
- `decision_distribution`: object
- `latency_ms`: object
- `failure_counts`: object mapping string to integer

`counts` required keys:

- `examples`: integer
- `answer_examples`: integer
- `follow_up_examples`: integer

`retrieval` required keys:

- `answer`: object
- `follow_up`: object
- `overall`: object

Each retrieval metrics object requires:

- `doc_recall_at_1`: number or null
- `doc_recall_at_3`: number or null
- `doc_recall_at_5`: number or null
- `doc_recall_at_10`: number or null
- `span_recall_at_5`: number or null
- `mrr_at_5`: number or null

`generation` required keys:

- `answer`: object

`generation.answer` required keys:

- `rouge_l`: number or null
- `token_f1`: number or null
- `exact_match`: number or null
- `sacrebleu`: number or null
- `citation_coverage`: number or null
- `end_to_end_success_rate`: number or null

`decision_distribution` required keys:

- `overall`: object mapping string to number
- `answer`: object mapping string to number
- `follow_up`: object mapping string to number

`latency_ms` required keys:

- `average`: number or null
- `p95`: number or null

### Eval Run `trace_index.json` Schema

Top-level required keys:

- `entries`: array of `TraceIndexEntry`

`TraceIndexEntry` required keys:

- `example_id`: string
- `trace_file`: string
- `graph_path`: array of strings
- `retrieval_attempts`: integer
- `final_query`: string
- `decision`: `Decision`
- `failure_label`: `FailureLabel` or null
- `total_latency_ms`: number
- `node_latency_ms`: object mapping string to array of numbers
- `retrieval_ranked_count`: integer
- `retrieved_count`: integer
- `fallback_count`: integer
- `fallback_nodes`: array of strings

`TraceIndexEntry` optional keys:

- none for v1

Rules:

- `trace_file` is the canonical storage key for the per-example trace file
- the full trace path is derived as `outputs/evals/runs/<eval_run_id>/traces/<trace_file>`
- `example_id` remains the canonical logical identity for routing and lookup
- the UI and CLI should resolve traces through `trace_index.json`, not by re-deriving filenames from `example_id`

Example:

```json
{
  "entries": [
    {
      "example_id": "dmv::1409501a35697e0ce68561e29577b90a::turn_2",
      "trace_file": "dmv-1409501a35697e0ce68561e29577b90a-turn-2.jsonl",
      "graph_path": [
        "prepare_query",
        "retrieve_docs",
        "grade_evidence",
        "generate_response",
        "finalize"
      ],
      "retrieval_attempts": 1,
      "final_query": "replace address on registration new york",
      "decision": "answer",
      "failure_label": null,
      "total_latency_ms": 1842.5,
      "node_latency_ms": {
        "prepare_query": [1.2],
        "retrieve_docs": [108.3],
        "grade_evidence": [422.6],
        "generate_response": [1201.9],
        "finalize": [0.7]
      },
      "retrieval_ranked_count": 5,
      "retrieved_count": 6,
      "fallback_count": 0,
      "fallback_nodes": []
    }
  ]
}
```

### Eval Run `predictions.jsonl` Record Schema

Required keys:

- `example_id`: string
- `target_mode`: `TargetMode`
- `target_turn_id`: integer
- `latest_user_utterance`: string or null
- `gold_doc_ids`: array of strings
- `gold_span_ids`: array of strings
- `target_text`: string
- `decision`: `Decision`
- `response_text`: string
- `citations`: array of `CitationRecord`
- `retrieval_ranked_chunks`: array of `RankedChunkRecordView`
- `retrieved_chunks`: array of `ChunkRecordView`
- `trace_summary`: `TraceSummary`
- `metrics`: `PerExampleMetrics`
- `failure_label`: `FailureLabel` or null

Optional keys:

- `runtime_error`: object

`runtime_error` required keys when present:

- `exception_type`: string
- `error`: string

### Eval Run `failures.jsonl` Record Schema

`failures.jsonl` uses the same record schema as `predictions.jsonl` with one extra rule:

- `failure_label` must be a non-null `FailureLabel`

### Eval Run `retrieval_examples.jsonl` Record Schema

Required keys:

- `example_id`: string
- `target_mode`: `TargetMode`
- `latest_user_utterance`: string or null
- `target_text`: string
- `gold_doc_ids`: array of strings
- `gold_span_ids`: array of strings
- `final_query`: string or null
- `retrieval_attempts`: integer or null
- `retrieval_ranked_chunks`: array of `RankedChunkRecordView`
- `retrieved_chunks`: array of `ChunkRecordView`
- `doc_recall_at_3`: number or null
- `span_recall_at_5`: number or null
- `mrr_at_5`: number or null
- `failure_label`: `FailureLabel` or null
- `decision`: `Decision`

Optional keys:

- none for v1

### Eval Run `manual_review.csv` Column Contract

Required columns, in order:

- `run_id`
- `example_id`
- `target_mode`
- `decision`
- `failure_label`
- `latest_user_utterance`
- `response_text`
- `gold_doc_ids`
- `gold_span_ids`
- `final_query`
- `retrieval_attempts`
- `retrieval_ranked_chunk_ids`
- `retrieved_chunk_ids`
- `citation_chunk_ids`
- `decision_correct`
- `evidence_relevant`
- `no_unsupported_claims`
- `clear`
- `citations_useful`
- `reviewer_notes`

## Eval Reports

Eval reports are not eval runs. They live in a separate namespace:

```text
outputs/evals/reports/<report_id>/
  manifest.json
  report.md
```

Examples:

- experiment summaries
- aggregate comparison notes

Rules:

- reports should not be mixed into the eval run namespace
- the UI should list reports separately from eval runs
- reports should have explicit metadata rather than relying on filename or file mtime heuristics

Required files:

- `manifest.json`
- `report.md`

### Eval Report `manifest.json` Schema

Required keys:

- `report_id`: string
- `created_at`: string
- `report_type`: `ReportType`
- `title`: string
- `related_run_ids`: array of strings

Optional keys:

- `domain`: string
- `split`: string
- `subset_label`: string
- `notes`: string

Rules for report metadata:

- `created_at` is the canonical timestamp for sorting and display
- `title` is the canonical UI label
- `report_type` should be a stable machine-readable value such as `experiment_summary`
- `related_run_ids` should list the eval runs the report summarizes or compares

## Standalone Runs

Each standalone `run` command owns one directory:

```text
outputs/runs/<run_id>/
  manifest.json
  result.json
  trace.jsonl
```

Required files:

- `manifest.json`
- `result.json`
- `trace.jsonl`

Rules:

- `result.json` should store the normalized final output returned by the runtime
- `trace.jsonl` should store the raw runtime event stream for that run
- standalone run execution should write `trace.jsonl` directly in place under `outputs/runs/<run_id>/`
- the UI should treat a standalone run as valid only when all three files are present

### Standalone Run `manifest.json` Schema

Required keys:

- `run_id`: string
- `created_at`: string
- `example_id`: string
- `domain`: string
- `provider`: object

`provider` required keys:

- `type`: string
- `chat_model`: string or null
- `embedding_type`: string or null
- `embedding_model`: string or null

Optional keys:

- `prompt_version`: string
- `notes`: string

Rules:

- `created_at` is the canonical timestamp for sorting and display
- `provider` is the canonical source for the standalone run provider/model snapshot shown in the UI

Example:

```json
{
  "run_id": "run-7df0f6c3fb11",
  "created_at": "2026-03-20T16:42:11.123456+01:00",
  "example_id": "dmv::1409501a35697e0ce68561e29577b90a::turn_2",
  "domain": "dmv",
  "provider": {
    "type": "ollama",
    "chat_model": "qwen3:8b-q4_K_M",
    "embedding_type": "ollama",
    "embedding_model": "qwen3-embedding:4b-q4_K_M"
  },
  "prompt_version": "v1"
}
```

## Result Shape

### Standalone Run `result.json` Schema

Required keys:

- `example_id`: string
- `latest_user_utterance`: string or null
- `decision`: `Decision`
- `response_text`: string
- `citations`: array of `CitationRecord`
- `confidence_label`: `ResponseConfidence`
- `retrieval_ranked_chunks`: array of `RankedChunkRecordView`
- `retrieved_chunks`: array of `ChunkRecordView`
- `evidence_grade`: object
- `trace_summary`: `TraceSummary`

Optional keys:

- none for v1

`evidence_grade` required keys:

- `verdict`: `EvidenceVerdict`
- `reason`: string
- `missing_information`: array of strings

Rules for retrieval list ordering:

- `retrieval_ranked_chunks` must preserve ranked retrieval order and each entry must carry a non-null `rank`
- `retrieved_chunks` must preserve the evidence-set order used downstream for grading, generation, verbose output, and citation validation

## Discovery Rules

Artifact discovery should be namespace-based, not heuristic-based.

That means:

- eval runs are listed from `outputs/evals/runs/`
- eval reports are listed from `outputs/evals/reports/`
- standalone runs are listed from `outputs/runs/`

The UI should fail closed on malformed directories instead of guessing artifact intent from partial files or names.

## Producer Responsibilities

The producers should write only to the new layout:

- `eval` writes all eval artifacts into `outputs/evals/runs/<eval_run_id>/`
- eval-linked traces are written directly into `outputs/evals/runs/<eval_run_id>/traces/`
- experiment and other report writers write `manifest.json` plus `report.md` into `outputs/evals/reports/<report_id>/`
- `run` writes standalone artifacts into `outputs/runs/<run_id>/`

This layout replaces the current mixed-root approach for generated outputs.

## Full-Switch Scope

The layout migration is a full-switch change, not a UI-only change.

It must update all consumers and defaults that currently assume the old layout, including:

- runtime and evaluation writers
- CLI review and inspection commands
- settings and runtime default output paths
- experiment and report writers
- project docs and walkthrough examples
- tests and fixtures that assert artifact locations or filenames

The repository should not ship in a mixed state where some commands write the new layout while other commands still read the old layout.

## Trace Write Model

Trace writes should go directly to final artifact paths.

That means:

- `run` allocates `<run_id>` and `outputs/runs/<run_id>/` before runtime execution starts
- `eval` allocates `<eval_run_id>` and its per-example trace destinations before each example run starts
- the runtime writes trace events in place to those final destinations
- the system should not stage traces in a global trace directory and later copy or move them into artifact directories

The runtime contract should therefore support externally assigned artifact identity and final trace destinations.

## Legacy Policy

Legacy generated artifacts are out of scope.

The plan assumes:

- old generated artifacts can be deleted
- the pipeline can be rerun to regenerate fresh artifacts under the new structure
- the Workbench will support only the new layout

## Implementation Order

1. update `run`, `eval`, and report writers to emit the new layout
2. update CLI review and inspection commands to read the new layout
3. update settings and runtime defaults to point at the new namespaces
4. update docs, walkthroughs, tests, and fixtures to the new layout
5. delete old generated artifacts
6. regenerate fresh runs and evals
7. build the UI against the new namespaces
