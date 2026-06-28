# SupportGraph Experiment Plan

## Purpose

This document defines how SupportGraph experiments should be recorded, compared, and reported.

The goal is to make future evals and experiments:

- reproducible
- easy to compare
- credible as a hiring artifact

## 1. Principles

- Every eval run must write a machine-readable run manifest.
- Every comparison should change one major variable at a time.
- Smoke tests, experiment tests, and full validation runs must be kept separate.
- Follow-up and clarification-style turns must not be mixed into the primary automatic answer-quality score.
- A run without config, metrics, and failure artifacts does not count as a valid experiment.

## 2. Required Run Artifacts

Each experiment run should write to:

`outputs/evals/runs/<run_id>/`

Required files:

- `manifest.json`
- `metrics.json`
- `failures.jsonl`
- `predictions.jsonl`
- `summary.md`

Optional files:

- `manual_review.csv`
- `retrieval_examples.jsonl`
- `trace_index.json`
- `traces/<trace_file>.jsonl`

Recommended `run_id` format:

`YYYYMMDD-HHMMSS-<short-slug>`

Example:

`20260318-143000-dmv-history-retry`

## 3. Run Manifest Schema

Each run must capture enough metadata to reproduce the result later.

Suggested schema:

```json
{
  "run_id": "20260318-143000-dmv-history-retry",
  "created_at": "2026-03-18T14:30:00+01:00",
  "git_commit": "abc1234",
  "git_dirty": false,
  "dataset_root": "/Users/yglk/coding/grounded-support-rag/multidoc2dial",
  "domains": ["dmv"],
  "split": "validation",
  "eval_subset": "frozen_experiment",
  "target_modes": ["answer"],
  "provider": {
    "type": "ollama",
    "base_url": "http://localhost:11434",
    "chat_model": "your-chat-model",
    "embedding_model": "your-embedding-model"
  },
  "chunking": {
    "strategy": "section_with_deterministic_subchunks",
    "max_tokens_per_chunk": 512
  },
  "retrieval": {
    "top_k": 5,
    "max_attempts": 2,
    "use_history": true
  },
  "graph": {
    "enable_retry": true,
    "decision_policy_version": "v1"
  },
  "prompt": {
    "answer_prompt_version": "v1",
    "query_prompt_version": "v1",
    "evidence_prompt_version": "v1"
  },
  "notes": "First history-aware retry run on DMV frozen subset"
}
```

Minimum required fields:

- `run_id`
- `created_at`
- `git_commit`
- `git_dirty`
- `domains`
- `split`
- `eval_subset`
- `target_modes`
- model/provider info
- retrieval config
- chunking config
- prompt versions

## 4. Eval Subsets

SupportGraph should use three evaluation tiers.

### 4.1 Smoke subset

Purpose:

- catch breakages quickly during development

Requirements:

- small
- deterministic
- committed to the repo

Recommended size:

- 25 `dmv` validation examples with `target_mode = answer`

### 4.2 Frozen experiment subset

Purpose:

- compare system variants quickly but fairly

Requirements:

- deterministic
- committed to the repo
- unchanged once chosen

Recommended size:

- 200 `dmv` validation examples with `target_mode = answer`

### 4.3 Full validation subset

Purpose:

- headline metrics

Requirements:

- all `dmv` validation examples with `target_mode = answer`
- separate reporting for `target_mode = follow_up`

## 5. Experiment Matrix

The MVP should support a small, clear experiment story.

Recommended comparison ladder:

### A0 Baseline retrieval-answer path

- section-aware chunks
- latest-user-focused query
- single retrieval attempt
- no retry

### A1 Add transcript-aware query construction

Changes from A0:

- use recent transcript context in query preparation

### A2 Add bounded retry path

Changes from A1:

- add evidence grading
- add one retrieval retry

### A3 Add deterministic subchunking for oversized sections

Changes from A2:

- split only large sections
- keep the rest section-sized

Rules:

- compare adjacent variants first
- do not change models and prompts in the same experiment unless the run is explicitly labeled as a prompt/model experiment
- if multiple variables change, the run belongs in exploration, not in the main experiment story

## 6. Metrics to Report

For `target_mode = answer`:

- `Doc Recall@3`
- `Span Recall@5`
- `MRR@5`
- `ROUGE-L`
- token-level `F1`
- citation coverage
- end-to-end automatic success rate

For `target_mode = follow_up`:

- `Doc Recall@3`
- `Span Recall@5`
- decision distribution
- manual review summary

## 7. Failure Analysis

Each run must write `failures.jsonl`.

Each failure record should include:

- `example_id`
- `target_mode`
- `gold_doc_ids`
- `gold_span_ids`
- retrieved chunks
- final decision
- response text
- metric breakdown
- short failure label

Suggested failure labels:

- `wrong_doc`
- `right_doc_wrong_section`
- `missed_history`
- `unsupported_answer`
- `weak_citations`
- `bad_clarification`
- `abstained_with_evidence`

## 8. Manual Review Rubric

Manual review is required for `target_mode = follow_up` and is useful for sampled answer failures.

Use a simple rubric with 0 or 1 per item:

- `decision_correct`
- `evidence_relevant`
- `no_unsupported_claims`
- `question_or_answer_is_clear`
- `citations_are_useful`

Recommended review sample sizes:

- 25 follow-up predictions per major run
- 10 answer failures per major run

## 9. Summary Template

Each run should produce a short `summary.md` with:

- what changed
- which subset was used
- key metrics
- biggest wins
- biggest regressions
- 3 to 5 representative failure patterns
- recommendation: keep, revert, or investigate

## 10. Portfolio-Ready Experiment Story

For a strong mid/senior presentation, the final README or case study should be able to answer:

- What was the baseline?
- What did history awareness change?
- What did retry change?
- What still failed after both?
- Which tradeoffs were worth the added complexity?

If SupportGraph can answer those questions with saved manifests, metrics, and example failures, the experiment story is strong.
