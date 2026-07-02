# Eval Expansion and Variance Attribution Tooling

> Sources: SupportGraph local codebase + plan, 2026-06-30
> Raw: [Eval Expansion and Variance Attribution Tooling](../../raw/supportgraph/2026-06-30-eval-expansion-variance-tooling.md)

## Overview

The Smoke-10 Generation Variance finding showed that single-run `incomplete_answer` counts over unchanged retrieval are generation variance, not a retrieval regression, but n=10 could not separate sampling noise from generation non-determinism. A set of CLI tools now supports validating a larger curated corpus-grounded eval set against the pinned chunk corpus and running a repeated-run variance study that asserts retrieval determinism and attributes the residual variance.

## New CLI Commands

- `validate-eval-examples` checks declared doc/span IDs exist in the chunk corpus, each required-point alias group has at least one alias whose tokens appear in the source span text, no duplicate `example_id`, and `answer_type` is valid. Errors gate promotion.
- `promote-eval-examples` merges verified candidates into `expanded.jsonl`, keeping existing rows and appending new `example_id`s, after running the validator.
- `eval-variance` runs the eval harness K times on the same subset/config/index, asserts retrieval signatures are identical across runs, and produces a report with per-metric mean ± 95% bootstrap CI, between-run std, n=10 vs n=50 sampling-noise CI width comparison, a recommended-K from the pilot between-run std, and an explicit sampling-noise-vs-generation-variance attribution.
- `model-ab-compatibility` is the blocking prerequisite for a model A/B: it confirms a candidate chat model supports `with_structured_output(..., method="json_schema")` by asserting the fallback-event count is ~0 on a small smoke run.

## Eval Subset Tiers

A new `EvalSubset.EXPANDED` tier sits between `SMOKE` and `FROZEN_EXPERIMENT` and loads from `data/eval_subsets/<domain>/expanded.jsonl` like `smoke.jsonl`. The curated 50-example candidate set covers definitions, procedures, and diagnoses across ConfigMap, Secret, Namespace, Ingress, StatefulSet, DaemonSet, Job/CronJob, HPA, PDB, PV/PVC, StorageClass, RBAC, probes, init containers, taints/tolerations, affinity, resource limits, QoS, rolling updates, rollbacks, scaling, debugging, CRDs, Operators, Helm, kubeadm, CNI, DNS, and troubleshooting scenarios.

## Candidate Validation and Promotion

Candidate files under `data/eval_subsets/<domain>/_candidates/` are staging artifacts for review. They are not used by `eval --subset expanded` until promoted into `data/eval_subsets/<domain>/expanded.jsonl`.

`validate-eval-examples` is a static quality gate, not a RAG performance run. It checks that each candidate is internally consistent and grounded in the local chunk corpus: unique `example_id`, valid `answer_type`, cited `doc_id` and `span_id` exist in `data/derived/chunks/<domain>.jsonl`, and every required-point alias group has at least one phrase whose tokens appear in the cited source span text. This catches stale span IDs and ungrounded labels before they become benchmark data.

`promote-eval-examples` reruns the validator, then appends only missing `example_id`s from the candidate file into `expanded.jsonl`. Promotion is the step that turns reviewed candidate rows into the real expanded eval set used by `eval --subset expanded` and `eval-variance --subset expanded`.

## Determinism-Diagnostic Knobs

`RuntimeConfig` / `RuntimeFileConfig` / `ProviderConfigLike` gained `chat_temperature` (default 0.0, preserved), `chat_seed` (default None), `openrouter_provider_order` (default None → floating routing), and `openrouter_allow_fallbacks` (default None). These exist to run a diagnostic determinism sweep (pin a single OpenRouter provider, set a seed, raise temperature) and are documented as diagnostic-only; the product relies on robustness, not a pinned RNG. OpenRouter provider routing is passed via `extra_body={"provider": {"order": [...], "allow_fallback": bool}}` through langchain-openai.

## Variance Attribution Method

The `eval-variance` harness disentangles two variance sources the single-run Smoke-10 baseline confounded:

- **Sampling noise** is fixed by a larger set and quantified by a bootstrap CI over examples from a single run.
- **Generation non-determinism** is exposed by repeated runs on the same set and quantified by the std of a metric across runs.

The retrieval-determinism check compares per-example retrieval signatures across runs; if they differ, the attribution is marked INVALID and the report says to investigate retrieval non-determinism before drawing generation-variance conclusions. The recommended-K computation uses `K = (z * sigma / desired_half_width) ** 2` from the pilot between-run std of `required_points_covered`.

## See Also

- [Kubernetes Baseline](kubernetes-baseline.md)
