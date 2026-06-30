# Eval Expansion and Variance Attribution Tooling

> Sources: SupportGraph local codebase + plan, 2026-06-30
> Raw: [Eval Expansion and Variance Attribution Tooling](../../raw/supportgraph/2026-06-30-eval-expansion-variance-tooling.md)

## Overview

The Smoke-10 Generation Variance finding showed that single-run `incomplete_answer` counts over unchanged retrieval are generation variance, not a retrieval regression, but n=10 could not separate sampling noise from generation non-determinism. A set of CLI tools now supports authoring a larger corpus-grounded eval set, validating labels against the pinned chunk corpus, and running a repeated-run variance study that asserts retrieval determinism and attributes the residual variance.

## New CLI Commands

- `draft-eval-examples` runs real retrieval per seed topic, picks the best retrieved chunk as the gold span source, and asks the chat model to draft a reference answer plus required-point alias groups constrained to the retrieved chunk text. Each candidate carries a `provenance` block (seed, retrieved chunk_ids, gold chunk/span IDs, source span text snippet) so labels are anchored to the pinned corpus. Output is appended idempotently to a candidates file.
- `validate-eval-examples` checks declared doc/span IDs exist in the chunk corpus, each required-point alias group has at least one alias whose tokens appear in the source span text, no duplicate `example_id`, and `answer_type` is valid. Errors gate promotion.
- `promote-eval-examples` merges verified candidates into `expanded.jsonl`, keeping existing rows and appending new `example_id`s, after running the validator.
- `eval-variance` runs the eval harness K times on the same subset/config/index, asserts retrieval signatures are identical across runs, and produces a report with per-metric mean ± 95% bootstrap CI, between-run std, n=10 vs n=50 sampling-noise CI width comparison, a recommended-K from the pilot between-run std, and an explicit sampling-noise-vs-generation-variance attribution.
- `model-ab-compatibility` is the blocking prerequisite for a model A/B: it confirms a candidate chat model supports `with_structured_output(..., method="json_schema")` by asserting the fallback-event count is ~0 on a small smoke run.

## Eval Subset Tiers

A new `EvalSubset.EXPANDED` tier sits between `SMOKE` and `FROZEN_EXPERIMENT` and loads from `data/eval_subsets/<domain>/expanded.jsonl` like `smoke.jsonl`. A 50-topic seed file covers definitions, procedures, and diagnoses across ConfigMap, Secret, Namespace, Ingress, StatefulSet, DaemonSet, Job/CronJob, HPA, PDB, PV/PVC, StorageClass, RBAC, probes, init containers, taints/tolerations, affinity, resource limits, QoS, rolling updates, rollbacks, scaling, debugging, CRDs, Operators, Helm, kubeadm, CNI, DNS, and troubleshooting scenarios.

## Determinism-Diagnostic Knobs

`RuntimeConfig` / `RuntimeFileConfig` / `ProviderConfigLike` gained `chat_temperature` (default 0.0, preserved), `chat_seed` (default None), `openrouter_provider_order` (default None → floating routing), and `openrouter_allow_fallbacks` (default None). These exist to run a diagnostic determinism sweep (pin a single OpenRouter provider, set a seed, raise temperature) and are documented as diagnostic-only; the product relies on robustness, not a pinned RNG. OpenRouter provider routing is passed via `extra_body={"provider": {"order": [...], "allow_fallback": bool}}` through langchain-openai.

## Variance Attribution Method

The `eval-variance` harness disentangles two variance sources the single-run Smoke-10 baseline confounded:

- **Sampling noise** is fixed by a larger set and quantified by a bootstrap CI over examples from a single run.
- **Generation non-determinism** is exposed by repeated runs on the same set and quantified by the std of a metric across runs.

The retrieval-determinism check compares per-example retrieval signatures across runs; if they differ, the attribution is marked INVALID and the report says to investigate retrieval non-determinism before drawing generation-variance conclusions. The recommended-K computation uses `K = (z * sigma / desired_half_width) ** 2` from the pilot between-run std of `required_points_covered`.

## See Also

- [Kubernetes Baseline](kubernetes-baseline.md)
