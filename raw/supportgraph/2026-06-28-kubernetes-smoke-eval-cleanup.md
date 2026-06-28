# Kubernetes Smoke Eval Cleanup

> Source: Local SupportGraph eval artifacts under `outputs/evals/runs/20260628-224937-kubernetes-smoke`
> Collected: 2026-06-28
> Published: Unknown

Context:
- Eval run: `20260628-224937-kubernetes-smoke`
- This was a baseline interpretation update after the RAG triad rubric changes.
- No index rebuild was needed.
- No retrieval tuning was done.

Before:
- Pods showed as a section mismatch after `_index` normalization.
- Services showed as a strict text-overlap miss despite correct source grounding.
- Deployments remained a retrieval miss.

After:
- Pods clears through rubric aliases and acceptable spans.
- Services clears through required-point and acceptable-span grading.
- Deployments is the only remaining smoke failure, labeled `retrieval_miss`.
- Smoke failure count is now `retrieval_miss: 1`.
