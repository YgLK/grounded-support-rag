Source URL: local eval artifacts
Collected: 2026-06-29
Published: 2026-06-29

# Kubernetes Smoke-10 Generation Variance

Canonical headline run: `outputs/evals/runs/20260629-002315-kubernetes-smoke`

Verification run (same config, fresh generation): `outputs/evals/runs/20260629-131400-kubernetes-smoke`

Config: `support_graph.kubernetes.toml` (identical between both runs; only `created_at`, `dataset_root`, and `notes` differ in the manifests).

## What changed

No code, prompt, retrieval, or config change. The verification run re-ran the same Smoke-10 eval against the same restored index to confirm the P1 dump-restore demo path. The only difference is a fresh LLM generation pass.

## Metrics comparison

| metric | headline (002315) | verify (131400) |
|---|---|---|
| Doc Recall@1 | 0.700 | 0.700 |
| Doc Recall@3 | 0.900 | 0.900 |
| Doc Recall@5 | 0.900 | 0.900 |
| Span Recall@5 | 1.000 | 1.000 |
| MRR@5 | 0.800 | 0.800 |
| ROUGE-L | 0.161 | 0.149 |
| Token F1 | 0.204 | 0.203 |
| Citation Coverage | 0.900 | 1.000 |
| Required points covered | 1.000 | 0.900 |
| Hit@5 | 1.000 | 1.000 |
| Graded MRR@5 | 0.850 | 0.850 |
| NDCG@5 | 0.974 | 0.946 |
| Failure labels | `weak_citations` 1 | `incomplete_answer` 3 |

Retrieval metrics are identical (deterministic over the same index). Generation and RAG-grounding metrics vary with the LLM pass.

## Per-example failure movement

| example | RPC headline | RPC verify | failure headline | failure verify |
|---|---|---|---|---|
| deployments | 1.0 | 0.667 | - | incomplete_answer |
| troubleshooting-deployment-rollout | 1.0 | 0.667 | - | incomplete_answer |
| troubleshooting-failedscheduling | 1.0 | 0.667 | - | incomplete_answer |
| troubleshooting-pvc-pending | 1.0 | 1.0 | weak_citations | - |
| (other 6 examples) | 1.0 | 1.0 | - | - |

Three examples flipped to `incomplete_answer` (required_points_covered dropped from 1.0 to 0.667, i.e. 2 of 3 required points covered). The PVC example flipped from `weak_citations` to passing (citation coverage rose to 1.0). The remaining six examples were stable.

## Root cause

The three `incomplete_answer` failures are generation omissions, not retrieval misses. In each case the expected evidence was retrieved but the generated answer omitted one required point:

- `deployments`: `concepts/workloads/controllers/deployment` was retrieved at rank 2, but the answer described Pods lifecycle and scaling without mentioning ReplicaSets (required point 2). The headline run's answer for the same example mentioned ReplicaSets and passed.
- `troubleshooting-failedscheduling`: debug docs (`tasks/debug/debug-application/debug-pods`, `manage-resources-containers`) were retrieved and cited, but the answer covered scheduler events and resource requests while omitting a third required point.
- `troubleshooting-deployment-rollout`: `concepts/workloads/controllers/deployment` and `tasks/run-application/update-deployment-rolling` were retrieved and cited, but the answer covered rollout history and stalled/failed detection while omitting a third required point.

The required-points grader is deterministic (token-overlap aliases), so the variance is entirely in the LLM's generated answer content. `openai/gpt-oss-120b:nitro` is non-deterministic across calls.

## Interpretation

- Retrieval is stable and reproducible: identical Doc Recall, Span Recall, MRR, Hit@5, and Graded MRR across runs over the same index.
- Generation is the variance source: ROUGE-L, F1, required-points coverage, and failure labels move with the LLM pass. The incomplete_answer bucket is fragile against generation omissions even when evidence is retrieved.
- The PVC `weak_citations` failure is the opposite case: it is stable across retrieval but its citation-coverage label depends on whether the gold `dynamic-provisioning` doc happens to be cited, which also varies with generation.

## Named follow-up

- Stabilize answer completeness with a stricter generation prompt or a lightweight post-generation required-points checklist that forces cited docs' concrete checklist items into the final answer. This is a generation-side fix; no reindex or retrieval change is warranted.
- Keep `20260629-002315-kubernetes-smoke` as the canonical headline baseline. Treat single-run incomplete_answer counts as generation variance, not a retrieval regression, when retrieval metrics are unchanged.

## Demo trace candidates (from the canonical headline run)

- Success: `kubernetes::pods::turn_2` — decision `answer`, cited `concepts/workloads/pods`, no failure.
- Failure: `kubernetes::troubleshooting-pvc-pending::turn_2` — `weak_citations`, well-grounded answer citing acceptable storage docs, legacy citation coverage expects the gold `dynamic-provisioning` doc.
