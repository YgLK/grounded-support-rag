# Kubernetes Baseline

> Sources: SupportGraph local eval artifacts, 2026-06-28
> Raw: [Kubernetes Baseline Eval](../../raw/supportgraph/2026-06-28-kubernetes-baseline-eval.md); [Kubernetes Normalized Smoke Eval](../../raw/supportgraph/2026-06-28-kubernetes-normalized-smoke-eval.md); [Kubernetes Smoke Eval Cleanup](../../raw/supportgraph/2026-06-28-kubernetes-smoke-eval-cleanup.md); [Kubernetes Smoke-10 Retrieval Baseline](../../raw/supportgraph/2026-06-28-kubernetes-smoke10-retrieval-baseline.md)

## Overview

The first end-to-end Kubernetes baseline ran against a pinned `kubernetes/website` docs snapshot, built a full 10,379-chunk index, and completed the 3-example Kubernetes smoke eval. A follow-up run after normalizing Hugo `_index.md` pages to website-style doc IDs improved document recall, confirming that the Pods failure was partly corpus-ID plumbing rather than pure retrieval quality. The current flagship baseline is a 10-example Kubernetes troubleshooting smoke eval. Retrieval is now stable: all examples hit expected or acceptable docs by top 3 and spans by top 5. Remaining failures are answer-coverage gaps, not retrieval misses.

## Baseline Run

- Docs ref: `main`, resolved to `2b654de67188ec248e641afc3edd33a5b2c982b6`
- Corpus: 1669 docs, 10379 chunks
- Index: `support_graph_kubernetes`, 10379 rows verified
- Embeddings: `openrouter / openai/text-embedding-3-small`
- Chat: `openrouter / openai/gpt-oss-120b:nitro`
- Eval run: `20260628-141342-kubernetes-smoke`
- Smoke subset: 3 answer examples

## Results

After `_index` normalization:

- Eval run: `20260628-213531-kubernetes-smoke`
- Doc Recall@3: 0.667
- Span Recall@5: 0.333
- MRR@5: 0.333
- ROUGE-L: 0.261
- Token F1: 0.291
- Citation coverage: 0.333
- End-to-end success: 0.000
- Failure labels: `right_doc_wrong_section` 1, `wrong_doc` 1, `unsupported_answer` 1

After RAG triad rubric cleanup:

- Eval run: `20260628-224937-kubernetes-smoke`
- Smoke subset: 3 answer examples
- Failure labels: `retrieval_miss` 1
- Cleared examples: `pods`, `services`
- Remaining failure: `deployments`
- Index rebuild: not needed

After hybrid retrieval/rerank fix and Smoke-10 expansion:

- Eval run: `20260628-233824-kubernetes-smoke`
- Smoke subset: 10 answer examples, including 7 troubleshooting scenarios
- Doc Recall@1: 0.700
- Doc Recall@3: 1.000
- Doc Recall@5: 1.000
- Span Recall@5: 1.000
- MRR@5: 0.833
- Citation coverage: 0.900
- Required points covered: 0.808
- NDCG@5: 0.993, max per-example NDCG: 1.000
- Failure labels: `incomplete_answer` 5
- Remaining gap: generation often gives useful but partial troubleshooting answers
- Index rebuild: not needed

Original baseline:

- Doc Recall@3: 0.333
- Span Recall@5: 0.333
- MRR@5: 0.167
- ROUGE-L: 0.240
- Token F1: 0.295
- Citation coverage: 0.333
- End-to-end success: 0.000
- Failure labels: `wrong_doc` 2, `unsupported_answer` 1

## Findings

The Pods example improved from `wrong_doc` to `right_doc_wrong_section` after `_index` normalization. The fixed run retrieves `concepts/workloads/pods` at rank 2, with useful evidence in `what-is-a-pod` while the legacy gold expected `overview`. The current rubric treats that as acceptable evidence when required points are covered, so Pods is no longer a smoke failure.

The Deployments retrieval miss was not an index problem. BM25 already found `concepts/workloads/controllers/deployment`; the merged keyword hits lacked comparable scores, so rerank sorted them behind dense-only tutorial chunks. The fix gives keyword-only hits dense-scale pseudo-distances and adds doc-path overlap to rerank. After that, Deployment hits expected/acceptable evidence; its remaining failure is answer completeness because generated answers often mention Pods but omit ReplicaSets / rollout wording.

The Services example retrieved and cited `concepts/services-networking/service` with doc recall, span recall, and citation coverage all at 1.0. The current rubric accepts the answer through required-point and acceptable-span grading, removing the old strict text-overlap false negative.

## Operational Notes

Local Ollama could not serve embeddings because the installed binary crashed during MLX/Metal initialization before command handling. OpenRouter `qwen/qwen3-embedding-8b` produced a 4096-dimensional probe vector but timed out during indexing on a small generated kubeadm docs chunk. OpenRouter `openai/text-embedding-3-small` embedded that chunk and completed the full index.

## Next Checks

- Improve answer generation for troubleshooting completeness, especially forcing cited docs' concrete checklist items into the final answer.
- Consider a stricter generation prompt or lightweight post-generation checklist for required points.
- Keep reindex off the table unless future artifacts show missing or wrong chunk metadata.

## RAG Triad Eval Upgrade (2026-06-28)

The smoke eval was upgraded from exact gold-doc/text matching to a RAG triad-style evaluation, addressing both next checks above. The smoke set now declares `expected_sources`, `acceptable_sources`, `required_points`, `forbidden_claims`, and `answer_type` per example, while keeping `gold_doc_ids`/`gold_span_ids` for legacy-metric continuity.

- Pods: `concepts/workloads/pods` is expected; `reference/glossary/pod` is acceptable. Required points cover smallest deployable object, one-or-more containers, shared networking/storage. This resolves the section-mismatch false negative: an answer grounded in the glossary or a non-`overview` section now passes when required points are covered.
- Services: accepted on required-points + source grounding rather than exact text overlap, fixing the semantic-correctness-but-text-miss failure.
- Deployments: `concepts/workloads/controllers/deployment` stays expected with `tasks/run-application/run-stateless-application-deployment` acceptable; the graded retrieval metrics (hit@k, precision@k, graded MRR, NDCG) now surface the retrieval miss clearly instead of collapsing it into `wrong_doc`.

New metrics emitted: `context_relevance`, `faithfulness`, `answer_relevance`, `answer_correctness` (with `reference_similarity` sub-bucket preserving ROUGE/F1/SacreBLEU), `required_points_covered`, and graded retrieval (`hit@k`, `precision@k`, `graded_mrr_at_k`, `ndcg_at_k` over expected=2/acceptable=1 graded relevance). Failure labels split `unsupported_answer` into `retrieval_miss`, `right_source_wrong_section`, `unfaithful_answer`, `incomplete_answer`, `irrelevant_answer`, and `weak_citations`. An optional LLM judge path (`RAGTriadJudge`) is wired but disabled by default; deterministic v1 proxies (citation validity, forbidden-claim absence, required-point token overlap) gate the default test suite without a live LLM.

The new smoke summary explains whether failures are retrieval, grounding, relevance, or correctness failures, and the manual review CSV carries the new RAG review columns (expected/acceptable sources, required points, per-example triad scores, judge rationale).

## Smoke-10 Flagship Baseline (2026-06-28)

The Kubernetes smoke set now covers `pods`, `deployments`, `services`, plus troubleshooting prompts for `CrashLoopBackOff`, `ImagePullBackOff`, `FailedScheduling`, stuck Deployment rollouts, Service DNS, PVC pending, and kubectl connectivity. The data test verifies all declared gold/acceptable doc IDs and span IDs exist in `data/derived/chunks/kubernetes.jsonl`.

The current Smoke-10 run with `support_graph.kubernetes.toml` produced retrieval-stable metrics: Doc Recall@3 `1.000`, Span Recall@5 `1.000`, and graded Hit@5 `1.000`. NDCG is now bounded (`0.993` average, `1.000` max), after deduplicating repeated doc IDs and preventing acceptable alternates from adding gain beyond the expected-source ideal. The only failure bucket is `incomplete_answer`; remaining failures are generation completeness or abstain issues, not retrieval misses.
