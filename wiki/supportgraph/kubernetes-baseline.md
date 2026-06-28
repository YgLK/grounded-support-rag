# Kubernetes Baseline

> Sources: SupportGraph local eval artifacts, 2026-06-28
> Raw: [Kubernetes Baseline Eval](../../raw/supportgraph/2026-06-28-kubernetes-baseline-eval.md); [Kubernetes Normalized Smoke Eval](../../raw/supportgraph/2026-06-28-kubernetes-normalized-smoke-eval.md)

## Overview

The first end-to-end Kubernetes baseline ran against a pinned `kubernetes/website` docs snapshot, built a full 10,379-chunk index, and completed the 3-example Kubernetes smoke eval. A follow-up run after normalizing Hugo `_index.md` pages to website-style doc IDs improved document recall, confirming that the Pods failure was partly corpus-ID plumbing rather than pure retrieval quality. The smoke set still is not a reliable quality gate: remaining failures include a section-level mismatch, a true deployment retrieval miss, and a strict generation/text-overlap miss.

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

The Pods example improved from `wrong_doc` to `right_doc_wrong_section` after `_index` normalization. The fixed run retrieves `concepts/workloads/pods` at rank 2, but the top matching section is `what-is-a-pod` while the smoke gold expects `overview`. The remaining failure is now section-level gold alignment, not doc ID drift.

The Deployments example retrieved broad tutorial and controller overview pages, not `concepts/workloads/controllers/deployment`. The generated answer was mostly correct, but evidence support did not line up with the curated target.

The Services example retrieved and cited `concepts/services-networking/service` with doc recall, span recall, and citation coverage all at 1.0. It still failed end-to-end, so the current text-overlap gate is stricter than semantic correctness for concise Kubernetes support answers.

## Operational Notes

Local Ollama could not serve embeddings because the installed binary crashed during MLX/Metal initialization before command handling. OpenRouter `qwen/qwen3-embedding-8b` produced a 4096-dimensional probe vector but timed out during indexing on a small generated kubeadm docs chunk. OpenRouter `openai/text-embedding-3-small` embedded that chunk and completed the full index.

## Next Checks

- Decide whether Kubernetes smoke gold spans should target semantic sections like `what-is-a-pod` rather than synthetic `overview` sections.
- Add a semantic or judge-backed generation check for short Kubernetes answer variants.

## RAG Triad Eval Upgrade (2026-06-28)

The smoke eval was upgraded from exact gold-doc/text matching to a RAG triad-style evaluation, addressing both next checks above. The smoke set now declares `expected_sources`, `acceptable_sources`, `required_points`, `forbidden_claims`, and `answer_type` per example, while keeping `gold_doc_ids`/`gold_span_ids` for legacy-metric continuity.

- Pods: `concepts/workloads/pods` is expected; `reference/glossary/pod` is acceptable. Required points cover smallest deployable object, one-or-more containers, shared networking/storage. This resolves the section-mismatch false negative: an answer grounded in the glossary or a non-`overview` section now passes when required points are covered.
- Services: accepted on required-points + source grounding rather than exact text overlap, fixing the semantic-correctness-but-text-miss failure.
- Deployments: `concepts/workloads/controllers/deployment` stays expected with `tasks/run-application/run-stateless-application-deployment` acceptable; the graded retrieval metrics (hit@k, precision@k, graded MRR, NDCG) now surface the retrieval miss clearly instead of collapsing it into `wrong_doc`.

New metrics emitted: `context_relevance`, `faithfulness`, `answer_relevance`, `answer_correctness` (with `reference_similarity` sub-bucket preserving ROUGE/F1/SacreBLEU), `required_points_covered`, and graded retrieval (`hit@k`, `precision@k`, `graded_mrr_at_k`, `ndcg_at_k` over expected=2/acceptable=1 graded relevance). Failure labels split `unsupported_answer` into `retrieval_miss`, `right_source_wrong_section`, `unfaithful_answer`, `incomplete_answer`, `irrelevant_answer`, and `weak_citations`. An optional LLM judge path (`RAGTriadJudge`) is wired but disabled by default; deterministic v1 proxies (citation validity, forbidden-claim absence, required-point token overlap) gate the default test suite without a live LLM.

The new smoke summary explains whether failures are retrieval, grounding, relevance, or correctness failures, and the manual review CSV carries the new RAG review columns (expected/acceptable sources, required points, per-example triad scores, judge rationale).
