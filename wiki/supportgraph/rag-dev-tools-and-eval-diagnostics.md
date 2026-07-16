# RAG Dev Tools and Eval Diagnostics

> Sources: SupportGraph planning conversation, 2026-07-08; SupportGraph local retrieval ablation run, 2026-07-08
> Raw: [RAG Dev Tools and Eval Diagnostics Plan](../../raw/supportgraph/2026-07-08-rag-dev-tools-eval-diagnostics-plan.md); [Retrieval Ablation Smoke Run](../../raw/supportgraph/2026-07-08-retrieval-ablation-smoke.md)

## Overview

SupportGraph needs separate diagnostic tools for retrieval and answer generation because they fail in different ways. Retrieval depends on the vector database, embeddings, chunk metadata, and rerank behavior. Generation depends on the chat model, prompt, retrieved evidence, citation selection, and answer completeness. The main eval harness remains the quality gate, while `dev_tools/` is the right home for exploratory diagnostics that are useful but not yet product CLI commands.

## Why Retrieval and Generation Need Separate Diagnostics

Retrieval diagnostics should answer whether the embedding/index layer can place related Kubernetes documentation near each other and retrieve expected or acceptable evidence. Generation diagnostics should answer whether the model used retrieved content faithfully, covered required points, avoided forbidden claims, and cited useful chunks.

Mixing both into a single headline score hides root causes. A run can have stable retrieval and unstable generation completeness, as shown by the Smoke-10 generation-variance finding. Conversely, answer wording can look weak under lexical metrics even when the answer is grounded and covers the intended points.

## Dev Tools Directory Convention

`dev_tools/` stores repo-local diagnostic tooling that is not part of the shipped `grounded-support-rag` CLI. Each tool gets its own directory and README. Generated outputs go under `outputs/dev_tools/<tool>/<run_id>/` and should not be committed by default.

Stable, repeated-use diagnostics can later graduate into the main CLI. Until then, keeping them in `dev_tools/` avoids mixing exploratory analysis code with product workflow code.

## Embedding-Space Diagnostics

The first retrieval diagnostic is `dev_tools/embedding_map/`. It exports pgvector chunk embeddings and metadata, computes PCA and UMAP projections, and writes interactive Plotly HTML maps. UMAP with cosine distance is the default nonlinear view because embedding debugging usually cares about local neighborhoods. PCA remains a cheap linear baseline and sanity check.

Embedding maps are useful because the smoke and expanded eval sets cover only a small slice of the corpus. A corpus-wide map can reveal unrelated docs clustering together, related docs splitting apart, short/title chunks dominating neighborhoods, or path-prefix groups that do not behave coherently.

These plots are diagnostic views, not retrieval metrics. Dimensionality reduction distorts distances. Retrieval judgment still comes from recall, MRR, NDCG, citation behavior, required-points coverage, and failure labels.

## Retrieval Ablation Finding

`dev_tools/retrieval_ablation/` is the higher-signal retrieval diagnostic because it directly compares dense-only, keyword-only, hybrid without rerank, and current hybrid rerank on the same eval examples. The 2026-07-08 Smoke-10 run shows dense retrieval is already doing real work: dense-only reached doc recall@3 0.700, span recall@5 1.000, and NDCG@5 0.862, while keyword-only reached doc recall@3 0.300, span recall@5 0.300, and NDCG@5 0.335.

Hybrid without rerank matched dense-only at top 5 in that run because appended keyword candidates did not enter the returned set before reranking. Current hybrid rerank improved doc recall@3 to 0.800, MRR@5 to 0.845, and NDCG@5 to 0.948, with three rerank rescue cases and one rerank regression. The baseline interpretation is that embeddings are valuable, keyword-only is weak, and rerank is carrying additional ranking quality rather than simply masking a broken dense retriever.

## Generation Metrics Interpretation

`ROUGE-L`, `token_f1`, `exact_match`, and `SacreBLEU` should stay in the `reference_similarity` bucket. They are useful trend indicators, but not primary correctness metrics for open-ended RAG answers because equivalent answers can use different wording.

The stronger default interpretation comes from expected/acceptable source retrieval, citation validity, citation coverage, required-points coverage, forbidden-claim checks, failure labels, and manual review rows. A generation-review dev tool should put lexical metrics next to those deterministic rubric signals so low lexical overlap is not mistaken for an automatically bad answer.

## See Also

- [Kubernetes Baseline](kubernetes-baseline.md)
- [Eval Expansion and Variance Attribution Tooling](eval-expansion-variance-tooling.md)
