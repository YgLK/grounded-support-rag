# Kubernetes Baseline

> Sources: SupportGraph local eval artifacts, 2026-06-28
> Raw: [Kubernetes Baseline Eval](../../raw/supportgraph/2026-06-28-kubernetes-baseline-eval.md)

## Overview

The first end-to-end Kubernetes baseline ran against a pinned `kubernetes/website` docs snapshot, built a full 10,379-chunk index, and completed the 3-example Kubernetes smoke eval. The baseline proves the corpus/index/eval path works, but the smoke set is not yet a reliable quality gate: two failures come from expected doc IDs that do not match the fetched Kubernetes website paths, and the only retrieval-success example still misses end-to-end generation success.

## Baseline Run

- Docs ref: `main`, resolved to `2b654de67188ec248e641afc3edd33a5b2c982b6`
- Corpus: 1669 docs, 10379 chunks
- Index: `support_graph_kubernetes`, 10379 rows verified
- Embeddings: `openrouter / openai/text-embedding-3-small`
- Chat: `openrouter / openai/gpt-oss-120b:nitro`
- Eval run: `20260628-141342-kubernetes-smoke`
- Smoke subset: 3 answer examples

## Results

- Doc Recall@3: 0.333
- Span Recall@5: 0.333
- MRR@5: 0.167
- ROUGE-L: 0.240
- Token F1: 0.295
- Citation coverage: 0.333
- End-to-end success: 0.000
- Failure labels: `wrong_doc` 2, `unsupported_answer` 1

## Findings

The Pods example retrieved strong evidence but failed against the curated gold IDs. The fetched docs expose the current page as `concepts/workloads/pods/_index` plus `reference/glossary/pod`; the smoke gold expects `concepts/workloads/pods`. This looks like a gold-ID normalization issue rather than a pure retrieval miss.

The Deployments example retrieved broad tutorial and controller overview pages, not `concepts/workloads/controllers/deployment`. The generated answer was mostly correct, but evidence support did not line up with the curated target.

The Services example retrieved and cited `concepts/services-networking/service` with doc recall, span recall, and citation coverage all at 1.0. It still failed end-to-end, so the current text-overlap gate is stricter than semantic correctness for concise Kubernetes support answers.

## Operational Notes

Local Ollama could not serve embeddings because the installed binary crashed during MLX/Metal initialization before command handling. OpenRouter `qwen/qwen3-embedding-8b` produced a 4096-dimensional probe vector but timed out during indexing on a small generated kubeadm docs chunk. OpenRouter `openai/text-embedding-3-small` embedded that chunk and completed the full index.

## Next Checks

- Normalize Kubernetes smoke gold doc IDs to fetched website doc IDs, especially `_index` pages.
- Re-run the smoke eval after gold-ID normalization before interpreting retrieval quality.
- Add a semantic or judge-backed generation check for short Kubernetes answer variants.
