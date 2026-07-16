# Retrieval Ablation

Dev-only retrieval diagnostic for measuring what each retrieval component adds.
This does not run answer generation or LLM judging.

## Run

```bash
uv run python -m dev_tools.retrieval_ablation.retrieval_ablation run \
  --config-file support_graph.kubernetes.toml \
  --domain kubernetes \
  --subset smoke
```

Useful variants:

```bash
uv run python -m dev_tools.retrieval_ablation.retrieval_ablation run \
  --config-file support_graph.kubernetes.toml \
  --domain kubernetes \
  --subset smoke \
  --top-k 10 \
  --run-id smoke-top10
```

## Modes

- `dense_only`: pgvector similarity only, no rerank.
- `keyword_only`: BM25 only, no rerank.
- `hybrid_no_rerank`: dense + BM25 merge, no rerank.
- `hybrid_rerank`: dense + BM25 merge with current heuristic rerank.

## Outputs

Outputs go under `outputs/dev_tools/retrieval_ablation/<run_id>/`:

- `manifest.json`
- `per_example.jsonl`
- `metrics.json`
- `comparison.csv`
- `summary.md`

Generated outputs should not be committed.

## Reading Results

- dense strong alone: embeddings are doing real retrieval work.
- keyword strong, dense weak: vector search is carrying less than expected.
- hybrid better than both: dense and lexical retrieval are complementary.
- hybrid no-rerank weak, hybrid rerank strong: reranker is carrying quality.
- all modes miss: likely corpus, chunking, query, or eval-label issue.

This is a retrieval diagnostic, not final answer quality. Generation still needs
required-points, citations, faithfulness, and failure-label checks.
