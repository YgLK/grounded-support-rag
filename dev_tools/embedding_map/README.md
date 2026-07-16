# Embedding Map

Diagnostic tool for inspecting the SupportGraph pgvector embedding space. This is
dev-only tooling, not part of the product CLI.

## What It Does

- Exports indexed chunk embeddings and metadata from Postgres/pgvector.
- Computes PCA and UMAP 2D projections.
- Writes interactive Plotly HTML maps.
- Writes a small summary with path-prefix purity and suspicious nearest neighbors.

UMAP and PCA plots are diagnostic views, not retrieval metrics. Use them to spot
embedding/corpus pathologies; judge retrieval with recall, MRR, NDCG, citations,
required points, and failure labels.

## Run

```bash
uv run python -m dev_tools.embedding_map.embedding_map run \
  --config-file support_graph.kubernetes.toml \
  --domain kubernetes
```

With an eval overlay:

```bash
uv run python -m dev_tools.embedding_map.embedding_map run \
  --config-file support_graph.kubernetes.toml \
  --domain kubernetes \
  --eval-run-id 20260629-002315-kubernetes-smoke \
  --color-by overlay_status
```

Step-by-step:

```bash
uv run python dev_tools/embedding_map/export_embeddings.py \
  --config-file support_graph.kubernetes.toml \
  --domain kubernetes \
  --output-dir outputs/dev_tools/embedding_map/manual

uv run python dev_tools/embedding_map/reduce_embeddings.py \
  --output-dir outputs/dev_tools/embedding_map/manual

uv run python dev_tools/embedding_map/plot_embeddings.py \
  --output-dir outputs/dev_tools/embedding_map/manual
```

## Inputs

- `--config-file`: defaults to `support_graph.kubernetes.toml`.
- `--domain`: defaults to `kubernetes`.
- `--sample-size`: optional deterministic even sample.
- `--doc-prefix`: optional repeatable filter for `doc_id` prefixes.
- `--eval-run-id`: optional eval run overlay from `outputs/evals/runs/<run_id>`.
- `--color-by`: `path_prefix`, `doc_id`, or `overlay_status`.

## Outputs

Outputs go under `outputs/dev_tools/embedding_map/<run_id>/` by default:

- `embeddings.jsonl` (heavy; includes 1536-dim vectors)
- `plot-data.jsonl` (lightweight plot cache; metadata + text, no vectors)
- `projection-pca.csv`
- `projection-umap.csv`
- `embedding-map-pca.html`
- `embedding-map-umap.html`
- `summary.md`

Generated outputs should not be committed by default.

## Iterating On The Frontend

The HTML maps only need projection coords + metadata/text, never the raw
vectors. The first `plot` run builds `plot-data.jsonl` (a slim cache) from
`embeddings.jsonl`; subsequent runs read that cache, so re-rendering after a
frontend tweak is fast and skips the O(n^2) summary diagnostics.

```bash
# Fast: re-render from the cache (no vectors loaded, no summary recompute)
uv run python dev_tools/embedding_map/plot_embeddings.py \
  --output-dir outputs/dev_tools/embedding_map/<run_id>

# Rebuild the cache and recompute summary.md from embeddings.jsonl
uv run python dev_tools/embedding_map/plot_embeddings.py \
  --output-dir outputs/dev_tools/embedding_map/<run_id> \
  --refresh-cache
```

For a ~10k-chunk run this drops re-render time from minutes to a couple of
seconds (234 MB -> ~18 MB read).

## Reading The Plots

Useful signs:

- Kubernetes topics cluster by path prefix.
- Same-doc chunks mostly stay near each other.
- Known eval evidence sits near related docs.

Suspicious signs:

- unrelated path prefixes are nearest neighbors.
- short title-like chunks dominate clusters.
- same-topic docs split into isolated islands without a content reason.
- eval retrieved chunks sit far from expected or acceptable sources.

PCA is a linear baseline. UMAP with cosine distance is the default visual map.
t-SNE is intentionally excluded from v1 because it is slower and easier to
over-interpret.
