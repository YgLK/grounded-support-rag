# SupportGraph Roadmap

## Current State

- Kubernetes corpus plumbing is implemented and unit-tested.
- CLI has `fetch-kubernetes-docs` for pinned `kubernetes/website` snapshots.
- Kubernetes documents load from Markdown into SupportGraph document/span records.
- Kubernetes evals use curated subsets under `data/eval_subsets/kubernetes/`.
- Full test suite passes: `uv run pytest`.

## Resume Here

1. Fetch a real Kubernetes docs snapshot.

   ```bash
   uv run grounded-support-rag fetch-kubernetes-docs --ref main --output raw/kubernetes/current --replace
   ```

2. Point local config at the corpus.

   ```toml
   [dataset]
   root = "raw/kubernetes/current"
   enabled_domains = ["kubernetes"]
   ```

3. Build chunks.

   ```bash
   uv run grounded-support-rag build-chunks --domain kubernetes
   ```

4. Index chunks once Postgres/vector config is ready.

   ```bash
   uv run grounded-support-rag index-docs --domain kubernetes
   ```

5. Run smoke eval.

   ```bash
   uv run grounded-support-rag eval --domain kubernetes --subset smoke
   ```

## Acceptance

- Fetch writes `raw/kubernetes/current/manifest.json` with requested ref and resolved SHA.
- Build chunks completes with nonzero Kubernetes documents/chunks.
- Index contains Kubernetes chunks.
- Smoke eval produces an eval run with predictions, retrieval examples, metrics, traces, and manual review CSV.
- If eval gives meaningful Kubernetes baseline findings, update `wiki/`.

## Known Constraints

- Live fetch needs network.
- Index/eval need configured providers and local services.
- Kubernetes has curated eval examples only; do not run dialogue/example generation for it.
- Generated corpus, logs, coverage, and run outputs should stay out of git.
