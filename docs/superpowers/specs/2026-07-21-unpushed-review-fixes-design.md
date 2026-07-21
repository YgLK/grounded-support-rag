# Unpushed Review Fixes

## Scope

Fix the seven confirmed regressions in `origin/dev..HEAD`. Preserve current CLI commands, hosted-evaluation contracts, and artifact formats. No unrelated refactors or new dependencies.

## Design

- LangSmith run tracing: keep the friendly SupportGraph run ID for output and graph state; pass a separate UUID to the LangSmith root trace.
- Immutable datasets: derive each LangSmith example UUID from the dataset content digest plus source `example_id`, so repeated publication is idempotent while distinct dataset versions cannot collide.
- Experiment latency: aggregate `outputs.trace_summary.latency_ms` across hosted example results into `metrics.latency_ms.average`, retaining the existing classification interface and guardrail.
- Hybrid retrieval without reranking: deterministically interleave dense and keyword candidates, deduplicate by chunk ID, then truncate. Dense-only and keyword-only behavior remains unchanged.
- Evaluation policies: explicitly map every accepted subset to a committed policy. `frozen_experiment` and `full_validation` use the expanded policy; smoke-derived experiments continue using smoke policy.
- Eval notes: store `--notes` in LangSmith experiment metadata when supplied.
- Small UMAP samples: retain the origin case for one point; use the deterministic PCA projection for two points; use UMAP for three or more.

## Error Handling

Existing explicit errors remain. Policy lookup fails only for unknown subsets. No fallback hides missing configuration or malformed inputs.

## Testing

Add one focused regression test per issue and observe each fail before production changes. Then run affected test modules, the full pytest suite, Ruff, type checking, and `git diff --check`.

## Non-goals

- Redesigning hosted evaluation.
- Changing policy thresholds.
- Adding retrieval fusion dependencies.
- Persisting standalone run artifacts.
