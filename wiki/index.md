# Knowledge Base Index

## supportgraph

SupportGraph run notes and baseline findings.

| Article | Summary | Updated |
|---------|---------|---------|
| [Kubernetes Baseline](supportgraph/kubernetes-baseline.md) | Kubernetes corpus/index/eval baseline; canonical Smoke-10 run `20260629-002315-kubernetes-smoke` clears incomplete answers with one PVC weak-citation failure. A same-config verification re-run confirmed retrieval is deterministic while generation completeness varies (incomplete_answer reappears on 3 examples when the LLM omits a retrieved required point). | 2026-06-29 |
| [Eval Expansion and Variance Attribution Tooling](supportgraph/eval-expansion-variance-tooling.md) | CLI tools for corpus-grounded eval authoring (draft/validate/promote), a repeated-run variance attribution study (eval-variance), a model A/B compatibility gate, an `EvalSubset.EXPANDED` tier, and determinism-diagnostic config knobs. 39 new tests; 188 passing. | 2026-06-30 |
| [ty Typecheck Cleanup Patterns](supportgraph/ty-typecheck-cleanup-patterns.md) | Memory aid for common ty failures in SupportGraph: mutable protocol members, `TypedDict` vs mutable `dict`, partial test fixtures, stale call signatures, erased async return types, and the preferred fixes. | 2026-07-02 |
