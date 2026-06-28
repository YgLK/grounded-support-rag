# Kubernetes Smoke-10 Retrieval Baseline

Source URL: Local SupportGraph eval artifacts
Collected: 2026-06-28
Published: 2026-06-28

Expanded Kubernetes smoke from 3 definition examples to 10 examples: Pods, Deployments, Services, CrashLoopBackOff, ImagePullBackOff, FailedScheduling, stuck Deployment rollout, Service DNS, PVC pending, and kubectl connectivity.

Run `20260628-231545-kubernetes-smoke` used:

- Config: `support_graph.kubernetes.toml`
- Chat: `openrouter / openai/gpt-oss-120b:nitro`
- Embeddings: `openrouter / openai/text-embedding-3-small`
- Index: `support_graph_kubernetes`
- Corpus: 1669 docs, 10379 chunks
- Retrieval: `top_k=5`, `candidate_k=12`, rerank enabled, BM25 hybrid enabled

Metrics:

- Examples: 10
- Doc Recall@1: 0.700
- Doc Recall@3: 1.000
- Doc Recall@5: 1.000
- Span Recall@5: 1.000
- MRR@5: 0.800
- ROUGE-L: 0.178
- Token F1: 0.237
- Citation coverage: 1.000
- RAG context relevance: 0.580
- Required points covered: 0.783
- Failure labels: `incomplete_answer` 6

Deployment retrieval miss investigation:

- Earlier run `20260628-224937-kubernetes-smoke` had one failure: `deployments` as `retrieval_miss`.
- Existing artifact showed BM25 found `concepts/workloads/controllers/deployment`, but keyword-only hits had no comparable dense score, so rerank effectively discarded them behind dense-only tutorial chunks.
- Retrieval fix: assign keyword-only hits dense-scale pseudo-distance and add doc-path overlap to rerank.
- No reindex required.
- After fix, Deployment is no longer a retrieval miss; it remains an answer-coverage issue because generated answers often omit ReplicaSets / rollout wording.

Smoke-10 interpretation:

- Retrieval is stable enough for a flagship Kubernetes story: all 10 examples hit an expected or acceptable source by top 3 and all expected/acceptable spans by top 5.
- Remaining failures are generation/rubric answer coverage, not retrieval misses or weak citations.
