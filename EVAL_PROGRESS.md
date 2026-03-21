# SupportGraph Eval Progress

This file tracks the recent smoke eval runs that were used to compare provider and model choices.

All runs below are on the `dmv validation / smoke` subset unless noted otherwise.

## Recent Smoke Runs

| Run ID | Examples | Chat | Embeddings | Attempts | Doc R@3 | Doc R@5 | Span R@5 | MRR@5 | ROUGE-L | F1 | Citation Cov. | E2E | Avg Latency |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20260321-235515-dmv-smoke` | 25 | `openrouter` `openai/gpt-oss-120b:nitro` | `ollama` `qwen3-embedding:4b-q4_K_M` | 2 | 0.760 | 0.760 | 0.307 | 0.573 | 0.165 | 0.211 | 0.240 | 0.160 | 2588 ms |
| `20260321-224607-dmv-smoke` | 25 | `ollama` `qwen3:8b-q4_K_M` | `ollama` `qwen3-embedding:4b-q4_K_M` | 2 | 0.680 | 0.760 | 0.320 | 0.565 | 0.158 | 0.188 | 0.100 | 0.160 | 27430 ms |
| `20260321-205603-dmv-smoke` | 25 | `ollama` `qwen3:8b-q4_K_M` | `openrouter` `qwen/qwen3-embedding-8b` | 2 | 0.120 | 0.160 | 0.153 | 0.090 | 0.100 | 0.126 | 0.000 | 0.000 | 25033 ms |
| `20260321-202806-dmv-smoke` | 25 | `openrouter` `openai/gpt-oss-120b:nitro` | `openrouter` `qwen/qwen3-embedding-8b` | 2 | 0.160 | 0.200 | 0.120 | 0.110 | 0.089 | 0.099 | 0.000 | 0.040 | 7319 ms |
| `20260321-200234-dmv-smoke` | 25 | `openrouter` `arcee-ai/trinity-large-preview:free` | `openrouter` `qwen/qwen3-embedding-8b` | 2 | 0.120 | 0.160 | 0.200 | 0.061 | 0.069 | 0.081 | 0.000 | 0.040 | 11167 ms |
| `20260320-182749-dmv-smoke` | 3 | `ollama` `qwen3:4b` | `ollama` `qwen3-embedding:4b-q4_K_M` | 2 | 1.000 | 1.000 | 0.000 | 0.833 | 0.097 | 0.144 | 0.000 | 0.000 | 83929 ms |

## Current Read

- The biggest quality jump came from switching embeddings back to Ollama. The `ollama chat + openrouter embeddings` run collapsed retrieval quality.
- The current best practical configuration is `openrouter` chat plus `ollama` embeddings from run `20260321-235515-dmv-smoke`.
- That best run improved over `ollama` chat plus `ollama` embeddings mainly on latency, answer rate, citation coverage, and small gains in ranking quality.
- The old March 20, 2026 Ollama run only covered 3 examples, so it is not a fair baseline against the 25-example runs.

## Important Caveat

Every run above used `max_retrieval_attempts = 2`.

That means retrieval metrics are still partially affected by chat-model-guided retry and refinement. For a cleaner isolation matrix, rerun the key configs with:

```toml
max_retrieval_attempts = 1
```

## Recommended Next Matrix

| Goal | Chat | Embeddings | Attempts |
| --- | --- | --- | ---: |
| Clean embedding check | `ollama` `qwen3:8b-q4_K_M` | `ollama` `qwen3-embedding:4b-q4_K_M` | 1 |
| Clean embedding check | `ollama` `qwen3:8b-q4_K_M` | `openrouter` `qwen/qwen3-embedding-8b` | 1 |
| Clean chat check | `ollama` `qwen3:8b-q4_K_M` | `ollama` `qwen3-embedding:4b-q4_K_M` | 1 |
| Clean chat check | `openrouter` `openai/gpt-oss-120b:nitro` | `ollama` `qwen3-embedding:4b-q4_K_M` | 1 |

## Source Artifacts

- [20260321-235515 manifest](/Users/yglk/coding/support-graph/outputs/evals/runs/20260321-235515-dmv-smoke/manifest.json)
- [20260321-235515 metrics](/Users/yglk/coding/support-graph/outputs/evals/runs/20260321-235515-dmv-smoke/metrics.json)
- [20260321-224607 manifest](/Users/yglk/coding/support-graph/outputs/evals/runs/20260321-224607-dmv-smoke/manifest.json)
- [20260321-224607 metrics](/Users/yglk/coding/support-graph/outputs/evals/runs/20260321-224607-dmv-smoke/metrics.json)
- [20260321-205603 manifest](/Users/yglk/coding/support-graph/outputs/evals/runs/20260321-205603-dmv-smoke/manifest.json)
- [20260321-205603 metrics](/Users/yglk/coding/support-graph/outputs/evals/runs/20260321-205603-dmv-smoke/metrics.json)
- [20260321-202806 manifest](/Users/yglk/coding/support-graph/outputs/evals/runs/20260321-202806-dmv-smoke/manifest.json)
- [20260321-202806 metrics](/Users/yglk/coding/support-graph/outputs/evals/runs/20260321-202806-dmv-smoke/metrics.json)
- [20260321-200234 manifest](/Users/yglk/coding/support-graph/outputs/evals/runs/20260321-200234-dmv-smoke/manifest.json)
- [20260321-200234 metrics](/Users/yglk/coding/support-graph/outputs/evals/runs/20260321-200234-dmv-smoke/metrics.json)
