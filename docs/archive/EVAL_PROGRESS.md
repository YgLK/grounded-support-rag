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

## Evaluation Methodology & Metric Validity

The evaluation harness in `support_graph/evaluation/evaluate.py` tracks three primary dimensions of performance:

### 1. Retrieval (Recall & MRR)
- **Doc Recall@K:** Binary "found it" metric. Returns 1.0 if any gold document ID is in the top K.
- **Span Recall@5:** Measures the fraction of specific "gold grounding sentences" found. This is the most rigorous retrieval metric for MultiDoc2Dial.
- **MRR@5:** Rewards the system for placing the correct document at the top of the results.

### 2. Generation (ROUGE-L, F1, BLEU)
- **ROUGE-L & Token F1:** Measure sequence and bag-of-words overlap. 
- **Exact Match (EM):** Strict string comparison. Rarely hit in conversational tasks but useful for short factual lookups.
- **SacreBLEU:** Standardized n-gram precision for translation-like quality.

### 3. Grounding & E2E Success
- **Citation Coverage:** Fraction of gold spans actually cited in the response.
- **Citations Valid:** Critical safety check ensuring the LLM only cites chunks that were actually retrieved (detects hallucinated citations).
- **End-to-End (E2E) Success:** A composite "Pass/Fail" requiring:
    - Decision to `answer`.
    - Successful retrieval (Doc Recall@3 > 0).
    - Valid citations.
    - Text similarity (ROUGE-L or F1) >= 0.35.

### 4. The RAG Triad (Reference-Free Evaluation)
Used for production monitoring or when "gold" answers are unavailable. These metrics are evaluated by an LLM:
- **Faithfulness (Groundedness):** Measures if the answer is derived *only* from the provided context. It penalizes "hallucinations" even if they are factually correct in general knowledge.
- **Answer Relevance:** Measures how well the response addresses the user's specific question, regardless of whether it used the context correctly.
- **Context Relevance:** Measures the quality of the retrieval—whether the retrieved snippets actually contain the information required to answer the query.

### Assessment
The metrics are highly appropriate for the MultiDoc2Dial dataset. The **0.35 E2E threshold** is a heuristic for "good enough" semantic similarity; while it may flag some technically correct but stylistically different answers as failures, the use of **Failure Labels** (e.g., `wrong_doc`, `weak_citations`) provides the necessary granularity for manual debugging.

**Potential Gaps:**
- The current metrics do not include an "LLM-as-a-judge" (e.g., G-Eval) to check for semantic correctness when ROUGE/F1 is low.
- "Found" documents are based on `doc_id` overlap; more granular paragraph-level `chunk_id` recall could be tracked to further refine retrieval tuning.

## LLM-as-a-Judge (Reference-Free) Insights

As of March 22, 2026, we have integrated a **RAG Triad** evaluation (Faithfulness, Answer Relevance, Context Relevance) to supplement deterministic metrics.

### Key Findings (Run `20260321-235515-dmv-smoke`)
A smoke test of 5 examples using the generation model as the judge revealed:

1.  **Retrieval is robust:** `Context Relevance` scores were high (0.80 - 1.00), confirming that the Ollama embeddings are finding the correct factual snippets.
2.  **The "Helpful Hallucination" Problem:** Several examples received a `Faithfulness` score of **0.00** despite being factually correct in the real world. The model (GPT-OSS-120B) frequently adds details from its pre-training (e.g., "you can also renew by mail") that are not present in the specific retrieved context.
3.  **Semantic vs. String Matching:** One example had a very low ROUGE-L (0.03) but a perfect `Answer Relevance` (1.00), proving that deterministic metrics are under-counting successful support interactions.

### Actionable Strategy
- **Prompt Engineering:** Tighten the system prompt to explicitly forbid adding information not found in the context (even if the model "knows" it to be true).
- **Abstention Logic:** Review "abstain" decisions; current results show the model sometimes abstains even when the judge finds the context 95% relevant.

## Update: March 22, 2026 (v2 + Hybrid Search Fix)

We discovered that initial Hybrid Search (Keyword + Vector) scores were lower than expected due to:
1.  **Keyword Pollution:** The keyword retriever (BM25) was trying to match meta-terms like "Domain: dmv" and "Latest user need:" which are part of the structured query but don't exist in the documents.
2.  **Config Override:** `support_graph.toml` was still forcing `prompt_version = "v1"`.

**Fixes Applied:**
- **Clean BM25 Query:** Hybrid search now uses only the `latest_user_need` for the keyword signal, keeping the full context for the vector signal.
- **v2 Activation:** Updated `support_graph.toml` and `RuntimeConfig` to default to `v2`.
- **Ensemble Import:** Fixed a typo in the `EnsembleRetriever` import path.

## Source Artifacts

- [20260321-235515 manifest](/Users/yglk/coding/grounded-support-rag/outputs/evals/runs/20260321-235515-dmv-smoke/manifest.json)
- [20260321-235515 metrics](/Users/yglk/coding/grounded-support-rag/outputs/evals/runs/20260321-235515-dmv-smoke/metrics.json)
- [20260321-224607 manifest](/Users/yglk/coding/grounded-support-rag/outputs/evals/runs/20260321-224607-dmv-smoke/manifest.json)
- [20260321-224607 metrics](/Users/yglk/coding/grounded-support-rag/outputs/evals/runs/20260321-224607-dmv-smoke/metrics.json)
- [20260321-205603 manifest](/Users/yglk/coding/grounded-support-rag/outputs/evals/runs/20260321-205603-dmv-smoke/manifest.json)
- [20260321-205603 metrics](/Users/yglk/coding/grounded-support-rag/outputs/evals/runs/20260321-205603-dmv-smoke/metrics.json)
- [20260321-202806 manifest](/Users/yglk/coding/grounded-support-rag/outputs/evals/runs/20260321-202806-dmv-smoke/manifest.json)
- [20260321-202806 metrics](/Users/yglk/coding/grounded-support-rag/outputs/evals/runs/20260321-202806-dmv-smoke/metrics.json)
- [20260321-200234 manifest](/Users/yglk/coding/grounded-support-rag/outputs/evals/runs/20260321-200234-dmv-smoke/manifest.json)
- [20260321-200234 metrics](/Users/yglk/coding/grounded-support-rag/outputs/evals/runs/20260321-200234-dmv-smoke/metrics.json)
