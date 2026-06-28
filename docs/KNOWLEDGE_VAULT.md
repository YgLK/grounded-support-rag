# SupportGraph Knowledge Vault

This document captures high-value architectural insights, evaluation strategies, and RAG optimization patterns discovered during the development of the SupportGraph system.

---

## 1. The RAG Triad (Reference-Free Evaluation)
In production settings where "Gold Answers" (human-written references) are unavailable, we use the **RAG Triad** to evaluate quality using an LLM-as-a-judge.

| Metric | Definition | What it catches |
| :--- | :--- | :--- |
| **Faithfulness** | Measures if the answer is derived *only* from the retrieved context. | **Hallucinations:** Prevents the model from using its own "internal" knowledge to answer. |
| **Answer Relevance** | Measures how well the response addresses the user's specific query. | **Avoidance:** Catches cases where the model gives a correct fact that doesn't actually help the user. |
| **Context Relevance** | Measures if the retrieved snippets actually contain the answer. | **Retriever Failure:** Identifies when the embedding model or search query is failing. |

**Pro Tip:** High Context Relevance + Low Faithfulness usually indicates a "Helpful Hallucination"—where the model knows the answer from its training data but it isn't in your specific manual.

---

## 2. Deterministic vs. Semantic Metrics
*   **ROUGE-L / Token F1:** Good for benchmarking against stable datasets, but "brittle." They penalize models for using synonyms (e.g., "Yes, you can" vs. "You are permitted").
*   **E2E Success Threshold:** We found that a ROUGE-L/F1 threshold of **0.35** is a reliable heuristic for "passable" similarity in this specific support domain.
*   **The Gap:** Always supplement string-matching (ROUGE) with Semantic Grading (LLM-Judge). A ROUGE score of 0.03 can still be a perfect 1.0 Answer Relevance score.

---

## 3. Prompt Engineering (v2 Strategy)
To move from a "prototype" to "production-grade" grounding, move from v1 (Zero-shot) to v2 (Guided) prompts:

*   **Negative Constraints:** Use phrases like *"If the information is not in the provided <context>, you MUST state that you do not know."*
*   **XML Tagging:** Surround context with `<context>` and `<chunk>` tags. LLMs are statistically better at "looking inside" structured tags than flat text blocks.
*   **Few-Shotting:** Include examples of "Over-reaching" (where the model added info) and label them as failures to teach the model the boundaries of its context.

---

## 4. Graph & Node Optimizations
*   **Intent Routing:** Add a "Router" node at the start. If the user says "Hello" or "Thanks," skip the expensive Vector Search and Grading steps. This saves ~40% latency on non-factual turns.
*   **Hybrid Search (BM25 + Vector):** Pure vector search often fails on technical acronyms (e.g., "MV-78B"). Use an `EnsembleRetriever` to combine BM25 (Keyword) and Vector (Semantic) search. 
    - **Recommended Weights:** 0.4 BM25 / 0.6 Vector.
*   **Neighbor Expansion:** When a chunk is retrieved, also pull the chunks immediately before and after it (by `section_id`). This provides the LLM with the "surrounding narrative" of the manual, which often contains crucial "if/then" conditions.
*   **Context Density:** Order chunks by relevance, but ensure that the "Anchor" (the chunk that actually matched the search) is clearly labeled so the LLM knows where to look first.

---

## 5. Advanced Resilience Patterns
*   **Self-Correction Node:** Instead of going straight from Generation to Finalize, add a "Self-Correction" step. An LLM checks the generated answer against the retrieved chunks. If the answer contains "internal knowledge" (hallucinations) not found in the context, the node triggers a retry with a "Strict Grounding" instruction.
*   **HyDE (Hypothetical Document Embeddings):** To improve vector search for conversational queries, generate a "hypothetical answer" first. Use the embedding of that fake answer to find real chunks. This shifts the search from "Question vs Document" to "Answer vs Document," which is often more accurate.
*   **Document-Level Routing:** In systems with many manuals, use a "Router" to identify the specific document title first. Filter the retrieval to *only* that document to eliminate noise from other irrelevant manuals.

---

## 6. Cost Management
*   **Context Truncation:** Limit each retrieved chunk to ~1,500 characters before sending it to the LLM. Most support answers are found in the first few sentences.
*   **Judgment Sampling:** Don't run the LLM-Judge on every single prediction during dev. Run it on a 5-10% random sample to get a "vibe check" without draining your API budget.
*   **Model Tiering:** Use cheaper models (e.g., GPT-4o-mini or Qwen-7B) for high-volume tasks like "Grading" and save the large models for "Final Answer Generation."

---

*Last Updated: March 22, 2026*
