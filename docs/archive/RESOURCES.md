# RAG Engineering Resources

This document contains a curated list of research papers, technical blogs, and documentation for the advanced RAG patterns implemented in this project.

## 1. Foundational Retrieval Patterns
*   **[Anthropic] Contextual Retrieval:** (Sept 2024) A must-read on how to prepend document context to chunks to prevent "retrieval drift."
    *   [Link](https://www.anthropic.com/news/contextual-retrieval)
*   **[LlamaIndex] Sentence Window Retrieval:** Conceptual guide for "Neighbor Expansion" (pulling surrounding context for isolated chunks).
    *   [Link](https://docs.llamaindex.ai/en/stable/examples/retrieval/sentence_window_retrieval/)
*   **[Cohere] The Power of Reranking:** Why vector search alone is insufficient and how secondary rerankers improve precision.
    *   [Link](https://txt.cohere.com/rerank/)

## 2. Evaluation & The RAG Triad
*   **[Arize Phoenix] The RAG Triad:** Definition of Faithfulness, Answer Relevance, and Context Relevance.
    *   [Link](https://docs.arize.com/phoenix/evaluation/llm-evals/rag-triad)
*   **[Ragas] Reference-Free Metrics:** Mathematical and conceptual definitions for the metrics used in our `judge.py`.
    *   [Link](https://docs.ragas.io/en/stable/concepts/metrics/index.html)
*   **[DeepLearning.AI] Quality & Safety for LLM Apps:** A short course covering evaluation loops.
    *   [Link](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/)

## 3. Query Optimization & Advanced Search
*   **[Research Paper] HyDE (Hypothetical Document Embeddings):** The original paper on using "fake" answers to improve retrieval.
    *   [Link](https://arxiv.org/abs/2212.10496)
*   **[LangChain] Multi-Query Retriever:** How to generate multiple search variations from a single user question.
    *   [Link](https://python.langchain.com/docs/how_to/MultiQueryRetriever/)
*   **[Pinecone] Semantic Routing:** Patterns for classifying intent before searching (like our Intent Router).
    *   [Link](https://www.pinecone.io/learn/series/rag/semantic-routing/)

## 4. Multi-Turn Dialogue (The MultiDoc2Dial Context)
*   **[Original Dataset Paper] MultiDoc2Dial:** The research paper describing the specific dataset used in this project.
    *   [Link](https://arxiv.org/abs/2109.12595)
*   **[Salesforce Research] Multi-document Dialogue:** Strategies for handling grounding across multiple source documents.
    *   [Link](https://github.com/alexa/multidoc2dial)

## 5. Architectural Blueprints
*   **[Modular RAG] Beyond Naive RAG:** A comprehensive map of the "Modular RAG" paradigm.
    *   [Link](https://arxiv.org/abs/2312.10997) (Original "Retrieval-Augmented Generation for LLMs" Survey)
