# `support_graph/retrieval/`

This package owns retrieval storage and query-time search. It bridges deterministic chunk artifacts with the pgvector-backed runtime.

## File Map

| File | Responsibility | Connects To |
| --- | --- | --- |
| `__init__.py` | Package marker for retrieval helpers. | No runtime logic. |
| `index.py` | Loads chunk artifacts, converts them into LangChain `Document` objects, builds embeddings, writes vectors into pgvector, and inspects collection state. | Used by the CLI `index-docs` command, embedding benchmarks, and index preflight checks. |
| `retrieve.py` | Builds retrieval queries from example context, applies metadata filters, normalizes vectorstore hits, performs deterministic reranking, and returns the ranked top-k chunks. | Used by `runtime/nodes.py` during graph execution. Its ranked output also feeds retrieval metrics in `evaluation/evaluate.py`. |

## How It Connects Later On

1. `index.py` is the write path: chunk JSONL in, pgvector rows out.
2. `retrieve.py` is the read path: runtime example in, ranked retrieval hits out.
3. Downstream, the runtime graph keeps two retrieval views so direct retrieval scoring stays separate from later evidence expansion.
4. `retrieval_ranked_chunks` comes directly from this package and is what evaluation uses for retrieval metrics.
5. `retrieved_chunks` is a later runtime expansion over those ranked hits, so this package intentionally stops before neighbor expansion and response generation.
