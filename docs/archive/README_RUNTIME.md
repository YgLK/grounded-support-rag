# `support_graph/runtime/`

This package owns the shared execution graph behind both `run` and `eval`. It combines retrieval, prompt-driven reasoning, retries, tracing, and optional hosted observability.

## File Map

| File | Responsibility | Connects To |
| --- | --- | --- |
| `__init__.py` | Package marker for runtime helpers. | No runtime logic. |
| `schemas.py` | Defines graph state, stream-event payloads, runtime resource containers, and Pydantic schemas for structured LLM outputs. | Shared by `graph.py`, `nodes.py`, and event consumers. |
| `prompts.py` | Holds versioned prompt templates and resolves the active prompt set from config. | Used by `nodes.py` when grading evidence and generating responses. |
| `llm_policy.py` | Wraps async LLM calls with retry rules and a shared concurrency semaphore keyed by provider and model. | Used by `nodes.py` for all structured chat-model calls. |
| `observability.py` | Configures optional LangSmith and OpenTelemetry support and exposes lightweight run and span contexts. | Used by `nodes.py` and `graph.py` to instrument the shared runtime. |
| `traces.py` | Writes local JSONL trace events and summarizes trace files for CLI and eval inspection. | Used during graph execution and later by `cli.py` and `evaluation/evaluate.py`. |
| `nodes.py` | Implements the actual node behavior: query preparation, retrieval, neighbor expansion, evidence grading, response generation, fallback handling, finalization, and runtime resource resolution. | Calls `retrieval/`, `providers.py`, `prompts.py`, `llm_policy.py`, and `traces.py`. |
| `graph.py` | Wires the LangGraph state machine, routes between nodes, emits graph events, and exposes `run_graph_async` plus event streaming. | Called by the CLI for `run` and by `evaluation/evaluate.py` for batch evaluation. |

## How It Connects Later On

1. `graph.py` defines the control flow, but `nodes.py` contains most of the execution logic.
2. `nodes.py` pulls ranked chunks from `retrieval/retrieve.py`, expands them when configured, and then decides whether the graph should answer, clarify, or abstain.
3. `prompts.py`, `llm_policy.py`, and `providers.py` keep model-specific concerns out of the control flow itself.
4. `traces.py` and `observability.py` run alongside every graph execution so both local artifact inspection and optional hosted tracing see the same path.
5. `evaluation/evaluate.py` reuses this exact graph, which is why runtime behavior and offline scoring stay aligned.
