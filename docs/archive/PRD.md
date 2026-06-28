# Product Requirements Document

## Project

**Name:** SupportGraph  
**Tagline:** Multi-document support assistant using LangChain, LangGraph, and grounded evaluation

## 1. Overview

SupportGraph is a document-grounded support assistant built on top of the MultiDoc2Dial dataset. Given a conversation transcript up to the next assistant turn, the system retrieves relevant documentation, decides whether it has enough evidence to answer, and returns a grounded support response with citations.

The project exists to demonstrate modern LLM application engineering with:

- LangChain for retrieval, prompts, and structured outputs
- LangGraph for conversation-aware workflow orchestration
- pgvector for RAG over an existing documentation corpus
- offline evaluation over a public benchmark-like dataset
- local-first tracing and optional hosted observability

## 2. Problem Statement

Most support assistants fail for one of three reasons:

- they answer without sufficient evidence
- they retrieve the wrong document or the wrong section
- they do not use conversation history effectively

MultiDoc2Dial is a strong fit because it already provides:

- a real document corpus
- multi-turn support dialogues
- gold references to supporting document spans

This allows SupportGraph to be built as a grounded assistant rather than a generic chatbot.

## 3. Goals

### 3.1 Primary goals

- Build a realistic multi-document support assistant over an existing corpus.
- Learn and demonstrate LangChain and LangGraph in a way that is technically justified.
- Measure retrieval quality, grounding quality, and response quality.
- Produce a portfolio-ready project with a clear product story and reproducible evaluation.

### 3.2 Secondary goals

- Show support for clarification, abstention, or retrieval retry when evidence is weak.
- Compare retrieval and workflow variants during development.
- Capture traces for debugging and failure analysis.

## 4. Non-Goals

- Training or fine-tuning a base model
- Building a general web-scale chatbot
- Supporting all possible datasets or domains in the first version
- Shipping a full production UI in the MVP
- Depending on a live external API as the only runtime option
- Treating the local `test` split as the headline benchmark before its provenance is confirmed

## 5. Users

### 5.1 Primary user

- A user asking a support-style question grounded in a documentation corpus.

### 5.2 Secondary user

- A recruiter, hiring manager, or interviewer evaluating the project as evidence of practical LLM systems experience.

## 6. Dataset Context

SupportGraph will use the local MultiDoc2Dial dataset located at:

`/Users/yglk/coding/grounded-support-rag/multidoc2dial`

Files:

- `multidoc2dial_doc.json`
- `multidoc2dial_dial_train.json`
- `multidoc2dial_dial_validation.json`
- `multidoc2dial_dial_test.json`

Observed local corpus summary:

- `488` documents total
- domains:
  - `dmv`: 149 docs
  - `ssa`: 109 docs
  - `va`: 138 docs
  - `studentaid`: 92 docs
- dialogue counts:
  - train: `3474`
  - validation: `661`
  - test: `661`

Important notes from the local copy:

- the bundled dataset README says the `test` file is a dummy file, but the local `multidoc2dial_dial_test.json` contains labeled `dial_data`
- MVP headline metrics will therefore use `validation`, not `test`
- observed agent turns in the local train and validation splits always include references, but loaders should still tolerate empty references defensively

## 7. Product Definition

### 7.1 Input

- a conversation transcript up to the next assistant turn
- optional domain restriction
- optional example metadata during offline evaluation

The system must not require a single fresh user utterance for every generation step. Some gold agent turns in MultiDoc2Dial are follow-up questions that occur after another agent turn, so the product contract is "generate the next assistant turn from the transcript so far," not only "answer the latest user message."

### 7.2 Output

- a structured assistant response
- decision: `answer`, `clarify`, or `abstain`
- citations to supporting document sections or spans
- trace metadata describing retrieval attempts and the final decision path

### 7.3 Product behavior

The system should not act like a free-form assistant. It should behave like a document-grounded support copilot:

- search the corpus
- use conversation context
- cite supporting evidence
- avoid unsupported answers

Decision rules at the product level:

- `answer` when evidence is sufficient and directly supports a response
- `clarify` when evidence is partially relevant but a missing condition blocks a safe answer
- `abstain` when the system still lacks usable evidence after the allowed retrieval attempts

### 7.4 Primary UX Surfaces

Even though the MVP is CLI-first, the user experience must be designed intentionally.

Primary surfaces:

- single-run CLI output for `grounded-support-rag run`
- evaluation CLI output for `grounded-support-rag eval`
- markdown artifacts for traces, summaries, and failure review

### 7.5 Interaction Hierarchy

For a single run, the output hierarchy must be:

1. last relevant user context
2. decision
3. response text
4. citations
5. next action or interpretation
6. trace summary

For evaluation output, the hierarchy must be:

1. run identity and subset
2. headline metrics
3. major failure categories
4. artifact paths for deeper inspection

### 7.6 Experience Principles

- default output should be concise and readable without raw JSON
- default output should show the last relevant user context before the system decision
- verbose output may reveal retrieval and graph details, but only after the primary answer view
- citations should be adjacent to the answer they support
- clarification should ask one concrete missing-condition question whenever possible
- abstention should explain what is missing and what the next step is
- the interface should use calm, direct language rather than chatbot filler

## 8. MVP Scope

### 8.1 Included in MVP

- local-first runtime with Postgres + pgvector and a documented Ollama-based model path
- `dmv` as the required end-to-end MVP domain
- `ssa` as the first scope expansion after the `dmv` path is stable
- indexing the existing document corpus into pgvector
- turn-level inference for agent responses
- a bounded LangGraph workflow with:
  - query preparation
  - retrieval
  - evidence check
  - at most one retrieval retry
  - final `answer` / `clarify` / `abstain` decision
- citations in every grounded final response
- offline evaluation on the validation split
- CLI-first interface
- human-readable default CLI output plus a verbose inspection mode
- markdown eval summaries and failure-review artifacts
- local traces and machine-readable failure reports

### 8.2 Excluded from MVP

- all four domains from day one
- a polished web frontend
- model fine-tuning
- complex tool ecosystems beyond retrieval and corpus inspection
- rerankers or hybrid retrieval in the first baseline
- human-in-the-loop review workflows

## 9. Why LangGraph Makes Sense Here

This project is not just a single `retrieve -> answer` step. The workflow naturally includes:

- conditioning on conversation history
- rewriting or compressing the current information need
- retrieving from multiple documents
- deciding whether evidence is sufficient
- retrieving again or answering
- optionally clarifying or abstaining

That is a real stateful graph workflow, not a forced one.

## 10. User Stories

- As a user, I want an answer that reflects the current conversation, not just my latest message in isolation.
- As a user, I want the answer to cite relevant documentation.
- As a user, I want the system to say it is unsure or ask for clarification when the evidence is weak.
- As a developer, I want retrieval and generation to be evaluated separately.
- As a reviewer, I want traces that explain why the system produced a given answer.
- As a developer, I want the project to boot locally without relying on a hosted API as the only supported path.

## 11. Functional Requirements

### 11.1 Data preparation

- The system must ingest MultiDoc2Dial documents and dialogues from local JSON files.
- The system must derive an indexable retrieval corpus from document sections or spans.
- The system must derive turn-level evaluation examples from dialogue data.
- The example builder must preserve full prior turn order and support target agent turns that are not immediately preceded by a user turn.

### 11.2 Retrieval

- The system must retrieve relevant documentation for a given conversation state.
- The system should support metadata filtering by domain and document ID.
- The system should support retrieval at section/span-aware granularity rather than only naive fixed chunks.
- The system must support a deterministic baseline retrieval path before any later reranking or hybrid work.

### 11.3 Conversation handling

- The system must use dialogue history when forming the retrieval query.
- The system must support at least one retry path when evidence is weak.
- The system must support `clarify` or `abstain` as a fallback.
- The system must generate the next assistant turn from the conversation transcript, not only from a mandatory latest user utterance.

### 11.4 Answer generation

- The system must generate an assistant response grounded in retrieved evidence.
- The final answer must include citations to source chunks or spans.
- The system must avoid unsupported claims when evidence is insufficient.
- The output contract must work for answer, clarification, and abstention responses.

### 11.5 Evaluation

- The system must support offline validation over held-out dialogue turns.
- The evaluation must measure retrieval quality and response quality separately.
- The evaluation must use the dataset's gold references where available.
- The evaluation must report answer-like turns separately from clarification-like turns, because only the first group is suitable for the primary automatic text-quality benchmark in the MVP.

### 11.6 UX and Reporting

- The default CLI output must be understandable without reading raw JSON.
- The system must support a verbose mode for retrieval and graph diagnostics.
- Every non-success state must include a clear next step.
- Evaluation runs must generate both machine-readable artifacts and a concise human-readable summary.
- The final design decisions for terminal and markdown surfaces must be captured in `DESIGN.md`.

## 12. Non-Functional Requirements

- The system must be runnable locally.
- The local bootstrap path must not require a hosted API as the only supported runtime.
- The retrieval index must be reproducible from raw dataset files.
- The workflow must be traceable step by step.
- The project should be simple enough to complete as a one-week MVP.
- The MVP repository layout should stay intentionally small and avoid premature package splitting.
- The CLI and markdown surfaces must remain readable in plain text without relying on color alone.
- A reviewer should be able to understand a run result and an eval result without opening raw JSON files.

## 13. Evaluation Strategy

SupportGraph will be evaluated at three levels.

### 13.1 Retrieval evaluation

Questions:

- Did the system retrieve the correct document?
- Did it retrieve text containing the gold supporting spans?

Metrics:

- Doc Recall@k
- Span Recall@k
- MRR

These retrieval metrics should be reported for:

- answer-like agent turns
- clarification-like agent turns

### 13.2 Response evaluation

Primary automatic response evaluation should focus on answer-like agent turns:

- `respond_solution`
- `respond_solution_positive`
- `respond_solution_negative`
- `respond_no_solution`

Questions:

- Did the response match the target answer reasonably well?
- Was it grounded in valid evidence?

Metrics:

- ROUGE-L or token-level F1
- citation coverage against gold references
- groundedness / support checks

### 13.3 Clarification evaluation

Clarification-style turns should be reported separately in the MVP:

- retrieval metrics still apply
- the system should report the count and rate of `clarify` decisions
- a capped manual review sample should be used for quality inspection

This avoids pretending that lexical overlap alone is a reliable metric for follow-up questions.

### 13.4 End-to-end evaluation

For the MVP, a turn is considered an end-to-end automatic success only on answer-like turns if:

- a gold doc or span is retrieved
- the generated answer clears a defined similarity threshold
- citations map to valid retrieved chunks

Clarification-style turns should be summarized separately rather than merged into the primary success rate.

## 14. Success Criteria

The MVP is successful if:

- it indexes the existing MultiDoc2Dial corpus end to end
- it can run the full workflow on `dmv` validation examples
- it returns cited responses
- retrieval metrics and response metrics can be reproduced automatically on answer-like validation turns
- the graph workflow is visible and explainable
- a user can understand one run output and one eval output without raw-log inspection
- the project is coherent as a hiring artifact

Stretch success after the core MVP is stable:

- extend the same benchmark flow to `ssa`

## 15. One-Week MVP

### Day-1 scope

- lock the runtime contract
- add the data-flow and graph-flow diagrams to the technical spec
- ingest `dmv`
- derive section-aware chunks
- create turn-level examples

### Day-2 scope

- build the pgvector index
- verify retrieval on a small `dmv` smoke set

### Day-3 to Day-4 scope

- implement the bounded LangGraph workflow
- add one retrieval retry branch
- generate structured responses with citations
- write local traces

### Day-5 to Day-6 scope

- build offline evaluation
- run the `dmv` validation benchmark on answer-like turns
- inspect failures
- add a manual review sample for clarification-style turns

### Day-7 scope

- clean CLI
- extend to `ssa` only if the `dmv` path is already stable
- write README with metrics, tradeoffs, and deferred work

## 16. Risks

- The graph could become more complex than the baseline justifies.
- Clarification-style turns are less straightforward to auto-score than answer-style turns.
- Section-aware retrieval may require more preprocessing than plain chunking.
- Local runtime setup could drift if the provider contract is not explicit.
- The local `test` split may look usable even though its benchmark meaning is unclear.

## 17. Mitigations

- Keep the graph bounded and explicit.
- Report clarification-style turns separately from the main automatic score.
- Prefer section/span-aware chunking derived from dataset annotations.
- Define the runtime, storage, and config contract before implementation.
- Use `validation` as the headline benchmark and treat `test` as out of scope for MVP reporting.

## 18. Recommended Next Step

Before implementation, the technical spec should define the exact artifacts and runtime contract:

- derived chunk schema
- derived turn-example schema
- LangGraph state schema
- decision rules for `answer` / `clarify` / `abstain`
- retrieval and answer evaluation rules
- local bootstrap contract
- CLI interaction design and reporting hierarchy

That technical spec should directly drive the implementation.
