# SupportGraph Design Guide

## Purpose

This document defines the user-facing design decisions for SupportGraph.

The MVP is CLI-first, so the design system is primarily about:

- terminal interaction design
- markdown artifact readability
- trust-building response behavior
- clear state handling for answer, clarification, abstention, setup, and failure

## 1. Product Posture

SupportGraph should feel like a trustworthy documentation operator, not a generic chatbot.

The product should feel:

- calm
- concrete
- evidence-led
- legible under pressure

The product should not feel:

- conversational for its own sake
- overconfident
- decorative
- like a raw JSON debugger unless the user explicitly asks for verbose output

## 2. Primary Surfaces

SupportGraph has three primary user-facing surfaces.

### 2.1 `grounded-support-rag run`

Audience:

- developer testing an example
- operator inspecting a single answer
- reviewer checking whether the system feels grounded

Goal:

- understand the system's decision in under 10 seconds

### 2.2 `grounded-support-rag eval`

Audience:

- engineer comparing runs
- reviewer scanning benchmark results

Goal:

- understand whether the system improved, regressed, or needs inspection

### 2.3 Markdown artifacts

Audience:

- hiring manager
- interviewer
- future you

Goal:

- explain the system clearly without opening raw logs

## 3. Information Hierarchy

Every major surface should follow the same hierarchy.

### 3.1 Single-run hierarchy

Order:

1. last relevant user context
2. decision
3. response or question shown to the user
4. citations
5. next action or interpretation
6. trace summary

If the output starts with raw chunk dumps, token counts, or JSON, the hierarchy is wrong.

### 3.2 Eval hierarchy

Order:

1. what run this is
2. which subset was evaluated
3. headline metrics
4. key deltas or failure counts
5. artifact paths for deeper inspection

### 3.3 Failure review hierarchy

Order:

1. failure label
2. what the system did
3. what the gold evidence was
4. why it likely failed
5. where to inspect more

## 4. Interaction Principles

### 4.1 Decision-first

The first prominent system state in a run should state whether the system chose:

- `answer`
- `clarify`
- `abstain`

One short context anchor may appear first if it helps the user understand what the system is reacting to.

### 4.1A Context anchor

Default `run` output should show the last relevant user turn before the decision.

Why:

- it immediately grounds the output
- it helps reviewers judge whether the answer matches the need
- it makes clarification and abstention decisions feel less arbitrary

This context anchor should be:

- one short block
- the last relevant user message, not the whole transcript
- above the decision

### 4.2 Evidence adjacent to claims

Citations should be visually close to the answer. Users should not need to scroll far away from the claim to see support.

### 4.3 Progressive disclosure

Default output should be concise and human-readable.

Verbose output may include:

- query text
- evidence grade
- retrieved chunks
- graph path

But those details belong behind `--verbose` or in artifact files, not in the default view.

### 4.4 One missing thing at a time

When clarifying, ask one concrete missing-condition question.

Bad:

- "Can you clarify your situation?"

Good:

- "Was your license already current before the address change?"

### 4.5 No fake certainty theater

Do not show fake numeric confidence like `0.79` to end users.

Use:

- `high`
- `medium`
- `low`

Only if the label helps interpretation.

### 4.6 Next-step guidance

Every non-success state should tell the user what to do next.

Examples:

- configure models
- build the index
- retry with `--verbose`
- inspect the failure artifact

## 5. Voice and Copy

Response tone should be:

- direct
- restrained
- helpful
- grounded in the source text

Copy rules:

- no hype
- no emoji
- no anthropomorphic phrases like "I think" or "I believe"
- no apology spam
- no filler before the answer
- if evidence is weak, say exactly what is missing

Clarification tone:

- specific
- narrow
- one question when possible

Abstention tone:

- clear about the limit
- explicit about what would unblock an answer

## 6. Terminal Layout Rules

### 6.1 `grounded-support-rag run`

Default layout:

```text
SupportGraph Run
Example: dmv::...::turn_6
Context
User: I moved and need to change the address on my license.

Decision: clarify

Response
Need one detail before I can answer: was your license already current?

Citations
[1] Top 5 DMV Mistakes and How to Avoid Them#3_0 :: spans 4,5

Trace
Attempts: 1
Path: prepare_query -> retrieve_docs -> grade_evidence -> finalize
```

### 6.2 `grounded-support-rag eval`

Default layout:

```text
SupportGraph Eval
Run: 20260318-143000-dmv-history-retry
Subset: dmv validation / answer

Headline Metrics
Doc Recall@3: ...
Span Recall@5: ...
ROUGE-L: ...
F1: ...

Failure Snapshot
wrong_doc: ...
missed_history: ...
unsupported_answer: ...

Artifacts
outputs/evals/runs/<run_id>/summary.md
outputs/evals/runs/<run_id>/failures.jsonl
```

### 6.3 Verbose mode

Verbose mode may add:

- full recent conversation context
- final query
- evidence grade
- retrieved chunk previews
- graph node path

It should append details after the main answer, not interrupt the primary reading flow.

## 7. Empty, Setup, and Error States

These states are part of the product.

### 7.1 Missing config

State:

- required config missing

Response should include:

- what is missing
- where config is loaded from
- the exact next command or file to inspect

### 7.2 Index missing

State:

- retrieval attempted before indexing

Response should include:

- that the corpus is not indexed yet
- the build/index command to run next

### 7.3 No evidence found

State:

- retrieval returns weak or empty evidence

Response should include:

- a plain-language statement that no sufficient support was found
- whether the system clarified or abstained
- a pointer to verbose mode or failure artifacts if relevant

### 7.4 No eval results yet

State:

- no outputs under `outputs/evals/runs/`

Response should include:

- how to run the first eval
- which subset is recommended first

## 8. Accessibility

Even in a terminal-first MVP, accessibility must be explicit.

Rules:

- do not rely on color alone to convey state
- headings and labels must be readable without ANSI color
- line length should remain readable in narrow terminals
- markdown summaries must make sense when read as plain text
- citations must use stable labels and ordering
- tables are optional, but critical information must still read well when wrapped

## 9. AI Slop Guardrails

These are failure patterns to avoid:

- raw JSON as the default experience
- giant metric dumps without explanation
- vague empty states like "No results found"
- unclear citations disconnected from the answer
- multiple competing summaries on the same screen
- a README that reads like generic AI-project marketing copy

## 10. Portfolio Artifact Rules

The final README and eval summaries should help a reviewer understand:

- one successful grounded example
- one clarification example
- one abstention or failure example
- what changed across experiments
- what still fails

If a reviewer must parse raw logs to understand the project, the design failed.
