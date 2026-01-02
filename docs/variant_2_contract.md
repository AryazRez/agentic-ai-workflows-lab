# Variant 2 Contract (MCP-Orchestrated Workflow)

This document defines the contract for Variant 2 of the agentic workflow system.

Variant 2 introduces MCP-style orchestration while preserving the semantics, artifacts, and auditability of Variant 1. This contract exists to prevent abstraction creep and to ensure that any benefits of MCP are measurable rather than cosmetic.

---

## Purpose of this contract

Variant 2 is not a rewrite. It is a controlled experiment.

This contract specifies:
- What Variant 2 is allowed to change
- What Variant 2 must preserve
- What constitutes a violation of design intent

If an implementation violates this contract, it is considered a failed variant regardless of code quality.

---

## Non-negotiable invariants

Variant 2 MUST preserve the following invariants from Variant 1.

### Inputs
- `inputs/run_config.json` schema is unchanged
- Same topic structure
- Same output configuration semantics

### Outputs
- Same output directory layout under `outputs/<run_id>/`
- Same artifact types:
  - topic research notes
  - prompt traces
  - synthesis output
  - short-form drafts (optional)
  - run manifest
- Artifacts remain human-readable and inspectable

### Run semantics
- A run has a clear start and end
- A run produces a bounded set of artifacts
- Partial runs are allowed but must be explicit
- Failures are surfaced, not hidden

Variant 2 may not introduce autonomous looping or open-ended execution.

---

## Conceptual components

Variant 2 introduces three explicit components.

---

## 1. RunContext (explicit shared state)

The RunContext represents the current state of a run.

### Required properties

The RunContext MUST include at least:

- `run_id`
- `topics`
- `current_topic_id` (or equivalent)
- `topic_outputs`
- `summary_output`
- `short_form_outputs`
- `errors`

### Rules

- Context updates must be explicit
- Context must be inspectable at any step
- Context must not be mutated implicitly by the model
- Context growth must be bounded

The model may read context. It may not redefine it.

---

## 2. Tools (first-class, schema-defined)

Each pipeline step is implemented as a tool.

### Required tools

Variant 2 MUST implement at least the following tools:

1. `generate_topic_notes`
2. `synthesize_notes`
3. `generate_short_form` (optional based on config)

### Tool contract

Each tool MUST define:
- Tool name
- Input schema
- Output schema
- Side effects (files written)

Tools:
- Do exactly one thing
- Do not call other tools
- Do not decide execution order
- Do not mutate global state outside declared outputs

Tools may fail. Failures must be returned explicitly.

---

## 3. Orchestrator (deterministic coordinator)

The orchestrator owns execution order and control flow.

### Orchestrator responsibilities

- Load run configuration
- Initialize RunContext
- Determine tool execution order
- Inject context into tool calls
- Capture tool outputs
- Persist artifacts
- Update run manifest
- Handle failures according to policy

### Orchestrator constraints

The orchestrator:
- Is deterministic
- Does not reason creatively
- Does not delegate control to the model
- Does not invent tools
- Does not modify schemas dynamically

The orchestrator coordinates. It does not think.

---

## Role of MCP

MCP is used to describe:
- Tool schemas
- Context interfaces
- Tool availability

MCP is NOT used to:
- Enable autonomous planning
- Allow unrestricted tool chaining
- Hide execution logic
- Replace explicit orchestration

If MCP obscures state or control flow, its use violates this contract.

---

## Prompt design constraints

In Variant 2:
- Prompts must be smaller than in Variant 1
- Prompts must not embed full run state
- Prompts must rely on structured context injection

Prompts are still versioned artifacts and must be traceable.

---

## Error handling contract

Variant 2 must preserve Variant 1 error semantics.

- Topic-level failures do not crash the run
- Errors are captured per tool invocation
- Errors are written to artifacts
- Errors are summarized in the run manifest

Variant 2 may not silently retry without recording the attempt.

---

## Observability requirements

Variant 2 MUST provide at least the same observability as Variant 1.

Required artifacts:
- Prompt traces for all model calls
- Tool input and output records (directly or indirectly)
- Run manifest with call counts and errors

If observability decreases, Variant 2 fails the contract.

---

## Explicitly out of scope

Variant 2 MUST NOT include:
- Autonomous planning
- Self-directed loops
- Multi-agent negotiation
- Tool discovery without constraints
- Dynamic schema creation
- Hidden memory mechanisms

Those patterns are intentionally excluded from this repository.

---

## Success criteria

Variant 2 is considered successful only if:

- The same workflow is easier to reason about
- Failures are more diagnosable
- Constraints are more enforceable
- Tradeoffs are clearer than in Variant 1

Cleaner code alone is not sufficient.

---

## What this contract enables

Because this contract exists:
- Variant 2 can be compared honestly to Variant 1
- MCP can be evaluated as an architectural tool, not a trend
- Abstraction benefits and costs become explicit

This document is the authority for Variant 2 design decisions.
