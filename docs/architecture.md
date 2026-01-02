# Architecture

This document describes the architecture of the workflow implemented in this repository and the boundaries that make variants comparable.

The repository is organized around a single core task implemented multiple ways:
AI-assisted research and content synthesis.

Variant 1 is the baseline implementation and serves as the reference architecture.

---

## Goals

The architecture is designed to make the following properties visible:

- What happened during a run (traceability)
- What inputs drove the run (reproducibility)
- Where state is stored (auditability)
- How failures surface (observability)

The architecture intentionally does not optimize for maximum automation. It optimizes for clarity.

---

## System boundaries

The system has three conceptual layers.

1. Orchestration layer
   - Owns the workflow sequence.
   - Owns error isolation and run boundaries.
   - Changes across variants.

2. Shared primitives
   - Schemas, prompts, and utilities used by all variants.
   - Designed to be stable across variants.

3. External dependencies
   - Model provider API (OpenAI in Variant 1).
   - File system (run artifacts and outputs).
   - Optional future: retrieval tools, databases, monitoring hooks.

---

## Data flow overview

High-level flow for all variants:

1. Load run configuration
2. For each topic:
   - Generate research notes
   - Persist outputs and trace
3. Synthesize across topics
   - Persist summary and trace
4. Optional short-form generation
   - Persist drafts and traces
5. Write run manifest

Variant 1 implements this flow with direct model calls. Later variants will preserve the same flow while changing orchestration mechanics.

---

## State model

The system is intentionally stateful, and state is persisted as files for transparency.

### Inputs

- `inputs/run_config.json`

This file defines:
- run id
- topics
- output settings

### Outputs (per run)

All run artifacts are stored under:

- `outputs/<run_id>/`

This directory is treated as an auditable record of:
- what was asked
- what was produced
- what prompts and responses were used
- what failed

Important: The repository uses `.gitignore` to avoid committing outputs. Outputs are considered runtime artifacts.

---

## Artifact layout (per run)

''''
outputs/<run_id>/
├── run_config.json
├── run_manifest.json
├── topics/
│ ├── <topic_id>/
│ │ ├── research_notes.md
│ │ ├── prompt_trace.json
│ │ └── error.json (only if failed)
│ └── ...
├── summary/
│ ├── brief.md
│ ├── prompt_trace.json
│ └── error.json (only if failed)
└── short_form/ (optional)
├── <platform>.md
├── <platform>_prompt_trace.json
└── error.json (only if failed)
''''


---

## Core components

### `shared/schemas.py`
Defines data structures for:
- topic specifications
- output configuration
- run configuration

This layer represents the stable contract that all variants should follow.

### `shared/prompts.py`
Defines prompt templates for:
- per-topic research notes
- synthesis across topics
- short-form generation

Prompts are treated as code. Changes to prompts are expected to cause behavior changes and should be versioned through commits.

### `shared/utils.py`
Defines utilities for:
- directory creation
- writing Markdown
- writing JSON with dataclass support
- timestamps

The utilities are intentionally simple. They provide observability but do not hide logic.

### `baseline_tool_calling/pipeline.py`
Implements Variant 1.

Responsibilities:
- load and validate config (lightweight)
- execute the workflow in a fixed sequence
- isolate failures per topic
- persist traces and artifacts
- write a run manifest

---

## Execution model

Variant 1 runs synchronously and sequentially.

Reasons:
- Simplicity and traceability
- Easy debugging
- Deterministic ordering of artifacts

Future variants may add concurrency, but only if they preserve observability and preserve comparable outputs.

---

## Error handling strategy

Variant 1 uses a simple policy:

- Topic-level failures do not crash the run.
- Failures are written to per-topic `error.json` files.
- A synthesis failure produces a summary error artifact and continues to manifest writing.
- The run manifest records all errors.

This policy favors partial outputs and explicit failure visibility over strict completeness.

---

## Observability and traceability

The system is designed to allow post-run inspection without rerunning.

Key mechanisms:
- Prompt traces are stored for all model calls.
- Run manifest records call counts and errors.
- Outputs are stored in human-readable Markdown.

Limitations in Variant 1:
- No source retrieval or citation grounding
- No structured evaluation metrics
- No token usage or cost accounting beyond call count

These limitations are intentional and are explored as tradeoffs in later variants.

---

## Variant comparison strategy

The architecture is intentionally designed to support comparability across variants.

Invariant across variants:
- input schema
- output artifacts and layout (as closely as possible)
- concept of a run manifest
- the core workflow stages

What varies across variants:
- how tools are represented
- how state transitions are managed
- how errors are handled
- what abstractions exist between steps

This separation is the central design decision of the repository.

---

## Future architecture extensions

Planned or possible extensions include:

- Retrieval and citation grounding
- Structured outputs with schema validation
- Resume-from-failure execution
- Token and cost tracking
- Lightweight evaluation harness for comparing variants
- Instrumentation hooks for monitoring

These extensions are intended to be layered in without rewriting Variant 1, so tradeoffs remain visible.
