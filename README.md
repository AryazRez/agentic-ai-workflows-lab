# Agentic AI Workflows Lab

## Problem statement

Modern AI systems increasingly rely on agentic workflows that coordinate language models, tools, and external services. New orchestration abstractions appear frequently, each promising cleaner integration, faster development, or better reliability.

The real problem is not choosing the newest abstraction.  
The real problem is understanding **how different orchestration and grounding choices change system behavior, failure modes, cost, and interpretability** when applied to the same task.

This repository explores that problem directly by holding the task constant and varying the orchestration and control strategy.

---

## Why this problem matters

In real-world environments, especially regulated or high-stakes settings, AI systems must remain:

- Auditable under scrutiny  
- Predictable under partial failure  
- Explicit about what is grounded versus inferred  
- Clear about where automation stops  

Many agentic demos optimize for speed or novelty.  
This project optimizes for **comparability, control, and judgment**.

---

## Who this repository is for

This repository is designed for two overlapping audiences:

**Engineers**
- Evaluating agent frameworks, orchestration layers, or grounding techniques  
- Debugging hallucinations, retries, or silent failures  
- Designing systems that must be observable and reproducible  

**Technical leaders**
- Assessing whether agentic systems are safe to deploy  
- Understanding cost, failure, and audit tradeoffs  
- Developing informed judgment beyond vendor demos  

No prior commitment to a specific framework is assumed.

---

## System overview

This repository implements the same AI-assisted workflow using multiple orchestration approaches and compares their behavior under identical inputs.

### Core task

**Grounded research and content synthesis**

**Input**
- A structured list of topics or questions

**Output**
- Grounded research notes with explicit citations  
- A synthesized cross-topic summary  
- Optional short-form content drafts (LinkedIn, YouTube Shorts)  

This task was chosen because it is:
- Non-clinical and non-proprietary  
- Sensitive to hallucination and context loss  
- Easy to evaluate qualitatively  
- Representative of real-world knowledge work  
- Reusable for communication and content generation  

---

## Orchestration and grounding variants

The same workflow is implemented using multiple strategies to expose tradeoffs.

### Variant 1: Baseline tool calling

A minimal implementation using direct LLM calls and explicit tool invocation.

**Characteristics**
- Simple control flow  
- Minimal abstraction  
- Explicit prompts and inputs  
- Easy to debug  
- Higher risk of duplicated logic and drift  

This variant serves as the control.

---

### Variant 2: MCP-style orchestration

The same workflow implemented using a Model Context Protocol style abstraction.

**Characteristics**
- Standardized tool schemas  
- Centralized tool definitions  
- Implicit state handling  
- Cleaner separation of concerns  
- Reduced visibility into intermediate decisions  

This variant explores what MCP simplifies and what it obscures.

---

### Variant 3: Grounded local-pack orchestration

A custom orchestration layer designed to enforce grounding and observability.

**Characteristics**
- Deterministic local evidence packs  
- Chunk-level retrieval with scoring and traces  
- Explicit validation gates that fail closed  
- Clear separation between evidence, inference, and unverified claims  
- Reproducible artifacts per run  

This variant prioritizes epistemic control over convenience.


---

## Repository structure

```
agentic-ai-workflows-lab/
├── README.md
├── docs/
│ ├── architecture.md
│ ├── tradeoffs.md
│ ├── failure_modes.md
│ ├── cost_and_latency.md
│ └── evidence_pack/
│ ├── manifest.json
│ └── sources/
├── shared/
│ ├── prompts.py
│ └── utils.py
├── baseline_tool_calling/
│ └── pipeline.py
├── mcp_orchestration/
│ └── pipeline.py
├── retrieval_grounded_orchestration/
│ └── pipeline.py
├── inputs/
│ └── run_config.json
├── outputs/
│ └── <run_id>/
└── requirements.txt
```


Each run produces a complete, inspectable artifact set including:
- retrieval traces  
- selected evidence  
- grounded research notes  
- synthesis prompts  
- short-form drafts  

---

## Grounding and validation model

Grounding is treated as a first-class system constraint.

Key properties:
- All claims must cite retrieved evidence chunks  
- Inferences are explicitly labeled  
- Missing or invalid evidence halts downstream synthesis  
- Retrieval and selection decisions are logged per topic  

The system is designed to fail loudly rather than hallucinate quietly.

---

## Example run artifacts

A single run produces:
- Topic-level evidence selection (`evidence.json`)  
- Retrieval scoring and traceability (`retrieval_trace.json`)  
- Grounded research notes (`research_notes.md`)  
- Cross-topic synthesis (`summary/brief.md`)  
- Platform-specific short-form drafts  

These artifacts are intended to be inspected, compared, and reused.

---

## Failure modes (intentionally documented)

This project explicitly documents where agentic systems break.

Examples include:
- Hallucinated synthesis when evidence is insufficient  
- Silent failures from empty or mis-tagged sources  
- Retry loops amplifying bad context  
- Cost escalation from uncontrolled retries  
- Loss of interpretability with deeper abstraction  

Failure analysis is treated as a first-class artifact, not a postmortem.

---

## Cost and latency considerations

Each orchestration strategy introduces different cost and latency profiles.

Factors explored include:
- Number of model calls  
- Prompt size growth  
- Retry behavior  
- Tool and retrieval overhead  

Measured observations and qualitative analysis are documented in `docs/cost_and_latency.md`.

---

## What this project is and is not

**This project is:**
- A controlled environment for reasoning about agentic systems  
- A demonstration of grounded AI content generation  
- A reference point for discussing orchestration tradeoffs  

**This project is not:**
- A framework  
- A product  
- A recommendation to deploy autonomous agents without oversight  

---

## Extension surface

This repository is intentionally extensible.

Possible extensions include:
- Additional orchestration strategies  
- Alternative grounding mechanisms  
- Structured evaluation metrics  
- Monitoring and observability hooks  
- Async execution variants  

The goal is not to chase tools, but to absorb change systematically.

---

## Final note

This repository is about making system behavior visible.

If you cannot explain how an agent reached an output,  
you should not trust it with decisions that matter.


