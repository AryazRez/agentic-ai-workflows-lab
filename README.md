# Agentic AI Workflows Lab

## Problem statement

Modern AI systems increasingly rely on agentic workflows that coordinate language models, tools, and external services. New orchestration abstractions appear frequently, each promising cleaner integration, faster development, or better reliability.

The problem is not choosing the newest abstraction.  
The problem is understanding **how different orchestration approaches change system behavior, failure modes, cost, and interpretability** when applied to the same task.

This repository explores that problem directly.

---

## Why this problem matters

In real-world environments, especially regulated or high-stakes settings, AI systems must remain:

- Understandable under audit  
- Predictable under failure  
- Adaptable as tooling evolves  
- Explicit about what is automated versus assistive  

Many agentic demos optimize for speed or novelty. This project optimizes for **comparability and judgment**.

---

## System overview

This repository implements the same AI-assisted workflow using multiple orchestration approaches and compares their behavior.

### Core task

**AI-assisted research and content synthesis**

**Input**
- A structured list of topics or questions

**Output**
- Structured research notes  
- A concise synthesized summary  
- Optional short-form content drafts  

This task was chosen because it is:
- Non-clinical and non-proprietary  
- Useful in practice  
- Sensitive to hallucination and context loss  
- Easy to evaluate qualitatively  
- Reusable for learning and content generation  

---

## Orchestration variants

The same workflow is implemented using three different orchestration strategies.

### Variant 1: Baseline tool calling

A minimal implementation using direct LLM calls and explicit tool invocation.

**Characteristics**
- Simple control flow  
- Minimal abstraction  
- Explicit prompts and inputs  
- Easy to debug  
- Higher risk of duplicated logic  

This variant serves as the **control**.

---

### Variant 2: MCP-based orchestration

The same workflow implemented using a Model Context Protocol (MCP) style abstraction.

**Characteristics**
- Standardized tool schemas  
- Centralized tool definitions  
- Implicit state handling  
- Cleaner separation of concerns  
- Additional abstraction layers  

This variant explores what MCP simplifies and what it obscures.

---

### Variant 3: Minimal custom orchestration layer

A thin, custom orchestration layer designed explicitly for this workflow.

**Characteristics**
- Explicit state management  
- Explicit retries and fallbacks  
- Minimal indirection  
- Higher upfront design cost  
- Clear ownership of failure handling  

This variant prioritizes transparency and control.

---

## Repository structure

agentic-ai-workflows-lab/
├── README.md
├── docs/
│ ├── architecture.md
│ ├── tradeoffs.md
│ ├── failure_modes.md
│ └── cost_and_latency.md
├── shared/
│ ├── schemas.py
│ ├── prompts.py
│ └── utils.py
├── baseline_tool_calling/
│ └── pipeline.py
├── mcp_orchestration/
│ └── pipeline.py
├── minimal_custom_layer/
│ └── pipeline.py
├── experiments/
│ └── comparisons.ipynb
└── requirements.txt



Each orchestration variant produces functionally equivalent outputs to enable meaningful comparison.

---

## Tradeoff summary (conceptual)

| Dimension           | Baseline Tool Calling | MCP Orchestration | Custom Layer |
|--------------------|-----------------------|-------------------|--------------|
| Transparency       | High                  | Medium            | High         |
| Setup complexity   | Low                   | Medium            | Medium       |
| Debuggability      | High                  | Medium            | High         |
| Abstraction reuse  | Low                   | High              | Medium       |
| Failure visibility | High                  | Medium            | High         |
| Adaptability       | Medium                | High              | High         |

Detailed discussion is provided in `docs/tradeoffs.md`.

---

## Failure modes (intentionally documented)

This project explicitly documents where agentic systems break.

Examples include:
- Hallucinated tool outputs  
- Silent failures due to schema mismatch  
- Retry loops amplifying bad context  
- Cost escalation from repeated calls  
- Latency spikes from over-orchestration  
- Loss of interpretability with deeper abstraction  

Failure analysis is treated as a first-class artifact.

---

## What I would and would not deploy

### I would deploy
- Assistive workflows with human review  
- Explicitly bounded automation  
- Systems with observable intermediate states  
- Pipelines with documented failure handling  

### I would not deploy
- Fully autonomous decision-making systems  
- Agent chains without auditability  
- Orchestration layers that hide state transitions  
- Systems where failure is silent or irreproducible  

This project is designed to make these boundaries visible.

---

## Cost and latency considerations

Each orchestration variant introduces different cost and latency profiles.

Key factors explored:
- Number of model calls  
- Prompt size growth  
- Retry behavior  
- Tool invocation overhead  

Measured observations and qualitative analysis are documented in `docs/cost_and_latency.md`.

---

## Future experiments

This repository is intentionally extensible.

Planned or potential extensions include:
- Adding additional orchestration variants  
- Swapping model providers  
- Introducing structured evaluation metrics  
- Adding lightweight monitoring hooks  
- Comparing synchronous versus asynchronous execution  

The goal is not to chase tools, but to **absorb change systematically**.

---

## Final note

This repository is not a framework and not a product.  
It is a controlled environment for developing and demonstrating judgment about agentic AI systems as orchestration tooling evolves.

---

