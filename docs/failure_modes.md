# Failure Modes Across Orchestration Variants

This document describes known and anticipated failure modes of agentic-style AI workflows implemented in this repository.

The intent is not to argue that any variant is “safe,” but to make risks explicit, observable, and diagnosable using concrete artifacts produced by the pipelines.

---

## Risk Framing and Scope

The controls described in this repository are intended to:
- improve detectability of failures
- reduce the likelihood of specific failure modes
- support post-hoc analysis, audit, and review

They do not eliminate risk. Residual risk remains and must be addressed through human oversight, governance processes, appropriate use restrictions, and periodic review of system behavior.

---

## Variant 1: Baseline Tool Calling

Variant 1 represents a minimal implementation with direct model calls and explicit file outputs. It prioritizes simplicity and transparency of prompt and output artifacts, but includes few structural controls.

### 1. Ungrounded content risk (confident, plausible, wrong)

**What happens**  
The model can generate content that sounds authoritative even when it is incorrect, incomplete, or outdated, particularly for fast-moving or ambiguous topics.

**Why it happens**  
Variant 1 does not perform retrieval or verification. Outputs are generated solely from the model’s internal knowledge and immediate prompt inputs.

**How to detect it**
- Compare outputs against authoritative references.
- Look for generic phrasing that lacks specifics.
- Review prompt traces:
  - `outputs/<run_id>/topics/<topic_id>/prompt_trace.json`
  - `outputs/<run_id>/summary/prompt_trace.json`

**Mitigation options**
- Introduce retrieval and explicit grounding to reduce the likelihood of ungrounded claims.
- Require uncertain claims to be labeled as assumptions.
- Add structured outputs listing claims and supporting evidence.

**Residual risk**
- Even with improved prompts, factual correctness is not guaranteed without external verification.

---

### 2. Fake citations and invented sources

**What happens**  
If prompts request sources or citations without retrieval, the model may invent plausible-looking references.

**Why it happens**  
The model optimizes for helpful-looking outputs and cannot reliably cite real sources without retrieval.

**Why this matters**  
In regulated or high-stakes settings, invented sources create false confidence and can contaminate downstream decisions.

**How to detect it**
- Vague or generic references.
- Citations that cannot be independently verified.

**Mitigation options**
- Do not allow citations unless retrieval exists.
- Clearly label references as unverified follow-ups when applicable.

**Residual risk**
- Human reviewers may still overlook fabricated references.

---

### 3. Constraint illusion (requirements present in prompt, not enforced)

**What happens**  
Constraints such as tone, length, or structure are requested in prompts but not enforced programmatically.

**Why it happens**  
Constraints are expressed in natural language without validation or enforcement.

**How to detect it**
- Outputs violate requested length or structure.
- Prompt traces show constraints that are not reflected in outputs.

**Mitigation options**
- Add post-generation validation checks.
- Introduce structured output schemas.

**Residual risk**
- Validation logic may lag behind evolving requirements.

---

### 4. Topic blending during synthesis (cross-contamination)

**What happens**  
The synthesis step may merge multiple topics into a single narrative, blurring attribution.

**Why it happens**  
Synthesis prompts encourage compression unless attribution is explicitly required.

**How to detect it**
- Claims in the summary cannot be traced to a specific topic.
- Compare per-topic notes with synthesized output.

**Mitigation options**
- Require topic attribution during synthesis.
- Split synthesis into per-topic extraction followed by cross-topic synthesis.

**Residual risk**
- Compression tradeoffs may still obscure nuance.

---

### 5. Silent degradation from prompt drift

**What happens**  
Small prompt changes or model updates can significantly alter output quality without obvious failures.

**Why it happens**  
LLM behavior is sensitive to phrasing and ordering.

**How to detect it**
- Compare outputs across similar runs.
- Review prompt traces for subtle differences.

**Mitigation options**
- Treat prompts as versioned code.
- Add lightweight regression checks.

**Residual risk**
- Model behavior may drift even with unchanged prompts.

---

### 6. Partial runs and incomplete outputs

**What happens**  
Individual topics may fail while the overall run continues, producing incomplete outputs.

**Why it happens**  
The pipeline isolates errors per topic by design.

**How to detect it**
- Review `run_manifest.json` for completed vs requested topics.
- Inspect per-topic `error.txt` files.

**Mitigation options**
- Add retry logic for transient failures.
- Implement resume or re-run capabilities.

**Residual risk**
- Incomplete runs may be misinterpreted if not reviewed carefully.

---

### 7. Cost and latency surprises

**What happens**  
Model call counts and response sizes can grow quickly with topic count and optional outputs.

**Why it happens**  
Call volume scales with topics, synthesis, and short-form generation.

**How to detect it**
- Review `call_count` in `run_manifest.json`.
- Inspect tool call logs.

**Mitigation options**
- Set hard limits per run.
- Cache outputs for unchanged topics.

**Residual risk**
- Costs may still vary due to model or input variability.

---

### 8. Output variability and reproducibility limits

**What happens**  
Outputs vary across runs even with similar configurations.

**Why it happens**  
Non-zero temperature and model updates introduce randomness.

**How to detect it**
- Compare outputs and prompt traces across runs.

**Mitigation options**
- Lower temperature for research outputs.
- Store model identifiers consistently.

**Residual risk**
- Full determinism is not achievable.

---

## Variant 2: Local MCP-Orchestration

Variant 2 introduces a tool registry, explicit run context, and structured tool call logging. These controls improve observability and reproducibility but do not guarantee correctness.

### 1. Observability mistaken for correctness

**What happens**  
The presence of detailed logs can be misinterpreted as correctness or validation.

**Mitigation**
- Explicitly document that logs support diagnosability, not truth.

**Residual risk**
- Logs may not be reviewed in practice.

---

### 2. Audit surface drift

**What happens**  
Multiple audit artifacts can become inconsistent as schemas evolve.

**Mitigation**
- Define a minimal canonical audit record.

**Residual risk**
- Inconsistencies may persist unnoticed.

---

### 3. Context-schema coupling risk

**What happens**  
Tools develop hidden dependencies on evolving context fields.

**Mitigation**
- Version context schemas.
- Validate required fields at invocation.

**Residual risk**
- Backward compatibility issues may still arise.

---

### 4. Tool call artifact growth

**What happens**  
Tool call logging generates many small files.

**Mitigation**
- Optional logging modes.
- Aggregate logs when appropriate.

**Residual risk**
- Disk and operational overhead remains.

---

## Variant 3: Retrieval-Grounded Orchestration

Variant 3 introduces a local evidence pack, deterministic retrieval, and enforced grounding of factual claims. These controls reduce hallucinated claims but introduce new risks.

### 1. Evidence coverage gaps

**What happens**  
Grounded outputs may be factually correct but shallow if evidence is incomplete.

**Detection**
- Review `evidence.json` and “Evidence gaps” sections.

**Mitigation**
- Expand and curate evidence packs.
- Require explicit uncertainty when evidence is thin.

**Residual risk**
- Grounded outputs may still omit important context.

---

### 2. Faithful citation of low-quality evidence

**What happens**  
The system will accurately cite evidence even if the evidence itself is weak or biased.

**Mitigation**
- Curate evidence sources.
- Add provenance metadata where feasible.

**Residual risk**
- Citations may reinforce weak assumptions.

---

### 3. Retrieval scoring bias

**What happens**  
Simple scoring may favor repeated terms over relevance.

**Mitigation**
- Improve retrieval heuristics.
- Periodically review retrieval traces.

**Residual risk**
- Retrieval bias may persist in edge cases.

---

### 4. Conceptual ambiguity despite grounding

**What happens**  
Facts are grounded, but reasoning may remain incomplete or ambiguous.

**Mitigation**
- Treat grounded outputs as decision support, not decisions.

**Residual risk**
- Human interpretation errors remain possible.

---

## Control Mapping Summary

| Control | Variant Introduced | Risk Addressed | Limitations |
|------|------------------|--------------|-------------|
| Prompt trace logging | Variant 1 | Undetectable prompt drift | Does not validate correctness |
| Tool call logging | Variant 2 | Opaque tool usage | Logs may not be reviewed |
| Orchestration separation | Variant 2 | Ad hoc tool invocation | Policy errors repeat deterministically |
| Evidence retrieval | Variant 3 | Hallucinated factual claims | Depends on evidence quality |
| Citation enforcement | Variant 3 | Invented sources | Citations may reinforce weak evidence |

---

## What Grounding Solves (and What It Does Not)

Retrieval grounding significantly reduces hallucinated factual claims and invented citations.

It does not guarantee:
- correctness of evidence
- completeness of coverage
- quality of reasoning
- alignment with human intent

Grounding should be treated as an enabling control, not a correctness guarantee.
