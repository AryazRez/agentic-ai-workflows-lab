# Tradeoffs

This document captures the intentional design tradeoffs made in this repository. Rather than optimizing for novelty or maximum capability, each variant is designed to make different system properties visible.

The purpose is not to declare a “best” approach, but to understand how different orchestration strategies affect transparency, control, and operational risk.

---

## Baseline philosophy

Variant 1 is deliberately simple. It uses direct model calls, explicit file outputs, and minimal abstraction.

This simplicity is not a limitation. It is the reference point against which all other variants are evaluated.

Later variants introduce orchestration layers. This document explains what is gained and what is lost as that happens.

---

## Tradeoff dimensions

The following dimensions are used consistently across variants.

---

## 1. Transparency vs abstraction

### Lower abstraction (Variant 1)
- Prompts, responses, and outputs are explicit.
- State transitions are visible on disk.
- Debugging is straightforward.
- Reviewers can inspect artifacts without understanding a framework.

**Cost**
- More boilerplate code.
- Repetition across steps.
- Less reuse across workflows.

### Higher abstraction (later variants)
- Cleaner high-level logic.
- Reusable components.
- Less code per workflow.

**Cost**
- State transitions become implicit.
- Failures may occur inside orchestration layers.
- Debugging often requires framework-specific knowledge.

---

## 2. Speed of development vs system clarity

### Faster iteration
- Abstractions reduce the amount of code needed to add features.
- New tools or agents can be plugged in quickly.

**Cost**
- It becomes harder to explain exactly what happened during a run.
- The system may “work” without being fully understood.

### Slower, explicit development
- Each step is designed deliberately.
- Failure handling is visible.
- Outputs map clearly to inputs.

**Cost**
- More upfront effort.
- Less immediate flexibility.

Variant 1 intentionally favors clarity over speed.

---

## 3. Flexibility vs enforceability

### Flexible outputs
- Freeform Markdown allows expressive responses.
- The model can adapt structure dynamically.

**Cost**
- No schema validation.
- Harder to programmatically assess correctness.
- Constraints are advisory, not enforced.

### Enforced structure (future variants)
- JSON schemas enable validation.
- Easier downstream automation.
- Clear failure signals when outputs are malformed.

**Cost**
- More brittle prompts.
- More work to design and maintain schemas.

Variant 1 accepts flexibility to surface where enforcement would matter.

---

## 4. Fault tolerance vs simplicity

### Simple failure handling (Variant 1)
- Per-topic failures do not crash the run.
- Errors are recorded and execution continues.

**Cost**
- No retries.
- No automatic recovery.
- Partial outputs are possible.

### More robust handling (future variants)
- Retries with backoff.
- Resume-from-failure support.
- Stronger guarantees about completeness.

**Cost**
- More complexity.
- More hidden state.
- Harder to reason about run boundaries.

Variant 1 intentionally avoids retries so failures are visible rather than masked.

---

## 5. Cost control vs completeness

### Minimal cost tracking
- Variant 1 records call count.
- Token usage and cost estimation are not tracked.

**Cost**
- Harder to predict spend.
- No automated budget enforcement.

### Detailed cost accounting (future variants)
- Token tracking per call.
- Cost per topic and per run.
- Budget-aware execution.

**Cost**
- Additional instrumentation.
- Dependency on provider metadata.

Variant 1 keeps cost transparent at a coarse level to avoid premature optimization.

---

## 6. Reproducibility vs adaptability

### Reproducibility
- Inputs and prompts are persisted.
- Outputs are stored per run.
- Manifests record metadata.

**Cost**
- Model behavior can still drift over time.
- Non-zero temperature introduces variability.

### Adaptability
- New models or prompts can be swapped easily.
- Behavior evolves as tooling evolves.

**Cost**
- Perfect reproducibility is not guaranteed.

This repository treats reproducibility as a goal, not an absolute guarantee.

---

## Summary view

| Dimension              | Variant 1 Bias |
|-----------------------|----------------|
| Transparency          | High           |
| Abstraction           | Low            |
| Debuggability         | High           |
| Flexibility           | High           |
| Enforcement           | Low            |
| Fault tolerance       | Basic          |
| Cost visibility       | Coarse         |
| Reproducibility       | Partial        |

Later variants intentionally move these sliders in different directions.

---

## Why these tradeoffs matter

In regulated, high-stakes, or long-lived systems, the wrong tradeoff is often more dangerous than an underpowered system.

This repository exists to make those tradeoffs explicit, observable, and comparable, rather than implicit or assumed.

---

## What comes next

- Variant 2 introduces orchestration via MCP-style abstractions.
- Variant 3 introduces a minimal custom orchestration layer.

Each variant will be evaluated against the same tradeoff dimensions described here.

---

## Evidence from Variant 1 vs Variant 2 runs

This section documents concrete differences observed between Variant 1 (baseline) and Variant 2 (local MCP-style orchestration), based on actual run artifacts committed during development.

The goal is not to declare a winner. The goal is to show what measurably changed.

---

## 1. Observability and auditability

### Variant 1
- Observability is limited to prompt traces per step.
- Debugging requires reading prompt text and inferred intent.
- There is no explicit record of “system intent” beyond code structure.

Artifacts:
- `topics/<topic_id>/prompt_trace.json`
- `summary/prompt_trace.json`
- `short_form/*_prompt_trace.json`

### Variant 2
- Introduces explicit tool call logs under `tool_calls/`.
- Each tool invocation records structured inputs and outputs.
- Execution order is explicit and inspectable.

Artifacts:
- `tool_calls/001_generate_topic_notes_input.json`
- `tool_calls/003_synthesize_notes_input.json`
- Corresponding output records

**Evidence**
- Variant 2 enables inspection of what the system attempted to do, not just what the model said.
- This improves diagnosability without changing core behavior.

**Tradeoff**
- Increased artifact volume.
- More surfaces to keep consistent.

---

## 2. Control of topic blending during synthesis

### Variant 1
- Synthesis input is a concatenated block of text.
- Topic boundaries are implicit.
- Claims in the summary are not easily attributable to source topics.

Observed behavior:
- Cross-topic blending without attribution.
- Difficult to determine which topic supported which claim.

### Variant 2
- Synthesis input is explicitly topic-keyed:
  - `topic_notes_by_id: { "mcp_basics": ..., "agent_failures": ... }`
- Prompts require per-topic takeaways and explicit attribution.

Observed behavior:
- Summary sections reference topic ids.
- Cross-topic connections are more transparent.
- Blending is visible rather than hidden.

**Evidence**
- `tool_calls/003_synthesize_notes_input.json` shows labeled inputs.
- `summary/brief.md` reflects topic-aware structure.

**Tradeoff**
- More verbose synthesis prompts.
- Slightly more rigid output structure.

---

## 3. Handling of uncertainty and overclaiming

### Variant 1
- Outputs often present speculative or generic statements as fact.
- When asked for sources, the model may invent plausible references.
- Uncertainty is implicit and easy to miss.

Observed behavior:
- Fabricated “Sources” sections.
- Confident tone even when claims are not verifiable.

### Variant 2 (tightened prompts)
- Explicit separation of:
  - facts
  - inferences
  - uncertainties
- Claims are accompanied by confidence ratings.
- Verification targets are listed without pretending validation.

Observed behavior:
- No invented citations.
- Claims marked Medium or Low confidence where appropriate.
- Clear articulation of what would need external verification.

**Evidence**
- `topics/mcp_basics/research_notes.md` includes:
  - “Claims and confidence” table
  - “What I would verify first” section
- No “Sources” sections appear in outputs.

**Tradeoff**
- Content is more conservative.
- Outputs may feel less “polished” or authoritative.

---

## 4. System behavior vs model capability

### What did not change
- Model used (`gpt-4o-mini`)
- Number of calls per run
- Overall latency characteristics
- Core task semantics

### What changed
- Structure of inputs to synthesis
- Explicitness of system intent
- Ability to audit and critique outputs post-run

**Conclusion**
Variant 2 did not make the model smarter.  
It made overclaiming easier to detect and easier to challenge.

---

## 5. Why this matters in high-stakes environments

In regulated or high-stakes settings:
- Silent overclaiming is more dangerous than explicit uncertainty.
- Auditability matters as much as output quality.
- Systems must support post hoc review, not just generation.

Variant 2 demonstrates that:
- MCP-style orchestration can improve system-level discipline
- without granting autonomy
- and without hiding control flow.

This is a meaningful, evidence-backed tradeoff rather than a theoretical benefit.

---

## Summary judgment

Variant 1 favors simplicity and speed of understanding.

Variant 2 introduces additional structure and observability that:
- improves diagnosability,
- reduces hidden failure modes,
- and forces intellectual honesty from the system.

Whether that tradeoff is worth it depends on the operating context. This repository exists to make that decision explicit rather than assumed.
