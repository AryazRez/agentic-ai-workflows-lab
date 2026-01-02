# Common agentic AI failure modes overview (local pack)

This file is a local evidence source. It is not authoritative by itself.
It exists to validate the mechanics of grounding, citation, and traceability.

## Chunk c_101
Misalignment: the system optimizes for a goal proxy that diverges from user intent, particularly when objectives are underspecified.

## Chunk c_102
Over-optimization: the system maximizes a metric and creates unintended side effects. Guardrails often fail when they are only described in prompts.

## Chunk c_103
Robustness failures: small changes in inputs cause disproportionate changes in outputs. This can appear as brittle behavior, tool misuse, or unsafe action selection.

## Chunk c_104
Explainability gaps: complex systems produce decisions that are hard to justify, which erodes trust and makes oversight difficult.

## Chunk c_105
Feedback loops: outputs influence future inputs and reinforce undesirable behavior, including bias amplification and reward hacking dynamics.
