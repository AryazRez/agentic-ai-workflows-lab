from __future__ import annotations


def build_research_prompt(topic) -> str:
    """
    Topic research prompt.

    Contract:
    - No citations, links, or "Sources" section.
    - Do not invent standards, papers, or references.
    - Distinguish clearly between facts, inferences, and uncertainty.
    """
    title = getattr(topic, "title", "")
    question = getattr(topic, "question", "")
    constraints = getattr(topic, "constraints", "")

    prompt = f"""
You are producing research notes for a deterministic pipeline.

Topic ID: {topic.id}
Title: {title}
Question: {question}

Constraints (if any):
{constraints}

Hard rules:
- Do NOT include citations, links, or a "Sources" section.
- Do NOT invent papers, standards, or references.
- Do NOT present speculative features as if they are definitive.
- If you are not confident a claim is true, label it as an inference or uncertainty.

Write research notes in exactly this structure:

## Key concepts (facts only)
- 3 to 7 bullets
- Only include statements you would be comfortable defending without external lookup
- If you cannot defend it, do not put it here

## Practical implications (inferences allowed)
- 3 to 7 bullets
- If a point is an inference, prefix it with "Inference:"
- Keep implications tied tightly to the key concepts above

## Risks and failure modes
- 3 to 7 bullets
- For each bullet, include a short "Why:" clause

## Claims and confidence
Create a table with 5 to 10 claims. Each row must include:
- Claim (one sentence)
- Confidence (High, Medium, Low)
- Basis (Fact from notes, or Inference)

Format:

| Claim | Confidence | Basis |
|---|---|---|
| ... | ... | ... |

## What I would verify first
- 3 to 5 bullets describing what you would check externally to increase confidence
- These are not sources, just verification targets (for example: official spec existence, terminology, scope)

Be concise and high signal. Use markdown headings and bullets only.
""".strip()

    return prompt


def build_synthesis_prompt(all_notes: str) -> str:
    """
    Synthesis prompt.

    Contract:
    - No citations, links, or "Sources" section.
    - Do not add external facts not supported by the provided notes.
    - Separate fact vs inference.
    - Attribute cross-topic claims by topic id.
    """
    prompt = f"""
You are synthesizing research notes from multiple topics for a deterministic pipeline.

Hard rules:
- Do NOT include citations, links, or a "Sources" section.
- Do NOT add external facts that are not supported by the provided notes.
- If you make an inference, label it explicitly as "Inference:".
- If something is unclear, say so.

Input notes (topic-labeled):
{all_notes}

Write the synthesis in this structure:

## Per-topic takeaways
For each topic id, include:
### <topic_id>
- 3 to 6 bullets
- Prefer facts over inferences
- If an item is an inference, prefix it with "Inference:"

## Cross-topic synthesis
- 5 to 9 bullets connecting ideas across topics
- Reference the topic ids in parentheses, for example: (mcp_basics, agent_failures)
- If a connection is speculative, prefix with "Inference:"

## Risks and failure modes
- 5 to 9 bullets summarizing the most important risks
- Ground each risk in at least one topic, referencing topic ids in parentheses

## Open questions
- 3 to 7 bullets listing what is missing, unclear, or would require verification
- This should be specific, not generic

Keep it sharp. Avoid generic filler.
""".strip()

    return prompt


def build_short_form_prompt(platform: str, summary_text: str) -> str:
    """
    Short-form generation prompt.

    Contract:
    - No citations, links, or sources.
    - Must be derived from the provided summary only.
    """
    platform_norm = (platform or "").strip().lower()

    if platform_norm == "linkedin":
        style = """
Write a LinkedIn post:
- 120 to 220 words
- Clear hook in the first 2 lines
- 3 to 6 short paragraphs or bullets
- Practical, non-hype tone
- End with 1 question to invite discussion
- No hashtags required
""".strip()
    elif platform_norm == "youtube_shorts":
        style = """
Write a YouTube Shorts script:
- 35 to 60 seconds
- Tight pacing, short sentences
- Start with a strong hook in the first 2 seconds
- Include 1 concrete example
- End with a simple call to action
""".strip()
    else:
        style = """
Write a short-form piece:
- concise
- practical
- derived from the provided summary
""".strip()

    prompt = f"""
You are generating short-form content from a provided summary for a deterministic pipeline.

Hard rules:
- Do NOT include citations, links, or a "Sources" section.
- Do NOT introduce new facts not present in the summary.
- If the summary is vague, stay faithful and do not fabricate details.

Platform: {platform}

Style requirements:
{style}

Summary to use:
{summary_text}

Output only the final draft. No meta commentary.
""".strip()

    return prompt
