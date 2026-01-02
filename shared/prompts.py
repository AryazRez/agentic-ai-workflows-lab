def build_research_prompt(topic):
    return f"""
You are assisting with technical research.

Question:
{topic.question}

Audience:
{topic.audience}

Constraints:
- Tone: {topic.constraints.get("tone", "neutral")}
- Max sources: {topic.constraints.get("max_sources", 5)}

Provide:
- Key concepts
- Practical implications
- Areas of uncertainty

Avoid hype. Be concise and factual.
""".strip()


def build_synthesis_prompt(all_notes: str):
    return f"""
You are synthesizing multiple research notes.

Notes:
{all_notes}

Produce:
- Key themes
- Common failure modes or risks
- Practical takeaways

Be clear and structured.
""".strip()


def build_short_form_prompt(platform: str, summary: str):
    return f"""
Create a short-form content draft for {platform}.

Source summary:
{summary}

Format:
- Hook (1 sentence)
- 3 bullet points
- Closing line

Avoid emojis. Avoid hype.
""".strip()
