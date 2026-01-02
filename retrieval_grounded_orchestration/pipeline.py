from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from openai import OpenAI

from shared.schemas import TopicSpec, OutputConfig, RunConfig
from shared.utils import ensure_dir, read_json, write_json, write_text


CLIENT = OpenAI()

CHUNK_HEADER_RE = re.compile(r"^##\s+Chunk\s+(?P<chunk_id>[a-zA-Z0-9_]+)\s*$", re.IGNORECASE)


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def abs_path(rel_or_abs: str) -> str:
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.join(repo_root(), rel_or_abs)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_topic_title(topic: TopicSpec) -> str:
    # Use whatever your schema provides, fall back to id
    if hasattr(topic, "title") and getattr(topic, "title"):
        return getattr(topic, "title")
    if hasattr(topic, "name") and getattr(topic, "name"):
        return getattr(topic, "name")
    return getattr(topic, "id", "topic")


def get_topic_question(topic: TopicSpec) -> str:
    # Prefer question, then prompt, then description, then empty string
    for field in ["question", "prompt", "description"]:
        if hasattr(topic, field) and getattr(topic, field):
            return getattr(topic, field)
    return ""


# ----------------------------
# Evidence pack loading + retrieval
# ----------------------------

def validate_evidence_pack(pack: Dict, manifest_path: str) -> None:
    sources = pack.get("sources", [])
    if not sources:
        raise RuntimeError(f"Evidence pack has no sources: {abs_path(manifest_path)}")

    bad = []
    for src in sources:
        source_id = src.get("source_id", "<missing_source_id>")
        rel_path = src.get("path")

        if not rel_path:
            bad.append({"source_id": source_id, "issue": "missing path"})
            continue

        ap = abs_path(rel_path)
        if not os.path.exists(ap):
            bad.append({"source_id": source_id, "issue": "file not found", "abs_path": ap})
            continue

        if os.path.getsize(ap) == 0:
            bad.append({"source_id": source_id, "issue": "file is empty (0 bytes)", "abs_path": ap})

    if bad:
        raise RuntimeError(
            "Evidence pack validation failed. Fix manifest paths or source files. "
            f"manifest={abs_path(manifest_path)} bad_sources={bad}"
        )


def load_evidence_pack(manifest_path: str) -> Dict:
    manifest_abs = abs_path(manifest_path)
    if not os.path.exists(manifest_abs):
        raise FileNotFoundError(f"Evidence pack manifest not found: {manifest_abs}")
    with open(manifest_abs, "r", encoding="utf-8") as f:
        pack = json.load(f)

    validate_evidence_pack(pack, manifest_path)
    return pack


def validate_source_file(source_abs: str) -> Tuple[bool, str]:
    """
    Returns (ok, reason). ok=False if file is missing, empty, or has no chunk headers.
    """
    if not os.path.exists(source_abs):
        return False, f"Evidence source not found: {source_abs}"

    try:
        size = os.path.getsize(source_abs)
    except OSError:
        size = -1

    if size == 0:
        return False, f"Evidence source is empty (0 bytes): {source_abs}"

    # Check for at least one chunk header to enforce the format contract
    has_chunk_header = False
    with open(source_abs, "r", encoding="utf-8") as f:
        for line in f:
            if CHUNK_HEADER_RE.match(line.strip()):
                has_chunk_header = True
                break

    if not has_chunk_header:
        return False, (
            "Evidence source has content but no chunk headers. "
            "Expected lines like '## Chunk c_001'. "
            f"Source: {source_abs}"
        )

    return True, ""


def load_source_chunks(source_path: str) -> List[Dict]:
    source_abs = abs_path(source_path)

    ok, reason = validate_source_file(source_abs)
    if not ok:
        raise FileNotFoundError(reason)

    chunks: List[Dict] = []
    current_id = None
    current_lines: List[str] = []

    with open(source_abs, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n")
            m = CHUNK_HEADER_RE.match(stripped.strip())
            if m:
                if current_id is not None:
                    text = "\n".join(current_lines).strip()
                    if text:
                        chunks.append({"chunk_id": current_id, "text": text})
                current_id = m.group("chunk_id")
                current_lines = []
            else:
                if current_id is not None:
                    current_lines.append(stripped)

    if current_id is not None:
        text = "\n".join(current_lines).strip()
        if text:
            chunks.append({"chunk_id": current_id, "text": text})

    return chunks


def tokenize_query(query: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9_]+", (query or "").lower())
    return [t for t in tokens if len(t) >= 3]


def score_chunk(query_terms: List[str], chunk_text: str) -> int:
    text = (chunk_text or "").lower()
    score = 0
    for term in query_terms:
        score += text.count(term)
    return score


def retrieve_evidence_for_topic(
    topic_id: str,
    query: str,
    top_k: int,
    manifest_path: str = "docs/evidence_pack/manifest.json",
) -> Tuple[Dict, Dict]:
    pack = load_evidence_pack(manifest_path)
    sources = pack.get("sources", [])

    query_terms = tokenize_query(query)
    candidates: List[Dict] = []
    invalid_tagged_sources: List[Dict] = []

    for src in sources:
        if topic_id not in (src.get("topic_tags") or []):
            continue

        source_id = src.get("source_id", "")
        source_path = src.get("path", "")
        title = src.get("title", "")

        # Validate and load chunks. If invalid, collect detail for a better error.
        source_abs = abs_path(source_path)
        ok, reason = validate_source_file(source_abs)
        if not ok:
            invalid_tagged_sources.append(
                {"source_id": source_id, "abs_path": source_abs, "reason": reason}
            )
            continue

        chunks = load_source_chunks(source_path)
        for ch in chunks:
            chunk_id = ch["chunk_id"]
            text = ch["text"]
            s = score_chunk(query_terms, text)
            candidates.append(
                {
                    "source_id": source_id,
                    "chunk_id": chunk_id,
                    "score": s,
                    "text": text,
                    "path": source_path,
                    "title": title,
                }
            )

    if not candidates:
        detail = ""
        if invalid_tagged_sources:
            detail = f" Likely causes: Tagged sources exist but are invalid: {invalid_tagged_sources}"
        raise RuntimeError(
            f"No evidence candidates found for topic_id='{topic_id}'."
            f"{detail} Check docs/evidence_pack/manifest.json topic_tags and source paths."
        )

    candidates_sorted = sorted(candidates, key=lambda x: (-x["score"], x["source_id"], x["chunk_id"]))
    selected = candidates_sorted[: max(1, top_k)]

    retrieved_at = utc_now_iso()

    evidence_json = {
        "topic_id": topic_id,
        "query": query,
        "retrieved_at": retrieved_at,
        "pack_id": pack.get("pack_id", ""),
        "selected_chunks": [
            {"source_id": c["source_id"], "chunk_id": c["chunk_id"], "title": c["title"], "text": c["text"]}
            for c in selected
        ],
    }

    retrieval_trace_json = {
        "topic_id": topic_id,
        "query": query,
        "retrieved_at": retrieved_at,
        "pack_id": pack.get("pack_id", ""),
        "query_terms": query_terms,
        "invalid_tagged_sources": invalid_tagged_sources,
        "candidates": [
            {
                "source_id": c["source_id"],
                "chunk_id": c["chunk_id"],
                "score": c["score"],
                "title": c["title"],
                "path": abs_path(c["path"]),
            }
            for c in candidates_sorted
        ],
        "selected": [{"source_id": c["source_id"], "chunk_id": c["chunk_id"], "score": c["score"]} for c in selected],
    }

    return evidence_json, retrieval_trace_json


# ----------------------------
# Prompts (grounded)
# ----------------------------

def build_grounded_topic_prompt(topic: TopicSpec, evidence: Dict) -> str:
    title = get_topic_title(topic)
    question = get_topic_question(topic)

    chunks = evidence.get("selected_chunks", [])
    formatted = []
    for c in chunks:
        formatted.append(f"[{c['chunk_id']}] {c['text']}")
    evidence_block = "\n\n".join(formatted).strip()

    return f"""
You are producing grounded research notes for a deterministic pipeline.

Topic ID: {topic.id}
Title: {title}
Question: {question}

Hard rules:
- Use ONLY the evidence chunks provided below.
- Do NOT add external facts.
- Every factual claim must include at least one chunk citation in parentheses, e.g. (c_001).
- If the evidence is insufficient, say so explicitly and mark claims as Unverified.

Evidence chunks:
{evidence_block}

Write in this structure:

## Key concepts (grounded)
- 3 to 7 bullets
- Each bullet must include citations like (c_001)

## Practical implications (inferences allowed)
- 3 to 7 bullets
- If inference, prefix with "Inference:"
- Still cite supporting chunks

## Risks and failure modes
- 3 to 7 bullets
- Each bullet must be: "Risk: ... Why: ... (c_###)"
- If you cannot cite a risk, prefix the whole bullet with "Inference:" and do not pretend it is grounded

## Claims table
| Claim | Confidence (High/Medium/Low) | Basis (Supported/Inference/Unverified) | Evidence |
|---|---|---|---|
| ... | ... | ... | c_001, c_003 |

## Evidence gaps
- 3 to 6 bullets listing what is missing from the evidence pack for this topic

No links. No sources section. Output markdown only.
""".strip()


def build_grounded_synthesis_prompt(topic_notes_by_id: Dict[str, str]) -> str:
    parts = []
    for tid, notes in topic_notes_by_id.items():
        parts.append(f"## Topic: {tid}\n{notes}")
    all_notes = "\n\n".join(parts)

    return f"""
You are synthesizing grounded notes from multiple topics.

Hard rules:
- Do NOT introduce new facts. Use only what is in the topic notes.
- Preserve citations present in the notes.
- If a statement has no citation available, label it "Unverified:".

Input topic notes:
{all_notes}

Write the synthesis in this structure:

## Per-topic takeaways
For each topic:
### <topic_id>
- 3 to 6 bullets
- Keep citations like (c_001)

## Cross-topic synthesis
- 5 to 9 bullets connecting topics
- Reference topic ids in parentheses like (mcp_basics, agent_failures)
- Keep citations where applicable

## Risks and failure modes
- 5 to 9 bullets
- Keep citations where applicable

## Open questions
- 3 to 7 bullets describing what is missing and would require better evidence

Output markdown only.
""".strip()


def build_short_form_prompt(platform: str, summary_text: str) -> str:
    platform_norm = (platform or "").strip().lower()

    if platform_norm == "linkedin":
        style = """
Write a LinkedIn post:
- 120 to 220 words
- Clear hook in the first 2 lines
- Practical, non-hype tone
- End with 1 question
- Do not include links
""".strip()
    elif platform_norm == "youtube_shorts":
        style = """
Write a YouTube Shorts script:
- 35 to 60 seconds
- Tight pacing
- Hook in first 2 seconds
- Include 1 concrete example
- No links
""".strip()
    else:
        style = "Write a concise short-form draft. No links."

    return f"""
You are generating short-form content from a grounded summary.

Hard rules:
- Use ONLY the summary provided.
- Do NOT add new facts.
- Do NOT include links or sources.

Platform: {platform}

Style:
{style}

Summary:
{summary_text}

Output only the draft.
""".strip()


def call_llm(model: str, prompt: str, temperature: float = 0.2) -> str:
    resp = CLIENT.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You are precise and follow grounding and formatting rules."},
            {"role": "user", "content": prompt},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


# ----------------------------
# Pipeline
# ----------------------------

def run_pipeline(config_path: str = "inputs/run_config.json") -> None:
    config_dict = read_json(config_path)
    model_name = config_dict.get("model", "gpt-4o-mini")

    topics = [TopicSpec(**t) for t in config_dict["topics"]]
    output = OutputConfig(**config_dict["output"])

    config = RunConfig(
        run_id=config_dict["run_id"],
        topics=topics,
        output=output,
    )

    run_id = config.run_id
    base_out = os.path.join("outputs", run_id)
    ensure_dir(base_out)

    manifest = {
        "run_id": run_id,
        "timestamp": utc_now_iso(),
        "model": model_name,
        "variant": "retrieval_grounded_local_pack",
        "topics_requested": len(config.topics),
        "topics_completed": 0,
        "call_count": 0,
        "short_form_enabled": bool(config.output.short_form_platforms),
        "short_form_platforms": config.output.short_form_platforms or [],
        "errors": [],
        "tools_available": [
            "retrieve_evidence",
            "generate_grounded_topic_notes",
            "synthesize_grounded_notes",
            "generate_short_form",
        ],
        "evidence_pack_manifest": abs_path("docs/evidence_pack/manifest.json"),
    }

    write_json(os.path.join(base_out, "run_config.json"), asdict(config))
    ensure_dir(os.path.join(base_out, "topics"))
    ensure_dir(os.path.join(base_out, "summary"))
    ensure_dir(os.path.join(base_out, "short_form"))
    ensure_dir(os.path.join(base_out, "tool_calls"))

    topic_notes_by_id: Dict[str, str] = {}

    for idx, topic in enumerate(config.topics, start=1):
        topic_dir = os.path.join(base_out, "topics", topic.id)
        ensure_dir(topic_dir)

        try:
            query = f"{topic.id} {get_topic_question(topic)}".strip()

            evidence_json, retrieval_trace_json = retrieve_evidence_for_topic(
                topic_id=topic.id,
                query=query,
                top_k=3,
                manifest_path="docs/evidence_pack/manifest.json",
            )

            write_json(os.path.join(topic_dir, "evidence.json"), evidence_json)
            write_json(os.path.join(topic_dir, "retrieval_trace.json"), retrieval_trace_json)

            write_json(
                os.path.join(base_out, "tool_calls", f"{idx:03d}_retrieve_evidence_input.json"),
                {"tool": "retrieve_evidence", "input": {"topic_id": topic.id, "query": query, "top_k": 3}},
            )
            write_json(
                os.path.join(base_out, "tool_calls", f"{idx:03d}_retrieve_evidence_output.json"),
                {"tool": "retrieve_evidence", "output": {"topic_id": topic.id, "selected": evidence_json["selected_chunks"]}},
            )

            prompt = build_grounded_topic_prompt(topic, evidence_json)

            write_json(
                os.path.join(base_out, "tool_calls", f"{idx:03d}_generate_grounded_topic_notes_input.json"),
                {
                    "tool": "generate_grounded_topic_notes",
                    "input": {
                        "topic_id": topic.id,
                        "selected_chunk_ids": [c["chunk_id"] for c in evidence_json["selected_chunks"]],
                    },
                },
            )

            notes = call_llm(model_name, prompt)
            manifest["call_count"] += 1

            write_text(os.path.join(topic_dir, "research_notes.md"), notes)
            write_json(os.path.join(topic_dir, "prompt_trace.json"), {"prompt": prompt, "response": notes})

            write_json(
                os.path.join(base_out, "tool_calls", f"{idx:03d}_generate_grounded_topic_notes_output.json"),
                {"tool": "generate_grounded_topic_notes", "output": {"topic_id": topic.id, "notes_path": os.path.join(topic_dir, "research_notes.md")}},
            )

            topic_notes_by_id[topic.id] = notes
            manifest["topics_completed"] += 1

        except Exception as e:
            err_msg = str(e)
            manifest["errors"].append({"topic_id": topic.id, "error": err_msg})
            write_text(os.path.join(topic_dir, "error.txt"), err_msg)

    if manifest["topics_completed"] == 0:
        manifest["errors"].append(
            {"stage": "halt", "error": "No topics completed. Skipping synthesis and short-form to avoid misleading outputs."}
        )
        write_json(os.path.join(base_out, "run_manifest.json"), manifest)
        return

    brief = ""
    try:
        synth_prompt = build_grounded_synthesis_prompt(topic_notes_by_id)

        write_json(
            os.path.join(base_out, "tool_calls", "900_synthesize_grounded_notes_input.json"),
            {"tool": "synthesize_grounded_notes", "input": {"topic_ids": list(topic_notes_by_id.keys())}},
        )

        brief = call_llm(model_name, synth_prompt)
        manifest["call_count"] += 1

        write_text(os.path.join(base_out, "summary", "brief.md"), brief)
        write_json(os.path.join(base_out, "summary", "prompt_trace.json"), {"prompt": synth_prompt, "response": brief})

        write_json(
            os.path.join(base_out, "tool_calls", "900_synthesize_grounded_notes_output.json"),
            {"tool": "synthesize_grounded_notes", "output": {"brief_path": os.path.join(base_out, "summary", "brief.md")}},
        )

    except Exception as e:
        manifest["errors"].append({"stage": "synthesis", "error": str(e)})

    if brief and manifest["short_form_enabled"]:
        for i, platform in enumerate(manifest["short_form_platforms"], start=1):
            try:
                sf_prompt = build_short_form_prompt(platform, brief)

                write_json(
                    os.path.join(base_out, "tool_calls", f"950_generate_short_form_{i:02d}_input.json"),
                    {"tool": "generate_short_form", "input": {"platform": platform}},
                )

                draft = call_llm(model_name, sf_prompt)
                manifest["call_count"] += 1

                out_path = os.path.join(base_out, "short_form", f"{platform}.md")
                trace_path = os.path.join(base_out, "short_form", f"{platform}_prompt_trace.json")

                write_text(out_path, draft)
                write_json(trace_path, {"prompt": sf_prompt, "response": draft})

                write_json(
                    os.path.join(base_out, "tool_calls", f"950_generate_short_form_{i:02d}_output.json"),
                    {"tool": "generate_short_form", "output": {"platform": platform, "path": out_path}},
                )

            except Exception as e:
                manifest["errors"].append({"stage": "short_form", "platform": platform, "error": str(e)})

    write_json(os.path.join(base_out, "run_manifest.json"), manifest)


if __name__ == "__main__":
    run_pipeline("inputs/run_config.json")
