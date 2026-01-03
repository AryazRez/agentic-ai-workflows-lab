from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------------------
# Paths and helpers
# --------------------------------------------------------------------------------------

def repo_root() -> Path:
    # Assumes this file lives at: <repo>/retrieval_grounded_orchestration/pipeline.py
    return Path(__file__).resolve().parents[1]


def abs_path(*parts: str) -> Path:
    return repo_root().joinpath(*parts).resolve()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_text(p: Path, s: str) -> None:
    ensure_dir(p.parent)
    p.write_text(s, encoding="utf-8")


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def write_json(p: Path, obj: Any) -> None:
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(p: Path) -> Any:
    return json.loads(read_text(p))


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    evidence_pack_manifest_relpath: str = "docs/evidence_pack/manifest.json"
    outputs_dir_relpath: str = "outputs"

    # "strict": fail topic on grounding validation failure
    # "soft": prefix offending bullets with "Unverified:" and continue
    grounding_mode: str = "strict"

    # If True, pipeline will stop after topic generation if zero topics complete
    halt_if_no_topics_completed: bool = True


# --------------------------------------------------------------------------------------
# Evidence pack validation
# --------------------------------------------------------------------------------------

def validate_source_file(source_file: Path) -> None:
    if not source_file.exists():
        raise FileNotFoundError(f"Evidence pack source file not found: {source_file}")
    if not source_file.is_file():
        raise ValueError(f"Evidence pack source path is not a file: {source_file}")
    if source_file.stat().st_size == 0:
        raise ValueError(f"Evidence pack source file is empty: {source_file}")


def validate_evidence_pack(manifest_path: Path) -> Dict[str, Any]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Evidence pack manifest not found: {manifest_path}")

    manifest = read_json(manifest_path)
    if "sources" not in manifest or not isinstance(manifest["sources"], list):
        raise ValueError("Evidence pack manifest must have a 'sources' list")

    for src in manifest["sources"]:
        if not isinstance(src, dict):
            raise ValueError("Each source entry must be an object")
        if "path" not in src:
            raise ValueError("Each source entry must contain 'path'")
        source_path = abs_path(src["path"])
        validate_source_file(source_path)

    return manifest


# --------------------------------------------------------------------------------------
# Chunking + retrieval
# --------------------------------------------------------------------------------------

# Accept citations in either form:
#   (c_001) or (c_0001) or bare c_001 / c_0001
# We capture the digits only.
_CITATION_DIGITS_RE = re.compile(r"(?:\(\s*)?\bc_(\d{3,6})\b(?:\s*\))?")

# Used to normalize bare citations to parenthesized form for output consistency.
_BARE_CIT_RE = re.compile(r"(?<!\()\bc_\d{3,6}\b(?!\))")

# Remove internal chunk headers embedded in evidence sources, e.g. "## Chunk c_102"
_INTERNAL_CHUNK_HEADER_RE = re.compile(r"^\s*##\s*Chunk\s+c_\d{1,6}\s*$", re.IGNORECASE)


def _clean_chunk_content(s: str) -> str:
    lines = s.splitlines()
    lines = [ln for ln in lines if not _INTERNAL_CHUNK_HEADER_RE.match(ln.strip())]
    return "\n".join(lines).strip()


def _chunk_markdown_to_chunks(md_text: str, chunk_prefix: str = "c_") -> List[Dict[str, Any]]:
    """
    Very simple chunker that splits on blank lines.
    Each chunk is a dict with chunk_id and content.
    """
    blocks = [b.strip() for b in md_text.split("\n\n") if b.strip()]
    chunks: List[Dict[str, Any]] = []
    for i, block in enumerate(blocks, start=1):
        chunk_id = f"{chunk_prefix}{i:03d}"
        chunks.append({"chunk_id": chunk_id, "content": _clean_chunk_content(block)})
    return chunks


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _topic_max_chunks(topic: Dict[str, Any], default: int = 4) -> int:
    """
    Derive max_chunks from either:
      - topic["max_chunks"] (legacy / direct)
      - topic["constraints"]["max_sources"] (RunConfig schema)
    """
    if isinstance(topic.get("max_chunks"), (int, str)):
        v = _safe_int(topic.get("max_chunks"), default)
        return max(1, v)

    constraints = topic.get("constraints")
    if isinstance(constraints, dict) and "max_sources" in constraints:
        v = _safe_int(constraints.get("max_sources"), default)
        return max(1, v)

    return max(1, default)


def retrieve_evidence_for_topic(
    topic_id: str,
    query: str,
    manifest: Dict[str, Any],
    max_chunks: int = 4
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Retrieves evidence for a topic using the local evidence pack.

    Returns:
      evidence_json: payload used for grounded prompting
      retrieval_trace_json: includes selected_chunks [{source_id, chunk_id, title}]
    """
    sources = manifest.get("sources", [])
    tagged_sources = []
    invalid_tagged_sources = []

    for src in sources:
        # Backwards compatible: manifest uses topic_tags, older schema used tags
        tags = src.get("topic_tags", src.get("tags", []))
        if topic_id in tags:
            source_path = abs_path(src["path"])
            try:
                validate_source_file(source_path)
                tagged_sources.append(src)
            except Exception as e:
                invalid_tagged_sources.append({"source": src, "error": str(e)})

    selected_chunks: List[Dict[str, Any]] = []
    evidence_items: List[Dict[str, Any]] = []
    scoring_debug: List[Dict[str, Any]] = []

    # Safety clamp
    if max_chunks < 1:
        max_chunks = 1

    for src in tagged_sources:
        source_id = src.get("source_id") or Path(src["path"]).stem
        title = src.get("title") or source_id
        source_path = abs_path(src["path"])
        md_text = read_text(source_path)
        chunks = _chunk_markdown_to_chunks(md_text)

        if not chunks:
            continue

        # Naive scoring: prefer chunks containing any query tokens
        q_tokens = [t for t in re.split(r"\W+", query.lower()) if t]
        scored: List[Tuple[int, Dict[str, Any]]] = []
        for ch in chunks:
            text = (ch.get("content") or "").lower()
            score = sum(1 for t in q_tokens if t in text)
            scored.append((score, ch))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Always include the first chunk (c_001) for a tagged source, then fill remaining slots by score.
        first = chunks[0]
        chosen: List[Dict[str, Any]] = [first]

        first_id = first["chunk_id"]
        for score, ch in scored:
            if ch["chunk_id"] == first_id:
                continue
            if len(chosen) >= max_chunks:
                break
            chosen.append(ch)

        # Debug record to explain why selection happened
        debug_top = []
        for score, ch in scored[: min(len(scored), 8)]:
            debug_top.append({
                "chunk_id": ch["chunk_id"],
                "score": score,
                "is_first": ch["chunk_id"] == first_id,
            })

        scoring_debug.append({
            "source_id": source_id,
            "title": title,
            "path": src["path"],
            "max_chunks": max_chunks,
            "first_chunk_id": first_id,
            "chosen_chunk_ids": [c["chunk_id"] for c in chosen],
            "top_scored": debug_top,
        })

        for ch in chosen:
            selected_chunks.append(
                {"source_id": source_id, "chunk_id": ch["chunk_id"], "title": title}
            )
            evidence_items.append(
                {
                    "source_id": source_id,
                    "title": title,
                    "chunk_id": ch["chunk_id"],
                    "content": ch.get("content") or "",
                }
            )

    evidence_json = {
        "topic_id": topic_id,
        "query": query,
        "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pack_id": manifest.get("pack_id", "evidence_pack_v1"),
        "selected_chunks": selected_chunks,
        "evidence_items": evidence_items,
    }

    retrieval_trace_json = {
        "topic_id": topic_id,
        "query": query,
        "selected_chunks": selected_chunks,
        "invalid_tagged_sources": invalid_tagged_sources,
        "scoring_debug": scoring_debug,
    }

    return evidence_json, retrieval_trace_json


# --------------------------------------------------------------------------------------
# Prompt builders
# --------------------------------------------------------------------------------------

def _extract_optional_prompt_hints(topic: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    audience = topic.get("audience")
    if not isinstance(audience, str) or not audience.strip():
        audience = None

    tone = None
    constraints = topic.get("constraints")
    if isinstance(constraints, dict):
        t = constraints.get("tone")
        if isinstance(t, str) and t.strip():
            tone = t.strip()

    return audience, tone


def build_grounded_topic_prompt(
    topic_id: str,
    query: str,
    evidence_json: Dict[str, Any],
    *,
    audience: Optional[str] = None,
    tone: Optional[str] = None,
) -> str:
    evidence_lines = []
    allowed_ids: List[str] = []

    for item in evidence_json.get("evidence_items", []):
        cid = item["chunk_id"]
        allowed_ids.append(cid)
        evidence_lines.append(f"[{cid}] {item['title']}\n{item['content']}\n")

    allowed_ids = sorted(set(allowed_ids))
    allowed_str = ", ".join(allowed_ids) if allowed_ids else "(none)"
    evidence_blob = "\n".join(evidence_lines).strip()

    style_lines = []
    if audience:
        style_lines.append(f"Audience: {audience}")
    if tone:
        style_lines.append(f"Tone: {tone}")
    style_block = ("\n".join(style_lines).strip() + "\n\n") if style_lines else ""

    return (
        f"You are writing research notes for topic_id={topic_id}.\n"
        f"User query: {query}\n\n"
        f"{style_block}"
        "You MUST ground claims in the provided evidence.\n\n"
        f"ALLOWED CITATIONS (use ONLY these exact ids): {allowed_str}\n\n"
        "Hard rules you must follow exactly:\n"
        "- Every bullet under '## Key concepts (grounded)' must include at least one citation.\n"
        "- Citations must ONLY come from the allowed list above.\n"
        "- Citations MUST be in parentheses, exactly like (c_002).\n"
        "- If a bullet cannot be cited, REMOVE the bullet entirely.\n"
        "- Do not invent facts or citations.\n"
        "- Final self-check before output: every Key concepts bullet has a valid citation.\n\n"
        "Write markdown with these sections:\n"
        "# Research notes\n"
        "## Key concepts (grounded)\n"
        "- Each bullet must end with a valid citation like (c_002)\n"
        "## Details\n"
        "- Additional notes, with citations as needed\n\n"
        "EVIDENCE:\n"
        f"{evidence_blob}\n"
    )


def build_grounded_synthesis_prompt(completed_topics: List[Dict[str, Any]]) -> str:
    parts = []
    for t in completed_topics:
        # Preserve topic boundaries explicitly to reduce blending.
        parts.append(f"## Topic: {t['topic_id']}\n{t['research_notes_markdown']}\n")
    joined = "\n".join(parts).strip()

    return (
        "You are synthesizing multiple grounded topic notes into a single summary.\n"
        "Do not invent facts. Do not add new claims that are not present in the topic notes.\n"
        "Keep topic attribution clear where relevant.\n\n"
        f"{joined}\n"
    )


def build_short_form_prompt(summary_markdown: str) -> str:
    return (
        "Rewrite the following summary into a short form, practical checklist.\n"
        "Do not add new facts.\n\n"
        f"{summary_markdown}\n"
    )


def build_grounding_retry_prompt(
    topic_id: str,
    query: str,
    evidence_json: Dict[str, Any],
    validation_errors: List[str],
    previous_markdown: str,
    *,
    audience: Optional[str] = None,
    tone: Optional[str] = None,
) -> str:
    evidence_lines = []
    allowed_ids: List[str] = []

    for item in evidence_json.get("evidence_items", []):
        cid = item["chunk_id"]
        allowed_ids.append(cid)
        evidence_lines.append(f"[{cid}] {item['title']}\n{item['content']}\n")

    allowed_ids = sorted(set(allowed_ids))
    allowed_str = ", ".join(allowed_ids) if allowed_ids else "(none)"
    evidence_blob = "\n".join(evidence_lines).strip()
    error_block = "\n".join(f"- {e}" for e in validation_errors)

    style_lines = []
    if audience:
        style_lines.append(f"Audience: {audience}")
    if tone:
        style_lines.append(f"Tone: {tone}")
    style_block = ("\n".join(style_lines).strip() + "\n\n") if style_lines else ""

    return (
        f"You previously generated research notes for topic_id={topic_id}, but they FAILED grounding validation.\n\n"
        "Grounding errors:\n"
        f"{error_block}\n\n"
        f"{style_block}"
        f"ALLOWED CITATIONS (use ONLY these exact ids): {allowed_str}\n\n"
        "Hard rules you must follow exactly:\n"
        "- Every bullet under '## Key concepts (grounded)' must include at least one citation.\n"
        "- Citations must ONLY reference the allowed citation ids.\n"
        "- Citations MUST be in parentheses, exactly like (c_002).\n"
        "- If a bullet cannot be cited, REMOVE the bullet entirely.\n"
        "- Do not invent facts or citations.\n"
        "- Final self-check before output: all Key concepts bullets are valid.\n\n"
        "Here is your previous output for reference only:\n"
        "-----\n"
        f"{previous_markdown}\n"
        "-----\n\n"
        "Now regenerate the FULL research notes from scratch, fully corrected.\n\n"
        "EVIDENCE:\n"
        f"{evidence_blob}\n"
    )


# --------------------------------------------------------------------------------------
# LLM call hook
# --------------------------------------------------------------------------------------

def call_llm(prompt: str, *, model: Optional[str] = None) -> str:
    from openai import OpenAI

    client = OpenAI()
    m = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    response = client.chat.completions.create(
        model=m,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Empty response content from model.")
    return content


# --------------------------------------------------------------------------------------
# Post generation grounding validator + metrics
# --------------------------------------------------------------------------------------

_KEY_CONCEPTS_HEADER_RE = re.compile(r"^##\s+Key concepts\s+\(grounded\)\s*$", re.IGNORECASE)
_ANY_HEADER_RE = re.compile(r"^##\s+.+$", re.IGNORECASE)
_BULLET_RE = re.compile(r"^(\s*[-*]\s+)(.+)$")


def _normalize_citations_to_parentheses(markdown: str) -> str:
    return _BARE_CIT_RE.sub(lambda m: f"({m.group(0)})", markdown)


def extract_citations(markdown: str) -> List[str]:
    digits = _CITATION_DIGITS_RE.findall(markdown)
    out: List[str] = []
    for d in digits:
        out.append(f"c_{int(d):03d}")
    return out


def _extract_key_concepts_section_lines(markdown: str) -> List[str]:
    lines = markdown.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if _KEY_CONCEPTS_HEADER_RE.match(line.strip()):
            start_idx = i + 1
            break
    if start_idx is None:
        return []

    section_lines: List[str] = []
    for j in range(start_idx, len(lines)):
        line = lines[j]
        if _ANY_HEADER_RE.match(line.strip()):
            break
        section_lines.append(line)
    return section_lines


def _selected_chunk_ids_from_trace(retrieval_trace: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    for ch in retrieval_trace.get("selected_chunks", []):
        cid = ch.get("chunk_id")
        if isinstance(cid, str):
            ids.append(cid)
    return sorted(set(ids))


def _extract_normalized_citations_from_line(line: str) -> List[str]:
    digits = _CITATION_DIGITS_RE.findall(line)
    return [f"c_{int(d):03d}" for d in digits]


def _strip_invalid_citations(markdown: str, invalid_ids: List[str]) -> str:
    if not invalid_ids:
        return markdown
    out = markdown
    for bad in invalid_ids:
        out = out.replace(f"({bad})", "")
    return out


def validate_and_optionally_rewrite_grounding(
    research_notes_md: str,
    selected_chunk_ids: List[str],
    *,
    grounding_mode: str
) -> Tuple[bool, str, List[str], Dict[str, Any]]:
    errors: List[str] = []
    selected_set = set(selected_chunk_ids)

    research_notes_md = _normalize_citations_to_parentheses(research_notes_md)

    citations = extract_citations(research_notes_md)
    invalid_citations = sorted({c for c in citations if c not in selected_set})
    total_citations = len(citations)
    invalid_citations_count = len(invalid_citations)

    if invalid_citations:
        errors.append(
            "Invalid citations found (not in selected chunk_ids): "
            + ", ".join(f"({c})" for c in invalid_citations)
        )
        research_notes_md = _strip_invalid_citations(research_notes_md, invalid_citations)

    section_lines = _extract_key_concepts_section_lines(research_notes_md)
    missing_key_concepts_section = len(section_lines) == 0
    if missing_key_concepts_section:
        errors.append("Missing section: ## Key concepts (grounded)")

    lines = research_notes_md.splitlines()

    key_concepts_bullets_total = 0
    key_concepts_bullets_missing_valid_citation = 0

    if not missing_key_concepts_section:
        start_idx = None
        for i, line in enumerate(lines):
            if _KEY_CONCEPTS_HEADER_RE.match(line.strip()):
                start_idx = i + 1
                break

        end_idx = len(lines)
        if start_idx is not None:
            for j in range(start_idx, len(lines)):
                if _ANY_HEADER_RE.match(lines[j].strip()):
                    end_idx = j
                    break

            rewritten_lines = list(lines)
            for idx in range(start_idx, end_idx):
                m = _BULLET_RE.match(lines[idx])
                if not m:
                    continue

                key_concepts_bullets_total += 1

                bullet_prefix = m.group(1)
                bullet_text = m.group(2)

                bullet_citations = _extract_normalized_citations_from_line(lines[idx])
                valid_in_bullet = [c for c in bullet_citations if c in selected_set]

                if len(valid_in_bullet) == 0:
                    key_concepts_bullets_missing_valid_citation += 1
                    msg = f"Uncited or invalidly cited bullet in Key concepts at line {idx + 1}"
                    errors.append(msg)
                    if grounding_mode == "soft":
                        if not bullet_text.lstrip().startswith("Unverified:"):
                            rewritten_lines[idx] = f"{bullet_prefix}Unverified: {bullet_text}"

            research_notes_md = "\n".join(rewritten_lines)

    ok = len(errors) == 0
    if grounding_mode == "soft":
        ok = True

    metrics: Dict[str, Any] = {
        "total_citations": total_citations,
        "invalid_citations_count": invalid_citations_count,
        "key_concepts_bullets_total": key_concepts_bullets_total,
        "key_concepts_bullets_missing_valid_citation": key_concepts_bullets_missing_valid_citation,
        "missing_key_concepts_section": missing_key_concepts_section,
    }

    return ok, research_notes_md, errors, metrics


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------

def run_pipeline(
    run_id: Optional[str] = None,
    *,
    config: Optional[PipelineConfig] = None,
    topics: Optional[List[Dict[str, Any]]] = None,
    model: Optional[str] = None,
    outputs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    cfg = config or PipelineConfig()
    if cfg.grounding_mode not in ("strict", "soft"):
        raise ValueError("grounding_mode must be 'strict' or 'soft'")

    manifest_path = abs_path(cfg.evidence_pack_manifest_relpath)
    manifest = validate_evidence_pack(manifest_path)

    if run_id is None:
        run_id = time.strftime("demo_v3_local_pack_%03d", time.localtime().tm_yday)

    out_root = outputs_dir if outputs_dir is not None else abs_path(cfg.outputs_dir_relpath)
    run_root = out_root / run_id
    topics_dir = run_root / "topics"
    summary_dir = run_root / "summary"
    short_form_dir = run_root / "short_form"
    tool_calls_dir = run_root / "tool_calls"

    if run_root.exists():
        suffix = 1
        while (out_root / f"{run_id}_{suffix:02d}").exists():
            suffix += 1
        run_id = f"{run_id}_{suffix:02d}"
        run_root = out_root / run_id
        topics_dir = run_root / "topics"
        summary_dir = run_root / "summary"
        short_form_dir = run_root / "short_form"
        tool_calls_dir = run_root / "tool_calls"

    ensure_dir(topics_dir)
    ensure_dir(summary_dir)
    ensure_dir(short_form_dir)
    ensure_dir(tool_calls_dir)

    if topics is None:
        tag_set = set()
        for src in manifest.get("sources", []):
            tags = src.get("topic_tags", src.get("tags", []))
            for tag in tags:
                if isinstance(tag, str) and tag.strip():
                    tag_set.add(tag.strip())
        derived = sorted(tag_set)
        topics = [{"topic_id": t, "query": t} for t in derived]

    resolved_model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    run_manifest: Dict[str, Any] = {
        "run_id": run_id,
        "variant": "retrieval_grounded_orchestration",
        "model": resolved_model,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "topics_total": len(topics),
        "topics_completed": 0,
        "call_count": 0,
        "errors": [],
        "grounding_mode": cfg.grounding_mode,
        "evidence_pack_manifest": str(Path(cfg.evidence_pack_manifest_relpath)),
        "topics_input": [
            {
                "topic_id": t.get("topic_id"),
                "query": t.get("query"),
                "audience": t.get("audience"),
                "constraints": t.get("constraints"),
            }
            for t in topics
            if isinstance(t, dict)
        ],
    }

    completed_topics_payload: List[Dict[str, Any]] = []
    topic_statuses: Dict[str, Any] = {}

    for t in topics:
        topic_id = t["topic_id"]
        query = t["query"]
        topic_out_dir = topics_dir / topic_id
        ensure_dir(topic_out_dir)

        topic_statuses[topic_id] = {"status": "started", "errors": []}

        audience, tone = _extract_optional_prompt_hints(t)

        try:
            max_chunks = _topic_max_chunks(t, default=4)

            evidence_json, retrieval_trace_json = retrieve_evidence_for_topic(
                topic_id=topic_id,
                query=query,
                manifest=manifest,
                max_chunks=max_chunks,
            )

            write_json(topic_out_dir / "evidence.json", evidence_json)
            write_json(topic_out_dir / "retrieval_trace.json", retrieval_trace_json)

            selected_chunk_ids = _selected_chunk_ids_from_trace(retrieval_trace_json)

            if len(selected_chunk_ids) == 0:
                msg = "No chunks selected for topic. Check evidence_pack manifest topic_tags for this topic_id."
                write_text(topic_out_dir / "error.txt", msg + "\n")
                topic_statuses[topic_id]["status"] = "failed"
                topic_statuses[topic_id]["errors"].append(msg)
                topic_statuses[topic_id]["attempts"] = 0
                run_manifest["errors"].append({
                    "topic_id": topic_id,
                    "stage": "retrieval",
                    "errors": [msg],
                })
                write_json(topic_out_dir / "grounding_validation.json", {
                    "ok": False,
                    "mode": cfg.grounding_mode,
                    "attempts": 0,
                    "selected_chunk_ids": selected_chunk_ids,
                    "errors": [msg],
                    "metrics": {
                        "total_citations": 0,
                        "invalid_citations_count": 0,
                        "key_concepts_bullets_total": 0,
                        "key_concepts_bullets_missing_valid_citation": 0,
                        "missing_key_concepts_section": True,
                    },
                })
                continue

            max_attempts = 3  # initial + up to two retries
            research_notes_md: Optional[str] = None
            ok = False
            rewritten_md = ""
            validation_errors: List[str] = []
            validation_metrics: Dict[str, Any] = {}

            prompt_trace: Dict[str, Any] = {
                "topic_id": topic_id,
                "query": query,
                "model": resolved_model,
                "attempts": [],
            }

            attempts_used = 0

            for attempt_index in range(max_attempts):
                if attempt_index == 0:
                    prompt = build_grounded_topic_prompt(
                        topic_id, query, evidence_json, audience=audience, tone=tone
                    )
                else:
                    prompt = build_grounding_retry_prompt(
                        topic_id=topic_id,
                        query=query,
                        evidence_json=evidence_json,
                        validation_errors=validation_errors,
                        previous_markdown=research_notes_md or "",
                        audience=audience,
                        tone=tone,
                    )

                run_manifest["call_count"] += 1
                research_notes_md = call_llm(prompt, model=resolved_model)
                attempts_used = attempt_index + 1

                ok, rewritten_md, validation_errors, validation_metrics = validate_and_optionally_rewrite_grounding(
                    research_notes_md,
                    selected_chunk_ids,
                    grounding_mode=cfg.grounding_mode,
                )

                prompt_trace["attempts"].append({
                    "attempt_index": attempt_index,
                    "ok": ok,
                    "prompt_chars": len(prompt),
                    "response_chars": len(research_notes_md or ""),
                    "validation_errors": list(validation_errors),
                    "selected_chunk_ids": list(selected_chunk_ids),
                })

                if ok:
                    break

            topic_statuses[topic_id]["attempts"] = attempts_used
            topic_statuses[topic_id]["selected_chunk_ids"] = list(selected_chunk_ids)
            topic_statuses[topic_id]["grounding_ok"] = ok
            topic_statuses[topic_id]["grounding_metrics"] = validation_metrics

            write_json(topic_out_dir / "prompt_trace.json", prompt_trace)

            research_notes_path = topic_out_dir / "research_notes.md"
            final_notes_to_write = rewritten_md if cfg.grounding_mode == "soft" else (research_notes_md or "")
            write_text(research_notes_path, final_notes_to_write)

            grounding_report = {
                "ok": ok,
                "mode": cfg.grounding_mode,
                "attempts": attempts_used,
                "selected_chunk_ids": selected_chunk_ids,
                "errors": validation_errors,
                "metrics": validation_metrics,
            }
            write_json(topic_out_dir / "grounding_validation.json", grounding_report)

            if not ok and cfg.grounding_mode == "strict":
                err_txt = (
                    "Grounding validation failed.\n\n"
                    f"topic_id: {topic_id}\n"
                    f"selected_chunk_ids: {selected_chunk_ids}\n"
                    f"attempts: {attempts_used}\n\n"
                    "Errors:\n"
                    + "\n".join(f"- {e}" for e in validation_errors)
                    + "\n"
                )
                write_text(topic_out_dir / "error.txt", err_txt)

                topic_statuses[topic_id]["status"] = "failed"
                topic_statuses[topic_id]["errors"].extend(validation_errors)
                run_manifest["errors"].append({
                    "topic_id": topic_id,
                    "stage": "post_generation_grounding_validation",
                    "errors": validation_errors,
                })
                continue

            topic_statuses[topic_id]["status"] = "completed"
            run_manifest["topics_completed"] += 1

            completed_topics_payload.append({
                "topic_id": topic_id,
                "query": query,
                "research_notes_markdown": final_notes_to_write,
                "selected_chunk_ids": selected_chunk_ids,
            })

        except Exception as e:
            msg = str(e)
            write_text(topic_out_dir / "error.txt", msg + "\n")
            topic_statuses[topic_id]["status"] = "failed"
            topic_statuses[topic_id]["errors"].append(msg)
            run_manifest["errors"].append({
                "topic_id": topic_id,
                "stage": "topic_generation",
                "errors": [msg],
            })

    run_manifest["topic_statuses"] = topic_statuses
    run_manifest["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    write_json(run_root / "run_manifest.json", run_manifest)

    if run_manifest["topics_completed"] == 0 and cfg.halt_if_no_topics_completed:
        return run_manifest

    try:
        synthesis_prompt = build_grounded_synthesis_prompt(completed_topics_payload)
        run_manifest["call_count"] += 1
        summary_md = call_llm(synthesis_prompt, model=resolved_model)
        write_text(summary_dir / "summary.md", summary_md)
    except Exception as e:
        run_manifest["errors"].append({
            "topic_id": None,
            "stage": "synthesis",
            "errors": [str(e)],
        })
        write_json(run_root / "run_manifest.json", run_manifest)
        return run_manifest

    try:
        short_prompt = build_short_form_prompt(summary_md)
        run_manifest["call_count"] += 1
        short_md = call_llm(short_prompt, model=resolved_model)
        write_text(short_form_dir / "short_form.md", short_md)
    except Exception as e:
        run_manifest["errors"].append({
            "topic_id": None,
            "stage": "short_form",
            "errors": [str(e)],
        })
        write_json(run_root / "run_manifest.json", run_manifest)
        return run_manifest

    write_json(run_root / "run_manifest.json", run_manifest)
    return run_manifest
