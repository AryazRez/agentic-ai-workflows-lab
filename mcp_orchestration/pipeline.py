import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI

from shared.schemas import TopicSpec, OutputConfig, RunConfig
from shared.prompts import (
    build_research_prompt,
    build_synthesis_prompt,
    build_short_form_prompt,
)
from shared.utils import ensure_dir, write_text, write_json, timestamp


client = OpenAI()


# -----------------------------
# Context and error structures
# -----------------------------

@dataclass
class ErrorRecord:
    step: str
    topic_id: Optional[str]
    error_type: str
    error_message: str


@dataclass
class RunPaths:
    base_out: str
    topics_dir: str
    summary_dir: str
    short_form_dir: str
    tool_calls_dir: str


@dataclass
class RunState:
    topic_ids_remaining: List[str] = field(default_factory=list)
    topic_ids_completed: List[str] = field(default_factory=list)
    topic_notes_by_id: Dict[str, str] = field(default_factory=dict)
    summary_text: Optional[str] = None
    short_form_by_platform: Dict[str, str] = field(default_factory=dict)


@dataclass
class RunTelemetry:
    call_count: int = 0
    errors: List[ErrorRecord] = field(default_factory=list)
    tool_call_seq: int = 0
    started_at: str = field(default_factory=timestamp)
    ended_at: Optional[str] = None


@dataclass
class RunContext:
    run_id: str
    config: RunConfig
    paths: RunPaths
    state: RunState = field(default_factory=RunState)
    telemetry: RunTelemetry = field(default_factory=RunTelemetry)


# -----------------------------
# Tool registry (local MCP-style)
# -----------------------------

@dataclass
class ToolSpec:
    name: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    handler: Callable[[RunContext, Dict[str, Any]], Dict[str, Any]]


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec):
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def list_tools(self) -> List[str]:
        return sorted(self._tools.keys())

    def invoke(self, ctx: RunContext, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        spec = self._tools[tool_name]

        # Record tool call for observability
        ctx.telemetry.tool_call_seq += 1
        seq = ctx.telemetry.tool_call_seq
        ensure_dir(ctx.paths.tool_calls_dir)
        write_json(
            os.path.join(ctx.paths.tool_calls_dir, f"{seq:03d}_{tool_name}_input.json"),
            {"tool": tool_name, "input": tool_input, "timestamp": timestamp()},
        )

        result = spec.handler(ctx, tool_input)

        write_json(
            os.path.join(ctx.paths.tool_calls_dir, f"{seq:03d}_{tool_name}_output.json"),
            {"tool": tool_name, "output": result, "timestamp": timestamp()},
        )

        return result


# -----------------------------
# Config loading
# -----------------------------

def load_config(path: str) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    topics = [TopicSpec(**t) for t in raw["topics"]]
    output = OutputConfig(**raw["output"])
    return RunConfig(run_id=raw["run_id"], topics=topics, output=output)


# -----------------------------
# Tools
# -----------------------------

def tool_generate_topic_notes(ctx: RunContext, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    topic: TopicSpec = tool_input["topic"]

    topic_dir = os.path.join(ctx.paths.topics_dir, topic.id)
    ensure_dir(topic_dir)

    try:
        prompt = build_research_prompt(topic)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        ctx.telemetry.call_count += 1

        text = response.choices[0].message.content or ""

        write_text(os.path.join(topic_dir, "research_notes.md"), text)
        write_json(os.path.join(topic_dir, "prompt_trace.json"), {"prompt": prompt, "response": text})

        return {
            "topic_id": topic.id,
            "notes": text,
            "artifacts_written": [
                os.path.join("topics", topic.id, "research_notes.md"),
                os.path.join("topics", topic.id, "prompt_trace.json"),
            ],
            "error": None,
        }

    except Exception as e:
        err = ErrorRecord(
            step="generate_topic_notes",
            topic_id=topic.id,
            error_type=e.__class__.__name__,
            error_message=str(e),
        )
        ctx.telemetry.errors.append(err)

        write_json(
            os.path.join(topic_dir, "error.json"),
            {
                "step": err.step,
                "topic_id": err.topic_id,
                "error_type": err.error_type,
                "error_message": err.error_message,
            },
        )

        return {
            "topic_id": topic.id,
            "notes": "",
            "artifacts_written": [os.path.join("topics", topic.id, "error.json")],
            "error": {
                "step": err.step,
                "topic_id": err.topic_id,
                "error_type": err.error_type,
                "error_message": err.error_message,
            },
        }


def tool_synthesize_notes(ctx: RunContext, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    notes_by_id: Dict[str, str] = tool_input["topic_notes_by_id"]

    ensure_dir(ctx.paths.summary_dir)

    labeled_notes = []
    for tid, notes in notes_by_id.items():
        labeled_notes.append(f"Topic: {tid}\n{notes}".strip())

    all_notes = "\n\n".join(labeled_notes).strip()
    if not all_notes:
        summary_text = "No topic notes were generated. See per-topic error files for details."
        write_text(os.path.join(ctx.paths.summary_dir, "brief.md"), summary_text)
        write_json(os.path.join(ctx.paths.summary_dir, "prompt_trace.json"), {"prompt": "", "response": summary_text})
        return {
            "summary_text": summary_text,
            "artifacts_written": [
                os.path.join("summary", "brief.md"),
                os.path.join("summary", "prompt_trace.json"),
            ],
            "error": None,
        }

    try:
        prompt = build_synthesis_prompt(all_notes)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        ctx.telemetry.call_count += 1

        summary_text = response.choices[0].message.content or ""
        write_text(os.path.join(ctx.paths.summary_dir, "brief.md"), summary_text)
        write_json(os.path.join(ctx.paths.summary_dir, "prompt_trace.json"), {"prompt": prompt, "response": summary_text})

        return {
            "summary_text": summary_text,
            "artifacts_written": [
                os.path.join("summary", "brief.md"),
                os.path.join("summary", "prompt_trace.json"),
            ],
            "error": None,
        }

    except Exception as e:
        err = ErrorRecord(
            step="synthesize_notes",
            topic_id=None,
            error_type=e.__class__.__name__,
            error_message=str(e),
        )
        ctx.telemetry.errors.append(err)

        write_json(
            os.path.join(ctx.paths.summary_dir, "error.json"),
            {
                "step": err.step,
                "topic_id": err.topic_id,
                "error_type": err.error_type,
                "error_message": err.error_message,
            },
        )
        write_text(os.path.join(ctx.paths.summary_dir, "brief.md"), "Synthesis failed. See error.json for details.")

        return {
            "summary_text": "",
            "artifacts_written": [
                os.path.join("summary", "brief.md"),
                os.path.join("summary", "error.json"),
            ],
            "error": {
                "step": err.step,
                "topic_id": err.topic_id,
                "error_type": err.error_type,
                "error_message": err.error_message,
            },
        }


def tool_generate_short_form(ctx: RunContext, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    platform: str = tool_input["platform"]
    summary_text: str = tool_input["summary_text"]

    ensure_dir(ctx.paths.short_form_dir)

    try:
        prompt = build_short_form_prompt(platform, summary_text)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        ctx.telemetry.call_count += 1

        draft_text = response.choices[0].message.content or ""

        write_text(os.path.join(ctx.paths.short_form_dir, f"{platform}.md"), draft_text)
        write_json(
            os.path.join(ctx.paths.short_form_dir, f"{platform}_prompt_trace.json"),
            {"prompt": prompt, "response": draft_text},
        )

        return {
            "platform": platform,
            "draft_text": draft_text,
            "artifacts_written": [
                os.path.join("short_form", f"{platform}.md"),
                os.path.join("short_form", f"{platform}_prompt_trace.json"),
            ],
            "error": None,
        }

    except Exception as e:
        err = ErrorRecord(
            step="generate_short_form",
            topic_id=None,
            error_type=e.__class__.__name__,
            error_message=str(e),
        )
        ctx.telemetry.errors.append(err)

        write_json(
            os.path.join(ctx.paths.short_form_dir, "error.json"),
            {
                "step": err.step,
                "topic_id": err.topic_id,
                "error_type": err.error_type,
                "error_message": err.error_message,
            },
        )

        return {
            "platform": platform,
            "draft_text": "",
            "artifacts_written": [os.path.join("short_form", "error.json")],
            "error": {
                "step": err.step,
                "topic_id": err.topic_id,
                "error_type": err.error_type,
                "error_message": err.error_message,
            },
        }


# -----------------------------
# Tool specs (schemas are simple dicts for now)
# -----------------------------

def build_registry() -> ToolRegistry:
    reg = ToolRegistry()

    reg.register(
        ToolSpec(
            name="generate_topic_notes",
            input_schema={"type": "object", "properties": {"topic": {"type": "object"}}, "required": ["topic"]},
            output_schema={
                "type": "object",
                "properties": {
                    "topic_id": {"type": "string"},
                    "notes": {"type": "string"},
                    "artifacts_written": {"type": "array"},
                    "error": {"type": ["object", "null"]},
                },
                "required": ["topic_id", "notes", "artifacts_written", "error"],
            },
            handler=tool_generate_topic_notes,
        )
    )

    reg.register(
        ToolSpec(
            name="synthesize_notes",
            input_schema={
                "type": "object",
                "properties": {"topic_notes_by_id": {"type": "object"}},
                "required": ["topic_notes_by_id"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "summary_text": {"type": "string"},
                    "artifacts_written": {"type": "array"},
                    "error": {"type": ["object", "null"]},
                },
                "required": ["summary_text", "artifacts_written", "error"],
            },
            handler=tool_synthesize_notes,
        )
    )

    reg.register(
        ToolSpec(
            name="generate_short_form",
            input_schema={
                "type": "object",
                "properties": {"platform": {"type": "string"}, "summary_text": {"type": "string"}},
                "required": ["platform", "summary_text"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "platform": {"type": "string"},
                    "draft_text": {"type": "string"},
                    "artifacts_written": {"type": "array"},
                    "error": {"type": ["object", "null"]},
                },
                "required": ["platform", "draft_text", "artifacts_written", "error"],
            },
            handler=tool_generate_short_form,
        )
    )

    return reg


# -----------------------------
# Orchestrator (deterministic)
# -----------------------------

def run_pipeline(config_path: str):
    config = load_config(config_path)

    base_out = os.path.join("outputs", config.run_id)
    paths = RunPaths(
        base_out=base_out,
        topics_dir=os.path.join(base_out, "topics"),
        summary_dir=os.path.join(base_out, "summary"),
        short_form_dir=os.path.join(base_out, "short_form"),
        tool_calls_dir=os.path.join(base_out, "tool_calls"),
    )

    ensure_dir(paths.base_out)
    ensure_dir(paths.topics_dir)
    ensure_dir(paths.summary_dir)

    ctx = RunContext(run_id=config.run_id, config=config, paths=paths)

    # Persist run config for reproducibility
    write_json(os.path.join(paths.base_out, "run_config.json"), config)

    registry = build_registry()

    # Initialize topic state
    ctx.state.topic_ids_remaining = [t.id for t in config.topics]

    # 1) Topic research notes
    for topic in config.topics:
        result = registry.invoke(ctx, "generate_topic_notes", {"topic": topic})

        if not result.get("error"):
            ctx.state.topic_notes_by_id[topic.id] = result.get("notes", "")
            ctx.state.topic_ids_completed.append(topic.id)

    # 2) Synthesis
    synth_result = registry.invoke(ctx, "synthesize_notes", {"topic_notes_by_id": ctx.state.topic_notes_by_id})
    if not synth_result.get("error"):
        ctx.state.summary_text = synth_result.get("summary_text")

    # 3) Short-form (optional)
    if config.output.include_short_form and ctx.state.summary_text:
        ensure_dir(paths.short_form_dir)
        for platform in config.output.short_form_platforms:
            sf_result = registry.invoke(
                ctx,
                "generate_short_form",
                {"platform": platform, "summary_text": ctx.state.summary_text},
            )
            if not sf_result.get("error"):
                ctx.state.short_form_by_platform[platform] = sf_result.get("draft_text", "")

    # Final manifest
    ctx.telemetry.ended_at = timestamp()

    write_json(
        os.path.join(paths.base_out, "run_manifest.json"),
        {
            "run_id": ctx.run_id,
            "timestamp": ctx.telemetry.ended_at,
            "model": "gpt-4o-mini",
            "variant": "mcp_orchestration_local",
            "topics_requested": len(config.topics),
            "topics_completed": len(ctx.state.topic_ids_completed),
            "call_count": ctx.telemetry.call_count,
            "short_form_enabled": config.output.include_short_form,
            "short_form_platforms": config.output.short_form_platforms if config.output.include_short_form else [],
            "errors": [
                {
                    "step": e.step,
                    "topic_id": e.topic_id,
                    "error_type": e.error_type,
                    "error_message": e.error_message,
                }
                for e in ctx.telemetry.errors
            ],
            "tools_available": registry.list_tools(),
        },
    )


if __name__ == "__main__":
    run_pipeline("inputs/run_config.json")
