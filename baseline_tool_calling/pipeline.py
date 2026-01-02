import json
import os
from openai import OpenAI

from shared.schemas import TopicSpec, OutputConfig, RunConfig
from shared.prompts import (
    build_research_prompt,
    build_synthesis_prompt,
    build_short_form_prompt
)
from shared.utils import ensure_dir, write_text, write_json, timestamp


client = OpenAI()


def load_config(path: str) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    topics = [TopicSpec(**t) for t in raw["topics"]]
    output = OutputConfig(**raw["output"])
    return RunConfig(run_id=raw["run_id"], topics=topics, output=output)


def run_pipeline(config_path: str):
    config = load_config(config_path)
    base_out = os.path.join("outputs", config.run_id)

    ensure_dir(base_out)

    # Track run metadata for auditability
    call_count = 0
    errors = []

    # Persist the run config for reproducibility
    write_json(os.path.join(base_out, "run_config.json"), config)

    topic_notes = []

    for topic in config.topics:
        topic_dir = os.path.join(base_out, "topics", topic.id)
        ensure_dir(topic_dir)

        try:
            prompt = build_research_prompt(topic)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            call_count += 1

            text = response.choices[0].message.content or ""

            write_text(os.path.join(topic_dir, "research_notes.md"), text)
            write_json(
                os.path.join(topic_dir, "prompt_trace.json"),
                {"prompt": prompt, "response": text}
            )

            topic_notes.append(text)

        except Exception as e:
            # Do not crash the entire run because one topic failed
            err = {
                "topic_id": topic.id,
                "step": "topic_research",
                "error_type": e.__class__.__name__,
                "error_message": str(e)
            }
            errors.append(err)
            write_json(os.path.join(topic_dir, "error.json"), err)

    # Synthesis step
    summary_dir = os.path.join(base_out, "summary")
    ensure_dir(summary_dir)

    all_notes = "\n\n".join(topic_notes).strip()
    if not all_notes:
        summary_text = "No topic notes were generated. See per-topic error files for details."
        write_text(os.path.join(summary_dir, "brief.md"), summary_text)
        write_json(
            os.path.join(summary_dir, "prompt_trace.json"),
            {"prompt": "", "response": summary_text}
        )
    else:
        try:
            summary_prompt = build_synthesis_prompt(all_notes)
            summary_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.2
            )
            call_count += 1

            summary_text = summary_response.choices[0].message.content or ""

            write_text(os.path.join(summary_dir, "brief.md"), summary_text)
            write_json(
                os.path.join(summary_dir, "prompt_trace.json"),
                {"prompt": summary_prompt, "response": summary_text}
            )

        except Exception as e:
            err = {
                "topic_id": None,
                "step": "synthesis",
                "error_type": e.__class__.__name__,
                "error_message": str(e)
            }
            errors.append(err)
            write_text(os.path.join(summary_dir, "brief.md"), "Synthesis failed. See run_manifest.json for details.")
            write_json(os.path.join(summary_dir, "error.json"), err)

    # Optional short-form drafts
    if config.output.include_short_form:
        short_dir = os.path.join(base_out, "short_form")
        ensure_dir(short_dir)

        try:
            # Load the summary we just wrote as the basis for short form
            summary_path = os.path.join(summary_dir, "brief.md")
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_for_short = f.read()

            for platform in config.output.short_form_platforms:
                sf_prompt = build_short_form_prompt(platform, summary_for_short)

                sf_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": sf_prompt}],
                    temperature=0.3
                )
                call_count += 1

                sf_text = sf_response.choices[0].message.content or ""

                write_text(os.path.join(short_dir, f"{platform}.md"), sf_text)
                write_json(
                    os.path.join(short_dir, f"{platform}_prompt_trace.json"),
                    {"prompt": sf_prompt, "response": sf_text}
                )

        except Exception as e:
            err = {
                "topic_id": None,
                "step": "short_form",
                "error_type": e.__class__.__name__,
                "error_message": str(e)
            }
            errors.append(err)
            write_json(os.path.join(short_dir, "error.json"), err)

    # Final manifest
    write_json(
        os.path.join(base_out, "run_manifest.json"),
        {
            "run_id": config.run_id,
            "timestamp": timestamp(),
            "model": "gpt-4o-mini",
            "topics_requested": len(config.topics),
            "topics_completed": len(topic_notes),
            "call_count": call_count,
            "short_form_enabled": config.output.include_short_form,
            "short_form_platforms": config.output.short_form_platforms if config.output.include_short_form else [],
            "errors": errors
        }
    )


if __name__ == "__main__":
    run_pipeline("inputs/run_config.json")
