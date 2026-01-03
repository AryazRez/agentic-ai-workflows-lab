import pytest
from pathlib import Path

from retrieval_grounded_orchestration import pipeline as pl


def _fake_retrieve_evidence_for_topic(topic_id: str, query: str, manifest: dict, max_chunks: int = 4):
    # Selected chunks are what grounding validation must enforce.
    retrieval_trace = {
        "topic_id": topic_id,
        "query": query,
        "selected_chunks": [
            {"source_id": "src_test", "chunk_id": "c_001", "title": "Test Source"},
            {"source_id": "src_test", "chunk_id": "c_002", "title": "Test Source"},
        ],
        "invalid_tagged_sources": [],
    }

    evidence = {
        "topic_id": topic_id,
        "query": query,
        "retrieved_at": "2026-01-02T00:00:00Z",
        "pack_id": "evidence_pack_v1",
        "selected_chunks": retrieval_trace["selected_chunks"],
        "evidence_items": [
            {"source_id": "src_test", "title": "Test Source", "chunk_id": "c_001", "content": "Alpha content"},
            {"source_id": "src_test", "title": "Test Source", "chunk_id": "c_002", "content": "Beta content"},
        ],
    }
    return evidence, retrieval_trace


def _fake_validate_evidence_pack(manifest_path: Path):
    # Avoid dependency on the real repo evidence pack in smoke tests.
    return {
        "pack_id": "evidence_pack_v1",
        "sources": [{"path": "docs/evidence_pack/sources/dummy.md", "tags": ["t1"]}],
    }


def test_smoke_strict_mode_passes_and_writes_artifacts(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.setattr(pl, "validate_evidence_pack", _fake_validate_evidence_pack)
    monkeypatch.setattr(pl, "retrieve_evidence_for_topic", _fake_retrieve_evidence_for_topic)

    def fake_call_llm(prompt: str, *, model=None) -> str:
        # 3 LLM calls expected when one topic completes and no retry is needed:
        # 1) topic research notes
        # 2) synthesis
        # 3) short form
        if "Write markdown with these sections" in prompt:
            return (
                "# Research notes\n"
                "## Key concepts (grounded)\n"
                "- Concept A is supported. (c_001)\n"
                "- Concept B is supported. (c_002)\n"
                "## Details\n"
                "- More detail. (c_001)\n"
            )
        if "You are synthesizing multiple grounded topic notes" in prompt:
            return "# Summary\n- Synthesis of grounded notes.\n"
        return "# Short form\n- Checklist item.\n"

    monkeypatch.setattr(pl, "call_llm", fake_call_llm)

    cfg = pl.PipelineConfig(grounding_mode="strict")
    topics = [{"topic_id": "t1", "query": "test query"}]

    # Act
    manifest = pl.run_pipeline(run_id="smoke_strict_pass", config=cfg, topics=topics, outputs_dir=tmp_path)

    # Assert run_manifest invariants
    assert manifest["topics_total"] == 1
    assert manifest["topics_completed"] == 1
    assert manifest["errors"] == []
    assert manifest["call_count"] == 3

    run_root = tmp_path / "smoke_strict_pass"
    assert (run_root / "run_manifest.json").exists()
    assert (run_root / "topics" / "t1" / "research_notes.md").exists()
    assert (run_root / "topics" / "t1" / "evidence.json").exists()
    assert (run_root / "topics" / "t1" / "retrieval_trace.json").exists()
    assert (run_root / "summary" / "summary.md").exists()
    assert (run_root / "short_form" / "short_form.md").exists()


def test_smoke_strict_mode_fails_topic_and_halts_if_none_completed(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "validate_evidence_pack", _fake_validate_evidence_pack)
    monkeypatch.setattr(pl, "retrieve_evidence_for_topic", _fake_retrieve_evidence_for_topic)

    def fake_call_llm(prompt: str, *, model=None) -> str:
        # All attempts are invalid (no citation and/or invalid citation).
        if "Write markdown with these sections" in prompt:
            return (
                "# Research notes\n"
                "## Key concepts (grounded)\n"
                "- This bullet has no citation.\n"
                "- This one cites an unselected chunk. (c_999)\n"
                "## Details\n"
                "- More detail. (c_001)\n"
            )
        if "FAILED grounding validation" in prompt:
            return (
                "# Research notes\n"
                "## Key concepts (grounded)\n"
                "- Still missing citation.\n"
                "## Details\n"
                "- Still fine detail. (c_001)\n"
            )
        return "UNEXPECTED PROMPT"

    monkeypatch.setattr(pl, "call_llm", fake_call_llm)

    cfg = pl.PipelineConfig(grounding_mode="strict", halt_if_no_topics_completed=True)
    topics = [{"topic_id": "t1", "query": "test query"}]

    manifest = pl.run_pipeline(run_id="smoke_strict_fail", config=cfg, topics=topics, outputs_dir=tmp_path)

    assert manifest["topics_total"] == 1
    assert manifest["topics_completed"] == 0
    # Pipeline uses max_attempts=3 (initial + up to two retries) before failing the topic.
    assert manifest["call_count"] == 3
    assert len(manifest["errors"]) == 1
    assert manifest["errors"][0]["stage"] == "post_generation_grounding_validation"

    run_root = tmp_path / "smoke_strict_fail"
    assert (run_root / "topics" / "t1" / "error.txt").exists()
    assert not (run_root / "summary" / "summary.md").exists()
    assert not (run_root / "short_form" / "short_form.md").exists()


def test_smoke_strict_mode_retry_succeeds_and_attempts_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "validate_evidence_pack", _fake_validate_evidence_pack)
    monkeypatch.setattr(pl, "retrieve_evidence_for_topic", _fake_retrieve_evidence_for_topic)

    call_counter = {"n": 0}

    def fake_call_llm(prompt: str, *, model=None) -> str:
        call_counter["n"] += 1

        # First attempt: invalid
        if "Write markdown with these sections" in prompt:
            return (
                "# Research notes\n"
                "## Key concepts (grounded)\n"
                "- Missing citation bullet.\n"
                "## Details\n"
                "- Detail is fine. (c_001)\n"
            )

        # Retry attempt: corrected
        if "FAILED grounding validation" in prompt:
            return (
                "# Research notes\n"
                "## Key concepts (grounded)\n"
                "- Now fixed with citation. (c_001)\n"
                "## Details\n"
                "- Detail still fine. (c_001)\n"
            )

        # Synthesis and short form after topic completes
        if "You are synthesizing multiple grounded topic notes" in prompt:
            return "# Summary\n- Retry success synthesis.\n"
        return "# Short form\n- Retry success checklist.\n"

    monkeypatch.setattr(pl, "call_llm", fake_call_llm)

    cfg = pl.PipelineConfig(grounding_mode="strict", halt_if_no_topics_completed=True)
    topics = [{"topic_id": "t1", "query": "test query"}]

    manifest = pl.run_pipeline(run_id="smoke_strict_retry_success", config=cfg, topics=topics, outputs_dir=tmp_path)

    assert manifest["topics_completed"] == 1
    # Calls: attempt1 + attempt2 + synthesis + short form = 4
    assert manifest["call_count"] == 4

    # attempts recorded for the topic
    assert manifest["topic_statuses"]["t1"]["attempts"] == 2

    run_root = tmp_path / "smoke_strict_retry_success"
    assert (run_root / "topics" / "t1" / "research_notes.md").exists()
    assert (run_root / "summary" / "summary.md").exists()
    assert (run_root / "short_form" / "short_form.md").exists()


def test_smoke_soft_mode_prefixes_unverified_and_continues(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "validate_evidence_pack", _fake_validate_evidence_pack)
    monkeypatch.setattr(pl, "retrieve_evidence_for_topic", _fake_retrieve_evidence_for_topic)

    def fake_call_llm(prompt: str, *, model=None) -> str:
        if "Write markdown with these sections" in prompt:
            return (
                "# Research notes\n"
                "## Key concepts (grounded)\n"
                "- Missing citation bullet.\n"
                "- Invalid citation bullet. (c_999)\n"
                "## Details\n"
                "- Detail is fine. (c_001)\n"
            )
        if "You are synthesizing multiple grounded topic notes" in prompt:
            return "# Summary\n- Soft mode synthesis.\n"
        return "# Short form\n- Soft mode checklist.\n"

    monkeypatch.setattr(pl, "call_llm", fake_call_llm)

    cfg = pl.PipelineConfig(grounding_mode="soft")
    topics = [{"topic_id": "t1", "query": "test query"}]

    manifest = pl.run_pipeline(run_id="smoke_soft", config=cfg, topics=topics, outputs_dir=tmp_path)

    assert manifest["topics_completed"] == 1
    assert manifest["call_count"] == 3

    notes_path = tmp_path / "smoke_soft" / "topics" / "t1" / "research_notes.md"
    notes = notes_path.read_text(encoding="utf-8")

    assert "- Unverified:" in notes
    assert (tmp_path / "smoke_soft" / "topics" / "t1" / "grounding_validation.json").exists()
    assert (tmp_path / "smoke_soft" / "summary" / "summary.md").exists()
    assert (tmp_path / "smoke_soft" / "short_form" / "short_form.md").exists()
