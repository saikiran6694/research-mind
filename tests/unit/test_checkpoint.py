import json
import pytest
from pathlib import Path
from unittest.mock import patch
from datetime import datetime

import utils.checkpoint as cp_module
from utils.checkpoint import (
    _topic_slug,
    save_checkpoint,
    load_checkpoint,
    clear_checkpoint,
    list_checkpoints,
    _serialise_state,
)
from models.schemas import ResearchQuery, ResearchDepth, Finding


@pytest.fixture(autouse=True)
def isolated_checkpoint_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cp_module, "CHECKPOINT_DIR", tmp_path)
    return tmp_path


class TestTopicSlug:
    def test_spaces_become_underscores(self):
        assert _topic_slug("hello world") == "hello_world"

    def test_special_chars_removed(self):
        slug = _topic_slug("AI & healthcare: 2024!")
        assert "&" not in slug
        assert ":" not in slug
        assert "!" not in slug

    def test_lowercased(self):
        assert _topic_slug("AI Healthcare") == "ai_healthcare"

    def test_truncated_to_60_chars(self):
        long_topic = "a" * 100
        assert len(_topic_slug(long_topic)) <= 60

    def test_short_topic_unchanged_length(self):
        slug = _topic_slug("short topic")
        assert len(slug) <= 60


class TestSerialiseState:
    def test_serialises_plain_values(self):
        state = {"phase": "searching", "iteration": 2, "gaps": ["gap1"]}
        result = _serialise_state(state)
        assert result["phase"] == "searching"
        assert result["iteration"] == 2

    def test_serialises_pydantic_model(self):
        query = ResearchQuery(topic="AI", depth=ResearchDepth.MEDIUM)
        state = {"query": query}
        result = _serialise_state(state)
        assert isinstance(result["query"], dict)
        assert result["query"]["topic"] == "AI"

    def test_serialises_list_of_pydantic_models(self):
        findings = [
            Finding(claim="c1", evidence=[], source_urls=[], confidence=0.8),
            Finding(claim="c2", evidence=[], source_urls=[], confidence=0.6),
        ]
        state = {"findings": findings}
        result = _serialise_state(state)
        assert isinstance(result["findings"], list)
        assert result["findings"][0]["claim"] == "c1"

    def test_non_serialisable_falls_back_to_str(self):
        state = {"obj": object()}
        result = _serialise_state(state)
        assert isinstance(result["obj"], str)

    def test_mixed_list(self):
        query = ResearchQuery(topic="AI")
        state = {"items": [query, "plain string", 42]}
        result = _serialise_state(state)
        assert isinstance(result["items"][0], dict)
        assert result["items"][1] == "plain string"
        assert result["items"][2] == 42


class TestSaveAndLoadCheckpoint:
    def test_save_creates_file(self, tmp_path):
        state = {"phase": "searching", "iteration": 1, "gaps": []}
        save_checkpoint("test topic", "searching", state)
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1

    def test_load_returns_none_when_no_checkpoint(self):
        assert load_checkpoint("nonexistent topic") is None

    def test_save_then_load_roundtrip(self):
        state = {"phase": "analyzing", "iteration": 2, "findings": []}
        save_checkpoint("my research topic", "analyzing", state)
        loaded = load_checkpoint("my research topic")
        assert loaded is not None
        assert loaded["_checkpoint_phase"] == "analyzing"
        assert loaded["_checkpoint_topic"] == "my research topic"

    def test_checkpoint_has_saved_at_timestamp(self):
        save_checkpoint("topic", "searching", {"phase": "searching"})
        loaded = load_checkpoint("topic")
        assert "_checkpoint_saved_at" in loaded
        # Should be a valid datetime string
        datetime.fromisoformat(loaded["_checkpoint_saved_at"])

    def test_checkpoint_saves_state_fields(self):
        state = {"phase": "synthesizing", "iteration": 3, "knowledge_gaps": ["gap A"]}
        save_checkpoint("ai topic", "synthesizing", state)
        loaded = load_checkpoint("ai topic")
        assert loaded["knowledge_gaps"] == ["gap A"]

    def test_same_topic_overwrites_checkpoint(self):
        save_checkpoint("topic", "searching", {"iteration": 1})
        save_checkpoint("topic", "analyzing", {"iteration": 2})
        loaded = load_checkpoint("topic")
        assert loaded["_checkpoint_phase"] == "analyzing"

    def test_slug_collision_across_similar_topics(self):
        save_checkpoint("AI in healthcare", "searching", {"iteration": 1})
        loaded = load_checkpoint("AI in healthcare")
        assert loaded is not None


class TestClearCheckpoint:
    def test_clear_removes_file(self, tmp_path):
        save_checkpoint("test topic", "done", {"phase": "done"})
        clear_checkpoint("test topic")
        assert load_checkpoint("test topic") is None

    def test_clear_nonexistent_is_noop(self):
        clear_checkpoint("topic that was never saved")  # should not raise


class TestListCheckpoints:
    def test_empty_dir_returns_empty(self):
        assert list_checkpoints() == []

    def test_lists_saved_checkpoints(self):
        save_checkpoint("topic A", "searching", {"phase": "searching"})
        save_checkpoint("topic B", "analyzing", {"phase": "analyzing"})
        results = list_checkpoints()
        assert len(results) == 2

    def test_metadata_fields_present(self):
        save_checkpoint("my topic", "done", {"phase": "done"})
        results = list_checkpoints()
        entry = results[0]
        assert "topic" in entry
        assert "phase" in entry
        assert "saved" in entry
        assert "file" in entry

    def test_checkpoint_topic_matches(self):
        save_checkpoint("specific topic", "searching", {"phase": "searching"})
        results = list_checkpoints()
        assert any(r["topic"] == "specific topic" for r in results)

    def test_checkpoint_phase_matches(self):
        save_checkpoint("topic", "synthesizing", {"phase": "synthesizing"})
        results = list_checkpoints()
        assert results[0]["phase"] == "synthesizing"

    def test_corrupt_file_skipped(self, tmp_path):
        (tmp_path / "corrupt.json").write_text("{ not valid json }")
        save_checkpoint("good topic", "done", {"phase": "done"})
        results = list_checkpoints()
        assert len(results) == 1
