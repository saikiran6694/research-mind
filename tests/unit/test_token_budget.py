import pytest
from utils.token_budget import (
    trim,
    trim_sources,
    build_sources_block,
    safe_json_list,
    SOURCE_CONTENT_CHARS,
    MAX_SOURCES_PER_BATCH,
)


class TestTrim:
    def test_short_text_unchanged(self):
        text = "Hello world"
        assert trim(text, 100) == text

    def test_exactly_at_budget_unchanged(self):
        text = "A" * 100
        assert trim(text, 100) == text

    def test_over_budget_is_truncated(self):
        text = "A" * 200
        result = trim(text, 100)
        assert len(result) < 200
        assert result.startswith("A" * 100)

    def test_truncation_adds_marker(self):
        text = "A" * 200
        result = trim(text, 100)
        assert "[trimmed for token budget]" in result

    def test_truncated_starts_with_correct_prefix(self):
        text = "Hello " * 50
        result = trim(text, 20)
        assert result[:20] == text[:20]

    def test_empty_string(self):
        assert trim("", 100) == ""

    def test_zero_budget_adds_marker(self):
        result = trim("some text", 0)
        assert "[trimmed for token budget]" in result


class TestTrimSources:
    def _make_source(self, url="https://example.com", content="x" * 1000):
        return {"url": url, "title": "Title", "content": content, "credibility_score": 0.8}

    def test_caps_at_max_sources(self):
        sources = [self._make_source(f"https://example{i}.com") for i in range(10)]
        result = trim_sources(sources, max_sources=3)
        assert len(result) == 3

    def test_trims_content_field(self):
        source = self._make_source(content="A" * 2000)
        result = trim_sources([source], max_sources=1, content_chars=100)
        assert len(result[0]["content"]) < 2000

    def test_does_not_mutate_original(self):
        original_content = "A" * 2000
        source = self._make_source(content=original_content)
        trim_sources([source], max_sources=1, content_chars=100)
        assert source["content"] == original_content

    def test_short_content_unchanged(self):
        source = self._make_source(content="Short content")
        result = trim_sources([source], max_sources=1, content_chars=500)
        assert result[0]["content"] == "Short content"

    def test_non_content_fields_preserved(self):
        source = self._make_source()
        result = trim_sources([source], max_sources=1)
        assert result[0]["url"] == source["url"]
        assert result[0]["title"] == source["title"]
        assert result[0]["credibility_score"] == source["credibility_score"]

    def test_empty_list(self):
        assert trim_sources([], max_sources=5) == []

    def test_fewer_sources_than_max_returns_all(self):
        sources = [self._make_source(f"https://ex{i}.com") for i in range(2)]
        result = trim_sources(sources, max_sources=10)
        assert len(result) == 2

    def test_missing_content_field_handled(self):
        source = {"url": "https://x.com", "title": "T", "credibility_score": 0.8}
        result = trim_sources([source], max_sources=1)
        assert result[0]["content"] == ""


class TestBuildSourcesBlock:
    def _make_source(self, i=1, url="https://ex.com", title="Title", content="Content here"):
        return {"url": url, "title": title, "content": content}

    def test_formats_single_source(self):
        sources = [self._make_source()]
        block = build_sources_block(sources)
        assert "SOURCE 1" in block
        assert "https://ex.com" in block
        assert "Content here" in block

    def test_formats_multiple_sources(self):
        sources = [
            {"url": "https://a.com", "title": "A", "content": "Content A"},
            {"url": "https://b.com", "title": "B", "content": "Content B"},
        ]
        block = build_sources_block(sources)
        assert "SOURCE 1" in block
        assert "SOURCE 2" in block
        assert "Content A" in block
        assert "Content B" in block

    def test_sources_separated(self):
        sources = [
            {"url": "https://a.com", "title": "A", "content": "A"},
            {"url": "https://b.com", "title": "B", "content": "B"},
        ]
        block = build_sources_block(sources)
        assert block.index("SOURCE 1") < block.index("SOURCE 2")

    def test_empty_list_returns_empty_string(self):
        assert build_sources_block([]) == ""

    def test_missing_title_uses_untitled(self):
        sources = [{"url": "https://x.com", "content": "C"}]
        block = build_sources_block(sources)
        assert "Untitled" in block

    def test_missing_url_handled(self):
        sources = [{"title": "T", "content": "C"}]
        block = build_sources_block(sources)
        assert "SOURCE 1" in block


class TestSafeJsonList:
    def test_returns_all_when_under_limit(self):
        data = [1, 2, 3]
        assert safe_json_list(data, 10) == [1, 2, 3]

    def test_caps_at_max_items(self):
        data = list(range(20))
        result = safe_json_list(data, 5)
        assert len(result) == 5
        assert result == [0, 1, 2, 3, 4]

    def test_empty_list(self):
        assert safe_json_list([], 10) == []

    def test_exactly_at_limit(self):
        data = list(range(5))
        assert safe_json_list(data, 5) == data

    def test_does_not_mutate_original(self):
        data = list(range(10))
        safe_json_list(data, 3)
        assert len(data) == 10
