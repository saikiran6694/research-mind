import json
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from graph.nodes import _parse_json, _depth_config, should_loop_back


# ── Pure helper tests (no mocking needed) ──────────────────────────────────

class TestParseJson:
    def test_plain_json_array(self):
        result = _parse_json('["a", "b", "c"]')
        assert result == ["a", "b", "c"]

    def test_plain_json_object(self):
        result = _parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_strips_markdown_json_fence(self):
        text = '```json\n["query 1", "query 2"]\n```'
        result = _parse_json(text)
        assert result == ["query 1", "query 2"]

    def test_strips_plain_markdown_fence(self):
        text = '```\n{"findings": []}\n```'
        result = _parse_json(text)
        assert result == {"findings": []}

    def test_strips_leading_trailing_whitespace(self):
        result = _parse_json('  ["a"]  ')
        assert result == ["a"]

    def test_invalid_json_raises(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _parse_json("not valid json at all")

    def test_nested_structure(self):
        data = {"findings": [{"claim": "x", "confidence": 0.9}]}
        result = _parse_json(json.dumps(data))
        assert result["findings"][0]["claim"] == "x"


class TestDepthConfig:
    def test_shallow_config(self):
        cfg = _depth_config("shallow")
        assert cfg["max_queries"] == 3
        assert cfg["max_sources"] == 3
        assert cfg["max_iterations"] == 1

    def test_medium_config(self):
        cfg = _depth_config("medium")
        assert cfg["max_queries"] == 4
        assert cfg["max_sources"] == 5
        assert cfg["max_iterations"] == 2

    def test_deep_config(self):
        cfg = _depth_config("deep")
        assert cfg["max_queries"] == 5
        assert cfg["max_sources"] == 7
        assert cfg["max_iterations"] == 3

    def test_unknown_depth_returns_medium_default(self):
        cfg = _depth_config("extreme")
        assert cfg == _depth_config("medium")


class TestShouldLoopBack:
    def _make_state(self, depth, iteration, gaps):
        from models.schemas import ResearchQuery, ResearchDepth
        query = ResearchQuery(topic="test", depth=ResearchDepth(depth))
        return {"query": query, "iteration": iteration, "knowledge_gaps": gaps}

    def test_loops_when_gaps_and_under_limit(self):
        state = self._make_state("medium", 1, ["gap A", "gap B"])
        assert should_loop_back(state) == "plan_searches"

    def test_synthesizes_when_no_gaps(self):
        state = self._make_state("medium", 1, [])
        assert should_loop_back(state) == "synthesize_report"

    def test_synthesizes_when_at_max_iteration(self):
        state = self._make_state("medium", 2, ["still gaps"])
        assert should_loop_back(state) == "synthesize_report"

    def test_synthesizes_when_over_max_iteration(self):
        state = self._make_state("shallow", 5, ["gap"])
        assert should_loop_back(state) == "synthesize_report"

    def test_shallow_loops_only_once(self):
        before = self._make_state("shallow", 0, ["gap"])
        after = self._make_state("shallow", 1, ["gap"])
        assert should_loop_back(before) == "plan_searches"
        assert should_loop_back(after) == "synthesize_report"

    def test_deep_allows_three_iterations(self):
        state = self._make_state("deep", 2, ["gap"])
        assert should_loop_back(state) == "plan_searches"
        state["iteration"] = 3
        assert should_loop_back(state) == "synthesize_report"


# ── Node function tests (external deps mocked) ─────────────────────────────

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.daily_calls = 0
    return llm


class TestPlanSearches:
    def test_returns_search_queries(self, sample_state, mock_llm):
        queries = ["AI in radiology", "ML clinical trials", "AI ethics medicine"]
        mock_llm.invoke.return_value = json.dumps(queries)

        with patch("graph.nodes.get_llm", return_value=mock_llm), \
             patch("graph.nodes.save_checkpoint"):
            from graph.nodes import plan_searches
            result = plan_searches(sample_state)

        assert result["phase"] == "searching"
        for q in queries:
            assert q in result["search_queries"]

    def test_deduplicates_queries(self, sample_state, mock_llm):
        existing = ["existing query"]
        sample_state["search_queries"] = existing
        mock_llm.invoke.return_value = json.dumps(["existing query", "new query"])

        with patch("graph.nodes.get_llm", return_value=mock_llm), \
             patch("graph.nodes.save_checkpoint"):
            from graph.nodes import plan_searches
            result = plan_searches(sample_state)

        assert result["search_queries"].count("existing query") == 1

    def test_falls_back_to_topic_on_invalid_llm_response(self, sample_state, mock_llm):
        mock_llm.invoke.return_value = "not valid json at all"

        with patch("graph.nodes.get_llm", return_value=mock_llm), \
             patch("graph.nodes.save_checkpoint"):
            from graph.nodes import plan_searches
            result = plan_searches(sample_state)

        assert sample_state["query"].topic in result["search_queries"]

    def test_falls_back_when_llm_returns_non_list(self, sample_state, mock_llm):
        mock_llm.invoke.return_value = '{"not": "a list"}'

        with patch("graph.nodes.get_llm", return_value=mock_llm), \
             patch("graph.nodes.save_checkpoint"):
            from graph.nodes import plan_searches
            result = plan_searches(sample_state)

        assert sample_state["query"].topic in result["search_queries"]

    def test_updates_messages(self, sample_state, mock_llm):
        mock_llm.invoke.return_value = json.dumps(["q1", "q2"])

        with patch("graph.nodes.get_llm", return_value=mock_llm), \
             patch("graph.nodes.save_checkpoint"):
            from graph.nodes import plan_searches
            result = plan_searches(sample_state)

        assert len(result["messages"]) > len(sample_state["messages"])

    def test_saves_checkpoint(self, sample_state, mock_llm):
        mock_llm.invoke.return_value = json.dumps(["q1"])

        with patch("graph.nodes.get_llm", return_value=mock_llm), \
             patch("graph.nodes.save_checkpoint") as mock_save:
            from graph.nodes import plan_searches
            plan_searches(sample_state)

        mock_save.assert_called_once()


class TestExecuteSearches:
    def _mock_search_result(self, url="https://nature.com/ai", content="x" * 500):
        return [{"url": url, "title": "AI Article", "content": content, "score": 0.8}]

    def test_adds_sources_to_state(self, sample_state):
        sample_state["search_queries"] = ["AI healthcare"]
        sample_state["raw_sources"] = []

        with patch("graph.nodes.search_and_extract") as mock_search, \
             patch("graph.nodes.scrape_url") as mock_scrape, \
             patch("graph.nodes.score_source_credibility") as mock_cred, \
             patch("graph.nodes.ingest_store", return_value=["chunk-1"]), \
             patch("graph.nodes.save_checkpoint"):

            mock_search.invoke.return_value = self._mock_search_result()
            mock_cred.invoke.return_value = 0.85

            from graph.nodes import execute_searches
            result = execute_searches(sample_state)

        assert len(result["raw_sources"]) > 0

    def test_skips_duplicate_urls(self, sample_state):
        url = "https://nature.com/ai"
        sample_state["search_queries"] = ["query"]
        sample_state["raw_sources"] = [{
            "url": url, "title": "T", "content": "c",
            "credibility_score": 0.9, "relevance_score": 0.8, "chunk_ids": [],
        }]

        with patch("graph.nodes.search_and_extract") as mock_search, \
             patch("graph.nodes.score_source_credibility") as mock_cred, \
             patch("graph.nodes.ingest_store", return_value=[]), \
             patch("graph.nodes.save_checkpoint"):

            mock_search.invoke.return_value = self._mock_search_result(url=url)
            mock_cred.invoke.return_value = 0.85

            from graph.nodes import execute_searches
            result = execute_searches(sample_state)

        assert len(result["raw_sources"]) == 1

    def test_skips_low_credibility_sources(self, sample_state):
        sample_state["search_queries"] = ["query"]
        sample_state["raw_sources"] = []

        with patch("graph.nodes.search_and_extract") as mock_search, \
             patch("graph.nodes.score_source_credibility") as mock_cred, \
             patch("graph.nodes.ingest_store", return_value=[]), \
             patch("graph.nodes.save_checkpoint"):

            mock_search.invoke.return_value = self._mock_search_result()
            mock_cred.invoke.return_value = 0.10  # below MIN_CREDIBILITY

            from graph.nodes import execute_searches
            result = execute_searches(sample_state)

        assert len(result["raw_sources"]) == 0

    def test_scrapes_short_content(self, sample_state):
        sample_state["search_queries"] = ["query"]
        sample_state["raw_sources"] = []
        short_content = "x" * 100  # < 400 chars

        with patch("graph.nodes.search_and_extract") as mock_search, \
             patch("graph.nodes.scrape_url") as mock_scrape, \
             patch("graph.nodes.score_source_credibility") as mock_cred, \
             patch("graph.nodes.ingest_store", return_value=["chunk-1"]), \
             patch("graph.nodes.save_checkpoint"):

            mock_search.invoke.return_value = self._mock_search_result(content=short_content)
            mock_scrape.invoke.return_value = {"success": True, "content": "Full scraped content " * 50}
            mock_cred.invoke.return_value = 0.90

            from graph.nodes import execute_searches
            result = execute_searches(sample_state)

        mock_scrape.invoke.assert_called_once()
        assert result["raw_sources"][0]["content"] != short_content

    def test_does_not_scrape_long_content(self, sample_state):
        sample_state["search_queries"] = ["query"]
        sample_state["raw_sources"] = []
        long_content = "x" * 500  # > 400 chars

        with patch("graph.nodes.search_and_extract") as mock_search, \
             patch("graph.nodes.scrape_url") as mock_scrape, \
             patch("graph.nodes.score_source_credibility") as mock_cred, \
             patch("graph.nodes.ingest_store", return_value=["chunk-1"]), \
             patch("graph.nodes.save_checkpoint"):

            mock_search.invoke.return_value = self._mock_search_result(content=long_content)
            mock_cred.invoke.return_value = 0.90

            from graph.nodes import execute_searches
            execute_searches(sample_state)

        mock_scrape.invoke.assert_not_called()

    def test_skips_empty_content(self, sample_state):
        sample_state["search_queries"] = ["query"]
        sample_state["raw_sources"] = []

        with patch("graph.nodes.search_and_extract") as mock_search, \
             patch("graph.nodes.scrape_url") as mock_scrape, \
             patch("graph.nodes.save_checkpoint"):

            mock_search.invoke.return_value = self._mock_search_result(content="")
            mock_scrape.invoke.return_value = {"success": False, "content": ""}

            from graph.nodes import execute_searches
            result = execute_searches(sample_state)

        assert len(result["raw_sources"]) == 0

    def test_respects_max_sources_cap(self, sample_state):
        sample_state["search_queries"] = ["q1", "q2", "q3", "q4"]
        sample_state["raw_sources"] = []

        def make_result(i):
            return [{"url": f"https://nature.com/art{i}", "title": "T", "content": "x" * 500, "score": 0.8}]

        with patch("graph.nodes.search_and_extract") as mock_search, \
             patch("graph.nodes.score_source_credibility") as mock_cred, \
             patch("graph.nodes.ingest_store", return_value=["chunk-1"]), \
             patch("graph.nodes.save_checkpoint"):

            mock_search.invoke.side_effect = [make_result(i) for i in range(10)]
            mock_cred.invoke.return_value = 0.90

            from graph.nodes import execute_searches
            result = execute_searches(sample_state)

        # medium depth → max 5 sources
        assert len(result["raw_sources"]) <= 5

    def test_sets_phase_to_analyzing(self, sample_state):
        sample_state["search_queries"] = ["query"]
        sample_state["raw_sources"] = []

        with patch("graph.nodes.search_and_extract") as mock_search, \
             patch("graph.nodes.score_source_credibility") as mock_cred, \
             patch("graph.nodes.ingest_store", return_value=[]), \
             patch("graph.nodes.save_checkpoint"):

            mock_search.invoke.return_value = []
            mock_cred.invoke.return_value = 0.90

            from graph.nodes import execute_searches
            result = execute_searches(sample_state)

        assert result["phase"] == "analyzing"


class TestAnalyseAndCritique:
    def test_merges_new_findings(self, sample_state, mock_llm):
        kb_doc = Document(page_content="background context", metadata={})
        new_finding = {
            "claim": "AI outperforms specialists in imaging",
            "evidence": ["Meta-analysis confirms"],
            "source_urls": ["https://pubmed.ncbi.nlm.nih.gov/789"],
            "confidence": 0.88,
            "contradictions": [],
        }
        mock_llm.invoke.return_value = json.dumps({
            "findings": [new_finding],
            "knowledge_gaps": ["cost data missing"],
            "critique_notes": ["small sample sizes"],
            "overall_quality": 0.75,
        })

        with patch("graph.nodes.get_llm", return_value=mock_llm), \
             patch("graph.nodes.semantic_search", return_value=[kb_doc]), \
             patch("graph.nodes.save_checkpoint"):
            from graph.nodes import analyse_and_critique
            result = analyse_and_critique(sample_state)

        claims = [f["claim"] for f in result["findings"]]
        assert "AI outperforms specialists in imaging" in claims
        assert "AI improves diagnostic accuracy" in claims  # from sample_state

    def test_deduplicates_findings_by_claim(self, sample_state, mock_llm):
        kb_doc = Document(page_content="context", metadata={})
        # Return a finding with the same claim as the one already in sample_state
        duplicate = {
            "claim": "AI improves diagnostic accuracy",
            "evidence": ["Different source"],
            "source_urls": ["https://other.com"],
            "confidence": 0.70,
            "contradictions": [],
        }
        mock_llm.invoke.return_value = json.dumps({
            "findings": [duplicate],
            "knowledge_gaps": [],
            "critique_notes": [],
            "overall_quality": 0.7,
        })

        with patch("graph.nodes.get_llm", return_value=mock_llm), \
             patch("graph.nodes.semantic_search", return_value=[kb_doc]), \
             patch("graph.nodes.save_checkpoint"):
            from graph.nodes import analyse_and_critique
            result = analyse_and_critique(sample_state)

        claims = [f["claim"] for f in result["findings"]]
        assert claims.count("AI improves diagnostic accuracy") == 1

    def test_increments_iteration(self, sample_state, mock_llm):
        kb_doc = Document(page_content="ctx", metadata={})
        mock_llm.invoke.return_value = json.dumps({
            "findings": [], "knowledge_gaps": [], "critique_notes": [], "overall_quality": 0.5
        })

        with patch("graph.nodes.get_llm", return_value=mock_llm), \
             patch("graph.nodes.semantic_search", return_value=[kb_doc]), \
             patch("graph.nodes.save_checkpoint"):
            from graph.nodes import analyse_and_critique
            result = analyse_and_critique(sample_state)

        assert result["iteration"] == sample_state["iteration"] + 1

    def test_handles_parse_failure_gracefully(self, sample_state, mock_llm):
        kb_doc = Document(page_content="ctx", metadata={})
        mock_llm.invoke.return_value = "this is not json"

        with patch("graph.nodes.get_llm", return_value=mock_llm), \
             patch("graph.nodes.semantic_search", return_value=[kb_doc]), \
             patch("graph.nodes.save_checkpoint"):
            from graph.nodes import analyse_and_critique
            result = analyse_and_critique(sample_state)

        # Should not raise; keeps existing findings
        assert "findings" in result
        assert result["knowledge_gaps"] == []

    def test_updates_knowledge_gaps(self, sample_state, mock_llm):
        kb_doc = Document(page_content="ctx", metadata={})
        mock_llm.invoke.return_value = json.dumps({
            "findings": [],
            "knowledge_gaps": ["gap A", "gap B"],
            "critique_notes": ["note"],
            "overall_quality": 0.6,
        })

        with patch("graph.nodes.get_llm", return_value=mock_llm), \
             patch("graph.nodes.semantic_search", return_value=[kb_doc]), \
             patch("graph.nodes.save_checkpoint"):
            from graph.nodes import analyse_and_critique
            result = analyse_and_critique(sample_state)

        assert "gap A" in result["knowledge_gaps"]
        assert "gap B" in result["knowledge_gaps"]


class TestSynthesizeReport:
    def _make_state_with_sources(self, sample_state, sample_source):
        sample_state["validated_sources"] = [sample_source]
        return sample_state

    def test_creates_research_report(self, sample_state, sample_source, mock_llm):
        state = self._make_state_with_sources(sample_state, sample_source)
        mock_llm.invoke.return_value = json.dumps({
            "topic": state["query"].topic,
            "summary": "AI is transforming healthcare.",
            "key_findings": [{
                "claim": "AI matches specialist accuracy",
                "evidence": ["Multiple studies"],
                "source_urls": ["https://nature.com/ai"],
                "confidence": 0.87,
                "contradictions": [],
            }],
            "gaps_identified": ["long-term data missing"],
            "follow_up_queries": ["AI safety in medicine"],
            "confidence_overall": 0.82,
        })

        with patch("graph.nodes.get_llm", return_value=mock_llm), \
             patch("graph.nodes.save_checkpoint"):
            from graph.nodes import synthesize_report
            result = synthesize_report(state)

        from models.schemas import ResearchReport
        assert isinstance(result["report"], ResearchReport)
        assert result["phase"] == "done"

    def test_report_includes_sources(self, sample_state, sample_source, mock_llm):
        state = self._make_state_with_sources(sample_state, sample_source)
        mock_llm.invoke.return_value = json.dumps({
            "topic": state["query"].topic,
            "summary": "Summary text.",
            "key_findings": [],
            "gaps_identified": [],
            "follow_up_queries": [],
            "confidence_overall": 0.7,
        })

        with patch("graph.nodes.get_llm", return_value=mock_llm), \
             patch("graph.nodes.save_checkpoint"):
            from graph.nodes import synthesize_report
            result = synthesize_report(state)

        assert len(result["report"].sources) > 0

    def test_uses_fallback_report_on_parse_failure(self, sample_state, sample_source, mock_llm):
        state = self._make_state_with_sources(sample_state, sample_source)
        mock_llm.invoke.return_value = "completely invalid json {"

        with patch("graph.nodes.get_llm", return_value=mock_llm), \
             patch("graph.nodes.save_checkpoint"):
            from graph.nodes import synthesize_report
            result = synthesize_report(state)

        from models.schemas import ResearchReport
        assert isinstance(result["report"], ResearchReport)
        assert result["report"].confidence_overall == 0.4  # fallback value

    def test_sets_word_count(self, sample_state, sample_source, mock_llm):
        state = self._make_state_with_sources(sample_state, sample_source)
        summary = "AI is transforming healthcare through improved diagnostics and efficiency."
        mock_llm.invoke.return_value = json.dumps({
            "topic": state["query"].topic,
            "summary": summary,
            "key_findings": [],
            "gaps_identified": [],
            "follow_up_queries": [],
            "confidence_overall": 0.75,
        })

        with patch("graph.nodes.get_llm", return_value=mock_llm), \
             patch("graph.nodes.save_checkpoint"):
            from graph.nodes import synthesize_report
            result = synthesize_report(state)

        assert result["report"].word_count == len(summary.split())
