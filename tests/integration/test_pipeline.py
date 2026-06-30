"""
Integration tests: run the full LangGraph pipeline with all external
calls mocked. Verifies the graph wiring, state flow, and node interactions
without hitting any real APIs.
"""
import json
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from models.schemas import ResearchQuery, ResearchDepth
from main import build_initial_state


PLANNER_RESPONSE = json.dumps([
    "AI diagnostic tools radiology",
    "machine learning clinical trials",
    "AI ethics healthcare",
])

ANALYSIS_RESPONSE = json.dumps({
    "findings": [
        {
            "claim": "AI matches radiologist accuracy in chest X-ray diagnosis",
            "evidence": ["Meta-analysis of 14 RCTs, AUC 0.96"],
            "source_urls": ["https://nature.com/ai-xray"],
            "confidence": 0.88,
            "contradictions": [],
        }
    ],
    "knowledge_gaps": [],  # empty → triggers synthesis in shallow run
    "critique_notes": ["sample sizes vary across studies"],
    "overall_quality": 0.80,
})

REPORT_RESPONSE = json.dumps({
    "topic": "artificial intelligence in healthcare",
    "summary": "AI is transforming healthcare diagnostics with high accuracy rates.",
    "key_findings": [
        {
            "claim": "AI matches radiologist accuracy",
            "evidence": ["Meta-analysis confirms AUC 0.96"],
            "source_urls": ["https://nature.com/ai-xray"],
            "confidence": 0.88,
            "contradictions": [],
        }
    ],
    "gaps_identified": ["long-term patient outcomes unclear"],
    "follow_up_queries": ["AI regulation healthcare", "AI bias clinical settings"],
    "confidence_overall": 0.84,
})

MOCK_SEARCH_RESULTS = [
    {
        "url": "https://nature.com/ai-xray",
        "title": "AI in Radiology",
        "content": "Comprehensive study of AI applications in diagnostic imaging. " * 30,
        "score": 0.92,
    }
]


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.daily_calls = 0
    llm.invoke.side_effect = [PLANNER_RESPONSE, ANALYSIS_RESPONSE, REPORT_RESPONSE]
    return llm


@pytest.fixture
def kb_docs():
    return [Document(page_content="Related healthcare AI context.", metadata={})]


@pytest.fixture
def shallow_state():
    query = ResearchQuery(
        topic="artificial intelligence in healthcare",
        depth=ResearchDepth.SHALLOW,
        focus_areas=["diagnostics"],
    )
    return build_initial_state(query)


class TestFullPipelineShallowRun:
    def test_pipeline_completes_without_error(self, shallow_state, mock_llm, kb_docs):
        from graph.builder import build_research_graph

        with patch("graph.nodes.get_llm", return_value=mock_llm), \
             patch("graph.nodes.search_and_extract") as mock_search, \
             patch("graph.nodes.scrape_url") as mock_scrape, \
             patch("graph.nodes.score_source_credibility") as mock_cred, \
             patch("graph.nodes.ingest_store", return_value=["chunk-1", "chunk-2"]), \
             patch("graph.nodes.semantic_search", return_value=kb_docs), \
             patch("graph.nodes.save_checkpoint"):

            mock_search.invoke.return_value = MOCK_SEARCH_RESULTS
            mock_cred.invoke.return_value = 0.92

            graph = build_research_graph()
            final_state = None
            for event in graph.stream(shallow_state):
                final_state = list(event.values())[0]

        assert final_state is not None
        assert final_state.get("error") is None

    def test_pipeline_produces_report(self, shallow_state, mock_llm, kb_docs):
        from graph.builder import build_research_graph
        from models.schemas import ResearchReport

        with patch("graph.nodes.get_llm", return_value=mock_llm), \
             patch("graph.nodes.search_and_extract") as mock_search, \
             patch("graph.nodes.scrape_url") as mock_scrape, \
             patch("graph.nodes.score_source_credibility") as mock_cred, \
             patch("graph.nodes.ingest_store", return_value=["chunk-1"]), \
             patch("graph.nodes.semantic_search", return_value=kb_docs), \
             patch("graph.nodes.save_checkpoint"):

            mock_search.invoke.return_value = MOCK_SEARCH_RESULTS
            mock_cred.invoke.return_value = 0.92

            graph = build_research_graph()
            final_state = None
            for event in graph.stream(shallow_state):
                node_name = list(event.keys())[0]
                final_state = list(event.values())[0]

        report = final_state.get("report")
        assert report is not None
        assert isinstance(report, ResearchReport)
        assert report.topic == shallow_state["query"].topic

    def test_pipeline_phase_transitions(self, shallow_state, mock_llm, kb_docs):
        from graph.builder import build_research_graph

        phases_seen = []

        with patch("graph.nodes.get_llm", return_value=mock_llm), \
             patch("graph.nodes.search_and_extract") as mock_search, \
             patch("graph.nodes.scrape_url"), \
             patch("graph.nodes.score_source_credibility") as mock_cred, \
             patch("graph.nodes.ingest_store", return_value=["chunk-1"]), \
             patch("graph.nodes.semantic_search", return_value=kb_docs), \
             patch("graph.nodes.save_checkpoint"):

            mock_search.invoke.return_value = MOCK_SEARCH_RESULTS
            mock_cred.invoke.return_value = 0.92

            graph = build_research_graph()
            for event in graph.stream(shallow_state):
                node_state = list(event.values())[0]
                phase = node_state.get("phase")
                if phase and phase not in phases_seen:
                    phases_seen.append(phase)

        assert "searching" in phases_seen
        assert "analyzing" in phases_seen
        assert "synthesizing" in phases_seen
        assert "done" in phases_seen

    def test_pipeline_iteration_count(self, shallow_state, mock_llm, kb_docs):
        from graph.builder import build_research_graph

        with patch("graph.nodes.get_llm", return_value=mock_llm), \
             patch("graph.nodes.search_and_extract") as mock_search, \
             patch("graph.nodes.scrape_url"), \
             patch("graph.nodes.score_source_credibility") as mock_cred, \
             patch("graph.nodes.ingest_store", return_value=["chunk-1"]), \
             patch("graph.nodes.semantic_search", return_value=kb_docs), \
             patch("graph.nodes.save_checkpoint"):

            mock_search.invoke.return_value = MOCK_SEARCH_RESULTS
            mock_cred.invoke.return_value = 0.92

            graph = build_research_graph()
            final_state = None
            for event in graph.stream(shallow_state):
                final_state = list(event.values())[0]

        # Shallow → max 1 iteration
        assert final_state["iteration"] <= 1

    def test_llm_called_expected_number_of_times(self, shallow_state, mock_llm, kb_docs):
        from graph.builder import build_research_graph

        with patch("graph.nodes.get_llm", return_value=mock_llm), \
             patch("graph.nodes.search_and_extract") as mock_search, \
             patch("graph.nodes.scrape_url"), \
             patch("graph.nodes.score_source_credibility") as mock_cred, \
             patch("graph.nodes.ingest_store", return_value=["chunk-1"]), \
             patch("graph.nodes.semantic_search", return_value=kb_docs), \
             patch("graph.nodes.save_checkpoint"):

            mock_search.invoke.return_value = MOCK_SEARCH_RESULTS
            mock_cred.invoke.return_value = 0.92

            graph = build_research_graph()
            for event in graph.stream(shallow_state):
                pass

        # shallow = plan(1) + analyse(1) + synthesize(1) = 3 calls
        assert mock_llm.invoke.call_count == 3

    def test_pipeline_checkpoints_saved_at_each_node(self, shallow_state, mock_llm, kb_docs):
        from graph.builder import build_research_graph

        with patch("graph.nodes.get_llm", return_value=mock_llm), \
             patch("graph.nodes.search_and_extract") as mock_search, \
             patch("graph.nodes.scrape_url"), \
             patch("graph.nodes.score_source_credibility") as mock_cred, \
             patch("graph.nodes.ingest_store", return_value=["chunk-1"]), \
             patch("graph.nodes.semantic_search", return_value=kb_docs), \
             patch("graph.nodes.save_checkpoint") as mock_save:

            mock_search.invoke.return_value = MOCK_SEARCH_RESULTS
            mock_cred.invoke.return_value = 0.92

            graph = build_research_graph()
            for event in graph.stream(shallow_state):
                pass

        # plan, execute, analyse, synthesize → 4 checkpoint saves
        assert mock_save.call_count == 4

    def test_pipeline_report_confidence_in_valid_range(self, shallow_state, mock_llm, kb_docs):
        from graph.builder import build_research_graph

        with patch("graph.nodes.get_llm", return_value=mock_llm), \
             patch("graph.nodes.search_and_extract") as mock_search, \
             patch("graph.nodes.scrape_url"), \
             patch("graph.nodes.score_source_credibility") as mock_cred, \
             patch("graph.nodes.ingest_store", return_value=["chunk-1"]), \
             patch("graph.nodes.semantic_search", return_value=kb_docs), \
             patch("graph.nodes.save_checkpoint"):

            mock_search.invoke.return_value = MOCK_SEARCH_RESULTS
            mock_cred.invoke.return_value = 0.92

            graph = build_research_graph()
            final_state = None
            for event in graph.stream(shallow_state):
                final_state = list(event.values())[0]

        confidence = final_state["report"].confidence_overall
        assert 0.0 <= confidence <= 1.0


class TestPipelineResumeFromCheckpoint:
    def test_restore_state_preserves_existing_findings(self, shallow_state):
        from main import restore_state
        checkpoint = {
            "phase": "synthesizing",
            "iteration": 1,
            "findings": [{"claim": "existing claim", "evidence": [], "source_urls": [], "confidence": 0.7, "contradictions": []}],
            "knowledge_gaps": [],
            "critique_notes": [],
            "search_queries": ["already done query"],
            "raw_sources": [],
            "validated_sources": [],
            "messages": [],
            "_checkpoint_phase": "synthesizing",
            "_checkpoint_topic": shallow_state["query"].topic,
        }
        restored = restore_state(checkpoint, shallow_state["query"])
        assert restored["findings"] == checkpoint["findings"]
        assert restored["search_queries"] == ["already done query"]

    def test_build_initial_state_has_correct_keys(self, shallow_state):
        required_keys = {
            "query", "messages", "iteration", "phase",
            "search_queries", "raw_sources", "validated_sources",
            "findings", "knowledge_gaps", "critique_notes",
            "report", "error",
        }
        assert required_keys.issubset(set(shallow_state.keys()))

    def test_build_initial_state_starts_at_planning(self, shallow_state):
        assert shallow_state["phase"] == "planning"
        assert shallow_state["iteration"] == 0
        assert shallow_state["findings"] == []
