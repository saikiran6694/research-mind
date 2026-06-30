import os
import json
import pytest

# Must be set before any project module is imported by pytest
os.environ.setdefault("GOOGLE_API_KEY", "fake-key-for-tests")
os.environ.setdefault("TAVILY_API_KEY", "fake-key-for-tests")

from models.schemas import (
    ResearchDepth,
    ResearchQuery,
    Finding,
    Source,
    ResearchReport,
)


@pytest.fixture
def sample_query():
    return ResearchQuery(
        topic="artificial intelligence in healthcare",
        depth=ResearchDepth.MEDIUM,
        focus_areas=["ethics", "clinical applications"],
    )


@pytest.fixture
def sample_finding():
    return {
        "claim": "AI improves diagnostic accuracy",
        "evidence": ["Study shows 94% accuracy in radiology"],
        "source_urls": ["https://pubmed.ncbi.nlm.nih.gov/123"],
        "confidence": 0.85,
        "contradictions": [],
    }


@pytest.fixture
def sample_source():
    return {
        "url": "https://nature.com/articles/ai-health",
        "title": "AI in Medicine",
        "content": "This peer-reviewed study examines the role of AI in healthcare diagnostics.",
        "credibility_score": 0.90,
        "relevance_score": 0.75,
        "chunk_ids": ["chunk-1", "chunk-2"],
    }


@pytest.fixture
def sample_state(sample_query, sample_finding, sample_source):
    return {
        "query": sample_query,
        "messages": [],
        "iteration": 0,
        "phase": "planning",
        "search_queries": [],
        "raw_sources": [sample_source],
        "validated_sources": [],
        "findings": [sample_finding],
        "knowledge_gaps": ["long-term outcomes not studied"],
        "critique_notes": [],
        "report": None,
        "error": None,
    }


@pytest.fixture
def planner_llm_response():
    return json.dumps([
        "AI diagnostic tools in radiology",
        "machine learning clinical trials",
        "AI ethics healthcare bias",
        "deep learning patient outcomes",
    ])


@pytest.fixture
def analysis_llm_response():
    return json.dumps({
        "findings": [
            {
                "claim": "AI outperforms radiologists in image analysis",
                "evidence": ["Meta-analysis of 14 studies shows AUC 0.96"],
                "source_urls": ["https://pubmed.ncbi.nlm.nih.gov/456"],
                "confidence": 0.88,
                "contradictions": [],
            }
        ],
        "knowledge_gaps": ["cost-effectiveness data missing"],
        "critique_notes": ["sample sizes small in several studies"],
        "overall_quality": 0.78,
    })


@pytest.fixture
def report_llm_response(sample_query):
    return json.dumps({
        "topic": sample_query.topic,
        "summary": "AI is revolutionizing healthcare by improving diagnostic accuracy and efficiency.",
        "key_findings": [
            {
                "claim": "AI matches or exceeds specialist accuracy",
                "evidence": ["Multiple meta-analyses confirm this"],
                "source_urls": ["https://nature.com/articles/ai-health"],
                "confidence": 0.87,
                "contradictions": [],
            }
        ],
        "gaps_identified": ["long-term patient outcome data lacking"],
        "follow_up_queries": ["AI regulation healthcare 2024", "AI bias clinical settings"],
        "confidence_overall": 0.82,
    })
