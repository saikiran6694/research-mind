import pytest
from pydantic import ValidationError
from models.schemas import ResearchDepth, ResearchQuery, Finding, Source, ResearchReport


class TestResearchDepth:
    def test_enum_values(self):
        assert ResearchDepth.SHALLOW == "shallow"
        assert ResearchDepth.MEDIUM == "medium"
        assert ResearchDepth.DEPTH == "deep"

    def test_from_string(self):
        assert ResearchDepth("shallow") == ResearchDepth.SHALLOW
        assert ResearchDepth("medium") == ResearchDepth.MEDIUM
        assert ResearchDepth("deep") == ResearchDepth.DEPTH

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            ResearchDepth("extreme")


class TestResearchQuery:
    def test_defaults(self):
        q = ResearchQuery(topic="climate change")
        assert q.depth == ResearchDepth.MEDIUM
        assert q.focus_areas == []
        assert q.exclude_domains == []

    def test_full_construction(self):
        q = ResearchQuery(
            topic="AI ethics",
            depth=ResearchDepth.DEPTH,
            focus_areas=["bias", "fairness"],
            exclude_domains=["reddit.com"],
        )
        assert q.topic == "AI ethics"
        assert q.depth == ResearchDepth.DEPTH
        assert "bias" in q.focus_areas
        assert "reddit.com" in q.exclude_domains

    def test_missing_topic_raises(self):
        with pytest.raises(ValidationError):
            ResearchQuery()


class TestFinding:
    def test_valid_finding(self):
        f = Finding(
            claim="AI improves accuracy",
            evidence=["Study A shows 94%"],
            source_urls=["https://pubmed.ncbi.nlm.nih.gov/1"],
            confidence=0.85,
        )
        assert f.confidence == 0.85
        assert f.contradictions == []

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            Finding(claim="x", evidence=[], source_urls=[], confidence=1.5)
        with pytest.raises(ValidationError):
            Finding(claim="x", evidence=[], source_urls=[], confidence=-0.1)

    def test_confidence_boundary_values(self):
        f_min = Finding(claim="x", evidence=[], source_urls=[], confidence=0.0)
        f_max = Finding(claim="x", evidence=[], source_urls=[], confidence=1.0)
        assert f_min.confidence == 0.0
        assert f_max.confidence == 1.0


class TestSource:
    def test_valid_source(self):
        s = Source(
            url="https://nature.com/ai",
            title="AI Study",
            content="Some content here.",
            credibility_score=0.9,
            relevance_score=0.8,
        )
        assert s.chunk_ids == []

    def test_score_bounds(self):
        with pytest.raises(ValidationError):
            Source(url="x", title="x", content="x", credibility_score=1.5, relevance_score=0.5)
        with pytest.raises(ValidationError):
            Source(url="x", title="x", content="x", credibility_score=0.5, relevance_score=-1.0)


class TestResearchReport:
    def test_valid_report(self, sample_query):
        f = Finding(
            claim="AI is useful",
            evidence=["Evidence A"],
            source_urls=["https://example.com"],
            confidence=0.8,
        )
        s = Source(
            url="https://example.com",
            title="Example",
            content="content",
            credibility_score=0.7,
            relevance_score=0.6,
        )
        r = ResearchReport(
            topic=sample_query.topic,
            summary="AI is transforming healthcare.",
            key_findings=[f],
            sources=[s],
            gaps_identified=["long-term data missing"],
            follow_up_queries=["AI safety"],
            word_count=5,
            confidence_overall=0.82,
        )
        assert r.topic == sample_query.topic
        assert len(r.key_findings) == 1
        assert len(r.sources) == 1

    def test_model_dump_is_serializable(self, sample_query):
        import json
        f = Finding(claim="x", evidence=[], source_urls=[], confidence=0.5)
        s = Source(url="u", title="t", content="c", credibility_score=0.5, relevance_score=0.5)
        r = ResearchReport(
            topic=sample_query.topic,
            summary="summary",
            key_findings=[f],
            sources=[s],
            gaps_identified=[],
            follow_up_queries=[],
            word_count=1,
            confidence_overall=0.5,
        )
        dumped = r.model_dump()
        assert json.dumps(dumped, default=str) is not None
