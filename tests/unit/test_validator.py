import pytest
from tools.validator import score_source_credibility, filter_sources, TRUSTED_DOMAINS, PENALIZED_DOMAINS


class TestScoreSourceCredibility:
    def _score(self, url, content_length=500):
        return score_source_credibility.invoke({"url": url, "content_length": content_length})

    def test_trusted_domain_scores_high(self):
        score = self._score("https://nature.com/articles/ai-2024")
        assert score >= 0.90

    def test_pubmed_scores_high(self):
        score = self._score("https://pubmed.ncbi.nlm.nih.gov/123456")
        assert score >= 0.90

    def test_arxiv_scores_high(self):
        score = self._score("https://arxiv.org/abs/2401.00001")
        assert score >= 0.85

    def test_penalized_domain_scores_lower(self):
        reddit = self._score("https://reddit.com/r/MachineLearning/comments/abc")
        unknown = self._score("https://someunknownblog.com/post/ai")
        assert reddit < unknown

    def test_unknown_domain_gets_base_score(self):
        score = self._score("https://totallynewsite.io/article")
        assert score == 0.50

    def test_long_content_boosts_score(self):
        short = self._score("https://somesite.com", content_length=100)
        long = self._score("https://somesite.com", content_length=2000)
        assert long > short

    def test_long_content_boost_is_0_05(self):
        base = self._score("https://somesite.com", content_length=100)
        boosted = self._score("https://somesite.com", content_length=2000)
        assert abs(boosted - base - 0.05) < 0.001

    def test_score_clamped_to_1(self):
        score = self._score("https://nature.com/article", content_length=5000)
        assert score <= 1.0

    def test_score_clamped_to_0(self):
        score = self._score("https://quora.com/What-is-AI", content_length=0)
        assert score >= 0.0

    def test_www_prefix_stripped(self):
        with_www = self._score("https://www.nature.com/article")
        without_www = self._score("https://nature.com/article")
        assert with_www == without_www

    def test_score_is_rounded_to_2_decimals(self):
        score = self._score("https://nature.com/article")
        assert score == round(score, 2)

    def test_all_trusted_domains_exceed_min_credibility(self):
        for domain in TRUSTED_DOMAINS:
            score = self._score(f"https://{domain}/some-article")
            assert score >= 0.35, f"{domain} scored below minimum: {score}"


class TestFilterSources:
    def _make_source(self, url, score):
        return {"url": url, "title": "T", "content": "C", "credibility_score": score}

    def test_filters_low_credibility(self):
        sources = [
            self._make_source("https://good.com", 0.80),
            self._make_source("https://bad.com", 0.20),
        ]
        result = filter_sources(sources)
        assert len(result) == 1
        assert result[0]["url"] == "https://good.com"

    def test_keeps_sources_at_threshold(self):
        sources = [self._make_source("https://ok.com", 0.35)]
        result = filter_sources(sources)
        assert len(result) == 1

    def test_empty_list_returns_empty(self):
        assert filter_sources([]) == []

    def test_all_filtered_returns_empty(self):
        sources = [self._make_source("https://junk.com", 0.10)]
        assert filter_sources(sources) == []

    def test_all_pass_when_all_above_threshold(self):
        sources = [
            self._make_source("https://a.com", 0.90),
            self._make_source("https://b.com", 0.80),
            self._make_source("https://c.com", 0.70),
        ]
        assert len(filter_sources(sources)) == 3

    def test_custom_min_score(self):
        sources = [
            self._make_source("https://a.com", 0.60),
            self._make_source("https://b.com", 0.40),
        ]
        result = filter_sources(sources, min_score=0.55)
        assert len(result) == 1
        assert result[0]["url"] == "https://a.com"

    def test_missing_credibility_score_excluded(self):
        sources = [{"url": "https://x.com", "title": "T", "content": "C"}]
        result = filter_sources(sources)
        assert result == []
