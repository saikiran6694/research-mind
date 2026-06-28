from langchain.tools import tool
from urllib.parse import urlparse

# Simple heuristic domain credibility scoring
TRUSTED_DOMAINS = {
    "nature.com": 0.95, "pubmed.ncbi.nlm.nih.gov": 0.95,
    "arxiv.org": 0.90,  "scholar.google.com": 0.85,
    "wikipedia.org": 0.70, "bbc.com": 0.80,
    "reuters.com": 0.85,   "nytimes.com": 0.75,
    "techcrunch.com": 0.65, "wired.com": 0.70,
    "medium.com": 0.55,    "towardsdatascience.com": 0.65,
}
PENALIZED_DOMAINS = {
    "reddit.com": -0.2, "quora.com": -0.15, "yahoo.com": -0.1,
}
MIN_CREDIBILITY = 0.35

@tool
def score_source_credibility(url: str, has_citations: bool = False,
                              has_author: bool = False, content_length: int = 0) -> float:
    """
    Score a source's credibility from 0.0 to 1.0 based on domain,
    content signals, and metadata.
    """
    domain = urlparse(url).netloc.replace("www.", "")
    base    = TRUSTED_DOMAINS.get(domain, 0.50)
    penalty = PENALIZED_DOMAINS.get(domain, 0.0)
    score   = base + penalty
    if content_length > 1500:
        score += 0.05
    return round(min(max(score, 0.0), 1.0), 2)


def filter_sources(sources: list[dict], min_score: float = MIN_CREDIBILITY) -> list[dict]:
    """Filter out low-credibility sources before expensive analysis."""
    return [s for s in sources if s.get("credibility_score", 0) >= min_score]