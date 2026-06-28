import os
from langchain.tools import tool

try:
    from tavily import TavilyClient
    _client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))
except Exception:
    _client = None


def _safe_client():
    if _client is None:
        raise RuntimeError("TavilyClient not initialised. Set TAVILY_API_KEY.")
    return _client


@tool
def web_search(query: str, max_results: int = 3) -> list[dict]:
    """Search the web. Returns list of {url, title, snippet}."""
    
    max_results = min(max_results, 3)
    try:
        results = _safe_client().search(
            query=query,
            max_results=max_results,
            search_depth="basic",   
            include_raw_content=False,
        )
        return [
            {"url": r["url"], "title": r["title"], "snippet": r.get("content", "")}
            for r in results.get("results", [])
        ]
    except Exception as e:
        print(f"[search] web_search error: {e}")
        return []
      
@tool
def search_and_extract(query: str) -> list[dict]:
    """
    Search and return raw page content where available.
    Fix 5: capped at 3 results to limit downstream scraping.
    """
    try:
        results = _safe_client().search(
            query=query,
            max_results=3,
            search_depth="advanced",
            include_raw_content=True,
        )
        return [
            {
                "url":     r["url"],
                "title":   r["title"],
                "content": (r.get("raw_content") or r.get("content", ""))[:4000],
                "score":   r.get("score", 0.5),
            }
            for r in results.get("results", [])
        ]
    except Exception as e:
        print(f"[search] search_and_extract error: {e}")
        return []

