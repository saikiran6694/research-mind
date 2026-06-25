from langchain.tools import tool
from tavily import TavilyClient
import os

client = TavilyClient(api_key="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")

@tool
def web_search(query: str, max_results: int = 2) -> list[dict]:
    """Search the web for a query. Returns list of {url, title, snippet}."""
    results = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
        include_raw_content=False
    )

    return [
        {
            "url": r['url'], 
            "title": r['title'], 
            "snippet": r.get("content", "") 
        }
        for r in results.get("results", [])
    ]


@tool
def search_and_extract(query: str) -> list[dict]:
    """Search the web and return full page content where available."""
    results = client.search(
        query=query,
        max_results=2,
        search_depth="advanced",
        include_raw_content=True
    )

    return [
        {
            "url": r["url"],
            "title": r["title"],
            "content": r.get("raw_content") or r.get("content", ""),
            "score": r.get("score", 0.5)
        }
        for r in results.get("results", [])
    ]