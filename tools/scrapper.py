import httpx
from bs4 import BeautifulSoup
from langchain.tools import tool

@tool
def scrape_url(url: str) -> dict:
    """
    Scrape the full text content from a URL.
    Returns {url, title, content, success}.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (ResearchBot/1.0)"}
        resp = httpx.get(url, headers=headers, timeout=10, follow_redirects=True)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "aside", "ads"]):
            tag.decompose()

        title = soup.find("title")
        title = title.get_text(strip=True) if title else url

        # Prefer article/main content
        main = soup.find("article") or soup.find("main") or soup.find("body")
        content = main.get_text(separator="\n", strip=True) if main else ""

        # Truncate to ~6000 chars to stay within context
        return {"url": url, "title": title, "content": content[:6000], "success": True}
    except Exception as e:
        return {"url": url, "title": "", "content": "", "success": False, "error": str(e)}