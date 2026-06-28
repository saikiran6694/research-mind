NODE_BUDGETS = {
    "plan_searches":        2_000,
    "analyze_and_critique": 3_500,   # largest — batches multiple sources
    "synthesize_report":    3_000,
    "fallback":             2_000,
}

SOURCE_CONTENT_CHARS = 700   # ~175 tokens per source
MAX_SOURCES_PER_BATCH = 5    # never send more than 5 sources at once
MAX_FINDINGS_IN_PROMPT = 10  # cap findings list in synthesis prompt


def trim(text: str, max_chars: int) -> str:
    """Hard-cap a string. Appends a marker so LLM knows it was cut."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[trimmed for token budget]"

def trim_sources(sources: list[dict],
                 max_sources: int = MAX_SOURCES_PER_BATCH,
                 content_chars: int = SOURCE_CONTENT_CHARS) -> list[dict]:
    """
    Take at most `max_sources` sources and trim each content field.
    Returns a new list — does not mutate originals.
    """
    trimmed = []
    for s in sources[:max_sources]:
        trimmed.append({
            **s,
            "content": trim(s.get("content", ""), content_chars)
        })
    return trimmed

def build_sources_block(sources: list[dict]) -> str:
    """Format trimmed sources into a single prompt block."""
    parts = []
    for i, s in enumerate(sources):
        parts.append(
            f"--- SOURCE {i+1}: {s.get('title', 'Untitled')} ---\n"
            f"URL: {s.get('url', '')}\n"
            f"{s.get('content', '')}"
        )
    return "\n\n".join(parts)


def safe_json_list(data: list, max_items: int) -> list:
    """Cap a list before embedding in a prompt."""
    return data[:max_items]