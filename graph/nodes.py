import json
import re

from models.schemas import Finding, ResearchReport, Source
from graph.state import ResearchState
from prompts.researchers import PLANNER_PROMPT, ANALYZER_CRITIQUE_PROMPT

from tools.search import search_and_extract
from tools.scrapper import scrape_url
from tools.validator import score_source_credibility, filter_sources
from tools.vector_store import ingest_store, semantic_search

from utils.checkpoint import save_checkpoint
from utils.rate_limiter import get_llm
from utils.token_budget import (
    MAX_SOURCES_PER_BATCH, NODE_BUDGETS, MAX_FINDINGS_IN_PROMPT, 
    trim, trim_sources, build_sources_block, safe_json_list
)


def _parse_json(text: str) -> dict | list:
    """Strip markdown fences and parse JSON from an LLM response."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    return json.loads(text)


def _depth_config(depth: str) -> dict:
    return {
        "shallow": {"max_queries": 3, "max_sources": 3, "max_iterations": 1},
        "medium":  {"max_queries": 4, "max_sources": 5, "max_iterations": 2},
        "deep":    {"max_queries": 5, "max_sources": 7, "max_iterations": 3},
    }.get(depth, {"max_queries": 4, "max_sources": 5, "max_iterations": 2})


def plan_searches(state: ResearchState) -> ResearchState:
    """Generate strategic search queries for the topic"""

    query = state['query']
    cfg = _depth_config(query.depth)
    n = cfg['max_queries']
    existing = state.get("search_queries", [])
    gaps     = state.get("knowledge_gaps", [])

    prompt = trim(PLANNER_PROMPT.format(
        n=n,
        topic=query.topic,
        focus_areas=", ".join(query.focus_areas) or "general overview",
        existing_queries=json.dumps(existing[-6:] if existing else []),
        knowledge_gaps=json.dumps(gaps) if gaps else []
    ), NODE_BUDGETS['plan_searches'])

    print(f"Planner Prompt: \n\n {prompt}\n\n")

    model = get_llm()
    response = model.invoke(prompt=prompt)

    print(f"Planner Response:\n\n {response}\n\n")
    try:
        queries = _parse_json(response)
        if not isinstance(queries, list):
            raise ValueError("Expected list")
    except Exception:
        queries = [query.topic] # fallback

    all_queries = existing + [q for q in queries if q not in existing]
    
    new_state = {
        **state,
        "phase": "searching",
        "search_queries": all_queries,
        "messages": state.get("messages", []) + [response],
    }

    save_checkpoint(topic=query.topic, completed_phase="searching", state=new_state)
    return new_state


def execute_searches(state: ResearchState) -> ResearchState:
    """Run searches and scrape content, storing to vector DB."""
    query = state['query']
    cfg = _depth_config(query.depth)
    max_src = cfg['max_sources']
    raw_sources = list(state.get("raw_sources", []))
    seen_urls = {s['url'] for s in raw_sources}

    batch_size = _depth_config(query.depth)['max_queries']
    recent_queries = state['search_queries'][-batch_size:]
    
    for query_str in recent_queries:
        if len(raw_sources) >= max_src:
            break

        results = search_and_extract.invoke({"query": query_str})
        for r in results:
            if len(raw_sources) >= max_src:
                break
            if r["url"] in seen_urls:
                continue
            seen_urls.add(r["url"])

            # Scrape full content if raw content is short
            content = r.get('content', "")
            if len(content) < 400:
                scraped = scrape_url.invoke({"url": r["url"]})
                if scraped['success']:
                    content = scraped['content']

            if not content:
                continue

            # Score credibility
            cred_score = score_source_credibility.invoke({
                "url": r["url"],
                "content_length": len(content)
            })

            if cred_score < 0.35:
                continue

            # Ingest into vector store
            chunk_ids = ingest_store(url=r["url"], title=r.get("title", ""), content=content)

            raw_sources.append({
                "url": r["url"],
                "title": r.get("title", "Untitled"),
                "content": content,
                "credibility_score": cred_score,
                "relevance_score": r.get("score", 0.5),
                "chunk_ids": chunk_ids
            })

    new_state = {**state, "phase": "analyzing", "raw_sources": raw_sources}

    save_checkpoint(query.topic, "analyzing", new_state)
    return new_state


def analyse_and_critique(state: ResearchState) -> ResearchState:
    """
    Analyze each source to extract structured findings.
    Critical review of findings to identify gaps and issues.
    """
    query = state['query']
    sources = filter_sources(state.get("raw_sources", []))

    trimmed = trim_sources(sources=sources, max_sources=MAX_SOURCES_PER_BATCH)
    
    kb_docs    = semantic_search(query.topic, k=3)
    kb_context = "\n".join(d.page_content for d in kb_docs)[:600]

    source_block = build_sources_block(sources=trimmed)

    prior_findings = safe_json_list(state.get("findings", []), MAX_FINDINGS_IN_PROMPT)

    prompt = trim(
        ANALYZER_CRITIQUE_PROMPT.format(
            topic=query.topic,
            focus_areas=', '.join(query.focus_areas) or 'general',
            kb_context=kb_context,
            sources_block=source_block,
            findings=json.dumps(prior_findings),
        ), NODE_BUDGETS['analyze_and_critique']
    )

    model = get_llm()
    response = model.invoke(prompt=prompt)

    try:
        data = _parse_json(response)
        new_findings = data.get("findings", [])
        gaps         = data.get("knowledge_gaps", [])
        critique     = data.get("critique_notes", [])
    except Exception as e:
        print(f"[analyze_and_critique] JSON parse failed: {e}")
        new_findings = []
        gaps         = []
        critique     = []

    # Merge with existing findings (dedup by claim)
    existing_claims = {f["claim"] for f in state.get("findings", [])}
    merged_findings = list(state.get("findings", []))
    for f in new_findings:
        if f.get("claim") not in existing_claims:
            merged_findings.append(f)
            existing_claims.add(f.get("claim"))

    new_state = {
        **state,
        "phase":             "synthesizing",
        "findings":          merged_findings,
        "knowledge_gaps":    gaps,
        "critique_notes":    critique,
        "validated_sources": filter_sources(state["raw_sources"]),
        "iteration":         state.get("iteration", 0) + 1,
    }

    save_checkpoint(topic=query.topic, completed_phase="synthesizing", state=new_state)
    return new_state

    
def synthesize_report(state: ResearchState) -> ResearchState:
    """Generate the final structured research report."""

    query    = state["query"]
    findings = safe_json_list(state.get("findings", []), MAX_FINDINGS_IN_PROMPT)
    gaps = state.get("knowledge_gaps", [])
    critique = state.get("critique_notes", [])
    sources = state.get("validated_sources", state.get("raw_sources", []))

    source_urls = [s["url"] for s in sources[:8]]

    prompt = trim(
        f"""You are a senior research analyst. Synthesise these research findings into
a structured report.

TOPIC: {query.topic}

KEY FINDINGS:
{json.dumps(findings, indent=2)}

KNOWLEDGE GAPS: {json.dumps(gaps)}
CRITIQUE NOTES: {json.dumps(critique)}
SOURCES USED: {json.dumps(source_urls)}

Write a comprehensive research report. Return ONLY valid JSON:
{{
  "topic": "{query.topic}",
  "summary": "3-4 paragraph executive summary",
  "key_findings": [
    {{
      "claim": "consolidated finding",
      "evidence": ["supporting evidence"],
      "source_urls": ["url"],
      "confidence": 0.0-1.0,
      "contradictions": []
    }}
  ],
  "gaps_identified": ["gap 1", "gap 2"],
  "follow_up_queries": ["query 1", "query 2", "query 3"],
  "confidence_overall": 0.0-1.0
}}""",
        NODE_BUDGETS["synthesize_report"],
    )
    
    model = get_llm()
    response = model.invoke(prompt=prompt)
    
    report = None
    try:
        data = _parse_json(response)
        data["sources"]    = [Source(**s) for s in sources[:10]]
        data["word_count"] = len(data.get("summary", "").split())
        report = ResearchReport(**data)
    except Exception as e:
        print(f"[synthesize_report] Failed to parse report: {e}")
        # Minimal fallback report
        fallback_summary = (
            f"Research on '{query.topic}' completed with "
            f"{len(state.get('findings', []))} findings."
        )
        report = ResearchReport(
            topic=query.topic,
            summary=fallback_summary,
            key_findings=[
                Finding(**f) for f in state.get("findings", [])[:5]
            ],
            sources=[Source(**s) for s in sources[:10]],
            gaps_identified=gaps,
            follow_up_queries=[],
            word_count=len(fallback_summary.split()),
            confidence_overall=0.4,
        )

    new_state = {**state, "phase": "done", "report": report}
    
    save_checkpoint(topic=query.topic,completed_phase="done",state=new_state)
    return new_state


def should_loop_back(state: ResearchState) -> str:
    """After critique, decide whether to search more or synthesize."""
    cfg       = _depth_config(state["query"].depth)
    max_iter  = cfg["max_iterations"]
    iteration = state.get("iteration", 0)
    gaps      = state.get("knowledge_gaps", [])

    if iteration < max_iter and len(gaps) > 0:
        print(f"[Router] Looping back — iteration {iteration}/{max_iter}, "
              f"{len(gaps)} gaps found.")
        return "plan_searches"

    print(f"[Router] Moving to synthesis — iteration {iteration}/{max_iter}.")
    return "synthesize_report"