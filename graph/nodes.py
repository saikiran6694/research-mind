import json
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage
from graph.state import ResearchState
from prompts.researchers import PLANNER_PROMPT, ANALYZER_PROMPT, CRITIQUE_PROMPT
from tools.search import search_and_extract
from tools.scrapper import scrape_url
from tools.validator import score_source_credibility
from tools.vector_store import ingest_store, semantic_search
from models.schemas import Source, ResearchReport
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
model_structured = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def _parse_json_response(response):
    """Strip markdown code fences (if any) and parse the LLM response content as JSON."""
    content = response.content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[len("json"):]
        content = content.strip()
    return json.loads(content)


def plan_searches(state: ResearchState) -> ResearchState:
    """Generate strategic search queries for the topic"""

    query = state['query']
    prompt = PLANNER_PROMPT.format(
        topic=query.topic,
        focus_areas=", ".join(query.focus_areas) or "general",
        existing_queries=state.get("search_queries", []),
        knowledge_gaps=state.get("knowledge_gaps", [])
    )

    response = model.invoke([HumanMessage(content=prompt)])
    try:
        queries = _parse_json_response(response)
    except Exception:
        queries = [query.topic] # fallback

    return {
        **state,
        "phase": "searching",
        "search_queries": state.get('search_queries', []) + queries,
        "messages": state.get("messages", []) + [response],
    }


def execute_searches(state: ResearchState) -> ResearchState:
    """Run searches and scrape content, storing to vector DB."""
    new_queries = state['search_queries'][-5:]
    raw_sources = list(state.get("raw_sources", []))
    seen_urls = {s['url'] for s in raw_sources}

    for query_str in new_queries:
        results = search_and_extract.invoke({"query": query_str})
        for r in results:
            if r["url"] in seen_urls:
                continue
            seen_urls.add(r["url"])

            # Scrape full content if raw content is short
            content = r.get('content', "")
            if len(content) < 500:
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

            # Ingest into vector store
            chunk_ids = ingest_store(url=r["url"], title=r.get("title", ""), content=content)

            raw_sources.append({
                "url": r["url"],
                "title": r.get("title", ""),
                "content": content,
                "credibility_score": cred_score,
                "relevance_score": r.get("score", 0.5),
                "chunk_ids": chunk_ids
            })

    return {**state, "phase": "analyzing", "raw_sources": raw_sources}


def analyse_sources(state: ResearchState) -> ResearchState:
    """Analyze each source to extract structured findings."""
    findings = list(state.get('findings', []))
    validated_sources: list[Source] = list(state.get('validated_sources', []))
    analyzed_urls = {s['url'] for s in validated_sources}

    for source in state.get("raw_sources", []):
        if source['url'] in analyzed_urls:
            continue

        if source['credibility_score'] < 0.3:
            continue

        # Pull relevant context from the vector store, excluding this source's own chunks
        kb_docs = semantic_search(query=state['query'].topic, k=8)
        kb_docs = [d for d in kb_docs if d.metadata.get("url") != source['url']][:4]
        kb_context = "\n\n".join([d.page_content.strip() for d in kb_docs])

        prompt = ANALYZER_PROMPT.format(
            topic=state['query'].topic,
            url=source['url'],
            title=source['title'],
            content=source['content'][:3000],
            kb_context=kb_context[:1500]
        )

        response = model.invoke([HumanMessage(content=prompt)])
        try:
            analysis = _parse_json_response(response)
        except Exception:
            continue

        for f in analysis.get('findings', []):
            findings.append({
                "claim": f["claim"],
                "evidence": f.get("evidence", []),
                "source_urls": [source["url"]],
                "confidence": f.get('confidence', 0.5),
                "contradictions": f.get("contradictions", [])
            })

        source['relevance_score'] = analysis.get("relevance_score", source['relevance_score'])
        validated_sources.append(source)

    return { 
        **state, 
        "phase": "critiquing", 
        "findings": findings, 
        "validated_sources": validated_sources 
    }


def critique_findings(state: ResearchState) -> ResearchState:
    """Critical review of findings to identify gaps and issues."""
    findings = json.dumps(state['findings'][:20], indent=2)
    source_urls = [s['url'] for s in state['validated_sources']]

    prompt = CRITIQUE_PROMPT.format(
        findings=findings,
        source_urls=source_urls
    )

    response = model.invoke([HumanMessage(content=prompt)])
    try:
        critique = _parse_json_response(response)
    except Exception:
        critique = {"issues": [], "missing_perspectives": [], "knowledge_gaps": [], "overall_quality": 0.6}

    gaps = critique.get("knowledge_gaps", []) + critique.get("missing_perspectives", [])

    return {
        **state,
        "phase": "synthesizing",
        "knowledge_gaps": gaps,
        "critique_notes": critique.get("issues", []),
        "iteration": state.get("iteration", 0) + 1,
        "messages": state["messages"] + [response]
    }
    
def synthesize_report(state: ResearchState) -> ResearchState:
    """Generate the final structured research report."""
    context = {
        "topic": state["query"].topic,
        "findings_count": len(state["findings"]),
        "sources_count": len(state["validated_sources"]),
        "findings": state["findings"][:25],
        "gaps": state["knowledge_gaps"],
        "critique": state["critique_notes"]
    }

    prompt = f"""You are a senior research analyst. Synthesize the following research into a 
comprehensive, well-structured report.

Research Context:
{json.dumps(context, indent=2)}

Produce a report with:
1. Executive Summary (2-3 paragraphs)
2. Key Findings (consolidated, de-duplicated, with confidence levels)
3. Areas of Consensus vs Controversy
4. Knowledge Gaps & Limitations
5. Recommended Follow-up Queries

Return as JSON matching ResearchReport schema:
{{
  "topic": str,
  "summary": str,
  "key_findings": [{{"claim": str, "evidence": [str], "source_urls": [str], "confidence": float, "contradictions": [str]}}],
  "gaps_identified": [str],
  "follow_up_queries": [str],
  "confidence_overall": float
}}"""
    
    response = model.invoke([HumanMessage(content=prompt)])
    try:
        report_data = _parse_json_response(response)
        report_data["sources"] = state["validated_sources"]
        report_data["word_count"] = len(report_data["summary"].split())
        report = ResearchReport(**report_data)
    except Exception as e:
        report = None

    return {**state, "phase": "done", "report": report}


def should_loop_back(state: ResearchState) -> str:
    """After critique, decide whether to search more or synthesize."""
    depth = state['query'].depth
    iteration = state.get("iteration", 0)
    gaps = state.get("knowledge_gaps", [])

    max_iterations = {"shallow": 1, "medium": 2, "deep": 3}[depth]

    if iteration < max_iterations and len(gaps) > 0:
        return "plan_searches" # loop back to fill gaps
    return "synthesize_report"