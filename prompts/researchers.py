PLANNER_PROMPT = """You are a research planning expert.

Given a research topic and focus areas, generate a strategic set of search queries 
that will comprehensively cover the topic from multiple areas.

Topic: {topic}
Focus areas: {focus_areas}
Existing queries already run: {existing_queries}
Knowledge gap identified so far: {knowledge_gaps}

Generate 3-5 percise , diverse search queries that will fill gaps and deepen coverage.
Return STRICTLY ONLY a JSON list of query strings. No explaination."""


ANALYZER_PROMPT = """You are a rigorous research analyst.

Analyse the following source content and extract key findings relevant to the research topic.

Topic: {topic}
Source URL: {url}
Source Title: {title}
Content: 
{content}

Previously retireved context from the knowledge base
{kb_context}

Extract:
1. Key claims with direct evidence from the text.
2. Any contradictions with prior findings (if any).
3. Confidence level (0.0-1.0) for each claim.
4. Relevance Score (0.0-1.0) for this source.

Return as JSON matching this schema:
{{
  "findings": [{{"claim": str, "evidence": [str], "confidence": float, "contradictions": [str]}}],
  "relevance_score": float,
  "key_quotes": [str]
}}"""


CRITIQUE_PROMPT = """You are a critical research reviewer.

Review the following set of findings and identify:
1. Logical inconsistency or contradictions between sources.
2. Claims that needs stronger evidence.
3. Important prespective or viewpoints that are missing.
4. Potential biases in the sources

Findings so far:
{findings}

Sources used:
{source_urls}

Return as JSON
{{
  "issues": [str],
  "missing_perspectives": [str],
  "knowledge_gaps": [str],
  "overall_quality": float
}}"""