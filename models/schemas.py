from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ResearchDepth(str, Enum):
    SHALLOW         = "shallow"
    MEDIUM          = "medium"
    DEPTH           = "deep"


class ResearchQuery(BaseModel):
    topic:              str
    depth:              ResearchDepth = ResearchDepth.MEDIUM
    focus_areas:        list[str] = []
    exclude_domains:    list[str] = []

class Finding(BaseModel):
    claim:              str
    evidence:           list[str]
    source_urls:        list[str]
    confidence:         float = Field(ge=0.0, le=1.0)
    contradictions:     list[str] = []

class Source(BaseModel):
    url:                str
    title:              str
    content:            str
    credibility_score:  float = Field(ge=0.0, le=1.0)
    relevance_score:    float = Field(ge=0.0, le=1.0)
    chunk_ids:          list[str] = []

class ResearchReport(BaseModel):
    topic:              str
    summary:            str
    key_findings:       list[Finding]
    sources:            list[Source]
    gaps_identified:    list[str]
    follow_up_queries:  list[str]
    word_count:         int
    confidence_overall: float
