from typing import TypedDict, Annotated, Optional
from models.schemas import ResearchQuery, Source, Finding, ResearchReport
from langgraph.graph.message import add_messages

class ResearchState(TypedDict):
    # query research
    query:                  ResearchQuery


    # Orchestration
    messages:               Annotated[list, add_messages]
    iteration:              int
    phase:                  str # planning | searching | analyzing | synthesizing | done


    #Accumulated data
    search_queries:         list[str]
    raw_sources:            list[Source]
    validated_sources:      list[Source]
    findings:               list[Finding]
    knowledge_gaps:         list[str]
    critique_notes:         list[str]


    # Output
    report:                 Optional[ResearchReport]
    error:                  Optional[str]