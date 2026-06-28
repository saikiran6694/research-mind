from langgraph.graph import StateGraph, END
from graph.state import ResearchState
from graph.nodes import (
    plan_searches, 
    execute_searches, 
    analyse_and_critique,
    synthesize_report, 
    should_loop_back
)

def build_research_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("plan_searches", plan_searches)
    graph.add_node("execute_searches", execute_searches)
    graph.add_node("analyze_and_critique", analyse_and_critique)
    graph.add_node("synthesize_report", synthesize_report)

    # Linear flow
    graph.set_entry_point("plan_searches")
    graph.add_edge("plan_searches", "execute_searches")
    graph.add_edge("execute_searches", "analyze_and_critique")

    # Conditional loop: re-search if gaps found, else synthesize
    graph.add_conditional_edges(
        "analyze_and_critique",
        should_loop_back,
        {
            "plan_searches": "plan_searches",
            "synthesize_report": "synthesize_report"
        }
    )

    graph.add_edge("synthesize_report", END)

    return graph.compile()