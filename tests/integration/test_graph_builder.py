import pytest
from unittest.mock import patch, MagicMock
from langgraph.graph import StateGraph


class TestBuildResearchGraph:
    @pytest.fixture
    def graph(self):
        from graph.builder import build_research_graph
        return build_research_graph()

    def test_graph_compiles_without_error(self, graph):
        assert graph is not None

    def test_graph_has_correct_nodes(self, graph):
        node_names = set(graph.nodes.keys())
        assert "plan_searches" in node_names
        assert "execute_searches" in node_names
        assert "analyze_and_critique" in node_names
        assert "synthesize_report" in node_names

    def test_graph_has_entry_point(self, graph):
        # LangGraph compiled graphs expose the entry via __start__
        assert "__start__" in graph.nodes

    def test_graph_is_callable(self, graph):
        assert callable(graph.invoke) or hasattr(graph, "stream")

    def test_conditional_edge_targets_are_valid_nodes(self, graph):
        # The should_loop_back router must route only to known nodes
        node_names = set(graph.nodes.keys())
        valid_targets = {"plan_searches", "synthesize_report", "__end__"}
        assert valid_targets.issubset(node_names | {"__end__"})


class TestShouldLoopBackRouting:
    """Verify the conditional edge resolves correctly under different states."""

    def _run_to_critique(self, state, mock_llm_response, mock_search_results):
        """Run graph up through analyse_and_critique and return that state."""
        from graph.nodes import _depth_config, should_loop_back
        from models.schemas import ResearchQuery, ResearchDepth
        return should_loop_back(state)

    def test_routes_to_plan_searches_when_gaps_remain(self):
        from models.schemas import ResearchQuery, ResearchDepth
        from graph.nodes import should_loop_back
        state = {
            "query": ResearchQuery(topic="AI", depth=ResearchDepth.MEDIUM),
            "iteration": 1,
            "knowledge_gaps": ["cost data missing", "long-term data absent"],
        }
        assert should_loop_back(state) == "plan_searches"

    def test_routes_to_synthesize_when_no_gaps(self):
        from models.schemas import ResearchQuery, ResearchDepth
        from graph.nodes import should_loop_back
        state = {
            "query": ResearchQuery(topic="AI", depth=ResearchDepth.MEDIUM),
            "iteration": 1,
            "knowledge_gaps": [],
        }
        assert should_loop_back(state) == "synthesize_report"

    def test_routes_to_synthesize_at_max_iteration(self):
        from models.schemas import ResearchQuery, ResearchDepth
        from graph.nodes import should_loop_back
        state = {
            "query": ResearchQuery(topic="AI", depth=ResearchDepth.SHALLOW),
            "iteration": 1,  # max for shallow
            "knowledge_gaps": ["still some gaps"],
        }
        assert should_loop_back(state) == "synthesize_report"
