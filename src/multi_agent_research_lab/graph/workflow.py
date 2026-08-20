"""LangGraph workflow for the multi-agent research system.

Nodes: supervisor → researcher / analyst / writer (conditional routing)
Stop condition: supervisor routes to "done" or max_iterations is reached.
"""

from __future__ import annotations

import logging
from typing import Any

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import DONE, SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node functions (LangGraph expects plain callables that accept / return dicts)
# ---------------------------------------------------------------------------

def _make_node(agent: Any):  # type: ignore[return]
    """Wrap a BaseAgent so LangGraph can call it with a plain dict state."""

    def node_fn(state_dict: dict[str, Any]) -> dict[str, Any]:
        state = ResearchState.model_validate(state_dict)
        with trace_span(
            f"agent.{agent.name}",
            {
                "query": state.request.query,
                "iteration": state.iteration,
                "source_count": len(state.sources),
            },
        ) as span:
            updated = agent.run(state)
            span["outputs"] = {
                "iteration": updated.iteration,
                "source_count": len(updated.sources),
                "route": updated.route_history[-1] if updated.route_history else None,
            }
        return updated.model_dump()

    node_fn.__name__ = agent.name
    return node_fn


def _route(state_dict: dict[str, Any]) -> str:
    """Conditional edge: read the last entry of route_history."""
    history: list[str] = state_dict.get("route_history", [])
    if not history:
        return DONE
    last = history[-1]
    if last in ("researcher", "analyst", "writer"):
        return last
    return DONE


class MultiAgentWorkflow:
    """Builds and runs the multi-agent LangGraph graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def build(self) -> Any:
        """Create a LangGraph StateGraph and return the compiled graph."""
        try:
            from langgraph.graph import END, StateGraph  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "langgraph not installed. Run: pip install -e '.[llm]'"
            ) from exc

        supervisor = SupervisorAgent()
        researcher = ResearcherAgent()
        analyst = AnalystAgent()
        writer = WriterAgent()

        # LangGraph uses TypedDict or plain dict as state; we serialise to/from ResearchState
        graph = StateGraph(dict)

        # Add nodes
        graph.add_node("supervisor", _make_node(supervisor))
        graph.add_node("researcher", _make_node(researcher))
        graph.add_node("analyst", _make_node(analyst))
        graph.add_node("writer", _make_node(writer))

        # Entry point
        graph.set_entry_point("supervisor")

        # Conditional routing from supervisor
        graph.add_conditional_edges(
            "supervisor",
            _route,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                DONE: END,
            },
        )

        # After each worker, return to supervisor
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")

        compiled = graph.compile()
        logger.info("MultiAgentWorkflow: graph compiled successfully")
        return compiled

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""
        graph = self.build()
        initial_dict = state.model_dump()

        logger.info(
            "MultiAgentWorkflow: starting run for query='%s'", state.request.query
        )
        with trace_span("workflow.multi_agent", {"query": state.request.query}) as span:
            result_dict = graph.invoke(initial_dict)
            span["outputs"] = {
                "route_history": result_dict.get("route_history", []),
                "has_final_answer": bool(result_dict.get("final_answer")),
            }

        final_state = ResearchState.model_validate(result_dict)
        logger.info(
            "MultiAgentWorkflow: finished | route_history=%s | errors=%s",
            final_state.route_history,
            final_state.errors,
        )
        return final_state
