"""Unit tests for SupervisorAgent routing policy.

These tests validate the routing logic WITHOUT calling any real LLM or search API.
"""

import os

from multi_agent_research_lab.agents.supervisor import DONE, SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState

_QUERY = ResearchQuery(query="Explain multi-agent systems in detail")


def _make_state(**kwargs) -> ResearchState:
    return ResearchState(request=_QUERY, **kwargs)


def test_supervisor_routes_to_researcher_when_no_sources() -> None:
    """With empty sources, supervisor should dispatch to researcher."""
    state = _make_state()
    agent = SupervisorAgent()
    result = agent.run(state)
    assert result.route_history[-1] == "researcher"
    assert result.iteration == 1


def test_supervisor_routes_to_analyst_when_sources_but_no_analysis() -> None:
    """With sources but no analysis notes, supervisor → analyst."""
    sources = [SourceDocument(title="Doc 1", snippet="Some text", url=None)]
    state = _make_state(sources=sources, research_notes="Some notes")
    agent = SupervisorAgent()
    result = agent.run(state)
    assert result.route_history[-1] == "analyst"


def test_supervisor_routes_to_writer_when_analysis_ready() -> None:
    """With sources + analysis but no answer, supervisor → writer."""
    sources = [SourceDocument(title="Doc 1", snippet="Some text", url=None)]
    state = _make_state(
        sources=sources,
        research_notes="Notes",
        analysis_notes="Analysis",
    )
    agent = SupervisorAgent()
    result = agent.run(state)
    assert result.route_history[-1] == "writer"


def test_supervisor_routes_to_done_when_complete() -> None:
    """With all fields populated, supervisor → done."""
    sources = [SourceDocument(title="Doc 1", snippet="Some text", url=None)]
    state = _make_state(
        sources=sources,
        research_notes="Notes",
        analysis_notes="Analysis",
        final_answer="Final answer here",
    )
    agent = SupervisorAgent()
    result = agent.run(state)
    assert result.route_history[-1] == DONE


def test_supervisor_enforces_max_iterations() -> None:
    """Supervisor stops at max_iterations even if fields are missing."""
    from multi_agent_research_lab.core.config import get_settings

    os.environ["MAX_ITERATIONS"] = "2"
    get_settings.cache_clear()

    try:
        # iteration=2 == max_iterations=2 → should stop
        state = ResearchState(request=_QUERY, iteration=2)
        agent = SupervisorAgent()
        result = agent.run(state)
        assert result.route_history[-1] == DONE
    finally:
        del os.environ["MAX_ITERATIONS"]
        get_settings.cache_clear()


def test_supervisor_records_trace_event() -> None:
    """Supervisor should add a trace event on each call."""
    state = _make_state()
    agent = SupervisorAgent()
    result = agent.run(state)
    assert any(e["name"] == "supervisor_route" for e in result.trace)
