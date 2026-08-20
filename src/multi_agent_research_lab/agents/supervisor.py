"""Supervisor / router agent.

Decides which worker agent to call next based on the current shared state,
and enforces the max_iterations guardrail.
"""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

# Sentinel written into route_history to signal end of workflow
DONE = "done"


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop.

    Routing policy (in priority order):
    1. If iteration >= max_iterations  → done  (guardrail)
    2. If sources is empty             → researcher
    3. If analysis_notes is None       → analyst
    4. If final_answer is None         → writer
    5. Otherwise                       → done
    """

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Append next route to state.route_history and return state."""
        settings = get_settings()
        max_iter = settings.max_iterations

        # Guardrail: prevent infinite loops
        if state.iteration >= max_iter:
            logger.warning(
                "Supervisor: max_iterations=%d reached after %d steps — stopping",
                max_iter,
                state.iteration,
            )
            state.record_route(DONE)
            state.add_trace_event(
                "supervisor_route",
                {"next": DONE, "reason": "max_iterations_reached", "iteration": state.iteration},
            )
            if state.errors:
                state.errors.append(f"Stopped at iteration {state.iteration} (max={max_iter})")
            return state

        # Determine next step
        if not state.sources:
            next_route = "researcher"
            reason = "no sources yet"
        elif state.analysis_notes is None:
            next_route = "analyst"
            reason = "sources available but no analysis"
        elif state.final_answer is None:
            next_route = "writer"
            reason = "analysis ready, need final answer"
        else:
            next_route = DONE
            reason = "all stages complete"

        logger.info(
            "Supervisor: iteration=%d → %s (%s)", state.iteration, next_route, reason
        )
        state.record_route(next_route)
        state.add_trace_event(
            "supervisor_route",
            {"next": next_route, "reason": reason, "iteration": state.iteration},
        )
        return state
