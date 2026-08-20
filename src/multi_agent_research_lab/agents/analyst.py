"""Analyst agent.

Reads research notes from shared state and produces structured analysis notes:
key claims, source reliability, conflicting evidence, and knowledge gaps.
"""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a Critical Analyst. Given research notes and retrieved sources, produce structured
analysis notes that:
1. List the KEY CLAIMS (3-7 bullet points) supported by the evidence
2. Assess SOURCE RELIABILITY (credible, mixed, or weak for each main source)
3. Identify CONFLICTING EVIDENCE or contradictions between sources
4. Flag KNOWLEDGE GAPS — important questions the sources do not answer
5. Rate overall EVIDENCE STRENGTH on a 1-10 scale with brief justification

Be objective, concise, and do NOT write the final answer yet."""

_USER_TEMPLATE = """\
Original Query: {query}

Research Notes:
{research_notes}

Sources Referenced:
{sources_block}

Produce structured analysis notes following the five sections above."""


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        query = state.request.query
        research_notes = state.research_notes or "(No research notes available)"

        sources_block = "\n".join(
            f"- [{i + 1}] {s.title} — {s.url or 'no URL'}"
            for i, s in enumerate(state.sources)
        ) or "(No sources)"

        logger.info("Analyst: analysing research notes (%d chars)", len(research_notes))

        try:
            response = self._llm.complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=_USER_TEMPLATE.format(
                    query=query,
                    research_notes=research_notes,
                    sources_block=sources_block,
                ),
            )
            state.analysis_notes = response.content
            metadata: dict = {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            }
        except Exception as exc:
            logger.error("Analyst: LLM call failed: %s", exc)
            state.errors.append(f"AnalystAgent error: {exc}")
            state.analysis_notes = f"Analysis could not be generated: {exc}"
            metadata = {"error": str(exc)}

        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=state.analysis_notes or "",
                metadata=metadata,
            )
        )
        state.add_trace_event(
            "analyst_done",
            {"analysis_length": len(state.analysis_notes or "")},
        )
        logger.info(
            "Analyst: produced %d chars of analysis notes",
            len(state.analysis_notes or ""),
        )
        return state
