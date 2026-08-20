"""Optional critic agent for fact-checking and citation coverage review."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a Fact-Checker and Quality Reviewer. Given a final answer and the sources used to
produce it, evaluate:
1. CITATION ACCURACY: are all [n] references present in the source list?
2. FACTUAL CONSISTENCY: are claims in the answer supported by the research notes?
3. COMPLETENESS: does the answer address the full query?
4. HALLUCINATION FLAGS: list any statements that appear unsupported or fabricated

Output a brief review report (bullet points), then a QUALITY SCORE from 0 to 10."""

_USER_TEMPLATE = """\
Query: {query}
Final Answer:
{final_answer}

Sources:
{sources_block}

Research Notes (for cross-check):
{research_notes}

Produce your quality review now."""


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings to state.agent_results."""
        query = state.request.query
        final_answer = state.final_answer or "(No answer generated)"
        research_notes = state.research_notes or "(No notes)"
        sources_block = "\n".join(
            f"[{i + 1}] {s.title}" for i, s in enumerate(state.sources)
        ) or "(No sources)"

        logger.info("Critic: reviewing final answer quality")

        try:
            response = self._llm.complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=_USER_TEMPLATE.format(
                    query=query,
                    final_answer=final_answer,
                    sources_block=sources_block,
                    research_notes=research_notes,
                ),
            )
            critic_output = response.content
            metadata: dict = {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            }
        except Exception as exc:
            logger.warning("Critic: review failed: %s", exc)
            critic_output = f"Critic review failed: {exc}"
            metadata = {"error": str(exc)}

        state.agent_results.append(
            AgentResult(agent=AgentName.CRITIC, content=critic_output, metadata=metadata)
        )
        state.add_trace_event("critic_done", {"review_length": len(critic_output)})
        return state
