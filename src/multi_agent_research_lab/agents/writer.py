"""Writer agent.

Synthesises all prior agent outputs into a polished final answer with
proper citations back to the retrieved sources.
"""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert Technical Writer. Using the research notes and analysis notes provided,
write a comprehensive, well-structured answer to the user's query.

Requirements:
- Start with a clear executive summary (2-3 sentences)
- Organise the body with headings and bullet points where helpful
- Cite sources using numbered references like [1], [2], etc. that map to the source list
- End with a brief "Key Takeaways" section (3-5 bullets)
- Tailor depth and terminology to the audience: {audience}
- Be factual — do not add information not supported by the research/analysis notes
- Total length: 400-700 words"""

_USER_TEMPLATE = """\
Query: {query}

Research Notes:
{research_notes}

Analysis Notes:
{analysis_notes}

Sources (for citation):
{sources_block}

Write the final answer now."""


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer` with citations."""
        query = state.request.query
        audience = state.request.audience
        research_notes = state.research_notes or "(No research notes)"
        analysis_notes = state.analysis_notes or "(No analysis notes)"

        sources_block = "\n".join(
            f"[{i + 1}] {s.title} — {s.url or 'no URL'}\n    {s.snippet[:150]}"
            for i, s in enumerate(state.sources)
        ) or "(No sources available)"

        logger.info("Writer: synthesising final answer for '%s'", query)

        try:
            response = self._llm.complete(
                system_prompt=_SYSTEM_PROMPT.format(audience=audience),
                user_prompt=_USER_TEMPLATE.format(
                    query=query,
                    research_notes=research_notes,
                    analysis_notes=analysis_notes,
                    sources_block=sources_block,
                ),
            )
            state.final_answer = response.content
            metadata: dict = {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
                "source_count": len(state.sources),
            }
        except Exception as exc:
            logger.error("Writer: LLM call failed: %s", exc)
            state.errors.append(f"WriterAgent error: {exc}")
            state.final_answer = (
                f"Error generating final answer: {exc}\n\n"
                f"Research notes summary:\n{research_notes[:500]}"
            )
            metadata = {"error": str(exc)}

        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=state.final_answer or "",
                metadata=metadata,
            )
        )
        state.add_trace_event(
            "writer_done",
            {"answer_length": len(state.final_answer or "")},
        )
        logger.info(
            "Writer: produced %d chars final answer", len(state.final_answer or "")
        )
        return state
