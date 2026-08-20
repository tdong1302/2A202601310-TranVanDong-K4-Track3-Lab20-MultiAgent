"""Researcher agent.

Searches for relevant sources and synthesises concise research notes.
"""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a Research Specialist. Given a research query and a set of retrieved source snippets,
produce concise, factual research notes that:
- Summarise the key facts from the sources
- Note the quality and relevance of each source briefly
- Flag any obvious gaps or contradictions in the evidence
- Are written in clear, neutral prose (3-5 paragraphs)
Do NOT write a final answer; only produce structured notes for the Analyst."""

_USER_TEMPLATE = """\
Research Query: {query}

Retrieved Sources:
{sources_block}

Write comprehensive research notes based on the sources above."""


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self) -> None:
        self._search = SearchClient()
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        query = state.request.query
        max_sources = state.request.max_sources

        logger.info("Researcher: searching for '%s' (max=%d)", query, max_sources)

        # 1. Search
        try:
            sources = self._search.search(query, max_results=max_sources)
        except Exception as exc:
            logger.error("Researcher: search failed: %s", exc)
            state.errors.append(f"ResearcherAgent search error: {exc}")
            sources = []

        state.sources = sources

        # 2. Synthesise notes via LLM
        if sources:
            sources_block = "\n\n".join(
                f"[Source {i + 1}] {s.title}\n"
                f"URL: {s.url or 'N/A'}\n"
                f"Snippet: {s.snippet}"
                for i, s in enumerate(sources)
            )
        else:
            sources_block = "(No external sources retrieved — rely on general knowledge)"

        try:
            response = self._llm.complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=_USER_TEMPLATE.format(
                    query=query, sources_block=sources_block
                ),
            )
            state.research_notes = response.content
            metadata: dict = {
                "source_count": len(sources),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            }
        except Exception as exc:
            logger.error("Researcher: LLM call failed: %s", exc)
            state.errors.append(f"ResearcherAgent LLM error: {exc}")
            state.research_notes = f"Research notes could not be generated: {exc}"
            metadata = {"error": str(exc)}

        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=state.research_notes or "",
                metadata=metadata,
            )
        )
        state.add_trace_event(
            "researcher_done",
            {"source_count": len(sources), "notes_length": len(state.research_notes or "")},
        )
        logger.info(
            "Researcher: found %d sources, notes=%d chars",
            len(sources),
            len(state.research_notes or ""),
        )
        return state
