"""Search client abstraction for ResearcherAgent.

Falls back to the offline corpus if TAVILY_API_KEY is not set.
"""

import json
import logging
import re
from pathlib import Path

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)

_CORPUS_DIR = (
    Path(__file__).resolve().parents[3]
    / "ai_agent_offline_research_corpus_v2"
    / "topics"
)


class SearchClient:
    """Provider-agnostic search client.

    Uses Tavily when TAVILY_API_KEY is configured; falls back to the offline
    research corpus bundled with the repo otherwise.
    """

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to *query*."""
        settings = get_settings()
        if settings.tavily_api_key:
            return self._tavily_search(query, max_results, settings.tavily_api_key)
        logger.info("No TAVILY_API_KEY — using offline corpus mock search")
        return self._offline_search(query, max_results)

    # ------------------------------------------------------------------
    # Tavily implementation
    # ------------------------------------------------------------------

    def _tavily_search(
        self, query: str, max_results: int, api_key: str
    ) -> list[SourceDocument]:
        try:
            from tavily import TavilyClient  # type: ignore[import-untyped]

            client = TavilyClient(api_key=api_key)
            response = client.search(query, max_results=max_results)
            results = response.get("results", [])
            return [
                SourceDocument(
                    title=r.get("title", "Untitled"),
                    url=r.get("url"),
                    snippet=r.get("content", ""),
                    metadata={"score": r.get("score", 0.0)},
                )
                for r in results[:max_results]
            ]
        except Exception as exc:
            logger.warning("Tavily search failed (%s); falling back to offline", exc)
            return self._offline_search(query, max_results)

    # ------------------------------------------------------------------
    # Offline corpus implementation
    # ------------------------------------------------------------------

    def _offline_search(self, query: str, max_results: int) -> list[SourceDocument]:
        """Simple keyword-matching search over the bundled JSON corpus."""
        query_lower = query.lower()
        keywords = set(re.findall(r"\w+", query_lower))
        # Remove stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "of",
                      "in", "to", "for", "with", "on", "at", "by", "from", "that", "this",
                      "it", "be", "as", "what", "how", "when", "why", "which", "who"}
        keywords -= stop_words

        candidates: list[tuple[float, SourceDocument]] = []

        if not _CORPUS_DIR.exists():
            logger.warning("Offline corpus not found at %s", _CORPUS_DIR)
            return self._hardcoded_fallback(query, max_results)

        for topic_file in sorted(_CORPUS_DIR.glob("*.json"))[:10]:
            try:
                data = json.loads(topic_file.read_text(encoding="utf-8"))
            except Exception:
                continue

            topic_name = data.get("topic", topic_file.stem)
            kb = data.get("knowledge_base", {})

            # Search in source_documents
            for doc in kb.get("source_documents", []):
                text = (
                    doc.get("title", "")
                    + " "
                    + doc.get("snippet", doc.get("abstract", ""))
                    + " "
                    + doc.get("content", "")[:500]
                ).lower()
                score = self._score(keywords, text)
                if score > 0:
                    candidates.append((
                        score,
                        SourceDocument(
                            title=doc.get("title", "Untitled"),
                            url=doc.get("url", doc.get("doi", None)),
                            snippet=(doc.get("snippet") or doc.get("abstract", ""))[:400],
                            metadata={"topic": topic_name, "source_id": doc.get("source_id", "")},
                        ),
                    ))

            # Search in knowledge_articles
            for art in kb.get("knowledge_articles", []):
                text = (art.get("title", "") + " " + art.get("summary", "")[:500]).lower()
                score = self._score(keywords, text)
                if score > 0:
                    candidates.append((
                        score,
                        SourceDocument(
                            title=art.get("title", "Untitled"),
                            url=None,
                            snippet=art.get("summary", "")[:400],
                            metadata={"topic": topic_name, "article_id": art.get("article_id", "")},
                        ),
                    ))

        # Sort by descending score
        candidates.sort(key=lambda x: x[0], reverse=True)
        results = [doc for _, doc in candidates[:max_results]]

        if not results:
            return self._hardcoded_fallback(query, max_results)
        return results

    @staticmethod
    def _score(keywords: set[str], text: str) -> float:
        if not keywords:
            return 0.0
        hits = sum(1 for kw in keywords if kw in text)
        return hits / len(keywords)

    @staticmethod
    def _hardcoded_fallback(query: str, max_results: int) -> list[SourceDocument]:
        """Ultra-minimal fallback so the system never returns empty hands."""
        return [
            SourceDocument(
                title=f"Background knowledge on: {query}",
                url=None,
                snippet=(
                    "This is a synthesized background document for the query. "
                    "Multi-agent systems divide complex tasks among specialised "
                    "agents (Researcher, Analyst, Writer) coordinated by a Supervisor. "
                    "Each agent maintains focused context and hands off structured "
                    "state to the next, improving quality on long-horizon tasks."
                ),
                metadata={"source": "fallback"},
            )
        ][:max_results]
