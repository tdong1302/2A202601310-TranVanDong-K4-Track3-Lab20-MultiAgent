"""Benchmark module: single-agent vs multi-agent comparison.

Measures latency, estimated token cost, citation coverage, and quality score
for each runner and returns structured BenchmarkMetrics.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


def _count_citations(text: str | None) -> int:
    """Count unique numeric citations like [1], [2], [3] in text."""
    if not text:
        return 0
    return len(set(re.findall(r"\[(\d+)\]", text)))


def _estimate_quality(state: ResearchState, query: str) -> float:
    """Heuristic quality score 0-10 based on content richness.

    Criteria:
    - Has final_answer with >100 chars  → +2
    - Citations present                 → +2
    - Research notes present            → +1
    - Analysis notes present            → +1
    - Sources retrieved (>0)            → +1
    - No errors                         → +1
    - Answer length >300 chars          → +1
    - Query terms appear in answer      → +1
    """
    score = 0.0
    answer = state.final_answer or ""

    if len(answer) > 100:
        score += 2
    if _count_citations(answer) > 0:
        score += 2
    if state.research_notes:
        score += 1
    if state.analysis_notes:
        score += 1
    if state.sources:
        score += 1
    if not state.errors:
        score += 1
    if len(answer) > 300:
        score += 1

    # Check query term coverage
    query_words = set(re.findall(r"\w+", query.lower())) - {
        "the", "a", "an", "is", "are", "and", "or", "of", "in", "to", "for"
    }
    if query_words and answer:
        coverage = sum(1 for w in query_words if w in answer.lower()) / len(query_words)
        if coverage > 0.5:
            score += 1

    return min(score, 10.0)


def _citation_coverage(state: ResearchState) -> float | None:
    """Fraction of retrieved sources that are cited in the final answer."""
    n_sources = len(state.sources)
    if n_sources == 0:
        return None
    n_cited = min(_count_citations(state.final_answer), n_sources)
    return n_cited / n_sources


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, cost, quality, citation coverage, and failure rate for one run."""
    logger.info("Benchmark: starting run '%s' for query='%s'", run_name, query)

    started = perf_counter()
    failed = False
    state: ResearchState | None = None

    try:
        state = runner(query)
        failed = bool(state.errors)
    except Exception as exc:
        logger.error("Benchmark: runner '%s' raised exception: %s", run_name, exc)
        failed = True
        from multi_agent_research_lab.core.schemas import ResearchQuery

        state = ResearchState(request=ResearchQuery(query=query))
        state.errors.append(str(exc))

    latency = perf_counter() - started

    # Aggregate token cost across all agent_results
    total_cost: float | None = None
    cost_values = [
        r.metadata.get("cost_usd")
        for r in state.agent_results
        if r.metadata.get("cost_usd") is not None
    ]
    if cost_values:
        total_cost = sum(float(c) for c in cost_values)

    quality = _estimate_quality(state, query) if not failed else 0.0
    citation_cov = _citation_coverage(state)
    failure_rate = 1.0 if failed else 0.0

    notes_parts = []
    if state.route_history:
        notes_parts.append(f"route: {' → '.join(state.route_history)}")
    if state.errors:
        notes_parts.append(f"errors: {'; '.join(state.errors[:2])}")
    notes = "; ".join(notes_parts)

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 3),
        estimated_cost_usd=round(total_cost, 6) if total_cost is not None else None,
        quality_score=round(quality, 1),
        citation_coverage=round(citation_cov, 3) if citation_cov is not None else None,
        failure_rate=failure_rate,
        notes=notes,
    )
    logger.info(
        "Benchmark: '%s' done | latency=%.2fs | quality=%.1f | cost=$%s",
        run_name,
        latency,
        quality,
        f"{total_cost:.5f}" if total_cost else "N/A",
    )
    return state, metrics
