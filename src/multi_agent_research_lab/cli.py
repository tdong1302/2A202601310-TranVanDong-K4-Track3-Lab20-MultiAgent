"""Command-line entrypoint for the lab starter."""

import time
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import (
    clear_trace_buffer,
    export_trace_json,
    trace_span,
)

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()

_BASELINE_SYSTEM = """\
You are an expert research assistant. Answer the user's query thoroughly, covering:
1. Key facts and definitions
2. Current state-of-the-art / recent developments
3. Trade-offs and limitations
4. Practical recommendations

Be factual, well-structured, and cite your reasoning. Length: 300-500 words."""


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline (one LLM call for search + analyse + write)."""
    _init()
    request = _parse_query(query)
    state = ResearchState(request=request)
    clear_trace_buffer()

    console.print(f"\n[bold cyan]Single-Agent Baseline[/bold cyan] | query: {query!r}\n")

    try:
        from multi_agent_research_lab.services.llm_client import LLMClient
        from multi_agent_research_lab.services.search_client import SearchClient

        # Gather sources first (same as multi-agent but in one agent)
        search_client = SearchClient()
        llm_client = LLMClient()

        t0 = time.perf_counter()
        sources = search_client.search(query, max_results=request.max_sources)
        state.sources = sources

        sources_block = "\n\n".join(
            f"[{i + 1}] {s.title}\n{s.snippet}" for i, s in enumerate(sources)
        ) or "(No external sources found)"

        user_prompt = (
            f"Query: {query}\n\n"
            f"Retrieved Sources:\n{sources_block}\n\n"
            "Answer the query comprehensively, citing sources with [n] notation."
        )

        with trace_span("baseline.single_agent", {"query": query}) as span:
            llm_response = llm_client.complete(
                system_prompt=_BASELINE_SYSTEM,
                user_prompt=user_prompt,
            )
            span["outputs"] = {
                "input_tokens": llm_response.input_tokens,
                "output_tokens": llm_response.output_tokens,
                "cost_usd": llm_response.cost_usd,
            }
        latency = time.perf_counter() - t0

        state.final_answer = llm_response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=llm_response.content,
                metadata={
                    "input_tokens": llm_response.input_tokens,
                    "output_tokens": llm_response.output_tokens,
                    "cost_usd": llm_response.cost_usd,
                    "latency_seconds": round(latency, 3),
                },
            )
        )

        console.print(Panel(state.final_answer or "", title="Answer", border_style="green"))

        # Show metrics
        table = Table(title="Baseline Metrics", show_header=True)
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Latency", f"{latency:.2f}s")
        table.add_row("Sources found", str(len(sources)))
        table.add_row("Input tokens", str(llm_response.input_tokens or "N/A"))
        table.add_row("Output tokens", str(llm_response.output_tokens or "N/A"))
        table.add_row(
            "Est. cost (USD)",
            f"${llm_response.cost_usd:.5f}" if llm_response.cost_usd else "N/A",
        )
        console.print(table)
        export_trace_json("reports/baseline_trace.json")

    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    except Exception as exc:
        console.print(Panel.fit(str(exc), title="Error", style="red"))
        raise typer.Exit(code=1) from exc


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the full multi-agent workflow (Supervisor → Researcher → Analyst → Writer)."""
    _init()
    state = ResearchState(request=_parse_query(query))
    clear_trace_buffer()
    console.print(f"\n[bold magenta]Multi-Agent Workflow[/bold magenta] | query: {query!r}\n")

    workflow = MultiAgentWorkflow()
    t0 = time.perf_counter()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    latency = time.perf_counter() - t0
    export_trace_json("reports/trace_export.json")

    # Print final answer
    console.print(
        Panel(
            result.final_answer or "(No answer generated)",
            title="Final Answer",
            border_style="magenta",
        )
    )

    # Route history
    console.print(f"\n[bold]Route history:[/bold] {' -> '.join(result.route_history)}")

    # Metrics table
    total_cost = sum(
        (r.metadata.get("cost_usd") or 0.0) for r in result.agent_results
    )
    total_in = sum(r.metadata.get("input_tokens") or 0 for r in result.agent_results)
    total_out = sum(r.metadata.get("output_tokens") or 0 for r in result.agent_results)

    table = Table(title="Multi-Agent Metrics", show_header=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Latency", f"{latency:.2f}s")
    table.add_row("Sources found", str(len(result.sources)))
    table.add_row("Iterations", str(result.iteration))
    table.add_row("Total input tokens", str(total_in))
    table.add_row("Total output tokens", str(total_out))
    table.add_row("Est. total cost (USD)", f"${total_cost:.5f}")
    table.add_row("Errors", str(len(result.errors)))
    console.print(table)

    if result.errors:
        console.print(Panel("\n".join(result.errors), title="Errors", style="red"))


if __name__ == "__main__":
    app()
