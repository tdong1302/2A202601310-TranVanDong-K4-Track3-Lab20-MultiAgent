"""Tracing hooks.

Provides a minimal span context manager that records duration and attributes,
and optionally integrates with LangSmith when LANGSMITH_API_KEY is configured.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

# In-memory trace buffer for the current process (used when no external provider)
_trace_buffer: list[dict[str, Any]] = []
_active_langsmith_run: ContextVar[Any | None] = ContextVar(
    "active_langsmith_run", default=None
)


def get_trace_buffer() -> list[dict[str, Any]]:
    """Return all spans recorded so far (for local inspection / export)."""
    return list(_trace_buffer)


def clear_trace_buffer() -> None:
    """Clear the in-memory trace buffer."""
    _trace_buffer.clear()


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Span context manager that records duration and stores the span locally.

    When LangSmith is configured, also logs to LangSmith via its run_tree API.
    """
    started = perf_counter()
    start_time = datetime.now(tz=UTC)
    start_iso = start_time.isoformat()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "start_time": start_iso,
        "duration_seconds": None,
        "error": None,
    }
    langsmith_run = _start_langsmith_run(name, span["attributes"])
    run_token = _active_langsmith_run.set(langsmith_run) if langsmith_run else None
    try:
        logger.debug("trace_span START | %s | attrs=%s", name, attributes)
        yield span
    except Exception as exc:
        span["error"] = str(exc)
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
        span["end_time"] = datetime.now(tz=UTC).isoformat()
        _trace_buffer.append(span)
        logger.debug(
            "trace_span END | %s | duration=%.3fs",
            name,
            span["duration_seconds"],
        )
        if langsmith_run:
            _finish_langsmith_run(langsmith_run, span)
        if run_token is not None:
            _active_langsmith_run.reset(run_token)


def _start_langsmith_run(name: str, inputs: dict[str, Any]) -> Any | None:
    """Start a LangSmith run, nesting it under the active span when present."""
    try:
        from multi_agent_research_lab.core.config import get_settings

        settings = get_settings()
        if not settings.langsmith_api_key:
            return

        from langsmith import Client  # type: ignore[import-untyped]
        from langsmith.run_trees import RunTree  # type: ignore[import-untyped]

        parent = _active_langsmith_run.get()
        if parent is not None:
            run = parent.create_child(name=name, run_type="chain", inputs=inputs)
        else:
            client = Client(api_key=settings.langsmith_api_key)
            run = RunTree(
                name=name,
                run_type="chain",
                inputs=inputs,
                project_name=settings.langsmith_project,
                ls_client=client,
            )
        run.post()
        logger.debug("LangSmith: started span '%s'", name)
        return run
    except Exception as exc:
        logger.debug("LangSmith span start skipped: %s", exc)
        return None


def _finish_langsmith_run(run: Any, span: dict[str, Any]) -> None:
    """Finish and patch a previously started LangSmith run."""
    try:
        outputs = span.get(
            "outputs", {"duration_seconds": span.get("duration_seconds")}
        )
        run.end(outputs=outputs, error=span.get("error"))
        run.patch()
        span["langsmith_run_id"] = str(run.id)
        span["langsmith_trace_id"] = str(run.trace_id)
    except Exception as exc:
        logger.debug("LangSmith span finish skipped: %s", exc)


def export_trace_json(path: str) -> None:
    """Export the in-memory trace buffer to a JSON file."""
    import pathlib

    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_trace_buffer, indent=2), encoding="utf-8")
    logger.info("Trace exported to %s (%d spans)", path, len(_trace_buffer))
