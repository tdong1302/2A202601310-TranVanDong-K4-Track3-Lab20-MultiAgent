"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

# Cost per 1K tokens in USD (gpt-4o-mini pricing)
_COST_INPUT_PER_1K = 0.000150
_COST_OUTPUT_PER_1K = 0.000600


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client that connects to OpenAI (or compatible API)."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Copy .env.example to .env and fill in your key."
            )
        try:
            from openai import OpenAI  # type: ignore[import-untyped]

            self._client = OpenAI(api_key=settings.openai_api_key)
            self._model = settings.openai_model
        except ImportError as exc:
            raise ImportError(
                "openai package not installed. Run: pip install -e '.[llm]'"
            ) from exc

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with retry, timeout, and token logging."""
        settings = get_settings()
        logger.debug("LLMClient.complete | model=%s", self._model)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=settings.timeout_seconds,
        )

        choice = response.choices[0]
        content = choice.message.content or ""
        usage = response.usage

        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None
        cost_usd = None
        if input_tokens is not None and output_tokens is not None:
            cost_usd = (input_tokens / 1000) * _COST_INPUT_PER_1K + (
                output_tokens / 1000
            ) * _COST_OUTPUT_PER_1K

        logger.info(
            "LLM | in=%s out=%s cost=$%.5f",
            input_tokens,
            output_tokens,
            cost_usd or 0,
        )
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
