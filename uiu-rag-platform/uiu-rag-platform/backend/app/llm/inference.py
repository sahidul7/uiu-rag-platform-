"""
UIU Academic RAG Platform — LLM Inference (Groq - FREE)

Groq provides free API access to:
  - llama-3.1-8b-instant   → fastest responses
  - llama-3.1-70b-versatile → best quality
  - mixtral-8x7b-32768     → long context

Free tier limits:
  - 30 requests/minute
  - 6000 tokens/minute
  - 500,000 tokens/day

Groq's API is OpenAI-compatible, so the interface is nearly identical.
"""
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from groq import AsyncGroq
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.exceptions import LLMError, LLMRateLimitError, LLMTimeoutError
from app.core.logging import get_logger
from app.prompts.builder import BuiltPrompt, CitationExtractor, prompt_builder

logger = get_logger(__name__)


# ── Provider Abstraction ──────────────────────────────────────────────────────
class BaseLLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str: ...

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[str, None]: ...


# ── Groq Provider ─────────────────────────────────────────────────────────────
class GroqProvider(BaseLLMProvider):
    """
    Free Groq API provider.
    Uses Llama 3.1 by default — excellent quality, fully free.
    """

    def __init__(self) -> None:
        self._client = AsyncGroq(
            api_key=settings.GROQ_API_KEY,
            timeout=settings.LLM_TIMEOUT,
            max_retries=0,
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(settings.LLM_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = settings.LLM_MAX_TOKENS,
        temperature: float = settings.LLM_TEMPERATURE,
    ) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            err_str = str(e).lower()
            if "rate limit" in err_str or "429" in err_str:
                raise LLMRateLimitError(f"Groq rate limit hit: {e}")
            if "timeout" in err_str:
                raise LLMTimeoutError(f"Groq timeout: {e}")
            raise LLMError(f"Groq API error: {e}")

    async def stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = settings.LLM_MAX_TOKENS,
        temperature: float = settings.LLM_TEMPERATURE,
    ) -> AsyncGenerator[str, None]:
        try:
            stream = await self._client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            err_str = str(e).lower()
            if "rate limit" in err_str:
                raise LLMRateLimitError(f"Groq stream rate limit: {e}")
            raise LLMError(f"Groq stream error: {e}")


# ── Provider Factory ──────────────────────────────────────────────────────────
class ProviderFactory:
    """
    Add new providers here without touching inference logic.
    Future: swap to OpenAI, Anthropic, or local Ollama any time.
    """
    _providers: dict[str, type[BaseLLMProvider]] = {
        "groq": GroqProvider,
        # "openai": OpenAIProvider,   # future
        # "ollama": OllamaProvider,   # future local LLM
    }

    @classmethod
    def create(cls, provider_name: str = "groq") -> BaseLLMProvider:
        provider_cls = cls._providers.get(provider_name)
        if not provider_cls:
            raise LLMError(f"Unknown provider: {provider_name}")
        return provider_cls()


# ── Inference Result ──────────────────────────────────────────────────────────
@dataclass
class InferenceResult:
    response_text: str
    citations: list[dict[str, Any]]
    source_map: dict[int, str]
    token_count: int
    latency_ms: int
    model: str
    was_compressed: bool


# ── LLM Inference Engine ──────────────────────────────────────────────────────
class LLMInference:
    def __init__(self) -> None:
        self._provider: BaseLLMProvider = ProviderFactory.create("groq")
        self._citation_extractor = CitationExtractor()

    async def query(
        self,
        built_prompt: BuiltPrompt,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> InferenceResult:
        """Non-streaming — full response at once."""
        messages = prompt_builder.build_messages(built_prompt, conversation_history)
        start_time = time.monotonic()

        response_text = await self._provider.complete(
            messages=messages,
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        )

        latency_ms = int((time.monotonic() - start_time) * 1000)

        citations = self._citation_extractor.extract(
            response_text,
            built_prompt.source_map,
            built_prompt.context_chunks,
        )

        logger.info(
            "Groq inference complete",
            model=settings.LLM_MODEL,
            latency_ms=latency_ms,
            citations_found=len(citations),
        )

        return InferenceResult(
            response_text=response_text,
            citations=citations,
            source_map=built_prompt.source_map,
            token_count=built_prompt.total_tokens,
            latency_ms=latency_ms,
            model=settings.LLM_MODEL,
            was_compressed=built_prompt.was_compressed,
        )

    async def stream_query(
        self,
        built_prompt: BuiltPrompt,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming — yields tokens as they arrive."""
        messages = prompt_builder.build_messages(built_prompt, conversation_history)

        async for token in self._provider.stream(
            messages=messages,
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        ):
            yield token


# Singleton
llm_inference = LLMInference()
