"""Retry decorators, rate-limiting, and resilient API wrappers for Groq and Cohere."""

import time
import logging
import threading

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)
from llama_index.embeddings.cohere import CohereEmbedding

logger = logging.getLogger(__name__)

_RETRYABLE_CODES = {429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    for attr in ("status_code", "status", "http_status"):
        code = getattr(exc, attr, None)
        if code is not None:
            try:
                if int(code) in _RETRYABLE_CODES:
                    return True
            except (ValueError, TypeError):
                pass
    cause = getattr(exc, "__cause__", None)
    if cause and cause is not exc:
        return _is_retryable(cause)
    return False


retry_groq_api = retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=2, min=5, max=120),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

retry_cohere_api = retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=2, min=10, max=120),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


class _TokenBucket:
    """Continuous token-bucket rate limiter."""

    def __init__(self, tokens_per_minute: int, initial_tokens: int | None = None):
        self._capacity = tokens_per_minute
        self._refill_rate = tokens_per_minute / 60.0
        self._tokens = float(initial_tokens if initial_tokens is not None else tokens_per_minute)
        self._last_ts = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(
            self._capacity,
            self._tokens + (now - self._last_ts) * self._refill_rate,
        )
        self._last_ts = now

    def acquire(self, tokens: int) -> None:
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return
            deficit = tokens - self._tokens
            wait = deficit / self._refill_rate
            logger.info("Rate limiter: waiting %.1fs for ~%d tokens", wait, tokens)
            time.sleep(wait)
            self._refill()
            self._tokens = max(0.0, self._tokens - tokens)


embed_limiter = _TokenBucket(tokens_per_minute=80_000, initial_tokens=15_000)
groq_limiter = _TokenBucket(tokens_per_minute=25_000, initial_tokens=5_000)


def _estimate_tokens(texts: list[str] | str) -> int:
    if isinstance(texts, str):
        return max(len(texts) // 4, 50)
    return sum(max(len(t) // 4, 50) for t in texts)


class RetryCohereEmbedding(CohereEmbedding):
    """CohereEmbedding with proactive rate-limiting and retry on 429 / 5xx."""

    @retry_cohere_api
    def _get_text_embedding(self, text: str) -> list[float]:
        embed_limiter.acquire(_estimate_tokens(text))
        return super()._get_text_embedding(text)

    @retry_cohere_api
    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        embed_limiter.acquire(_estimate_tokens(texts))
        return super()._get_text_embeddings(texts)

    @retry_cohere_api
    def _get_query_embedding(self, query: str) -> list[float]:
        embed_limiter.acquire(_estimate_tokens(query))
        return super()._get_query_embedding(query)
