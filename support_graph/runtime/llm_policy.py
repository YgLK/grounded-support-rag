"""Async LLM retry and concurrency helpers."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import TypeVar

from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt
from tenacity.wait import wait_exponential_jitter

__all__ = [
    "LLMCallTimeoutError",
    "ainvoke_with_retry",
    "is_retryable_exception",
]


_T = TypeVar("_T")


class LLMCallTimeoutError(TimeoutError):
    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds
        super().__init__(f"LLM call timed out after {timeout_seconds:.1f}s")


def _exception_status_code(exc: BaseException) -> int | None:
    direct_status = getattr(exc, "status_code", None)
    if isinstance(direct_status, int):
        return direct_status
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status
    return None


def is_retryable_exception(exc: BaseException) -> bool:
    """Determine if an LLM call failure is safe to retry.

    - Returns False for intentional timeouts (LLMCallTimeoutError)
    - Returns True for connection errors, HTTP 429 rate limits, or HTTP 5xx errors
    - Uses regex only as a fallback for transient unstructured error text
    """
    if isinstance(exc, LLMCallTimeoutError):
        return False
    if isinstance(exc, (TimeoutError, ConnectionError, asyncio.TimeoutError, OSError)):
        return True

    retryable_exception_names = {
        "connecttimeout",
        "readtimeout",
        "apiconnectionerror",
        "internalservererror",
        "serviceunavailableerror",
        "ratelimiterror",
    }
    retryable_message_pattern = re.compile(
        r"(temporar(?:y|ily)|try again|connection (?:reset|aborted|dropped)|tim(?:e|ed) out)",
        re.IGNORECASE,
    )

    exception_name = type(exc).__name__.lower()
    if exception_name in retryable_exception_names:
        return True

    status_code = _exception_status_code(exc)
    if status_code is not None and (status_code == 429 or 500 <= status_code < 600):
        return True

    return bool(retryable_message_pattern.search(str(exc)))


async def ainvoke_with_retry(
    operation: Callable[[], Awaitable[_T]],
    *,
    timeout_seconds: float | None,
    max_attempts: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
) -> _T:
    """Execute an async LLM call with a configured retry and timeout policy.

    - Wraps the call in `asyncio.wait_for` if a timeout is configured
    - Uses `tenacity` for exponential backoff with jitter
    - Only retries exceptions that pass `is_retryable_exception`
    - Reraises the final exception if max attempts are exceeded
    """
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")
    if base_delay_seconds < 0.0:
        raise ValueError("base_delay_seconds must be non-negative.")
    if max_delay_seconds < base_delay_seconds:
        raise ValueError(
            "max_delay_seconds must be greater than or equal to base_delay_seconds."
        )
    if timeout_seconds is not None and timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive when provided.")
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(
            initial=base_delay_seconds,
            max=max_delay_seconds,
        ),
        retry=retry_if_exception(is_retryable_exception),
        reraise=True,
    ):
        with attempt:
            if timeout_seconds is None:
                return await operation()
            try:
                return await asyncio.wait_for(operation(), timeout=timeout_seconds)
            except asyncio.TimeoutError as exc:
                raise LLMCallTimeoutError(timeout_seconds) from exc
    raise RuntimeError("LLM retry policy exhausted without returning or raising.")
