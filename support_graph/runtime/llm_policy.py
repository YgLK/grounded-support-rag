"""Async LLM retry and concurrency helpers."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import TypeVar

from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt
from tenacity.wait import wait_exponential_jitter

from support_graph.config.runtime import RuntimeConfigLike
from support_graph.providers import chat_provider_base_url


_T = TypeVar("_T")
SemaphoreKey = tuple[int, str, str, str, int]
_RETRYABLE_MESSAGE_PATTERN = re.compile(
    r"(429|rate limit|timeout|temporar|try again|connection|unavailable|bad gateway|service unavailable|internal server error)",
    re.IGNORECASE,
)
_SHARED_SEMAPHORES: dict[SemaphoreKey, asyncio.Semaphore] = {}


def _max_concurrency(config: RuntimeConfigLike) -> int:
    assert config.llm_max_concurrency > 0
    return config.llm_max_concurrency


def _semaphore_key(config: RuntimeConfigLike) -> SemaphoreKey:
    return (
        id(asyncio.get_running_loop()),
        config.provider_type,
        chat_provider_base_url(config) or "",
        config.chat_model or "",
        _max_concurrency(config),
    )


def shared_llm_semaphore(config: RuntimeConfigLike) -> asyncio.Semaphore:
    key = _semaphore_key(config)
    semaphore = _SHARED_SEMAPHORES.get(key)
    if semaphore is None:
        semaphore = asyncio.Semaphore(key[-1])
        _SHARED_SEMAPHORES[key] = semaphore
    return semaphore


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
    if isinstance(exc, (TimeoutError, ConnectionError, asyncio.TimeoutError, OSError)):
        return True

    exception_name = type(exc).__name__.lower()
    if exception_name in {
        "connecttimeout",
        "readtimeout",
        "apiconnectionerror",
        "internalservererror",
        "serviceunavailableerror",
        "ratelimiterror",
    }:
        return True

    status_code = _exception_status_code(exc)
    if status_code == 429:
        return True
    if status_code is not None and 500 <= status_code < 600:
        return True
    if _RETRYABLE_MESSAGE_PATTERN.search(str(exc)):
        return True
    return False


async def ainvoke_with_retry(
    operation: Callable[[], Awaitable[_T]],
    *,
    semaphore: asyncio.Semaphore,
    max_attempts: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
) -> _T:
    assert max_attempts > 0
    assert base_delay_seconds >= 0.0
    assert max_delay_seconds >= base_delay_seconds
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
            async with semaphore:
                return await operation()
    raise RuntimeError("LLM retry policy exhausted without returning or raising.")


__all__ = [
    "ainvoke_with_retry",
    "is_retryable_exception",
    "shared_llm_semaphore",
]
