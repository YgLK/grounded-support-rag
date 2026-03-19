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
_RETRYABLE_MESSAGE_PATTERN = re.compile(
    r"(429|rate limit|timeout|temporar|try again|connection|unavailable|bad gateway|service unavailable|internal server error)",
    re.IGNORECASE,
)
_SHARED_SEMAPHORES: dict[tuple[int, str, str, str, int], asyncio.Semaphore] = {}


def shared_llm_semaphore(config: RuntimeConfigLike) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    max_concurrency = max(1, int(config.llm_max_concurrency))
    key = (
        id(loop),
        str(config.provider_type),
        str(chat_provider_base_url(config) or ""),
        str(config.chat_model or ""),
        max_concurrency,
    )
    semaphore = _SHARED_SEMAPHORES.get(key)
    if semaphore is None:
        semaphore = asyncio.Semaphore(max_concurrency)
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

    return bool(_RETRYABLE_MESSAGE_PATTERN.search(str(exc)))


async def ainvoke_with_retry(
    operation: Callable[[], Awaitable[_T]],
    *,
    semaphore: asyncio.Semaphore,
    max_attempts: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
) -> _T:
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max(1, int(max_attempts))),
        wait=wait_exponential_jitter(
            initial=max(0.0, float(base_delay_seconds)),
            max=max(float(base_delay_seconds), float(max_delay_seconds)),
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
