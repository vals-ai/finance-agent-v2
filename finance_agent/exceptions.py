import logging
from collections.abc import Callable
from typing import Any

import aiohttp
import backoff
import httpx
from tavily import UsageLimitExceededError

logger = logging.getLogger(__name__)


class RetryExhaustedError(Exception):
    """Raised when all retry attempts for an HTTP status code have been exhausted."""

    pass


RetryPolicy = dict[int, int]  # {status_code: max_tries}


def _get_status_code(exception: Exception) -> int | None:
    if isinstance(exception, aiohttp.ClientResponseError):
        return exception.status
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code
    if isinstance(exception, UsageLimitExceededError):
        return 429
    return None


def retry_http_errors(*status_codes: int, max_tries: int = 100) -> Callable:
    """Simple retry decorator: retries any of the given status codes up to max_tries."""
    policy = {code: max_tries for code in status_codes}
    return retry_with_policy(policy)


def retry_with_policy(policy: RetryPolicy) -> Callable:
    """Retry decorator with per-status-code max tries.

    Usage:
        @retry_with_policy({429: 20, 503: 3})
        async def fetch(url): ...
    """
    overall_max = max(policy.values()) if policy else 1

    def decorator(func: Callable) -> Callable:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt_counts: dict[int, int] = {}

            def should_retry(exception: Exception) -> bool:
                code = _get_status_code(exception)
                if code is None:
                    for c in policy:
                        if str(c) in str(exception):
                            code = c
                            break
                if code is None or code not in policy:
                    return False

                attempt_counts[code] = attempt_counts.get(code, 0) + 1
                if attempt_counts[code] > policy[code]:
                    return False

                logger.error(f"{code} error: {exception}")
                return True

            @backoff.on_exception(
                backoff.expo,
                Exception,
                max_tries=overall_max,
                max_value=120,
                base=2,
                factor=3,
                jitter=backoff.full_jitter,
                giveup=lambda e: not should_retry(e),
            )
            async def _retrying(*a: Any, **kw: Any) -> Any:
                return await func(*a, **kw)

            return await _retrying(*args, **kwargs)

        return wrapper

    return decorator


FALLBACK_RETRY_POLICY: RetryPolicy = {429: 5, 503: 5}

URL_RETRY_POLICIES: dict[str, RetryPolicy] = {
    "sec.gov": {429: 20, 503: 20},
}


def get_retry_policy(url: str) -> RetryPolicy:
    """Get the retry policy for a URL based on domain patterns."""
    for pattern, policy in URL_RETRY_POLICIES.items():
        if pattern in url:
            return policy
    return FALLBACK_RETRY_POLICY
