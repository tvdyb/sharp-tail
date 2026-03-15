"""Async rate limiter using token bucket algorithm."""

from __future__ import annotations

import asyncio
import time


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, rate_per_second: float = 5.0) -> None:
        self._rate = rate_per_second
        self._tokens = rate_per_second
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available, then consume it."""
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self._rate, self._tokens + elapsed * self._rate)
                self._last_refill = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

                wait_time = (1.0 - self._tokens) / self._rate
                await asyncio.sleep(wait_time)
