"""Gamma Markets API client for market metadata and resolved markets."""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from polymir.api.rate_limiter import RateLimiter
from polymir.config import APIConfig
from polymir.models import Market, Token

logger = structlog.get_logger()


class GammaAPIError(Exception):
    def __init__(self, status: int, message: str) -> None:
        self.status = status
        super().__init__(f"Gamma API error {status}: {message}")


class GammaClient:
    """Async client for the Gamma Markets API (market metadata)."""

    def __init__(self, config: APIConfig | None = None) -> None:
        self._config = config or APIConfig()
        self._limiter = RateLimiter(self._config.rate_limit_per_second)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> GammaClient:
        self._session = aiohttp.ClientSession(
            base_url=self._config.gamma_base_url,
            timeout=aiohttp.ClientTimeout(total=self._config.request_timeout),
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("GammaClient must be used as async context manager")
        return self._session

    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, max=60),
        reraise=True,
    )
    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        await self._limiter.acquire()
        session = self._ensure_session()
        async with session.get(path, params=params) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise GammaAPIError(resp.status, text)
            return await resp.json()

    async def get_markets(
        self,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Market]:
        """Fetch markets, optionally filtered by status."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        data = await self._get("/markets", params=params)
        return [self._parse_market(m) for m in data]

    async def get_resolved_markets(self, limit: int = 100, offset: int = 0) -> list[Market]:
        """Fetch all resolved markets."""
        return await self.get_markets(status="resolved", limit=limit, offset=offset)

    async def get_all_resolved_markets(self, page_size: int = 100) -> list[Market]:
        """Paginate through all resolved markets."""
        all_markets: list[Market] = []
        offset = 0
        while True:
            batch = await self.get_resolved_markets(limit=page_size, offset=offset)
            if not batch:
                break
            all_markets.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size
        logger.info("fetched_resolved_markets", count=len(all_markets))
        return all_markets

    async def get_market(self, condition_id: str) -> Market:
        """Fetch a single market by condition ID."""
        data = await self._get(f"/markets/{condition_id}")
        return self._parse_market(data)

    def _parse_market(self, data: dict[str, Any]) -> Market:
        tokens = []
        for t in data.get("tokens", []):
            tokens.append(
                Token(
                    token_id=t.get("token_id", t.get("tokenId", "")),
                    outcome=t.get("outcome", ""),
                    price=float(t.get("price", 0)),
                    winner=bool(t.get("winner", False)),
                )
            )
        return Market(
            condition_id=data.get("condition_id", data.get("conditionId", "")),
            question=data.get("question", ""),
            slug=data.get("slug", ""),
            status=data.get("status", ""),
            outcome=data.get("outcome", ""),
            end_date=data.get("end_date") or data.get("endDate"),
            resolution_date=data.get("resolution_date") or data.get("resolutionDate"),
            tokens=tokens,
            volume=float(data.get("volume", 0)),
            liquidity=float(data.get("liquidity", 0)),
        )
