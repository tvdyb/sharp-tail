"""Polymarket CLOB API client for orderbook, trades, and order management."""

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
from polymir.models import OrderBook, OrderBookLevel, Trade

logger = structlog.get_logger()


class ClobAPIError(Exception):
    """Raised when the CLOB API returns an error response."""

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        super().__init__(f"CLOB API error {status}: {message}")


class ClobClient:
    """Async client for the Polymarket CLOB API."""

    def __init__(self, config: APIConfig | None = None) -> None:
        self._config = config or APIConfig()
        self._limiter = RateLimiter(self._config.rate_limit_per_second)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> ClobClient:
        self._session = aiohttp.ClientSession(
            base_url=self._config.clob_base_url,
            timeout=aiohttp.ClientTimeout(total=self._config.request_timeout),
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("ClobClient must be used as async context manager")
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
                raise ClobAPIError(resp.status, text)
            return await resp.json()

    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, max=60),
        reraise=True,
    )
    async def _post(self, path: str, json: dict[str, Any] | None = None) -> Any:
        await self._limiter.acquire()
        session = self._ensure_session()
        headers = self._auth_headers()
        async with session.post(path, json=json, headers=headers) as resp:
            if resp.status not in (200, 201):
                text = await resp.text()
                raise ClobAPIError(resp.status, text)
            return await resp.json()

    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, max=60),
        reraise=True,
    )
    async def _delete(self, path: str, params: dict[str, Any] | None = None) -> Any:
        await self._limiter.acquire()
        session = self._ensure_session()
        headers = self._auth_headers()
        async with session.delete(path, params=params, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise ClobAPIError(resp.status, text)
            return await resp.json()

    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._config.api_key:
            headers["POLY_API_KEY"] = self._config.api_key
            headers["POLY_API_SECRET"] = self._config.api_secret
            headers["POLY_PASSPHRASE"] = self._config.api_passphrase
        return headers

    async def get_orderbook(self, token_id: str) -> OrderBook:
        """Fetch current order book for a token."""
        data = await self._get("/book", params={"token_id": token_id})
        return OrderBook(
            asset_id=token_id,
            bids=[OrderBookLevel(price=float(b["price"]), size=float(b["size"])) for b in data.get("bids", [])],
            asks=[OrderBookLevel(price=float(a["price"]), size=float(a["size"])) for a in data.get("asks", [])],
        )

    async def get_trades(
        self, asset_id: str | None = None, maker_address: str | None = None, limit: int = 100
    ) -> list[Trade]:
        """Fetch recent trades, optionally filtered by asset or maker."""
        params: dict[str, Any] = {"limit": limit}
        if asset_id:
            params["asset_id"] = asset_id
        if maker_address:
            params["maker_address"] = maker_address
        data = await self._get("/trades", params=params)
        return [Trade.model_validate(t) for t in data]

    async def get_trades_for_wallet(self, wallet: str, limit: int = 500) -> list[Trade]:
        """Fetch trades for a specific wallet address."""
        params: dict[str, Any] = {"limit": limit, "maker_address": wallet}
        data = await self._get("/trades", params=params)
        return [Trade.model_validate(t) for t in data]

    async def place_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> dict[str, Any]:
        """Place a limit order. Returns order details including order_id."""
        payload = {
            "tokenID": token_id,
            "side": side,
            "price": price,
            "size": size,
            "type": "GTC",
        }
        result = await self._post("/order", json=payload)
        logger.info(
            "order_placed",
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            order_id=result.get("orderID"),
        )
        return result

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel an open order."""
        result = await self._delete("/order", params={"orderID": order_id})
        logger.info("order_cancelled", order_id=order_id)
        return result

    async def get_order(self, order_id: str) -> dict[str, Any]:
        """Get order status."""
        return await self._get(f"/order/{order_id}")
