"""Live trading client using py-clob-client for authenticated order signing."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from py_clob_client.client import ClobClient as PyClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, PartialCreateOrderOptions

from polymir.config import APIConfig
from polymir.models import OrderBook, OrderBookLevel, Trade

logger = structlog.get_logger()


class LiveClobClient:
    """Async-compatible wrapper around py-clob-client for live order execution.

    Uses the official Polymarket SDK for EIP-712 order signing, which requires
    a private key.  Read-only operations (orderbook, trades) use the unsigned
    REST API.  Order placement goes through the SDK's create_and_post_order
    which handles nonce generation, signing, and submission.
    """

    def __init__(self, config: APIConfig) -> None:
        self._config = config
        self._client: PyClobClient | None = None

    async def __aenter__(self) -> LiveClobClient:
        cfg = self._config
        creds = None
        if cfg.api_key:
            creds = ApiCreds(
                api_key=cfg.api_key,
                api_secret=cfg.api_secret,
                api_passphrase=cfg.api_passphrase,
            )

        self._client = PyClobClient(
            host=cfg.clob_base_url,
            chain_id=cfg.chain_id,
            key=cfg.private_key or None,
            creds=creds,
        )

        # Derive API creds from private key if not provided
        if cfg.private_key and not cfg.api_key:
            logger.info("deriving_api_creds", msg="No API key provided, deriving from private key")
            self._client.set_api_creds(self._client.create_or_derive_api_creds())

        logger.info("live_client_connected", host=cfg.clob_base_url, chain_id=cfg.chain_id)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self._client = None

    def _ensure_client(self) -> PyClobClient:
        if self._client is None:
            raise RuntimeError("LiveClobClient must be used as async context manager")
        return self._client

    async def get_orderbook(self, token_id: str) -> OrderBook:
        """Fetch current order book for a token."""
        client = self._ensure_client()
        data = await asyncio.to_thread(client.get_order_book, token_id)
        return OrderBook(
            asset_id=token_id,
            bids=[
                OrderBookLevel(price=float(b.get("price", 0)), size=float(b.get("size", 0)))
                for b in (data.get("bids") or [])
            ],
            asks=[
                OrderBookLevel(price=float(a.get("price", 0)), size=float(a.get("size", 0)))
                for a in (data.get("asks") or [])
            ],
        )

    async def get_trades(
        self, asset_id: str | None = None, maker_address: str | None = None, limit: int = 100
    ) -> list[Trade]:
        """Fetch recent trades."""
        client = self._ensure_client()
        params: dict[str, Any] = {"limit": limit}
        if asset_id:
            params["asset_id"] = asset_id
        if maker_address:
            params["maker_address"] = maker_address
        data = await asyncio.to_thread(client.get_trades, params=params)
        return [Trade.model_validate(t) for t in data]

    async def place_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> dict[str, Any]:
        """Place a signed limit order via py-clob-client."""
        client = self._ensure_client()

        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size,
            side="BUY" if side.upper() == "BUY" else "SELL",
        )

        # Determine tick size from price precision
        tick_size = _infer_tick_size(price)

        result = await asyncio.to_thread(
            client.create_and_post_order,
            order_args,
            PartialCreateOrderOptions(tick_size=tick_size),
        )

        order_id = ""
        if isinstance(result, dict):
            order_id = result.get("orderID", result.get("id", ""))
        logger.info(
            "order_placed",
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            order_id=order_id,
        )
        return result if isinstance(result, dict) else {"orderID": str(result)}

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel an open order."""
        client = self._ensure_client()
        result = await asyncio.to_thread(client.cancel_orders, [order_id])
        logger.info("order_cancelled", order_id=order_id)
        return result if isinstance(result, dict) else {}

    async def get_order(self, order_id: str) -> dict[str, Any]:
        """Get order status."""
        client = self._ensure_client()
        result = await asyncio.to_thread(client.get_order, order_id)
        return result if isinstance(result, dict) else {}


def _infer_tick_size(price: float) -> str:
    """Infer the appropriate tick size for a given price.

    Polymarket uses tick sizes of 0.1, 0.01, 0.001, or 0.0001.
    Most markets use 0.01 (cent precision).
    """
    # Check how many decimal places the price has
    price_str = f"{price:.4f}".rstrip("0")
    decimals = len(price_str.split(".")[1]) if "." in price_str else 0

    if decimals <= 1:
        return "0.1"
    elif decimals <= 2:
        return "0.01"
    elif decimals <= 3:
        return "0.001"
    return "0.0001"
