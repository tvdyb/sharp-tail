"""WebSocket client for real-time Polymarket trade feed."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Callable

import structlog
import websockets
from websockets.exceptions import ConnectionClosed

from polymir.config import APIConfig
from polymir.models import Trade

logger = structlog.get_logger()


class TradeWebSocket:
    """WebSocket client that streams trades for subscribed markets/assets."""

    def __init__(self, config: APIConfig | None = None) -> None:
        self._config = config or APIConfig()
        self._ws: Any = None
        self._subscribed_assets: set[str] = set()
        self._running = False

    async def connect(self) -> None:
        """Establish WebSocket connection."""
        self._ws = await websockets.connect(self._config.ws_url)
        self._running = True
        logger.info("ws_connected", url=self._config.ws_url)

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("ws_disconnected")

    async def subscribe(self, asset_ids: list[str]) -> None:
        """Subscribe to trade updates for given asset IDs."""
        if not self._ws:
            raise RuntimeError("Not connected")
        msg = json.dumps({"type": "subscribe", "assets_ids": asset_ids})
        await self._ws.send(msg)
        self._subscribed_assets.update(asset_ids)
        logger.info("ws_subscribed", asset_count=len(asset_ids))

    async def unsubscribe(self, asset_ids: list[str]) -> None:
        """Unsubscribe from trade updates."""
        if not self._ws:
            return
        msg = json.dumps({"type": "unsubscribe", "assets_ids": asset_ids})
        await self._ws.send(msg)
        self._subscribed_assets -= set(asset_ids)

    async def stream_trades(
        self,
        filter_fn: Callable[[dict[str, Any]], bool] | None = None,
    ) -> AsyncIterator[Trade]:
        """Yield trades from the WebSocket feed.

        Args:
            filter_fn: Optional filter applied to raw message dicts before parsing.
        """
        if not self._ws:
            raise RuntimeError("Not connected")

        while self._running:
            try:
                raw = await self._ws.recv()
                data = json.loads(raw)

                # Skip non-trade messages (heartbeats, acks, etc.)
                if not isinstance(data, dict) or data.get("type") not in ("trade", "last_trade_price"):
                    continue

                trades = data.get("trades", [data]) if "trades" in data else [data]
                for t in trades:
                    if filter_fn and not filter_fn(t):
                        continue
                    try:
                        yield Trade.model_validate(t)
                    except Exception:
                        logger.warning("ws_parse_error", raw_trade=t)

            except ConnectionClosed:
                logger.warning("ws_connection_closed, reconnecting...")
                await self._reconnect()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("ws_stream_error")
                await asyncio.sleep(1)

    async def _reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        delay = 1.0
        for attempt in range(self._config.max_retries):
            try:
                self._ws = await websockets.connect(self._config.ws_url)
                if self._subscribed_assets:
                    await self.subscribe(list(self._subscribed_assets))
                logger.info("ws_reconnected", attempt=attempt + 1)
                return
            except Exception:
                logger.warning("ws_reconnect_failed", attempt=attempt + 1, delay=delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._config.backoff_max)
        logger.error("ws_reconnect_exhausted")
        self._running = False
