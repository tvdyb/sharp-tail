"""Trade Monitor: watch ranked wallets for new position entries."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import structlog

from polymir.api.clob import ClobClient
from polymir.api.websocket import TradeWebSocket
from polymir.config import AppConfig
from polymir.db import Database
from polymir.models import Trade, TradeSignal, WalletScore

logger = structlog.get_logger()


class TradeMonitor:
    """Watches top-ranked wallets and emits signals on new position entries."""

    def __init__(self, config: AppConfig, db: Database) -> None:
        self._config = config
        self._db = db
        self._watched_wallets: dict[str, WalletScore] = {}
        self._signal_queue: asyncio.Queue[TradeSignal] = asyncio.Queue()
        self._running = False

    @property
    def signal_queue(self) -> asyncio.Queue[TradeSignal]:
        return self._signal_queue

    async def load_watchlist(self) -> None:
        """Load top wallets from database into watchlist."""
        top = await self._db.get_top_wallets(limit=self._config.top_wallets)
        self._watched_wallets = {w.address: w for w in top}
        logger.info("watchlist_loaded", wallet_count=len(self._watched_wallets))

    async def run(self) -> None:
        """Start monitoring with WebSocket primary and polling fallback."""
        await self.load_watchlist()
        if not self._watched_wallets:
            logger.warning("no_wallets_to_watch")
            return

        self._running = True
        # Run WebSocket and polling concurrently
        await asyncio.gather(
            self._run_websocket(),
            self._run_polling(),
        )

    async def stop(self) -> None:
        self._running = False

    async def _run_websocket(self) -> None:
        """Primary: stream trades via WebSocket and filter for watched wallets."""
        ws = TradeWebSocket(self._config.api)
        try:
            await ws.connect()
            # We subscribe to all assets — the filter_fn handles wallet filtering
            async for trade in ws.stream_trades(filter_fn=self._is_watched_trade_raw):
                if not self._running:
                    break
                signal = self._trade_to_signal(trade)
                if signal:
                    await self._emit_signal(signal)
        except asyncio.CancelledError:
            pass
        finally:
            await ws.disconnect()

    async def _run_polling(self) -> None:
        """Fallback: poll CLOB API for recent trades by watched wallets."""
        async with ClobClient(self._config.api) as clob:
            seen_trade_ids: set[str] = set()
            while self._running:
                for wallet_addr in list(self._watched_wallets):
                    try:
                        trades = await clob.get_trades_for_wallet(wallet_addr, limit=20)
                        for trade in trades:
                            if trade.id and trade.id not in seen_trade_ids:
                                seen_trade_ids.add(trade.id)
                                signal = self._trade_to_signal(trade)
                                if signal:
                                    await self._emit_signal(signal)
                    except Exception:
                        logger.exception("polling_error", wallet=wallet_addr)

                # Cap memory of seen IDs
                if len(seen_trade_ids) > 50_000:
                    seen_trade_ids = set(list(seen_trade_ids)[-25_000:])

                await asyncio.sleep(self._config.execution.poll_interval_s)

    def _is_watched_trade_raw(self, raw: dict[str, Any]) -> bool:
        """Filter function for WebSocket stream — checks if trade is from a watched wallet."""
        owner = raw.get("owner", raw.get("maker_address", ""))
        return owner in self._watched_wallets

    def _trade_to_signal(self, trade: Trade) -> TradeSignal | None:
        """Convert a trade to a signal if it's a new position entry from a watched wallet."""
        if trade.owner not in self._watched_wallets:
            return None
        # Only signal on BUY trades (new position entries)
        if trade.side != "BUY":
            return None

        return TradeSignal(
            wallet_address=trade.owner,
            market_id=trade.market,
            asset_id=trade.asset_id,
            side=trade.side,
            size=trade.size,
            price=trade.price,
            detected_at=datetime.utcnow(),
        )

    async def _emit_signal(self, signal: TradeSignal) -> None:
        """Put signal on queue and log."""
        await self._signal_queue.put(signal)
        logger.info(
            "trade_signal",
            wallet=signal.wallet_address,
            market=signal.market_id,
            asset=signal.asset_id,
            side=signal.side,
            size=signal.size,
            price=signal.price,
        )
