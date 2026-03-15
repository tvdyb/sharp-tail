"""Mirror Executor: replicate detected trades with slippage controls."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import structlog

from polymir.api.clob import ClobClient
from polymir.config import AppConfig
from polymir.db import Database
from polymir.models import (
    MirrorDecision,
    MirrorTrade,
    OrderBook,
    TradeSignal,
)
from polymir.monitor import TradeMonitor

logger = structlog.get_logger()


class MirrorExecutor:
    """Processes trade signals and executes mirror trades with safety checks."""

    def __init__(self, config: AppConfig, db: Database) -> None:
        self._config = config
        self._db = db

    async def run(self) -> None:
        """Start monitor and process signals as they arrive."""
        monitor = TradeMonitor(self._config, self._db)
        # Run monitor and executor concurrently
        monitor_task = asyncio.create_task(monitor.run())
        try:
            async with ClobClient(self._config.api) as clob:
                while True:
                    signal = await monitor.signal_queue.get()
                    await self._process_signal(signal, clob)
        except asyncio.CancelledError:
            await monitor.stop()
            monitor_task.cancel()

    async def _process_signal(self, signal: TradeSignal, clob: ClobClient) -> MirrorTrade:
        """Evaluate and potentially execute a mirror trade for a signal."""
        exec_cfg = self._config.execution
        log = logger.bind(
            wallet=signal.wallet_address,
            market=signal.market_id,
            asset=signal.asset_id,
            side=signal.side,
        )

        # Check stale signal
        age_s = (datetime.utcnow() - signal.detected_at).total_seconds()
        if age_s > exec_cfg.stale_signal_timeout_s:
            log.info("skip_stale", age_s=age_s)
            return await self._record(signal, MirrorDecision.SKIP_STALE)

        # Fetch orderbook
        book = await clob.get_orderbook(signal.asset_id)

        # Check liquidity
        total_liq = book.total_liquidity()
        if total_liq < exec_cfg.min_liquidity_usd:
            log.info("skip_liquidity", liquidity=total_liq)
            return await self._record(
                signal, MirrorDecision.SKIP_LIQUIDITY, metadata={"liquidity": total_liq}
            )

        # Check spread
        spread_pct = book.spread_pct
        if spread_pct is not None and spread_pct > exec_cfg.max_spread_pct:
            log.info("skip_spread", spread_pct=spread_pct)
            return await self._record(
                signal, MirrorDecision.SKIP_SPREAD, metadata={"spread_pct": spread_pct}
            )

        # Compute position size (scale by liquidity, cap at max)
        order_size = self._compute_size(signal, book, exec_cfg)

        # Estimate slippage
        fill_price = book.estimate_fill_price(order_size, signal.side)
        if fill_price is None:
            log.info("skip_liquidity_fill", order_size=order_size)
            return await self._record(signal, MirrorDecision.SKIP_LIQUIDITY)

        midpoint = book.midpoint or signal.price
        estimated_slippage = abs(fill_price - midpoint) / midpoint if midpoint > 0 else 0

        if estimated_slippage > exec_cfg.max_slippage_pct:
            log.info("skip_slippage", estimated=estimated_slippage)
            return await self._record(
                signal,
                MirrorDecision.SKIP_SLIPPAGE,
                estimated_slippage=estimated_slippage,
            )

        # Compute order price with aggression
        order_price = self._compute_price(book, signal.side, exec_cfg.aggression)

        # Place order
        try:
            result = await clob.place_order(
                token_id=signal.asset_id,
                side=signal.side,
                price=order_price,
                size=order_size,
            )
            order_id = result.get("orderID", "")
        except Exception:
            log.exception("order_error")
            return await self._record(signal, MirrorDecision.ERROR)

        # Wait for fill with timeout
        filled = await self._wait_for_fill(clob, order_id, exec_cfg.fill_timeout_s)

        if not filled:
            try:
                await clob.cancel_order(order_id)
            except Exception:
                log.warning("cancel_failed", order_id=order_id)
            return await self._record(
                signal,
                MirrorDecision.SKIP_FILL_TIMEOUT,
                order_price=order_price,
                order_size=order_size,
                order_id=order_id,
                estimated_slippage=estimated_slippage,
            )

        # Record successful execution
        realized_slippage = abs(filled["avg_price"] - midpoint) / midpoint if midpoint > 0 else 0
        log.info(
            "trade_executed",
            order_id=order_id,
            price=order_price,
            size=order_size,
            estimated_slippage=estimated_slippage,
            realized_slippage=realized_slippage,
        )
        return await self._record(
            signal,
            MirrorDecision.EXECUTE,
            order_price=order_price,
            order_size=order_size,
            estimated_slippage=estimated_slippage,
            realized_slippage=realized_slippage,
            fill_price=filled["avg_price"],
            fill_size=filled["filled_size"],
            order_id=order_id,
        )

    @staticmethod
    def _compute_size(
        signal: TradeSignal, book: OrderBook, exec_cfg: Any
    ) -> float:
        """Scale order size by liquidity and cap at max position."""
        total_liq = book.total_liquidity()
        # Scale: use at most 10% of available liquidity
        liq_scaled = min(signal.size, total_liq * 0.10 / (signal.price or 1.0))
        # Cap at max position
        max_contracts = exec_cfg.max_position_usd / (signal.price or 1.0)
        return min(liq_scaled, max_contracts)

    @staticmethod
    def _compute_price(book: OrderBook, side: str, aggression: float) -> float:
        """Compute limit order price based on midpoint and aggression.

        aggression=0: midpoint
        aggression>0: move toward the opposite side (more aggressive)
        """
        mid = book.midpoint
        if mid is None:
            return 0.0
        spread = book.spread or 0.0
        if side == "BUY":
            return mid + aggression * spread * 0.5
        else:
            return mid - aggression * spread * 0.5

    async def _wait_for_fill(
        self, clob: ClobClient, order_id: str, timeout_s: float
    ) -> dict[str, Any] | None:
        """Poll order status until filled or timeout."""
        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            try:
                order = await clob.get_order(order_id)
                status = order.get("status", "")
                if status == "filled":
                    return {
                        "avg_price": float(order.get("avg_fill_price", 0)),
                        "filled_size": float(order.get("filled_size", 0)),
                    }
                if status in ("cancelled", "expired"):
                    return None
            except Exception:
                pass
            await asyncio.sleep(2.0)
        return None

    async def _record(
        self,
        signal: TradeSignal,
        decision: MirrorDecision,
        order_price: float | None = None,
        order_size: float | None = None,
        estimated_slippage: float | None = None,
        realized_slippage: float | None = None,
        fill_price: float | None = None,
        fill_size: float | None = None,
        order_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MirrorTrade:
        trade = MirrorTrade(
            signal=signal,
            decision=decision,
            order_price=order_price,
            order_size=order_size,
            estimated_slippage=estimated_slippage,
            realized_slippage=realized_slippage,
            fill_price=fill_price,
            fill_size=fill_size,
            order_id=order_id,
            executed_at=datetime.utcnow() if decision == MirrorDecision.EXECUTE else None,
            metadata=metadata or {},
        )
        await self._db.record_mirror_trade(trade)
        return trade
