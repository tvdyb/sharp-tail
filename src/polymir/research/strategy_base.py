"""Base class for alpha research strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any

import structlog

from polymir.research.models import (
    PriceSnapshot,
    ResearchMarket,
    Signal,
    StrategyResult,
    Trade,
)

logger = structlog.get_logger()


class Strategy(ABC):
    """Abstract base class for research strategies."""

    name: str = "base"

    def __init__(self, markets: list[ResearchMarket], prices: list[PriceSnapshot]) -> None:
        self._markets = markets
        self._prices = prices
        self._market_map: dict[str, ResearchMarket] = {m.market_id: m for m in markets}
        self._price_map: dict[str, list[PriceSnapshot]] = {}
        for p in prices:
            self._price_map.setdefault(p.market_id, []).append(p)
        # Sort each market's prices by time
        for mid in self._price_map:
            self._price_map[mid].sort(key=lambda x: x.timestamp)

    @abstractmethod
    def generate_signals(self, as_of: datetime) -> list[Signal]:
        """Generate trading signals using only data available before as_of."""
        ...

    def backtest(
        self,
        start: date,
        end: date,
        capital: float = 10_000.0,
        fee_rate: float = 0.02,
        max_position_frac: float = 0.10,
        **params: Any,
    ) -> StrategyResult:
        """Backtest the strategy over a date range.

        Uses resolved market outcomes for P&L calculation.
        Quarter-Kelly sizing, capped at max_position_frac of capital.
        """
        result = StrategyResult(
            strategy_name=self.name,
            params=params,
            start_date=start,
            end_date=end,
        )

        # Get all unique timestamps from price data within range
        start_dt = datetime.combine(start, datetime.min.time())
        end_dt = datetime.combine(end, datetime.max.time())

        all_timestamps = sorted(set(
            p.timestamp for p in self._prices
            if start_dt <= p.timestamp <= end_dt
        ))

        if not all_timestamps:
            logger.info("no_timestamps_in_range", strategy=self.name, start=start, end=end)
            return result

        # Sample at daily frequency for signal generation
        daily_timestamps = self._sample_daily(all_timestamps)

        for as_of in daily_timestamps:
            signals = self.generate_signals(as_of)

            for signal in signals:
                market = self._market_map.get(signal.market_id)
                if not market:
                    continue

                # Only trade resolved markets in backtest (we need outcome for P&L)
                if market.status != "resolved" or not market.outcome:
                    continue

                # Quarter-Kelly sizing
                if signal.confidence <= 0:
                    continue
                edge = signal.confidence - signal.entry_price if signal.direction == "BUY" else signal.entry_price - signal.confidence
                if edge <= 0:
                    continue
                odds = (1.0 - signal.entry_price) / signal.entry_price if signal.entry_price > 0 else 0
                if odds <= 0:
                    continue
                kelly = edge / (1.0 - signal.entry_price) if signal.entry_price < 1 else 0
                quarter_kelly = kelly * 0.25
                position_frac = min(quarter_kelly * signal.size_fraction, max_position_frac)
                size_usd = capital * position_frac

                if size_usd < 1.0:
                    continue

                # Determine outcome
                resolved_yes = market.outcome.lower() in ("yes", "1", "true")
                if signal.direction == "BUY":
                    exit_price = 1.0 if resolved_yes else 0.0
                else:
                    exit_price = 0.0 if resolved_yes else 1.0

                # Calculate P&L
                entry_price = signal.entry_price
                spread_cost = 0.01  # assume 1c spread crossing
                effective_entry = entry_price + spread_cost if signal.direction == "BUY" else entry_price - spread_cost
                effective_entry = max(0.01, min(0.99, effective_entry))

                contracts = size_usd / effective_entry
                if signal.direction == "BUY":
                    gross_pnl = (exit_price - effective_entry) * contracts
                else:
                    gross_pnl = (effective_entry - exit_price) * contracts

                fees = abs(gross_pnl) * fee_rate if gross_pnl > 0 else 0.0
                net_pnl = gross_pnl - fees

                holding_hours = 0.0
                if market.resolution_date:
                    holding_hours = (market.resolution_date - as_of).total_seconds() / 3600.0

                trade = Trade(
                    signal=signal,
                    entry_time=as_of,
                    exit_time=market.resolution_date,
                    entry_price=effective_entry,
                    exit_price=exit_price,
                    size_usd=size_usd,
                    pnl=gross_pnl,
                    fees=fees,
                    net_pnl=net_pnl,
                    holding_period_hours=max(0, holding_hours),
                    slippage=spread_cost,
                )
                result.trades.append(trade)

        result.trades.sort(key=lambda t: t.entry_time)
        logger.info(
            "backtest_complete",
            strategy=self.name,
            trades=len(result.trades),
            total_pnl=result.total_pnl,
            sharpe=result.sharpe_ratio,
        )
        return result

    def _get_prices_before(self, market_id: str, as_of: datetime) -> list[PriceSnapshot]:
        """Get price history for a market before as_of (no look-ahead)."""
        snapshots = self._price_map.get(market_id, [])
        return [p for p in snapshots if p.timestamp < as_of]

    def _get_latest_price(self, market_id: str, as_of: datetime) -> float | None:
        """Get the most recent price before as_of."""
        history = self._get_prices_before(market_id, as_of)
        if not history:
            return None
        return history[-1].price

    def _resolved_markets(self) -> list[ResearchMarket]:
        """Get all resolved markets."""
        return [m for m in self._markets if m.status == "resolved"]

    def _active_markets_at(self, as_of: datetime) -> list[ResearchMarket]:
        """Markets that were active at the given time."""
        result = []
        for m in self._markets:
            created = m.creation_date
            resolved = m.resolution_date or m.end_date
            if created and created <= as_of:
                if resolved is None or resolved > as_of:
                    result.append(m)
        return result

    @staticmethod
    def _sample_daily(timestamps: list[datetime]) -> list[datetime]:
        """Sample one timestamp per day."""
        seen_dates: set[date] = set()
        daily: list[datetime] = []
        for ts in timestamps:
            d = ts.date()
            if d not in seen_dates:
                seen_dates.add(d)
                daily.append(ts)
        return daily
