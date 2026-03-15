"""Strategy 6: Time Decay and Theta Harvesting.

Thesis: Markets near 0.50 with long time to resolution carry implicit "theta" —
the time value of uncertainty. We sell this uncertainty premium by taking positions
in markets where price overstates uncertainty.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import structlog

from polymir.research.models import PriceSnapshot, ResearchMarket, Signal
from polymir.research.strategy_base import Strategy

logger = structlog.get_logger()


class ThetaHarvestingStrategy(Strategy):
    name = "theta_harvesting"

    def __init__(
        self,
        markets: list[ResearchMarket],
        prices: list[PriceSnapshot],
        min_days_at_mid: int = 30,
        mid_range: tuple[float, float] = (0.35, 0.65),
        vol_threshold: float = 0.05,
        holding_period_days: int = 14,
    ) -> None:
        super().__init__(markets, prices)
        self._min_days_at_mid = min_days_at_mid
        self._mid_range = mid_range
        self._vol_threshold = vol_threshold
        self._holding_period_days = holding_period_days

    def _days_at_midrange(
        self, history: list[PriceSnapshot], as_of: datetime
    ) -> float:
        """Count how many days the price has been in the mid-range."""
        if not history:
            return 0.0
        lo, hi = self._mid_range
        consecutive_days = 0.0
        # Walk backward from most recent
        for i in range(len(history) - 1, -1, -1):
            if lo <= history[i].price <= hi:
                if i > 0:
                    dt = (history[i].timestamp - history[i - 1].timestamp).total_seconds() / 86400
                    consecutive_days += min(dt, 7)  # cap gaps
                else:
                    consecutive_days += 1
            else:
                break
        return consecutive_days

    def _recent_volatility(self, history: list[PriceSnapshot], window_days: int = 14) -> float:
        """Compute recent price volatility."""
        if len(history) < 3:
            return float("inf")
        cutoff = history[-1].timestamp - timedelta(days=window_days)
        recent = [p for p in history if p.timestamp >= cutoff]
        if len(recent) < 3:
            return float("inf")
        prices = [p.price for p in recent]
        returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] > 0:
                returns.append(prices[i] - prices[i - 1])
        if len(returns) < 2:
            return float("inf")
        return float(np.std(returns))

    def _estimate_hazard_rate(
        self, category: str, as_of: datetime
    ) -> float:
        """Estimate resolution hazard rate by category from historical data."""
        resolved = [
            m for m in self._resolved_markets()
            if m.category.value == category
            and m.resolution_date and m.resolution_date < as_of
            and m.creation_date
        ]
        if len(resolved) < 5:
            return 0.5  # default prior

        lifetimes = [
            (m.resolution_date - m.creation_date).total_seconds() / 86400  # type: ignore[operator]
            for m in resolved
            if m.creation_date and m.resolution_date
        ]
        if not lifetimes:
            return 0.5
        # Hazard = 1/mean_lifetime (exponential model)
        mean_life = np.mean(lifetimes)
        return 1.0 / max(mean_life, 1.0)

    def generate_signals(self, as_of: datetime) -> list[Signal]:
        signals = []

        for market in self._active_markets_at(as_of):
            history = self._get_prices_before(market.market_id, as_of)
            if len(history) < 5:
                continue

            current_price = history[-1].price
            lo, hi = self._mid_range

            # Check if price is in mid-range
            if not (lo <= current_price <= hi):
                continue

            # Check how long it's been at mid-range
            days_at_mid = self._days_at_midrange(history, as_of)
            if days_at_mid < self._min_days_at_mid:
                continue

            # Check volatility
            vol = self._recent_volatility(history)
            if vol > self._vol_threshold:
                continue

            # Estimate hazard rate
            hazard = self._estimate_hazard_rate(market.category.value, as_of)

            # Time to resolution
            if market.end_date:
                days_to_end = (market.end_date - as_of).total_seconds() / 86400
            else:
                days_to_end = 365  # default for open-ended

            # Theta signal: price has been stable near 0.50, sell the uncertainty premium
            # Direction: bet on continuation (price stays near current level)
            # In practice: buy if slightly above 0.50 (bet it stays YES-ish)
            # sell if slightly below 0.50 (bet it stays NO-ish)
            if current_price >= 0.50:
                direction = "BUY"
                confidence = current_price + 0.02  # slight edge
            else:
                direction = "SELL"
                confidence = 1.0 - current_price + 0.02

            # Size inversely proportional to time-to-resolution (more theta near expiry)
            time_decay_factor = max(0.1, min(1.0, 30.0 / max(days_to_end, 1)))

            signals.append(Signal(
                strategy_name=self.name,
                market_id=market.market_id,
                token_id=market.token_ids[0] if market.token_ids else "",
                direction=direction,
                confidence=confidence,
                entry_price=current_price,
                size_fraction=time_decay_factor * 0.5,
                metadata={
                    "days_at_mid": days_at_mid,
                    "volatility": vol,
                    "hazard_rate": hazard,
                    "days_to_end": days_to_end,
                    "time_decay_factor": time_decay_factor,
                },
            ))

        return signals
