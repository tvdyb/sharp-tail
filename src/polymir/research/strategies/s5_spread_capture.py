"""Strategy 5: Spread-Capture Market Making in Thin Markets.

Thesis: Many PM markets have wide spreads with thin books. By providing
passive liquidity in markets with stable fair prices, you capture the spread
minus adverse selection costs.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import structlog

from polymir.research.models import PriceSnapshot, ResearchMarket, Signal
from polymir.research.strategy_base import Strategy

logger = structlog.get_logger()


class SpreadCaptureStrategy(Strategy):
    name = "spread_capture"

    def __init__(
        self,
        markets: list[ResearchMarket],
        prices: list[PriceSnapshot],
        min_spread: float = 0.05,
        vol_ceiling: float = 0.15,
        min_time_to_resolution_hours: float = 48.0,
        spread_width_frac: float = 1.0,
        inventory_limit: int = 5,
    ) -> None:
        super().__init__(markets, prices)
        self._min_spread = min_spread
        self._vol_ceiling = vol_ceiling
        self._min_time_to_resolution_hours = min_time_to_resolution_hours
        self._spread_width_frac = spread_width_frac
        self._inventory_limit = inventory_limit
        self._inventory: dict[str, int] = {}  # market_id -> net position count

    def _estimate_spread(self, history: list[PriceSnapshot]) -> float | None:
        """Estimate spread from price oscillation patterns."""
        if len(history) < 3:
            return None
        prices = [p.price for p in history[-10:]]
        if len(prices) < 3:
            return None
        # Use range of recent prices as spread proxy
        return max(prices) - min(prices)

    def _compute_realized_vol(self, history: list[PriceSnapshot], window_days: int = 7) -> float:
        """Compute realized volatility from price history."""
        if len(history) < 3:
            return float("inf")
        prices = [p.price for p in history]
        returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] > 0:
                returns.append((prices[i] - prices[i - 1]) / prices[i - 1])
        if len(returns) < 2:
            return float("inf")
        return float(np.std(returns))

    def generate_signals(self, as_of: datetime) -> list[Signal]:
        signals = []

        for market in self._active_markets_at(as_of):
            # Check time to resolution
            if market.end_date:
                hours_to_end = (market.end_date - as_of).total_seconds() / 3600
                if hours_to_end < self._min_time_to_resolution_hours:
                    continue
            else:
                continue

            history = self._get_prices_before(market.market_id, as_of)
            if len(history) < 5:
                continue

            # Estimate spread
            est_spread = self._estimate_spread(history)
            if est_spread is None or est_spread < self._min_spread:
                continue

            # Check realized volatility
            realized_vol = self._compute_realized_vol(history)
            if realized_vol > self._vol_ceiling:
                continue

            # Spread should be significantly wider than volatility
            if est_spread < 3 * realized_vol:
                continue

            # Check inventory limit
            current_inventory = abs(self._inventory.get(market.market_id, 0))
            if current_inventory >= self._inventory_limit:
                continue

            current_price = history[-1].price
            if current_price <= 0.05 or current_price >= 0.95:
                continue

            half_spread = (est_spread * self._spread_width_frac) / 2

            # Market making: place bid and ask around midpoint
            bid_price = current_price - half_spread
            ask_price = current_price + half_spread

            if bid_price > 0.02:
                signals.append(Signal(
                    strategy_name=self.name,
                    market_id=market.market_id,
                    token_id=market.token_ids[0] if market.token_ids else "",
                    direction="BUY",
                    confidence=current_price + 0.01,  # slight edge from spread
                    entry_price=bid_price,
                    size_fraction=0.3,
                    metadata={
                        "signal_type": "market_making_bid",
                        "est_spread": est_spread,
                        "realized_vol": realized_vol,
                        "spread_vol_ratio": est_spread / max(realized_vol, 1e-6),
                        "hours_to_end": hours_to_end,
                        "midpoint": current_price,
                    },
                ))

            if ask_price < 0.98:
                signals.append(Signal(
                    strategy_name=self.name,
                    market_id=market.market_id,
                    token_id=market.token_ids[0] if market.token_ids else "",
                    direction="SELL",
                    confidence=1.0 - current_price + 0.01,
                    entry_price=ask_price,
                    size_fraction=0.3,
                    metadata={
                        "signal_type": "market_making_ask",
                        "est_spread": est_spread,
                        "realized_vol": realized_vol,
                        "spread_vol_ratio": est_spread / max(realized_vol, 1e-6),
                        "hours_to_end": hours_to_end,
                        "midpoint": current_price,
                    },
                ))

        return signals
