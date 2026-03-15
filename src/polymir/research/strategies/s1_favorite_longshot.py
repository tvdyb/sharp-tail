"""Strategy 1: Favorite-Longshot Bias (Terminal Convergence Mispricing).

Thesis: Prediction markets exhibit the favorite-longshot bias — longshots are
systematically overpriced and favorites underpriced. We build a calibration curve
and trade when realized frequency diverges from implied probability.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import structlog

from polymir.research.models import PriceSnapshot, ResearchMarket, Signal
from polymir.research.strategy_base import Strategy

logger = structlog.get_logger()


class FavoriteLongshotStrategy(Strategy):
    name = "favorite_longshot"

    def __init__(
        self,
        markets: list[ResearchMarket],
        prices: list[PriceSnapshot],
        entry_hours: float = 72.0,
        price_bins: int = 10,
        min_bin_count: int = 10,
        category_filter: str | None = None,
    ) -> None:
        super().__init__(markets, prices)
        self._entry_hours = entry_hours
        self._price_bins = price_bins
        self._min_bin_count = min_bin_count
        self._category_filter = category_filter
        self._calibration: dict[int, tuple[float, float, int]] | None = None

    def _build_calibration(self, as_of: datetime) -> dict[int, tuple[float, float, int]]:
        """Build calibration curve from resolved markets before as_of.

        Returns: bin_index -> (implied_prob_midpoint, realized_frequency, count)
        """
        bins: dict[int, list[bool]] = {i: [] for i in range(self._price_bins)}
        bin_width = 1.0 / self._price_bins

        for market in self._resolved_markets():
            if market.resolution_date and market.resolution_date >= as_of:
                continue
            if self._category_filter and market.category.value != self._category_filter:
                continue

            resolved_yes = market.outcome.lower() in ("yes", "1", "true")
            history = self._get_prices_before(market.market_id, as_of)
            if not history:
                continue

            # Get price near resolution (within entry window)
            if market.resolution_date:
                cutoff = market.resolution_date - timedelta(hours=self._entry_hours)
                late_prices = [p for p in history if p.timestamp >= cutoff]
                if not late_prices:
                    continue
                price = late_prices[0].price
            else:
                continue

            if price <= 0 or price >= 1:
                continue

            bin_idx = min(int(price / bin_width), self._price_bins - 1)
            bins[bin_idx].append(resolved_yes)

        result = {}
        for bin_idx, outcomes in bins.items():
            if len(outcomes) >= self._min_bin_count:
                implied_mid = (bin_idx + 0.5) * bin_width
                realized = sum(outcomes) / len(outcomes)
                result[bin_idx] = (implied_mid, realized, len(outcomes))

        return result

    def generate_signals(self, as_of: datetime) -> list[Signal]:
        """Generate signals based on calibration curve mispricing."""
        # Rebuild calibration periodically (use cached if recent)
        self._calibration = self._build_calibration(as_of)
        if not self._calibration:
            return []

        signals = []
        bin_width = 1.0 / self._price_bins

        for market in self._active_markets_at(as_of):
            if self._category_filter and market.category.value != self._category_filter:
                continue

            # Check if market is within entry window of resolution
            if market.end_date:
                hours_to_end = (market.end_date - as_of).total_seconds() / 3600
                if hours_to_end > self._entry_hours or hours_to_end < 0:
                    continue
            else:
                continue

            price = self._get_latest_price(market.market_id, as_of)
            if price is None or price <= 0.05 or price >= 0.95:
                continue

            bin_idx = min(int(price / bin_width), self._price_bins - 1)
            cal = self._calibration.get(bin_idx)
            if not cal:
                continue

            implied, realized, count = cal
            edge = realized - price

            if abs(edge) < 0.03:  # minimum 3% edge
                continue

            if edge > 0:
                # Favorites underpriced: buy
                signals.append(Signal(
                    strategy_name=self.name,
                    market_id=market.market_id,
                    token_id=market.token_ids[0] if market.token_ids else "",
                    direction="BUY",
                    confidence=realized,
                    entry_price=price,
                    metadata={
                        "implied": implied,
                        "realized": realized,
                        "edge": edge,
                        "bin_count": count,
                        "hours_to_end": hours_to_end,
                    },
                ))
            else:
                # Longshots overpriced: sell
                signals.append(Signal(
                    strategy_name=self.name,
                    market_id=market.market_id,
                    token_id=market.token_ids[0] if market.token_ids else "",
                    direction="SELL",
                    confidence=1.0 - realized,
                    entry_price=price,
                    metadata={
                        "implied": implied,
                        "realized": realized,
                        "edge": edge,
                        "bin_count": count,
                        "hours_to_end": hours_to_end,
                    },
                ))

        return signals

    def get_calibration_data(self, as_of: datetime) -> dict[str, Any]:
        """Get calibration curve data for plotting."""
        cal = self._build_calibration(as_of)
        return {
            "bins": {
                k: {"implied": v[0], "realized": v[1], "count": v[2]}
                for k, v in cal.items()
            },
            "entry_hours": self._entry_hours,
            "price_bins": self._price_bins,
        }
