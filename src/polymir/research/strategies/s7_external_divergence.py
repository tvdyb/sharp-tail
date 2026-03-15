"""Strategy 7: External Source Divergence.

Thesis: PM prices reflect PM participants' beliefs, but external forecasting
sources sometimes have better information. When external signals diverge
from PM prices, bet on convergence toward the external source.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import structlog

from polymir.research.models import PriceSnapshot, ResearchMarket, Signal
from polymir.research.strategy_base import Strategy

logger = structlog.get_logger()


class ExternalDivergenceStrategy(Strategy):
    name = "external_divergence"

    def __init__(
        self,
        markets: list[ResearchMarket],
        prices: list[PriceSnapshot],
        divergence_threshold: float = 0.10,
        external_data: dict[str, list[tuple[datetime, float]]] | None = None,
    ) -> None:
        super().__init__(markets, prices)
        self._divergence_threshold = divergence_threshold
        self._external_data = external_data or {}
        # Track historical accuracy of external vs PM
        self._accuracy_by_category: dict[str, float] = {}
        if not self._external_data:
            self._build_synthetic_external()

    def _build_synthetic_external(self) -> None:
        """Build synthetic external signals from resolved market data.

        For backtesting without live external feeds, we simulate external
        signals using price-derived models per category:
        - Crypto: use price momentum as proxy for spot price signal
        - Economics: use cross-market consensus as proxy for FedWatch
        - Other: use historical base rates as prior
        """
        for market in self._resolved_markets():
            history = self._price_map.get(market.market_id, [])
            if len(history) < 5:
                continue

            signals: list[tuple[datetime, float]] = []
            for i in range(2, len(history)):
                # Simple model: external signal = smoothed price + noise reduction
                window = history[max(0, i - 5):i]
                prices = [p.price for p in window]
                smoothed = np.mean(prices)

                # Category-specific adjustments
                if market.category.value == "crypto":
                    # Crypto: momentum signal
                    if len(prices) >= 3:
                        trend = prices[-1] - prices[0]
                        smoothed = np.clip(smoothed + trend * 0.3, 0.01, 0.99)
                elif market.category.value == "economics":
                    # Economics: mean-reverting to historical base rates
                    base_rate = 0.5
                    smoothed = smoothed * 0.7 + base_rate * 0.3

                signals.append((history[i].timestamp, float(smoothed)))

            if signals:
                self._external_data[market.market_id] = signals

        # Compute historical accuracy by category
        self._compute_accuracy()

    def _compute_accuracy(self) -> None:
        """Compute how accurate external signals were vs PM for resolved markets."""
        category_correct: dict[str, list[bool]] = {}

        for market in self._resolved_markets():
            ext = self._external_data.get(market.market_id)
            if not ext:
                continue

            resolved_yes = market.outcome.lower() in ("yes", "1", "true")
            resolution_val = 1.0 if resolved_yes else 0.0

            # Get last external signal before resolution
            if market.resolution_date:
                pre_res = [e for e in ext if e[0] < market.resolution_date]
            else:
                pre_res = ext

            if not pre_res:
                continue

            ext_price = pre_res[-1][1]
            pm_history = self._price_map.get(market.market_id, [])
            pm_prices = [p for p in pm_history if market.resolution_date and p.timestamp < market.resolution_date]
            if not pm_prices:
                continue

            pm_price = pm_prices[-1].price

            # Was external closer to truth?
            ext_err = abs(ext_price - resolution_val)
            pm_err = abs(pm_price - resolution_val)
            ext_better = ext_err < pm_err

            cat = market.category.value
            category_correct.setdefault(cat, []).append(ext_better)

        for cat, results in category_correct.items():
            self._accuracy_by_category[cat] = sum(results) / len(results) if results else 0.5

    def _get_external_price(self, market_id: str, as_of: datetime) -> float | None:
        """Get external signal for a market at a given time (no look-ahead)."""
        ext = self._external_data.get(market_id)
        if not ext:
            return None
        valid = [e for e in ext if e[0] < as_of]
        if not valid:
            return None
        return valid[-1][1]

    def generate_signals(self, as_of: datetime) -> list[Signal]:
        signals = []

        for market in self._active_markets_at(as_of):
            pm_price = self._get_latest_price(market.market_id, as_of)
            if pm_price is None:
                continue

            ext_price = self._get_external_price(market.market_id, as_of)
            if ext_price is None:
                continue

            divergence = ext_price - pm_price

            if abs(divergence) < self._divergence_threshold:
                continue

            # Check if external source is historically accurate for this category
            cat_accuracy = self._accuracy_by_category.get(market.category.value, 0.5)
            if cat_accuracy < 0.52:  # external must be better than coin flip
                continue

            direction = "BUY" if divergence > 0 else "SELL"
            # Confidence weighted by historical accuracy
            confidence = min(cat_accuracy + abs(divergence) * 0.3, 0.80)

            signals.append(Signal(
                strategy_name=self.name,
                market_id=market.market_id,
                token_id=market.token_ids[0] if market.token_ids else "",
                direction=direction,
                confidence=confidence,
                entry_price=pm_price,
                size_fraction=cat_accuracy - 0.5,  # scale by edge
                metadata={
                    "pm_price": pm_price,
                    "ext_price": ext_price,
                    "divergence": divergence,
                    "category": market.category.value,
                    "cat_accuracy": cat_accuracy,
                },
            ))

        return signals
