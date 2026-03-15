"""Strategy 2: Multi-Outcome Relative Value and Sum Arbitrage.

Thesis: In neg_risk events with N outcomes, prices should sum to ~$1.00.
Deviations create positive-EV trades. Adjacent outcomes in ordinal events
have an implied distribution that should be smooth — kinks are mispricings.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import structlog

from polymir.research.models import PriceSnapshot, ResearchMarket, Signal
from polymir.research.strategy_base import Strategy

logger = structlog.get_logger()


class MultiOutcomeArbStrategy(Strategy):
    name = "multi_outcome_arb"

    def __init__(
        self,
        markets: list[ResearchMarket],
        prices: list[PriceSnapshot],
        deviation_threshold: float = 0.03,
        hold_hours: float = 24.0,
    ) -> None:
        super().__init__(markets, prices)
        self._deviation_threshold = deviation_threshold
        self._hold_hours = hold_hours
        self._event_groups: dict[str, list[ResearchMarket]] = {}
        self._build_event_groups()

    def _build_event_groups(self) -> None:
        """Group markets by event_id for multi-outcome analysis."""
        for m in self._markets:
            if m.neg_risk and m.event_id:
                self._event_groups.setdefault(m.event_id, []).append(m)
        # Only keep events with multiple outcomes
        self._event_groups = {
            eid: markets for eid, markets in self._event_groups.items()
            if len(markets) >= 2
        }

    def generate_signals(self, as_of: datetime) -> list[Signal]:
        signals = []

        for event_id, event_markets in self._event_groups.items():
            # Get current prices for all outcomes
            outcome_prices: list[tuple[ResearchMarket, float]] = []
            for market in event_markets:
                price = self._get_latest_price(market.market_id, as_of)
                if price is not None and price > 0:
                    outcome_prices.append((market, price))

            if len(outcome_prices) < 2:
                continue

            # Check sum deviation
            price_sum = sum(p for _, p in outcome_prices)
            deviation = price_sum - 1.0

            if abs(deviation) < self._deviation_threshold:
                continue

            # Sort by relative mispricing
            n = len(outcome_prices)
            fair_share = 1.0 / n  # naive equal weight as starting point

            for market, price in outcome_prices:
                # Compute how far this outcome is from its "fair" share
                # adjusted by the sum deviation
                adjusted_fair = price * (1.0 / price_sum) if price_sum > 0 else fair_share

                if deviation > self._deviation_threshold:
                    # Sum > 1: outcomes overpriced on average, sell the most overpriced
                    if price > adjusted_fair + 0.02:
                        signals.append(Signal(
                            strategy_name=self.name,
                            market_id=market.market_id,
                            token_id=market.token_ids[0] if market.token_ids else "",
                            direction="SELL",
                            confidence=1.0 - adjusted_fair,
                            entry_price=price,
                            size_fraction=min(abs(deviation), 0.5),
                            metadata={
                                "event_id": event_id,
                                "price_sum": price_sum,
                                "deviation": deviation,
                                "n_outcomes": n,
                                "adjusted_fair": adjusted_fair,
                            },
                        ))
                elif deviation < -self._deviation_threshold:
                    # Sum < 1: outcomes underpriced on average, buy the most underpriced
                    if price < adjusted_fair - 0.02:
                        signals.append(Signal(
                            strategy_name=self.name,
                            market_id=market.market_id,
                            token_id=market.token_ids[0] if market.token_ids else "",
                            direction="BUY",
                            confidence=adjusted_fair,
                            entry_price=price,
                            size_fraction=min(abs(deviation), 0.5),
                            metadata={
                                "event_id": event_id,
                                "price_sum": price_sum,
                                "deviation": deviation,
                                "n_outcomes": n,
                                "adjusted_fair": adjusted_fair,
                            },
                        ))

        return signals

    def get_deviation_history(self, as_of: datetime) -> dict[str, list[tuple[datetime, float]]]:
        """Track sum-of-prices deviation over time for each event."""
        result: dict[str, list[tuple[datetime, float]]] = {}

        for event_id, event_markets in self._event_groups.items():
            # Get all timestamps from any market in the event
            all_ts: set[datetime] = set()
            for m in event_markets:
                for p in self._get_prices_before(m.market_id, as_of):
                    all_ts.add(p.timestamp)

            deviations: list[tuple[datetime, float]] = []
            for ts in sorted(all_ts):
                price_sum = 0.0
                n_found = 0
                for m in event_markets:
                    price = self._get_latest_price(m.market_id, ts)
                    if price is not None:
                        price_sum += price
                        n_found += 1
                if n_found == len(event_markets):
                    deviations.append((ts, price_sum - 1.0))

            if deviations:
                result[event_id] = deviations

        return result
