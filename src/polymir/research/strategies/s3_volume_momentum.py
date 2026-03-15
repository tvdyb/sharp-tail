"""Strategy 3: Volume-Weighted Price Momentum with Reversal Detection.

Thesis: Large volume at stable prices signals informed accumulation.
Large volume with sharp price moves signals uninformed momentum that reverses.
Kyle's lambda (price impact per unit volume) predicts future direction.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import structlog

from polymir.research.models import PriceSnapshot, ResearchMarket, Signal
from polymir.research.strategy_base import Strategy

logger = structlog.get_logger()


class VolumeMomentumStrategy(Strategy):
    name = "volume_momentum"

    def __init__(
        self,
        markets: list[ResearchMarket],
        prices: list[PriceSnapshot],
        volume_window_hours: float = 24.0,
        momentum_threshold: float = 0.10,
        reversal_lookback_hours: float = 48.0,
    ) -> None:
        super().__init__(markets, prices)
        self._volume_window_hours = volume_window_hours
        self._momentum_threshold = momentum_threshold
        self._reversal_lookback_hours = reversal_lookback_hours

    def generate_signals(self, as_of: datetime) -> list[Signal]:
        signals = []
        window = timedelta(hours=self._volume_window_hours)

        for market in self._active_markets_at(as_of):
            history = self._get_prices_before(market.market_id, as_of)
            if len(history) < 3:
                continue

            # Get recent window
            window_start = as_of - window
            recent = [p for p in history if p.timestamp >= window_start]
            if len(recent) < 2:
                continue

            # Price change over window
            price_start = recent[0].price
            price_end = recent[-1].price
            if price_start <= 0:
                continue
            price_change = (price_end - price_start) / price_start

            # Volume proxy: count of price observations (we don't have tick volume)
            volume_proxy = len(recent)

            # Compute lambda estimate: |price_change| / volume_proxy
            abs_change = abs(price_change)
            if volume_proxy > 0:
                kyle_lambda = abs_change / volume_proxy
            else:
                continue

            # Detect momentum (large move on activity)
            if abs_change >= self._momentum_threshold and volume_proxy >= 3:
                # Momentum reversal signal: bet on mean reversion
                direction = "SELL" if price_change > 0 else "BUY"
                # Confidence based on historical mean reversion rates
                confidence = min(0.5 + abs_change * 0.5, 0.75)

                signals.append(Signal(
                    strategy_name=self.name,
                    market_id=market.market_id,
                    token_id=market.token_ids[0] if market.token_ids else "",
                    direction=direction,
                    confidence=confidence,
                    entry_price=price_end,
                    metadata={
                        "signal_type": "momentum_reversal",
                        "price_change": price_change,
                        "volume_proxy": volume_proxy,
                        "kyle_lambda": kyle_lambda,
                    },
                ))

            # Detect informed accumulation (high volume, low price change)
            elif volume_proxy >= 5 and abs_change < 0.03:
                # Price absorbing lots of observation without moving
                # Trade in direction of slight drift
                if abs(price_change) < 0.005:
                    continue  # truly flat, no signal

                direction = "BUY" if price_change > 0 else "SELL"
                confidence = 0.55  # slight edge

                signals.append(Signal(
                    strategy_name=self.name,
                    market_id=market.market_id,
                    token_id=market.token_ids[0] if market.token_ids else "",
                    direction=direction,
                    confidence=confidence,
                    entry_price=price_end,
                    size_fraction=0.5,  # lower sizing for weaker signal
                    metadata={
                        "signal_type": "informed_accumulation",
                        "price_change": price_change,
                        "volume_proxy": volume_proxy,
                        "kyle_lambda": kyle_lambda,
                    },
                ))

        return signals
