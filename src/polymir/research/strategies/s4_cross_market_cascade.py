"""Strategy 4: Cross-Market Information Cascade Lag.

Thesis: When news moves one market, related markets should reprice, but PM
participants update related markets slowly, creating a tradeable lead-lag structure.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import structlog

from polymir.research.models import PriceSnapshot, ResearchMarket, Signal
from polymir.research.strategy_base import Strategy

logger = structlog.get_logger()


class CrossMarketCascadeStrategy(Strategy):
    name = "cross_market_cascade"

    def __init__(
        self,
        markets: list[ResearchMarket],
        prices: list[PriceSnapshot],
        move_threshold: float = 0.05,
        lag_hours: float = 6.0,
        min_correlation: float = 0.3,
    ) -> None:
        super().__init__(markets, prices)
        self._move_threshold = move_threshold
        self._lag_hours = lag_hours
        self._min_correlation = min_correlation
        self._relationship_graph: dict[str, list[str]] = {}
        self._build_relationships()

    def _build_relationships(self) -> None:
        """Build market relationship graph based on events, category, and keywords."""
        # Same event siblings
        event_groups: dict[str, list[str]] = {}
        for m in self._markets:
            if m.event_id:
                event_groups.setdefault(m.event_id, []).append(m.market_id)

        for event_id, market_ids in event_groups.items():
            for mid in market_ids:
                siblings = [s for s in market_ids if s != mid]
                self._relationship_graph.setdefault(mid, []).extend(siblings)

        # Same category + overlapping time period
        by_category: dict[str, list[ResearchMarket]] = {}
        for m in self._markets:
            by_category.setdefault(m.category.value, []).append(m)

        for cat, cat_markets in by_category.items():
            for i, m1 in enumerate(cat_markets):
                for m2 in cat_markets[i + 1:]:
                    if self._time_overlap(m1, m2):
                        self._relationship_graph.setdefault(m1.market_id, []).append(m2.market_id)
                        self._relationship_graph.setdefault(m2.market_id, []).append(m1.market_id)

        # Keyword overlap
        for i, m1 in enumerate(self._markets):
            kw1 = self._extract_keywords(m1.question)
            for m2 in self._markets[i + 1:]:
                kw2 = self._extract_keywords(m2.question)
                overlap = kw1 & kw2
                if len(overlap) >= 2:
                    self._relationship_graph.setdefault(m1.market_id, []).append(m2.market_id)
                    self._relationship_graph.setdefault(m2.market_id, []).append(m1.market_id)

        # Deduplicate
        for mid in self._relationship_graph:
            self._relationship_graph[mid] = list(set(self._relationship_graph[mid]))

    @staticmethod
    def _time_overlap(m1: ResearchMarket, m2: ResearchMarket) -> bool:
        """Check if two markets have overlapping active periods."""
        s1 = m1.creation_date or datetime.min
        e1 = m1.resolution_date or m1.end_date or datetime.max
        s2 = m2.creation_date or datetime.min
        e2 = m2.resolution_date or m2.end_date or datetime.max
        return s1 <= e2 and s2 <= e1

    @staticmethod
    def _extract_keywords(text: str) -> set[str]:
        """Extract significant keywords from market question."""
        stop_words = {
            "will", "the", "be", "in", "by", "on", "at", "to", "of", "a", "an",
            "is", "are", "was", "were", "has", "have", "had", "do", "does", "did",
            "this", "that", "it", "for", "with", "from", "or", "and", "not", "yes",
            "no", "before", "after", "more", "than", "above", "below", "over",
            "under", "between", "during", "what", "who", "when", "where", "how",
        }
        words = set(re.findall(r"\b[a-z]+\b", text.lower()))
        return words - stop_words

    def generate_signals(self, as_of: datetime) -> list[Signal]:
        signals = []
        window = timedelta(hours=self._lag_hours)

        for market in self._active_markets_at(as_of):
            history = self._get_prices_before(market.market_id, as_of)
            if len(history) < 2:
                continue

            # Check for recent significant move
            recent_start = as_of - window
            recent = [p for p in history if p.timestamp >= recent_start]
            if len(recent) < 2:
                continue

            price_before = recent[0].price
            price_now = recent[-1].price
            if price_before <= 0:
                continue
            move = (price_now - price_before) / price_before

            if abs(move) < self._move_threshold:
                continue

            # This market moved significantly — check related markets
            related_ids = self._relationship_graph.get(market.market_id, [])
            for related_id in related_ids:
                related = self._market_map.get(related_id)
                if not related:
                    continue

                # Only signal active related markets
                active = self._active_markets_at(as_of)
                if related not in active:
                    continue

                related_history = self._get_prices_before(related_id, as_of)
                if len(related_history) < 2:
                    continue

                # Check if related has already repriced
                related_recent = [p for p in related_history if p.timestamp >= recent_start]
                if len(related_recent) < 2:
                    related_price = related_history[-1].price
                    related_move = 0.0
                else:
                    related_price = related_recent[-1].price
                    if related_recent[0].price > 0:
                        related_move = (related_price - related_recent[0].price) / related_recent[0].price
                    else:
                        continue

                # If related hasn't moved proportionally, trade it
                expected_move = move * self._min_correlation  # conservative estimate
                gap = expected_move - related_move

                if abs(gap) < 0.02:
                    continue

                direction = "BUY" if gap > 0 else "SELL"
                confidence = min(0.5 + abs(gap), 0.7)

                signals.append(Signal(
                    strategy_name=self.name,
                    market_id=related_id,
                    token_id=related.token_ids[0] if related.token_ids else "",
                    direction=direction,
                    confidence=confidence,
                    entry_price=related_price,
                    size_fraction=0.5,
                    metadata={
                        "primary_market": market.market_id,
                        "primary_move": move,
                        "related_move": related_move,
                        "expected_move": expected_move,
                        "gap": gap,
                    },
                ))

        return signals
