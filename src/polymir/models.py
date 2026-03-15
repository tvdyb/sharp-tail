"""Pydantic models for Polymarket API responses and internal data."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── API Response Models ──────────────────────────────────────────────


class MarketOutcome(str, Enum):
    YES = "Yes"
    NO = "No"


class MarketStatus(str, Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    CLOSED = "closed"


class Market(BaseModel):
    """Market metadata from Gamma API."""

    condition_id: str
    question: str
    slug: str = ""
    status: str = ""
    outcome: str = ""
    end_date: datetime | None = None
    resolution_date: datetime | None = None
    tokens: list[Token] = Field(default_factory=list)
    volume: float = 0.0
    liquidity: float = 0.0

    @property
    def is_resolved(self) -> bool:
        return self.status == MarketStatus.RESOLVED


class Token(BaseModel):
    """A tradeable token (YES or NO outcome) within a market."""

    token_id: str
    outcome: str = ""
    price: float = 0.0
    winner: bool = False


# Allow Market to reference Token via forward ref
Market.model_rebuild()


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class Trade(BaseModel):
    """A single trade from the CLOB API."""

    id: str = ""
    taker_order_id: str = ""
    market: str = ""
    asset_id: str = ""
    side: str = ""
    size: float = 0.0
    price: float = 0.0
    timestamp: datetime | None = None
    owner: str = ""  # wallet address


class OrderBookLevel(BaseModel):
    """A single price level in the order book."""

    price: float
    size: float


class OrderBook(BaseModel):
    """Order book snapshot for a token."""

    asset_id: str = ""
    bids: list[OrderBookLevel] = Field(default_factory=list)
    asks: list[OrderBookLevel] = Field(default_factory=list)

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def midpoint(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_pct(self) -> float | None:
        mid = self.midpoint
        sprd = self.spread
        if mid and sprd is not None and mid > 0:
            return sprd / mid
        return None

    def total_liquidity(self, side: str = "both") -> float:
        """Total size * price across all levels."""
        total = 0.0
        if side in ("both", "bid"):
            total += sum(l.price * l.size for l in self.bids)
        if side in ("both", "ask"):
            total += sum(l.price * l.size for l in self.asks)
        return total

    def estimate_fill_price(self, size: float, side: str) -> float | None:
        """Estimate average fill price for a given order size.

        Args:
            size: Number of contracts to fill.
            side: "BUY" uses asks, "SELL" uses bids.

        Returns:
            Weighted average fill price, or None if insufficient liquidity.
        """
        levels = self.asks if side == "BUY" else self.bids
        remaining = size
        cost = 0.0
        for level in levels:
            fill = min(remaining, level.size)
            cost += fill * level.price
            remaining -= fill
            if remaining <= 0:
                break
        if remaining > 0:
            return None  # insufficient liquidity
        return cost / size if size > 0 else None


# ── Internal Models ──────────────────────────────────────────────────


class WalletScore(BaseModel):
    """Computed score for a wallet."""

    address: str
    win_rate: float = 0.0
    avg_roi: float = 0.0
    consistency: float = 0.0
    recency_score: float = 0.0
    hold_ratio: float = 0.0
    resolved_market_count: int = 0
    composite_score: float = 0.0
    scored_at: datetime = Field(default_factory=datetime.utcnow)


class TradeSignal(BaseModel):
    """Signal emitted when a tracked wallet enters a new position."""

    wallet_address: str
    market_id: str
    asset_id: str
    side: str
    size: float
    price: float
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class MirrorDecision(str, Enum):
    EXECUTE = "execute"
    SKIP_SLIPPAGE = "skip_slippage"
    SKIP_SPREAD = "skip_spread"
    SKIP_LIQUIDITY = "skip_liquidity"
    SKIP_STALE = "skip_stale"
    SKIP_FILL_TIMEOUT = "skip_fill_timeout"
    ERROR = "error"


class MirrorTrade(BaseModel):
    """Record of a mirror trade decision."""

    signal: TradeSignal
    decision: MirrorDecision
    order_price: float | None = None
    order_size: float | None = None
    estimated_slippage: float | None = None
    realized_slippage: float | None = None
    fill_price: float | None = None
    fill_size: float | None = None
    order_id: str | None = None
    executed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
