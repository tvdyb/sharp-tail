"""Data structures for backtesting."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from polymir.models import OrderBook, OrderBookLevel


@dataclass
class HistoricalTrade:
    """A historical wallet trade for backtesting."""

    wallet: str
    market_id: str
    asset_id: str
    side: str
    size: float
    price: float
    timestamp: datetime
    market_resolved_price: float = 0.0  # 1.0 if won, 0.0 if lost
    market_resolution_date: datetime | None = None
    market_category: str = ""


@dataclass
class HistoricalOrderbook:
    """Snapshot of orderbook at a point in time."""

    asset_id: str
    timestamp: datetime
    bids: list[tuple[float, float]]  # (price, size)
    asks: list[tuple[float, float]]

    def to_orderbook(self) -> OrderBook:
        return OrderBook(
            asset_id=self.asset_id,
            bids=[OrderBookLevel(price=p, size=s) for p, s in self.bids],
            asks=[OrderBookLevel(price=p, size=s) for p, s in self.asks],
        )


@dataclass
class TradeRecord:
    """Detailed record of a single backtest trade decision."""

    timestamp: datetime
    wallet: str
    market_id: str
    asset_id: str
    side: str
    signal_price: float
    fill_price: float | None = None
    size: float = 0.0
    pnl: float = 0.0
    estimated_slippage: float = 0.0
    realized_slippage: float = 0.0
    fee: float = 0.0
    decision: str = "execute"
    wallet_score: float = 0.0
    wallet_rank: int = 0
    market_category: str = ""
    market_resolved_price: float = 0.0
