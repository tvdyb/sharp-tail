"""Data structures for backtesting."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


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
    fee: float = 0.0
    decision: str = "execute"
    wallet_score: float = 0.0
    wallet_rank: int = 0
    market_category: str = ""
    market_resolved_price: float = 0.0
