"""Models for the alpha research platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any


class MarketCategory(str, Enum):
    POLITICS = "politics"
    SPORTS = "sports"
    CRYPTO = "crypto"
    WEATHER = "weather"
    CULTURE = "culture"
    ECONOMICS = "economics"
    SCIENCE = "science"
    OTHER = "other"


@dataclass
class ResearchMarket:
    """Enriched market data for research."""

    market_id: str
    condition_id: str
    question: str
    slug: str
    category: MarketCategory
    creation_date: datetime | None
    end_date: datetime | None
    resolution_date: datetime | None
    outcome: str  # "Yes", "No", or ""
    total_volume: float
    liquidity: float
    token_ids: list[str]
    neg_risk: bool
    event_id: str
    status: str

    # Derived fields
    total_lifetime_days: float = 0.0
    first_trade_timestamp: datetime | None = None


@dataclass
class PriceSnapshot:
    """A single price observation for a market token."""

    market_id: str
    token_id: str
    timestamp: datetime
    price: float
    volume_bucket: float = 0.0


@dataclass
class Signal:
    """A trading signal from a strategy."""

    strategy_name: str
    market_id: str
    token_id: str
    direction: str  # "BUY" or "SELL"
    confidence: float  # 0 to 1
    entry_price: float
    target_price: float | None = None
    stop_price: float | None = None
    size_fraction: float = 1.0  # fraction of Kelly
    metadata: dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Trade:
    """A simulated trade in the backtest."""

    signal: Signal
    entry_time: datetime
    exit_time: datetime | None = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    size_usd: float = 0.0
    pnl: float = 0.0
    fees: float = 0.0
    net_pnl: float = 0.0
    holding_period_hours: float = 0.0
    slippage: float = 0.0


@dataclass
class StrategyResult:
    """Result of backtesting a single strategy."""

    strategy_name: str
    trades: list[Trade] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    start_date: date | None = None
    end_date: date | None = None

    @property
    def pnl_series(self) -> list[float]:
        return [t.net_pnl for t in self.trades]

    @property
    def cumulative_pnl(self) -> list[float]:
        cum = []
        total = 0.0
        for p in self.pnl_series:
            total += p
            cum.append(total)
        return cum

    @property
    def total_pnl(self) -> float:
        return sum(self.pnl_series)

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.net_pnl > 0) / len(self.trades)

    @property
    def avg_winner(self) -> float:
        winners = [t.net_pnl for t in self.trades if t.net_pnl > 0]
        return sum(winners) / len(winners) if winners else 0.0

    @property
    def avg_loser(self) -> float:
        losers = [t.net_pnl for t in self.trades if t.net_pnl <= 0]
        return sum(losers) / len(losers) if losers else 0.0

    @property
    def avg_holding_hours(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.holding_period_hours for t in self.trades) / len(self.trades)

    @property
    def sharpe_ratio(self) -> float:
        import math
        from statistics import mean, stdev

        pnl = self.pnl_series
        if len(pnl) < 2:
            return 0.0
        avg = mean(pnl)
        sd = stdev(pnl)
        if sd == 0:
            return 0.0
        # Estimate trades per year from actual entry timestamps
        if len(self.trades) >= 2:
            sorted_trades = sorted(self.trades, key=lambda t: t.entry_time)
            span = (sorted_trades[-1].entry_time - sorted_trades[0].entry_time).total_seconds()
            if span > 0:
                trades_per_year = len(self.trades) / (span / (365.25 * 86400))
                return (avg / sd) * math.sqrt(trades_per_year)
        return (avg / sd) * math.sqrt(252)  # fallback

    @property
    def sortino_ratio(self) -> float:
        import math
        from statistics import mean

        pnl = self.pnl_series
        if len(pnl) < 2:
            return 0.0
        avg = mean(pnl)
        downside = [min(p, 0) ** 2 for p in pnl]
        down_dev = math.sqrt(sum(downside) / len(downside))
        if down_dev == 0:
            return 0.0
        return (avg / down_dev) * math.sqrt(252)

    @property
    def calmar_ratio(self) -> float:
        dd = self.max_drawdown
        if dd == 0:
            return 0.0
        total = self.total_pnl
        return total / dd

    @property
    def max_drawdown(self) -> float:
        cum = self.cumulative_pnl
        if not cum:
            return 0.0
        peak = cum[0]
        max_dd = 0.0
        for val in cum:
            peak = max(peak, val)
            dd = peak - val
            max_dd = max(max_dd, dd)
        return max_dd

    @property
    def max_drawdown_pct(self) -> float:
        cum = self.cumulative_pnl
        if not cum:
            return 0.0
        capital = 10_000.0  # starting capital
        peak = capital + cum[0]
        max_dd_pct = 0.0
        for val in cum:
            current = capital + val
            peak = max(peak, current)
            if peak > 0:
                dd_pct = (peak - current) / peak
                max_dd_pct = max(max_dd_pct, dd_pct)
        return max_dd_pct

    @property
    def return_skewness(self) -> float:
        from scipy.stats import skew
        pnl = self.pnl_series
        if len(pnl) < 3:
            return 0.0
        return float(skew(pnl))

    @property
    def return_kurtosis(self) -> float:
        from scipy.stats import kurtosis
        pnl = self.pnl_series
        if len(pnl) < 4:
            return 3.0
        return float(kurtosis(pnl, fisher=False))

    def metrics_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy_name,
            "total_pnl": self.total_pnl,
            "trade_count": self.trade_count,
            "win_rate": self.win_rate,
            "avg_winner": self.avg_winner,
            "avg_loser": self.avg_loser,
            "sharpe": self.sharpe_ratio,
            "sortino": self.sortino_ratio,
            "calmar": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "avg_holding_hours": self.avg_holding_hours,
            "skewness": self.return_skewness,
            "kurtosis": self.return_kurtosis,
        }


@dataclass
class DataQualityReport:
    """Report on data quality metrics."""

    total_markets: int = 0
    markets_by_category: dict[str, int] = field(default_factory=dict)
    markets_by_year: dict[int, int] = field(default_factory=dict)
    markets_by_status: dict[str, int] = field(default_factory=dict)
    total_price_observations: int = 0
    markets_excluded_low_volume: int = 0
    markets_excluded_few_observations: int = 0
    volume_distribution: dict[str, float] = field(default_factory=dict)
    lifetime_distribution: dict[str, float] = field(default_factory=dict)
    observations_per_market: dict[str, float] = field(default_factory=dict)
