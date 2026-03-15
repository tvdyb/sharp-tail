"""Backtest result metrics: Sharpe, Sortino, drawdown, etc."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean, stdev

from polymir.backtest.data import TradeRecord


@dataclass
class BacktestResult:
    """Aggregate results from a backtest run with full metric suite."""

    trade_records: list[TradeRecord] = field(default_factory=list)
    latency_s: int = 60
    top_n: int = 50
    fee_rate: float = 0.0
    config_label: str = ""

    # ── Derived counts ────────────────────────────────────────────

    @property
    def trades_executed(self) -> int:
        return sum(1 for t in self.trade_records if t.decision == "execute")

    @property
    def trades_skipped(self) -> int:
        return sum(1 for t in self.trade_records if t.decision != "execute")

    @property
    def skip_reasons(self) -> dict[str, int]:
        reasons: dict[str, int] = {}
        for t in self.trade_records:
            if t.decision != "execute":
                reasons[t.decision] = reasons.get(t.decision, 0) + 1
        return reasons

    # ── PnL series ────────────────────────────────────────────────

    @property
    def pnl_series(self) -> list[float]:
        return [t.pnl for t in self.trade_records if t.decision == "execute"]

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

    # ── Core metrics ──────────────────────────────────────────────

    @property
    def win_rate(self) -> float:
        pnl = self.pnl_series
        if not pnl:
            return 0.0
        return sum(1 for p in pnl if p > 0) / len(pnl)

    @property
    def avg_profit(self) -> float:
        pnl = self.pnl_series
        return mean(pnl) if pnl else 0.0

    @property
    def median_profit(self) -> float:
        pnl = sorted(self.pnl_series)
        if not pnl:
            return 0.0
        n = len(pnl)
        if n % 2 == 0:
            return (pnl[n // 2 - 1] + pnl[n // 2]) / 2
        return pnl[n // 2]

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio (risk-free rate = 0 for prediction markets)."""
        pnl = self.pnl_series
        if len(pnl) < 2:
            return 0.0
        avg = mean(pnl)
        sd = stdev(pnl)
        if sd == 0:
            return 0.0
        return (avg / sd) * math.sqrt(252)

    @property
    def sortino_ratio(self) -> float:
        """Annualized Sortino ratio (downside deviation only)."""
        pnl = self.pnl_series
        if len(pnl) < 2:
            return 0.0
        avg = mean(pnl)
        downside = [min(p, 0) ** 2 for p in pnl]
        down_dev = math.sqrt(mean(downside)) if downside else 0.0
        if down_dev == 0:
            return 0.0
        return (avg / down_dev) * math.sqrt(252)

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown as a dollar amount."""
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
    def max_drawdown_duration(self) -> int:
        """Duration of max drawdown in number of trades."""
        cum = self.cumulative_pnl
        if not cum:
            return 0
        peak = cum[0]
        max_dur = 0
        current_dur = 0
        for val in cum:
            if val >= peak:
                peak = val
                current_dur = 0
            else:
                current_dur += 1
                max_dur = max(max_dur, current_dur)
        return max_dur

    # ── Slippage stats ────────────────────────────────────────────

    @property
    def slippage_estimates(self) -> list[float]:
        return [t.estimated_slippage for t in self.trade_records if t.decision == "execute"]

    @property
    def slippage_realized(self) -> list[float]:
        return [t.realized_slippage for t in self.trade_records if t.decision == "execute"]

    @property
    def avg_estimated_slippage(self) -> float:
        s = self.slippage_estimates
        return mean(s) if s else 0.0

    @property
    def avg_realized_slippage(self) -> float:
        s = self.slippage_realized
        return mean(s) if s else 0.0

    # ── Time-based breakdowns ─────────────────────────────────────

    def monthly_returns(self) -> dict[str, float]:
        """PnL grouped by month (YYYY-MM)."""
        monthly: dict[str, float] = defaultdict(float)
        for t in self.trade_records:
            if t.decision == "execute":
                key = t.timestamp.strftime("%Y-%m")
                monthly[key] += t.pnl
        return dict(sorted(monthly.items()))

    def weekly_returns(self) -> dict[str, float]:
        """PnL grouped by ISO week (YYYY-WNN)."""
        weekly: dict[str, float] = defaultdict(float)
        for t in self.trade_records:
            if t.decision == "execute":
                iso = t.timestamp.isocalendar()
                key = f"{iso[0]}-W{iso[1]:02d}"
                weekly[key] += t.pnl
        return dict(sorted(weekly.items()))

    # ── Per-category breakdown ────────────────────────────────────

    def pnl_by_category(self) -> dict[str, float]:
        by_cat: dict[str, float] = defaultdict(float)
        for t in self.trade_records:
            if t.decision == "execute":
                by_cat[t.market_category or "unknown"] += t.pnl
        return dict(by_cat)

    # ── Wallet concentration ──────────────────────────────────────

    def pnl_by_wallet(self) -> dict[str, float]:
        by_wallet: dict[str, float] = defaultdict(float)
        for t in self.trade_records:
            if t.decision == "execute":
                by_wallet[t.wallet] += t.pnl
        return dict(sorted(by_wallet.items(), key=lambda x: x[1], reverse=True))

    def top_wallet_concentration(self, n: int = 5) -> float:
        """Fraction of total PnL from top-n wallets."""
        total = self.total_pnl
        if total == 0:
            return 0.0
        by_wallet = self.pnl_by_wallet()
        top_pnl = sum(list(by_wallet.values())[:n])
        return top_pnl / total if total != 0 else 0.0

    # ── Per-market PnL ────────────────────────────────────────────

    def pnl_by_market(self) -> dict[str, float]:
        by_market: dict[str, float] = defaultdict(float)
        for t in self.trade_records:
            if t.decision == "execute":
                by_market[t.market_id] += t.pnl
        return dict(by_market)

    # ── Summary ───────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"=== Backtest Results (latency={self.latency_s}s, top_n={self.top_n}) ===",
            f"Trades executed: {self.trades_executed}",
            f"Trades skipped:  {self.trades_skipped}",
            f"Skip breakdown:  {self.skip_reasons}",
            f"Total PnL:       ${self.total_pnl:,.2f}",
            f"Win rate:        {self.win_rate:.1%}",
            f"Avg profit:      ${self.avg_profit:,.2f}",
            f"Median profit:   ${self.median_profit:,.2f}",
            f"Sharpe ratio:    {self.sharpe_ratio:.2f}",
            f"Sortino ratio:   {self.sortino_ratio:.2f}",
            f"Max drawdown:    ${self.max_drawdown:,.2f}",
            f"Max DD duration: {self.max_drawdown_duration} trades",
            f"Avg est slip:    {self.avg_estimated_slippage:.4f}",
            f"Avg real slip:   {self.avg_realized_slippage:.4f}",
            f"Fee rate:        {self.fee_rate:.4f}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON export."""
        return {
            "latency_s": self.latency_s,
            "top_n": self.top_n,
            "fee_rate": self.fee_rate,
            "config_label": self.config_label,
            "trades_executed": self.trades_executed,
            "trades_skipped": self.trades_skipped,
            "skip_reasons": self.skip_reasons,
            "total_pnl": self.total_pnl,
            "win_rate": self.win_rate,
            "avg_profit": self.avg_profit,
            "median_profit": self.median_profit,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "avg_estimated_slippage": self.avg_estimated_slippage,
            "avg_realized_slippage": self.avg_realized_slippage,
            "monthly_returns": self.monthly_returns(),
            "pnl_by_category": self.pnl_by_category(),
        }
