"""Tests for backtest engine."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from polymir.backtest import (
    BacktestEngine,
    BacktestResult,
    HistoricalTrade,
)
from polymir.backtest.data import TradeRecord
from polymir.config import AppConfig, ExecutionConfig


def _make_trades(
    count: int = 10,
    price: float = 0.55,
    resolved_price: float = 1.0,
) -> list[HistoricalTrade]:
    now = datetime.utcnow()
    return [
        HistoricalTrade(
            wallet=f"w{i % 3}",
            market_id="m1",
            asset_id="a1",
            side="BUY",
            size=100,
            price=price,
            timestamp=now - timedelta(hours=count - i),
            market_resolved_price=resolved_price,
        )
        for i in range(count)
    ]


class TestBacktestResult:
    def test_empty_result(self):
        r = BacktestResult()
        assert r.total_pnl == 0
        assert r.win_rate == 0
        assert r.sharpe_ratio == 0

    def test_summary_format(self):
        records = [
            TradeRecord(
                timestamp=datetime.utcnow(),
                wallet="w0", market_id="m1", asset_id="a1",
                side="BUY", signal_price=0.55, decision="execute",
                pnl=p, size=100, fill_price=0.55,
            )
            for p in [10, -5, 15, -2, 8]
        ] + [
            TradeRecord(
                timestamp=datetime.utcnow(),
                wallet="w0", market_id="m1", asset_id="a1",
                side="BUY", signal_price=0.55, decision="stale",
            ),
            TradeRecord(
                timestamp=datetime.utcnow(),
                wallet="w0", market_id="m1", asset_id="a1",
                side="BUY", signal_price=0.55, decision="duplicate_market",
            ),
        ]
        r = BacktestResult(trade_records=records)
        s = r.summary()
        assert "Trades executed: 5" in s
        assert "Trades skipped:  2" in s

    def test_win_rate(self):
        records = [
            TradeRecord(
                timestamp=datetime.utcnow(),
                wallet="w0", market_id="m1", asset_id="a1",
                side="BUY", signal_price=0.55, decision="execute",
                pnl=p, size=100, fill_price=0.55,
            )
            for p in [10, -5, 15, -2, 8]
        ]
        r = BacktestResult(trade_records=records)
        assert r.win_rate == pytest.approx(0.6)

    def test_sortino_ratio(self):
        records = [
            TradeRecord(
                timestamp=datetime.utcnow(),
                wallet="w0", market_id="m1", asset_id="a1",
                side="BUY", signal_price=0.55, decision="execute",
                pnl=p, size=100, fill_price=0.55,
            )
            for p in [10, -5, 15, -2, 8]
        ]
        r = BacktestResult(trade_records=records)
        assert r.sortino_ratio != 0

    def test_max_drawdown(self):
        # PnL series: 10, -20, 5 -> cumulative: 10, -10, -5
        # Peak at 10, drop to -10, dd = 20
        records = [
            TradeRecord(
                timestamp=datetime.utcnow(),
                wallet="w0", market_id="m1", asset_id="a1",
                side="BUY", signal_price=0.55, decision="execute",
                pnl=p, size=100, fill_price=0.55,
            )
            for p in [10, -20, 5]
        ]
        r = BacktestResult(trade_records=records)
        assert r.max_drawdown == pytest.approx(20.0)

    def test_monthly_returns(self):
        now = datetime.utcnow()
        records = [
            TradeRecord(
                timestamp=now,
                wallet="w0", market_id="m1", asset_id="a1",
                side="BUY", signal_price=0.55, decision="execute",
                pnl=10.0, size=100, fill_price=0.55,
            ),
        ]
        r = BacktestResult(trade_records=records)
        monthly = r.monthly_returns()
        assert len(monthly) == 1
        assert list(monthly.values())[0] == pytest.approx(10.0)


class TestBacktestEngine:
    @pytest.mark.asyncio
    async def test_basic_run(self):
        config = AppConfig(
            execution=ExecutionConfig(
                max_position_usd=10000,
            )
        )
        engine = BacktestEngine(config, latency_s=30)
        trades = _make_trades(count=5, price=0.55, resolved_price=1.0)
        result = await engine.run(trades=trades)
        assert result.trades_executed > 0
        assert result.total_pnl > 0

    @pytest.mark.asyncio
    async def test_stale_signal_skip(self):
        config = AppConfig(
            execution=ExecutionConfig(stale_signal_timeout_s=10)
        )
        engine = BacktestEngine(config, latency_s=60)
        trades = _make_trades(count=3)
        result = await engine.run(trades=trades)
        assert result.trades_skipped == 3
        assert result.skip_reasons.get("stale") == 3

    @pytest.mark.asyncio
    async def test_date_filter(self):
        now = datetime.utcnow()
        trades = [
            HistoricalTrade(
                wallet="w", market_id="m", asset_id="a",
                side="BUY", size=100, price=0.55,
                timestamp=now - timedelta(days=10),
                market_resolved_price=1.0,
            ),
            HistoricalTrade(
                wallet="w", market_id="m", asset_id="a",
                side="BUY", size=100, price=0.55,
                timestamp=now - timedelta(days=1),
                market_resolved_price=1.0,
            ),
        ]
        config = AppConfig(
            execution=ExecutionConfig(
                max_position_usd=10000,
            )
        )
        engine = BacktestEngine(config, latency_s=30)
        start = (now - timedelta(days=5)).strftime("%Y-%m-%d")
        result = await engine.run(trades=trades, start_date=start)
        assert result.trades_executed + result.trades_skipped <= 1

    @pytest.mark.asyncio
    async def test_losing_trades_negative_pnl(self):
        config = AppConfig(
            execution=ExecutionConfig(
                max_position_usd=10000,
            )
        )
        engine = BacktestEngine(config, latency_s=30)
        trades = _make_trades(count=3, price=0.55, resolved_price=0.0)
        result = await engine.run(trades=trades)
        assert result.trades_executed > 0
        assert result.total_pnl < 0

    @pytest.mark.asyncio
    async def test_fee_deduction(self):
        config = AppConfig(
            execution=ExecutionConfig(
                max_position_usd=10000, fee_rate=0.01,
            )
        )
        engine = BacktestEngine(config, latency_s=30)
        trades = _make_trades(count=1, price=0.55, resolved_price=1.0)
        result = await engine.run(trades=trades)
        assert result.trades_executed == 1
        # With fees, PnL should be less than without
        executed = [t for t in result.trade_records if t.decision == "execute"][0]
        assert executed.fee > 0

    @pytest.mark.asyncio
    async def test_point_in_time_scoring(self):
        """Verify PIT scoring: wallets only qualify based on past-resolved markets."""
        from polymir.scanner import WalletMarketResult
        from polymir.config import ScoringConfig

        now = datetime.utcnow()

        # Market M1 resolved 10 days ago, M2 resolved 5 days ago
        wallet_results = [
            WalletMarketResult(
                wallet="good_wallet", market_id=f"m{i}", won=True, roi=0.5,
                held_to_expiration=True, total_bought=100, total_sold=0,
                resolution_date=now - timedelta(days=10 - i),
            )
            for i in range(5)
        ]

        # Trade from good_wallet 3 days ago (only m0, m1 resolved before trade)
        trades = [
            HistoricalTrade(
                wallet="good_wallet", market_id="m_new", asset_id="a_new",
                side="BUY", size=50, price=0.50,
                timestamp=now - timedelta(days=3),
                market_resolved_price=1.0,
            ),
        ]

        config = AppConfig(
            scoring=ScoringConfig(min_resolved_markets=3),
            execution=ExecutionConfig(
                max_position_usd=10000,
            ),
        )
        engine = BacktestEngine(config, latency_s=30, top_n=10)
        result = await engine.run(trades=trades, wallet_results=wallet_results)

        # With min_resolved_markets=3, the wallet needs 3 markets resolved
        # before the trade timestamp (now - 3 days).
        # Markets m0..m4 resolved at now-10, now-9, ..., now-6 days.
        # All 5 resolved before the trade at now-3 days. So wallet qualifies.
        assert result.trades_executed == 1
