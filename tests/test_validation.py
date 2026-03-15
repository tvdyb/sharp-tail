"""Tests for statistical validation functions."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from polymir.backtest import BacktestResult, HistoricalTrade
from polymir.backtest.data import TradeRecord
from polymir.backtest.validation import (
    bootstrap_confidence_intervals,
    deflated_sharpe_ratio,
    holm_bonferroni_correction,
    in_out_sample_split,
    minimum_track_record_length,
    randomized_baseline,
    walk_forward,
)
from polymir.config import AppConfig, ExecutionConfig, ScoringConfig
from polymir.scanner import WalletMarketResult


def _now():
    return datetime.utcnow()


def _make_trade_records(count: int = 20, base_pnl: float = 5.0) -> list[TradeRecord]:
    now = _now()
    return [
        TradeRecord(
            timestamp=now - timedelta(hours=count - i),
            wallet=f"w{i % 3}",
            market_id=f"m{i}",
            asset_id=f"a{i}",
            side="BUY",
            signal_price=0.55,
            fill_price=0.56,
            size=100,
            pnl=base_pnl if i % 3 != 0 else -base_pnl * 0.5,
            decision="execute",
            market_resolved_price=1.0,
        )
        for i in range(count)
    ]


class TestBootstrapCI:
    def test_positive_returns(self):
        records = _make_trade_records(50, base_pnl=10.0)
        result = BacktestResult(trade_records=records)
        vr = bootstrap_confidence_intervals(result, n_bootstrap=1000, seed=42)
        assert vr.passed  # Lower CI of Sharpe should be positive
        assert "sharpe_ci_95" in vr.details

    def test_negative_returns(self):
        records = _make_trade_records(50, base_pnl=-10.0)
        result = BacktestResult(trade_records=records)
        vr = bootstrap_confidence_intervals(result, n_bootstrap=1000, seed=42)
        assert not vr.passed  # Negative returns -> CI includes 0

    def test_insufficient_trades(self):
        records = _make_trade_records(2, base_pnl=10.0)
        result = BacktestResult(trade_records=records)
        vr = bootstrap_confidence_intervals(result, n_bootstrap=100, seed=42)
        assert vr.passed  # Insufficient data -> passes (conservative)


class TestDeflatedSharpe:
    def test_high_sharpe_few_strategies(self):
        vr = deflated_sharpe_ratio(
            observed_sharpe=2.0,
            n_trades=500,
            n_strategies_tested=5,
        )
        assert vr.details["observed_sharpe"] == 2.0
        assert "p_value" in vr.details

    def test_low_sharpe_many_strategies(self):
        vr = deflated_sharpe_ratio(
            observed_sharpe=0.5,
            n_trades=100,
            n_strategies_tested=1000,
        )
        assert not vr.passed  # Should fail: low Sharpe after many tests

    def test_insufficient_data(self):
        vr = deflated_sharpe_ratio(
            observed_sharpe=2.0,
            n_trades=1,
            n_strategies_tested=1,
        )
        assert vr.passed  # Insufficient data -> passes


class TestMinTrackRecord:
    def test_sufficient_length(self):
        vr = minimum_track_record_length(
            observed_sharpe=2.0,
            n_trades=500,
        )
        assert vr.passed
        assert vr.details["actual_trades"] >= vr.details["min_trades_needed"]

    def test_insufficient_length(self):
        vr = minimum_track_record_length(
            observed_sharpe=0.3,
            n_trades=10,
        )
        assert not vr.passed

    def test_negative_sharpe(self):
        vr = minimum_track_record_length(
            observed_sharpe=-0.5,
            n_trades=100,
        )
        assert not vr.passed


class TestHolmBonferroni:
    def test_significant_results(self):
        p_values = {"test_a": 0.001, "test_b": 0.01, "test_c": 0.04}
        vr = holm_bonferroni_correction(p_values)
        assert vr.passed  # At least one significant after correction

    def test_no_significant_results(self):
        p_values = {"test_a": 0.5, "test_b": 0.6, "test_c": 0.7}
        vr = holm_bonferroni_correction(p_values)
        assert not vr.passed

    def test_empty(self):
        vr = holm_bonferroni_correction({})
        assert vr.passed


class TestWalkForward:
    @pytest.mark.asyncio
    async def test_basic_walk_forward(self):
        now = _now()
        # Create trades spanning 12 months
        trades = []
        wallet_results = []
        for month in range(12):
            for day in range(5):
                ts = now - timedelta(days=365 - month * 30 - day)
                trades.append(HistoricalTrade(
                    wallet="w0", market_id=f"m{month}_{day}",
                    asset_id=f"a{month}_{day}", side="BUY",
                    size=50, price=0.50, timestamp=ts,
                    market_resolved_price=1.0,
                ))

        for month in range(12):
            res_date = now - timedelta(days=365 - month * 30)
            for w in range(3):
                wallet_results.append(WalletMarketResult(
                    wallet=f"w{w}", market_id=f"m{month}_scored",
                    won=True, roi=0.1,
                    held_to_expiration=True, total_bought=100, total_sold=0,
                    resolution_date=res_date,
                ))

        config = AppConfig(
            scoring=ScoringConfig(min_resolved_markets=2),
            execution=ExecutionConfig(
                max_position_usd=10000,
            ),
        )
        vr = await walk_forward(
            trades=trades,
            wallet_results=wallet_results,
            config=config,
            train_months=4,
            test_months=2,
            latency_s=30,
            top_n=10,
        )
        assert vr.method == "walk_forward"
        assert "windows" in vr.details

    @pytest.mark.asyncio
    async def test_empty_trades(self):
        config = AppConfig()
        vr = await walk_forward(
            trades=[], wallet_results=[], config=config,
        )
        assert vr.passed


class TestInOutSample:
    @pytest.mark.asyncio
    async def test_basic_split(self):
        now = _now()
        trades = [
            HistoricalTrade(
                wallet="w0", market_id=f"m{i}",
                asset_id=f"a{i}", side="BUY",
                size=50, price=0.50,
                timestamp=now - timedelta(days=100 - i),
                market_resolved_price=1.0,
            )
            for i in range(20)
        ]

        wallet_results = [
            WalletMarketResult(
                wallet="w0", market_id=f"wr{i}",
                won=True, roi=0.1,
                held_to_expiration=True, total_bought=100, total_sold=0,
                resolution_date=now - timedelta(days=200 - i * 10),
            )
            for i in range(10)
        ]

        config = AppConfig(
            scoring=ScoringConfig(min_resolved_markets=2),
            execution=ExecutionConfig(
                max_position_usd=10000,
            ),
        )
        vr = await in_out_sample_split(
            trades=trades,
            wallet_results=wallet_results,
            config=config,
            latency_s=30,
            top_n=10,
        )
        assert vr.method == "in_out_sample"
        assert "in_sample_sharpe" in vr.details
        assert "out_of_sample_sharpe" in vr.details


class TestRandomizedBaseline:
    @pytest.mark.asyncio
    async def test_basic_randomized(self):
        now = _now()
        trades = [
            HistoricalTrade(
                wallet="w0", market_id=f"m{i}",
                asset_id=f"a{i}", side="BUY",
                size=50, price=0.50,
                timestamp=now - timedelta(hours=i),
                market_resolved_price=1.0,
            )
            for i in range(10)
        ]

        wallet_results = [
            WalletMarketResult(
                wallet=f"w{w}", market_id=f"wr{i}",
                won=w == 0,  # Only w0 wins
                roi=0.3 if w == 0 else -0.1,
                held_to_expiration=True, total_bought=100, total_sold=0,
                resolution_date=now - timedelta(days=50 + i),
            )
            for w in range(3)
            for i in range(5)
        ]

        config = AppConfig(
            scoring=ScoringConfig(min_resolved_markets=2),
            execution=ExecutionConfig(
                max_position_usd=10000,
            ),
        )
        vr = await randomized_baseline(
            trades=trades,
            wallet_results=wallet_results,
            config=config,
            iterations=20,  # Low for test speed
            latency_s=30,
            top_n=10,
            seed=42,
        )
        assert vr.method == "randomized_baseline"
        assert "p_value" in vr.details
        assert "real_sharpe" in vr.details
