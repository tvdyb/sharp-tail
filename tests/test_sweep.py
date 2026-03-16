"""Tests for parameter sweep."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from polymir.backtest import HistoricalTrade
from polymir.backtest.sweep import SweepConfig, build_heatmap_data, run_sweep
from polymir.config import AppConfig, ExecutionConfig, ScoringConfig
from polymir.scanner import WalletMarketResult


class TestSweep:
    @pytest.mark.asyncio
    async def test_basic_sweep(self):
        now = datetime.utcnow()
        trades = [
            HistoricalTrade(
                wallet="w0", market_id=f"m{i}",
                asset_id=f"a{i}", side="BUY",
                size=50, price=0.50,
                timestamp=now - timedelta(hours=i),
                market_resolved_price=1.0,
            )
            for i in range(5)
        ]

        config = AppConfig(
            execution=ExecutionConfig(
                max_position_usd=10000,
            ),
        )
        sweep_config = SweepConfig(
            latencies=[30, 60],
            top_ns=[10],
        )
        results = await run_sweep(
            trades=trades,
            wallet_results=[],
            config=config,
            sweep_config=sweep_config,
        )
        assert len(results) == 2  # 2 latencies * 1 top_n
        for label, result in results.items():
            assert "lat=" in label
            assert result.trades_executed >= 0

    @pytest.mark.asyncio
    async def test_sweep_with_wallet_results(self):
        now = datetime.utcnow()
        trades = [
            HistoricalTrade(
                wallet="w0", market_id=f"m{i}",
                asset_id=f"a{i}", side="BUY",
                size=50, price=0.50,
                timestamp=now - timedelta(hours=i),
                market_resolved_price=1.0,
            )
            for i in range(5)
        ]
        wallet_results = [
            WalletMarketResult(
                wallet="w0", market_id=f"wr{i}",
                won=True, roi=0.08 + 0.01 * i,  # vary ROIs for Sharpe scoring
                held_to_expiration=True, total_bought=100, total_sold=0,
                resolution_date=now - timedelta(days=30 + i),
            )
            for i in range(5)
        ]

        config = AppConfig(
            scoring=ScoringConfig(min_resolved_markets=2),
            execution=ExecutionConfig(
                max_position_usd=10000,
            ),
        )
        sweep_config = SweepConfig(latencies=[30], top_ns=[5, 10])
        results = await run_sweep(
            trades=trades,
            wallet_results=wallet_results,
            config=config,
            sweep_config=sweep_config,
        )
        assert len(results) == 2


class TestHeatmapData:
    def test_build_heatmap(self):
        from polymir.backtest import BacktestResult
        from polymir.backtest.data import TradeRecord

        now = datetime.utcnow()
        results = {}
        for lat in [30, 60]:
            for top in [10, 20]:
                label = f"lat={lat}_top={top}_fee=0.0_pos=1000.0"
                records = [
                    TradeRecord(
                        timestamp=now, wallet="w0", market_id="m0",
                        asset_id="a0", side="BUY", signal_price=0.55,
                        fill_price=0.56, size=100, pnl=10.0,
                        decision="execute", market_resolved_price=1.0,
                    )
                ]
                results[label] = BacktestResult(trade_records=records)

        data = build_heatmap_data(results, x_param="lat", y_param="top")
        assert "x_values" in data
        assert "y_values" in data
        assert "z_matrix" in data
        assert len(data["x_values"]) == 2
        assert len(data["y_values"]) == 2
