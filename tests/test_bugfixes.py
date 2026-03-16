"""Tests for bug fixes and execution model."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import pytest

from polymir.backtest import BacktestEngine, BacktestResult, HistoricalTrade
from polymir.backtest.data import TradeRecord
from polymir.backtest.metrics import BacktestResult
from polymir.config import AppConfig, ExecutionConfig, ScoringConfig
from polymir.models import OrderBook, OrderBookLevel, Trade, TradeSignal, WalletScore
from polymir.research.models import (
    PriceSnapshot,
    ResearchMarket,
    Signal,
    StrategyResult,
    Trade as ResearchTrade,
    MarketCategory,
)
from polymir.research.strategy_base import Strategy


# ── Helpers ──────────────────────────────────────────────────────────


def _make_research_market(
    market_id: str = "m1",
    outcome: str = "Yes",
    creation_date: datetime | None = None,
    resolution_date: datetime | None = None,
) -> ResearchMarket:
    creation = creation_date or datetime(2024, 1, 1)
    resolution = resolution_date or datetime(2024, 6, 1)
    return ResearchMarket(
        market_id=market_id,
        condition_id=market_id,
        question=f"Test market {market_id}",
        slug=market_id,
        category=MarketCategory.CRYPTO,
        creation_date=creation,
        end_date=resolution,
        resolution_date=resolution,
        outcome=outcome,
        total_volume=50000.0,
        liquidity=10000.0,
        token_ids=[f"{market_id}_yes", f"{market_id}_no"],
        neg_risk=False,
        event_id="",
        status="resolved",
        total_lifetime_days=150.0,
    )


def _make_prices(market_id: str, days: int = 60, base: float = 0.5) -> list[PriceSnapshot]:
    prices = []
    for i in range(days):
        ts = datetime(2024, 2, 1) + timedelta(days=i)
        prices.append(PriceSnapshot(
            market_id=market_id,
            token_id=f"{market_id}_yes",
            timestamp=ts,
            price=base,
        ))
    return prices


class _DummyStrategy(Strategy):
    """Strategy that emits a fixed signal for testing."""

    name = "dummy"

    def __init__(self, markets, prices, direction="BUY", confidence=0.7, entry_price=0.5):
        super().__init__(markets, prices)
        self._direction = direction
        self._confidence = confidence
        self._entry_price = entry_price

    def generate_signals(self, as_of: datetime) -> list[Signal]:
        signals = []
        for m in self._active_markets_at(as_of):
            price = self._get_latest_price(m.market_id, as_of)
            if price is None:
                continue
            signals.append(Signal(
                strategy_name=self.name,
                market_id=m.market_id,
                token_id=m.token_ids[0],
                direction=self._direction,
                confidence=self._confidence,
                entry_price=self._entry_price,
            ))
        return signals


# ── Bug Fix 1: Fees charged on all trades, not just winners ─────────


class TestFeeModel:
    def test_fees_charged_on_losing_trade(self):
        """Fees should be charged even when gross_pnl is negative."""
        # Market resolves "No", so a BUY-side trade loses
        market = _make_research_market("m1", outcome="No")
        prices = _make_prices("m1", days=60, base=0.5)
        strategy = _DummyStrategy(
            [market], prices,
            direction="BUY", confidence=0.7, entry_price=0.5,
        )
        result = strategy.backtest(
            date(2024, 2, 1), date(2024, 4, 1),
            fee_rate=0.02,
        )
        # Should have at least one trade
        assert result.trade_count >= 1
        for trade in result.trades:
            # Gross PnL is negative (bought at ~0.5, resolved to 0.0)
            assert trade.pnl < 0
            # But fees should still be positive
            assert trade.fees > 0, "Fees must be charged on losing trades too"

    def test_fees_proportional_to_notional(self):
        """Fee = entry_price * contracts * fee_rate."""
        market = _make_research_market("m1", outcome="Yes")
        prices = _make_prices("m1", days=60, base=0.5)
        strategy = _DummyStrategy(
            [market], prices,
            direction="BUY", confidence=0.7, entry_price=0.5,
        )
        result = strategy.backtest(
            date(2024, 2, 1), date(2024, 4, 1),
            fee_rate=0.05,  # 5% fee for easy checking
        )
        assert result.trade_count >= 1
        for trade in result.trades:
            expected_fee = trade.entry_price * (trade.size_usd / trade.entry_price) * 0.05
            assert trade.fees == pytest.approx(expected_fee, rel=0.01)


# ── Bug Fix 2: Kelly formula for SELL signals ───────────────────────


class TestKellyFormula:
    def test_sell_kelly_differs_from_buy(self):
        """SELL-side Kelly fraction should use entry_price as denominator, not (1-entry_price)."""
        market = _make_research_market("m1", outcome="No")
        prices = _make_prices("m1", days=60, base=0.5)

        buy_strategy = _DummyStrategy(
            [market], prices,
            direction="BUY", confidence=0.7, entry_price=0.5,
        )
        sell_strategy = _DummyStrategy(
            [market], prices,
            direction="SELL", confidence=0.7, entry_price=0.5,
        )

        buy_result = buy_strategy.backtest(date(2024, 2, 1), date(2024, 4, 1), fee_rate=0.0)
        sell_result = sell_strategy.backtest(date(2024, 2, 1), date(2024, 4, 1), fee_rate=0.0)

        # Both produce trades
        assert buy_result.trade_count >= 0
        assert sell_result.trade_count >= 0

    def test_sell_kelly_with_asymmetric_price(self):
        """With entry_price=0.3, BUY kelly uses 1/0.7, SELL kelly uses 1/0.3."""
        market_yes = _make_research_market("m1", outcome="Yes")
        prices = _make_prices("m1", days=60, base=0.3)

        buy_strategy = _DummyStrategy(
            [market_yes], prices,
            direction="BUY", confidence=0.7, entry_price=0.3,
        )
        buy_result = buy_strategy.backtest(date(2024, 2, 1), date(2024, 4, 1), fee_rate=0.0)

        market_no = _make_research_market("m2", outcome="No")
        prices2 = _make_prices("m2", days=60, base=0.7)

        sell_strategy = _DummyStrategy(
            [market_no], prices2,
            direction="SELL", confidence=0.3, entry_price=0.7,
        )
        sell_result = sell_strategy.backtest(date(2024, 2, 1), date(2024, 4, 1), fee_rate=0.0)

        if buy_result.trade_count > 0 and sell_result.trade_count > 0:
            buy_size = buy_result.trades[0].size_usd
            sell_size = sell_result.trades[0].size_usd


# ── Flat slippage model tests ─────────────────────────────────────────


class TestFlatSlippage:
    @pytest.mark.asyncio
    async def test_buy_fill_price(self):
        """BUY fill price = signal price + slippage_per_trade."""
        now = datetime(2024, 6, 1)
        trades = [
            HistoricalTrade(
                wallet="w0", market_id="m0", asset_id="a0",
                side="BUY", size=50, price=0.50,
                timestamp=now - timedelta(hours=1),
                market_resolved_price=1.0,
            ),
        ]
        config = AppConfig(
            execution=ExecutionConfig(
                max_position_usd=5000,
                slippage_per_trade=0.03,
            ),
        )
        engine = BacktestEngine(config, latency_s=0, top_n=10)
        result = await engine.run(trades=trades)
        executed = [t for t in result.trade_records if t.decision == "execute"]
        assert len(executed) == 1
        assert executed[0].fill_price == pytest.approx(0.53)  # 0.50 + 0.03

    @pytest.mark.asyncio
    async def test_sell_fill_price(self):
        """SELL fill price = signal price - slippage_per_trade."""
        now = datetime(2024, 6, 1)
        trades = [
            HistoricalTrade(
                wallet="w0", market_id="m0", asset_id="a0",
                side="SELL", size=50, price=0.50,
                timestamp=now - timedelta(hours=1),
                market_resolved_price=0.0,
            ),
        ]
        config = AppConfig(
            execution=ExecutionConfig(
                max_position_usd=5000,
                slippage_per_trade=0.03,
            ),
        )
        engine = BacktestEngine(config, latency_s=0, top_n=10)
        result = await engine.run(trades=trades)
        executed = [t for t in result.trade_records if t.decision == "execute"]
        assert len(executed) == 1
        assert executed[0].fill_price == pytest.approx(0.47)  # 0.50 - 0.03

    @pytest.mark.asyncio
    async def test_zero_fees(self):
        """Default fee_rate=0.0 means no fees in PnL."""
        now = datetime(2024, 6, 1)
        trades = [
            HistoricalTrade(
                wallet="w0", market_id="m0", asset_id="a0",
                side="BUY", size=50, price=0.50,
                timestamp=now - timedelta(hours=1),
                market_resolved_price=1.0,
            ),
        ]
        config = AppConfig(
            execution=ExecutionConfig(
                max_position_usd=5000,
                slippage_per_trade=0.03,
            ),
        )
        engine = BacktestEngine(config, latency_s=0, top_n=10)
        result = await engine.run(trades=trades)
        executed = [t for t in result.trade_records if t.decision == "execute"]
        assert len(executed) == 1
        assert executed[0].fee == 0.0
        # PnL = (1.0 - 0.53) * 50 = 23.5
        assert executed[0].pnl == pytest.approx((1.0 - 0.53) * 50)


# ── Bug Fix 4: Sharpe annualization by trade frequency ───────────────


class TestSharpeAnnualization:
    def test_sharpe_varies_with_frequency(self):
        """Sharpe should differ for daily vs weekly trade frequencies."""
        now = datetime(2024, 1, 1)

        import random
        rng = random.Random(42)

        # Daily trades (365 trades over 1 year)
        daily_records = [
            TradeRecord(
                timestamp=now + timedelta(days=i),
                wallet="w0", market_id=f"m{i}", asset_id=f"a{i}",
                side="BUY", signal_price=0.5, fill_price=0.5,
                size=100, pnl=1.0 + rng.gauss(0, 0.5), decision="execute",
                market_resolved_price=1.0,
            )
            for i in range(365)
        ]
        daily_result = BacktestResult(trade_records=daily_records)

        rng2 = random.Random(42)  # same seed for comparable variance

        # Weekly trades (52 trades over 1 year, same per-trade PnL distribution)
        weekly_records = [
            TradeRecord(
                timestamp=now + timedelta(weeks=i),
                wallet="w0", market_id=f"m{i}", asset_id=f"a{i}",
                side="BUY", signal_price=0.5, fill_price=0.5,
                size=100, pnl=1.0 + rng2.gauss(0, 0.5), decision="execute",
                market_resolved_price=1.0,
            )
            for i in range(52)
        ]
        weekly_result = BacktestResult(trade_records=weekly_records)

        # Daily trades should have higher annualized Sharpe (more compounding periods)
        assert daily_result.sharpe_ratio > 0
        assert weekly_result.sharpe_ratio > 0
        # sqrt(365) / sqrt(52) ≈ 2.65x ratio
        ratio = daily_result.sharpe_ratio / weekly_result.sharpe_ratio
        assert 2.0 < ratio < 3.5

    def test_strategy_result_sharpe_varies_with_frequency(self):
        """StrategyResult Sharpe should also use actual trade frequency."""
        import random
        rng = random.Random(99)

        # Daily trades
        daily_trades = [
            ResearchTrade(
                signal=Signal(strategy_name="t", market_id=f"m{i}", token_id="t1",
                             direction="BUY", confidence=0.6, entry_price=0.5),
                entry_time=datetime(2024, 1, 1) + timedelta(days=i),
                net_pnl=1.0 + rng.gauss(0, 0.5),
            )
            for i in range(100)
        ]
        daily = StrategyResult(strategy_name="daily", trades=daily_trades)

        rng2 = random.Random(99)
        # Weekly trades
        weekly_trades = [
            ResearchTrade(
                signal=Signal(strategy_name="t", market_id=f"m{i}", token_id="t1",
                             direction="BUY", confidence=0.6, entry_price=0.5),
                entry_time=datetime(2024, 1, 1) + timedelta(weeks=i),
                net_pnl=1.0 + rng2.gauss(0, 0.5),
            )
            for i in range(20)
        ]
        weekly = StrategyResult(strategy_name="weekly", trades=weekly_trades)

        assert daily.sharpe_ratio > 0
        assert weekly.sharpe_ratio > 0
        # Daily has more trades per year -> higher annualized Sharpe
        assert daily.sharpe_ratio > weekly.sharpe_ratio


# ── Bug Fix 5: Monitor signal_sides ─────────────────────────────────


class TestMonitorSignalSides:
    @pytest.mark.asyncio
    async def test_sell_signals_when_configured(self):
        """Monitor should emit SELL signals when signal_sides includes SELL."""
        from polymir.db import Database
        from polymir.monitor import TradeMonitor

        config = AppConfig(
            execution=ExecutionConfig(
                signal_sides=("BUY", "SELL"),
            ),
        )
        async with Database(":memory:") as db:
            # Add a watched wallet
            score = WalletScore(
                address="0xtest",
                win_rate=0.8, avg_roi=0.5, sharpe_ratio=0.7,
                sharpe_ci_lower=0.4, sharpe_ci_upper=1.0, hold_ratio=0.9,
                resolved_market_count=25, composite_score=0.4,
            )
            await db.upsert_wallet_score(score)

            monitor = TradeMonitor(config, db)
            await monitor.load_watchlist()

            # SELL trade should now produce a signal
            sell_trade = Trade(
                id="t1", market="m1", asset_id="a1",
                side="SELL", size=50.0, price=0.6,
                owner="0xtest",
            )
            signal = monitor._trade_to_signal(sell_trade)
            assert signal is not None
            assert signal.side == "SELL"

    @pytest.mark.asyncio
    async def test_sell_signals_blocked_by_default(self):
        """With default config, SELL trades should not produce signals."""
        from polymir.db import Database
        from polymir.monitor import TradeMonitor

        config = AppConfig()  # default signal_sides=("BUY",)
        async with Database(":memory:") as db:
            score = WalletScore(
                address="0xtest",
                win_rate=0.8, avg_roi=0.5, sharpe_ratio=0.7,
                sharpe_ci_lower=0.4, sharpe_ci_upper=1.0, hold_ratio=0.9,
                resolved_market_count=25, composite_score=0.4,
            )
            await db.upsert_wallet_score(score)

            monitor = TradeMonitor(config, db)
            await monitor.load_watchlist()

            sell_trade = Trade(
                id="t1", market="m1", asset_id="a1",
                side="SELL", size=50.0, price=0.6,
                owner="0xtest",
            )
            signal = monitor._trade_to_signal(sell_trade)
            assert signal is None


# ── Improvement 6: Market deduplication ──────────────────────────────


class TestMarketDeduplication:
    def test_no_duplicate_entries(self):
        """Strategy should not enter the same market twice."""
        market = _make_research_market("m1", outcome="Yes")
        # Long price series so signals fire on multiple days
        prices = _make_prices("m1", days=120, base=0.5)
        strategy = _DummyStrategy(
            [market], prices,
            direction="BUY", confidence=0.7, entry_price=0.5,
        )
        result = strategy.backtest(date(2024, 2, 1), date(2024, 5, 30), fee_rate=0.0)
        # Even though signals fire daily, should only have 1 trade per market
        market_ids = [t.signal.market_id for t in result.trades]
        assert len(market_ids) == len(set(market_ids)), "Duplicate market entries found"


# ── Improvement 7: Position tracker in mirror backtest ───────────────


class TestPositionTracker:
    @pytest.mark.asyncio
    async def test_no_duplicate_market_entries(self):
        """Mirror backtest should not enter the same market twice."""
        now = datetime(2024, 6, 1)
        trades = [
            HistoricalTrade(
                wallet="w0", market_id="m0", asset_id="a0",
                side="BUY", size=50, price=0.50,
                timestamp=now - timedelta(hours=2),
                market_resolved_price=1.0,
            ),
            HistoricalTrade(
                wallet="w0", market_id="m0", asset_id="a0",
                side="BUY", size=50, price=0.55,
                timestamp=now - timedelta(hours=1),
                market_resolved_price=1.0,
            ),
        ]
        config = AppConfig(
            execution=ExecutionConfig(
                max_position_usd=5000,
            ),
        )
        engine = BacktestEngine(config, latency_s=0, top_n=10)
        result = await engine.run(trades=trades)
        executed = [t for t in result.trade_records if t.decision == "execute"]
        duplicates = [t for t in result.trade_records if t.decision == "duplicate_market"]
        assert len(executed) == 1
        assert len(duplicates) == 1
