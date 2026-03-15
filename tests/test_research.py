"""Tests for the alpha research platform."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from polymir.data.collector import DataCollector, categorize_market
from polymir.research.models import (
    DataQualityReport,
    MarketCategory,
    PriceSnapshot,
    ResearchMarket,
    Signal,
    StrategyResult,
    Trade,
)
from polymir.research.engine import ResearchEngine, _param_combinations
from polymir.research.strategy_base import Strategy
from polymir.research.strategies.s1_favorite_longshot import FavoriteLongshotStrategy
from polymir.research.strategies.s2_multi_outcome_arb import MultiOutcomeArbStrategy
from polymir.research.strategies.s3_volume_momentum import VolumeMomentumStrategy
from polymir.research.strategies.s4_cross_market_cascade import CrossMarketCascadeStrategy
from polymir.research.strategies.s5_spread_capture import SpreadCaptureStrategy
from polymir.research.strategies.s6_theta_harvesting import ThetaHarvestingStrategy
from polymir.research.strategies.s7_external_divergence import ExternalDivergenceStrategy
from polymir.research.validation import (
    deflated_sharpe,
    naive_benchmark_comparison,
    check_contemporaneous_prices,
    check_no_lookahead,
)


# ── Fixtures ──────────────────────────────────────────────────────


def _make_market(
    market_id: str = "m1",
    question: str = "Will Bitcoin hit 100k?",
    category: MarketCategory = MarketCategory.CRYPTO,
    status: str = "resolved",
    outcome: str = "Yes",
    volume: float = 50000.0,
    neg_risk: bool = False,
    event_id: str = "",
    creation_date: datetime | None = None,
    end_date: datetime | None = None,
    resolution_date: datetime | None = None,
) -> ResearchMarket:
    creation = creation_date or datetime(2024, 1, 1)
    end = end_date or datetime(2024, 6, 1)
    resolution = resolution_date or datetime(2024, 6, 1)
    return ResearchMarket(
        market_id=market_id,
        condition_id=market_id,
        question=question,
        slug=market_id,
        category=category,
        creation_date=creation,
        end_date=end,
        resolution_date=resolution,
        outcome=outcome,
        total_volume=volume,
        liquidity=10000.0,
        token_ids=[f"{market_id}_yes", f"{market_id}_no"],
        neg_risk=neg_risk,
        event_id=event_id,
        status=status,
        total_lifetime_days=150.0,
    )


def _make_prices(
    market_id: str = "m1",
    start: datetime | None = None,
    days: int = 30,
    base_price: float = 0.5,
    drift: float = 0.01,
) -> list[PriceSnapshot]:
    start = start or datetime(2024, 3, 1)
    prices = []
    price = base_price
    for i in range(days):
        ts = start + timedelta(days=i)
        prices.append(PriceSnapshot(
            market_id=market_id,
            token_id=f"{market_id}_yes",
            timestamp=ts,
            price=max(0.01, min(0.99, price)),
        ))
        price += drift
    return prices


def _sample_markets_and_prices(n: int = 10) -> tuple[list[ResearchMarket], list[PriceSnapshot]]:
    markets = []
    prices = []
    for i in range(n):
        mid = f"m{i}"
        outcome = "Yes" if i % 2 == 0 else "No"
        drift = 0.005 if outcome == "Yes" else -0.005
        m = _make_market(market_id=mid, outcome=outcome)
        markets.append(m)
        prices.extend(_make_prices(
            market_id=mid,
            start=datetime(2024, 1, 1),
            days=150,
            base_price=0.4 + (i * 0.05),
            drift=drift,
        ))
    return markets, prices


# ── Categorization Tests ──────────────────────────────────────────


class TestCategorization:
    def test_politics(self) -> None:
        assert categorize_market("Will Trump win the 2024 election?", "trump-2024") == MarketCategory.POLITICS

    def test_crypto(self) -> None:
        assert categorize_market("Will Bitcoin exceed $100,000?", "btc-100k") == MarketCategory.CRYPTO

    def test_economics(self) -> None:
        assert categorize_market("Will the Fed cut rates in March?", "fed-march") == MarketCategory.ECONOMICS

    def test_sports(self) -> None:
        assert categorize_market("Who will win the Super Bowl?", "super-bowl") == MarketCategory.SPORTS

    def test_weather(self) -> None:
        assert categorize_market("Will there be a hurricane in Florida?", "hurricane-fl") == MarketCategory.WEATHER

    def test_unknown(self) -> None:
        assert categorize_market("Random nonsense question xyz", "xyz") == MarketCategory.OTHER


# ── Model Tests ───────────────────────────────────────────────────


class TestModels:
    def test_strategy_result_empty(self) -> None:
        result = StrategyResult(strategy_name="test")
        assert result.total_pnl == 0.0
        assert result.trade_count == 0
        assert result.sharpe_ratio == 0.0
        assert result.win_rate == 0.0

    def test_strategy_result_with_trades(self) -> None:
        trades = [
            Trade(
                signal=Signal(strategy_name="t", market_id="m1", token_id="t1",
                             direction="BUY", confidence=0.6, entry_price=0.5),
                entry_time=datetime(2024, 1, 1),
                net_pnl=100.0,
                holding_period_hours=24.0,
            ),
            Trade(
                signal=Signal(strategy_name="t", market_id="m2", token_id="t2",
                             direction="BUY", confidence=0.7, entry_price=0.4),
                entry_time=datetime(2024, 1, 2),
                net_pnl=-50.0,
                holding_period_hours=48.0,
            ),
        ]
        result = StrategyResult(strategy_name="test", trades=trades)
        assert result.total_pnl == 50.0
        assert result.trade_count == 2
        assert result.win_rate == 0.5
        assert result.avg_winner == 100.0
        assert result.avg_loser == -50.0

    def test_data_quality_report(self) -> None:
        report = DataQualityReport(total_markets=100)
        assert report.total_markets == 100
        assert report.total_price_observations == 0


# ── Strategy Tests ────────────────────────────────────────────────


class TestFavoriteLongshot:
    def test_generates_signals(self) -> None:
        markets, prices = _sample_markets_and_prices(20)
        strategy = FavoriteLongshotStrategy(
            markets=markets, prices=prices,
            entry_hours=2400, price_bins=5, min_bin_count=2,
        )
        # Generate signals at a time when markets should be near resolution
        signals = strategy.generate_signals(datetime(2024, 5, 30))
        # May or may not generate signals depending on calibration
        assert isinstance(signals, list)

    def test_backtest_runs(self) -> None:
        markets, prices = _sample_markets_and_prices(20)
        strategy = FavoriteLongshotStrategy(
            markets=markets, prices=prices,
            entry_hours=2400, price_bins=5, min_bin_count=2,
        )
        result = strategy.backtest(date(2024, 1, 1), date(2024, 6, 1))
        assert isinstance(result, StrategyResult)
        assert result.strategy_name == "favorite_longshot"


class TestMultiOutcomeArb:
    def test_groups_events(self) -> None:
        markets = [
            _make_market("a1", neg_risk=True, event_id="e1"),
            _make_market("a2", neg_risk=True, event_id="e1"),
            _make_market("a3", neg_risk=True, event_id="e1"),
            _make_market("b1", neg_risk=False, event_id="e2"),
        ]
        prices = []
        for m in markets:
            prices.extend(_make_prices(m.market_id, base_price=0.3))
        strategy = MultiOutcomeArbStrategy(markets=markets, prices=prices)
        assert "e1" in strategy._event_groups
        assert len(strategy._event_groups["e1"]) == 3
        assert "e2" not in strategy._event_groups


class TestVolumeMomentum:
    def test_backtest(self) -> None:
        markets, prices = _sample_markets_and_prices(10)
        strategy = VolumeMomentumStrategy(markets=markets, prices=prices)
        result = strategy.backtest(date(2024, 1, 1), date(2024, 6, 1))
        assert isinstance(result, StrategyResult)


class TestCrossMarketCascade:
    def test_builds_relationships(self) -> None:
        markets = [
            _make_market("m1", question="Will Trump win?", event_id="e1"),
            _make_market("m2", question="Will Trump lose?", event_id="e1"),
            _make_market("m3", question="Something unrelated xyz"),
        ]
        prices = []
        for m in markets:
            prices.extend(_make_prices(m.market_id))
        strategy = CrossMarketCascadeStrategy(markets=markets, prices=prices)
        # m1 and m2 share event_id and keyword "trump"
        assert "m2" in strategy._relationship_graph.get("m1", [])


class TestSpreadCapture:
    def test_backtest(self) -> None:
        markets, prices = _sample_markets_and_prices(5)
        strategy = SpreadCaptureStrategy(markets=markets, prices=prices)
        result = strategy.backtest(date(2024, 1, 1), date(2024, 6, 1))
        assert isinstance(result, StrategyResult)


class TestThetaHarvesting:
    def test_backtest(self) -> None:
        markets, prices = _sample_markets_and_prices(5)
        strategy = ThetaHarvestingStrategy(
            markets=markets, prices=prices,
            min_days_at_mid=5,  # lower threshold for test data
        )
        result = strategy.backtest(date(2024, 1, 1), date(2024, 6, 1))
        assert isinstance(result, StrategyResult)


class TestExternalDivergence:
    def test_builds_synthetic(self) -> None:
        markets, prices = _sample_markets_and_prices(5)
        strategy = ExternalDivergenceStrategy(markets=markets, prices=prices)
        assert len(strategy._external_data) > 0

    def test_backtest(self) -> None:
        markets, prices = _sample_markets_and_prices(5)
        strategy = ExternalDivergenceStrategy(markets=markets, prices=prices)
        result = strategy.backtest(date(2024, 1, 1), date(2024, 6, 1))
        assert isinstance(result, StrategyResult)


# ── Engine Tests ──────────────────────────────────────────────────


class TestResearchEngine:
    def test_param_combinations(self) -> None:
        grid = {"a": [1, 2], "b": [10, 20]}
        combos = _param_combinations(grid)
        assert len(combos) == 4
        assert {"a": 1, "b": 10} in combos

    def test_correlation_matrix(self) -> None:
        r1 = StrategyResult(strategy_name="s1", trades=[
            Trade(signal=Signal(strategy_name="s1", market_id="m1", token_id="t1",
                               direction="BUY", confidence=0.6, entry_price=0.5),
                  entry_time=datetime(2024, 1, i), net_pnl=float(i))
            for i in range(1, 11)
        ])
        r2 = StrategyResult(strategy_name="s2", trades=[
            Trade(signal=Signal(strategy_name="s2", market_id="m1", token_id="t1",
                               direction="BUY", confidence=0.6, entry_price=0.5),
                  entry_time=datetime(2024, 1, i), net_pnl=float(-i))
            for i in range(1, 11)
        ])
        corr = ResearchEngine.correlation_matrix({"s1": r1, "s2": r2})
        assert "s1" in corr
        assert corr["s1"]["s1"] == pytest.approx(1.0, abs=0.01)
        assert corr["s1"]["s2"] == pytest.approx(-1.0, abs=0.01)

    def test_optimal_weights(self) -> None:
        r1 = StrategyResult(strategy_name="s1", trades=[
            Trade(signal=Signal(strategy_name="s1", market_id="m1", token_id="t1",
                               direction="BUY", confidence=0.6, entry_price=0.5),
                  entry_time=datetime(2024, 1, i), net_pnl=10.0)
            for i in range(1, 11)
        ])
        weights = ResearchEngine.optimal_weights({"s1": r1})
        assert weights["s1"] == pytest.approx(1.0)


# ── Validation Tests ──────────────────────────────────────────────


class TestValidation:
    def test_deflated_sharpe(self) -> None:
        result = deflated_sharpe(2.0, 100, 7, 4)
        assert result.test_name == "deflated_sharpe"
        assert "p_value" in result.details

    def test_deflated_sharpe_low(self) -> None:
        result = deflated_sharpe(0.1, 10, 7, 4)
        assert not result.passed  # low Sharpe shouldn't pass

    def test_naive_benchmark(self) -> None:
        trades = [
            Trade(
                signal=Signal(strategy_name="t", market_id=f"m{i}", token_id="t1",
                             direction="BUY", confidence=0.6, entry_price=0.5),
                entry_time=datetime(2024, 1, i + 1),
                net_pnl=float(i * (-1) ** i),
            )
            for i in range(20)
        ]
        result_obj = StrategyResult(strategy_name="test", trades=trades)
        val = naive_benchmark_comparison(result_obj, [], [])
        assert val.test_name == "naive_benchmark"

    def test_contemporaneous_prices(self) -> None:
        markets = [
            _make_market("a1", neg_risk=True, event_id="e1"),
            _make_market("a2", neg_risk=True, event_id="e1"),
        ]
        prices = []
        for m in markets:
            prices.extend(_make_prices(m.market_id))
        result = check_contemporaneous_prices(markets, prices)
        assert result.passed


# ── Data Collector Tests ──────────────────────────────────────────


class TestDataCollector:
    def test_parse_datetime_iso(self) -> None:
        dt = DataCollector._parse_dt("2024-01-15T12:00:00Z")
        assert dt is not None
        assert dt.year == 2024

    def test_parse_datetime_epoch(self) -> None:
        dt = DataCollector._parse_dt(1705312800)
        assert dt is not None

    def test_parse_datetime_none(self) -> None:
        assert DataCollector._parse_dt(None) is None
        assert DataCollector._parse_dt("") is None

    def test_compute_quality_report(self) -> None:
        markets = [_make_market(f"m{i}") for i in range(5)]
        prices = []
        for m in markets:
            prices.extend(_make_prices(m.market_id, days=10))
        collector = DataCollector()
        report = collector.compute_quality_report(markets, prices)
        assert report.total_markets == 5
        assert report.total_price_observations == 50

    def test_filter_markets(self) -> None:
        markets = [
            _make_market("m1", volume=50000),
            _make_market("m2", volume=500),  # low volume
        ]
        prices = _make_prices("m1", days=10) + _make_prices("m2", days=2)
        collector = DataCollector()
        filtered_m, filtered_p = collector.filter_markets(markets, prices, min_volume=1000, min_observations=5)
        assert len(filtered_m) == 1
        assert filtered_m[0].market_id == "m1"
