"""Integration tests: scanner -> monitor -> executor pipeline with mocked APIs."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from aioresponses import aioresponses

from polymir.config import APIConfig, AppConfig, ExecutionConfig, ScoringConfig
from polymir.db import Database
from polymir.executor import MirrorExecutor
from polymir.models import (
    MirrorDecision,
    OrderBook,
    OrderBookLevel,
    Trade,
    TradeSignal,
    WalletScore,
)
from polymir.monitor import TradeMonitor
from polymir.scanner import WalletScanner


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def db(tmp_path):
    db_path = str(tmp_path / "integration_test.db")
    async with Database(db_path) as database:
        yield database


@pytest.fixture
def api_config():
    return APIConfig(
        clob_base_url="http://clob.test",
        gamma_base_url="http://gamma.test",
        rate_limit_per_second=1000.0,  # no throttle in tests
    )


@pytest.fixture
def config(api_config):
    return AppConfig(
        api=api_config,
        scoring=ScoringConfig(min_resolved_markets=2),  # low threshold for tests
        execution=ExecutionConfig(
            min_liquidity_usd=100.0,  # low threshold for tests
            max_slippage_pct=0.05,
            max_spread_pct=0.10,
            max_position_usd=500.0,
            stale_signal_timeout_s=300.0,
            fill_timeout_s=0.5,  # short timeout for tests
        ),
        top_wallets=10,
    )


# ── Helper data ──────────────────────────────────────────────────────

WALLET_A = "0xaaaa"
WALLET_B = "0xbbbb"
MARKET_1 = "market_001"
TOKEN_YES = "token_yes_001"
TOKEN_NO = "token_no_001"

NOW = datetime.now(timezone.utc).replace(tzinfo=None)


def _resolved_market(condition_id: str, question: str) -> dict:
    return {
        "condition_id": condition_id,
        "question": question,
        "slug": condition_id,
        "status": "resolved",
        "outcome": "Yes",
        "tokens": [
            {"token_id": f"{condition_id}_yes", "outcome": "Yes", "price": 1.0, "winner": True},
            {"token_id": f"{condition_id}_no", "outcome": "No", "price": 0.0, "winner": False},
        ],
        "volume": 50000,
        "liquidity": 20000,
    }


def _trade(owner: str, asset_id: str, side: str, size: float, price: float) -> dict:
    return {
        "id": f"trade_{owner}_{asset_id}_{side}_{size}",
        "taker_order_id": "taker_001",
        "market": MARKET_1,
        "asset_id": asset_id,
        "side": side,
        "size": size,
        "price": price,
        "owner": owner,
    }


def _orderbook_response(bid: float = 0.60, ask: float = 0.62, depth: float = 500) -> dict:
    return {
        "bids": [{"price": str(bid), "size": str(depth)}],
        "asks": [{"price": str(ask), "size": str(depth)}],
    }


# ── Tests ────────────────────────────────────────────────────────────


class TestScannerToDatabase:
    """Scanner scores wallets and persists to DB, which monitor can then read."""

    @pytest.mark.asyncio
    async def test_scanner_scores_and_monitor_loads(self, config, db):
        """Full pipeline: scanner scores wallets, monitor loads them as watchlist."""
        markets = [_resolved_market(f"mkt_{i}", f"Question {i}") for i in range(3)]

        with aioresponses() as mock:
            # Gamma: return 3 resolved markets, then empty page
            mock.get(
                "http://gamma.test/markets?limit=100&offset=0&status=resolved",
                payload=markets,
            )
            mock.get(
                "http://gamma.test/markets?limit=100&offset=100&status=resolved",
                payload=[],
            )

            # CLOB: for each market's 2 tokens, return trades
            # Wallet A: bought YES tokens in all 3 markets (winning side)
            # Wallet B: bought NO tokens in all 3 markets (losing side)
            for i in range(3):
                yes_id = f"mkt_{i}_yes"
                no_id = f"mkt_{i}_no"
                mock.get(
                    f"http://clob.test/trades?limit=500&asset_id={yes_id}",
                    payload=[
                        _trade(WALLET_A, yes_id, "BUY", 100, 0.55),
                    ],
                )
                mock.get(
                    f"http://clob.test/trades?limit=500&asset_id={no_id}",
                    payload=[
                        _trade(WALLET_B, no_id, "BUY", 100, 0.45),
                    ],
                )

            scanner = WalletScanner(config, db)
            scores = await scanner.run()

        # Wallet A should be scored (3 markets >= min 2)
        assert len(scores) >= 1
        wallet_a_score = next((s for s in scores if s.address == WALLET_A), None)
        assert wallet_a_score is not None
        assert wallet_a_score.win_rate == 1.0  # won all 3

        wallet_b_score = next((s for s in scores if s.address == WALLET_B), None)
        assert wallet_b_score is not None
        assert wallet_b_score.win_rate == 0.0  # lost all 3

        # Wallet A should rank higher
        assert wallet_a_score.composite_score > wallet_b_score.composite_score

        # Now verify monitor can load the watchlist from DB
        monitor = TradeMonitor(config, db)
        await monitor.load_watchlist()
        assert WALLET_A in monitor._watched_wallets
        assert WALLET_B in monitor._watched_wallets


class TestMonitorSignalGeneration:
    """Monitor generates signals that the executor can process."""

    @pytest.mark.asyncio
    async def test_trade_signal_from_watched_wallet(self, config, db):
        """Monitor correctly converts a watched wallet's BUY trade to a signal."""
        # Pre-populate a watched wallet in DB
        score = WalletScore(
            address=WALLET_A,
            win_rate=0.8,
            avg_roi=0.5,
            consistency=0.7,
            recency_score=0.6,
            hold_ratio=0.9,
            resolved_market_count=25,
            composite_score=0.75,
        )
        await db.upsert_wallet_score(score)

        monitor = TradeMonitor(config, db)
        await monitor.load_watchlist()

        # Simulate a trade from the watched wallet
        trade = Trade(
            id="t_001",
            market=MARKET_1,
            asset_id=TOKEN_YES,
            side="BUY",
            size=50.0,
            price=0.60,
            owner=WALLET_A,
        )
        signal = monitor._trade_to_signal(trade)
        assert signal is not None
        assert signal.wallet_address == WALLET_A
        assert signal.asset_id == TOKEN_YES
        assert signal.side == "BUY"
        assert signal.size == 50.0

    @pytest.mark.asyncio
    async def test_sell_trade_not_signaled(self, config, db):
        """Monitor ignores SELL trades (not new position entries)."""
        score = WalletScore(
            address=WALLET_A,
            win_rate=0.8,
            avg_roi=0.5,
            consistency=0.7,
            recency_score=0.6,
            hold_ratio=0.9,
            resolved_market_count=25,
            composite_score=0.75,
        )
        await db.upsert_wallet_score(score)

        monitor = TradeMonitor(config, db)
        await monitor.load_watchlist()

        trade = Trade(
            id="t_002",
            market=MARKET_1,
            asset_id=TOKEN_YES,
            side="SELL",
            size=50.0,
            price=0.70,
            owner=WALLET_A,
        )
        signal = monitor._trade_to_signal(trade)
        assert signal is None

    @pytest.mark.asyncio
    async def test_unwatched_wallet_not_signaled(self, config, db):
        """Monitor ignores trades from wallets not in watchlist."""
        monitor = TradeMonitor(config, db)
        await monitor.load_watchlist()  # empty watchlist

        trade = Trade(
            id="t_003",
            market=MARKET_1,
            asset_id=TOKEN_YES,
            side="BUY",
            size=50.0,
            price=0.60,
            owner="0xunknown",
        )
        signal = monitor._trade_to_signal(trade)
        assert signal is None


class TestExecutorDecisions:
    """Executor makes correct decisions given signals and orderbook state."""

    @pytest.mark.asyncio
    async def test_execute_valid_signal(self, config, db):
        """Executor places an order for a valid signal with good liquidity."""
        executor = MirrorExecutor(config, db)

        signal = TradeSignal(
            wallet_address=WALLET_A,
            market_id=MARKET_1,
            asset_id=TOKEN_YES,
            side="BUY",
            size=50.0,
            price=0.60,
            detected_at=NOW,
        )

        with aioresponses() as mock:
            # Good orderbook
            mock.get(
                f"http://clob.test/book?token_id={TOKEN_YES}",
                payload=_orderbook_response(0.59, 0.61, 500),
            )
            # Order placed successfully
            mock.post(
                "http://clob.test/order",
                payload={"orderID": "order_123", "status": "open"},
            )
            # Order fills
            mock.get(
                "http://clob.test/order/order_123",
                payload={"status": "filled", "avg_fill_price": "0.605", "filled_size": "50"},
            )

            async with __import__("polymir.api.clob", fromlist=["ClobClient"]).ClobClient(config.api) as clob:
                result = await executor._process_signal(signal, clob)

        assert result.decision == MirrorDecision.EXECUTE
        assert result.fill_price == 0.605
        assert result.order_id == "order_123"

        # Verify recorded in DB
        trades = await db.get_mirror_trades(wallet=WALLET_A)
        assert len(trades) == 1
        assert trades[0]["decision"] == "execute"

    @pytest.mark.asyncio
    async def test_skip_low_liquidity(self, config, db):
        """Executor skips when orderbook liquidity is below threshold."""
        executor = MirrorExecutor(config, db)

        signal = TradeSignal(
            wallet_address=WALLET_A,
            market_id=MARKET_1,
            asset_id=TOKEN_YES,
            side="BUY",
            size=50.0,
            price=0.60,
            detected_at=NOW,
        )

        with aioresponses() as mock:
            # Thin orderbook: only $1 total liquidity
            mock.get(
                f"http://clob.test/book?token_id={TOKEN_YES}",
                payload=_orderbook_response(0.59, 0.61, 1),
            )

            from polymir.api.clob import ClobClient

            async with ClobClient(config.api) as clob:
                result = await executor._process_signal(signal, clob)

        assert result.decision == MirrorDecision.SKIP_LIQUIDITY

    @pytest.mark.asyncio
    async def test_skip_wide_spread(self, config, db):
        """Executor skips when spread exceeds threshold."""
        executor = MirrorExecutor(config, db)

        signal = TradeSignal(
            wallet_address=WALLET_A,
            market_id=MARKET_1,
            asset_id=TOKEN_YES,
            side="BUY",
            size=50.0,
            price=0.60,
            detected_at=NOW,
        )

        with aioresponses() as mock:
            # Wide spread: 0.40 to 0.60 = 33% spread
            mock.get(
                f"http://clob.test/book?token_id={TOKEN_YES}",
                payload=_orderbook_response(0.40, 0.60, 500),
            )

            from polymir.api.clob import ClobClient

            async with ClobClient(config.api) as clob:
                result = await executor._process_signal(signal, clob)

        assert result.decision == MirrorDecision.SKIP_SPREAD

    @pytest.mark.asyncio
    async def test_skip_stale_signal(self, config, db):
        """Executor skips when signal is too old."""
        executor = MirrorExecutor(config, db)

        # Signal from 10 minutes ago (exceeds 300s timeout)
        from datetime import timedelta

        signal = TradeSignal(
            wallet_address=WALLET_A,
            market_id=MARKET_1,
            asset_id=TOKEN_YES,
            side="BUY",
            size=50.0,
            price=0.60,
            detected_at=NOW - timedelta(minutes=10),
        )

        from polymir.api.clob import ClobClient

        async with ClobClient(config.api) as clob:
            result = await executor._process_signal(signal, clob)

        assert result.decision == MirrorDecision.SKIP_STALE


class TestEndToEndPipeline:
    """Full pipeline: score -> detect -> execute with mocked APIs."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, config, db):
        """Score wallets, then process a signal from a top wallet through executor."""
        # Step 1: Score wallets via scanner
        markets = [_resolved_market(f"mkt_{i}", f"Q{i}") for i in range(3)]

        with aioresponses() as mock:
            mock.get(
                "http://gamma.test/markets?limit=100&offset=0&status=resolved",
                payload=markets,
            )
            mock.get(
                "http://gamma.test/markets?limit=100&offset=100&status=resolved",
                payload=[],
            )
            for i in range(3):
                yes_id = f"mkt_{i}_yes"
                no_id = f"mkt_{i}_no"
                mock.get(
                    f"http://clob.test/trades?limit=500&asset_id={yes_id}",
                    payload=[_trade(WALLET_A, yes_id, "BUY", 100, 0.55)],
                )
                mock.get(
                    f"http://clob.test/trades?limit=500&asset_id={no_id}",
                    payload=[],
                )

            scanner = WalletScanner(config, db)
            scores = await scanner.run()

        assert any(s.address == WALLET_A for s in scores)

        # Step 2: Monitor loads watchlist
        monitor = TradeMonitor(config, db)
        await monitor.load_watchlist()
        assert WALLET_A in monitor._watched_wallets

        # Step 3: Simulate detection of a new trade
        new_trade = Trade(
            id="t_new",
            market="new_market",
            asset_id="new_token",
            side="BUY",
            size=75.0,
            price=0.55,
            owner=WALLET_A,
        )
        signal = monitor._trade_to_signal(new_trade)
        assert signal is not None

        # Step 4: Executor processes signal
        executor = MirrorExecutor(config, db)
        with aioresponses() as mock:
            mock.get(
                "http://clob.test/book?token_id=new_token",
                payload=_orderbook_response(0.54, 0.56, 500),
            )
            mock.post(
                "http://clob.test/order",
                payload={"orderID": "order_e2e", "status": "open"},
            )
            mock.get(
                "http://clob.test/order/order_e2e",
                payload={"status": "filled", "avg_fill_price": "0.555", "filled_size": "75"},
            )

            from polymir.api.clob import ClobClient

            async with ClobClient(config.api) as clob:
                result = await executor._process_signal(signal, clob)

        assert result.decision == MirrorDecision.EXECUTE
        assert result.fill_price == 0.555

        # Verify the full chain is recorded in DB
        mirror_trades = await db.get_mirror_trades(wallet=WALLET_A)
        assert len(mirror_trades) == 1
        assert mirror_trades[0]["decision"] == "execute"
        assert mirror_trades[0]["order_id"] == "order_e2e"

        # Verify slippage stats
        stats = await db.get_slippage_stats()
        assert stats["trade_count"] == 1


class TestResearchPipelineIntegration:
    """End-to-end test: runs all 7 strategies on synthetic data."""

    def test_full_research_pipeline_synthetic(self, tmp_path):
        """Create synthetic markets + prices, run all 7 strategies, validate, generate report."""
        from datetime import date, datetime, timedelta

        from polymir.research.models import (
            MarketCategory,
            PriceSnapshot,
            ResearchMarket,
        )
        from polymir.research.pipeline import run_research, run_validation, generate_full_report

        import random
        rng = random.Random(42)

        # Create 20 synthetic resolved markets
        markets = []
        prices = []
        start_dt = datetime(2024, 1, 1)

        categories = [MarketCategory.POLITICS, MarketCategory.SPORTS, MarketCategory.CRYPTO,
                       MarketCategory.ECONOMICS, MarketCategory.WEATHER]

        for i in range(20):
            mid = f"market_{i}"
            outcome = "Yes" if rng.random() > 0.4 else "No"
            cat = categories[i % len(categories)]
            creation = start_dt + timedelta(days=i * 5)
            resolution = creation + timedelta(days=60 + rng.randint(0, 30))

            markets.append(ResearchMarket(
                market_id=mid,
                condition_id=mid,
                question=f"Will event {i} happen?",
                slug=mid,
                category=cat,
                creation_date=creation,
                end_date=resolution,
                resolution_date=resolution,
                outcome=outcome,
                total_volume=rng.uniform(10000, 500000),
                liquidity=rng.uniform(5000, 50000),
                token_ids=[f"{mid}_yes", f"{mid}_no"],
                neg_risk=i < 5,  # first 5 markets are neg-risk (for multi-outcome arb)
                event_id=f"event_{i // 3}",  # group some markets by event
                status="resolved",
                total_lifetime_days=60.0,
            ))

            # Generate 120 days of price data per market
            base_price = 0.3 + rng.random() * 0.4
            drift = 0.002 if outcome == "Yes" else -0.002
            price = base_price
            for d in range(120):
                ts = creation + timedelta(days=d)
                price = max(0.05, min(0.95, price + drift + rng.gauss(0, 0.01)))
                prices.append(PriceSnapshot(
                    market_id=mid,
                    token_id=f"{mid}_yes",
                    timestamp=ts,
                    price=price,
                    volume_bucket=rng.uniform(100, 5000),
                ))

        # Run all 7 strategies
        start = date(2024, 1, 15)
        end = date(2024, 8, 1)
        results = run_research(markets, prices, start=start, end=end)

        assert len(results) == 7, f"Expected 7 strategies, got {len(results)}"
        for name, result in results.items():
            assert result.strategy_name == name
            # Not all strategies will produce trades, but the result should exist
            assert result is not None

        # Run validation on strategies that produced trades
        validations = run_validation(markets, prices, results, start=start, end=end)
        # Some strategies may not have enough trades for validation, that's OK
        assert isinstance(validations, dict)

        # Generate report
        report = generate_full_report(
            markets, prices, results, validations,
            output_dir=str(tmp_path),
        )
        assert "Alpha Research Report" in report
        # Check report file exists
        import os
        assert os.path.exists(tmp_path / "research_report.md")
