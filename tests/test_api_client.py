"""Tests for API clients with mocked HTTP responses."""

from __future__ import annotations

import pytest
from aioresponses import aioresponses

from polymir.api.clob import ClobAPIError, ClobClient
from polymir.api.gamma import GammaAPIError, GammaClient
from polymir.config import APIConfig


@pytest.fixture
def api_config():
    return APIConfig(
        clob_base_url="http://test-clob",
        gamma_base_url="http://test-gamma",
        rate_limit_per_second=100.0,  # no rate limiting in tests
    )


class TestClobClient:
    @pytest.mark.asyncio
    async def test_get_orderbook(self, api_config):
        with aioresponses() as mocked:
            mocked.get(
                "http://test-clob/book?token_id=tok1",
                payload={
                    "bids": [{"price": "0.48", "size": "200"}],
                    "asks": [{"price": "0.52", "size": "150"}],
                },
            )
            async with ClobClient(api_config) as client:
                book = await client.get_orderbook("tok1")
                assert book.asset_id == "tok1"
                assert len(book.bids) == 1
                assert len(book.asks) == 1
                assert book.best_bid == pytest.approx(0.48)
                assert book.best_ask == pytest.approx(0.52)

    @pytest.mark.asyncio
    async def test_get_orderbook_empty(self, api_config):
        with aioresponses() as mocked:
            mocked.get(
                "http://test-clob/book?token_id=tok1",
                payload={"bids": [], "asks": []},
            )
            async with ClobClient(api_config) as client:
                book = await client.get_orderbook("tok1")
                assert book.midpoint is None

    @pytest.mark.asyncio
    async def test_get_trades(self, api_config):
        with aioresponses() as mocked:
            mocked.get(
                "http://test-clob/trades?limit=10&asset_id=a1",
                payload=[
                    {
                        "id": "t1",
                        "market": "m1",
                        "asset_id": "a1",
                        "owner": "0xabc",
                        "side": "BUY",
                        "size": 100,
                        "price": 0.55,
                    }
                ],
            )
            async with ClobClient(api_config) as client:
                trades = await client.get_trades(asset_id="a1", limit=10)
                assert len(trades) == 1
                assert trades[0].id == "t1"

    @pytest.mark.asyncio
    async def test_place_order(self, api_config):
        config = APIConfig(
            clob_base_url="http://test-clob",
            rate_limit_per_second=100.0,
            api_key="key",
            api_secret="secret",
            api_passphrase="pass",
        )
        with aioresponses() as mocked:
            mocked.post(
                "http://test-clob/order",
                payload={"orderID": "ord1", "status": "live"},
            )
            async with ClobClient(config) as client:
                result = await client.place_order("tok1", "BUY", 0.50, 100)
                assert result["orderID"] == "ord1"

    @pytest.mark.asyncio
    async def test_api_error(self, api_config):
        with aioresponses() as mocked:
            mocked.get("http://test-clob/book?token_id=tok1", status=500, body="Internal Error")
            async with ClobClient(api_config) as client:
                with pytest.raises(ClobAPIError) as exc_info:
                    await client.get_orderbook("tok1")
                assert exc_info.value.status == 500

    @pytest.mark.asyncio
    async def test_cancel_order(self, api_config):
        config = APIConfig(
            clob_base_url="http://test-clob",
            rate_limit_per_second=100.0,
            api_key="key",
            api_secret="secret",
            api_passphrase="pass",
        )
        with aioresponses() as mocked:
            mocked.delete(
                "http://test-clob/order?orderID=ord1",
                payload={"status": "cancelled"},
            )
            async with ClobClient(config) as client:
                result = await client.cancel_order("ord1")
                assert result["status"] == "cancelled"


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_rate_limiting_throttles(self):
        """Rate limiter actually throttles when tokens are exhausted."""
        import time
        from polymir.api.rate_limiter import RateLimiter

        limiter = RateLimiter(rate_per_second=2.0)
        # Burn through initial tokens
        await limiter.acquire()
        await limiter.acquire()
        # Third acquire should block
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.1  # had to wait for token refill

    @pytest.mark.asyncio
    async def test_rate_limiting_allows_burst(self):
        """Rate limiter allows burst up to bucket size."""
        from polymir.api.rate_limiter import RateLimiter

        limiter = RateLimiter(rate_per_second=10.0)
        # Should be able to acquire 10 tokens immediately
        for _ in range(10):
            await limiter.acquire()


class TestClobClientRetry:
    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self, api_config):
        """Client retries on transient connection errors."""
        import aiohttp

        config = APIConfig(
            clob_base_url="http://test-clob",
            rate_limit_per_second=100.0,
            max_retries=3,
        )
        with aioresponses() as mocked:
            # First call fails, second succeeds
            mocked.get(
                "http://test-clob/book?token_id=tok1",
                exception=aiohttp.ClientConnectionError("connection reset"),
            )
            mocked.get(
                "http://test-clob/book?token_id=tok1",
                payload={"bids": [{"price": "0.48", "size": "100"}], "asks": [{"price": "0.52", "size": "100"}]},
            )
            async with ClobClient(config) as client:
                book = await client.get_orderbook("tok1")
                assert book.best_bid == pytest.approx(0.48)


class TestGammaClient:
    @pytest.mark.asyncio
    async def test_get_resolved_markets(self, api_config):
        with aioresponses() as mocked:
            mocked.get(
                "http://test-gamma/markets?limit=10&offset=0&status=resolved",
                payload=[
                    {
                        "condition_id": "c1",
                        "question": "Will X happen?",
                        "status": "resolved",
                        "outcome": "Yes",
                        "volume": 50000,
                        "liquidity": 10000,
                        "tokens": [
                            {"token_id": "t1", "outcome": "Yes", "price": 1.0, "winner": True},
                            {"token_id": "t2", "outcome": "No", "price": 0.0, "winner": False},
                        ],
                    }
                ],
            )
            async with GammaClient(api_config) as client:
                markets = await client.get_resolved_markets(limit=10)
                assert len(markets) == 1
                assert markets[0].condition_id == "c1"
                assert markets[0].is_resolved
                assert len(markets[0].tokens) == 2
                assert markets[0].tokens[0].winner is True

    @pytest.mark.asyncio
    async def test_get_all_resolved_markets_pagination(self, api_config):
        with aioresponses() as mocked:
            # First page: full
            mocked.get(
                "http://test-gamma/markets?limit=2&offset=0&status=resolved",
                payload=[
                    {"condition_id": f"c{i}", "question": f"Q{i}", "status": "resolved"}
                    for i in range(2)
                ],
            )
            # Second page: partial (signals end)
            mocked.get(
                "http://test-gamma/markets?limit=2&offset=2&status=resolved",
                payload=[{"condition_id": "c2", "question": "Q2", "status": "resolved"}],
            )
            async with GammaClient(api_config) as client:
                markets = await client.get_all_resolved_markets(page_size=2)
                assert len(markets) == 3

    @pytest.mark.asyncio
    async def test_api_error(self, api_config):
        with aioresponses() as mocked:
            mocked.get(
                "http://test-gamma/markets?limit=100&offset=0&status=resolved",
                status=429,
                body="Rate limited",
            )
            async with GammaClient(api_config) as client:
                with pytest.raises(GammaAPIError) as exc_info:
                    await client.get_resolved_markets()
                assert exc_info.value.status == 429

    @pytest.mark.asyncio
    async def test_get_single_market(self, api_config):
        with aioresponses() as mocked:
            mocked.get(
                "http://test-gamma/markets/c1",
                payload={
                    "condition_id": "c1",
                    "question": "Test?",
                    "status": "active",
                    "tokens": [],
                },
            )
            async with GammaClient(api_config) as client:
                market = await client.get_market("c1")
                assert market.condition_id == "c1"
                assert not market.is_resolved
