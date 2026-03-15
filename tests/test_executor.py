"""Tests for mirror executor decision logic."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from polymir.config import ExecutionConfig
from polymir.executor import MirrorExecutor
from polymir.models import OrderBook, OrderBookLevel, TradeSignal


def _make_book(bids: list[tuple], asks: list[tuple]) -> OrderBook:
    return OrderBook(
        asset_id="test",
        bids=[OrderBookLevel(price=p, size=s) for p, s in bids],
        asks=[OrderBookLevel(price=p, size=s) for p, s in asks],
    )


class TestComputeSize:
    def test_caps_at_max_position(self):
        signal = TradeSignal(
            wallet_address="w", market_id="m", asset_id="a",
            side="BUY", size=10000, price=0.50,
        )
        book = _make_book([(0.48, 5000)], [(0.52, 5000)])
        exec_cfg = ExecutionConfig(max_position_usd=100)
        size = MirrorExecutor._compute_size(signal, book, exec_cfg)
        # Max 100 USD / 0.50 = 200 contracts
        assert size <= 200

    def test_scales_by_liquidity(self):
        signal = TradeSignal(
            wallet_address="w", market_id="m", asset_id="a",
            side="BUY", size=10000, price=0.50,
        )
        # Very thin book
        book = _make_book([(0.48, 10)], [(0.52, 10)])
        exec_cfg = ExecutionConfig(max_position_usd=100000)
        size = MirrorExecutor._compute_size(signal, book, exec_cfg)
        # Should be limited by 10% of liquidity
        assert size < 10000


class TestComputePrice:
    def test_midpoint_no_aggression(self):
        book = _make_book([(0.48, 100)], [(0.52, 100)])
        price = MirrorExecutor._compute_price(book, "BUY", aggression=0.0)
        assert price == pytest.approx(0.50)

    def test_aggressive_buy(self):
        book = _make_book([(0.48, 100)], [(0.52, 100)])
        price = MirrorExecutor._compute_price(book, "BUY", aggression=0.5)
        # Midpoint 0.50 + 0.5 * 0.04 * 0.5 = 0.50 + 0.01 = 0.51
        assert price == pytest.approx(0.51)

    def test_aggressive_sell(self):
        book = _make_book([(0.48, 100)], [(0.52, 100)])
        price = MirrorExecutor._compute_price(book, "SELL", aggression=0.5)
        # Midpoint 0.50 - 0.5 * 0.04 * 0.5 = 0.50 - 0.01 = 0.49
        assert price == pytest.approx(0.49)

    def test_empty_book(self):
        book = _make_book([], [])
        price = MirrorExecutor._compute_price(book, "BUY", aggression=0.0)
        assert price == 0.0
