"""Tests for Pydantic models and orderbook logic."""

from __future__ import annotations

import pytest

from polymir.models import OrderBook, OrderBookLevel, Trade, WalletScore


class TestOrderBook:
    def _make_book(self, bids: list[tuple], asks: list[tuple]) -> OrderBook:
        return OrderBook(
            asset_id="test",
            bids=[OrderBookLevel(price=p, size=s) for p, s in bids],
            asks=[OrderBookLevel(price=p, size=s) for p, s in asks],
        )

    def test_midpoint(self):
        book = self._make_book([(0.48, 100)], [(0.52, 100)])
        assert book.midpoint == pytest.approx(0.50)

    def test_spread(self):
        book = self._make_book([(0.48, 100)], [(0.52, 100)])
        assert book.spread == pytest.approx(0.04)

    def test_spread_pct(self):
        book = self._make_book([(0.48, 100)], [(0.52, 100)])
        assert book.spread_pct == pytest.approx(0.08)

    def test_empty_book(self):
        book = self._make_book([], [])
        assert book.midpoint is None
        assert book.spread is None
        assert book.spread_pct is None
        assert book.best_bid is None
        assert book.best_ask is None

    def test_total_liquidity(self):
        book = self._make_book([(0.50, 200)], [(0.55, 100)])
        bid_liq = 0.50 * 200
        ask_liq = 0.55 * 100
        assert book.total_liquidity("bid") == pytest.approx(bid_liq)
        assert book.total_liquidity("ask") == pytest.approx(ask_liq)
        assert book.total_liquidity("both") == pytest.approx(bid_liq + ask_liq)

    def test_estimate_fill_price_buy_single_level(self):
        book = self._make_book([], [(0.55, 200)])
        fill = book.estimate_fill_price(100, "BUY")
        assert fill == pytest.approx(0.55)

    def test_estimate_fill_price_buy_multiple_levels(self):
        book = self._make_book([], [(0.55, 50), (0.60, 100)])
        # 50 @ 0.55 + 50 @ 0.60 = 27.5 + 30 = 57.5 / 100 = 0.575
        fill = book.estimate_fill_price(100, "BUY")
        assert fill == pytest.approx(0.575)

    def test_estimate_fill_price_insufficient_liquidity(self):
        book = self._make_book([], [(0.55, 10)])
        fill = book.estimate_fill_price(100, "BUY")
        assert fill is None

    def test_estimate_fill_price_sell(self):
        book = self._make_book([(0.50, 200)], [])
        fill = book.estimate_fill_price(100, "SELL")
        assert fill == pytest.approx(0.50)

    def test_estimate_fill_price_zero_size(self):
        book = self._make_book([], [(0.55, 200)])
        fill = book.estimate_fill_price(0, "BUY")
        assert fill is None


class TestWalletScore:
    def test_creation(self):
        ws = WalletScore(
            address="0xabc",
            win_rate=0.65,
            avg_roi=0.12,
            sharpe_ratio=0.80,
            sharpe_ci_lower=0.50,
            sharpe_ci_upper=1.10,
            hold_ratio=0.90,
            resolved_market_count=25,
            composite_score=0.72,
        )
        assert ws.address == "0xabc"
        assert ws.composite_score == 0.72


class TestTrade:
    def test_trade_parse(self):
        t = Trade(
            id="t1",
            market="m1",
            asset_id="a1",
            owner="0xabc",
            side="BUY",
            size=100.0,
            price=0.55,
        )
        assert t.side == "BUY"
        assert t.size == 100.0
