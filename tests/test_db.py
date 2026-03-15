"""Tests for the database layer."""

from __future__ import annotations

from datetime import datetime

import pytest

from polymir.db import Database
from polymir.models import MirrorDecision, MirrorTrade, TradeSignal, WalletScore


class TestDatabaseWalletScores:
    @pytest.mark.asyncio
    async def test_upsert_and_retrieve(self, db):
        score = WalletScore(
            address="0xabc",
            win_rate=0.65,
            avg_roi=0.12,
            consistency=0.80,
            recency_score=0.70,
            hold_ratio=0.90,
            resolved_market_count=25,
            composite_score=0.72,
            scored_at=datetime(2024, 6, 1),
        )
        await db.upsert_wallet_score(score)
        result = await db.get_wallet_score("0xabc")
        assert result is not None
        assert result.address == "0xabc"
        assert result.composite_score == pytest.approx(0.72)

    @pytest.mark.asyncio
    async def test_top_wallets_ordering(self, db):
        for i, score_val in enumerate([0.5, 0.9, 0.7]):
            await db.upsert_wallet_score(
                WalletScore(
                    address=f"0x{i}",
                    win_rate=0.5,
                    avg_roi=0.1,
                    consistency=0.5,
                    recency_score=0.5,
                    hold_ratio=0.5,
                    resolved_market_count=20,
                    composite_score=score_val,
                    scored_at=datetime(2024, 6, 1),
                )
            )
        top = await db.get_top_wallets(limit=2)
        assert len(top) == 2
        assert top[0].composite_score == pytest.approx(0.9)
        assert top[1].composite_score == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_get_nonexistent_wallet(self, db):
        result = await db.get_wallet_score("0xnotfound")
        assert result is None


class TestDatabaseMirrorTrades:
    @pytest.mark.asyncio
    async def test_record_and_retrieve(self, db):
        signal = TradeSignal(
            wallet_address="0xabc",
            market_id="m1",
            asset_id="a1",
            side="BUY",
            size=100,
            price=0.55,
        )
        trade = MirrorTrade(
            signal=signal,
            decision=MirrorDecision.EXECUTE,
            order_price=0.55,
            order_size=50,
            estimated_slippage=0.01,
        )
        row_id = await db.record_mirror_trade(trade)
        assert row_id > 0

        trades = await db.get_mirror_trades(wallet="0xabc")
        assert len(trades) == 1
        assert trades[0]["decision"] == "execute"

    @pytest.mark.asyncio
    async def test_filter_by_decision(self, db):
        for decision in [MirrorDecision.EXECUTE, MirrorDecision.SKIP_SLIPPAGE, MirrorDecision.EXECUTE]:
            signal = TradeSignal(
                wallet_address="0xabc",
                market_id="m1",
                asset_id="a1",
                side="BUY",
                size=100,
                price=0.55,
            )
            await db.record_mirror_trade(MirrorTrade(signal=signal, decision=decision))

        executed = await db.get_mirror_trades(decision=MirrorDecision.EXECUTE)
        assert len(executed) == 2
        skipped = await db.get_mirror_trades(decision=MirrorDecision.SKIP_SLIPPAGE)
        assert len(skipped) == 1

    @pytest.mark.asyncio
    async def test_slippage_stats(self, db):
        for est, real in [(0.01, 0.015), (0.02, 0.018)]:
            signal = TradeSignal(
                wallet_address="0xabc",
                market_id="m1",
                asset_id="a1",
                side="BUY",
                size=100,
                price=0.55,
            )
            await db.record_mirror_trade(
                MirrorTrade(
                    signal=signal,
                    decision=MirrorDecision.EXECUTE,
                    estimated_slippage=est,
                    realized_slippage=real,
                )
            )
        stats = await db.get_slippage_stats()
        assert stats["trade_count"] == 2
        assert stats["avg_estimated"] == pytest.approx(0.015)
        assert stats["avg_realized"] == pytest.approx(0.0165)
