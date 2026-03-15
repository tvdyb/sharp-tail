"""Tests for wallet scoring model."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from polymir.config import ScoringConfig
from polymir.scanner import WalletMarketResult, compute_wallet_score, _normalize_roi


def _make_results(
    wallet: str = "0xabc",
    count: int = 25,
    win_fraction: float = 0.6,
    roi_base: float = 0.10,
    hold_fraction: float = 0.8,
    days_ago_start: int = 180,
) -> list[WalletMarketResult]:
    """Generate synthetic wallet market results."""
    results = []
    now = datetime.utcnow()
    for i in range(count):
        won = i < int(count * win_fraction)
        roi = roi_base if won else -roi_base * 0.5
        days_ago = days_ago_start - (i * days_ago_start // count)
        results.append(
            WalletMarketResult(
                wallet=wallet,
                market_id=f"m{i}",
                won=won,
                roi=roi,
                held_to_expiration=i < int(count * hold_fraction),
                total_bought=100,
                total_sold=20 if i < int(count * hold_fraction) else 100,
                resolution_date=now - timedelta(days=days_ago),
            )
        )
    return results


class TestComputeWalletScore:
    def test_basic_scoring(self):
        results = _make_results()
        config = ScoringConfig()
        score = compute_wallet_score("0xabc", results, config)
        assert score is not None
        assert score.address == "0xabc"
        assert 0 < score.composite_score < 1
        assert score.win_rate == pytest.approx(0.6)
        assert score.resolved_market_count == 25

    def test_below_min_markets_returns_none(self):
        results = _make_results(count=5)
        config = ScoringConfig(min_resolved_markets=20)
        score = compute_wallet_score("0xabc", results, config)
        assert score is None

    def test_perfect_wallet(self):
        results = _make_results(
            count=30, win_fraction=1.0, roi_base=0.20, hold_fraction=1.0
        )
        config = ScoringConfig()
        score = compute_wallet_score("0xabc", results, config)
        assert score is not None
        assert score.win_rate == 1.0
        assert score.hold_ratio == 1.0
        assert score.composite_score > 0.7

    def test_terrible_wallet(self):
        results = _make_results(
            count=25, win_fraction=0.1, roi_base=0.05, hold_fraction=0.3
        )
        config = ScoringConfig()
        score = compute_wallet_score("0xabc", results, config)
        assert score is not None
        assert score.win_rate == pytest.approx(0.1, abs=0.05)
        assert score.composite_score < 0.5

    def test_high_win_rate_beats_low(self):
        config = ScoringConfig()
        high_wr = compute_wallet_score(
            "high", _make_results(win_fraction=0.9), config
        )
        low_wr = compute_wallet_score(
            "low", _make_results(win_fraction=0.2), config
        )
        assert high_wr is not None and low_wr is not None
        assert high_wr.composite_score > low_wr.composite_score

    def test_recency_weighting(self):
        """Recent results should be weighted more heavily."""
        config = ScoringConfig()
        recent = _make_results(days_ago_start=30)
        old = _make_results(days_ago_start=365)
        score_recent = compute_wallet_score("recent", recent, config)
        score_old = compute_wallet_score("old", old, config)
        assert score_recent is not None and score_old is not None
        assert score_recent.recency_score > score_old.recency_score

    def test_consistency_scoring(self):
        """Consistent returns should score higher than volatile."""
        config = ScoringConfig()
        # Consistent: all same ROI
        consistent = [
            WalletMarketResult(
                wallet="c", market_id=f"m{i}", won=True, roi=0.10,
                held_to_expiration=True, total_bought=100, total_sold=0,
                resolution_date=datetime.utcnow() - timedelta(days=i),
            )
            for i in range(25)
        ]
        # Volatile: alternating high/low ROI
        volatile = [
            WalletMarketResult(
                wallet="v", market_id=f"m{i}", won=True,
                roi=0.50 if i % 2 == 0 else -0.30,
                held_to_expiration=True, total_bought=100, total_sold=0,
                resolution_date=datetime.utcnow() - timedelta(days=i),
            )
            for i in range(25)
        ]
        sc = compute_wallet_score("c", consistent, config)
        sv = compute_wallet_score("v", volatile, config)
        assert sc is not None and sv is not None
        assert sc.consistency > sv.consistency


class TestNormalizeRoi:
    def test_zero(self):
        assert _normalize_roi(0) == pytest.approx(0.5)

    def test_positive(self):
        assert _normalize_roi(1.0) > 0.5

    def test_negative(self):
        assert _normalize_roi(-1.0) < 0.5

    def test_bounded(self):
        assert 0 < _normalize_roi(100) <= 1.0
        assert 0 <= _normalize_roi(-100) < 1.0
