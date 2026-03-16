"""Tests for wallet scoring model (dual-path: Sharpe CI + consistency)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from polymir.config import ScoringConfig
from polymir.scanner import WalletMarketResult, compute_wallet_score, _z_score


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
        config = ScoringConfig(min_resolved_markets=20)
        score = compute_wallet_score("0xabc", results, config)
        assert score is not None
        assert score.address == "0xabc"
        # Win rate computed on held positions only: 15 wins out of 20 held = 0.75
        assert score.win_rate == pytest.approx(0.75)
        assert score.resolved_market_count == 25
        # Sharpe CI fields populated
        assert score.sharpe_ratio != 0.0
        assert score.sharpe_ci_lower < score.sharpe_ratio < score.sharpe_ci_upper
        # composite_score == sharpe point estimate (not ci_lower)
        assert score.composite_score == pytest.approx(score.sharpe_ratio)

    def test_below_min_markets_returns_none(self):
        results = _make_results(count=5)
        config = ScoringConfig(min_resolved_markets=20)
        score = compute_wallet_score("0xabc", results, config)
        assert score is None

    def test_below_min_hold_ratio_returns_none(self):
        """Wallets that exit early too often should be filtered out."""
        results = _make_results(count=25, hold_fraction=0.3)
        config = ScoringConfig(min_resolved_markets=20, min_hold_ratio=0.70)
        score = compute_wallet_score("0xabc", results, config)
        assert score is None

    def test_perfect_wallet(self):
        """All wins with varying ROIs → high Sharpe and positive rating."""
        now = datetime.utcnow()
        results = [
            WalletMarketResult(
                wallet="0xabc", market_id=f"m{i}", won=True,
                roi=0.15 + 0.01 * i,  # slight variance so stdev > 0
                held_to_expiration=True, total_bought=100, total_sold=20,
                resolution_date=now - timedelta(days=i),
            )
            for i in range(30)
        ]
        config = ScoringConfig(min_resolved_markets=20)
        score = compute_wallet_score("0xabc", results, config)
        assert score is not None
        assert score.win_rate == 1.0
        assert score.hold_ratio == 1.0
        assert score.sharpe_ratio > 0
        assert score.composite_score > 0

    def test_high_win_rate_beats_low(self):
        """High win rate wallet scores higher; low win rate filtered by significance gate."""
        config = ScoringConfig(min_resolved_markets=20)
        high_wr = compute_wallet_score(
            "high", _make_results(win_fraction=0.8, hold_fraction=1.0), config
        )
        # Low win rate (0.3) has negative Sharpe → CI lower bound < 0 → filtered out
        low_wr = compute_wallet_score(
            "low", _make_results(win_fraction=0.3, hold_fraction=1.0), config
        )
        assert high_wr is not None
        assert high_wr.sharpe_ratio > 0
        assert low_wr is None  # filtered by significance gate

    def test_longer_track_record_tightens_ci(self):
        """More markets → narrower CI."""
        config = ScoringConfig(min_resolved_markets=20)
        short = compute_wallet_score(
            "short", _make_results(count=25, win_fraction=0.7), config
        )
        long = compute_wallet_score(
            "long", _make_results(count=100, win_fraction=0.7), config
        )
        assert short is not None and long is not None
        short_width = short.sharpe_ci_upper - short.sharpe_ci_lower
        long_width = long.sharpe_ci_upper - long.sharpe_ci_lower
        assert long_width < short_width

    def test_consistent_beats_volatile(self):
        """Consistent returns → higher Sharpe; volatile may not pass significance gate."""
        config = ScoringConfig(min_resolved_markets=20)
        # Consistent: moderate spread of positive ROIs (stdev ~0.03)
        consistent = [
            WalletMarketResult(
                wallet="c", market_id=f"m{i}", won=True,
                roi=0.10 + 0.02 * (i % 5 - 2),  # 0.06 to 0.14
                held_to_expiration=True, total_bought=100, total_sold=0,
                resolution_date=datetime.utcnow() - timedelta(days=i),
            )
            for i in range(25)
        ]
        sc = compute_wallet_score("c", consistent, config)
        assert sc is not None
        assert sc.sharpe_ratio > 1.0  # high Sharpe from consistent positive returns

        # Volatile: wild swings — may not pass significance gate
        volatile = [
            WalletMarketResult(
                wallet="v", market_id=f"m{i}", won=True,
                roi=0.50 if i % 2 == 0 else -0.30,
                held_to_expiration=True, total_bought=100, total_sold=0,
                resolution_date=datetime.utcnow() - timedelta(days=i),
            )
            for i in range(25)
        ]
        sv = compute_wallet_score("v", volatile, config)
        # Volatile wallet has low Sharpe (~0.28) with wide CI → filtered out
        assert sv is None

    def test_hold_ratio_filter(self):
        """Wallets below min_hold_ratio are filtered out."""
        now = datetime.utcnow()
        config = ScoringConfig(min_resolved_markets=20, min_hold_ratio=0.5)

        def _make_held_results(hold_frac: float) -> list[WalletMarketResult]:
            results = []
            count = 25
            held_count = int(count * hold_frac)
            for i in range(count):
                held = i < held_count
                won = (i % 10) < 7
                results.append(WalletMarketResult(
                    wallet="w", market_id=f"m{i}", won=won,
                    roi=0.10 if won else -0.05,
                    held_to_expiration=held,
                    total_bought=100, total_sold=20 if held else 100,
                    resolution_date=now - timedelta(days=180 - i * 7),
                ))
            return results

        high_hold = compute_wallet_score("holder", _make_held_results(1.0), config)
        low_hold = compute_wallet_score("flipper", _make_held_results(0.6), config)
        assert high_hold is not None and low_hold is not None
        assert high_hold.hold_ratio > low_hold.hold_ratio

    def test_significance_gate_filters_weak_sharpe(self):
        """Wallet with positive but weak Sharpe and CI_lower < 0 is filtered out."""
        config = ScoringConfig(min_resolved_markets=10)
        # 55% win rate → small positive Sharpe but wide CI
        results = _make_results(count=12, win_fraction=0.55, hold_fraction=1.0)
        score = compute_wallet_score("0xweak", results, config)
        assert score is None  # CI lower bound < 0

    def test_composite_score_is_sharpe_point_estimate(self):
        """For Sharpe-path wallets, composite_score equals the Sharpe point estimate."""
        config = ScoringConfig(min_resolved_markets=20)
        results = _make_results(win_fraction=0.8, hold_fraction=1.0)
        score = compute_wallet_score("0xtest", results, config)
        assert score is not None
        assert score.composite_score == pytest.approx(score.sharpe_ratio)
        assert score.composite_score != pytest.approx(score.sharpe_ci_lower)

    def test_12_markets_passes_new_default(self):
        """A wallet with 12 markets now passes the min_resolved_markets=10 default."""
        config = ScoringConfig()  # uses new defaults: min_resolved_markets=10
        results = _make_results(count=12, win_fraction=0.8, hold_fraction=1.0)
        score = compute_wallet_score("0xnew", results, config)
        assert score is not None
        assert score.sharpe_ratio > 0


class TestConsistencyPath:
    """Tests for the consistency-based scoring path (near-zero ROI stdev)."""

    def test_consistency_path_all_wins_near_zero_stdev(self):
        """15 markets, all wins, near-zero stdev → scores via consistency path."""
        now = datetime.utcnow()
        results = [
            WalletMarketResult(
                wallet="0xcons", market_id=f"m{i}", won=True,
                roi=0.001,  # all identical → stdev = 0
                held_to_expiration=True, total_bought=100, total_sold=0,
                resolution_date=now - timedelta(days=i),
            )
            for i in range(15)
        ]
        config = ScoringConfig(min_resolved_markets=10)
        score = compute_wallet_score("0xcons", results, config)
        assert score is not None
        # Consistency path: Sharpe fields are 0
        assert score.sharpe_ratio == 0.0
        assert score.sharpe_ci_lower == 0.0
        assert score.sharpe_ci_upper == 0.0
        # composite_score = win_rate * log(n_held)
        assert score.composite_score == pytest.approx(1.0 * math.log(15))
        assert score.win_rate == 1.0

    def test_consistency_path_requires_min_markets(self):
        """Consistency path requires len(held) >= min_resolved_markets."""
        now = datetime.utcnow()
        results = [
            WalletMarketResult(
                wallet="0xfew", market_id=f"m{i}", won=True,
                roi=0.001, held_to_expiration=True,
                total_bought=100, total_sold=0,
                resolution_date=now - timedelta(days=i),
            )
            for i in range(10)
        ]
        # Pass the total results count check but use high min for consistency
        config = ScoringConfig(min_resolved_markets=10, min_roi_stdev=0.01)
        score = compute_wallet_score("0xfew", results, config)
        assert score is not None  # 10 held >= min 10

    def test_consistency_path_rejects_low_win_rate(self):
        """Consistency path requires win_rate >= 0.85."""
        now = datetime.utcnow()
        results = [
            WalletMarketResult(
                wallet="0xlow", market_id=f"m{i}",
                won=i < 10,  # 10 wins out of 15 = 66%
                roi=0.001, held_to_expiration=True,
                total_bought=100, total_sold=0,
                resolution_date=now - timedelta(days=i),
            )
            for i in range(15)
        ]
        config = ScoringConfig(min_resolved_markets=10)
        score = compute_wallet_score("0xlow", results, config)
        assert score is None

    def test_consistency_path_uses_config_min_roi_stdev(self):
        """The min_roi_stdev threshold comes from config, not hardcoded."""
        now = datetime.utcnow()
        results = [
            WalletMarketResult(
                wallet="0xcfg", market_id=f"m{i}", won=True,
                roi=0.001 + 0.0001 * i,  # stdev ~0.0004
                held_to_expiration=True, total_bought=100, total_sold=0,
                resolution_date=now - timedelta(days=i),
            )
            for i in range(15)
        ]
        # With default min_roi_stdev=0.001, stdev ~0.0004 < threshold → consistency path
        config = ScoringConfig(min_resolved_markets=10)
        score = compute_wallet_score("0xcfg", results, config)
        assert score is not None
        assert score.sharpe_ratio == 0.0  # consistency path

        # With very low threshold, same wallet goes through Sharpe path
        config2 = ScoringConfig(min_resolved_markets=10, min_roi_stdev=0.0001)
        score2 = compute_wallet_score("0xcfg", results, config2)
        assert score2 is not None
        assert score2.sharpe_ratio != 0.0  # Sharpe path


class TestZScore:
    def test_95_confidence(self):
        z = _z_score(0.95)
        assert z == pytest.approx(1.96, abs=0.01)

    def test_99_confidence(self):
        z = _z_score(0.99)
        assert z == pytest.approx(2.576, abs=0.01)

    def test_90_confidence(self):
        z = _z_score(0.90)
        assert z == pytest.approx(1.645, abs=0.01)
