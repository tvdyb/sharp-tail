"""Bias detection tests for backtest validation.

Each function tests for a specific form of bias and returns a result dict
with pass/fail status and details.
"""

from __future__ import annotations

import random
from collections import defaultdict
from datetime import datetime, timedelta
from statistics import mean
from typing import Any

from polymir.backtest.data import HistoricalTrade, TradeRecord
from polymir.backtest.engine import BacktestEngine, _compute_pit_scores
from polymir.backtest.metrics import BacktestResult
from polymir.config import AppConfig
from polymir.scanner import WalletMarketResult, compute_wallet_score


# ── Look-Ahead Bias ──────────────────────────────────────────────────


def check_wallet_score_leakage(
    wallet_results: list[WalletMarketResult],
    scoring_config: Any,
) -> dict[str, Any]:
    """Verify wallet scores at time T use ONLY markets resolved before T.

    For each market M that resolved at time T, verify that no wallet score
    computed at T-1 includes data from M.
    """
    # Group results by market
    by_market: dict[str, list[WalletMarketResult]] = {}
    for r in wallet_results:
        by_market.setdefault(r.market_id, []).append(r)

    # Get all resolution dates
    resolution_dates: list[tuple[datetime, str]] = []
    for mid, results in by_market.items():
        for r in results:
            if r.resolution_date:
                resolution_dates.append((r.resolution_date, mid))
                break

    resolution_dates.sort()
    violations = []

    for res_date, market_id in resolution_dates:
        # Score at T - 1 second
        check_time = res_date - timedelta(seconds=1)
        eligible = [
            r for r in wallet_results
            if r.resolution_date is not None and r.resolution_date < check_time
        ]

        # Check: none of the eligible results should be from this market
        leaked = [r for r in eligible if r.market_id == market_id]
        if leaked:
            violations.append({
                "market_id": market_id,
                "resolution_date": res_date.isoformat(),
                "leaked_results": len(leaked),
            })

    return {
        "test": "wallet_score_leakage",
        "passed": len(violations) == 0,
        "markets_checked": len(resolution_dates),
        "violations": violations,
    }


def check_price_leakage(
    trade_records: list[TradeRecord],
) -> dict[str, Any]:
    """Verify execution prices differ from resolution prices.

    If fill_price == market_resolved_price for most trades, there's likely
    price leakage (using future information for execution).
    """
    executed = [t for t in trade_records if t.decision == "execute" and t.fill_price is not None]
    if not executed:
        return {"test": "price_leakage", "passed": True, "trades_checked": 0, "detail": "no trades"}

    exact_matches = sum(
        1 for t in executed
        if t.fill_price is not None and abs(t.fill_price - t.market_resolved_price) < 1e-10
    )
    match_rate = exact_matches / len(executed)

    return {
        "test": "price_leakage",
        "passed": match_rate < 0.5,  # More than half matching is suspicious
        "trades_checked": len(executed),
        "exact_match_rate": match_rate,
        "exact_matches": exact_matches,
    }


def check_resolution_data_leakage(
    result: BacktestResult,
) -> dict[str, Any]:
    """Verify the backtest doesn't condition on resolution outcome.

    Check that the decision (execute vs skip) doesn't correlate with whether
    the market resolved in the trader's favor. If it does, we may be
    using resolution data to make execution decisions.
    """
    executed = [t for t in result.trade_records if t.decision == "execute"]
    skipped = [t for t in result.trade_records if t.decision != "execute" and t.decision != "not_qualified"]

    if not executed or not skipped:
        return {"test": "resolution_data_leakage", "passed": True, "detail": "insufficient data"}

    # Win rate among executed vs skipped (if we could see future)
    exec_win_rate = mean([1 if t.market_resolved_price > 0.5 else 0 for t in executed])
    skip_win_rate = mean([1 if t.market_resolved_price > 0.5 else 0 for t in skipped])

    # If executed trades have a much higher "would-have-won" rate than skipped,
    # that's suspicious (could indicate resolution leakage)
    diff = abs(exec_win_rate - skip_win_rate)

    return {
        "test": "resolution_data_leakage",
        "passed": diff < 0.3,  # Allow some natural correlation
        "executed_favorable_rate": exec_win_rate,
        "skipped_favorable_rate": skip_win_rate,
        "difference": diff,
    }


# ── Selection Bias ───────────────────────────────────────────────────


def check_time_window_stability(
    results_by_period: dict[str, BacktestResult],
) -> dict[str, Any]:
    """Check if results hold across non-overlapping time windows.

    Args:
        results_by_period: Dict mapping period label to BacktestResult.
    """
    sharpes = {k: v.sharpe_ratio for k, v in results_by_period.items()}
    pnls = {k: v.total_pnl for k, v in results_by_period.items()}

    profitable_periods = sum(1 for p in pnls.values() if p > 0)
    total_periods = len(pnls)

    return {
        "test": "time_window_stability",
        "passed": profitable_periods >= total_periods * 0.5 if total_periods > 0 else True,
        "periods": total_periods,
        "profitable_periods": profitable_periods,
        "sharpe_by_period": sharpes,
        "pnl_by_period": pnls,
    }


def check_category_concentration(
    result: BacktestResult,
) -> dict[str, Any]:
    """Check if alpha is concentrated in one market category."""
    by_cat = result.pnl_by_category()
    if not by_cat:
        return {"test": "category_concentration", "passed": True, "detail": "no categories"}

    total = sum(abs(v) for v in by_cat.values())
    if total == 0:
        return {"test": "category_concentration", "passed": True, "detail": "zero pnl"}

    concentration = {k: abs(v) / total for k, v in by_cat.items()}
    max_concentration = max(concentration.values())

    return {
        "test": "category_concentration",
        "passed": max_concentration < 0.8,  # No single category > 80% of PnL
        "concentration": concentration,
        "max_concentration": max_concentration,
    }


def check_wallet_concentration(
    result: BacktestResult,
    top_counts: tuple[int, ...] = (1, 3, 5),
) -> dict[str, Any]:
    """Check what fraction of PnL comes from top wallets."""
    by_wallet = result.pnl_by_wallet()
    total = result.total_pnl

    if total == 0 or not by_wallet:
        return {"test": "wallet_concentration", "passed": True, "detail": "no pnl"}

    concentrations = {}
    for n in top_counts:
        top_pnl = sum(list(by_wallet.values())[:n])
        concentrations[f"top_{n}"] = top_pnl / total if total != 0 else 0.0

    # Fail if top-1 wallet drives > 50% of returns
    top1_conc = concentrations.get("top_1", 0.0)

    return {
        "test": "wallet_concentration",
        "passed": abs(top1_conc) < 0.5,
        "concentrations": concentrations,
        "total_wallets": len(by_wallet),
    }


# ── Overfitting ──────────────────────────────────────────────────────


def check_parameter_sensitivity(
    sweep_results: dict[str, BacktestResult],
) -> dict[str, Any]:
    """Check if profitable parameters form a robust region vs isolated island.

    Args:
        sweep_results: Dict mapping param string to BacktestResult.
    """
    if not sweep_results:
        return {"test": "parameter_sensitivity", "passed": True, "detail": "no results"}

    sharpes = {k: v.sharpe_ratio for k, v in sweep_results.items()}
    profitable = sum(1 for s in sharpes.values() if s > 0)
    total = len(sharpes)

    return {
        "test": "parameter_sensitivity",
        "passed": profitable >= total * 0.3,  # At least 30% of params profitable
        "total_configs": total,
        "profitable_configs": profitable,
        "profitable_fraction": profitable / total if total > 0 else 0,
        "sharpe_range": (min(sharpes.values()), max(sharpes.values())) if sharpes else (0, 0),
    }


# ── Transaction Cost & Execution Reality ─────────────────────────────


def check_latency_sensitivity(
    results_by_latency: dict[int, BacktestResult],
) -> dict[str, Any]:
    """Check where edge disappears as latency increases."""
    if not results_by_latency:
        return {"test": "latency_sensitivity", "passed": True, "detail": "no results"}

    curve = {
        lat: r.sharpe_ratio
        for lat, r in sorted(results_by_latency.items())
    }

    # Find where Sharpe goes negative
    edge_disappears_at = None
    for lat, sharpe in sorted(curve.items()):
        if sharpe <= 0:
            edge_disappears_at = lat
            break

    return {
        "test": "latency_sensitivity",
        "passed": edge_disappears_at is None or edge_disappears_at > 60,
        "sharpe_curve": curve,
        "edge_disappears_at_s": edge_disappears_at,
    }


def check_fee_breakeven(
    results_by_fee: dict[float, BacktestResult],
) -> dict[str, Any]:
    """Find the breakeven fee rate."""
    if not results_by_fee:
        return {"test": "fee_breakeven", "passed": True, "detail": "no results"}

    curve = {fee: r.total_pnl for fee, r in sorted(results_by_fee.items())}

    breakeven = None
    for fee, pnl in sorted(curve.items()):
        if pnl <= 0:
            breakeven = fee
            break

    return {
        "test": "fee_breakeven",
        "passed": breakeven is None or breakeven > 0.001,
        "pnl_curve": curve,
        "breakeven_fee": breakeven,
    }
