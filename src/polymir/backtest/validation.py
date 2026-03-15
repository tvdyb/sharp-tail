"""Statistical validation: walk-forward, bootstrap, deflated Sharpe."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, stdev
from typing import Any

from polymir.backtest.data import HistoricalTrade
from polymir.backtest.engine import BacktestEngine
from polymir.backtest.metrics import BacktestResult
from polymir.config import AppConfig
from polymir.scanner import WalletMarketResult


@dataclass
class ValidationResult:
    """Result of a validation procedure."""

    method: str
    passed: bool
    details: dict[str, Any]

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.method}: {self.details}"


# ── Walk-Forward Validation ──────────────────────────────────────────


async def walk_forward(
    trades: list[HistoricalTrade],
    wallet_results: list[WalletMarketResult],
    config: AppConfig,
    train_months: int = 6,
    test_months: int = 2,
    latency_s: int = 60,
    top_n: int = 50,
) -> ValidationResult:
    """Rolling walk-forward: train on N months, test on M months, roll forward.

    Returns aggregate out-of-sample performance.
    """
    if not trades:
        return ValidationResult("walk_forward", True, {"detail": "no trades"})

    sorted_trades = sorted(trades, key=lambda t: t.timestamp)
    start = sorted_trades[0].timestamp
    end = sorted_trades[-1].timestamp

    oos_results: list[BacktestResult] = []
    window_start = start

    while True:
        train_end = _add_months(window_start, train_months)
        test_end = _add_months(train_end, test_months)

        if train_end >= end:
            break

        # Test on out-of-sample period
        test_trades = [
            t for t in sorted_trades
            if train_end <= t.timestamp < test_end
        ]

        if test_trades:
            # Use only wallet results resolved before the test period
            pit_results = [
                r for r in wallet_results
                if r.resolution_date is not None and r.resolution_date < train_end
            ]

            engine = BacktestEngine(config, latency_s=latency_s, top_n=top_n)
            result = await engine.run(
                trades=test_trades,
                wallet_results=pit_results,
            )
            oos_results.append(result)

        window_start = _add_months(window_start, test_months)

    if not oos_results:
        return ValidationResult("walk_forward", True, {"detail": "insufficient data for windows"})

    # Aggregate OOS metrics
    oos_pnl = sum(r.total_pnl for r in oos_results)
    oos_trades = sum(r.trades_executed for r in oos_results)
    all_pnl = []
    for r in oos_results:
        all_pnl.extend(r.pnl_series)

    oos_sharpe = 0.0
    if len(all_pnl) >= 2:
        avg = mean(all_pnl)
        sd = stdev(all_pnl)
        if sd > 0:
            oos_sharpe = (avg / sd) * math.sqrt(252)

    return ValidationResult(
        "walk_forward",
        passed=oos_sharpe > 0,
        details={
            "windows": len(oos_results),
            "oos_total_pnl": oos_pnl,
            "oos_trades": oos_trades,
            "oos_sharpe": oos_sharpe,
            "per_window_pnl": [r.total_pnl for r in oos_results],
            "per_window_sharpe": [r.sharpe_ratio for r in oos_results],
        },
    )


# ── In-Sample / Out-of-Sample ────────────────────────────────────────


async def in_out_sample_split(
    trades: list[HistoricalTrade],
    wallet_results: list[WalletMarketResult],
    config: AppConfig,
    train_fraction: float = 0.6,
    latency_s: int = 60,
    top_n: int = 50,
) -> ValidationResult:
    """Split data temporally: optimize on first N%, test on last (1-N)%."""
    if not trades:
        return ValidationResult("in_out_sample", True, {"detail": "no trades"})

    sorted_trades = sorted(trades, key=lambda t: t.timestamp)
    split_idx = int(len(sorted_trades) * train_fraction)
    train_trades = sorted_trades[:split_idx]
    test_trades = sorted_trades[split_idx:]

    if not train_trades or not test_trades:
        return ValidationResult("in_out_sample", True, {"detail": "insufficient data"})

    split_time = test_trades[0].timestamp

    engine = BacktestEngine(config, latency_s=latency_s, top_n=top_n)

    # In-sample
    is_wr = [r for r in wallet_results if r.resolution_date is not None and r.resolution_date < split_time]
    is_result = await engine.run(trades=train_trades, wallet_results=is_wr)

    # Out-of-sample
    oos_result = await engine.run(trades=test_trades, wallet_results=is_wr)

    return ValidationResult(
        "in_out_sample",
        passed=oos_result.sharpe_ratio > 0,
        details={
            "split_date": split_time.isoformat(),
            "in_sample_sharpe": is_result.sharpe_ratio,
            "in_sample_pnl": is_result.total_pnl,
            "in_sample_trades": is_result.trades_executed,
            "out_of_sample_sharpe": oos_result.sharpe_ratio,
            "out_of_sample_pnl": oos_result.total_pnl,
            "out_of_sample_trades": oos_result.trades_executed,
        },
    )


# ── Randomized Baseline ─────────────────────────────────────────────


async def randomized_baseline(
    trades: list[HistoricalTrade],
    wallet_results: list[WalletMarketResult],
    config: AppConfig,
    iterations: int = 1000,
    latency_s: int = 60,
    top_n: int = 50,
    seed: int | None = None,
) -> ValidationResult:
    """Shuffle wallet labels and run same strategy to test if scoring has edge.

    If random wallets produce similar returns, the scoring model has no value.
    """
    if not trades:
        return ValidationResult("randomized_baseline", True, {"detail": "no trades"})

    rng = random.Random(seed)

    # First run with real labels
    engine = BacktestEngine(config, latency_s=latency_s, top_n=top_n)
    real_result = await engine.run(trades=trades, wallet_results=wallet_results)
    real_sharpe = real_result.sharpe_ratio

    # Collect all unique wallets
    all_wallets = list({r.wallet for r in wallet_results})

    random_sharpes: list[float] = []
    for _ in range(iterations):
        # Shuffle wallet labels in results
        wallet_map = dict(zip(all_wallets, rng.sample(all_wallets, len(all_wallets))))
        shuffled_results = []
        for r in wallet_results:
            shuffled_results.append(WalletMarketResult(
                wallet=wallet_map[r.wallet],
                market_id=r.market_id,
                won=r.won,
                roi=r.roi,
                held_to_expiration=r.held_to_expiration,
                total_bought=r.total_bought,
                total_sold=r.total_sold,
                resolution_date=r.resolution_date,
            ))

        rand_result = await engine.run(trades=trades, wallet_results=shuffled_results)
        random_sharpes.append(rand_result.sharpe_ratio)

    # P-value: fraction of random runs that beat real
    p_value = sum(1 for s in random_sharpes if s >= real_sharpe) / iterations

    return ValidationResult(
        "randomized_baseline",
        passed=p_value < 0.05,
        details={
            "real_sharpe": real_sharpe,
            "random_mean_sharpe": mean(random_sharpes) if random_sharpes else 0,
            "random_std_sharpe": stdev(random_sharpes) if len(random_sharpes) >= 2 else 0,
            "p_value": p_value,
            "iterations": iterations,
        },
    )


# ── Bootstrap Confidence Intervals ──────────────────────────────────


def bootstrap_confidence_intervals(
    result: BacktestResult,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int | None = None,
) -> ValidationResult:
    """Bootstrap trade-level returns for confidence intervals on Sharpe and PnL."""
    pnl = result.pnl_series
    if len(pnl) < 5:
        return ValidationResult("bootstrap_ci", True, {"detail": "insufficient trades"})

    rng = random.Random(seed)
    boot_sharpes: list[float] = []
    boot_pnls: list[float] = []

    for _ in range(n_bootstrap):
        sample = [rng.choice(pnl) for _ in range(len(pnl))]
        boot_pnls.append(sum(sample))
        if len(sample) >= 2:
            avg = mean(sample)
            sd = stdev(sample)
            if sd > 0:
                boot_sharpes.append((avg / sd) * math.sqrt(252))
            else:
                boot_sharpes.append(0.0)

    alpha = (1 - ci_level) / 2
    boot_sharpes.sort()
    boot_pnls.sort()

    lo_idx = int(alpha * n_bootstrap)
    hi_idx = int((1 - alpha) * n_bootstrap) - 1

    sharpe_ci = (boot_sharpes[lo_idx], boot_sharpes[hi_idx])
    pnl_ci = (boot_pnls[lo_idx], boot_pnls[hi_idx])

    return ValidationResult(
        "bootstrap_ci",
        passed=sharpe_ci[0] > 0,  # Lower bound of CI is positive
        details={
            "sharpe_ci_95": sharpe_ci,
            "pnl_ci_95": pnl_ci,
            "bootstrap_mean_sharpe": mean(boot_sharpes),
            "bootstrap_std_sharpe": stdev(boot_sharpes) if len(boot_sharpes) >= 2 else 0,
            "n_bootstrap": n_bootstrap,
        },
    )


# ── Deflated Sharpe Ratio ────────────────────────────────────────────


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trades: int,
    n_strategies_tested: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> ValidationResult:
    """Compute deflated Sharpe ratio (Bailey & Lopez de Prado).

    Accounts for the number of strategy variants tested and
    non-normal return distribution.
    """
    if n_trades < 2 or n_strategies_tested < 1:
        return ValidationResult("deflated_sharpe", True, {"detail": "insufficient data"})

    # Expected max Sharpe under null (Euler-Mascheroni approximation)
    euler_mascheroni = 0.5772
    expected_max_sharpe = (
        (1 - euler_mascheroni) * math.sqrt(2 * math.log(n_strategies_tested))
        + euler_mascheroni * math.sqrt(2 * math.log(n_strategies_tested))
        if n_strategies_tested > 1
        else 0.0
    )

    # Variance of Sharpe ratio estimator
    sharpe_var = (
        1
        + 0.5 * observed_sharpe**2
        - skewness * observed_sharpe
        + ((kurtosis - 3) / 4) * observed_sharpe**2
    ) / n_trades

    if sharpe_var <= 0:
        return ValidationResult("deflated_sharpe", False, {"detail": "negative variance"})

    # Test statistic
    test_stat = (observed_sharpe - expected_max_sharpe) / math.sqrt(sharpe_var)

    # Approximate p-value using normal CDF
    p_value = 0.5 * (1 + math.erf(-test_stat / math.sqrt(2)))

    return ValidationResult(
        "deflated_sharpe",
        passed=p_value < 0.05,
        details={
            "observed_sharpe": observed_sharpe,
            "expected_max_sharpe": expected_max_sharpe,
            "test_statistic": test_stat,
            "p_value": p_value,
            "n_strategies_tested": n_strategies_tested,
            "sharpe_variance": sharpe_var,
        },
    )


# ── Minimum Track Record Length ──────────────────────────────────────


def minimum_track_record_length(
    observed_sharpe: float,
    n_trades: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    target_sharpe: float = 0.0,
) -> ValidationResult:
    """Compute min backtest length for Sharpe to be significant (Bailey & LdP)."""
    if n_trades < 2:
        return ValidationResult("min_track_record", True, {"detail": "insufficient data"})

    sharpe_diff = observed_sharpe - target_sharpe
    if sharpe_diff <= 0:
        return ValidationResult(
            "min_track_record",
            False,
            {"detail": "observed Sharpe <= target", "observed": observed_sharpe},
        )

    # MinTRL formula
    var_term = (
        1
        - skewness * observed_sharpe
        + ((kurtosis - 1) / 4) * observed_sharpe**2
    )

    # z-score for 95% confidence
    z = 1.96
    min_trl = var_term * (z / sharpe_diff) ** 2

    return ValidationResult(
        "min_track_record",
        passed=n_trades >= min_trl,
        details={
            "min_trades_needed": math.ceil(min_trl),
            "actual_trades": n_trades,
            "observed_sharpe": observed_sharpe,
            "sufficient": n_trades >= min_trl,
        },
    )


# ── Multiple Comparison Correction ──────────────────────────────────


def holm_bonferroni_correction(
    p_values: dict[str, float],
    alpha: float = 0.05,
) -> ValidationResult:
    """Apply Holm-Bonferroni correction to multiple p-values."""
    if not p_values:
        return ValidationResult("holm_bonferroni", True, {"detail": "no p-values"})

    n = len(p_values)
    sorted_pvals = sorted(p_values.items(), key=lambda x: x[1])

    adjusted: dict[str, dict[str, Any]] = {}
    for i, (name, pval) in enumerate(sorted_pvals):
        threshold = alpha / (n - i)
        adjusted[name] = {
            "raw_p": pval,
            "threshold": threshold,
            "significant": pval < threshold,
        }

    any_significant = any(v["significant"] for v in adjusted.values())

    return ValidationResult(
        "holm_bonferroni",
        passed=any_significant,
        details={
            "adjusted_results": adjusted,
            "n_tests": n,
        },
    )


# ── Helper ───────────────────────────────────────────────────────────


def _add_months(dt: datetime, months: int) -> datetime:
    """Add months to a datetime (approximate)."""
    month = dt.month - 1 + months
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, 28)
    return dt.replace(year=year, month=month, day=day)
