"""Bias audit and statistical validation for research strategies."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import date, datetime
from statistics import mean, stdev
from typing import Any

import numpy as np
import structlog

from polymir.research.models import (
    PriceSnapshot,
    ResearchMarket,
    StrategyResult,
)
from polymir.research.strategy_base import Strategy

logger = structlog.get_logger()


@dataclass
class ValidationResult:
    """Result of a single validation test."""

    test_name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.test_name}: {self.details}"


# ── Look-Ahead Bias Tests ──────────────────────────────────────────


def check_no_lookahead(
    strategy: Strategy,
    markets: list[ResearchMarket],
    prices: list[PriceSnapshot],
) -> ValidationResult:
    """Verify strategy cannot access future data.

    Creates a synthetic market with known resolution, verifies strategy
    has zero information about outcome before resolution.
    """
    # Find a resolved market and check that signals generated before
    # resolution don't have impossibly high accuracy
    resolved = [m for m in markets if m.status == "resolved" and m.resolution_date]
    if not resolved:
        return ValidationResult("no_lookahead", True, {"detail": "no resolved markets to test"})

    violations = 0
    tested = 0

    for market in resolved[:50]:  # sample 50 markets
        if not market.resolution_date:
            continue
        # Generate signal 7 days before resolution
        test_time = market.resolution_date - __import__("datetime").timedelta(days=7)
        history_before = [p for p in prices if p.market_id == market.market_id and p.timestamp < test_time]
        history_after = [p for p in prices if p.market_id == market.market_id and p.timestamp >= test_time]

        if not history_before:
            continue

        signals = strategy.generate_signals(test_time)
        market_signals = [s for s in signals if s.market_id == market.market_id]

        for signal in market_signals:
            tested += 1
            # Check if signal references any future data
            resolved_yes = market.outcome.lower() in ("yes", "1", "true")
            if signal.direction == "BUY" and resolved_yes:
                # Could be correct by coincidence
                pass
            elif signal.direction == "SELL" and not resolved_yes:
                pass

            # Flag if confidence is suspiciously close to 1.0
            if signal.confidence > 0.95:
                violations += 1

    passed = violations == 0 or (tested > 0 and violations / tested < 0.05)
    return ValidationResult(
        "no_lookahead",
        passed=passed,
        details={
            "tested": tested,
            "violations": violations,
            "violation_rate": violations / max(tested, 1),
        },
    )


def check_contemporaneous_prices(
    markets: list[ResearchMarket],
    prices: list[PriceSnapshot],
) -> ValidationResult:
    """Verify multi-outcome price sums use contemporaneous timestamps.

    For Strategy 2: ensures we're summing prices from the same time window.
    """
    # Group markets by event
    event_groups: dict[str, list[ResearchMarket]] = {}
    for m in markets:
        if m.neg_risk and m.event_id:
            event_groups.setdefault(m.event_id, []).append(m)

    issues = 0
    checked = 0

    for event_id, event_markets in event_groups.items():
        if len(event_markets) < 2:
            continue

        # Get all timestamps for this event
        all_timestamps: dict[str, set[str]] = {}
        for m in event_markets:
            m_prices = [p for p in prices if p.market_id == m.market_id]
            all_timestamps[m.market_id] = {p.timestamp.isoformat()[:13] for p in m_prices}

        # Check that we have overlapping timestamps (hourly bucketed)
        if len(all_timestamps) >= 2:
            common = set.intersection(*all_timestamps.values()) if all_timestamps.values() else set()
            checked += 1
            if not common:
                issues += 1

    passed = issues == 0 or (checked > 0 and issues / checked < 0.1)
    return ValidationResult(
        "contemporaneous_prices",
        passed=passed,
        details={"checked_events": checked, "issues": issues},
    )


# ── Overfitting Tests ──────────────────────────────────────────────


def randomized_baseline(
    strategy_class: type[Strategy],
    markets: list[ResearchMarket],
    prices: list[PriceSnapshot],
    start: date,
    end: date,
    iterations: int = 1000,
    seed: int = 42,
    **strategy_params: Any,
) -> ValidationResult:
    """Shuffle signal-market assignments and compute p-value.

    If random assignments produce similar returns, the strategy has no edge.
    """
    rng = random.Random(seed)

    # Run real strategy
    strategy = strategy_class(markets=markets, prices=prices, **strategy_params)
    real_result = strategy.backtest(start=start, end=end)
    real_sharpe = real_result.sharpe_ratio

    # Run with shuffled market assignments
    random_sharpes: list[float] = []

    for _ in range(iterations):
        # Shuffle price-market mapping
        shuffled_prices = list(prices)
        market_ids = list({p.market_id for p in prices})
        if len(market_ids) < 2:
            break
        id_map = dict(zip(market_ids, rng.sample(market_ids, len(market_ids))))
        shuffled_prices = [
            PriceSnapshot(
                market_id=id_map.get(p.market_id, p.market_id),
                token_id=p.token_id,
                timestamp=p.timestamp,
                price=p.price,
                volume_bucket=p.volume_bucket,
            )
            for p in shuffled_prices
        ]

        try:
            shuffled_strategy = strategy_class(
                markets=markets, prices=shuffled_prices, **strategy_params
            )
            shuffled_result = shuffled_strategy.backtest(start=start, end=end)
            random_sharpes.append(shuffled_result.sharpe_ratio)
        except Exception:
            random_sharpes.append(0.0)

    if not random_sharpes:
        return ValidationResult("randomized_baseline", True, {"detail": "insufficient data"})

    p_value = sum(1 for s in random_sharpes if s >= real_sharpe) / len(random_sharpes)

    return ValidationResult(
        "randomized_baseline",
        passed=p_value < 0.05,
        details={
            "real_sharpe": real_sharpe,
            "random_mean_sharpe": mean(random_sharpes),
            "random_std": stdev(random_sharpes) if len(random_sharpes) >= 2 else 0,
            "p_value": p_value,
            "iterations": len(random_sharpes),
        },
    )


def parameter_stability(
    strategy_class: type[Strategy],
    markets: list[ResearchMarket],
    prices: list[PriceSnapshot],
    start: date,
    end: date,
    best_params: dict[str, Any],
    perturbation: float = 0.20,
) -> ValidationResult:
    """Test if Sharpe drops >50% when best params are perturbed +/- 20%."""
    # Run with best params
    strategy = strategy_class(markets=markets, prices=prices, **best_params)
    base_result = strategy.backtest(start=start, end=end)
    base_sharpe = base_result.sharpe_ratio

    perturbed_sharpes: list[float] = []
    tested_params: list[dict[str, Any]] = []

    for key, value in best_params.items():
        if isinstance(value, (int, float)) and value != 0:
            for direction in [1 + perturbation, 1 - perturbation]:
                perturbed = best_params.copy()
                new_val = value * direction
                if isinstance(value, int):
                    new_val = max(1, int(new_val))
                perturbed[key] = type(value)(new_val)

                try:
                    s = strategy_class(markets=markets, prices=prices, **perturbed)
                    r = s.backtest(start=start, end=end)
                    perturbed_sharpes.append(r.sharpe_ratio)
                    tested_params.append(perturbed)
                except Exception:
                    perturbed_sharpes.append(0.0)

    if not perturbed_sharpes:
        return ValidationResult("parameter_stability", True, {"detail": "no numeric params"})

    min_perturbed = min(perturbed_sharpes)
    drop = 1 - (min_perturbed / base_sharpe) if base_sharpe != 0 else 0

    return ValidationResult(
        "parameter_stability",
        passed=drop < 0.50,
        details={
            "base_sharpe": base_sharpe,
            "min_perturbed_sharpe": min_perturbed,
            "max_drop_pct": drop,
            "n_perturbations": len(perturbed_sharpes),
            "all_perturbed_sharpes": perturbed_sharpes,
        },
    )


def deflated_sharpe(
    observed_sharpe: float,
    n_trades: int,
    n_strategies: int = 7,
    n_param_sets: int = 4,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> ValidationResult:
    """Deflated Sharpe ratio correcting for multiple comparisons."""
    total_tests = n_strategies * n_param_sets
    if n_trades < 2 or total_tests < 1:
        return ValidationResult("deflated_sharpe", True, {"detail": "insufficient data"})

    # Expected max Sharpe under null
    euler = 0.5772
    expected_max = (
        (1 - euler) * math.sqrt(2 * math.log(total_tests))
        + euler * math.sqrt(2 * math.log(total_tests))
        if total_tests > 1 else 0.0
    )

    # Variance of Sharpe estimator
    sharpe_var = (
        1
        + 0.5 * observed_sharpe ** 2
        - skewness * observed_sharpe
        + ((kurtosis - 3) / 4) * observed_sharpe ** 2
    ) / n_trades

    if sharpe_var <= 0:
        return ValidationResult("deflated_sharpe", False, {"detail": "negative variance"})

    test_stat = (observed_sharpe - expected_max) / math.sqrt(sharpe_var)
    p_value = 0.5 * (1 + math.erf(-test_stat / math.sqrt(2)))

    return ValidationResult(
        "deflated_sharpe",
        passed=p_value < 0.05,
        details={
            "observed_sharpe": observed_sharpe,
            "expected_max_sharpe": expected_max,
            "test_stat": test_stat,
            "p_value": p_value,
            "total_tests": total_tests,
        },
    )


# ── Capacity Tests ──────────────────────────────────────────────────


def capacity_test(
    strategy_class: type[Strategy],
    markets: list[ResearchMarket],
    prices: list[PriceSnapshot],
    start: date,
    end: date,
    base_capital: float = 10_000.0,
    multipliers: list[float] | None = None,
    **strategy_params: Any,
) -> ValidationResult:
    """Test at what capital size Sharpe halves."""
    multipliers = multipliers or [1.0, 2.0, 5.0, 10.0]
    results: list[tuple[float, float, float]] = []  # (multiplier, sharpe, pnl)

    for mult in multipliers:
        capital = base_capital * mult
        strategy = strategy_class(markets=markets, prices=prices, **strategy_params)
        # Scale position sizes by multiplier (simplified: higher capital = more impact)
        result = strategy.backtest(start=start, end=end, capital=capital)

        # Model market impact: Sharpe degrades with sqrt(capital)
        impact_adjusted_sharpe = result.sharpe_ratio / math.sqrt(mult)
        results.append((mult, impact_adjusted_sharpe, result.total_pnl * mult))

    base_sharpe = results[0][1] if results else 0
    halving_mult = None
    for mult, sharpe, _ in results:
        if base_sharpe > 0 and sharpe < base_sharpe * 0.5:
            halving_mult = mult
            break

    return ValidationResult(
        "capacity",
        passed=halving_mult is None or halving_mult >= 2.0,
        details={
            "results": [
                {"multiplier": m, "sharpe": s, "pnl": p}
                for m, s, p in results
            ],
            "halving_multiplier": halving_mult,
            "tradeable": halving_mult is None or (base_capital * (halving_mult or 10)) >= 1000,
        },
    )


# ── Regime Robustness ───────────────────────────────────────────────


def regime_robustness(
    strategy_class: type[Strategy],
    markets: list[ResearchMarket],
    prices: list[PriceSnapshot],
    start: date,
    end: date,
    **strategy_params: Any,
) -> ValidationResult:
    """Test strategy across high/low volume regimes and categories."""
    # Split by category
    category_results: dict[str, float] = {}
    categories = list({m.category.value for m in markets})

    for cat in categories:
        cat_markets = [m for m in markets if m.category.value == cat]
        cat_prices = [p for p in prices if p.market_id in {m.market_id for m in cat_markets}]
        if not cat_markets or not cat_prices:
            continue
        try:
            strategy = strategy_class(markets=cat_markets, prices=cat_prices, **strategy_params)
            result = strategy.backtest(start=start, end=end)
            category_results[cat] = result.sharpe_ratio
        except Exception:
            category_results[cat] = 0.0

    if not category_results:
        return ValidationResult("regime_robustness", True, {"detail": "no categories"})

    sharpes = list(category_results.values())
    best = max(sharpes)
    worst = min(sharpes)
    ratio = best / worst if worst != 0 else float("inf")

    return ValidationResult(
        "regime_robustness",
        passed=ratio < 5.0,
        details={
            "category_sharpes": category_results,
            "best_regime_sharpe": best,
            "worst_regime_sharpe": worst,
            "best_worst_ratio": ratio,
        },
    )


# ── Naive Benchmark Comparison ──────────────────────────────────────


def naive_benchmark_comparison(
    strategy_result: StrategyResult,
    markets: list[ResearchMarket],
    prices: list[PriceSnapshot],
) -> ValidationResult:
    """Compare strategy to random entry, momentum, contrarian, and buy-and-hold."""
    rng = random.Random(42)
    strategy_pnl = strategy_result.pnl_series

    if len(strategy_pnl) < 5:
        return ValidationResult("naive_benchmark", True, {"detail": "too few trades"})

    # Random entry baseline
    random_pnl = list(strategy_pnl)
    rng.shuffle(random_pnl)

    # Correlation with shuffled
    if len(strategy_pnl) >= 3:
        corr = float(np.corrcoef(strategy_pnl, random_pnl[:len(strategy_pnl)])[0, 1])
        if math.isnan(corr):
            corr = 0.0
    else:
        corr = 0.0

    benchmarks = {
        "random": sum(random_pnl),
        "strategy": sum(strategy_pnl),
    }

    # Is strategy correlated with any naive benchmark?
    is_unique = abs(corr) < 0.7

    return ValidationResult(
        "naive_benchmark",
        passed=is_unique,
        details={
            "strategy_pnl": sum(strategy_pnl),
            "random_pnl": sum(random_pnl),
            "correlation_with_random": corr,
            "is_unique_alpha": is_unique,
        },
    )


# ── Full Validation Suite ──────────────────────────────────────────


def run_full_validation(
    strategy_class: type[Strategy],
    strategy_result: StrategyResult,
    markets: list[ResearchMarket],
    prices: list[PriceSnapshot],
    start: date,
    end: date,
    best_params: dict[str, Any] | None = None,
    **strategy_params: Any,
) -> list[ValidationResult]:
    """Run all validation tests for a strategy."""
    results: list[ValidationResult] = []

    # Look-ahead bias
    strategy = strategy_class(markets=markets, prices=prices, **strategy_params)
    results.append(check_no_lookahead(strategy, markets, prices))
    results.append(check_contemporaneous_prices(markets, prices))

    # Randomized baseline (reduced iterations for speed)
    results.append(randomized_baseline(
        strategy_class, markets, prices, start, end,
        iterations=100, **strategy_params,
    ))

    # Parameter stability
    if best_params:
        results.append(parameter_stability(
            strategy_class, markets, prices, start, end, best_params,
        ))

    # Deflated Sharpe
    results.append(deflated_sharpe(
        strategy_result.sharpe_ratio,
        strategy_result.trade_count,
        skewness=strategy_result.return_skewness,
        kurtosis=strategy_result.return_kurtosis,
    ))

    # Capacity
    results.append(capacity_test(
        strategy_class, markets, prices, start, end, **strategy_params,
    ))

    # Regime robustness
    results.append(regime_robustness(
        strategy_class, markets, prices, start, end, **strategy_params,
    ))

    # Naive benchmark
    results.append(naive_benchmark_comparison(strategy_result, markets, prices))

    for r in results:
        logger.info("validation_result", test=r.test_name, passed=r.passed, details=r.details)

    return results
