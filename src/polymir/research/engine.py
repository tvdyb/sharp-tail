"""Research backtest engine — runs strategies with proper validation."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, datetime, timedelta
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


class ResearchEngine:
    """Runs strategy backtests with parameter sweeps and portfolio analysis."""

    def __init__(
        self,
        markets: list[ResearchMarket],
        prices: list[PriceSnapshot],
        capital: float = 10_000.0,
        fee_rate: float = 0.02,
        max_position_frac: float = 0.10,
    ) -> None:
        self._markets = markets
        self._prices = prices
        self._capital = capital
        self._fee_rate = fee_rate
        self._max_position_frac = max_position_frac

    def run_strategy(
        self,
        strategy: Strategy,
        start: date,
        end: date,
        **params: Any,
    ) -> StrategyResult:
        """Run a single strategy backtest."""
        return strategy.backtest(
            start=start,
            end=end,
            capital=self._capital,
            fee_rate=self._fee_rate,
            max_position_frac=self._max_position_frac,
            **params,
        )

    def parameter_sweep(
        self,
        strategy_class: type[Strategy],
        start: date,
        end: date,
        param_grid: dict[str, list[Any]],
    ) -> list[StrategyResult]:
        """Sweep over parameter combinations."""
        results = []
        combos = _param_combinations(param_grid)

        for combo in combos:
            try:
                strategy = strategy_class(
                    markets=self._markets,
                    prices=self._prices,
                    **combo,
                )
                result = self.run_strategy(strategy, start, end)
                result.params = combo
                results.append(result)
                logger.info(
                    "sweep_result",
                    strategy=strategy.name,
                    params=combo,
                    sharpe=result.sharpe_ratio,
                    pnl=result.total_pnl,
                )
            except Exception as e:
                logger.error("sweep_error", params=combo, error=str(e))

        return results

    def walk_forward(
        self,
        strategy_class: type[Strategy],
        start: date,
        end: date,
        train_months: int = 6,
        test_months: int = 2,
        best_params: dict[str, Any] | None = None,
        param_grid: dict[str, list[Any]] | None = None,
        optimize_per_window: bool = False,
    ) -> list[StrategyResult]:
        """Walk-forward out-of-sample testing.

        Args:
            optimize_per_window: If True and param_grid is provided, re-optimize
                parameters on each training window before testing OOS.
        """
        oos_results = []
        current = start

        while True:
            train_end = _add_months_date(current, train_months)
            test_end = _add_months_date(train_end, test_months)

            if train_end >= end:
                break

            actual_test_end = min(test_end, end)

            try:
                if optimize_per_window and param_grid:
                    # Re-optimize on training window
                    sweep_results = self.parameter_sweep(
                        strategy_class, current, train_end, param_grid,
                    )
                    if sweep_results:
                        best = max(sweep_results, key=lambda r: r.sharpe_ratio)
                        params = best.params
                    else:
                        params = best_params or {}
                else:
                    params = best_params or {}

                strategy = strategy_class(
                    markets=self._markets,
                    prices=self._prices,
                    **params,
                )
                result = self.run_strategy(strategy, train_end, actual_test_end)
                result.params["window"] = f"{train_end} to {actual_test_end}"
                oos_results.append(result)
                logger.info(
                    "walk_forward_window",
                    strategy=strategy.name,
                    window=f"{train_end}-{actual_test_end}",
                    trades=result.trade_count,
                    pnl=result.total_pnl,
                )
            except Exception as e:
                logger.error("walk_forward_error", error=str(e))

            current = _add_months_date(current, test_months)

        return oos_results

    @staticmethod
    def correlation_matrix(results: dict[str, StrategyResult]) -> dict[str, dict[str, float]]:
        """Compute return correlation matrix between strategies."""
        names = list(results.keys())
        if len(names) < 2:
            return {}

        # Align returns by trade index (simple approach)
        series = {}
        for name, result in results.items():
            series[name] = result.pnl_series

        matrix: dict[str, dict[str, float]] = {}
        for n1 in names:
            matrix[n1] = {}
            for n2 in names:
                s1 = series[n1]
                s2 = series[n2]
                min_len = min(len(s1), len(s2))
                if min_len < 3:
                    matrix[n1][n2] = 0.0
                else:
                    corr = float(np.corrcoef(s1[:min_len], s2[:min_len])[0, 1])
                    matrix[n1][n2] = corr if not math.isnan(corr) else 0.0

        return matrix

    @staticmethod
    def portfolio_metrics(
        results: dict[str, StrategyResult],
        weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Compute combined portfolio metrics."""
        if not results:
            return {}

        if weights is None:
            n = len(results)
            weights = {name: 1.0 / n for name in results}

        # Combine P&L series (weighted)
        max_len = max(len(r.pnl_series) for r in results.values())
        combined_pnl = []

        for i in range(max_len):
            period_pnl = 0.0
            for name, result in results.items():
                w = weights.get(name, 0)
                if i < len(result.pnl_series):
                    period_pnl += result.pnl_series[i] * w
            combined_pnl.append(period_pnl)

        total_pnl = sum(combined_pnl)
        cum_pnl = []
        total = 0.0
        for p in combined_pnl:
            total += p
            cum_pnl.append(total)

        # Metrics
        sharpe = 0.0
        if len(combined_pnl) >= 2:
            avg = mean(combined_pnl)
            sd = stdev(combined_pnl)
            if sd > 0:
                sharpe = (avg / sd) * math.sqrt(252)

        max_dd = 0.0
        if cum_pnl:
            peak = cum_pnl[0]
            for val in cum_pnl:
                peak = max(peak, val)
                max_dd = max(max_dd, peak - val)

        return {
            "total_pnl": total_pnl,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "trade_count": sum(r.trade_count for r in results.values()),
            "weights": weights,
            "cumulative_pnl": cum_pnl,
        }

    @staticmethod
    def optimal_weights(
        results: dict[str, StrategyResult],
    ) -> dict[str, float]:
        """Mean-variance optimal weights (long-only, sum-to-1)."""
        names = list(results.keys())
        n = len(names)
        if n == 0:
            return {}
        if n == 1:
            return {names[0]: 1.0}

        # Build return matrix
        max_len = max(len(r.pnl_series) for r in results.values())
        returns_matrix = np.zeros((max_len, n))
        for j, name in enumerate(names):
            pnl = results[name].pnl_series
            for i in range(min(len(pnl), max_len)):
                returns_matrix[i, j] = pnl[i]

        # Mean returns and covariance
        mean_returns = np.mean(returns_matrix, axis=0)
        cov = np.cov(returns_matrix.T)

        if cov.ndim < 2:
            return {name: 1.0 / n for name in names}

        # Simple inverse-variance weighting (more robust than full MVO)
        try:
            diag = np.diag(cov)
            diag = np.where(diag > 0, diag, 1e-6)
            inv_var = 1.0 / diag
            weights = inv_var / inv_var.sum()
            return {names[i]: float(weights[i]) for i in range(n)}
        except Exception:
            return {name: 1.0 / n for name in names}


def _param_combinations(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Generate all combinations from a parameter grid."""
    if not grid:
        return [{}]

    keys = list(grid.keys())
    values = list(grid.values())

    combos = [{}]
    for key, vals in zip(keys, values):
        new_combos = []
        for combo in combos:
            for val in vals:
                new_combo = combo.copy()
                new_combo[key] = val
                new_combos.append(new_combo)
        combos = new_combos

    return combos


def _add_months_date(d: date, months: int) -> date:
    """Add months to a date."""
    month = d.month - 1 + months
    year = d.year + month // 12
    month = month % 12 + 1
    day = min(d.day, 28)
    return date(year, month, day)
