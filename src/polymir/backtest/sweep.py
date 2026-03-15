"""Parameter sweep for backtest optimization."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

import structlog

from polymir.backtest.data import HistoricalTrade
from polymir.backtest.engine import BacktestEngine
from polymir.backtest.metrics import BacktestResult
from polymir.config import AppConfig, ExecutionConfig
from polymir.scanner import WalletMarketResult

logger = structlog.get_logger()


@dataclass
class SweepConfig:
    """Configuration for parameter sweep."""

    latencies: list[int] = None  # type: ignore[assignment]
    top_ns: list[int] = None  # type: ignore[assignment]
    fee_rates: list[float] = None  # type: ignore[assignment]
    max_position_usds: list[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.latencies is None:
            self.latencies = [30, 60, 120, 300, 600]
        if self.top_ns is None:
            self.top_ns = [10, 20, 50, 100]
        if self.fee_rates is None:
            self.fee_rates = [0.0]
        if self.max_position_usds is None:
            self.max_position_usds = [1000.0]


async def run_sweep(
    trades: list[HistoricalTrade],
    wallet_results: list[WalletMarketResult],
    config: AppConfig,
    sweep_config: SweepConfig | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, BacktestResult]:
    """Run parameter sweep across latency and top-n values.

    Returns dict mapping parameter label to BacktestResult.
    """
    if sweep_config is None:
        sweep_config = SweepConfig()

    results: dict[str, BacktestResult] = {}
    total = (
        len(sweep_config.latencies)
        * len(sweep_config.top_ns)
        * len(sweep_config.fee_rates)
        * len(sweep_config.max_position_usds)
    )
    done = 0

    for latency in sweep_config.latencies:
        for top_n in sweep_config.top_ns:
            for fee_rate in sweep_config.fee_rates:
                for max_pos in sweep_config.max_position_usds:
                    label = f"lat={latency}_top={top_n}_fee={fee_rate}_pos={max_pos}"

                    # Build config variant
                    exec_cfg = ExecutionConfig(
                        stale_signal_timeout_s=config.execution.stale_signal_timeout_s,
                        fill_timeout_s=config.execution.fill_timeout_s,
                        aggression=config.execution.aggression,
                        max_position_usd=max_pos,
                        poll_interval_s=config.execution.poll_interval_s,
                        fee_rate=fee_rate,
                        slippage_per_trade=config.execution.slippage_per_trade,
                    )
                    variant = AppConfig(
                        api=config.api,
                        scoring=config.scoring,
                        execution=exec_cfg,
                        db_path=config.db_path,
                        top_wallets=top_n,
                        log_level=config.log_level,
                    )

                    engine = BacktestEngine(variant, latency_s=latency, top_n=top_n)
                    result = await engine.run(
                        trades=trades,
                        wallet_results=wallet_results,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    result.config_label = label
                    results[label] = result

                    done += 1
                    if done % 10 == 0 or done == total:
                        logger.info("sweep_progress", done=done, total=total)

    return results


def sweep_results_to_json(results: dict[str, BacktestResult]) -> str:
    """Serialize sweep results to JSON."""
    data = {label: r.to_dict() for label, r in results.items()}
    return json.dumps(data, indent=2, default=str)


def build_heatmap_data(
    results: dict[str, BacktestResult],
    x_param: str = "lat",
    y_param: str = "top",
    metric: str = "sharpe_ratio",
) -> dict[str, Any]:
    """Extract heatmap data from sweep results.

    Args:
        results: Sweep results dict.
        x_param: Parameter name prefix for x-axis.
        y_param: Parameter name prefix for y-axis.
        metric: Metric to plot.

    Returns:
        Dict with x_values, y_values, and z_matrix.
    """
    x_vals: set[str] = set()
    y_vals: set[str] = set()
    data: dict[tuple[str, str], float] = {}

    for label, result in results.items():
        params = dict(p.split("=") for p in label.split("_") if "=" in p)
        x_val = params.get(x_param, "")
        y_val = params.get(y_param, "")
        x_vals.add(x_val)
        y_vals.add(y_val)
        data[(x_val, y_val)] = getattr(result, metric, 0.0)

    x_sorted = sorted(x_vals, key=lambda v: float(v) if v.replace(".", "").isdigit() else v)
    y_sorted = sorted(y_vals, key=lambda v: float(v) if v.replace(".", "").isdigit() else v)

    z_matrix = []
    for y in y_sorted:
        row = [data.get((x, y), 0.0) for x in x_sorted]
        z_matrix.append(row)

    return {
        "x_values": x_sorted,
        "y_values": y_sorted,
        "z_matrix": z_matrix,
        "metric": metric,
    }
