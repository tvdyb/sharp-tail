"""End-to-end research pipeline orchestrator."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import structlog

from polymir.config import APIConfig
from polymir.data.collector import DataCollector
from polymir.research.engine import ResearchEngine
from polymir.research.models import (
    DataQualityReport,
    PriceSnapshot,
    ResearchMarket,
    StrategyResult,
)
from polymir.research.report import generate_report
from polymir.research.strategy_base import Strategy
from polymir.research.strategies.s1_favorite_longshot import FavoriteLongshotStrategy
from polymir.research.strategies.s2_multi_outcome_arb import MultiOutcomeArbStrategy
from polymir.research.strategies.s3_volume_momentum import VolumeMomentumStrategy
from polymir.research.strategies.s4_cross_market_cascade import CrossMarketCascadeStrategy
from polymir.research.strategies.s5_spread_capture import SpreadCaptureStrategy
from polymir.research.strategies.s6_theta_harvesting import ThetaHarvestingStrategy
from polymir.research.strategies.s7_external_divergence import ExternalDivergenceStrategy
from polymir.research.validation import ValidationResult, run_full_validation

logger = structlog.get_logger()

STRATEGY_REGISTRY: dict[int, tuple[type[Strategy], dict[str, Any]]] = {
    1: (FavoriteLongshotStrategy, {"entry_hours": 72.0, "price_bins": 10}),
    2: (MultiOutcomeArbStrategy, {"deviation_threshold": 0.03}),
    3: (VolumeMomentumStrategy, {"volume_window_hours": 24.0, "momentum_threshold": 0.10}),
    4: (CrossMarketCascadeStrategy, {"move_threshold": 0.05, "lag_hours": 6.0}),
    5: (SpreadCaptureStrategy, {"min_spread": 0.05, "vol_ceiling": 0.15}),
    6: (ThetaHarvestingStrategy, {"min_days_at_mid": 30, "vol_threshold": 0.05}),
    7: (ExternalDivergenceStrategy, {"divergence_threshold": 0.10}),
}

STRATEGY_NAMES: dict[int, str] = {
    1: "favorite_longshot",
    2: "multi_outcome_arb",
    3: "volume_momentum",
    4: "cross_market_cascade",
    5: "spread_capture",
    6: "theta_harvesting",
    7: "external_divergence",
}


async def collect_data(
    config: APIConfig | None = None,
    db_path: str = "research.db",
) -> tuple[list[ResearchMarket], list[PriceSnapshot], DataQualityReport]:
    """Phase 1: Collect all market data."""
    async with DataCollector(config=config, db_path=db_path) as collector:
        return await collector.run()


async def load_data(
    db_path: str = "research.db",
) -> tuple[list[ResearchMarket], list[PriceSnapshot]]:
    """Load previously collected data."""
    collector = DataCollector(db_path=db_path)
    async with collector:
        return await collector.load_from_db()


def run_research(
    markets: list[ResearchMarket],
    prices: list[PriceSnapshot],
    strategy_num: int | None = None,
    start: date | None = None,
    end: date | None = None,
) -> dict[str, StrategyResult]:
    """Phase 2-3: Run strategy backtests."""
    # Determine date range from data
    if start is None:
        timestamps = [p.timestamp for p in prices if p.timestamp]
        start = min(timestamps).date() if timestamps else date(2023, 1, 1)
    if end is None:
        timestamps = [p.timestamp for p in prices if p.timestamp]
        end = max(timestamps).date() if timestamps else date(2025, 12, 31)

    engine = ResearchEngine(markets, prices)
    results: dict[str, StrategyResult] = {}

    strategies_to_run = (
        {strategy_num: STRATEGY_REGISTRY[strategy_num]}
        if strategy_num and strategy_num in STRATEGY_REGISTRY
        else STRATEGY_REGISTRY
    )

    for num, (cls, default_params) in strategies_to_run.items():
        name = STRATEGY_NAMES[num]
        logger.info("running_strategy", number=num, name=name)

        try:
            strategy = cls(markets=markets, prices=prices, **default_params)
            result = engine.run_strategy(strategy, start, end)
            results[name] = result
            logger.info(
                "strategy_complete",
                name=name,
                trades=result.trade_count,
                pnl=result.total_pnl,
                sharpe=result.sharpe_ratio,
            )
        except Exception as e:
            logger.error("strategy_failed", name=name, error=str(e))
            results[name] = StrategyResult(strategy_name=name)

    return results


def run_validation(
    markets: list[ResearchMarket],
    prices: list[PriceSnapshot],
    strategy_results: dict[str, StrategyResult],
    start: date | None = None,
    end: date | None = None,
) -> dict[str, list[ValidationResult]]:
    """Phase 4: Run bias audit and statistical validation."""
    if start is None:
        timestamps = [p.timestamp for p in prices]
        start = min(timestamps).date() if timestamps else date(2023, 1, 1)
    if end is None:
        timestamps = [p.timestamp for p in prices]
        end = max(timestamps).date() if timestamps else date(2025, 12, 31)

    all_validations: dict[str, list[ValidationResult]] = {}

    for num, (cls, default_params) in STRATEGY_REGISTRY.items():
        name = STRATEGY_NAMES[num]
        result = strategy_results.get(name)
        if not result or result.trade_count == 0:
            continue

        logger.info("validating_strategy", name=name)

        try:
            validations = run_full_validation(
                strategy_class=cls,
                strategy_result=result,
                markets=markets,
                prices=prices,
                start=start,
                end=end,
                best_params=default_params,
                **default_params,
            )
            all_validations[name] = validations
        except Exception as e:
            logger.error("validation_failed", name=name, error=str(e))

    return all_validations


def generate_full_report(
    markets: list[ResearchMarket],
    prices: list[PriceSnapshot],
    strategy_results: dict[str, StrategyResult],
    validation_results: dict[str, list[ValidationResult]],
    data_quality: DataQualityReport | None = None,
    output_dir: str = "output",
) -> str:
    """Phase 5: Generate research report."""
    engine = ResearchEngine(markets, prices)

    # Correlation matrix
    corr = engine.correlation_matrix(strategy_results)

    # Portfolio metrics
    weights = engine.optimal_weights(strategy_results)
    portfolio = engine.portfolio_metrics(strategy_results, weights)

    # Walk-forward (simplified: use default params)
    timestamps = [p.timestamp for p in prices]
    start = min(timestamps).date() if timestamps else date(2023, 1, 1)
    end = max(timestamps).date() if timestamps else date(2025, 12, 31)

    walk_forward_results: dict[str, list[StrategyResult]] = {}
    for num, (cls, default_params) in STRATEGY_REGISTRY.items():
        name = STRATEGY_NAMES[num]
        if name not in strategy_results or strategy_results[name].trade_count == 0:
            continue
        try:
            wf = engine.walk_forward(cls, start, end, best_params=default_params)
            if wf:
                walk_forward_results[name] = wf
        except Exception as e:
            logger.error("walk_forward_failed", name=name, error=str(e))

    return generate_report(
        strategy_results=strategy_results,
        validation_results=validation_results,
        portfolio_metrics=portfolio,
        correlation_matrix=corr,
        data_quality=data_quality,
        walk_forward_results=walk_forward_results,
        output_dir=output_dir,
    )


async def full_pipeline(
    config: APIConfig | None = None,
    db_path: str = "research.db",
    output_dir: str = "output",
    strategy_num: int | None = None,
) -> str:
    """Run the complete pipeline: collect -> research -> validate -> report."""
    logger.info("pipeline_starting")

    # Phase 1: Data Collection
    logger.info("phase_1_data_collection")
    try:
        markets, prices, quality = await collect_data(config, db_path)
    except Exception as e:
        logger.warning("collection_failed_trying_cache", error=str(e))
        markets, prices = await load_data(db_path)
        quality = DataCollector(db_path=db_path).compute_quality_report(markets, prices)

    # Filter
    collector = DataCollector(db_path=db_path)
    markets, prices = collector.filter_markets(markets, prices)
    logger.info("data_ready", markets=len(markets), prices=len(prices))

    # Phase 2-3: Research
    logger.info("phase_2_3_research")
    results = run_research(markets, prices, strategy_num=strategy_num)

    # Phase 4: Validation
    logger.info("phase_4_validation")
    validations = run_validation(markets, prices, results)

    # Phase 5: Report
    logger.info("phase_5_report")
    report = generate_full_report(
        markets, prices, results, validations,
        data_quality=quality,
        output_dir=output_dir,
    )

    logger.info("pipeline_complete", report_path=f"{output_dir}/research_report.md")
    return report
