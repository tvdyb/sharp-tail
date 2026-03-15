#!/usr/bin/env python3
"""Collect real Polymarket data and generate comprehensive PDF research report."""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from polymir.config import APIConfig
from polymir.data.collector import DataCollector
from polymir.research.pipeline import (
    STRATEGY_NAMES,
    STRATEGY_REGISTRY,
    run_research,
    run_validation,
)
from polymir.research.engine import ResearchEngine
from polymir.logging import setup_logging

setup_logging("INFO")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "research.db")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


async def collect():
    """Collect real data from Polymarket APIs."""
    config = APIConfig()
    async with DataCollector(config=config, db_path=DB_PATH) as collector:
        markets, prices, quality = await collector.run()
    return markets, prices, quality


async def load():
    """Load previously collected data."""
    collector = DataCollector(db_path=DB_PATH)
    async with collector:
        markets, prices = await collector.load_from_db()
    quality = DataCollector(db_path=DB_PATH).compute_quality_report(markets, prices)
    return markets, prices, quality


async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "charts"), exist_ok=True)

    # Phase 1: Collect data (or load from cache)
    if os.path.exists(DB_PATH):
        print("Loading cached data from research.db ...")
        markets, prices, quality = await load()
        if len(markets) < 100:
            print(f"Only {len(markets)} markets cached, re-collecting...")
            markets, prices, quality = await collect()
    else:
        print("Collecting data from Polymarket APIs...")
        markets, prices, quality = await collect()

    # Filter
    collector = DataCollector(db_path=DB_PATH)
    markets, prices = collector.filter_markets(markets, prices, min_volume=500, min_observations=3)
    print(f"\nData ready: {len(markets)} markets, {len(prices)} price observations")
    print(f"Categories: {quality.markets_by_category}")

    # Phase 2: Run all strategies
    print("\nRunning 7 strategies...")
    results = run_research(markets, prices)
    for name, r in results.items():
        print(f"  {name}: {r.trade_count} trades, PnL=${r.total_pnl:,.2f}, Sharpe={r.sharpe_ratio:.2f}")

    # Phase 3: Validation
    print("\nRunning validation...")
    validations = run_validation(markets, prices, results)

    # Phase 4: Portfolio analysis
    engine = ResearchEngine(markets, prices)
    corr = engine.correlation_matrix(results)
    weights = engine.optimal_weights(results)
    portfolio = engine.portfolio_metrics(results, weights)

    # Walk-forward
    from datetime import date
    timestamps = [p.timestamp for p in prices if p.timestamp]
    start = min(timestamps).date() if timestamps else date(2023, 1, 1)
    end = max(timestamps).date() if timestamps else date(2025, 12, 31)

    wf_results = {}
    for num, (cls, params) in STRATEGY_REGISTRY.items():
        name = STRATEGY_NAMES[num]
        if name in results and results[name].trade_count > 0:
            try:
                wf = engine.walk_forward(cls, start, end, best_params=params)
                if wf:
                    wf_results[name] = wf
            except Exception:
                pass

    # Phase 5: Generate PDF
    print("\nGenerating PDF report...")
    from polymir.research.pdf_report import generate_pdf_report

    pdf_path = generate_pdf_report(
        strategy_results=results,
        validation_results=validations,
        portfolio_metrics=portfolio,
        correlation_matrix=corr,
        data_quality=quality,
        walk_forward_results=wf_results,
        markets=markets,
        prices=prices,
        output_dir=OUTPUT_DIR,
    )
    print(f"\nReport generated: {pdf_path}")


if __name__ == "__main__":
    asyncio.run(main())
