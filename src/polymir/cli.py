"""CLI entry points for polymir."""

from __future__ import annotations

import asyncio
import json

import click
import structlog

from polymir.logging import setup_logging

logger = structlog.get_logger()


@click.group()
@click.option("--log-level", default="INFO", help="Logging level")
def main(log_level: str) -> None:
    """Polymarket wallet mirror trading system."""
    setup_logging(log_level)


@main.command()
@click.option("--min-markets", default=20, help="Minimum resolved markets for scoring")
def scan(min_markets: int) -> None:
    """Score wallets by historical profitability on resolved markets."""
    async def _run() -> None:
        from polymir.scanner import WalletScanner
        from polymir.config import AppConfig
        from polymir.db import Database

        config = AppConfig.from_env()
        async with Database(config.db_path) as db:
            scanner = WalletScanner(config, db)
            await scanner.run()

    asyncio.run(_run())


@main.command()
@click.option("--top", default=50, help="Number of top wallets to watch")
def monitor(top: int) -> None:
    """Watch tracked wallets for new position entries."""
    async def _run() -> None:
        from polymir.monitor import TradeMonitor
        from polymir.config import AppConfig
        from polymir.db import Database

        config = AppConfig.from_env()
        async with Database(config.db_path) as db:
            mon = TradeMonitor(config, db)
            await mon.run()

    asyncio.run(_run())


@main.command()
def trade() -> None:
    """Start live mirror execution."""
    async def _run() -> None:
        from polymir.executor import MirrorExecutor
        from polymir.config import AppConfig
        from polymir.db import Database

        config = AppConfig.from_env()
        async with Database(config.db_path) as db:
            executor = MirrorExecutor(config, db)
            await executor.run()

    asyncio.run(_run())


@main.command()
@click.option("--latency", default=60, help="Simulated latency in seconds")
@click.option("--top-n", default=50, help="Number of top wallets to mirror")
@click.option("--start", default=None, help="Backtest start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="Backtest end date (YYYY-MM-DD)")
@click.option("--sweep", is_flag=True, help="Run parameter sweep")
@click.option("--output", default=None, help="Output file for results (JSON)")
def backtest(
    latency: int,
    top_n: int,
    start: str | None,
    end: str | None,
    sweep: bool,
    output: str | None,
) -> None:
    """Run backtest simulation of mirror strategy."""
    async def _run() -> None:
        from polymir.backtest.engine import BacktestEngine
        from polymir.config import AppConfig

        config = AppConfig.from_env()

        if sweep:
            from polymir.backtest.sweep import SweepConfig, run_sweep, sweep_results_to_json

            sweep_config = SweepConfig()
            results = await run_sweep(
                trades=[],
                wallet_results=[],
                config=config,
                sweep_config=sweep_config,
                start_date=start,
                end_date=end,
            )
            for label, result in sorted(
                results.items(), key=lambda x: x[1].sharpe_ratio, reverse=True
            ):
                click.echo(f"{label}: Sharpe={result.sharpe_ratio:.2f} PnL=${result.total_pnl:,.2f}")

            if output:
                with open(output, "w") as f:
                    f.write(sweep_results_to_json(results))
                click.echo(f"\nResults saved to {output}")
        else:
            engine = BacktestEngine(config, latency_s=latency, top_n=top_n)
            result = await engine.run(start_date=start, end_date=end)
            click.echo(result.summary())

            if output:
                with open(output, "w") as f:
                    json.dump(result.to_dict(), f, indent=2, default=str)
                click.echo(f"\nResults saved to {output}")

    asyncio.run(_run())


@main.command()
@click.option("--output-dir", default="reports", help="Output directory for report")
def report(output_dir: str) -> None:
    """Generate full backtest report with bias tests."""
    async def _run() -> None:
        from polymir.backtest.engine import BacktestEngine
        from polymir.backtest.report import generate_report
        from polymir.config import AppConfig

        config = AppConfig.from_env()
        engine = BacktestEngine(config)
        result = await engine.run()
        report_text = generate_report(result, output_dir=output_dir)
        click.echo(f"Report generated at {output_dir}/backtest_report.md")

    asyncio.run(_run())


# ── Alpha Research Commands ────────────────────────────────────────


@main.command()
@click.option("--db", default="research.db", help="Database path for research data")
def collect(db: str) -> None:
    """Collect market data for alpha research."""
    async def _run() -> None:
        from polymir.config import APIConfig
        from polymir.research.pipeline import collect_data

        config = APIConfig.from_env()
        markets, prices, quality = await collect_data(config, db_path=db)
        click.echo(f"Collected {len(markets)} markets, {len(prices)} price observations")
        click.echo(f"Categories: {quality.markets_by_category}")
        click.echo(f"Excluded (low volume): {quality.markets_excluded_low_volume}")
        click.echo(f"Excluded (few obs): {quality.markets_excluded_few_observations}")

    asyncio.run(_run())


@main.command()
@click.option("--strategy", default="all", help="Strategy number (1-7) or 'all'")
@click.option("--db", default="research.db", help="Database path for research data")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
def research(strategy: str, db: str, start: str | None, end: str | None) -> None:
    """Run alpha strategy backtests."""
    async def _run() -> None:
        from datetime import date as date_type

        from polymir.research.pipeline import load_data, run_research

        markets, prices = await load_data(db_path=db)
        click.echo(f"Loaded {len(markets)} markets, {len(prices)} prices")

        start_date = date_type.fromisoformat(start) if start else None
        end_date = date_type.fromisoformat(end) if end else None

        strategy_num = int(strategy) if strategy != "all" else None
        results = run_research(markets, prices, strategy_num=strategy_num,
                              start=start_date, end=end_date)

        for name, result in results.items():
            click.echo(f"\n{'='*60}")
            click.echo(f"Strategy: {name}")
            click.echo(f"  Trades:  {result.trade_count}")
            click.echo(f"  P&L:     ${result.total_pnl:,.2f}")
            click.echo(f"  Sharpe:  {result.sharpe_ratio:.2f}")
            click.echo(f"  Sortino: {result.sortino_ratio:.2f}")
            click.echo(f"  Max DD:  ${result.max_drawdown:,.2f}")
            click.echo(f"  Win %:   {result.win_rate:.1%}")

    asyncio.run(_run())


@main.command()
@click.option("--db", default="research.db", help="Database path for research data")
def validate(db: str) -> None:
    """Run bias audit and statistical validation on strategy results."""
    async def _run() -> None:
        from polymir.research.pipeline import load_data, run_research, run_validation

        markets, prices = await load_data(db_path=db)
        results = run_research(markets, prices)
        validations = run_validation(markets, prices, results)

        for name, vals in validations.items():
            click.echo(f"\n{'='*60}")
            click.echo(f"Strategy: {name}")
            for v in vals:
                status = "PASS" if v.passed else "FAIL"
                click.echo(f"  [{status}] {v.test_name}")
                if "p_value" in v.details:
                    click.echo(f"           p-value: {v.details['p_value']:.4f}")

    asyncio.run(_run())


@main.command(name="research-report")
@click.option("--db", default="research.db", help="Database path for research data")
@click.option("--output-dir", default="output", help="Output directory")
def research_report(db: str, output_dir: str) -> None:
    """Generate full alpha research report with charts."""
    async def _run() -> None:
        from polymir.data.collector import DataCollector
        from polymir.research.pipeline import (
            generate_full_report,
            load_data,
            run_research,
            run_validation,
        )

        markets, prices = await load_data(db_path=db)
        click.echo(f"Loaded {len(markets)} markets, {len(prices)} prices")

        collector = DataCollector(db_path=db)
        quality = collector.compute_quality_report(markets, prices)

        results = run_research(markets, prices)
        validations = run_validation(markets, prices, results)
        report_text = generate_full_report(
            markets, prices, results, validations,
            data_quality=quality, output_dir=output_dir,
        )
        click.echo(f"\nReport generated at {output_dir}/research_report.md")

    asyncio.run(_run())


@main.command()
@click.option("--db", default="research.db", help="Database path")
@click.option("--output-dir", default="output", help="Output directory")
@click.option("--strategy", default=None, help="Strategy number (1-7) or None for all")
def pipeline(db: str, output_dir: str, strategy: str | None) -> None:
    """Run the complete research pipeline: collect -> research -> validate -> report."""
    async def _run() -> None:
        from polymir.config import APIConfig
        from polymir.research.pipeline import full_pipeline

        config = APIConfig.from_env()
        strategy_num = int(strategy) if strategy else None
        await full_pipeline(
            config=config,
            db_path=db,
            output_dir=output_dir,
            strategy_num=strategy_num,
        )
        click.echo(f"\nPipeline complete. Report at {output_dir}/research_report.md")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
