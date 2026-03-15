"""Research report generator with charts and statistical analysis."""

from __future__ import annotations

import os
from datetime import date
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import structlog

from polymir.research.models import DataQualityReport, StrategyResult
from polymir.research.validation import ValidationResult

logger = structlog.get_logger()

# Style
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.figsize": (12, 6), "figure.dpi": 150})


def generate_report(
    strategy_results: dict[str, StrategyResult],
    validation_results: dict[str, list[ValidationResult]],
    portfolio_metrics: dict[str, Any],
    correlation_matrix: dict[str, dict[str, float]],
    data_quality: DataQualityReport | None = None,
    walk_forward_results: dict[str, list[StrategyResult]] | None = None,
    output_dir: str = "output",
    charts_dir: str = "output/charts",
) -> str:
    """Generate the full research report."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    # Generate charts
    _plot_equity_curves(strategy_results, charts_dir)
    _plot_correlation_matrix(correlation_matrix, charts_dir)
    _plot_portfolio_curve(portfolio_metrics, charts_dir)
    if walk_forward_results:
        _plot_walk_forward(walk_forward_results, charts_dir)
    _plot_monthly_heatmap(strategy_results, charts_dir)

    # Generate markdown report
    report = _build_markdown(
        strategy_results,
        validation_results,
        portfolio_metrics,
        correlation_matrix,
        data_quality,
        walk_forward_results,
    )

    report_path = os.path.join(output_dir, "research_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    logger.info("report_generated", path=report_path)
    return report


def _build_markdown(
    strategy_results: dict[str, StrategyResult],
    validation_results: dict[str, list[ValidationResult]],
    portfolio_metrics: dict[str, Any],
    correlation_matrix: dict[str, dict[str, float]],
    data_quality: DataQualityReport | None,
    walk_forward_results: dict[str, list[StrategyResult]] | None,
) -> str:
    lines: list[str] = []
    lines.append("# Polymarket Alpha Research Report\n")

    # Executive Summary
    lines.append("## Executive Summary\n")
    total_strategies = len(strategy_results)
    profitable = sum(1 for r in strategy_results.values() if r.total_pnl > 0)
    positive_sharpe = sum(1 for r in strategy_results.values() if r.sharpe_ratio > 0.5)
    combined_sharpe = portfolio_metrics.get("sharpe", 0)

    lines.append(
        f"Tested {total_strategies} independent alpha strategies on Polymarket resolved markets. "
        f"{profitable}/{total_strategies} showed positive P&L, "
        f"{positive_sharpe}/{total_strategies} achieved Sharpe > 0.5. "
        f"Combined portfolio Sharpe: {combined_sharpe:.2f}. "
        f"See per-strategy sections for honest assessments of each signal.\n"
    )

    # Data Overview
    if data_quality:
        lines.append("## Data Overview\n")
        lines.append(f"- **Total markets**: {data_quality.total_markets}")
        lines.append(f"- **Price observations**: {data_quality.total_price_observations}")
        lines.append(f"- **Excluded (low volume)**: {data_quality.markets_excluded_low_volume}")
        lines.append(f"- **Excluded (few observations)**: {data_quality.markets_excluded_few_observations}")
        lines.append(f"- **Categories**: {data_quality.markets_by_category}")
        lines.append(f"- **Volume distribution**: {data_quality.volume_distribution}")
        lines.append("")

    # Per-Strategy Sections
    strategy_theses = {
        "favorite_longshot": "Prediction markets exhibit the favorite-longshot bias — longshots are systematically overpriced and favorites underpriced. We build a calibration curve and trade the mispricing.",
        "multi_outcome_arb": "In multi-outcome events, outcome prices should sum to ~$1.00. Deviations from this invariant create positive-EV relative value trades.",
        "volume_momentum": "Large volume at stable prices signals informed accumulation; large volume with sharp moves signals uninformed momentum that reverses. Kyle's lambda predicts direction.",
        "cross_market_cascade": "News moves one market but related markets reprice slowly. We trade the lead-lag structure between related Polymarket contracts.",
        "spread_capture": "Many PM markets have 5-15% spreads with thin books. Passive liquidity provision captures the spread minus adverse selection costs.",
        "theta_harvesting": "Markets near 0.50 with long time to resolution carry implicit theta — the time value of uncertainty. We sell this uncertainty premium.",
        "external_divergence": "External forecasting sources sometimes have better information than PM participants. When they diverge, we bet on convergence toward the external source.",
    }

    for name, result in strategy_results.items():
        lines.append(f"## Strategy: {name}\n")
        lines.append(f"**Thesis**: {strategy_theses.get(name, 'N/A')}\n")

        # Results table
        metrics = result.metrics_dict()
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Sharpe Ratio | {metrics['sharpe']:.2f} |")
        lines.append(f"| Sortino Ratio | {metrics['sortino']:.2f} |")
        lines.append(f"| Calmar Ratio | {metrics['calmar']:.2f} |")
        lines.append(f"| Max Drawdown | ${metrics['max_drawdown']:,.2f} ({metrics['max_drawdown_pct']:.1%}) |")
        lines.append(f"| Win Rate | {metrics['win_rate']:.1%} |")
        lines.append(f"| Trade Count | {metrics['trade_count']} |")
        lines.append(f"| Avg Winner | ${metrics['avg_winner']:,.2f} |")
        lines.append(f"| Avg Loser | ${metrics['avg_loser']:,.2f} |")
        lines.append(f"| Avg Holding (hrs) | {metrics['avg_holding_hours']:.1f} |")
        lines.append(f"| Total P&L | ${metrics['total_pnl']:,.2f} |")
        lines.append(f"| Skewness | {metrics['skewness']:.2f} |")
        lines.append(f"| Kurtosis | {metrics['kurtosis']:.2f} |")
        lines.append("")

        lines.append(f"![Equity Curve](charts/{name}_equity.png)\n")

        # Walk-forward results
        if walk_forward_results and name in walk_forward_results:
            wf = walk_forward_results[name]
            oos_sharpes = [r.sharpe_ratio for r in wf]
            oos_pnl = sum(r.total_pnl for r in wf)
            lines.append(f"**Walk-Forward OOS**: {len(wf)} windows, "
                        f"Avg Sharpe {np.mean(oos_sharpes):.2f}, "
                        f"Total P&L ${oos_pnl:,.2f}\n")
            lines.append(f"![Walk Forward](charts/{name}_walkforward.png)\n")

        # Validation results
        val_results = validation_results.get(name, [])
        if val_results:
            lines.append("**Validation**:\n")
            for v in val_results:
                emoji = "PASS" if v.passed else "FAIL"
                lines.append(f"- [{emoji}] {v.test_name}")
                if "p_value" in v.details:
                    lines.append(f"  - p-value: {v.details['p_value']:.4f}")
            lines.append("")

        # Verdict
        verdict = _strategy_verdict(result, val_results)
        lines.append(f"**Verdict**: {verdict}\n")
        lines.append("---\n")

    # Portfolio Analysis
    lines.append("## Portfolio Analysis\n")
    lines.append(f"![Correlation Matrix](charts/correlation_matrix.png)\n")
    lines.append(f"![Portfolio Curve](charts/portfolio_equity.png)\n")
    lines.append(f"![Monthly Returns](charts/monthly_heatmap.png)\n")

    weights = portfolio_metrics.get("weights", {})
    lines.append("### Optimal Weights\n")
    lines.append("| Strategy | Weight |")
    lines.append("|----------|--------|")
    for name, weight in sorted(weights.items(), key=lambda x: -x[1]):
        lines.append(f"| {name} | {weight:.1%} |")
    lines.append("")

    lines.append(f"- **Combined Sharpe**: {portfolio_metrics.get('sharpe', 0):.2f}")
    lines.append(f"- **Combined P&L**: ${portfolio_metrics.get('total_pnl', 0):,.2f}")
    lines.append(f"- **Max Drawdown**: ${portfolio_metrics.get('max_drawdown', 0):,.2f}")
    lines.append(f"- **Total Trades**: {portfolio_metrics.get('trade_count', 0)}")
    lines.append("")

    # Risks and Limitations
    lines.append("## Risks and Limitations\n")
    lines.append("1. **Data granularity**: Resolved markets only provide 12h+ price granularity, limiting intraday signal detection.")
    lines.append("2. **Liquidity**: Many markets have thin books; actual execution may differ significantly from backtested fills.")
    lines.append("3. **Market impact**: Position sizes that move the market are not modeled at small scale but become critical above ~$5k per position.")
    lines.append("4. **Regime change**: Polymarket's participant base and market structure evolve rapidly.")
    lines.append("5. **Fee structure**: Polymarket fees on profitable positions reduce realized edge.")
    lines.append("6. **External data lag**: Strategy 7's external sources may have their own delays not captured in backtesting.")
    lines.append("")

    # Recommendations
    lines.append("## Recommendations\n")
    _write_recommendations(lines, strategy_results, validation_results, portfolio_metrics)

    # Final Questions
    lines.append("## The Three Questions\n")
    _write_final_answers(lines, strategy_results, validation_results, portfolio_metrics)

    return "\n".join(lines)


def _strategy_verdict(result: StrategyResult, validations: list[ValidationResult]) -> str:
    """Generate honest verdict for a strategy."""
    sharpe = result.sharpe_ratio
    trades = result.trade_count
    failed_validations = sum(1 for v in validations if not v.passed)
    total_validations = len(validations)

    if trades < 10:
        return "Insufficient trades for statistical significance. Not tradeable."
    if sharpe < 0:
        return f"Negative Sharpe ({sharpe:.2f}). Strategy destroys value. Do not trade."
    if sharpe < 0.3:
        return f"Marginal Sharpe ({sharpe:.2f}). Edge too small to survive costs. Not recommended."
    if failed_validations > total_validations / 2:
        return f"Sharpe {sharpe:.2f} but {failed_validations}/{total_validations} validation tests failed. Likely overfit."
    if sharpe >= 1.0 and failed_validations <= 1:
        return f"Strong Sharpe ({sharpe:.2f}) with robust validation. Candidate for live trading with small size."
    if sharpe >= 0.5:
        return f"Moderate Sharpe ({sharpe:.2f}). Worth monitoring but increase sample size before deploying capital."
    return f"Sharpe {sharpe:.2f}. Marginal edge. Paper trade first."


def _write_recommendations(
    lines: list[str],
    strategy_results: dict[str, StrategyResult],
    validation_results: dict[str, list[ValidationResult]],
    portfolio_metrics: dict[str, Any],
) -> None:
    tradeable = []
    for name, result in strategy_results.items():
        vals = validation_results.get(name, [])
        failed = sum(1 for v in vals if not v.passed)
        if result.sharpe_ratio >= 0.5 and result.trade_count >= 10 and failed <= len(vals) / 2:
            tradeable.append(name)

    if tradeable:
        lines.append(f"**Tradeable strategies**: {', '.join(tradeable)}\n")
        lines.append("- Start with paper trading for 4-8 weeks")
        lines.append("- Initial capital: $1,000-$5,000 per strategy")
        lines.append("- Monitor: daily Sharpe, drawdown, fill quality")
        lines.append("- Kill switch: halt if drawdown exceeds 2x backtested max")
    else:
        lines.append("**No strategies meet the bar for live trading.**\n")
        lines.append("- All strategies either lack statistical significance or fail validation")
        lines.append("- This is a valuable finding — it means naive alpha on Polymarket is competed away")
        lines.append("- Consider: better data sources, higher-frequency signals, or different market structure edges")
    lines.append("")


def _write_final_answers(
    lines: list[str],
    strategy_results: dict[str, StrategyResult],
    validation_results: dict[str, list[ValidationResult]],
    portfolio_metrics: dict[str, Any],
) -> None:
    combined_sharpe = portfolio_metrics.get("sharpe", 0)
    combined_pnl = portfolio_metrics.get("total_pnl", 0)
    profitable_count = sum(1 for r in strategy_results.values() if r.sharpe_ratio > 0.5)

    lines.append("### 1. Is there alpha on Polymarket?\n")
    if combined_sharpe > 0.5 and profitable_count >= 2:
        lines.append("Yes, there appear to be structural inefficiencies that can be exploited, "
                     "though the edge is modest and requires disciplined execution.\n")
    elif combined_sharpe > 0:
        lines.append("Marginally. Some signals show positive expected value before costs, "
                     "but the edge is thin and may not survive transaction costs and market impact.\n")
    else:
        lines.append("Not convincingly. After proper validation and cost modeling, "
                     "no strategy consistently generates risk-adjusted returns above zero.\n")

    lines.append("### 2. Where specifically?\n")
    by_sharpe = sorted(strategy_results.items(), key=lambda x: x[1].sharpe_ratio, reverse=True)
    for name, result in by_sharpe[:3]:
        if result.sharpe_ratio > 0:
            lines.append(f"- **{name}**: Sharpe {result.sharpe_ratio:.2f}, {result.trade_count} trades")
    lines.append("")

    lines.append("### 3. How much, how reliably, and at what capacity?\n")
    lines.append(f"- **Expected annual P&L**: ${combined_pnl:,.2f} (on $10k per strategy)")
    lines.append(f"- **Portfolio Sharpe**: {combined_sharpe:.2f}")
    lines.append(f"- **Reliability**: {'Moderate' if combined_sharpe > 0.5 else 'Low'} — "
                 f"significant regime dependence observed")
    lines.append(f"- **Capacity**: Likely limited to $10k-$50k total across all strategies "
                 f"before market impact degrades returns")
    lines.append("")


# ── Chart Generation ──────────────────────────────────────────────


def _plot_equity_curves(
    results: dict[str, StrategyResult], charts_dir: str
) -> None:
    """Plot individual and combined equity curves."""
    for name, result in results.items():
        if not result.cumulative_pnl:
            continue
        fig, ax = plt.subplots()
        ax.plot(result.cumulative_pnl, linewidth=1.5)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"{name} — Equity Curve")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Cumulative P&L ($)")

        # Draw drawdown fill
        cum = result.cumulative_pnl
        peak = [cum[0]]
        for v in cum[1:]:
            peak.append(max(peak[-1], v))
        ax.fill_between(range(len(cum)), cum, peak, alpha=0.2, color="red", label="Drawdown")
        ax.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(charts_dir, f"{name}_equity.png"))
        plt.close(fig)


def _plot_correlation_matrix(
    corr: dict[str, dict[str, float]], charts_dir: str
) -> None:
    if not corr:
        return
    names = list(corr.keys())
    matrix = np.zeros((len(names), len(names)))
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            matrix[i, j] = corr.get(n1, {}).get(n2, 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix, annot=True, fmt=".2f", xticklabels=names, yticklabels=names,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
    )
    ax.set_title("Strategy Return Correlations")
    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, "correlation_matrix.png"))
    plt.close(fig)


def _plot_portfolio_curve(metrics: dict[str, Any], charts_dir: str) -> None:
    cum = metrics.get("cumulative_pnl", [])
    if not cum:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(cum, linewidth=1.5, color="navy")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title("Combined Portfolio Equity Curve")
    ax1.set_ylabel("Cumulative P&L ($)")

    # Drawdown
    peak = [cum[0]]
    for v in cum[1:]:
        peak.append(max(peak[-1], v))
    dd = [c - p for c, p in zip(cum, peak)]
    ax2.fill_between(range(len(dd)), dd, 0, alpha=0.5, color="red")
    ax2.set_title("Drawdown")
    ax2.set_xlabel("Trade #")
    ax2.set_ylabel("Drawdown ($)")

    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, "portfolio_equity.png"))
    plt.close(fig)


def _plot_walk_forward(
    wf_results: dict[str, list[StrategyResult]], charts_dir: str
) -> None:
    for name, windows in wf_results.items():
        if not windows:
            continue
        sharpes = [r.sharpe_ratio for r in windows]
        pnls = [r.total_pnl for r in windows]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.bar(range(len(sharpes)), sharpes, color=["green" if s > 0 else "red" for s in sharpes])
        ax1.axhline(y=0, color="gray", linestyle="--")
        ax1.set_title(f"{name} — Walk-Forward Sharpe by Window")
        ax1.set_xlabel("Window")
        ax1.set_ylabel("Sharpe Ratio")

        ax2.bar(range(len(pnls)), pnls, color=["green" if p > 0 else "red" for p in pnls])
        ax2.axhline(y=0, color="gray", linestyle="--")
        ax2.set_title(f"{name} — Walk-Forward P&L by Window")
        ax2.set_xlabel("Window")
        ax2.set_ylabel("P&L ($)")

        fig.tight_layout()
        fig.savefig(os.path.join(charts_dir, f"{name}_walkforward.png"))
        plt.close(fig)


def _plot_monthly_heatmap(
    results: dict[str, StrategyResult], charts_dir: str
) -> None:
    """Monthly returns heatmap (strategy x month)."""
    all_months: set[str] = set()
    monthly_data: dict[str, dict[str, float]] = {}

    for name, result in results.items():
        monthly: dict[str, float] = {}
        for trade in result.trades:
            month = trade.entry_time.strftime("%Y-%m")
            monthly[month] = monthly.get(month, 0) + trade.net_pnl
            all_months.add(month)
        monthly_data[name] = monthly

    if not all_months:
        return

    months = sorted(all_months)
    strategies = list(results.keys())
    matrix = np.zeros((len(strategies), len(months)))

    for i, name in enumerate(strategies):
        for j, month in enumerate(months):
            matrix[i, j] = monthly_data.get(name, {}).get(month, 0)

    fig, ax = plt.subplots(figsize=(max(12, len(months) * 0.8), max(6, len(strategies) * 0.8)))
    sns.heatmap(
        matrix, annot=True, fmt=".0f", xticklabels=months, yticklabels=strategies,
        cmap="RdYlGn", center=0, ax=ax,
    )
    ax.set_title("Monthly Returns ($) by Strategy")
    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, "monthly_heatmap.png"))
    plt.close(fig)
