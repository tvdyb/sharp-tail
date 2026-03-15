"""Markdown report generation for backtest results."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from polymir.backtest.metrics import BacktestResult
from polymir.backtest.validation import ValidationResult


def generate_report(
    result: BacktestResult,
    bias_results: list[dict[str, Any]] | None = None,
    validation_results: list[ValidationResult] | None = None,
    sweep_results: dict[str, BacktestResult] | None = None,
    output_dir: str = "reports",
) -> str:
    """Generate a comprehensive markdown report.

    Args:
        result: Primary backtest result.
        bias_results: Results from bias tests.
        validation_results: Results from validation procedures.
        sweep_results: Results from parameter sweep.
        output_dir: Directory to save report and charts.

    Returns:
        Markdown string.
    """
    os.makedirs(output_dir, exist_ok=True)

    sections = []
    sections.append("# Backtest Report")
    sections.append(f"\nGenerated: {datetime.utcnow().isoformat()}\n")

    # Summary
    sections.append("## Summary\n")
    sections.append("```")
    sections.append(result.summary())
    sections.append("```\n")

    # Monthly returns
    sections.append("## Monthly Returns\n")
    monthly = result.monthly_returns()
    if monthly:
        sections.append("| Month | PnL |")
        sections.append("|-------|-----|")
        for month, pnl in monthly.items():
            sections.append(f"| {month} | ${pnl:,.2f} |")
    else:
        sections.append("No monthly data available.\n")

    # Skip breakdown
    sections.append("\n## Trade Decision Breakdown\n")
    sections.append(f"- Executed: {result.trades_executed}")
    for reason, count in result.skip_reasons.items():
        sections.append(f"- Skipped ({reason}): {count}")

    # Wallet concentration
    sections.append("\n## Wallet Concentration\n")
    by_wallet = result.pnl_by_wallet()
    if by_wallet:
        sections.append("| Rank | Wallet | PnL | % of Total |")
        sections.append("|------|--------|-----|------------|")
        total = result.total_pnl or 1.0
        for i, (wallet, pnl) in enumerate(list(by_wallet.items())[:10], 1):
            pct = (pnl / total * 100) if total != 0 else 0
            sections.append(f"| {i} | {wallet[:10]}... | ${pnl:,.2f} | {pct:.1f}% |")

        sections.append(f"\nTop-1 concentration: {result.top_wallet_concentration(1):.1%}")
        sections.append(f"Top-3 concentration: {result.top_wallet_concentration(3):.1%}")
        sections.append(f"Top-5 concentration: {result.top_wallet_concentration(5):.1%}")

    # Category breakdown
    sections.append("\n## Performance by Category\n")
    by_cat = result.pnl_by_category()
    if by_cat:
        sections.append("| Category | PnL |")
        sections.append("|----------|-----|")
        for cat, pnl in sorted(by_cat.items(), key=lambda x: x[1], reverse=True):
            sections.append(f"| {cat} | ${pnl:,.2f} |")

    # Bias tests
    if bias_results:
        sections.append("\n## Bias Tests\n")
        sections.append("| Test | Status | Details |")
        sections.append("|------|--------|---------|")
        for br in bias_results:
            status = "PASS" if br.get("passed") else "FAIL"
            name = br.get("test", "unknown")
            detail_keys = [k for k in br if k not in ("test", "passed")]
            detail = ", ".join(f"{k}={br[k]}" for k in detail_keys[:3])
            sections.append(f"| {name} | {status} | {detail} |")

    # Validation results
    if validation_results:
        sections.append("\n## Statistical Validation\n")
        for vr in validation_results:
            status = "PASS" if vr.passed else "FAIL"
            sections.append(f"### {vr.method} [{status}]\n")
            for k, v in vr.details.items():
                if isinstance(v, float):
                    sections.append(f"- **{k}**: {v:.4f}")
                elif isinstance(v, list) and len(v) <= 10:
                    sections.append(f"- **{k}**: {v}")
                elif isinstance(v, dict) and len(v) <= 10:
                    sections.append(f"- **{k}**: {v}")
                else:
                    sections.append(f"- **{k}**: {v}")

    # Parameter sweep
    if sweep_results:
        sections.append("\n## Parameter Sweep\n")
        sections.append("| Config | Sharpe | PnL | Trades | Win Rate |")
        sections.append("|--------|--------|-----|--------|----------|")
        sorted_sweep = sorted(sweep_results.items(), key=lambda x: x[1].sharpe_ratio, reverse=True)
        for label, sr in sorted_sweep[:20]:
            sections.append(
                f"| {label} | {sr.sharpe_ratio:.2f} | ${sr.total_pnl:,.2f} "
                f"| {sr.trades_executed} | {sr.win_rate:.1%} |"
            )

    # Honest assessment
    sections.append("\n## Assessment\n")
    sections.append(_generate_assessment(result, bias_results, validation_results))

    # Charts note
    sections.append("\n## Charts\n")
    _save_chart_data(result, output_dir)
    sections.append(f"Chart data saved to `{output_dir}/chart_data.json`.")
    sections.append("Use matplotlib or your preferred tool to render:\n")
    sections.append("- Cumulative PnL curve")
    sections.append("- Drawdown overlay")
    sections.append("- Monthly returns bar chart")
    sections.append("- Parameter sensitivity heatmap (if sweep data available)")

    report = "\n".join(sections)

    report_path = os.path.join(output_dir, "backtest_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    return report


def _generate_assessment(
    result: BacktestResult,
    bias_results: list[dict[str, Any]] | None,
    validation_results: list[ValidationResult] | None,
) -> str:
    """Generate honest assessment of strategy viability."""
    issues = []

    if result.trades_executed < 30:
        issues.append("Insufficient trade count for statistical significance.")

    if result.sharpe_ratio < 0.5:
        issues.append(f"Low Sharpe ratio ({result.sharpe_ratio:.2f}).")

    if result.max_drawdown > abs(result.total_pnl) * 0.5:
        issues.append(f"Large drawdown (${result.max_drawdown:,.2f}) relative to total PnL.")

    if result.top_wallet_concentration(1) > 0.5:
        issues.append("Returns heavily concentrated in a single wallet.")

    if bias_results:
        failed = [b["test"] for b in bias_results if not b.get("passed")]
        if failed:
            issues.append(f"Failed bias tests: {', '.join(failed)}")

    if validation_results:
        failed_val = [v.method for v in validation_results if not v.passed]
        if failed_val:
            issues.append(f"Failed validation: {', '.join(failed_val)}")

    if not issues:
        return (
            "The strategy shows promising results across the tested period. "
            "All bias tests pass and statistical validation is positive. "
            "However, past performance does not guarantee future results. "
            "Monitor live performance closely against backtest expectations."
        )

    return (
        "**Concerns identified:**\n\n"
        + "\n".join(f"- {issue}" for issue in issues)
        + "\n\nThese issues should be investigated before deploying capital."
    )


def _save_chart_data(result: BacktestResult, output_dir: str) -> None:
    """Save chart data as JSON for external rendering."""
    data = {
        "cumulative_pnl": result.cumulative_pnl,
        "pnl_series": result.pnl_series,
        "monthly_returns": result.monthly_returns(),
        "executed_timestamps": [
            t.timestamp.isoformat()
            for t in result.trade_records
            if t.decision == "execute"
        ],
    }
    path = os.path.join(output_dir, "chart_data.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
