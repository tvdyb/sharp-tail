"""Tests for report generation."""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pytest

from polymir.backtest import BacktestResult
from polymir.backtest.data import TradeRecord
from polymir.backtest.report import generate_report
from polymir.backtest.validation import ValidationResult


def _make_result() -> BacktestResult:
    now = datetime.utcnow()
    records = [
        TradeRecord(
            timestamp=now - timedelta(hours=20 - i),
            wallet=f"w{i % 3}",
            market_id=f"m{i}",
            asset_id=f"a{i}",
            side="BUY",
            signal_price=0.55,
            fill_price=0.56,
            size=100,
            pnl=10.0 if i % 3 != 0 else -5.0,
            decision="execute",
            market_resolved_price=1.0 if i % 3 != 0 else 0.0,
            market_category="politics" if i % 2 == 0 else "sports",
        )
        for i in range(20)
    ] + [
        TradeRecord(
            timestamp=now - timedelta(hours=i),
            wallet="w0", market_id=f"skip{i}", asset_id=f"as{i}",
            side="BUY", signal_price=0.55,
            decision="stale",
        )
        for i in range(5)
    ]
    return BacktestResult(trade_records=records)


class TestReportGeneration:
    def test_generates_markdown(self, tmp_path):
        result = _make_result()
        report = generate_report(result, output_dir=str(tmp_path))
        assert "# Backtest Report" in report
        assert "Trades executed:" in report
        assert "Sharpe ratio:" in report

    def test_creates_output_files(self, tmp_path):
        result = _make_result()
        generate_report(result, output_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "backtest_report.md")
        assert os.path.exists(tmp_path / "chart_data.json")

    def test_includes_bias_results(self, tmp_path):
        result = _make_result()
        bias_results = [
            {"test": "price_leakage", "passed": True, "exact_match_rate": 0.01},
            {"test": "wallet_score_leakage", "passed": False, "violations": 3},
        ]
        report = generate_report(result, bias_results=bias_results, output_dir=str(tmp_path))
        assert "Bias Tests" in report
        assert "PASS" in report
        assert "FAIL" in report

    def test_includes_validation_results(self, tmp_path):
        result = _make_result()
        validations = [
            ValidationResult("bootstrap_ci", True, {"sharpe_ci_95": (0.5, 2.0)}),
            ValidationResult("walk_forward", False, {"oos_sharpe": -0.3}),
        ]
        report = generate_report(
            result, validation_results=validations, output_dir=str(tmp_path))
        assert "Statistical Validation" in report
        assert "bootstrap_ci" in report
        assert "walk_forward" in report

    def test_includes_sweep_results(self, tmp_path):
        result = _make_result()
        sweep = {
            "config_a": _make_result(),
            "config_b": _make_result(),
        }
        report = generate_report(result, sweep_results=sweep, output_dir=str(tmp_path))
        assert "Parameter Sweep" in report

    def test_honest_assessment_with_issues(self, tmp_path):
        # Small number of trades -> should flag
        records = [
            TradeRecord(
                timestamp=datetime.utcnow(),
                wallet="w0", market_id="m0", asset_id="a0",
                side="BUY", signal_price=0.55, fill_price=0.56,
                size=100, pnl=1.0, decision="execute",
                market_resolved_price=1.0,
            )
        ]
        result = BacktestResult(trade_records=records)
        report = generate_report(result, output_dir=str(tmp_path))
        assert "Concerns" in report or "insufficient" in report.lower() or "Insufficient" in report
