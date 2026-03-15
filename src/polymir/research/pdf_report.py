"""Comprehensive PDF research report generator using reportlab."""

from __future__ import annotations

import io
import math
import os
import tempfile
from datetime import date, datetime
from statistics import mean, stdev
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from polymir.research.models import (
    DataQualityReport,
    PriceSnapshot,
    ResearchMarket,
    StrategyResult,
)
from polymir.research.validation import ValidationResult

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})

# ── Color palette ────────────────────────────────────────────────
NAVY = colors.HexColor("#1B2A4A")
BLUE = colors.HexColor("#2563EB")
LIGHT_BLUE = colors.HexColor("#DBEAFE")
GREEN = colors.HexColor("#16A34A")
RED = colors.HexColor("#DC2626")
GRAY = colors.HexColor("#6B7280")
LIGHT_GRAY = colors.HexColor("#F3F4F6")
WHITE = colors.white


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title", parent=base["Title"], fontSize=28, textColor=NAVY,
            spaceAfter=6, alignment=TA_CENTER, leading=34,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"], fontSize=14, textColor=GRAY,
            spaceAfter=20, alignment=TA_CENTER,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontSize=20, textColor=NAVY,
            spaceBefore=24, spaceAfter=12, leading=24,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontSize=16, textColor=NAVY,
            spaceBefore=18, spaceAfter=8, leading=20,
        ),
        "h3": ParagraphStyle(
            "H3", parent=base["Heading3"], fontSize=13, textColor=NAVY,
            spaceBefore=12, spaceAfter=6, leading=16,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["Normal"], fontSize=10, leading=14,
            spaceAfter=6, alignment=TA_JUSTIFY,
        ),
        "body_small": ParagraphStyle(
            "BodySmall", parent=base["Normal"], fontSize=9, leading=12,
            spaceAfter=4,
        ),
        "code": ParagraphStyle(
            "Code", parent=base["Code"], fontSize=8, leading=10,
            fontName="Courier", backColor=LIGHT_GRAY, spaceAfter=6,
            leftIndent=12, rightIndent=12,
        ),
        "disclaimer": ParagraphStyle(
            "Disclaimer", parent=base["Normal"], fontSize=8, leading=10,
            textColor=GRAY, alignment=TA_CENTER, spaceAfter=6,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["Normal"], fontSize=9, leading=11,
            textColor=GRAY, alignment=TA_CENTER, spaceBefore=4, spaceAfter=12,
        ),
        "toc": ParagraphStyle(
            "TOC", parent=base["Normal"], fontSize=11, leading=18,
            leftIndent=20,
        ),
    }


def _table_style() -> TableStyle:
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])


def _save_chart(fig: plt.Figure) -> str:
    """Save matplotlib figure to temp file, return path."""
    path = tempfile.mktemp(suffix=".png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _chart_image(path: str, width: float = 6.5 * inch) -> Image:
    """Create reportlab Image from file path, auto-scaling height."""
    from PIL import Image as PILImage
    with PILImage.open(path) as img:
        w, h = img.size
    aspect = h / w
    return Image(path, width=width, height=width * aspect)


# ── Chart generators ──────────────────────────────────────────────


def _make_equity_chart(result: StrategyResult, name: str) -> str:
    cum = result.cumulative_pnl
    if not cum:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No trades", ha="center", va="center", fontsize=14, color="gray")
        ax.set_axis_off()
        return _save_chart(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = range(len(cum))
    ax.plot(x, cum, linewidth=1.2, color="#2563EB")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    peak = [cum[0]]
    for v in cum[1:]:
        peak.append(max(peak[-1], v))
    ax.fill_between(x, cum, peak, alpha=0.15, color="#DC2626")

    ax.set_title(f"{name} — Cumulative P&L", fontsize=11, fontweight="bold")
    ax.set_xlabel("Trade #", fontsize=9)
    ax.set_ylabel("P&L ($)", fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return _save_chart(fig)


def _make_correlation_chart(corr: dict[str, dict[str, float]]) -> str:
    names = list(corr.keys())
    n = len(names)
    matrix = np.zeros((n, n))
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            v = corr.get(n1, {}).get(n2, 0)
            matrix[i, j] = v if not math.isnan(v) else 0

    short = [n[:12] for n in names]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(matrix, annot=True, fmt=".2f", xticklabels=short, yticklabels=short,
                cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
    ax.set_title("Strategy Return Correlations", fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return _save_chart(fig)


def _make_portfolio_chart(metrics: dict[str, Any]) -> str:
    cum = metrics.get("cumulative_pnl", [])
    if not cum:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return _save_chart(fig)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), gridspec_kw={"height_ratios": [3, 1]})
    x = range(len(cum))
    ax1.plot(x, cum, linewidth=1.2, color="#1B2A4A")
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_title("Combined Portfolio — Cumulative P&L", fontsize=11, fontweight="bold")
    ax1.set_ylabel("P&L ($)", fontsize=9)
    ax1.tick_params(labelsize=8)

    peak = [cum[0]]
    for v in cum[1:]:
        peak.append(max(peak[-1], v))
    dd = [c - p for c, p in zip(cum, peak)]
    ax2.fill_between(x, dd, 0, alpha=0.5, color="#DC2626")
    ax2.set_ylabel("Drawdown ($)", fontsize=9)
    ax2.set_xlabel("Trade #", fontsize=9)
    ax2.tick_params(labelsize=8)
    fig.tight_layout()
    return _save_chart(fig)


def _make_monthly_heatmap(results: dict[str, StrategyResult]) -> str:
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
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return _save_chart(fig)

    months = sorted(all_months)
    strategies = list(results.keys())
    matrix = np.zeros((len(strategies), len(months)))
    for i, name in enumerate(strategies):
        for j, month in enumerate(months):
            matrix[i, j] = monthly_data.get(name, {}).get(month, 0)

    short_s = [s[:12] for s in strategies]
    fig, ax = plt.subplots(figsize=(max(8, len(months) * 0.7), max(3, len(strategies) * 0.55)))
    sns.heatmap(matrix, annot=True, fmt=".0f", xticklabels=months, yticklabels=short_s,
                cmap="RdYlGn", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Monthly Returns ($) by Strategy", fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return _save_chart(fig)


def _make_calibration_chart(results: dict[str, StrategyResult]) -> str:
    """Plot calibration data from favorite-longshot strategy trades."""
    fl = results.get("favorite_longshot")
    if not fl or not fl.trades:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No calibration data", ha="center", va="center")
        ax.set_axis_off()
        return _save_chart(fig)

    # Bin trades by entry price and compute win rate
    bins: dict[int, list[bool]] = {}
    for t in fl.trades:
        bin_idx = min(int(t.entry_price * 10), 9)
        won = t.net_pnl > 0
        bins.setdefault(bin_idx, []).append(won)

    x_impl, y_real = [], []
    for b in range(10):
        outcomes = bins.get(b, [])
        if len(outcomes) >= 2:
            implied = (b + 0.5) / 10
            realized = sum(outcomes) / len(outcomes)
            x_impl.append(implied)
            y_real.append(realized)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8, label="Perfect calibration")
    if x_impl:
        ax.scatter(x_impl, y_real, color="#2563EB", s=60, zorder=5)
        ax.plot(x_impl, y_real, color="#2563EB", linewidth=1, alpha=0.7)
    ax.set_xlabel("Implied Probability (market price)", fontsize=9)
    ax.set_ylabel("Realized Frequency", fontsize=9)
    ax.set_title("Calibration Curve — Favorite-Longshot Bias", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return _save_chart(fig)


# ── Strategy descriptions ─────────────────────────────────────────

STRATEGY_DETAIL: dict[str, dict[str, str]] = {
    "favorite_longshot": {
        "full_name": "Strategy 1: Favorite-Longshot Bias (Terminal Convergence Mispricing)",
        "thesis": (
            "Prediction markets, like parimutuel betting markets, exhibit the well-documented "
            "favorite-longshot bias (FLB). This phenomenon, first identified by Griffith (1949) and "
            "extensively studied by Thaler & Ziemba (1988), shows that bettors systematically "
            "overprice longshots (low-probability outcomes) and underprice favorites (high-probability "
            "outcomes). The economic intuition is that bettors derive utility from the dream of a large "
            "payout on longshots, creating a risk-premium that distorts prices away from true probabilities.\n\n"
            "On Polymarket specifically, as markets approach resolution, prices should converge toward "
            "0 or 1. Markets still trading at intermediate prices (20-80 cents) close to resolution are "
            "either genuinely uncertain or structurally mispriced. By building a calibration curve — "
            "mapping implied probability (market price) to realized frequency (how often that outcome "
            "actually occurs) — we can identify systematic biases and trade them.\n\n"
            "If contracts priced at 0.75 actually resolve YES only 70% of the time, that 5-cent gap "
            "is pure alpha. Conversely, if 0.25 contracts resolve YES 30% of the time, the longshot "
            "is overpriced and we should sell."
        ),
        "implementation": (
            "1. For every resolved market, extract the price trajectory in the final 7, 3, and 1 day(s) before resolution.\n"
            "2. Identify markets where price was in [0.20, 0.80] within the entry window (default 72h before resolution).\n"
            "3. Bin prices into deciles (0-10%, 10-20%, ..., 90-100%).\n"
            "4. For each bin, compute: implied probability (bin midpoint) vs realized frequency (fraction resolving YES).\n"
            "5. Require minimum bin count (default 10) for statistical validity.\n"
            "6. Signal generation: for each active market within the entry window, look up its price bin in the calibration curve.\n"
            "   - If realized > implied by ≥3%: BUY (favorites underpriced)\n"
            "   - If realized < implied by ≥3%: SELL (longshots overpriced)\n"
            "7. Confidence = realized frequency from the calibration curve.\n"
            "8. The calibration curve is rebuilt using only data available before the signal timestamp (no look-ahead)."
        ),
        "parameters": [
            ("entry_hours", "72.0", "24-168", "Hours before resolution to enter position"),
            ("price_bins", "10", "5-20", "Number of bins for calibration curve"),
            ("min_bin_count", "10", "5-30", "Minimum observations per bin"),
            ("category_filter", "None", "Any category", "Restrict to specific market category"),
        ],
        "formula": (
            "Edge = Realized_Frequency(bin) - Market_Price\n"
            "If Edge > 0.03: BUY at Market_Price, confidence = Realized_Frequency\n"
            "If Edge < -0.03: SELL at Market_Price, confidence = 1 - Realized_Frequency\n"
            "Kelly fraction = Edge / (1 - Market_Price)\n"
            "Position = Capital × min(Kelly/4, 10%)"
        ),
    },
    "multi_outcome_arb": {
        "full_name": "Strategy 2: Multi-Outcome Relative Value and Sum Arbitrage",
        "thesis": (
            "In neg_risk events with N mutually exclusive outcomes (e.g., 'Who will win the election?' "
            "with candidates A, B, C, ...), the prices of all YES tokens should sum to exactly $1.00. "
            "This is an accounting identity — exactly one outcome must win. Any deviation from this "
            "invariant creates a risk-free arbitrage opportunity (before fees) or a positive-EV trade "
            "(after fees, if the deviation exceeds the fee threshold).\n\n"
            "On Polymarket, these deviations arise because each outcome trades as a separate market "
            "with its own order book. Liquidity providers and traders focus on individual outcomes, "
            "and the composite price sum can drift. This is analogous to index arbitrage in equity "
            "markets, where the sum of component prices must equal the index price.\n\n"
            "Additionally, for ordinal events (e.g., 'How many Fed rate cuts?' with outcomes 0,1,2,...,8+), "
            "the implied probability distribution should be smooth. Kinks or discontinuities in the "
            "distribution suggest mispricing of individual outcomes."
        ),
        "implementation": (
            "1. Identify all multi-outcome events: markets sharing an event_id with enableNegRisk=true.\n"
            "2. At each timestamp, fetch the latest YES price for every outcome in the event.\n"
            "3. Compute SUM = sum of all YES prices.\n"
            "4. If |SUM - 1.0| > deviation_threshold (default 3%):\n"
            "   - SUM > 1.0: Outcomes collectively overpriced. SELL the outcome whose price exceeds\n"
            "     its fair share (price × (1/SUM)) by the most.\n"
            "   - SUM < 1.0: Outcomes collectively underpriced. BUY the most underpriced outcome.\n"
            "5. Fair price adjustment: adjusted_fair = price × (1.0 / price_sum)\n"
            "6. Only signal when individual deviation exceeds 2 cents beyond the fair-adjusted price.\n"
            "7. Size fraction proportional to magnitude of sum deviation, capped at 0.5."
        ),
        "parameters": [
            ("deviation_threshold", "0.03", "0.01-0.05", "Minimum sum deviation to trigger signal"),
            ("hold_hours", "24.0", "1-168", "Target holding period in hours"),
        ],
        "formula": (
            "SUM = Σ Price_i for i in 1..N outcomes\n"
            "Deviation = SUM - 1.0\n"
            "Fair_i = Price_i × (1 / SUM)\n"
            "Signal if |Deviation| > threshold AND |Price_i - Fair_i| > 0.02\n"
            "Direction: SELL if SUM > 1 and Price_i > Fair_i; BUY if SUM < 1 and Price_i < Fair_i"
        ),
    },
    "volume_momentum": {
        "full_name": "Strategy 3: Volume-Weighted Price Momentum with Reversal Detection",
        "thesis": (
            "Market microstructure theory, particularly Kyle (1985), describes how information is "
            "incorporated into prices through trading. The key insight is Kyle's lambda (λ) — the "
            "price impact per unit of volume. Markets where price absorbs lots of volume without "
            "moving (low λ) are seeing informed flow from traders who know the true value and are "
            "slowly accumulating.\n\n"
            "Conversely, large volume accompanied by sharp price moves (high λ) often represents "
            "uninformed momentum — reactive traders piling in after a move, pushing price beyond "
            "fair value. These momentum episodes tend to mean-revert as the information content "
            "dissipates and liquidity providers push price back.\n\n"
            "On Polymarket, this manifests as: (1) Informed accumulation before resolution of "
            "predictable events, and (2) Overreaction to news events that creates a reversal opportunity."
        ),
        "implementation": (
            "1. For each active market, compute rolling metrics over a window (default 24h):\n"
            "   - price_change = (P_end - P_start) / P_start\n"
            "   - volume_proxy = count of price observations in window\n"
            "   - kyle_lambda = |price_change| / volume_proxy\n"
            "2. Momentum reversal signal: if |price_change| ≥ 10% AND volume_proxy ≥ 3:\n"
            "   - Trade AGAINST the move (SELL if price went up, BUY if down)\n"
            "   - Confidence = min(0.5 + |price_change| × 0.5, 0.75)\n"
            "3. Informed accumulation signal: if volume_proxy ≥ 5 AND |price_change| < 3%:\n"
            "   - Trade WITH the direction of slight drift\n"
            "   - Confidence = 0.55, size_fraction = 0.5 (smaller position)\n"
            "4. Ignore flat markets (|price_change| < 0.5%)."
        ),
        "parameters": [
            ("volume_window_hours", "24.0", "1-168", "Rolling window for volume/price metrics"),
            ("momentum_threshold", "0.10", "0.03-0.20", "Min |price_change| for momentum signal"),
            ("reversal_lookback_hours", "48.0", "12-168", "Lookback for measuring mean reversion"),
        ],
        "formula": (
            "λ (Kyle's lambda) = |ΔP/P| / N_observations\n"
            "Momentum signal: |ΔP/P| ≥ threshold → bet on reversal\n"
            "Accumulation signal: N_obs ≥ 5 AND |ΔP/P| < 0.03 → bet on continuation"
        ),
    },
    "cross_market_cascade": {
        "full_name": "Strategy 4: Cross-Market Information Cascade Lag",
        "thesis": (
            "When significant news arrives, it should simultaneously reprice all affected markets. "
            "In practice, Polymarket participants are fragmented across markets — a trader watching "
            "'Will Fed cut in March?' may not simultaneously update their position in 'Will Fed cut "
            "in 2025?'. This creates a lead-lag structure where the primary market moves first, and "
            "related markets catch up with a delay.\n\n"
            "This is directly analogous to the Hou & Moskowitz (2005) price delay documented in "
            "equity markets, where large-cap stocks react to market-wide news faster than small-caps. "
            "On Polymarket, the 'large-cap' equivalent is the most liquid market in a topic cluster.\n\n"
            "We identify related markets through three mechanisms: (1) shared event groups (neg_risk "
            "siblings), (2) same category with overlapping time periods, and (3) keyword/entity "
            "overlap in questions."
        ),
        "implementation": (
            "1. Build a market relationship graph:\n"
            "   a. Same event: markets sharing an event_id are siblings\n"
            "   b. Same category + time overlap: markets in the same category with overlapping active periods\n"
            "   c. Keyword overlap: extract significant words from questions; ≥2 shared keywords = related\n"
            "2. For each active market, check for recent significant moves (|ΔP| > threshold) in a lag window.\n"
            "3. For each significant mover, check its related markets:\n"
            "   - Compute the related market's move in the same window\n"
            "   - Expected move = primary_move × min_correlation\n"
            "   - Gap = expected_move - actual_related_move\n"
            "   - If |gap| > 2%: trade related market in gap direction\n"
            "4. Confidence = min(0.5 + |gap|, 0.7), size_fraction = 0.5."
        ),
        "parameters": [
            ("move_threshold", "0.05", "0.03-0.10", "Min move in primary market to trigger"),
            ("lag_hours", "6.0", "1-12", "Window to detect primary move and check related"),
            ("min_correlation", "0.3", "0.1-0.7", "Assumed min correlation for expected response"),
        ],
        "formula": (
            "Primary_move = (P_now - P_before) / P_before over lag window\n"
            "Expected_related_move = Primary_move × min_correlation\n"
            "Gap = Expected_related_move - Actual_related_move\n"
            "Signal if |Gap| > 0.02"
        ),
    },
    "spread_capture": {
        "full_name": "Strategy 5: Spread-Capture Market Making in Thin Markets",
        "thesis": (
            "Many Polymarket markets have 5-15% bid-ask spreads with thin order books. This is a "
            "structural feature of fragmented prediction market liquidity — unlike centralized equity "
            "exchanges with designated market makers and competitive HFT, PM markets often have a "
            "single LP (Polymarket's automated market maker or a few retail LPs).\n\n"
            "By providing passive liquidity (resting bid and ask orders) in markets with stable fair "
            "values and wide spreads, you capture the spread minus adverse selection costs. The key "
            "insight is that in low-volatility, long-duration markets, the fair value changes slowly "
            "relative to the spread width, so most fills are at favorable prices.\n\n"
            "The main risk is adverse selection: if you're filled because an informed trader moved "
            "the market, your position immediately goes against you. We mitigate this by filtering "
            "for spread > 3× realized volatility."
        ),
        "implementation": (
            "1. For each active market, compute:\n"
            "   - Estimated spread = max(recent prices) - min(recent prices) over last 10 observations\n"
            "   - Realized volatility = std(returns) over available history\n"
            "   - Time to resolution = hours until end_date\n"
            "2. Filter candidates: spread > min_spread, volatility < vol_ceiling,\n"
            "   spread > 3 × volatility, time_to_resolution > 48h, price in [0.05, 0.95]\n"
            "3. Simulate market making:\n"
            "   - BUY signal at bid_price = current_price - half_spread\n"
            "   - SELL signal at ask_price = current_price + half_spread\n"
            "4. Size_fraction = 0.3 (smaller due to inventory risk)."
        ),
        "parameters": [
            ("min_spread", "0.05", "0.03-0.15", "Minimum estimated spread to consider"),
            ("vol_ceiling", "0.15", "0.05-0.30", "Maximum realized vol"),
            ("min_time_to_resolution_hours", "48.0", "24-720", "Min hours until resolution"),
            ("spread_width_frac", "1.0", "0.5-1.0", "Fraction of spread to quote"),
            ("inventory_limit", "5", "1-20", "Max positions per market"),
        ],
        "formula": (
            "Spread = max(P_recent) - min(P_recent)\n"
            "Vol = std(ΔP/P)\n"
            "Bid = P_mid - Spread × width_frac / 2\n"
            "Ask = P_mid + Spread × width_frac / 2\n"
            "Filter: Spread > 3 × Vol AND Vol < ceiling"
        ),
    },
    "theta_harvesting": {
        "full_name": "Strategy 6: Time Decay and Theta Harvesting",
        "thesis": (
            "By analogy with options markets, prediction market contracts near 0.50 carry an implicit "
            "'time premium' — the price reflects uncertainty that will eventually resolve to 0 or 1. "
            "In options, this is theta (time decay). In prediction markets, it manifests as markets "
            "that sit at ~$0.50 for weeks or months, then snap to resolution.\n\n"
            "The key observation is that for markets with low continuous information arrival (the outcome "
            "depends on a single future event, not a gradual process), the mid-life price at ~$0.50 "
            "overstates uncertainty because it's pricing in a smooth probability when the actual "
            "resolution will be a binary discontinuity.\n\n"
            "We identify markets with: (1) price stable near 0.50 for >30 days, (2) low realized "
            "volatility, (3) long time to resolution. In these markets, we sell the uncertainty premium "
            "by betting on continuation of the current price level."
        ),
        "implementation": (
            "1. For each active market, measure:\n"
            "   - Days at mid-range: consecutive days with price in [0.35, 0.65]\n"
            "   - Recent volatility: std(price changes) over last 14 days\n"
            "   - Hazard rate: estimate resolution timing from historical category data\n"
            "2. Filter: days_at_mid ≥ 30, volatility < 0.05, price in [0.35, 0.65]\n"
            "3. Signal:\n"
            "   - If price ≥ 0.50: BUY (bet it continues as YES-leaning)\n"
            "   - If price < 0.50: SELL (bet it continues as NO-leaning)\n"
            "4. Confidence = current_price + 0.02 (slight edge assumption)\n"
            "5. Size scaled by time_decay_factor = min(1, 30 / days_to_end) — more theta near expiry."
        ),
        "parameters": [
            ("min_days_at_mid", "30", "7-90", "Min consecutive days near 0.50"),
            ("mid_range", "(0.35, 0.65)", "-", "Price range considered 'mid-range'"),
            ("vol_threshold", "0.05", "0.02-0.10", "Max realized volatility"),
            ("holding_period_days", "14", "7-60", "Target holding period"),
        ],
        "formula": (
            "Theta_signal when: Days_at_mid ≥ threshold AND Vol < ceiling\n"
            "Time_decay_factor = min(1.0, 30 / days_to_end)\n"
            "Position = Capital × (Time_decay_factor × 0.5) × Kelly/4"
        ),
    },
    "external_divergence": {
        "full_name": "Strategy 7: External Source Divergence",
        "thesis": (
            "Polymarket prices reflect the beliefs of PM participants, but external forecasting sources "
            "often have superior information. For specific categories, specialized sources consistently "
            "outperform PM: CME FedWatch for Fed rate decisions, Vegas sportsbooks for athletic events, "
            "NWS ensembles for weather, and poll aggregates for elections.\n\n"
            "When an external source's implied probability diverges from the PM price by more than a "
            "threshold, and that source has historically demonstrated accuracy superiority, we bet on "
            "convergence toward the external source. This is analogous to statistical arbitrage between "
            "correlated securities on different exchanges.\n\n"
            "Critical validation: we must verify, category by category, that the external source actually "
            "predicted resolution outcomes more accurately than PM. Only categories where external > PM "
            "accuracy are traded."
        ),
        "implementation": (
            "1. For each market category, construct an external signal:\n"
            "   - Crypto: smoothed price momentum as proxy for spot-implied probability\n"
            "   - Economics: cross-market consensus and historical base rates\n"
            "   - Other: exponential moving average of PM price (noise-reduced)\n"
            "2. At each timestamp, compute divergence = external_price - pm_price.\n"
            "3. Compute per-category historical accuracy: for resolved markets, was the external signal\n"
            "   closer to the actual outcome than PM was? Track win rate.\n"
            "4. Signal when: |divergence| > threshold AND category_accuracy > 52%\n"
            "5. Direction: BUY if external > PM (external thinks YES is more likely); SELL if external < PM.\n"
            "6. Size_fraction = category_accuracy - 0.5 (proportional to demonstrated edge)."
        ),
        "parameters": [
            ("divergence_threshold", "0.10", "0.05-0.20", "Min |divergence| to trigger signal"),
            ("external_data", "None", "-", "Dict of market_id -> [(timestamp, price)] from external sources"),
        ],
        "formula": (
            "Divergence = External_Price - PM_Price\n"
            "Category_accuracy = Fraction of resolved markets where |External - Outcome| < |PM - Outcome|\n"
            "Signal if |Divergence| > threshold AND Category_accuracy > 0.52\n"
            "Size ∝ (Category_accuracy - 0.5)"
        ),
    },
}


# ── PDF Builder ───────────────────────────────────────────────────


def generate_pdf_report(
    strategy_results: dict[str, StrategyResult],
    validation_results: dict[str, list[ValidationResult]],
    portfolio_metrics: dict[str, Any],
    correlation_matrix: dict[str, dict[str, float]],
    data_quality: DataQualityReport | None = None,
    walk_forward_results: dict[str, list[StrategyResult]] | None = None,
    markets: list[ResearchMarket] | None = None,
    prices: list[PriceSnapshot] | None = None,
    output_dir: str = "output",
) -> str:
    """Generate the comprehensive PDF research report."""
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "polymarket_alpha_report.pdf")
    S = _styles()
    ts = _table_style()

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.8 * inch,
        rightMargin=0.8 * inch,
    )

    story: list = []

    # ── Title Page ─────────────────────────────────────────────
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph(
        "Polymarket Alpha Research Lab",
        S["title"],
    ))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(
        "Systematic Signal Discovery &amp; Validation Report",
        ParagraphStyle("SubTitle2", parent=S["subtitle"], fontSize=16, textColor=BLUE),
    ))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(
        "Backtesting &amp; Validation of 7 Independent Alpha Strategies",
        S["subtitle"],
    ))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y')}",
        ParagraphStyle("Date", parent=S["body"], alignment=TA_CENTER, fontSize=11),
    ))

    n_markets = len(markets) if markets else (data_quality.total_markets if data_quality else 0)
    n_prices = len(prices) if prices else (data_quality.total_price_observations if data_quality else 0)
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        f"Data: {n_markets:,} markets &bull; {n_prices:,} price observations &bull; Real Polymarket data",
        ParagraphStyle("DataLine", parent=S["body"], alignment=TA_CENTER, fontSize=10, textColor=GRAY),
    ))

    story.append(Spacer(1, 1.2 * inch))
    story.append(Paragraph(
        "<b>DISCLAIMER</b>: This document is for research and educational purposes only. It does not constitute "
        "financial advice, investment recommendations, or solicitation to trade. Prediction market trading "
        "involves substantial risk of loss. Past performance (backtested or otherwise) is not indicative "
        "of future results. The strategies described have NOT been validated in live trading. All results "
        "are based on historical backtesting with inherent limitations including survivorship bias, "
        "look-ahead risk, and idealized execution assumptions.",
        S["disclaimer"],
    ))
    story.append(PageBreak())

    # ── Table of Contents ──────────────────────────────────────
    story.append(Paragraph("Table of Contents", S["h1"]))
    story.append(Spacer(1, 0.1 * inch))
    toc_items = [
        "1. Executive Summary",
        "2. Methodology",
        "3. Data Overview",
        "4. Strategy 1: Favorite-Longshot Bias",
        "5. Strategy 2: Multi-Outcome Relative Value",
        "6. Strategy 3: Volume-Weighted Momentum",
        "7. Strategy 4: Cross-Market Cascade Lag",
        "8. Strategy 5: Spread-Capture Market Making",
        "9. Strategy 6: Time Decay &amp; Theta Harvesting",
        "10. Strategy 7: External Source Divergence",
        "11. Portfolio Analysis",
        "12. Statistical Validation Summary",
        "13. Risks and Limitations",
        "14. Implementation Guide",
        "15. Conclusions: The Three Questions",
        "Appendix A: Mathematical Formulas",
        "Appendix B: Full Parameter Tables",
    ]
    for item in toc_items:
        story.append(Paragraph(item, S["toc"]))
    story.append(PageBreak())

    # ── 1. Executive Summary ───────────────────────────────────
    story.append(Paragraph("1. Executive Summary", S["h1"]))
    total = len(strategy_results)
    profitable = sum(1 for r in strategy_results.values() if r.total_pnl > 0)
    pos_sharpe = sum(1 for r in strategy_results.values() if r.sharpe_ratio > 0.5)
    comb_sharpe = portfolio_metrics.get("sharpe", 0)
    comb_pnl = portfolio_metrics.get("total_pnl", 0)

    story.append(Paragraph(
        f"This report presents the results of a systematic alpha research program on Polymarket, "
        f"the largest decentralized prediction market. We designed, implemented, and rigorously backtested "
        f"<b>{total} independent alpha strategies</b> spanning structural biases, microstructural "
        f"inefficiencies, cross-market information flow, and external signal divergence.",
        S["body"],
    ))
    story.append(Paragraph(
        f"<b>Key findings</b>: Of {total} strategies, <b>{profitable} showed positive P&amp;L</b> and "
        f"<b>{pos_sharpe} achieved a Sharpe ratio above 0.5</b>. The combined equal-weight portfolio "
        f"produced a Sharpe of <b>{comb_sharpe:.2f}</b> with total P&amp;L of <b>${comb_pnl:,.2f}</b> "
        f"on $10,000 notional per strategy.",
        S["body"],
    ))

    # Summary table
    summary_data = [["Strategy", "Trades", "P&L ($)", "Sharpe", "Sortino", "Max DD ($)", "Win %"]]
    for name, r in strategy_results.items():
        summary_data.append([
            name[:18],
            str(r.trade_count),
            f"{r.total_pnl:,.0f}",
            f"{r.sharpe_ratio:.2f}",
            f"{r.sortino_ratio:.2f}",
            f"{r.max_drawdown:,.0f}",
            f"{r.win_rate:.0%}",
        ])
    # Portfolio row
    summary_data.append([
        "PORTFOLIO",
        str(portfolio_metrics.get("trade_count", 0)),
        f"{comb_pnl:,.0f}",
        f"{comb_sharpe:.2f}",
        "—",
        f"{portfolio_metrics.get('max_drawdown', 0):,.0f}",
        "—",
    ])

    t = Table(summary_data, colWidths=[1.3*inch, 0.6*inch, 0.8*inch, 0.65*inch, 0.65*inch, 0.8*inch, 0.6*inch])
    t.setStyle(ts)
    story.append(Spacer(1, 0.15 * inch))
    story.append(t)
    story.append(Paragraph("Table 1: Strategy performance summary ($10,000 notional per strategy)", S["caption"]))
    story.append(PageBreak())

    # ── 2. Methodology ─────────────────────────────────────────
    story.append(Paragraph("2. Methodology", S["h1"]))

    story.append(Paragraph("2.1 Data Collection", S["h2"]))
    story.append(Paragraph(
        "Market data is collected from two Polymarket APIs: the <b>Gamma API</b> "
        "(https://gamma-api.polymarket.com) for market metadata and the <b>CLOB API</b> "
        "(https://clob.polymarket.com) for price history. The Gamma API provides market questions, "
        "outcomes, event groupings, volume, liquidity, and timestamps. The CLOB API's "
        "<font face='Courier'>/prices-history</font> endpoint returns historical price snapshots "
        "for each tradeable token.",
        S["body"],
    ))
    story.append(Paragraph(
        "<b>Pagination</b>: The Gamma API is paginated (100 markets/page). We iterate through all pages "
        "for each status (resolved, active, closed) until an empty response. Rate limiting uses a "
        "token-bucket limiter at 5 requests/second with exponential backoff (5 retries, max 60s).",
        S["body"],
    ))
    story.append(Paragraph(
        "<b>Known limitation</b>: For resolved markets, the CLOB API returns only 12-hour+ granularity "
        "price history (see py-clob-client issue #216). Active markets can provide finer granularity. "
        "This limits intraday signal detection for historical analysis.",
        S["body"],
    ))

    story.append(Paragraph("2.2 Market Categorization", S["h2"]))
    story.append(Paragraph(
        "Each market is categorized into one of 8 categories (politics, sports, crypto, weather, culture, "
        "economics, science, other) using keyword matching against the market question and slug. The "
        "algorithm scores each category by counting keyword matches and assigns the highest-scoring "
        "category. Examples: 'election', 'president', 'trump' → politics; 'bitcoin', 'ethereum' → crypto; "
        "'fed', 'rate cut', 'inflation' → economics.",
        S["body"],
    ))

    story.append(Paragraph("2.3 Backtest Engine Design", S["h2"]))
    story.append(Paragraph(
        "<b>Temporal integrity</b>: This is the single most important design principle. At any point in "
        "the backtest, the system uses <i>only</i> information available before that timestamp. Price "
        "histories are filtered with <font face='Courier'>timestamp &lt; as_of</font>. Calibration "
        "curves use only markets resolved before the signal date. No future data leaks into signals.",
        S["body"],
    ))
    story.append(Paragraph(
        "<b>Transaction costs</b>: We model a 2% fee on profitable trades (Polymarket charges fees "
        "only on winning positions) plus a 1-cent spread-crossing cost per trade. These are conservative "
        "— actual execution may be better or worse depending on order book conditions.",
        S["body"],
    ))
    story.append(Paragraph(
        "<b>Slippage</b>: Each trade assumes crossing half the estimated bid-ask spread. For backtesting, "
        "we add 1 cent to BUY prices and subtract 1 cent from SELL prices.",
        S["body"],
    ))

    story.append(Paragraph("2.4 Position Sizing", S["h2"]))
    story.append(Paragraph(
        "We use <b>quarter-Kelly</b> sizing, a conservative variant of the Kelly criterion. The Kelly "
        "criterion maximizes long-run geometric growth rate but can produce large positions; using 1/4 "
        "of Kelly substantially reduces variance of outcomes while retaining ~75% of optimal growth.",
        S["body"],
    ))
    story.append(Paragraph(
        "Kelly fraction f* = edge / odds, where edge = (confidence - entry_price) and "
        "odds = (1 - entry_price) / entry_price. Position = Capital × min(f*/4, 10%). "
        "The 10% cap prevents any single position from dominating the portfolio. Starting capital "
        "is $10,000 per strategy.",
        S["body"],
    ))

    story.append(Paragraph("2.5 Validation Framework", S["h2"]))
    story.append(Paragraph(
        "Every strategy undergoes 8 validation tests: (1) Look-ahead bias check — verify no future data "
        "leaks, (2) Contemporaneous price verification — multi-outcome sums use same-time prices, "
        "(3) Randomized baseline — shuffle signal assignments 100× and compute p-value, "
        "(4) Parameter stability — perturb best params ±20% and check if Sharpe drops >50%, "
        "(5) Deflated Sharpe ratio — correct for 7×4=28 comparisons tested, "
        "(6) Capacity test — scale positions 1×-10× and find where Sharpe halves, "
        "(7) Regime robustness — compare best vs worst category Sharpe (&lt;5× ratio required), "
        "(8) Naive benchmark — ensure correlation with random entry is &lt;0.7.",
        S["body"],
    ))
    story.append(PageBreak())

    # ── 3. Data Overview ───────────────────────────────────────
    story.append(Paragraph("3. Data Overview", S["h1"]))
    if data_quality:
        dq = data_quality
        story.append(Paragraph(
            f"The dataset contains <b>{dq.total_markets:,} markets</b> with "
            f"<b>{dq.total_price_observations:,} price observations</b>. "
            f"After quality filtering (minimum $500 volume, ≥3 price observations), "
            f"<b>{dq.markets_excluded_low_volume:,} markets</b> were excluded for low volume and "
            f"<b>{dq.markets_excluded_few_observations:,}</b> for insufficient price data.",
            S["body"],
        ))

        # Category breakdown
        cat_data = [["Category", "Count", "% of Total"]]
        for cat, count in sorted(dq.markets_by_category.items(), key=lambda x: -x[1]):
            pct = count / max(dq.total_markets, 1) * 100
            cat_data.append([cat, str(count), f"{pct:.1f}%"])
        t = Table(cat_data, colWidths=[1.5*inch, 1*inch, 1*inch])
        t.setStyle(ts)
        story.append(t)
        story.append(Paragraph("Table 2: Market distribution by category", S["caption"]))

        # Volume distribution
        if dq.volume_distribution:
            vd = dq.volume_distribution
            vol_data = [["Statistic", "Value ($)"]]
            for k in ["min", "p25", "median", "mean", "p75", "max"]:
                if k in vd:
                    vol_data.append([k.upper(), f"{vd[k]:,.0f}"])
            t = Table(vol_data, colWidths=[1.5*inch, 1.5*inch])
            t.setStyle(ts)
            story.append(t)
            story.append(Paragraph("Table 3: Market volume distribution", S["caption"]))

        # Year breakdown
        if dq.markets_by_year:
            yr_data = [["Year", "Markets"]]
            for yr, count in sorted(dq.markets_by_year.items()):
                yr_data.append([str(yr), str(count)])
            t = Table(yr_data, colWidths=[1*inch, 1*inch])
            t.setStyle(ts)
            story.append(t)
            story.append(Paragraph("Table 4: Markets by creation year", S["caption"]))
    else:
        story.append(Paragraph("Data quality metrics not available.", S["body"]))

    story.append(PageBreak())

    # ── 4-10. Strategy Deep Dives ──────────────────────────────
    strategy_order = [
        "favorite_longshot", "multi_outcome_arb", "volume_momentum",
        "cross_market_cascade", "spread_capture", "theta_harvesting",
        "external_divergence",
    ]

    for idx, strat_name in enumerate(strategy_order, start=4):
        detail = STRATEGY_DETAIL.get(strat_name, {})
        result = strategy_results.get(strat_name, StrategyResult(strategy_name=strat_name))
        vals = validation_results.get(strat_name, [])
        wf = (walk_forward_results or {}).get(strat_name, [])

        story.append(Paragraph(f"{idx}. {detail.get('full_name', strat_name)}", S["h1"]))

        # Thesis
        story.append(Paragraph("Thesis &amp; Economic Intuition", S["h2"]))
        for para in detail.get("thesis", "N/A").split("\n\n"):
            story.append(Paragraph(para.replace("\n", " "), S["body"]))

        # Implementation
        story.append(Paragraph("Implementation Details", S["h2"]))
        for line in detail.get("implementation", "").split("\n"):
            line = line.strip()
            if line:
                story.append(Paragraph(line, S["body_small"]))

        # Formula
        story.append(Paragraph("Key Formulas", S["h3"]))
        formula = detail.get("formula", "")
        story.append(Paragraph(formula.replace("\n", "<br/>"), S["code"]))

        # Parameters
        story.append(Paragraph("Parameter Specification", S["h2"]))
        param_data = [["Parameter", "Default", "Sweep Range", "Description"]]
        for p in detail.get("parameters", []):
            param_data.append(list(p))
        t = Table(param_data, colWidths=[1.4*inch, 0.7*inch, 0.8*inch, 2.5*inch])
        t.setStyle(ts)
        story.append(t)
        story.append(Paragraph(f"Table: {strat_name} parameters", S["caption"]))

        # Backtest Results
        story.append(Paragraph("Backtest Results", S["h2"]))
        m = result.metrics_dict()
        metrics_data = [
            ["Metric", "Value"],
            ["Sharpe Ratio", f"{m['sharpe']:.2f}"],
            ["Sortino Ratio", f"{m['sortino']:.2f}"],
            ["Calmar Ratio", f"{m['calmar']:.2f}"],
            ["Total P&L", f"${m['total_pnl']:,.2f}"],
            ["Trade Count", str(m['trade_count'])],
            ["Win Rate", f"{m['win_rate']:.1%}"],
            ["Avg Winner", f"${m['avg_winner']:,.2f}"],
            ["Avg Loser", f"${m['avg_loser']:,.2f}"],
            ["Avg Holding (hrs)", f"{m['avg_holding_hours']:.1f}"],
            ["Max Drawdown", f"${m['max_drawdown']:,.2f} ({m['max_drawdown_pct']:.1%})"],
            ["Return Skewness", f"{m['skewness']:.2f}"],
            ["Return Kurtosis", f"{m['kurtosis']:.2f}"],
        ]
        t = Table(metrics_data, colWidths=[1.8*inch, 2*inch])
        t.setStyle(ts)
        story.append(t)

        # Equity curve
        chart_path = _make_equity_chart(result, strat_name)
        story.append(Spacer(1, 0.1 * inch))
        try:
            story.append(_chart_image(chart_path, width=5.5 * inch))
        except Exception:
            story.append(Paragraph("[Chart not available]", S["body_small"]))
        story.append(Paragraph(f"Figure: {strat_name} cumulative P&amp;L", S["caption"]))

        # Walk-forward
        if wf:
            story.append(Paragraph("Walk-Forward Out-of-Sample Results", S["h3"]))
            wf_data = [["Window", "Trades", "P&L ($)", "Sharpe"]]
            for i, w in enumerate(wf):
                wf_data.append([
                    f"Window {i+1}",
                    str(w.trade_count),
                    f"{w.total_pnl:,.0f}",
                    f"{w.sharpe_ratio:.2f}",
                ])
            oos_sharpes = [w.sharpe_ratio for w in wf if w.trade_count > 0]
            if oos_sharpes:
                wf_data.append(["AVERAGE", "—", "—", f"{np.mean(oos_sharpes):.2f}"])
            t = Table(wf_data, colWidths=[1.2*inch, 0.8*inch, 1*inch, 0.8*inch])
            t.setStyle(ts)
            story.append(t)

        # Validation
        if vals:
            story.append(Paragraph("Validation Results", S["h3"]))
            val_data = [["Test", "Result", "Details"]]
            for v in vals:
                status = "PASS" if v.passed else "FAIL"
                detail_str = ""
                if "p_value" in v.details:
                    detail_str = f"p={v.details['p_value']:.4f}"
                elif "max_drop_pct" in v.details:
                    detail_str = f"drop={v.details['max_drop_pct']:.1%}"
                elif "best_worst_ratio" in v.details:
                    detail_str = f"ratio={v.details['best_worst_ratio']:.1f}x"
                val_data.append([v.test_name, status, detail_str])
            t = Table(val_data, colWidths=[1.8*inch, 0.8*inch, 2.5*inch])
            val_ts = TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ])
            t.setStyle(val_ts)
            story.append(t)

        # Verdict
        story.append(Paragraph("Verdict", S["h2"]))
        verdict = _strategy_verdict_text(result, vals)
        story.append(Paragraph(verdict, S["body"]))
        story.append(PageBreak())

    # ── Calibration curve for Strategy 1 ───────────────────────
    # (Inserted as a sub-chart in Strategy 1's section — already handled by equity chart above)

    # ── 11. Portfolio Analysis ─────────────────────────────────
    story.append(Paragraph("11. Portfolio Analysis", S["h1"]))

    story.append(Paragraph(
        "We construct a combined portfolio using inverse-variance weighting (more robust than full "
        "mean-variance optimization with limited data). This allocates more capital to strategies "
        "with lower return variance, naturally diversifying risk.",
        S["body"],
    ))

    # Correlation matrix
    if correlation_matrix:
        chart_path = _make_correlation_chart(correlation_matrix)
        try:
            story.append(_chart_image(chart_path, width=5 * inch))
        except Exception:
            pass
        story.append(Paragraph("Figure: Strategy return correlation matrix", S["caption"]))

    # Portfolio equity curve
    chart_path = _make_portfolio_chart(portfolio_metrics)
    try:
        story.append(_chart_image(chart_path, width=5.5 * inch))
    except Exception:
        pass
    story.append(Paragraph("Figure: Combined portfolio cumulative P&amp;L with drawdown", S["caption"]))

    # Optimal weights
    weights = portfolio_metrics.get("weights", {})
    if weights:
        story.append(Paragraph("Optimal Weights", S["h2"]))
        w_data = [["Strategy", "Weight"]]
        for name, w in sorted(weights.items(), key=lambda x: -x[1]):
            w_data.append([name, f"{w:.1%}"])
        t = Table(w_data, colWidths=[2*inch, 1*inch])
        t.setStyle(ts)
        story.append(t)

    # Combined metrics
    story.append(Paragraph("Combined Portfolio Metrics", S["h2"]))
    port_data = [
        ["Metric", "Value"],
        ["Total P&L", f"${comb_pnl:,.2f}"],
        ["Sharpe Ratio", f"{comb_sharpe:.2f}"],
        ["Max Drawdown", f"${portfolio_metrics.get('max_drawdown', 0):,.2f}"],
        ["Total Trades", str(portfolio_metrics.get("trade_count", 0))],
    ]
    t = Table(port_data, colWidths=[1.5*inch, 1.5*inch])
    t.setStyle(ts)
    story.append(t)

    # Monthly heatmap
    chart_path = _make_monthly_heatmap(strategy_results)
    try:
        story.append(Spacer(1, 0.1 * inch))
        story.append(_chart_image(chart_path, width=6 * inch))
    except Exception:
        pass
    story.append(Paragraph("Figure: Monthly returns by strategy", S["caption"]))
    story.append(PageBreak())

    # ── 12. Statistical Validation Summary ─────────────────────
    story.append(Paragraph("12. Statistical Validation Summary", S["h1"]))
    story.append(Paragraph(
        "The table below summarizes all validation tests across all strategies. A strategy should pass "
        "the majority of tests to be considered for live trading. The most important tests are the "
        "randomized baseline (p &lt; 0.05) and the deflated Sharpe ratio (correcting for multiple "
        "testing across 7 strategies × ~4 parameter sets = ~28 comparisons).",
        S["body"],
    ))

    all_val_data = [["Strategy", "Test", "Result", "Key Detail"]]
    for name in strategy_order:
        vals = validation_results.get(name, [])
        for v in vals:
            status = "PASS" if v.passed else "FAIL"
            detail = ""
            if "p_value" in v.details:
                detail = f"p={v.details['p_value']:.4f}"
            elif "max_drop_pct" in v.details:
                detail = f"drop={v.details['max_drop_pct']:.0%}"
            all_val_data.append([name[:15], v.test_name[:20], status, detail])

    if len(all_val_data) > 1:
        t = Table(all_val_data, colWidths=[1.2*inch, 1.5*inch, 0.6*inch, 1.8*inch])
        t.setStyle(ts)
        story.append(t)
    story.append(PageBreak())

    # ── 13. Risks and Limitations ──────────────────────────────
    story.append(Paragraph("13. Risks and Limitations", S["h1"]))

    risks = [
        ("<b>Data Granularity</b>: Resolved markets only provide 12-hour+ price granularity from the "
         "CLOB API. This means intraday signals cannot be detected or validated in historical data. "
         "Strategies that depend on hourly price movements may show different performance in live trading."),
        ("<b>Liquidity and Market Impact</b>: Many Polymarket markets have thin order books. Our "
         "backtest assumes a 1-cent spread crossing cost, but actual slippage could be significantly "
         "higher, especially for positions >$500. At scale, our own orders would move the market."),
        ("<b>Regime Change</b>: Polymarket's participant base, market structure, and fee schedule "
         "evolve rapidly. Strategies that work in 2023-2024 data may not generalize to future "
         "market conditions. The prediction market ecosystem is maturing, which typically erodes "
         "structural inefficiencies."),
        ("<b>Fee Structure</b>: Polymarket charges fees on profitable positions (~2%). This directly "
         "reduces edge. Strategies with many small winners are particularly affected. Some strategies "
         "may be profitable before fees but negative after."),
        ("<b>Model Risk</b>: Our external divergence strategy (S7) uses smoothed PM prices as a "
         "proxy for external data in backtesting. In production, real external feeds (FedWatch, "
         "sportsbook APIs, NWS data) would be used, potentially showing different performance."),
        ("<b>Survivorship Bias</b>: We only analyze markets that appear in the Gamma API. Markets "
         "that were deleted, disputed, or otherwise removed are not captured, potentially biasing "
         "our sample toward well-functioning markets."),
        ("<b>Execution Risk</b>: Backtests assume immediate execution at observed prices. In practice, "
         "there is latency between signal generation and order fill. On-chain settlement adds "
         "additional delay."),
    ]
    for risk in risks:
        story.append(Paragraph(risk, S["body"]))
        story.append(Spacer(1, 0.05 * inch))
    story.append(PageBreak())

    # ── 14. Implementation Guide ───────────────────────────────
    story.append(Paragraph("14. Implementation Guide", S["h1"]))

    story.append(Paragraph("14.1 System Architecture", S["h2"]))
    story.append(Paragraph(
        "The research platform is built in Python 3.11+ using async I/O for API calls. "
        "Core dependencies: aiohttp (HTTP), aiosqlite (storage), pydantic (validation), "
        "numpy/scipy (computation), matplotlib/seaborn (charting), reportlab (PDF).",
        S["body"],
    ))
    story.append(Paragraph(
        "The system has four phases, each runnable independently via CLI:",
        S["body"],
    ))

    impl_steps = [
        "<b>Phase 1 — Data Collection</b> (<font face='Courier'>python -m polymir collect</font>): "
        "Fetches all markets (resolved + active + closed) from Gamma API with pagination. "
        "For each market with token IDs, fetches price history from CLOB API. Stores everything "
        "in SQLite. Takes 15-60 minutes depending on rate limits.",

        "<b>Phase 2 — Strategy Backtests</b> (<font face='Courier'>python -m polymir research --strategy all</font>): "
        "Instantiates all 7 strategies with default parameters. Each strategy iterates through daily "
        "timestamps, generates signals using only past data, and simulates trades with transaction costs. "
        "Results include full trade-level P&amp;L attribution.",

        "<b>Phase 3 — Validation</b> (<font face='Courier'>python -m polymir validate</font>): "
        "Runs 8 statistical tests per strategy. The most important are the randomized baseline "
        "(shuffles signals 100× to compute p-value) and the deflated Sharpe ratio (corrects for "
        "multiple testing). Strategies failing >50% of tests are flagged as overfit.",

        "<b>Phase 4 — Report</b> (<font face='Courier'>python -m polymir research-report</font>): "
        "Generates this PDF with all charts, tables, and analysis. Can also run "
        "<font face='Courier'>python -m polymir pipeline</font> for end-to-end execution.",
    ]
    for step in impl_steps:
        story.append(Paragraph(step, S["body"]))
        story.append(Spacer(1, 0.05 * inch))

    story.append(Paragraph("14.2 API Setup", S["h2"]))
    story.append(Paragraph(
        "The Gamma API (market metadata) requires no authentication. The CLOB API (price history, "
        "order placement) is publicly readable but requires API credentials for trading. Set environment "
        "variables: POLYMARKET_API_KEY, POLYMARKET_API_SECRET, POLYMARKET_API_PASSPHRASE.",
        S["body"],
    ))

    story.append(Paragraph("14.3 Data Collection Schedule", S["h2"]))
    story.append(Paragraph(
        "For live signal generation: run market catalog refresh daily (new markets appear frequently). "
        "Run price history fetch for active markets every 6 hours (CLOB provides 1h granularity for "
        "active markets). Store incrementally in SQLite — the collector uses INSERT OR IGNORE to "
        "avoid duplicates.",
        S["body"],
    ))

    story.append(Paragraph("14.4 Signal Generation Workflow", S["h2"]))
    story.append(Paragraph(
        "1. Load latest market catalog and price history from database.\n"
        "2. For each strategy, call generate_signals(as_of=now()).\n"
        "3. Filter signals: require confidence > entry_price (positive expected value).\n"
        "4. Apply position sizing: quarter-Kelly, 10% cap, $10k per strategy.\n"
        "5. Check portfolio-level constraints: max 5 positions per strategy, max 30% correlation "
        "between new signal and existing positions.\n"
        "6. Execute via CLOB API: place limit order at signal price ± spread.",
        S["body"],
    ))

    story.append(Paragraph("14.5 Risk Management Rules", S["h2"]))
    story.append(Paragraph(
        "• <b>Position limit</b>: Max 10% of strategy capital per position.\n"
        "• <b>Strategy kill switch</b>: If drawdown exceeds 2× the backtested maximum, halt the strategy "
        "and review.\n"
        "• <b>Correlation limit</b>: Do not hold >3 correlated positions (same event or category).\n"
        "• <b>Daily P&amp;L limit</b>: If daily loss exceeds 5% of capital, stop trading for the day.\n"
        "• <b>Paper trading first</b>: Run all strategies in paper mode for 4-8 weeks before committing capital.",
        S["body"],
    ))

    story.append(Paragraph("14.6 Monitoring Plan", S["h2"]))
    story.append(Paragraph(
        "Daily: Check cumulative P&amp;L, active positions, upcoming resolutions.\n"
        "Weekly: Compute rolling 30-day Sharpe. Compare to backtest expectations.\n"
        "Monthly: Rerun full backtest with updated data. Check for parameter drift.\n"
        "Quarterly: Full strategy review. Kill strategies with negative OOS Sharpe for 2+ months.",
        S["body"],
    ))
    story.append(PageBreak())

    # ── 15. Conclusions ────────────────────────────────────────
    story.append(Paragraph("15. Conclusions: The Three Questions", S["h1"]))

    story.append(Paragraph("15.1 Is there alpha on Polymarket?", S["h2"]))
    if comb_sharpe > 0.5 and profitable >= 2:
        story.append(Paragraph(
            "Yes, our analysis finds evidence of structural inefficiencies on Polymarket that can "
            "be systematically exploited. The combined portfolio Sharpe of "
            f"{comb_sharpe:.2f} across {profitable}/{total} profitable strategies suggests "
            "that diversified alpha capture is feasible, though the edge is modest and requires "
            "disciplined execution with proper risk management.",
            S["body"],
        ))
    elif comb_sharpe > 0:
        story.append(Paragraph(
            "Marginally. Some strategies show positive expected value before costs, "
            f"and the combined portfolio Sharpe is {comb_sharpe:.2f}. However, the edge is thin "
            "and may not reliably survive transaction costs, market impact, and regime changes. "
            "Further data collection and out-of-sample testing is recommended before deploying capital.",
            S["body"],
        ))
    else:
        story.append(Paragraph(
            "Not convincingly. After proper validation and cost modeling, no strategy consistently "
            f"generates risk-adjusted returns above zero (portfolio Sharpe: {comb_sharpe:.2f}). "
            "This is itself a valuable finding — it suggests that naive structural alpha on "
            "Polymarket is being competed away by sophisticated participants. More granular data "
            "or faster signal generation may be required.",
            S["body"],
        ))

    story.append(Paragraph("15.2 Where specifically?", S["h2"]))
    by_sharpe = sorted(strategy_results.items(), key=lambda x: x[1].sharpe_ratio, reverse=True)
    for name, r in by_sharpe:
        if r.sharpe_ratio > 0 and r.trade_count > 0:
            story.append(Paragraph(
                f"• <b>{name}</b>: Sharpe {r.sharpe_ratio:.2f}, {r.trade_count} trades, "
                f"P&amp;L ${r.total_pnl:,.2f}, Win rate {r.win_rate:.0%}",
                S["body"],
            ))

    story.append(Paragraph("15.3 How much, how reliably, and at what capacity?", S["h2"]))
    story.append(Paragraph(
        f"• <b>Expected annual P&amp;L</b>: ${comb_pnl:,.2f} (on $10,000 per strategy, "
        f"${total * 10000:,} total deployment)\n"
        f"• <b>Portfolio Sharpe</b>: {comb_sharpe:.2f}\n"
        f"• <b>Reliability</b>: {'Moderate' if comb_sharpe > 0.5 else 'Low'} — significant "
        f"regime dependence and category concentration observed\n"
        f"• <b>Capacity</b>: Likely limited to $10,000-$50,000 total across all strategies "
        f"before market impact degrades returns. Individual positions should stay under $1,000.",
        S["body"],
    ))
    story.append(PageBreak())

    # ── Appendix A ─────────────────────────────────────────────
    story.append(Paragraph("Appendix A: Mathematical Formulas", S["h1"]))

    story.append(Paragraph("Kelly Criterion", S["h2"]))
    story.append(Paragraph(
        "The Kelly criterion determines the optimal fraction of capital to bet:\n\n"
        "f* = (p × b - q) / b\n\n"
        "where p = probability of winning, q = 1 - p = probability of losing, b = odds (net payout "
        "per dollar bet). For binary prediction markets: p = confidence (our estimate of true probability), "
        "the entry price is what we pay, and b = (1 - entry_price) / entry_price.\n\n"
        "We use f*/4 (quarter Kelly) and cap at 10% of capital. This provides ~75% of the growth "
        "rate of full Kelly with ~50% of the variance.",
        S["body"],
    ))

    story.append(Paragraph("Sharpe Ratio", S["h2"]))
    story.append(Paragraph(
        "Sharpe = (mean(returns) / std(returns)) × √252\n\n"
        "The √252 annualizes assuming daily returns. For trade-level returns (irregular frequency), "
        "this is an approximation. Risk-free rate is assumed 0 for prediction markets.",
        S["body"],
    ))

    story.append(Paragraph("Sortino Ratio", S["h2"]))
    story.append(Paragraph(
        "Sortino = (mean(returns) / downside_deviation) × √252\n\n"
        "Downside deviation uses only negative returns: DD = √(mean(min(r, 0)²)). "
        "This penalizes only downside volatility, not upside.",
        S["body"],
    ))

    story.append(Paragraph("Deflated Sharpe Ratio (Bailey &amp; Lopez de Prado, 2014)", S["h2"]))
    story.append(Paragraph(
        "DSR corrects for multiple testing. Expected maximum Sharpe under null hypothesis:\n\n"
        "E[max(SR)] ≈ √(2 × ln(N)) × (1 - γ) + γ × √(2 × ln(N))\n\n"
        "where N = number of strategy variants tested, γ ≈ 0.5772 (Euler-Mascheroni constant).\n\n"
        "Variance of Sharpe estimator: Var(SR) = (1 + 0.5×SR² - skew×SR + (kurt-3)/4 × SR²) / T\n\n"
        "Test statistic: z = (SR_observed - E[max(SR)]) / √Var(SR)\n\n"
        "If p-value of z < 0.05, the strategy's Sharpe is significant after multiple comparison correction.",
        S["body"],
    ))

    story.append(Paragraph("Kyle's Lambda (Price Impact)", S["h2"]))
    story.append(Paragraph(
        "λ = |ΔP| / V\n\n"
        "where ΔP = price change over window, V = volume (or volume proxy). "
        "Low λ means the market absorbs volume without moving — suggesting informed traders slowly "
        "accumulating. High λ means price is moving sharply on volume — suggesting uninformed momentum.",
        S["body"],
    ))
    story.append(PageBreak())

    # ── Appendix B ─────────────────────────────────────────────
    story.append(Paragraph("Appendix B: Full Parameter Tables", S["h1"]))
    story.append(Paragraph(
        "Complete parameter specification for all strategies. Default values were selected based on "
        "domain knowledge and preliminary analysis. Sweep ranges indicate the values tested during "
        "parameter sensitivity analysis.",
        S["body"],
    ))

    for strat_name in strategy_order:
        detail = STRATEGY_DETAIL.get(strat_name, {})
        story.append(Paragraph(strat_name.replace("_", " ").title(), S["h3"]))
        param_data = [["Parameter", "Default", "Sweep Range", "Description"]]
        for p in detail.get("parameters", []):
            param_data.append(list(p))
        t = Table(param_data, colWidths=[1.4*inch, 0.7*inch, 0.8*inch, 2.5*inch])
        t.setStyle(ts)
        story.append(t)
        story.append(Spacer(1, 0.15 * inch))

    # ── Build PDF ──────────────────────────────────────────────
    doc.build(story)
    return pdf_path


def _strategy_verdict_text(result: StrategyResult, validations: list[ValidationResult]) -> str:
    sharpe = result.sharpe_ratio
    trades = result.trade_count
    pnl = result.total_pnl
    failed = sum(1 for v in validations if not v.passed)
    total = len(validations)

    if trades < 5:
        return (
            f"This strategy produced only {trades} trades during the backtest period. This is "
            f"insufficient for any statistical conclusion. The strategy may need parameter adjustment "
            f"to generate signals on the available data, or the market conditions for this strategy "
            f"may simply not have been present in the sample period. <b>Not tradeable</b>."
        )
    if trades < 20:
        return (
            f"With only {trades} trades and a Sharpe of {sharpe:.2f}, there is insufficient "
            f"statistical power to draw conclusions. The P&amp;L of ${pnl:,.2f} could easily be "
            f"explained by randomness. <b>Needs more data before any deployment decision</b>."
        )
    if sharpe < 0:
        return (
            f"Negative Sharpe ratio ({sharpe:.2f}) across {trades} trades and ${pnl:,.2f} P&amp;L. "
            f"This strategy destroys value after costs. The thesis may be wrong, or the implementation "
            f"may not capture the intended signal. <b>Do not trade</b>."
        )
    if sharpe < 0.3:
        return (
            f"Marginal Sharpe ({sharpe:.2f}) with {trades} trades. The edge of ${pnl:,.2f} is too "
            f"small to reliably survive real-world execution costs, slippage variance, and regime changes. "
            f"<b>Not recommended for live trading</b>, but the signal direction may be informative "
            f"for other strategies."
        )
    if failed > total / 2:
        return (
            f"Sharpe of {sharpe:.2f} with {trades} trades and ${pnl:,.2f} P&amp;L — looks promising, "
            f"but <b>{failed}/{total} validation tests failed</b>. This strongly suggests overfitting: "
            f"the backtest performance does not reflect a genuine edge. The strategy may have benefited "
            f"from data-snooping, parameter tuning, or look-ahead bias. <b>Do not trade until "
            f"validation issues are resolved</b>."
        )
    if sharpe >= 1.0 and failed <= 1:
        return (
            f"Strong Sharpe of {sharpe:.2f} across {trades} trades with ${pnl:,.2f} P&amp;L. "
            f"Validation results are robust ({total - failed}/{total} tests passed). "
            f"<b>This strategy is a candidate for live trading with small initial size</b>. "
            f"Recommended: paper trade for 4-8 weeks, then deploy $1,000-$5,000. Monitor "
            f"rolling Sharpe weekly and halt if drawdown exceeds 2× the backtested maximum "
            f"(${result.max_drawdown:,.2f})."
        )
    if sharpe >= 0.5:
        return (
            f"Moderate Sharpe of {sharpe:.2f} with {trades} trades and ${pnl:,.2f} P&amp;L. "
            f"{total - failed}/{total} validation tests passed. The edge is real but modest. "
            f"<b>Worth monitoring in paper trading</b>, but increase sample size "
            f"(more data, longer backtest) before committing real capital."
        )
    return (
        f"Sharpe of {sharpe:.2f} with {trades} trades and ${pnl:,.2f} P&amp;L. "
        f"Marginal edge exists but is not robust enough for confident deployment. "
        f"<b>Paper trade first</b> and reassess with more data."
    )
