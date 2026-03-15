#!/usr/bin/env python3
"""Generate a full backtest report as PDF with charts and bias analysis."""

import asyncio
import json
import math
import os
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from polymir.backtest import BacktestEngine, BacktestResult, HistoricalTrade
from polymir.backtest.bias_tests import (
    check_category_concentration,
    check_fee_breakeven,
    check_latency_sensitivity,
    check_parameter_sensitivity,
    check_price_leakage,
    check_resolution_data_leakage,
    check_time_window_stability,
    check_wallet_concentration,
    check_wallet_score_leakage,
)
from polymir.backtest.data import TradeRecord
from polymir.backtest.sweep import SweepConfig, build_heatmap_data, run_sweep
from polymir.backtest.validation import (
    bootstrap_confidence_intervals,
    deflated_sharpe_ratio,
    holm_bonferroni_correction,
    in_out_sample_split,
    minimum_track_record_length,
    randomized_baseline,
    walk_forward,
)
from polymir.config import AppConfig, ExecutionConfig, ScoringConfig
from polymir.scanner import WalletMarketResult, compute_wallet_score

OUTPUT_DIR = "reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Real data fetching from Polymarket APIs ─────────────────────────

CACHE_PATH = os.path.join(OUTPUT_DIR, "polymarket_data_cache.json")

# Category mapping for Gamma API tags
_CATEGORY_MAP = {
    "politics": "politics",
    "sports": "sports",
    "crypto": "crypto",
    "pop-culture": "entertainment",
    "business": "finance",
    "science": "science",
    "world": "politics",
    "finance": "finance",
    "weather": "weather",
}


def _api_get(url: str, retries: int = 3) -> dict | list | None:
    """Fetch JSON from a URL with retries and rate limiting."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "polymir-backtest/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))
            else:
                print(f"  WARNING: Failed to fetch {url}: {e}")
                return None


def _fetch_resolved_markets(limit: int = 200) -> list[dict]:
    """Fetch high-volume resolved markets from the Gamma API."""
    markets = []
    offset = 0
    batch = 100
    while len(markets) < limit:
        url = (
            f"https://gamma-api.polymarket.com/markets?"
            f"closed=true&limit={batch}&offset={offset}"
            f"&order=volumeClob&ascending=false"
        )
        data = _api_get(url)
        if not data or not isinstance(data, list):
            break
        markets.extend(data)
        if len(data) < batch:
            break
        offset += batch
        time.sleep(0.3)
    return markets[:limit]


def _fetch_trades_for_market(condition_id: str, limit: int = 500) -> list[dict]:
    """Fetch trades for a specific market from the Data API.

    Uses the `market=` parameter which correctly filters by conditionId.
    """
    all_trades = []
    # The Data API may paginate; fetch in batches
    # Use `market=` param (not `conditionId=` which doesn't filter)
    offset = 0
    while len(all_trades) < limit:
        batch = min(500, limit - len(all_trades))
        url = (
            f"https://data-api.polymarket.com/trades?"
            f"market={condition_id}&limit={batch}&offset={offset}"
        )
        data = _api_get(url)
        if not data or not isinstance(data, list):
            break
        all_trades.extend(data)
        if len(data) < batch:
            break
        offset += len(data)
        time.sleep(0.1)
    return all_trades


def _parse_gamma_market(m: dict) -> dict | None:
    """Parse a Gamma API market entry into a structured dict."""
    condition_id = m.get("conditionId", "")
    if not condition_id:
        return None

    # Need resolution outcome
    outcome_prices = m.get("outcomePrices", "")
    if isinstance(outcome_prices, str):
        try:
            outcome_prices = json.loads(outcome_prices) if outcome_prices else []
        except json.JSONDecodeError:
            outcome_prices = []

    clob_token_ids = m.get("clobTokenIds", "")
    if isinstance(clob_token_ids, str):
        try:
            clob_token_ids = json.loads(clob_token_ids) if clob_token_ids else []
        except json.JSONDecodeError:
            clob_token_ids = []

    if not outcome_prices or not clob_token_ids:
        return None

    # Determine winner: outcome with price ~1.0 after resolution
    winner_idx = -1
    for i, p in enumerate(outcome_prices):
        try:
            if float(p) > 0.9:
                winner_idx = i
                break
        except (ValueError, TypeError):
            continue
    if winner_idx < 0:
        return None

    closed_time = m.get("endDate") or m.get("closedTime") or m.get("resolutionDate")
    if not closed_time:
        return None

    try:
        if isinstance(closed_time, str):
            # Handle various ISO formats
            closed_time = closed_time.replace("Z", "+00:00")
            if "+" in closed_time:
                closed_time = closed_time.split("+")[0]
            res_date = datetime.fromisoformat(closed_time)
        else:
            res_date = datetime.utcfromtimestamp(closed_time)
    except (ValueError, TypeError, OSError):
        return None

    # Map tags to category
    tags_raw = m.get("tags", [])
    if isinstance(tags_raw, str):
        try:
            tags_raw = json.loads(tags_raw)
        except json.JSONDecodeError:
            tags_raw = []
    category = "other"
    for tag in (tags_raw or []):
        if isinstance(tag, str) and tag.lower() in _CATEGORY_MAP:
            category = _CATEGORY_MAP[tag.lower()]
            break

    return {
        "condition_id": condition_id,
        "question": m.get("question", ""),
        "category": category,
        "resolution_date": res_date.isoformat(),
        "winner_idx": winner_idx,
        "clob_token_ids": clob_token_ids,
        "volume": float(m.get("volume", 0) or 0),
        "outcomes": m.get("outcomes", []),
    }


def fetch_real_data(max_markets: int = 200, trades_per_market: int = 500):
    """Fetch real Polymarket data: resolved markets and their trades.

    Returns:
        (trades, wallet_results, wallet_aliases)
    """
    # Check cache first
    if os.path.exists(CACHE_PATH):
        print(f"  Loading cached data from {CACHE_PATH}...")
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        return _cache_to_objects(cache)

    print("  Fetching resolved markets from Gamma API...")
    raw_markets = _fetch_resolved_markets(limit=max_markets)
    print(f"  Got {len(raw_markets)} raw markets")

    parsed_markets = []
    for m in raw_markets:
        parsed = _parse_gamma_market(m)
        if parsed:
            parsed_markets.append(parsed)

    print(f"  {len(parsed_markets)} markets have valid resolution data")

    # Fetch trades for each market
    all_market_trades: dict[str, list[dict]] = {}
    for i, pm in enumerate(parsed_markets):
        if i % 20 == 0:
            print(f"  Fetching trades: {i}/{len(parsed_markets)} markets...")
        cid = pm["condition_id"]
        raw_trades = _fetch_trades_for_market(cid, limit=trades_per_market)
        if raw_trades:
            all_market_trades[cid] = raw_trades
        time.sleep(0.15)  # Rate limit

    print(f"  Got trades for {len(all_market_trades)} markets")

    # Build cache
    cache = {
        "markets": parsed_markets,
        "trades": all_market_trades,
        "fetched_at": datetime.utcnow().isoformat(),
    }
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)
    print(f"  Cached data to {CACHE_PATH}")

    return _cache_to_objects(cache)


def _cache_to_objects(cache: dict):
    """Convert cached JSON data into HistoricalTrade, WalletMarketResult, etc."""
    markets_by_cid = {}
    for pm in cache["markets"]:
        markets_by_cid[pm["condition_id"]] = pm

    trades: list[HistoricalTrade] = []
    wallet_market_positions: dict[str, dict[str, dict]] = {}  # wallet -> market_id -> position info
    wallet_aliases: dict[str, str] = {}

    all_market_trades = cache.get("trades", {})

    for cid, raw_trades in all_market_trades.items():
        pm = markets_by_cid.get(cid)
        if not pm:
            continue

        res_date = datetime.fromisoformat(pm["resolution_date"])
        winner_idx = pm["winner_idx"]
        clob_token_ids = pm["clob_token_ids"]
        category = pm["category"]

        # Determine which token IDs are winners
        winner_token_id = clob_token_ids[winner_idx] if winner_idx < len(clob_token_ids) else None

        # Group trades by wallet and token
        wallet_token_trades: dict[str, dict[str, list]] = {}  # wallet -> token -> [trades]

        for t in raw_trades:
            wallet = (t.get("proxyWallet") or t.get("maker") or "").lower()
            if not wallet or wallet == "0x" or len(wallet) < 10:
                continue

            asset = t.get("asset", "")
            side = t.get("side", "BUY").upper()
            try:
                size = float(t.get("size", 0))
                price = float(t.get("price", 0))
            except (ValueError, TypeError):
                continue

            if size <= 0 or price <= 0 or price > 1.0:
                continue

            # Parse timestamp
            ts_raw = t.get("timestamp") or t.get("matchTime") or t.get("createdAt")
            if not ts_raw:
                continue
            try:
                if isinstance(ts_raw, (int, float)):
                    ts = datetime.utcfromtimestamp(ts_raw)
                else:
                    ts_str = str(ts_raw).replace("Z", "+00:00")
                    if "+" in ts_str:
                        ts_str = ts_str.split("+")[0]
                    ts = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError, OSError):
                continue

            # Store alias from username field
            uname = t.get("name") or t.get("pseudonym") or t.get("userName") or ""
            # Skip auto-generated names (look like wallet addresses)
            if uname and not uname.startswith("0x") and wallet not in wallet_aliases:
                wallet_aliases[wallet] = uname

            # Track positions
            wallet_token_trades.setdefault(wallet, {}).setdefault(asset, []).append({
                "side": side,
                "size": size,
                "price": price,
                "timestamp": ts,
            })

            # Determine resolved price for this specific token
            if asset == winner_token_id:
                resolved_price = 1.0
            elif asset in clob_token_ids:
                resolved_price = 0.0
            else:
                resolved_price = 0.0

            trades.append(HistoricalTrade(
                wallet=wallet,
                market_id=cid,
                asset_id=asset,
                side=side,
                size=size,
                price=price,
                timestamp=ts,
                market_resolved_price=resolved_price,
                market_resolution_date=res_date,
                market_category=category,
            ))

        # Build WalletMarketResult for each wallet in this market
        for wallet, token_trades in wallet_token_trades.items():
            for token_id, t_list in token_trades.items():
                buys = [t for t in t_list if t["side"] == "BUY"]
                sells = [t for t in t_list if t["side"] == "SELL"]
                total_bought = sum(t["size"] for t in buys)
                total_sold = sum(t["size"] for t in sells)
                net_position = total_bought - total_sold

                if net_position <= 0:
                    continue

                # Weighted avg entry price
                buy_cost = sum(t["price"] * t["size"] for t in buys)
                avg_entry = buy_cost / total_bought if total_bought > 0 else 0.0

                is_winner = (token_id == winner_token_id)
                won = is_winner and net_position > 0

                # ROI
                payout = net_position * 1.0 if is_winner else 0.0
                cost = net_position * avg_entry if avg_entry > 0 else 0.0
                roi = (payout - cost) / cost if cost > 0 else 0.0

                key = f"{wallet}_{cid}_{token_id}"
                if wallet not in wallet_market_positions:
                    wallet_market_positions[wallet] = {}
                wallet_market_positions[wallet][key] = {
                    "market_id": cid,
                    "won": won,
                    "roi": roi,
                    "held_to_expiration": net_position > 0,
                    "total_bought": total_bought,
                    "total_sold": total_sold,
                    "resolution_date": res_date,
                }

    # Convert wallet positions to WalletMarketResult list
    wallet_results: list[WalletMarketResult] = []
    for wallet, positions in wallet_market_positions.items():
        for key, pos in positions.items():
            wallet_results.append(WalletMarketResult(
                wallet=wallet,
                market_id=pos["market_id"],
                won=pos["won"],
                roi=pos["roi"],
                held_to_expiration=pos["held_to_expiration"],
                total_bought=pos["total_bought"],
                total_sold=pos["total_sold"],
                resolution_date=pos["resolution_date"],
            ))

    print(f"  Processed: {len(trades)} trades, {len(wallet_results)} wallet results, "
          f"{len(wallet_aliases)} wallet aliases")

    return trades, wallet_results, wallet_aliases


# ── Chart generation ─────────────────────────────────────────────────


def plot_cumulative_pnl(result: BacktestResult, path: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[3, 1], sharex=True)
    cum = result.cumulative_pnl
    if not cum:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    x = range(len(cum))
    ax1.plot(x, cum, color="#2196F3", linewidth=1.5)
    ax1.fill_between(x, cum, alpha=0.1, color="#2196F3")
    ax1.set_ylabel("Cumulative PnL ($)")
    ax1.set_title("Cumulative PnL with Drawdown Overlay")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.grid(True, alpha=0.3)

    # Drawdown
    peak = cum[0]
    dd = []
    for v in cum:
        peak = max(peak, v)
        dd.append(v - peak)
    ax2.fill_between(x, dd, color="#F44336", alpha=0.4)
    ax2.set_ylabel("Drawdown ($)")
    ax2.set_xlabel("Trade #")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_returns(result: BacktestResult, path: str):
    monthly = result.monthly_returns()
    if not monthly:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No monthly data", ha="center", va="center")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    months = list(monthly.keys())
    values = list(monthly.values())
    colors_list = ["#4CAF50" if v >= 0 else "#F44336" for v in values]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(months)), values, color=colors_list, alpha=0.8)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("PnL ($)")
    ax.set_title("Monthly Returns")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_latency_curve(results: dict[int, BacktestResult], path: str):
    if not results:
        return
    lats = sorted(results.keys())
    sharpes = [results[l].sharpe_ratio for l in lats]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lats, sharpes, "o-", color="#9C27B0", linewidth=2, markersize=8)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Mirror Latency (seconds)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Latency Degradation Curve")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(heatmap_data: dict, path: str):
    x_vals = heatmap_data["x_values"]
    y_vals = heatmap_data["y_values"]
    z = heatmap_data["z_matrix"]

    if not z or not z[0]:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(z, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(x_vals, fontsize=9)
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels(y_vals, fontsize=9)
    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Top-N Wallets")
    ax.set_title(f"Parameter Sensitivity: {heatmap_data['metric']}")

    for i in range(len(y_vals)):
        for j in range(len(x_vals)):
            ax.text(j, i, f"{z[i][j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Sharpe Ratio")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_walk_forward(wf_details: dict, path: str):
    pnls = wf_details.get("per_window_pnl", [])
    if not pnls:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(1, len(pnls) + 1)
    colors_list = ["#4CAF50" if p >= 0 else "#F44336" for p in pnls]
    ax.bar(x, pnls, color=colors_list, alpha=0.8)
    ax.set_xlabel("Walk-Forward Window")
    ax.set_ylabel("Out-of-Sample PnL ($)")
    ax.set_title("Walk-Forward Validation: OOS Performance by Window")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_randomized_baseline(details: dict, path: str):
    real_sharpe = details.get("real_sharpe", 0)
    random_mean = details.get("random_mean_sharpe", 0)
    random_std = details.get("random_std_sharpe", 1)

    import numpy as np
    x = np.linspace(random_mean - 3 * max(random_std, 0.1), random_mean + 3 * max(random_std, 0.1), 200)
    if random_std > 0:
        y = (1 / (random_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - random_mean) / random_std) ** 2)
    else:
        y = np.zeros_like(x)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(x, y, alpha=0.3, color="#9E9E9E", label="Random Baseline Distribution")
    ax.plot(x, y, color="#9E9E9E", linewidth=1.5)
    ax.axvline(x=real_sharpe, color="#F44336", linewidth=2, linestyle="--", label=f"Actual Sharpe = {real_sharpe:.2f}")
    ax.set_xlabel("Sharpe Ratio")
    ax.set_ylabel("Density")
    ax.set_title(f"Randomized Baseline Test (p={details.get('p_value', 'N/A'):.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── PDF generation ───────────────────────────────────────────────────


def _std_table_style(header_color="#1565C0", alt_color="#E3F2FD"):
    """Return a reusable TableStyle."""
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(alt_color)]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])


def build_pdf(
    result: BacktestResult,
    bias_results: list[dict],
    validation_results: list,
    sweep_results: dict[str, BacktestResult],
    latency_results: dict[int, BacktestResult],
    wf_details: dict,
    rand_details: dict,
    heatmap_data: dict,
    wallet_scores: list,
    wallet_aliases: dict[str, str],
    output_path: str,
):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("CustomTitle", parent=styles["Title"], fontSize=22, spaceAfter=20)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceAfter=10, spaceBefore=16)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceAfter=8, spaceBefore=12)
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, spaceAfter=6, leading=14)
    mono = ParagraphStyle("Mono", parent=styles["Code"], fontSize=9, spaceAfter=6, leading=12)
    small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=8, spaceAfter=4, leading=10)
    bullet = ParagraphStyle("Bullet", parent=body, leftIndent=20, bulletIndent=8)

    elements = []

    # ────────────────────────────────────────────────────────────────────
    # TITLE PAGE
    # ────────────────────────────────────────────────────────────────────
    elements.append(Paragraph("Polymarket Wallet Mirror Strategy", title_style))
    elements.append(Paragraph("Backtest &amp; Bias Audit Report", styles["Heading2"]))
    elements.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", small))
    elements.append(Spacer(1, 20))

    # ────────────────────────────────────────────────────────────────────
    # 1. STRATEGY OVERVIEW — What This Is
    # ────────────────────────────────────────────────────────────────────
    elements.append(Paragraph("1. Strategy Overview", h1))

    elements.append(Paragraph("<b>What is Polymarket?</b>", body))
    elements.append(Paragraph(
        "Polymarket (polymarket.com) is a decentralized prediction market built on the Polygon blockchain. "
        "Users trade binary outcome contracts — for example, 'Will X happen by date Y?' — where YES tokens "
        "resolve to $1.00 if the event occurs and $0.00 if it does not. Prices between $0 and $1 reflect "
        "the market's implied probability. Trading is done via a Central Limit Order Book (CLOB) with "
        "limit orders, just like a stock exchange.", body))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("<b>What is the 'Wallet Mirror' strategy?</b>", body))
    elements.append(Paragraph(
        "The core idea is simple: some Polymarket wallets are consistently profitable. They buy contracts "
        "at low prices, hold them to expiration, and win more often than they lose. If we can identify "
        "these 'sharp' wallets and copy their trades in real time, we capture their information edge. "
        "This is sometimes called 'copy trading' or 'following the sharps'.", body))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("<b>How does it work end to end?</b>", body))
    elements.append(Paragraph("&bull; <b>Step 1 — Discover:</b> Scan all historically resolved Polymarket markets. "
        "For each market, fetch every trade by wallet address. Reconstruct who held what at expiration and "
        "whether they won or lost.", bullet))
    elements.append(Paragraph("&bull; <b>Step 2 — Score:</b> Score each wallet with a composite formula "
        "(win rate, ROI, consistency, recency, hold ratio). Rank the top wallets.", bullet))
    elements.append(Paragraph("&bull; <b>Step 3 — Monitor:</b> Watch the top-ranked wallets in real time via "
        "WebSocket feed and polling. When a sharp wallet buys a contract, we get a signal.", bullet))
    elements.append(Paragraph("&bull; <b>Step 4 — Validate:</b> Before copying the trade, check the orderbook. "
        "Is there enough liquidity? Is the spread tight? Can we get filled without excessive slippage?", bullet))
    elements.append(Paragraph("&bull; <b>Step 5 — Execute:</b> Place a limit order mirroring the sharp wallet's "
        "position. Cap position size, deduct fees, and log every decision.", bullet))
    elements.append(Paragraph("&bull; <b>Step 6 — Wait:</b> Hold the position until the market resolves. "
        "The contract pays $1.00 if correct, $0.00 if wrong.", bullet))

    elements.append(PageBreak())

    # ────────────────────────────────────────────────────────────────────
    # 2. HOW WE FIND SHARP WALLETS
    # ────────────────────────────────────────────────────────────────────
    elements.append(Paragraph("2. How We Find Sharp Wallets", h1))

    elements.append(Paragraph("<b>Data sources</b>", body))
    elements.append(Paragraph(
        "We pull data from two Polymarket APIs: the Gamma API (gamma-api.polymarket.com) for market "
        "metadata and resolution outcomes, and the CLOB API (clob.polymarket.com) for individual trade "
        "records per wallet. All markets that have resolved (outcome is known) are analyzed.", body))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("<b>Per-wallet analysis</b>", body))
    elements.append(Paragraph(
        "For each resolved market, we reconstruct every wallet's position at expiration: "
        "net contracts held = total bought - total sold. A wallet 'won' if it held a positive "
        "position in the winning token. ROI = (payout - cost) / cost, where payout = position * $1 "
        "if won, $0 if lost; cost = position * volume-weighted average entry price.", body))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("<b>Minimum qualification</b>", body))
    elements.append(Paragraph(
        f"A wallet must have participated in at least <b>20 resolved markets</b> (configurable) to be scored. "
        "This filters out lucky one-timers. Only wallets that held to expiration are considered — "
        "short-term flippers are excluded.", body))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("<b>Composite scoring formula</b>", body))
    elements.append(Paragraph(
        "Each wallet receives a score from 0 to 1, computed as a weighted sum of five components:", body))
    elements.append(Spacer(1, 6))

    score_data = [
        ["Component", "Weight", "What it measures", "Formula"],
        ["Win Rate", "30%", "Prediction accuracy", "wins / total_markets"],
        ["ROI", "25%", "Profitability per market",
         "sigmoid(avg_roi * 5), normalized to [0,1]"],
        ["Consistency", "20%", "Steady vs. volatile returns",
         "1 / (1 + stdev(roi) / |mean(roi)|)"],
        ["Recency", "15%", "Recent performance weighted more",
         "exp(-ln2/90d * age) * outcome, averaged"],
        ["Hold Ratio", "10%", "Conviction: holds to expiration",
         "held_count / total_markets"],
    ]
    st = Table(score_data, colWidths=[1.1 * inch, 0.6 * inch, 1.8 * inch, 3 * inch])
    st.setStyle(_std_table_style("#4CAF50", "#E8F5E9"))
    elements.append(st)

    elements.append(Spacer(1, 8))
    elements.append(Paragraph(
        "<b>Final score</b> = 0.30 * WinRate + 0.25 * NormalizedROI + 0.20 * Consistency "
        "+ 0.15 * RecencyScore + 0.10 * HoldRatio", mono))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "The recency component uses exponential decay with a 90-day half-life: "
        "a market that resolved yesterday has ~2x the weight of one that resolved 90 days ago. "
        "This ensures the ranking adapts as wallets improve or degrade over time.", body))
    elements.append(Paragraph(
        "Wallets are ranked by composite score and the top N (default 50) form the watchlist.", body))

    # Wallet Leaderboard — built from actual executed trades in the backtest
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("<b>Wallets that generated trades in this backtest</b>", body))
    elements.append(Paragraph(
        "These wallets actually had trades executed during the backtest. They qualified as "
        "top-ranked at the point in time when each trade occurred (point-in-time scoring). "
        "Wallets are ranked here by total PnL contribution.", body))
    elements.append(Spacer(1, 6))

    # Aggregate per-wallet stats from executed trades
    _wallet_stats: dict[str, dict] = {}
    for tr in result.trade_records:
        if tr.decision != "execute":
            continue
        w = tr.wallet
        if w not in _wallet_stats:
            _wallet_stats[w] = {"pnl": 0.0, "trades": 0, "wins": 0, "score": tr.wallet_score}
        _wallet_stats[w]["pnl"] += tr.pnl
        _wallet_stats[w]["trades"] += 1
        if tr.pnl > 0:
            _wallet_stats[w]["wins"] += 1
        # Keep the latest score seen
        if tr.wallet_score > 0:
            _wallet_stats[w]["score"] = tr.wallet_score

    _sorted_wallets = sorted(_wallet_stats.items(), key=lambda x: x[1]["pnl"], reverse=True)

    if _sorted_wallets:
        ws_data = [["#", "Wallet", "Alias", "PnL", "Trades", "Win%"]]
        for i, (addr, stats) in enumerate(_sorted_wallets[:20], 1):
            short_addr = f"{addr[:8]}...{addr[-6:]}" if len(addr) > 16 else addr
            alias = wallet_aliases.get(addr, "—")
            wr = f"{stats['wins'] / stats['trades']:.0%}" if stats["trades"] > 0 else "—"
            ws_data.append([
                str(i),
                short_addr,
                alias,
                f"${stats['pnl']:,.2f}",
                str(stats["trades"]),
                wr,
            ])
        wst = Table(ws_data, colWidths=[0.4 * inch, 1.5 * inch, 1.5 * inch, 1.0 * inch, 0.7 * inch, 0.6 * inch])
        wst.setStyle(_std_table_style("#1565C0", "#E3F2FD"))
        elements.append(wst)

        # Full address reference
        elements.append(Spacer(1, 10))
        elements.append(Paragraph("<b>Full wallet addresses (verifiable on polymarket.com/profile/ADDRESS)</b>", body))
        elements.append(Spacer(1, 4))
        for i, (addr, stats) in enumerate(_sorted_wallets[:20], 1):
            alias = wallet_aliases.get(addr, "")
            alias_str = f" ({alias})" if alias else ""
            elements.append(Paragraph(
                f"&nbsp;&nbsp;#{i}: <font face='Courier'>{addr}</font>{alias_str} "
                f"— ${stats['pnl']:+,.2f} across {stats['trades']} trades", small))

    elements.append(PageBreak())

    # ────────────────────────────────────────────────────────────────────
    # 3. WHAT WE TRADE AND HOW
    # ────────────────────────────────────────────────────────────────────
    elements.append(Paragraph("3. What We Trade and How We Execute", h1))

    elements.append(Paragraph("<b>What we trade</b>", body))
    elements.append(Paragraph(
        "We trade binary outcome tokens on Polymarket's CLOB. Each market (e.g. 'Will Bitcoin exceed "
        "$100k by June 2025?') has YES and NO tokens. We only mirror BUY trades — when a sharp wallet "
        "buys YES or NO tokens, we buy the same token. We hold until expiration and collect $1.00 per "
        "token if correct, $0.00 if wrong. Markets span categories: politics, sports, crypto, finance, "
        "weather, entertainment, and more.", body))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("<b>Real-time monitoring</b>", body))
    elements.append(Paragraph(
        "Primary: WebSocket stream from Polymarket's CLOB (wss://ws-subscriptions-clob.polymarket.com). "
        "Fallback: poll the CLOB API every 5 seconds for recent trades by watched wallets. "
        "When a sharp wallet places a BUY trade, we immediately receive a signal.", body))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("<b>Pre-trade validation filters</b>", body))
    elements.append(Paragraph(
        "Every signal passes through five filters. The first failure skips the trade:", body))
    elements.append(Spacer(1, 4))

    filter_data = [
        ["Filter", "Default Threshold", "Why"],
        ["Stale signal", "300 seconds", "If too much time has passed, the market has moved"],
        ["Min liquidity", "$10,000", "Thin markets = can't fill without huge slippage"],
        ["Max spread", "3%", "Wide spread = we'd trade against ourselves"],
        ["Max slippage", "2%", "Estimated fill price too far from midpoint"],
        ["Max position", "$1,000", "Cap exposure per trade"],
    ]
    ft = Table(filter_data, colWidths=[1.2 * inch, 1.3 * inch, 4 * inch])
    ft.setStyle(_std_table_style("#FF9800", "#FFF3E0"))
    elements.append(ft)

    elements.append(Spacer(1, 8))
    elements.append(Paragraph("<b>Position sizing</b>", body))
    elements.append(Paragraph(
        "Order size = min(wallet's trade size, max_position_usd / price). "
        "We never take a position larger than $1,000 (configurable) to limit per-trade risk.", body))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("<b>Order placement and fill simulation</b>", body))
    elements.append(Paragraph(
        "We walk the orderbook to compute a realistic fill price. For a BUY, we consume ask levels "
        "from best (lowest) to worst until our order is fully filled. The fill price is the "
        "volume-weighted average across all consumed levels. This is NOT a midpoint order — it "
        "accounts for the actual depth and shape of the order book. If there isn't enough liquidity "
        "to fill the order, the trade is skipped. In live trading, we place a limit order and wait "
        "up to 60 seconds for a fill; if not filled, we cancel.", body))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("<b>Fees</b>", body))
    elements.append(Paragraph(
        "Polymarket charges <b>zero trading fees</b>. There are no maker or taker fees on the CLOB. "
        "The only costs are slippage (the difference between the expected fill price and the actual "
        "fill price) and Polygon network gas fees (typically &lt; $0.01 per transaction, negligible).", body))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("<b>PnL calculation</b>", body))
    elements.append(Paragraph(
        "PnL per trade = (market_resolved_price - fill_price) * order_size. "
        "For a BUY at $0.55 with a winning resolution at $1.00 and size 100: "
        "PnL = ($1.00 - $0.55) * 100 = $45.00. "
        "For a losing resolution at $0.00: PnL = ($0.00 - $0.55) * 100 = -$55.00. "
        "The only real cost is slippage — getting filled at a worse price than the signal price.", body))

    elements.append(PageBreak())

    # ────────────────────────────────────────────────────────────────────
    # 4. BACKTEST METHODOLOGY
    # ────────────────────────────────────────────────────────────────────
    elements.append(Paragraph("4. Backtest Methodology", h1))

    elements.append(Paragraph("<b>Point-in-time scoring (no look-ahead bias)</b>", body))
    elements.append(Paragraph(
        "The most important design decision in this backtest is <b>point-in-time (PIT) wallet scoring</b>. "
        "At each historical trade timestamp T, wallet scores are computed using ONLY markets that "
        "resolved BEFORE T. This simulates what information would have been available in real time. "
        "A binary search (bisect) efficiently finds the cutoff in the sorted resolution date list.", body))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("<b>Latency simulation</b>", body))
    elements.append(Paragraph(
        f"This backtest simulates a <b>{result.latency_s}-second delay</b> between when the sharp wallet "
        "trades and when our mirror order would execute. During this delay, the orderbook may change. "
        "The realized slippage model increases slippage linearly with latency: "
        "realized_slippage = estimated_slippage * (1 + latency_s / 600).", body))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("<b>Orderbook simulation</b>", body))
    elements.append(Paragraph(
        "Historical orderbook snapshots are used when available. When no snapshot exists for a given "
        "asset/time, a synthetic orderbook is generated with a 2% spread and 500 contracts of depth "
        "on each side, centered on the trade price.", body))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("<b>Data used in this report</b>", body))
    n_unique_wallets = len(set(t.wallet for t in result.trade_records))
    n_unique_markets = len(set(t.market_id for t in result.trade_records))
    elements.append(Paragraph(
        f"This report uses <b>real Polymarket historical data</b> fetched from the Gamma API "
        f"(market metadata and resolutions) and the Data API (trade records). The dataset includes "
        f"~{n_unique_wallets} unique wallets across ~{n_unique_markets} resolved markets. "
        f"All wallet addresses, trade prices, outcomes, and timestamps are real on-chain data "
        f"verifiable on polymarket.com.", body))

    elements.append(PageBreak())

    # ────────────────────────────────────────────────────────────────────
    # 5. CONFIGURATION REFERENCE
    # ────────────────────────────────────────────────────────────────────
    elements.append(Paragraph("5. Configuration Reference", h1))

    elements.append(Paragraph(
        "Below are the exact parameter values used in this backtest. Tuning these "
        "directly affects performance — the parameter sweep in section 10 tests sensitivity.", body))
    elements.append(Spacer(1, 6))

    cfg_data = [
        ["Parameter", "Value", "Description"],
        ["Scoring: min_resolved_markets", "5", "Min markets for wallet qualification"],
        ["Scoring: recency_half_life_days", "90", "Decay half-life for recency weighting"],
        ["Scoring: win_rate_weight", "0.30", "Weight of win rate in composite score"],
        ["Scoring: roi_weight", "0.25", "Weight of normalized ROI"],
        ["Scoring: consistency_weight", "0.20", "Weight of consistency metric"],
        ["Scoring: recency_weight", "0.15", "Weight of recency score"],
        ["Scoring: hold_ratio_weight", "0.10", "Weight of hold-to-expiration ratio"],
        ["Execution: max_slippage_pct", "5%", "Max allowed slippage per trade"],
        ["Execution: max_spread_pct", "10%", "Max bid-ask spread to trade"],
        ["Execution: min_liquidity_usd", "$0", "Min orderbook depth required"],
        ["Execution: max_position_usd", "$5,000", "Max position size per trade"],
        ["Execution: stale_signal_timeout", "600s", "Signal expiry time"],
        ["Execution: fees", "$0", "Polymarket has zero trading fees"],
        ["Engine: latency_s", "60s", "Simulated mirror latency"],
        ["Engine: top_n", "20", "Number of top wallets to mirror"],
    ]
    ct = Table(cfg_data, colWidths=[2.3 * inch, 0.8 * inch, 3.4 * inch])
    ct.setStyle(_std_table_style("#607D8B", "#ECEFF1"))
    elements.append(ct)

    elements.append(PageBreak())

    # ────────────────────────────────────────────────────────────────────
    # 6. EXECUTIVE SUMMARY & KEY METRICS
    # ────────────────────────────────────────────────────────────────────
    elements.append(Paragraph("6. Backtest Results", h1))
    verdict = "promising" if result.sharpe_ratio > 0.5 else "inconclusive" if result.sharpe_ratio > 0 else "negative"
    summary_text = (
        f"The mirror strategy was backtested over <b>real Polymarket historical data</b> "
        f"with <b>{result.trades_executed}</b> executed trades and <b>{result.trades_skipped}</b> skipped trades. "
        f"The overall verdict is <b>{verdict}</b> with a Sharpe ratio of <b>{result.sharpe_ratio:.2f}</b>."
    )
    elements.append(Paragraph(summary_text, body))

    # Key Metrics Table
    elements.append(Spacer(1, 10))
    metrics_data = [
        ["Metric", "Value"],
        ["Total PnL", f"${result.total_pnl:,.2f}"],
        ["Trades Executed", str(result.trades_executed)],
        ["Trades Skipped", str(result.trades_skipped)],
        ["Win Rate", f"{result.win_rate:.1%}"],
        ["Avg Profit/Trade", f"${result.avg_profit:,.2f}"],
        ["Median Profit/Trade", f"${result.median_profit:,.2f}"],
        ["Sharpe Ratio", f"{result.sharpe_ratio:.2f}"],
        ["Sortino Ratio", f"{result.sortino_ratio:.2f}"],
        ["Max Drawdown", f"${result.max_drawdown:,.2f}"],
        ["Max DD Duration", f"{result.max_drawdown_duration} trades"],
        ["Avg Est. Slippage", f"{result.avg_estimated_slippage:.4f}"],
        ["Trading Fees", "$0 (Polymarket has zero fees)"],
    ]
    t = Table(metrics_data, colWidths=[2.5 * inch, 2.5 * inch])
    t.setStyle(_std_table_style())
    elements.append(t)

    # Skip Breakdown
    elements.append(Spacer(1, 10))
    skip = result.skip_reasons
    if skip:
        elements.append(Paragraph("Skip Reason Breakdown:", h2))
        elements.append(Paragraph(
            "Trades are skipped when pre-trade validation filters fail. Here is the breakdown:", body))
        skip_data = [["Reason", "Count", "Meaning"]]
        skip_meanings = {
            "not_qualified": "Wallet did not have enough history to rank in top-N at that point in time",
            "stale": "Signal was too old by the time we could act",
            "liquidity": "Orderbook depth below minimum threshold",
            "spread": "Bid-ask spread too wide",
            "slippage": "Estimated fill price deviated too much from midpoint",
            "no_fill": "Could not estimate a fill from the orderbook",
        }
        for k, v in sorted(skip.items(), key=lambda x: -x[1]):
            skip_data.append([k, str(v), skip_meanings.get(k, "")])
        st2 = Table(skip_data, colWidths=[1.5 * inch, 0.8 * inch, 4.2 * inch])
        st2.setStyle(_std_table_style("#FF9800", "#FFF3E0"))
        elements.append(st2)

    # PnL Chart
    elements.append(PageBreak())
    elements.append(Paragraph("7. Performance Charts", h1))

    pnl_chart = os.path.join(OUTPUT_DIR, "cumulative_pnl.png")
    plot_cumulative_pnl(result, pnl_chart)
    if os.path.exists(pnl_chart):
        elements.append(Image(pnl_chart, width=6.5 * inch, height=3.9 * inch))
        elements.append(Spacer(1, 10))

    monthly_chart = os.path.join(OUTPUT_DIR, "monthly_returns.png")
    plot_monthly_returns(result, monthly_chart)
    if os.path.exists(monthly_chart):
        elements.append(Image(monthly_chart, width=6.5 * inch, height=2.6 * inch))

    # Monthly Returns Table
    elements.append(Spacer(1, 10))
    monthly = result.monthly_returns()
    if monthly:
        m_data = [["Month", "PnL", "Trades"]]
        # Count trades per month
        trades_per_month = {}
        for tr in result.trade_records:
            if tr.decision == "execute":
                key = tr.timestamp.strftime("%Y-%m")
                trades_per_month[key] = trades_per_month.get(key, 0) + 1
        for month, pnl in list(monthly.items())[:18]:
            m_data.append([month, f"${pnl:,.2f}", str(trades_per_month.get(month, 0))])
        mt = Table(m_data, colWidths=[1.5 * inch, 2 * inch, 1.5 * inch])
        mt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565C0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#E3F2FD")]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        elements.append(Paragraph("Monthly Returns:", h2))
        elements.append(mt)

    # Wallet Concentration
    elements.append(PageBreak())
    elements.append(Paragraph("8. Wallet Concentration Analysis", h1))
    by_wallet = result.pnl_by_wallet()
    if by_wallet:
        total_pnl = result.total_pnl or 1.0
        wc_data = [["Rank", "Wallet", "PnL", "% of Total"]]
        for i, (wallet, pnl) in enumerate(list(by_wallet.items())[:15], 1):
            pct = (pnl / total_pnl * 100) if total_pnl != 0 else 0
            wc_data.append([str(i), wallet, f"${pnl:,.2f}", f"{pct:.1f}%"])
        wt = Table(wc_data, colWidths=[0.6 * inch, 1.5 * inch, 1.8 * inch, 1.2 * inch])
        wt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565C0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#E3F2FD")]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        elements.append(wt)
        elements.append(Spacer(1, 10))
        elements.append(Paragraph(f"Top-1 concentration: {result.top_wallet_concentration(1):.1%}", body))
        elements.append(Paragraph(f"Top-3 concentration: {result.top_wallet_concentration(3):.1%}", body))
        elements.append(Paragraph(f"Top-5 concentration: {result.top_wallet_concentration(5):.1%}", body))

    # Category Breakdown
    by_cat = result.pnl_by_category()
    if by_cat:
        elements.append(Spacer(1, 10))
        elements.append(Paragraph("Performance by Market Category:", h2))
        cat_data = [["Category", "PnL"]]
        for cat, pnl in sorted(by_cat.items(), key=lambda x: -x[1]):
            cat_data.append([cat, f"${pnl:,.2f}"])
        ct = Table(cat_data, colWidths=[2.5 * inch, 2.5 * inch])
        ct.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4CAF50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        elements.append(ct)

    # Bias Tests
    elements.append(PageBreak())
    elements.append(Paragraph("9. Bias Audit Results", h1))
    elements.append(Paragraph(
        "Backtests are prone to biases that inflate apparent returns. Each test below checks for a "
        "specific form of statistical error. A FAIL means the backtest may be unreliable in that dimension.",
        body,
    ))
    elements.append(Spacer(1, 10))

    # Descriptions of each bias test
    bias_descriptions = {
        "wallet_score_leakage": "Checks if wallet scores used future information (markets that hadn't "
            "resolved yet at trade time). If scores include future outcomes, the ranking is artificially good.",
        "price_leakage": "Checks if the fill price suspiciously matches the resolution price. "
            "If fill == resolved price for many trades, the backtest may be using future prices.",
        "resolution_data_leakage": "Checks if the decision to execute vs. skip is correlated with "
            "the eventual market outcome. If we only trade winners, something is leaking.",
        "category_concentration": "Checks if profits come from a single market category (e.g. only politics). "
            "Over-concentration means the edge may not generalize.",
        "wallet_concentration": "Checks if profits come from just 1-2 wallets. "
            "High concentration means the strategy depends on specific individuals, not a systematic edge.",
        "time_window_stability": "Runs the backtest on sub-periods (quarters). If only 1 quarter is profitable "
            "and the rest are flat/negative, the overall result may be driven by a fluke.",
        "parameter_sensitivity": "Runs a parameter sweep and checks what fraction of configs are profitable. "
            "If only a narrow parameter set works, the strategy may be overfit to those exact settings.",
        "latency_sensitivity": "Tests performance at different mirror latencies (30s to 600s). "
            "If Sharpe collapses at realistic latencies, the edge doesn't survive execution delay.",
        "slippage_sensitivity": "Tests performance under increasing slippage/cost assumptions. "
            "Checks whether the edge is robust to realistic transaction costs.",
        "fee_breakeven": "Finds the fee level at which the strategy breaks even. "
            "If breakeven fee is close to actual fees, the margin of safety is thin.",
    }

    bias_data = [["Test", "Status", "Key Finding"]]
    for br in bias_results:
        status = "PASS" if br.get("passed") else "FAIL"
        name = br.get("test", "unknown")
        detail = ""
        for k in br:
            if k in ("test", "passed"):
                continue
            v = br[k]
            if isinstance(v, float):
                detail = f"{k}={v:.4f}"
                break
            elif isinstance(v, int):
                detail = f"{k}={v}"
                break
            elif isinstance(v, list) and len(v) <= 3:
                detail = f"{k}={v}"
                break
        bias_data.append([name, status, detail[:60]])

    bt = Table(bias_data, colWidths=[2.5 * inch, 0.8 * inch, 3 * inch])
    bt.setStyle(_std_table_style("#9C27B0", "#F3E5F5"))
    elements.append(bt)

    # Detailed bias findings with descriptions
    elements.append(Spacer(1, 10))
    for br in bias_results:
        name = br.get("test", "unknown")
        status = "PASS" if br.get("passed") else "FAIL"
        status_color = "#4CAF50" if br.get("passed") else "#F44336"
        elements.append(Paragraph(
            f'<b>{name}</b> <font color="{status_color}">[{status}]</font>', body))
        desc = bias_descriptions.get(name, "")
        if desc:
            elements.append(Paragraph(f"<i>{desc}</i>", small))
        for k, v in br.items():
            if k in ("test", "passed"):
                continue
            if isinstance(v, (dict, list)) and len(str(v)) > 100:
                continue
            elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{k}: {v}", small))

    # Statistical Validation
    elements.append(PageBreak())
    elements.append(Paragraph("10. Statistical Validation", h1))
    elements.append(Paragraph(
        "These tests apply rigorous statistical methods to determine whether the backtest results "
        "are genuine or could be explained by chance, data mining, or overfitting.", body))
    elements.append(Spacer(1, 6))

    val_descriptions = {
        "bootstrap_ci": "Resamples the trade PnL series 5,000 times to compute a 95% confidence "
            "interval for the Sharpe ratio. PASS = lower bound > 0 (Sharpe is positive with 95% confidence).",
        "deflated_sharpe": "Adjusts the Sharpe ratio for the number of strategy variations tested "
            "(Bailey &amp; Lopez de Prado, 2014). More variations tested = higher bar. PASS = p &lt; 0.05.",
        "minimum_track_record": "Calculates the minimum number of trades needed for the observed Sharpe "
            "to be statistically significant. PASS = actual trades >= minimum needed.",
        "walk_forward": "Divides the data into rolling 4-month train / 2-month test windows. "
            "Trains the strategy on each window and tests on the next. PASS = average OOS Sharpe > 0.",
        "in_out_sample": "Splits data 60/40 by time. Trains on the first 60% and tests on the last 40%. "
            "PASS = out-of-sample Sharpe > 0. A large gap between IS and OOS suggests overfitting.",
        "randomized_baseline": "Shuffles wallet labels 100 times and re-runs the backtest each time. "
            "If the real strategy beats 95% of random permutations, the edge is likely real (p &lt; 0.05).",
        "holm_bonferroni": "Applies the Holm-Bonferroni correction to all p-values from the tests above, "
            "controlling the family-wise error rate when running multiple hypothesis tests.",
    }

    for vr in validation_results:
        status = "PASS" if vr.passed else "FAIL"
        color = "#4CAF50" if vr.passed else "#F44336"
        elements.append(Paragraph(
            f'<font color="{color}"><b>[{status}]</b></font> {vr.method}',
            h2,
        ))
        desc = val_descriptions.get(vr.method, "")
        if desc:
            elements.append(Paragraph(f"<i>{desc}</i>", small))
        for k, v in vr.details.items():
            if isinstance(v, float):
                elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;<b>{k}</b>: {v:.4f}", body))
            elif isinstance(v, (list, tuple)) and len(v) <= 10:
                if all(isinstance(x, float) for x in v):
                    formatted = [f"{x:.3f}" for x in v]
                    elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;<b>{k}</b>: {formatted}", body))
                else:
                    elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;<b>{k}</b>: {v}", body))
            elif isinstance(v, dict) and len(v) <= 10:
                elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;<b>{k}</b>: {v}", body))
            else:
                elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;<b>{k}</b>: {v}", body))
        elements.append(Spacer(1, 6))

    # Charts: Latency, Heatmap, Walk-Forward, Randomized Baseline
    elements.append(PageBreak())
    elements.append(Paragraph("11. Sensitivity Analysis Charts", h1))
    elements.append(Paragraph(
        "These charts visualize how the strategy's edge degrades under different conditions. "
        "A robust strategy should maintain positive performance across a range of parameters.", body))

    latency_chart = os.path.join(OUTPUT_DIR, "latency_curve.png")
    plot_latency_curve(latency_results, latency_chart)
    if os.path.exists(latency_chart):
        elements.append(Paragraph("Latency Degradation:", h2))
        elements.append(Paragraph(
            "Shows Sharpe ratio vs. mirror latency (30s to 600s). A steep drop means the edge "
            "is time-sensitive — you need fast infrastructure to capture it.", small))
        elements.append(Image(latency_chart, width=6 * inch, height=3 * inch))
        elements.append(Spacer(1, 10))

    heatmap_chart = os.path.join(OUTPUT_DIR, "param_heatmap.png")
    if heatmap_data and heatmap_data.get("z_matrix"):
        plot_heatmap(heatmap_data, heatmap_chart)
        if os.path.exists(heatmap_chart):
            elements.append(Paragraph("Parameter Sensitivity Heatmap:", h2))
            elements.append(Paragraph(
                "Shows Sharpe ratio across combinations of latency (x-axis) and number of wallets "
                "mirrored (y-axis). Green = good, red = bad. A consistent green region means the "
                "strategy is robust to parameter choices.", small))
            elements.append(Image(heatmap_chart, width=6 * inch, height=3.75 * inch))

    elements.append(PageBreak())
    wf_chart = os.path.join(OUTPUT_DIR, "walk_forward.png")
    if wf_details.get("per_window_pnl"):
        plot_walk_forward(wf_details, wf_chart)
        if os.path.exists(wf_chart):
            elements.append(Paragraph("Walk-Forward Validation:", h2))
            elements.append(Paragraph(
                "Each bar is the out-of-sample PnL for one rolling test window. Green = profitable, "
                "red = losing. Consistently green bars mean the strategy works across different "
                "time periods, not just one lucky stretch.", small))
            elements.append(Image(wf_chart, width=6 * inch, height=3 * inch))
            elements.append(Spacer(1, 10))

    rand_chart = os.path.join(OUTPUT_DIR, "randomized_baseline.png")
    if rand_details:
        plot_randomized_baseline(rand_details, rand_chart)
        if os.path.exists(rand_chart):
            elements.append(Paragraph("Randomized Baseline Distribution:", h2))
            elements.append(Paragraph(
                "The gray curve shows the distribution of Sharpe ratios when wallet labels are "
                "randomly shuffled (destroying any real edge). The red dashed line is our actual "
                "Sharpe. If our line is far to the right of the distribution, the edge is real — "
                "not explainable by chance.", small))
            elements.append(Image(rand_chart, width=6 * inch, height=3 * inch))

    # Sweep Results Table
    if sweep_results:
        elements.append(PageBreak())
        elements.append(Paragraph("12. Parameter Sweep Results", h1))
        elements.append(Paragraph(
            f"Tested {len(sweep_results)} parameter combinations. "
            f"Top 20 shown below, sorted by Sharpe ratio.",
            body,
        ))
        elements.append(Spacer(1, 10))

        sw_data = [["Config", "Sharpe", "PnL", "Trades", "Win Rate"]]
        sorted_sweep = sorted(sweep_results.items(), key=lambda x: x[1].sharpe_ratio, reverse=True)
        for label, sr in sorted_sweep[:20]:
            # Shorten label
            short = label.replace("_", " ").replace("=", ":")
            if len(short) > 40:
                short = short[:37] + "..."
            sw_data.append([
                short,
                f"{sr.sharpe_ratio:.2f}",
                f"${sr.total_pnl:,.0f}",
                str(sr.trades_executed),
                f"{sr.win_rate:.0%}",
            ])
        swt = Table(sw_data, colWidths=[2.8 * inch, 0.8 * inch, 1.2 * inch, 0.8 * inch, 0.9 * inch])
        swt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565C0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#E3F2FD")]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        elements.append(swt)

    # Assessment
    elements.append(PageBreak())
    elements.append(Paragraph("13. Honest Assessment", h1))
    elements.append(Paragraph(
        "This section deliberately looks for reasons the strategy might NOT work. "
        "Every backtest looks good on paper — the goal here is to identify risks before real capital is at stake.",
        body,
    ))
    elements.append(Spacer(1, 6))

    issues = []
    if result.trades_executed < 30:
        issues.append("Insufficient trade count for statistical significance — need 50+ trades minimum.")
    if result.sharpe_ratio < 0.5:
        issues.append(f"Marginal Sharpe ratio ({result.sharpe_ratio:.2f}) — edge may not survive transaction costs at scale.")
    if result.max_drawdown > abs(result.total_pnl) * 0.5 and result.total_pnl > 0:
        issues.append(f"Drawdown (${result.max_drawdown:,.2f}) is large relative to total PnL (${result.total_pnl:,.2f}). "
                       "You'd need to stomach significant losses before seeing recovery.")
    if result.top_wallet_concentration(1) > 0.5:
        issues.append(f"Top-1 wallet concentration is {result.top_wallet_concentration(1):.0%} — "
                       "returns depend on a single trader. If they stop trading or change strategy, "
                       "your edge disappears.")
    failed_bias = [b["test"] for b in bias_results if not b.get("passed")]
    if failed_bias:
        issues.append(f"Failed bias tests: {', '.join(failed_bias)}. "
                       "These failures suggest the backtest results may be inflated by statistical artifacts.")
    failed_val = [v.method for v in validation_results if not v.passed]
    if failed_val:
        issues.append(f"Failed statistical validation: {', '.join(failed_val)}. "
                       "These failures indicate the observed edge may not be statistically robust.")

    if not issues:
        elements.append(Paragraph(
            "The strategy shows robust results across all tests. Bias audit passed, "
            "statistical validation is positive, and performance is consistent across time windows. "
            "This report is based on real Polymarket historical data. "
            "Recommended next step: paper trade for 2-4 weeks to validate execution assumptions.",
            body,
        ))
    else:
        elements.append(Paragraph("<b>Concerns identified:</b>", body))
        for issue in issues:
            elements.append(Paragraph(f"&bull; {issue}", body))

    elements.append(Spacer(1, 14))
    elements.append(Paragraph("<b>Structural risks (regardless of backtest results):</b>", body))
    elements.append(Paragraph(
        "&bull; <b>Alpha decay:</b> If many participants adopt this strategy, sharp wallet trades "
        "will move the market before mirrors can execute, compressing the edge to zero.", body))
    elements.append(Paragraph(
        "&bull; <b>Wallet rotation:</b> Profitable wallets may create new addresses, change strategy, "
        "or stop trading. The scoring model uses recency weighting to partially address this, but "
        "there is an inherent lag.", body))
    elements.append(Paragraph(
        "&bull; <b>Execution risk:</b> Real execution involves network latency, gas fees (Polygon), "
        "partial fills, and API rate limits. The backtest approximates these but cannot fully replicate them.", body))
    elements.append(Paragraph(
        "&bull; <b>Regulatory risk:</b> Prediction market regulations vary by jurisdiction and are evolving. "
        "Automated trading may face additional scrutiny.", body))
    elements.append(Paragraph(
        "&bull; <b>Historical data limitations:</b> Real historical data may not perfectly reflect "
        "future conditions. Liquidity profiles, participant behavior, and market structure can shift. "
        "Past performance is not indicative of future results.", body))

    elements.append(Spacer(1, 14))
    elements.append(Paragraph("<b>Recommended next steps:</b>", body))
    elements.append(Paragraph(
        "&bull; Re-run periodically with fresh data to track performance stability over time.", body))
    elements.append(Paragraph(
        "&bull; Paper trade for 2-4 weeks to validate execution assumptions before deploying capital.", body))
    elements.append(Paragraph(
        "&bull; Start with small position sizes ($100-200/trade) and scale up only after live validation.", body))
    elements.append(Paragraph(
        "&bull; Monitor wallet ranking stability — if top wallets churn rapidly, the signal is unreliable.", body))

    elements.append(Spacer(1, 20))
    elements.append(Paragraph(
        "<i>This report was generated programmatically. All bias tests and statistical "
        "validation procedures are implemented in src/polymir/backtest/ and have "
        "corresponding unit tests. Source code: src/polymir/</i>",
        small,
    ))

    doc.build(elements)
    print(f"PDF saved to: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────


async def main():
    print("Fetching real Polymarket data...")
    trades, wallet_results, wallet_aliases = fetch_real_data(max_markets=200)
    print(f"  {len(trades)} trades, {len(wallet_results)} wallet results")

    config = AppConfig(
        scoring=ScoringConfig(min_resolved_markets=5),
        execution=ExecutionConfig(
            max_position_usd=5000,
            stale_signal_timeout_s=600,
        ),
        top_wallets=20,
    )

    # Compute wallet scores for the leaderboard
    print("Computing wallet scores...")
    from collections import defaultdict
    wr_by_wallet: dict[str, list[WalletMarketResult]] = defaultdict(list)
    for wr in wallet_results:
        wr_by_wallet[wr.wallet].append(wr)
    wallet_scores = []
    for wallet, results_list in wr_by_wallet.items():
        score = compute_wallet_score(wallet, results_list, config.scoring)
        if score is not None:
            wallet_scores.append(score)
    wallet_scores.sort(key=lambda s: s.composite_score, reverse=True)
    print(f"  {len(wallet_scores)} wallets qualified")

    # Run primary backtest
    print("Running primary backtest...")
    engine = BacktestEngine(config, latency_s=60, top_n=20)
    result = await engine.run(
        trades=trades,
        wallet_results=wallet_results,
    )
    print(result.summary())
    print()

    # Run bias tests
    print("Running bias tests...")
    bias_results = []

    bias_results.append(check_wallet_score_leakage(wallet_results, config.scoring))
    bias_results.append(check_price_leakage(result.trade_records))
    bias_results.append(check_resolution_data_leakage(result))
    bias_results.append(check_category_concentration(result))
    bias_results.append(check_wallet_concentration(result))

    # Time window stability: split into quarters
    print("  Time window stability...")
    sorted_trades = sorted(trades, key=lambda t: t.timestamp)
    if sorted_trades:
        quarter_len = len(sorted_trades) // 4
        quarters = {}
        for q in range(4):
            q_trades = sorted_trades[q * quarter_len:(q + 1) * quarter_len]
            if q_trades:
                q_engine = BacktestEngine(config, latency_s=60, top_n=20)
                q_result = await q_engine.run(trades=q_trades, wallet_results=wallet_results)
                quarters[f"Q{q + 1}"] = q_result
        bias_results.append(check_time_window_stability(quarters))

    # Parameter sensitivity
    print("  Parameter sensitivity...")
    sweep_config = SweepConfig(
        latencies=[30, 60, 120, 300],
        top_ns=[10, 20, 50],
    )
    sweep_results = await run_sweep(
        trades=trades,
        wallet_results=wallet_results,
        config=config,
        sweep_config=sweep_config,
    )
    bias_results.append(check_parameter_sensitivity(sweep_results))

    # Latency sensitivity
    print("  Latency sensitivity...")
    latency_results = {}
    for lat in [30, 60, 120, 300, 600]:
        e = BacktestEngine(config, latency_s=lat, top_n=20)
        latency_results[lat] = await e.run(trades=trades, wallet_results=wallet_results)
    bias_results.append(check_latency_sensitivity(latency_results))

    # Fee breakeven
    print("  Fee breakeven...")
    fee_results = {}
    # Fee breakeven: hypothetical test — Polymarket has no fees, but
    # this tests robustness if a fee were ever introduced or if trading
    # on a different venue with fees.
    for fee in [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]:
        fee_cfg = AppConfig(
            scoring=config.scoring,
            execution=ExecutionConfig(
                max_position_usd=5000,
                stale_signal_timeout_s=600,
                fee_rate=fee,
            ),
            top_wallets=20,
        )
        e = BacktestEngine(fee_cfg, latency_s=60, top_n=20)
        fee_results[fee] = await e.run(trades=trades, wallet_results=wallet_results)
    bias_results.append(check_fee_breakeven(fee_results))

    for br in bias_results:
        status = "PASS" if br.get("passed") else "FAIL"
        print(f"  [{status}] {br.get('test')}")

    # Statistical validation
    print("\nRunning statistical validation...")
    validation_results = []

    print("  Bootstrap CI...")
    vr = bootstrap_confidence_intervals(result, n_bootstrap=5000, seed=42)
    validation_results.append(vr)
    print(f"    [{('PASS' if vr.passed else 'FAIL')}] Sharpe CI: {vr.details.get('sharpe_ci_95')}")

    print("  Deflated Sharpe...")
    vr = deflated_sharpe_ratio(
        observed_sharpe=result.sharpe_ratio,
        n_trades=result.trades_executed,
        n_strategies_tested=len(sweep_results),
    )
    validation_results.append(vr)
    print(f"    [{('PASS' if vr.passed else 'FAIL')}] p-value: {vr.details.get('p_value', 'N/A'):.4f}")

    print("  Min track record length...")
    vr = minimum_track_record_length(
        observed_sharpe=result.sharpe_ratio,
        n_trades=result.trades_executed,
    )
    validation_results.append(vr)
    print(f"    [{('PASS' if vr.passed else 'FAIL')}] min={vr.details.get('min_trades_needed')}, actual={vr.details.get('actual_trades')}")

    print("  Walk-forward validation...")
    wf = await walk_forward(
        trades=trades,
        wallet_results=wallet_results,
        config=config,
        train_months=4,
        test_months=2,
        latency_s=60,
        top_n=20,
    )
    validation_results.append(wf)
    print(f"    [{('PASS' if wf.passed else 'FAIL')}] OOS Sharpe: {wf.details.get('oos_sharpe', 0):.2f}")

    print("  In/Out sample split...")
    ios = await in_out_sample_split(
        trades=trades,
        wallet_results=wallet_results,
        config=config,
        latency_s=60,
        top_n=20,
    )
    validation_results.append(ios)
    print(f"    [{('PASS' if ios.passed else 'FAIL')}] IS Sharpe: {ios.details.get('in_sample_sharpe', 0):.2f}, OOS: {ios.details.get('out_of_sample_sharpe', 0):.2f}")

    print("  Randomized baseline (100 iterations)...")
    rb = await randomized_baseline(
        trades=trades,
        wallet_results=wallet_results,
        config=config,
        iterations=100,
        latency_s=60,
        top_n=20,
        seed=42,
    )
    validation_results.append(rb)
    print(f"    [{('PASS' if rb.passed else 'FAIL')}] p-value: {rb.details.get('p_value', 'N/A'):.3f}")

    print("  Holm-Bonferroni correction...")
    p_values = {}
    for vr_item in validation_results:
        if "p_value" in vr_item.details:
            p_values[vr_item.method] = vr_item.details["p_value"]
    if p_values:
        hb = holm_bonferroni_correction(p_values)
        validation_results.append(hb)
        print(f"    [{('PASS' if hb.passed else 'FAIL')}]")

    # Heatmap data
    heatmap_data = build_heatmap_data(sweep_results, x_param="lat", y_param="top")

    # Build PDF
    print("\nGenerating PDF report...")
    pdf_path = os.path.join(OUTPUT_DIR, "backtest_report.pdf")
    build_pdf(
        result=result,
        bias_results=bias_results,
        validation_results=validation_results,
        sweep_results=sweep_results,
        latency_results=latency_results,
        wf_details=wf.details,
        rand_details=rb.details,
        heatmap_data=heatmap_data,
        wallet_scores=wallet_scores,
        wallet_aliases=wallet_aliases,
        output_path=pdf_path,
    )

    print(f"\nDone! Report at: {pdf_path}")
    print(f"Charts at: {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
