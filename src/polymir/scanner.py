"""Wallet Scanner: discover and rank wallets by historical profitability."""

from __future__ import annotations

import math
from datetime import datetime
from statistics import mean, stdev
from typing import Any

import structlog

from polymir.api.clob import ClobClient
from polymir.api.gamma import GammaClient
from polymir.config import AppConfig, ScoringConfig
from polymir.db import Database
from polymir.models import Market, Trade, WalletScore

logger = structlog.get_logger()


class WalletScanner:
    """Scans resolved markets and scores wallets by profitability."""

    def __init__(self, config: AppConfig, db: Database) -> None:
        self._config = config
        self._db = db

    async def run(self) -> list[WalletScore]:
        """Run full wallet scoring pipeline.

        1. Fetch all resolved markets
        2. For each market, fetch wallet activity at expiration
        3. Score each wallet
        4. Persist scores to database
        """
        async with GammaClient(self._config.api) as gamma, ClobClient(self._config.api) as clob:
            markets = await gamma.get_all_resolved_markets()
            logger.info("scanner_markets_fetched", count=len(markets))

            # Collect per-wallet data across all markets
            wallet_data: dict[str, list[WalletMarketResult]] = {}
            for market in markets:
                results = await self._analyze_market(market, clob)
                for r in results:
                    wallet_data.setdefault(r.wallet, []).append(r)

            # Score wallets
            scores = []
            for wallet, results in wallet_data.items():
                score = compute_wallet_score(
                    wallet, results, self._config.scoring
                )
                if score is not None:
                    scores.append(score)
                    await self._db.upsert_wallet_score(score)

            scores.sort(key=lambda s: s.composite_score, reverse=True)
            logger.info("scanner_complete", wallets_scored=len(scores))
            return scores

    async def _analyze_market(
        self, market: Market, clob: ClobClient
    ) -> list[WalletMarketResult]:
        """Analyze a single resolved market to extract wallet results."""
        results: list[WalletMarketResult] = []
        winner_token_ids = {t.token_id for t in market.tokens if t.winner}

        for token in market.tokens:
            trades = await clob.get_trades(asset_id=token.token_id, limit=500)
            # Group trades by wallet
            wallet_trades: dict[str, list[Trade]] = {}
            for trade in trades:
                wallet_trades.setdefault(trade.owner, []).append(trade)

            for wallet, wtrades in wallet_trades.items():
                if not wallet:
                    continue
                total_bought = sum(t.size for t in wtrades if t.side == "BUY")
                total_sold = sum(t.size for t in wtrades if t.side == "SELL")
                net_position = total_bought - total_sold

                if total_bought == 0:
                    continue  # sell-only positions

                avg_entry = self._weighted_avg_entry(wtrades)

                # held_to_expiration: kept >50% of position through resolution
                exit_ratio = total_sold / total_bought if total_bought > 0 else 1.0
                held_to_expiration = exit_ratio < 0.5

                is_winner_token = token.token_id in winner_token_ids

                if net_position > 0:
                    won = is_winner_token
                    payout = net_position * 1.0 if is_winner_token else 0.0
                    cost = net_position * avg_entry if avg_entry else 0.0
                    roi = (payout - cost) / cost if cost > 0 else 0.0
                else:
                    # Fully exited before resolution — realized P&L
                    sell_revenue = sum(
                        t.price * t.size for t in wtrades if t.side == "SELL"
                    )
                    cost = total_bought * avg_entry if avg_entry else 0.0
                    roi = (sell_revenue - cost) / cost if cost > 0 else 0.0
                    won = roi > 0

                results.append(
                    WalletMarketResult(
                        wallet=wallet,
                        market_id=market.condition_id,
                        won=won,
                        roi=roi,
                        held_to_expiration=held_to_expiration,
                        total_bought=total_bought,
                        total_sold=total_sold,
                        resolution_date=market.resolution_date,
                    )
                )
        return results

    @staticmethod
    def _weighted_avg_entry(trades: list[Trade]) -> float:
        """Compute volume-weighted average entry price for BUY trades."""
        buys = [t for t in trades if t.side == "BUY"]
        total_size = sum(t.size for t in buys)
        if total_size == 0:
            return 0.0
        return sum(t.price * t.size for t in buys) / total_size


class WalletMarketResult:
    """Result of a wallet's participation in a single resolved market."""

    __slots__ = (
        "wallet",
        "market_id",
        "won",
        "roi",
        "held_to_expiration",
        "total_bought",
        "total_sold",
        "resolution_date",
    )

    def __init__(
        self,
        wallet: str,
        market_id: str,
        won: bool,
        roi: float,
        held_to_expiration: bool,
        total_bought: float,
        total_sold: float,
        resolution_date: datetime | None,
    ) -> None:
        self.wallet = wallet
        self.market_id = market_id
        self.won = won
        self.roi = roi
        self.held_to_expiration = held_to_expiration
        self.total_bought = total_bought
        self.total_sold = total_sold
        self.resolution_date = resolution_date


def compute_wallet_score(
    wallet: str,
    results: list[WalletMarketResult],
    config: ScoringConfig,
    as_of: datetime | None = None,
) -> WalletScore | None:
    """Compute wallet rating from the lower CI bound of per-market Sharpe.

    The rating naturally rewards:
    - High risk-adjusted returns (high Sharpe)
    - Long track records (more markets → tighter CI → higher floor)
    - Holding to expiration (hard filter + scored on held positions only)

    Args:
        wallet: Wallet address.
        results: Market results to score against.
        config: Scoring configuration.
        as_of: Unused, kept for backtest engine API compatibility.

    Returns None if the wallet doesn't meet minimum filters.
    """
    if len(results) < config.min_resolved_markets:
        return None

    # Hold-to-expiration ratio — filter out wallets that exit early
    held_count = sum(1 for r in results if r.held_to_expiration)
    hold_ratio = held_count / len(results)
    if hold_ratio < config.min_hold_ratio:
        return None

    # Score only on held-to-expiration positions
    held_results = [r for r in results if r.held_to_expiration]
    if len(held_results) < 2:
        return None

    # Win rate (held positions only)
    wins = sum(1 for r in held_results if r.won)
    win_rate = wins / len(held_results)

    # Per-market ROIs (held positions only)
    rois = [r.roi for r in held_results]
    avg_roi = mean(rois)

    # Sharpe ratio = mean(roi) / stdev(roi), capped to avoid numerical blow-up
    sd = stdev(rois)
    if sd < 0.01:
        # Near-zero variance means all positions have ~identical ROI
        # (e.g. buying at $0.999).  Not a meaningful signal — skip.
        return None
    sharpe = max(min(avg_roi / sd, 10.0), -10.0)

    # Sharpe CI: SE(sharpe) ≈ sqrt((1 + sharpe²/2) / n)
    n = len(rois)
    se = math.sqrt((1.0 + sharpe ** 2 / 2.0) / n)

    # z-score for desired confidence level
    z = _z_score(config.ci_confidence)
    ci_lower = sharpe - z * se
    ci_upper = sharpe + z * se

    return WalletScore(
        address=wallet,
        win_rate=win_rate,
        avg_roi=avg_roi,
        sharpe_ratio=sharpe,
        sharpe_ci_lower=ci_lower,
        sharpe_ci_upper=ci_upper,
        hold_ratio=hold_ratio,
        resolved_market_count=len(results),
        composite_score=ci_lower,
    )


def _z_score(confidence: float) -> float:
    """Approximate z-score for a two-tailed confidence level.

    Uses the rational approximation from Abramowitz & Stegun.
    """
    alpha = (1.0 - confidence) / 2.0
    # Rational approximation of the inverse normal CDF
    t = math.sqrt(-2.0 * math.log(alpha))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
