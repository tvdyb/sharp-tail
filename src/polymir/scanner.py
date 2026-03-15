"""Wallet Scanner: discover and rank wallets by historical profitability."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
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
                net_position = sum(
                    t.size if t.side == "BUY" else -t.size for t in wtrades
                )
                avg_entry = self._weighted_avg_entry(wtrades)
                held_to_expiration = net_position > 0
                won = held_to_expiration and token.token_id in winner_token_ids
                # ROI: if won, payout is 1.0 per share; if lost, payout is 0
                payout = net_position * 1.0 if won else 0.0
                cost = net_position * avg_entry if avg_entry else 0.0
                roi = (payout - cost) / cost if cost > 0 else 0.0

                total_bought = sum(t.size for t in wtrades if t.side == "BUY")
                total_sold = sum(t.size for t in wtrades if t.side == "SELL")

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
    """Compute composite score for a wallet from its market results.

    Args:
        wallet: Wallet address.
        results: Market results to score against.
        config: Scoring configuration.
        as_of: Point-in-time reference for recency weighting.
               If None, uses current UTC time. For backtesting, pass
               the simulation timestamp to prevent look-ahead bias.

    Returns None if the wallet doesn't meet the minimum volume filter.
    """
    if len(results) < config.min_resolved_markets:
        return None

    now = as_of if as_of is not None else datetime.utcnow()

    # Win rate
    wins = sum(1 for r in results if r.won)
    win_rate = wins / len(results)

    # Average ROI
    rois = [r.roi for r in results]
    avg_roi = mean(rois) if rois else 0.0

    # Consistency (inverse coefficient of variation; higher = more consistent)
    if len(rois) > 1 and mean(rois) != 0:
        cv = stdev(rois) / abs(mean(rois))
        consistency = 1.0 / (1.0 + cv)
    else:
        consistency = 0.5

    # Recency weighting (exponential decay)
    half_life = timedelta(days=config.recency_half_life_days)
    decay = math.log(2) / half_life.total_seconds()
    recency_scores = []
    for r in results:
        if r.resolution_date:
            age = (now - r.resolution_date).total_seconds()
            weight = math.exp(-decay * max(age, 0))
        else:
            weight = 0.5
        recency_scores.append(weight * (1.0 if r.won else 0.0))
    recency_score = mean(recency_scores) if recency_scores else 0.0

    # Hold-to-expiration ratio
    held_count = sum(1 for r in results if r.held_to_expiration)
    hold_ratio = held_count / len(results)

    # Composite score
    composite = (
        config.win_rate_weight * win_rate
        + config.roi_weight * _normalize_roi(avg_roi)
        + config.consistency_weight * consistency
        + config.recency_weight * recency_score
        + config.hold_ratio_weight * hold_ratio
    )

    return WalletScore(
        address=wallet,
        win_rate=win_rate,
        avg_roi=avg_roi,
        consistency=consistency,
        recency_score=recency_score,
        hold_ratio=hold_ratio,
        resolved_market_count=len(results),
        composite_score=composite,
    )


def _normalize_roi(roi: float) -> float:
    """Normalize ROI to [0, 1] range using sigmoid-like transform."""
    return 1.0 / (1.0 + math.exp(-roi * 5))
