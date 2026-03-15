"""Core backtest engine with point-in-time wallet scoring (no look-ahead bias)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import structlog

from polymir.backtest.data import HistoricalTrade, TradeRecord
from polymir.backtest.metrics import BacktestResult
from polymir.config import AppConfig
from polymir.scanner import WalletMarketResult, compute_wallet_score

logger = structlog.get_logger()


class BacktestEngine:
    """Replays historical wallet activity with point-in-time scoring.

    Key design: at each trade timestamp T, wallet scores are computed using
    ONLY markets that resolved BEFORE T. This prevents look-ahead bias.
    """

    def __init__(
        self,
        config: AppConfig,
        latency_s: int = 60,
        top_n: int | None = None,
    ) -> None:
        self._config = config
        self._latency_s = latency_s
        self._top_n = top_n or config.top_wallets

    async def run(
        self,
        trades: list[HistoricalTrade] | None = None,
        wallet_results: list[WalletMarketResult] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        **kwargs: Any,
    ) -> BacktestResult:
        """Run backtest with point-in-time wallet scoring.

        Args:
            trades: Historical trades to replay chronologically.
            wallet_results: Per-wallet market results for scoring. If provided,
                scores are computed point-in-time. If None, all trades are
                treated as coming from qualified wallets.
            start_date: Start date filter (YYYY-MM-DD).
            end_date: End date filter (YYYY-MM-DD).
        """
        if trades is None:
            trades = []

        trades = sorted(trades, key=lambda t: t.timestamp)

        if start_date:
            start = datetime.fromisoformat(start_date)
            trades = [t for t in trades if t.timestamp >= start]
        if end_date:
            end = datetime.fromisoformat(end_date)
            trades = [t for t in trades if t.timestamp <= end]

        # Pre-sort wallet results by resolution date for efficient PIT filtering
        sorted_wr = sorted(
            wallet_results or [],
            key=lambda r: r.resolution_date or datetime.min,
        )

        # Cache: maps resolution_date cutoff index -> {wallet: score}
        # We rebuild when the set of resolved markets changes
        score_cache: dict[str, float] = {}
        rank_cache: list[str] = []
        last_cutoff_idx = -1

        result = BacktestResult(
            latency_s=self._latency_s,
            top_n=self._top_n,
            fee_rate=self._config.execution.fee_rate,
        )
        exec_cfg = self._config.execution

        # Position tracker: market_id -> total exposure (USD)
        open_positions: dict[str, float] = {}
        total_exposure = 0.0
        max_total_exposure = exec_cfg.max_position_usd * self._top_n

        for trade in trades:
            # Point-in-time wallet scoring
            if sorted_wr:
                cutoff_idx = _bisect_resolved_before(sorted_wr, trade.timestamp)
                if cutoff_idx != last_cutoff_idx:
                    score_cache, rank_cache = _compute_pit_scores(
                        sorted_wr[:cutoff_idx],
                        self._config.scoring,
                        trade.timestamp,
                        self._top_n,
                    )
                    last_cutoff_idx = cutoff_idx

                wallet_score = score_cache.get(trade.wallet, 0.0)
                try:
                    wallet_rank = rank_cache.index(trade.wallet) + 1
                except ValueError:
                    wallet_rank = 0

                if trade.wallet not in score_cache:
                    result.trade_records.append(TradeRecord(
                        timestamp=trade.timestamp,
                        wallet=trade.wallet,
                        market_id=trade.market_id,
                        asset_id=trade.asset_id,
                        side=trade.side,
                        signal_price=trade.price,
                        decision="not_qualified",
                        market_category=trade.market_category,
                        market_resolved_price=trade.market_resolved_price,
                    ))
                    continue
            else:
                wallet_score = 1.0
                wallet_rank = 1

            # Stale signal check
            if self._latency_s > exec_cfg.stale_signal_timeout_s:
                result.trade_records.append(TradeRecord(
                    timestamp=trade.timestamp,
                    wallet=trade.wallet,
                    market_id=trade.market_id,
                    asset_id=trade.asset_id,
                    side=trade.side,
                    signal_price=trade.price,
                    decision="stale",
                    wallet_score=wallet_score,
                    wallet_rank=wallet_rank,
                    market_category=trade.market_category,
                    market_resolved_price=trade.market_resolved_price,
                ))
                continue

            # Per-market position limit: don't double up
            if trade.market_id in open_positions:
                result.trade_records.append(TradeRecord(
                    timestamp=trade.timestamp,
                    wallet=trade.wallet,
                    market_id=trade.market_id,
                    asset_id=trade.asset_id,
                    side=trade.side,
                    signal_price=trade.price,
                    decision="duplicate_market",
                    wallet_score=wallet_score,
                    wallet_rank=wallet_rank,
                    market_category=trade.market_category,
                    market_resolved_price=trade.market_resolved_price,
                ))
                continue

            # Total exposure check
            trade_exposure = min(trade.size * trade.price, exec_cfg.max_position_usd)
            if total_exposure + trade_exposure > max_total_exposure:
                result.trade_records.append(TradeRecord(
                    timestamp=trade.timestamp,
                    wallet=trade.wallet,
                    market_id=trade.market_id,
                    asset_id=trade.asset_id,
                    side=trade.side,
                    signal_price=trade.price,
                    decision="exposure_limit",
                    wallet_score=wallet_score,
                    wallet_rank=wallet_rank,
                    market_category=trade.market_category,
                    market_resolved_price=trade.market_resolved_price,
                ))
                continue

            # Simple slippage model: flat cost per trade from midpoint
            slippage = exec_cfg.slippage_per_trade

            # Fill price = signal price + slippage for BUY, - slippage for SELL
            if trade.side == "BUY":
                fill_price = min(trade.price + slippage, 0.99)
            else:
                fill_price = max(trade.price - slippage, 0.01)

            # Position sizing (keep max_position_usd cap)
            max_contracts = exec_cfg.max_position_usd / (fill_price or 1.0)
            order_size = min(trade.size, max_contracts)

            # PnL
            pnl = (trade.market_resolved_price - fill_price) * order_size
            if trade.side == "SELL":
                pnl = (fill_price - trade.market_resolved_price) * order_size

            # Fees (Polymarket has no fees, but configurable)
            fee = fill_price * order_size * exec_cfg.fee_rate
            pnl -= fee

            # Track position
            position_usd = fill_price * order_size
            open_positions[trade.market_id] = position_usd
            total_exposure += position_usd

            result.trade_records.append(TradeRecord(
                timestamp=trade.timestamp,
                wallet=trade.wallet,
                market_id=trade.market_id,
                asset_id=trade.asset_id,
                side=trade.side,
                signal_price=trade.price,
                fill_price=fill_price,
                size=order_size,
                pnl=pnl,
                fee=fee,
                decision="execute",
                wallet_score=wallet_score,
                wallet_rank=wallet_rank,
                market_category=trade.market_category,
                market_resolved_price=trade.market_resolved_price,
            ))

        logger.info("backtest_complete", **{
            "trades_executed": result.trades_executed,
            "trades_skipped": result.trades_skipped,
            "total_pnl": result.total_pnl,
            "sharpe": result.sharpe_ratio,
        })
        return result


def _bisect_resolved_before(
    sorted_results: list[WalletMarketResult],
    cutoff: datetime,
) -> int:
    """Return index of first result with resolution_date >= cutoff.

    All results before this index resolved strictly before cutoff.
    """
    lo, hi = 0, len(sorted_results)
    while lo < hi:
        mid = (lo + hi) // 2
        rd = sorted_results[mid].resolution_date
        if rd is not None and rd < cutoff:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _compute_pit_scores(
    results: list[WalletMarketResult],
    scoring_config: Any,
    as_of: datetime,
    top_n: int,
) -> tuple[dict[str, float], list[str]]:
    """Compute point-in-time wallet scores from resolved market results.

    Returns:
        Tuple of (wallet -> score dict, ranked wallet list).
        Only top_n wallets are included.
    """
    # Group by wallet
    by_wallet: dict[str, list[WalletMarketResult]] = {}
    for r in results:
        by_wallet.setdefault(r.wallet, []).append(r)

    # Score each wallet
    scored: list[tuple[str, float]] = []
    for wallet, wr_list in by_wallet.items():
        score = compute_wallet_score(wallet, wr_list, scoring_config, as_of=as_of)
        if score is not None:
            scored.append((wallet, score.composite_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_n]

    score_dict = {w: s for w, s in top}
    rank_list = [w for w, _ in top]
    return score_dict, rank_list
