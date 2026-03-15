"""Core backtest engine with point-in-time wallet scoring (no look-ahead bias)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

import structlog

from polymir.backtest.data import HistoricalOrderbook, HistoricalTrade, TradeRecord
from polymir.backtest.metrics import BacktestResult
from polymir.config import AppConfig
from polymir.models import OrderBook, OrderBookLevel
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
        orderbooks: list[HistoricalOrderbook] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> BacktestResult:
        """Run backtest with point-in-time wallet scoring.

        Args:
            trades: Historical trades to replay chronologically.
            wallet_results: Per-wallet market results for scoring. If provided,
                scores are computed point-in-time. If None, all trades are
                treated as coming from qualified wallets.
            orderbooks: Historical orderbook snapshots for slippage estimation.
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

        book_lookup = _build_book_lookup(orderbooks or [])

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
            exec_time = trade.timestamp + timedelta(seconds=self._latency_s)

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

            book, is_synthetic = _get_book_at(
                trade.asset_id, exec_time, book_lookup, trade,
                spread=exec_cfg.synthetic_book_spread,
                top_size=exec_cfg.synthetic_book_top_size,
            )
            if is_synthetic:
                result.synthetic_book_count += 1
            else:
                result.real_book_count += 1

            # Liquidity check
            total_liq = book.total_liquidity()
            if total_liq < exec_cfg.min_liquidity_usd:
                result.trade_records.append(TradeRecord(
                    timestamp=trade.timestamp,
                    wallet=trade.wallet,
                    market_id=trade.market_id,
                    asset_id=trade.asset_id,
                    side=trade.side,
                    signal_price=trade.price,
                    decision="liquidity",
                    wallet_score=wallet_score,
                    wallet_rank=wallet_rank,
                    market_category=trade.market_category,
                    market_resolved_price=trade.market_resolved_price,
                ))
                continue

            # Spread check
            spread_pct = book.spread_pct
            if spread_pct is not None and spread_pct > exec_cfg.max_spread_pct:
                result.trade_records.append(TradeRecord(
                    timestamp=trade.timestamp,
                    wallet=trade.wallet,
                    market_id=trade.market_id,
                    asset_id=trade.asset_id,
                    side=trade.side,
                    signal_price=trade.price,
                    decision="spread",
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

            # Position sizing
            max_contracts = exec_cfg.max_position_usd / (trade.price or 1.0)
            order_size = min(trade.size, max_contracts)

            # Slippage estimation
            fill_price = book.estimate_fill_price(order_size, trade.side)
            if fill_price is None:
                result.trade_records.append(TradeRecord(
                    timestamp=trade.timestamp,
                    wallet=trade.wallet,
                    market_id=trade.market_id,
                    asset_id=trade.asset_id,
                    side=trade.side,
                    signal_price=trade.price,
                    decision="no_fill",
                    wallet_score=wallet_score,
                    wallet_rank=wallet_rank,
                    market_category=trade.market_category,
                    market_resolved_price=trade.market_resolved_price,
                ))
                continue

            midpoint = book.midpoint or trade.price
            est_slippage = abs(fill_price - midpoint) / midpoint if midpoint > 0 else 0

            if est_slippage > exec_cfg.max_slippage_pct:
                result.trade_records.append(TradeRecord(
                    timestamp=trade.timestamp,
                    wallet=trade.wallet,
                    market_id=trade.market_id,
                    asset_id=trade.asset_id,
                    side=trade.side,
                    signal_price=trade.price,
                    decision="slippage",
                    estimated_slippage=est_slippage,
                    wallet_score=wallet_score,
                    wallet_rank=wallet_rank,
                    market_category=trade.market_category,
                    market_resolved_price=trade.market_resolved_price,
                ))
                continue

            # Execute trade
            pnl = (trade.market_resolved_price - fill_price) * order_size
            if trade.side == "SELL":
                pnl = (fill_price - trade.market_resolved_price) * order_size

            # Fees
            fee = fill_price * order_size * exec_cfg.fee_rate
            pnl -= fee

            # Sqrt-time impact model (more realistic than linear)
            realized_slippage = est_slippage * math.sqrt(1.0 + self._latency_s / 60.0)

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
                estimated_slippage=est_slippage,
                realized_slippage=realized_slippage,
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


def _build_book_lookup(
    orderbooks: list[HistoricalOrderbook],
) -> dict[str, list[HistoricalOrderbook]]:
    """Group orderbook snapshots by asset_id, sorted by time."""
    lookup: dict[str, list[HistoricalOrderbook]] = {}
    for ob in orderbooks:
        lookup.setdefault(ob.asset_id, []).append(ob)
    for books in lookup.values():
        books.sort(key=lambda b: b.timestamp)
    return lookup


def _get_book_at(
    asset_id: str,
    at_time: datetime,
    lookup: dict[str, list[HistoricalOrderbook]],
    trade: HistoricalTrade,
    spread: float = 0.04,
    top_size: float = 100.0,
) -> tuple[OrderBook, bool]:
    """Get closest orderbook snapshot at or before at_time.

    Returns:
        Tuple of (OrderBook, is_synthetic).
    """
    books = lookup.get(asset_id, [])
    best = None
    for b in books:
        if b.timestamp <= at_time:
            best = b
        else:
            break
    if best:
        return best.to_orderbook(), False
    return _synthetic_book(trade, spread=spread, top_size=top_size), True


def _synthetic_book(
    trade: HistoricalTrade,
    spread: float = 0.04,
    top_size: float = 100.0,
) -> OrderBook:
    """Create a synthetic multi-level orderbook from a single trade price.

    Uses two price levels with decaying depth to be more conservative
    than the old single-level 500-contract book.
    """
    mid = trade.price
    return OrderBook(
        asset_id=trade.asset_id,
        bids=[
            OrderBookLevel(price=max(0.01, mid - spread / 2), size=top_size),
            OrderBookLevel(price=max(0.01, mid - spread), size=top_size * 0.5),
        ],
        asks=[
            OrderBookLevel(price=min(0.99, mid + spread / 2), size=top_size),
            OrderBookLevel(price=min(0.99, mid + spread), size=top_size * 0.5),
        ],
    )
