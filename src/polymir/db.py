"""SQLite database layer with async access via aiosqlite."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import aiosqlite
import structlog

from polymir.models import MirrorDecision, MirrorTrade, TradeSignal, WalletScore

logger = structlog.get_logger()

SCHEMA = """
CREATE TABLE IF NOT EXISTS wallets (
    address TEXT PRIMARY KEY,
    first_seen TEXT,
    last_seen TEXT,
    total_trades INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS wallet_scores (
    address TEXT NOT NULL,
    win_rate REAL NOT NULL,
    avg_roi REAL NOT NULL,
    sharpe_ratio REAL NOT NULL,
    sharpe_ci_lower REAL NOT NULL,
    sharpe_ci_upper REAL NOT NULL,
    hold_ratio REAL NOT NULL,
    resolved_market_count INTEGER NOT NULL,
    composite_score REAL NOT NULL,
    scored_at TEXT NOT NULL,
    PRIMARY KEY (address, scored_at)
);

CREATE TABLE IF NOT EXISTS markets (
    condition_id TEXT PRIMARY KEY,
    question TEXT,
    status TEXT,
    outcome TEXT,
    end_date TEXT,
    resolution_date TEXT,
    volume REAL DEFAULT 0,
    liquidity REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,
    market TEXT,
    asset_id TEXT,
    owner TEXT,
    side TEXT,
    size REAL,
    price REAL,
    timestamp TEXT
);

CREATE TABLE IF NOT EXISTS mirror_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    wallet_address TEXT NOT NULL,
    market_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    signal_side TEXT NOT NULL,
    signal_size REAL NOT NULL,
    signal_price REAL NOT NULL,
    detected_at TEXT NOT NULL,
    decision TEXT NOT NULL,
    order_price REAL,
    order_size REAL,
    estimated_slippage REAL,
    realized_slippage REAL,
    fill_price REAL,
    fill_size REAL,
    order_id TEXT,
    executed_at TEXT,
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_wallet_scores_composite ON wallet_scores(composite_score DESC);
CREATE INDEX IF NOT EXISTS idx_wallet_scores_address ON wallet_scores(address);
CREATE INDEX IF NOT EXISTS idx_trades_owner ON trades(owner);
CREATE INDEX IF NOT EXISTS idx_trades_market ON trades(market);
CREATE INDEX IF NOT EXISTS idx_mirror_trades_wallet ON mirror_trades(wallet_address);
"""


class Database:
    """Async SQLite database manager."""

    def __init__(self, db_path: str = "polymir.db") -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()
        logger.info("db_connected", path=self._db_path)

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> Database:
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    def _ensure_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected")
        return self._conn

    # ── Wallet Scores ────────────────────────────────────────────

    async def upsert_wallet_score(self, score: WalletScore) -> None:
        conn = self._ensure_conn()
        await conn.execute(
            """INSERT OR REPLACE INTO wallet_scores
               (address, win_rate, avg_roi, sharpe_ratio, sharpe_ci_lower,
                sharpe_ci_upper, hold_ratio, resolved_market_count,
                composite_score, scored_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                score.address,
                score.win_rate,
                score.avg_roi,
                score.sharpe_ratio,
                score.sharpe_ci_lower,
                score.sharpe_ci_upper,
                score.hold_ratio,
                score.resolved_market_count,
                score.composite_score,
                score.scored_at.isoformat(),
            ),
        )
        await conn.commit()

    async def get_top_wallets(self, limit: int = 50) -> list[WalletScore]:
        conn = self._ensure_conn()
        cursor = await conn.execute(
            """SELECT DISTINCT address, win_rate, avg_roi, sharpe_ratio,
                      sharpe_ci_lower, sharpe_ci_upper, hold_ratio,
                      resolved_market_count, composite_score, scored_at
               FROM wallet_scores
               WHERE scored_at = (SELECT MAX(scored_at) FROM wallet_scores ws2 WHERE ws2.address = wallet_scores.address)
               ORDER BY composite_score DESC
               LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            WalletScore(
                address=r["address"],
                win_rate=r["win_rate"],
                avg_roi=r["avg_roi"],
                sharpe_ratio=r["sharpe_ratio"],
                sharpe_ci_lower=r["sharpe_ci_lower"],
                sharpe_ci_upper=r["sharpe_ci_upper"],
                hold_ratio=r["hold_ratio"],
                resolved_market_count=r["resolved_market_count"],
                composite_score=r["composite_score"],
                scored_at=datetime.fromisoformat(r["scored_at"]),
            )
            for r in rows
        ]

    async def get_wallet_score(self, address: str) -> WalletScore | None:
        conn = self._ensure_conn()
        cursor = await conn.execute(
            """SELECT * FROM wallet_scores WHERE address = ? ORDER BY scored_at DESC LIMIT 1""",
            (address,),
        )
        r = await cursor.fetchone()
        if not r:
            return None
        return WalletScore(
            address=r["address"],
            win_rate=r["win_rate"],
            avg_roi=r["avg_roi"],
            sharpe_ratio=r["sharpe_ratio"],
            sharpe_ci_lower=r["sharpe_ci_lower"],
            sharpe_ci_upper=r["sharpe_ci_upper"],
            hold_ratio=r["hold_ratio"],
            resolved_market_count=r["resolved_market_count"],
            composite_score=r["composite_score"],
            scored_at=datetime.fromisoformat(r["scored_at"]),
        )

    # ── Mirror Trades ────────────────────────────────────────────

    async def record_mirror_trade(self, trade: MirrorTrade) -> int:
        conn = self._ensure_conn()
        import json

        cursor = await conn.execute(
            """INSERT INTO mirror_trades
               (wallet_address, market_id, asset_id, signal_side, signal_size,
                signal_price, detected_at, decision, order_price, order_size,
                estimated_slippage, realized_slippage, fill_price, fill_size,
                order_id, executed_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade.signal.wallet_address,
                trade.signal.market_id,
                trade.signal.asset_id,
                trade.signal.side,
                trade.signal.size,
                trade.signal.price,
                trade.signal.detected_at.isoformat(),
                trade.decision.value,
                trade.order_price,
                trade.order_size,
                trade.estimated_slippage,
                trade.realized_slippage,
                trade.fill_price,
                trade.fill_size,
                trade.order_id,
                trade.executed_at.isoformat() if trade.executed_at else None,
                json.dumps(trade.metadata),
            ),
        )
        await conn.commit()
        return cursor.lastrowid or 0

    async def get_mirror_trades(
        self,
        wallet: str | None = None,
        decision: MirrorDecision | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        conn = self._ensure_conn()
        query = "SELECT * FROM mirror_trades WHERE 1=1"
        params: list[Any] = []
        if wallet:
            query += " AND wallet_address = ?"
            params.append(wallet)
        if decision:
            query += " AND decision = ?"
            params.append(decision.value)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ── Slippage Tracking ────────────────────────────────────────

    async def get_slippage_stats(self) -> dict[str, float]:
        """Get aggregate slippage statistics for executed trades."""
        conn = self._ensure_conn()
        cursor = await conn.execute(
            """SELECT
                 AVG(estimated_slippage) as avg_estimated,
                 AVG(realized_slippage) as avg_realized,
                 COUNT(*) as trade_count
               FROM mirror_trades
               WHERE decision = 'execute' AND estimated_slippage IS NOT NULL"""
        )
        row = await cursor.fetchone()
        if not row:
            return {"avg_estimated": 0.0, "avg_realized": 0.0, "trade_count": 0}
        return {
            "avg_estimated": row["avg_estimated"] or 0.0,
            "avg_realized": row["avg_realized"] or 0.0,
            "trade_count": row["trade_count"],
        }
