#!/usr/bin/env python3
"""Launch the sharp-tail live mirror trading system.

Usage:
    # Dry run (no real orders) with 1 contract per trade:
    python run_live.py --dry-run --size 1

    # Live trading with 1 contract per trade:
    python run_live.py --size 1

    # Live trading with default liquidity-based sizing:
    python run_live.py

Prerequisites:
    1. Copy .env.example to .env and fill in your credentials
    2. Run the wallet scanner first (or let this script do it):
       python -m polymir scan
    3. Ensure you have MATIC for gas and USDC for trading on Polygon

Environment variables (or .env file):
    POLYMARKET_PRIVATE_KEY  - Ethereum private key (hex, with or without 0x prefix)
    POLYMARKET_API_KEY      - (optional) CLOB API key — derived from private key if omitted
    POLYMARKET_API_SECRET   - (optional) CLOB API secret
    POLYMARKET_API_PASSPHRASE - (optional) CLOB API passphrase
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import structlog

from polymir.config import AppConfig, APIConfig, ScoringConfig, ExecutionConfig
from polymir.db import Database
from polymir.executor import MirrorExecutor
from polymir.logging import setup_logging


logger = structlog.get_logger()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sharp-tail live mirror trader")
    p.add_argument("--size", type=float, default=1.0,
                    help="Contracts per trade (default: 1)")
    p.add_argument("--dry-run", action="store_true",
                    help="Log signals without placing orders")
    p.add_argument("--top-wallets", type=int, default=20,
                    help="Number of top wallets to mirror (default: 20)")
    p.add_argument("--db", default="polymir.db",
                    help="Database path (default: polymir.db)")
    p.add_argument("--scan-first", action="store_true",
                    help="Run wallet scanner before starting (requires API access)")
    p.add_argument("--log-level", default="INFO",
                    help="Log level (default: INFO)")
    return p.parse_args()


async def run(args: argparse.Namespace) -> None:
    setup_logging(args.log_level)

    # Validate credentials
    private_key = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
    if not private_key and not args.dry_run:
        print("ERROR: POLYMARKET_PRIVATE_KEY not set. Use --dry-run to test without a key.")
        sys.exit(1)

    api_config = APIConfig(
        api_key=os.environ.get("POLYMARKET_API_KEY", ""),
        api_secret=os.environ.get("POLYMARKET_API_SECRET", ""),
        api_passphrase=os.environ.get("POLYMARKET_API_PASSPHRASE", ""),
        private_key=private_key,
    )

    config = AppConfig(
        api=api_config,
        scoring=ScoringConfig(),
        execution=ExecutionConfig(
            fixed_size_contracts=args.size,
            max_position_usd=1_000.0,
            stale_signal_timeout_s=300.0,
            min_liquidity_usd=500.0,  # lower bar for 1-contract trades
            max_spread_pct=0.05,      # 5% spread tolerance
            max_slippage_pct=0.03,    # 3% slippage tolerance
        ),
        db_path=args.db,
        top_wallets=args.top_wallets,
    )

    async with Database(config.db_path) as db:
        # Optionally scan wallets first
        if args.scan_first:
            from polymir.scanner import WalletScanner
            logger.info("scanning_wallets")
            scanner = WalletScanner(config, db)
            scores = await scanner.run()
            logger.info("scan_complete", qualified=len(scores))
            if not scores:
                print("No qualified wallets found. Run with more market data first.")
                sys.exit(1)

        # Check watchlist
        top = await db.get_top_wallets(limit=config.top_wallets)
        if not top:
            print("No wallets in database. Run 'python -m polymir scan' first, or use --scan-first.")
            sys.exit(1)

        mode = "DRY RUN" if args.dry_run else "LIVE"
        print(f"\n{'='*60}")
        print(f"  Sharp-tail Mirror Trader — {mode}")
        print(f"  Watching {len(top)} wallets")
        print(f"  Size: {args.size} contract(s) per trade")
        print(f"  DB: {args.db}")
        print(f"{'='*60}\n")

        for i, w in enumerate(top[:5], 1):
            print(f"  #{i} {w.address}  score={w.composite_score:.3f}  win={w.win_rate:.0%}  mkts={w.resolved_market_count}")
        if len(top) > 5:
            print(f"  ... and {len(top) - 5} more")
        print()

        executor = MirrorExecutor(config, db, dry_run=args.dry_run)
        await executor.run()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
