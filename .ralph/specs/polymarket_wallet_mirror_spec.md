# Technical Specification: Polymarket Wallet Mirror Trading System

## 1. Polymarket API Reference

### 1.1 CLOB API (Central Limit Order Book)

Base URL: `https://clob.polymarket.com`

Key endpoints:
- `GET /markets` -- list active markets with token IDs
- `GET /book?token_id={id}` -- full orderbook for a market outcome token
- `GET /trades?market={id}` -- recent trades for a market
- `GET /trades?maker_address={addr}` -- trades by a specific wallet
- `POST /order` -- place a limit order (requires API key + CLOB credentials)
- `DELETE /order/{id}` -- cancel an open order

WebSocket: `wss://ws-subscriptions-clob.polymarket.com/ws/`
- Subscribe to trade events filtered by market or maker address
- Messages are JSON with fields: market, price, size, side, maker_address, timestamp

Rate limits: 100 req/min for REST, monitor headers for dynamic limits.

### 1.2 Gamma Markets API

Base URL: `https://gamma-api.polymarket.com`

Key endpoints:
- `GET /markets` -- market metadata including question, description, outcomes, resolution status
- `GET /markets?closed=true` -- resolved markets with outcome data
- `GET /markets/{id}` -- single market detail with resolution info

Use this API for: market discovery, resolution status, outcome labels. Do NOT use for orderbook or trade data (use CLOB for that).

### 1.3 Wallet Activity Data

Polymarket trades are onchain (Polygon) but the CLOB API exposes trade history by maker address. For historical analysis of resolved markets:
- Pull resolved markets from Gamma API
- For each resolved market, pull trades from CLOB API filtered by that market
- Group trades by maker_address to reconstruct per-wallet positions at expiration
- A wallet "held to expiration" if their net position at the last trade before resolution was nonzero

Alternative data source: Dune Analytics or direct Polygon RPC for more complete historical data if the CLOB API history is limited.

## 2. Database Schema

```sql
CREATE TABLE wallets (
    address TEXT PRIMARY KEY,
    first_seen_at TIMESTAMP,
    last_scored_at TIMESTAMP,
    total_markets_participated INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE wallet_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    wallet_address TEXT REFERENCES wallets(address),
    scored_at TIMESTAMP NOT NULL,
    win_rate REAL,
    avg_roi REAL,
    consistency_score REAL,
    recency_score REAL,
    hold_ratio REAL,
    composite_score REAL,
    markets_counted INTEGER,
    UNIQUE(wallet_address, scored_at)
);

CREATE TABLE markets (
    id TEXT PRIMARY KEY,
    question TEXT,
    outcome_yes TEXT,
    outcome_no TEXT,
    resolution_status TEXT,  -- 'open', 'resolved_yes', 'resolved_no'
    resolved_at TIMESTAMP,
    total_volume REAL,
    token_id_yes TEXT,
    token_id_no TEXT
);

CREATE TABLE wallet_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    wallet_address TEXT REFERENCES wallets(address),
    market_id TEXT REFERENCES markets(id),
    side TEXT,  -- 'yes' or 'no'
    avg_entry_price REAL,
    total_size REAL,
    held_to_expiration BOOLEAN,
    pnl REAL,
    UNIQUE(wallet_address, market_id, side)
);

CREATE TABLE mirror_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at TIMESTAMP NOT NULL,
    source_wallet TEXT REFERENCES wallets(address),
    market_id TEXT REFERENCES markets(id),
    side TEXT,
    source_price REAL,
    source_size REAL,
    wallet_composite_score REAL
);

CREATE TABLE mirror_executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER REFERENCES mirror_signals(id),
    decision TEXT NOT NULL,  -- 'executed', 'skipped_slippage', 'skipped_spread', 'skipped_liquidity', 'skipped_stale', 'skipped_other'
    decision_reason TEXT,
    attempted_at TIMESTAMP,
    order_price REAL,
    order_size REAL,
    estimated_slippage_bps REAL,
    actual_fill_price REAL,
    actual_fill_size REAL,
    realized_slippage_bps REAL,
    fill_time_seconds REAL,
    orderbook_best_bid REAL,
    orderbook_best_ask REAL,
    orderbook_depth_usd REAL,
    pnl REAL  -- filled in after market resolves
);
```

## 3. Wallet Scoring Formula

```
composite_score = (
    w_wr * win_rate +
    w_roi * normalized_roi +
    w_cons * consistency_score +
    w_rec * recency_score +
    w_hold * hold_ratio
)
```

Default weights (configurable):
- w_wr = 0.25
- w_roi = 0.30
- w_cons = 0.15
- w_rec = 0.15
- w_hold = 0.15

Normalization: each component scaled to [0, 1] range across the wallet universe before weighting.

Recency decay: `recency_weight(t) = exp(-lambda * days_since_resolution)` where lambda defaults to `ln(2) / 90` (90-day half-life).

Minimum filter: wallets with fewer than 20 resolved markets are excluded from ranking entirely.

## 4. Execution Parameters (Defaults)

| Parameter | Default | Description |
|---|---|---|
| max_slippage_bps | 200 | Skip trade if estimated slippage > 2% |
| max_spread_bps | 300 | Skip trade if bid-ask spread > 3% |
| min_liquidity_usd | 10000 | Skip market if total book depth < $10k |
| stale_signal_seconds | 300 | Skip if > 5 min since wallet traded |
| order_aggression | 0.0 | 0.0 = midpoint, 1.0 = cross the spread |
| fill_timeout_seconds | 60 | Cancel and re-evaluate after 60s |
| max_position_usd | 500 | Maximum position size per mirror trade |
| max_portfolio_exposure_usd | 10000 | Total across all open mirror positions |
| top_n_wallets | 50 | Number of top-scored wallets to track |
| rescore_interval_hours | 24 | Frequency of wallet re-scoring |

## 5. Project Structure

```
polymir/
├── pyproject.toml
├── README.md
├── config.toml              # Default configuration
├── src/
│   └── polymir/
│       ├── __init__.py
│       ├── __main__.py      # CLI entry point (click or argparse)
│       ├── config.py        # Configuration loader (TOML + env vars)
│       ├── api/
│       │   ├── __init__.py
│       │   ├── clob.py      # CLOB REST + WebSocket client
│       │   ├── gamma.py     # Gamma Markets API client
│       │   ├── models.py    # Pydantic response models
│       │   └── rate_limit.py # Rate limiter implementation
│       ├── db/
│       │   ├── __init__.py
│       │   ├── schema.py    # Schema definitions and migrations
│       │   ├── repository.py # Data access layer
│       │   └── models.py    # SQLAlchemy or dataclass models
│       ├── scanner/
│       │   ├── __init__.py
│       │   ├── fetcher.py   # Resolved market + wallet data fetching
│       │   ├── scorer.py    # Wallet scoring model
│       │   └── pipeline.py  # Full scan orchestration
│       ├── monitor/
│       │   ├── __init__.py
│       │   ├── watchlist.py # Wallet watchlist manager
│       │   ├── detector.py  # Trade detection (WS + polling)
│       │   └── signals.py   # Signal emission
│       ├── executor/
│       │   ├── __init__.py
│       │   ├── orderbook.py # Orderbook analysis + slippage estimation
│       │   ├── filters.py   # Pre-trade filters (slippage, spread, liquidity, stale)
│       │   ├── sizer.py     # Position sizing logic
│       │   ├── placer.py    # Order placement + fill management
│       │   └── logger.py    # Execution decision logging
│       └── backtest/
│           ├── __init__.py
│           ├── data.py      # Historical data loader
│           ├── engine.py    # Backtest replay engine
│           └── metrics.py   # Performance metrics
├── tests/
│   ├── conftest.py          # Shared fixtures, mock API responses
│   ├── test_api/
│   ├── test_scanner/
│   ├── test_monitor/
│   ├── test_executor/
│   ├── test_backtest/
│   └── test_integration/
└── scripts/
    └── seed_historical.py   # One-off script to backfill resolved market data
```

## 6. Key Design Decisions

**Why hold-to-expiration wallets?** Most Polymarket alpha comes from information edge on binary outcomes, not from trading the spread. Wallets that buy and hold to expiration are expressing a directional view and collecting the full $1 payout on wins. This is a cleaner signal than wallets that trade in and out (which could be market makers or noise traders).

**Why latency-aware execution?** By the time we detect a wallet's trade and place our own, the market may have moved. The system must model this explicitly rather than assuming we get the same price. The backtest with configurable latency is critical for honest performance estimation.

**Why SQLite first?** The system runs on a single Mac Mini. SQLite is zero-config, fast for this scale (< 100k wallets, < 10k markets), and easy to back up. The schema is designed to be portable to Postgres when/if needed.

**Why async throughout?** We need to simultaneously monitor WebSocket feeds for multiple wallets, check orderbooks, and place orders. Blocking I/O would create bottlenecks that directly translate to worse execution latency and more slippage.
