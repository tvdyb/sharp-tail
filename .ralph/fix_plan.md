# Fix Plan -- Polymarket Wallet Mirror Trading System

## Phase 1: Project Scaffolding & API Layer

- [x] Initialize project with pyproject.toml, src layout, pytest config, structlog setup
- [x] Create Polymarket API client module with rate limiting and exponential backoff
  - [x] REST client for CLOB endpoints (orderbook, markets, trades)
  - [x] REST client for Gamma Markets API (market metadata, resolved markets)
  - [x] WebSocket client for real-time trade feed
  - [x] Response models (Pydantic) for all API types
- [x] Write tests for API client with mocked responses (happy path + error cases + rate limit handling)
- [x] Set up SQLite database schema: wallets, wallet_scores, markets, trades, mirror_trades, execution_log
- [x] Database access layer with async sqlite (aiosqlite) and migration support

## Phase 2: Wallet Scanner

- [x] Build resolved market fetcher: pull all resolved markets and their outcomes from Gamma API
- [x] Build wallet activity fetcher: for each resolved market, pull all wallets that held positions at expiration
- [x] Implement wallet scoring model:
  - [x] Win rate calculation (resolved-in-favor / total-resolved)
  - [x] Per-market ROI calculation
  - [x] Consistency score (inverse coefficient of variation of per-market returns)
  - [x] Recency weighting (exponential decay with configurable half-life)
  - [x] Volume filter (minimum resolved market count threshold)
  - [x] Hold-to-expiration ratio
  - [x] Composite score combining all factors with configurable weights
- [x] Persist wallet scores to database with timestamp for historical tracking
- [x] CLI command: `python -m polymir scan` to run full wallet scoring pipeline
- [x] Tests for scoring model with synthetic wallet data (known-outcome edge cases)
- [x] Tests for full scanner pipeline with mocked API data

## Phase 3: Trade Monitor

- [ ] Implement wallet watchlist manager (load top-N wallets from database by score)
- [ ] Build real-time trade detection via Polymarket WebSocket feed
  - [ ] Filter for trades by watched wallet addresses
  - [ ] Detect new position entries (distinguish from exits, increases, decreases)
  - [ ] Handle WebSocket reconnection and heartbeat
- [ ] Build polling-based fallback for trade detection (REST-based, configurable interval)
- [ ] Emit trade signals as async events (asyncio.Queue or similar)
- [ ] CLI command: `python -m polymir monitor` to start watching wallets
- [ ] Tests for trade detection logic with simulated WebSocket messages
- [ ] Tests for signal emission and filtering

## Phase 4: Mirror Executor

- [ ] Implement orderbook fetcher for target market at execution time
- [ ] Slippage estimator: given order size and current book depth, estimate expected fill price
- [ ] Spread checker: compute bid-ask spread, reject if too wide
- [ ] Liquidity filter: skip thin markets below configurable liquidity threshold
- [ ] Stale signal filter: skip if time since wallet trade exceeds configurable timeout
- [ ] Position sizer: scale order size based on liquidity, wallet conviction (bet size relative to wallet history), and configurable max position
- [ ] Order placer: submit limit order via CLOB API at midpoint or better
  - [ ] Configurable aggression parameter (how far past midpoint to place)
  - [ ] Fill timeout with cancel-and-reeval logic
- [ ] Execution logger: log every decision (trade, skip, reject) with full context to database and structlog
- [ ] CLI command: `python -m polymir trade` to start live mirror execution
- [ ] Tests for slippage estimation with synthetic orderbooks
- [ ] Tests for execution decision logic (slippage reject, spread reject, liquidity skip, stale skip)
- [ ] Tests for order placement with mocked CLOB API

## Phase 5: Backtest Harness

- [ ] Historical data loader: fetch or load cached resolved market data with price history
- [ ] Simulated orderbook snapshots (or use historical trade data to reconstruct approximate book state)
- [ ] Backtest engine that replays wallet activity chronologically
  - [ ] Configurable mirror latency parameter (30s, 60s, 120s, 300s)
  - [ ] Apply same scoring, filtering, and execution logic as live system
  - [ ] Track simulated fills with slippage model
- [ ] Performance metrics calculator: PnL curve, Sharpe, win rate, avg slippage, trade count, skip breakdown
- [ ] CLI command: `python -m polymir backtest --latency 60 --start 2024-01-01`
- [ ] Tests for backtest engine with known-outcome synthetic data
- [ ] Validate backtest results against hand-calculated examples

## Phase 6: Integration & Polish

- [ ] Integration test: scanner -> monitor -> executor pipeline with mocked APIs end-to-end
- [ ] Configuration file support (TOML or YAML) for all tunable parameters
- [ ] README with setup instructions, architecture overview, and usage examples
- [ ] Realized vs. estimated slippage tracking and reporting
- [ ] Graceful shutdown handling (SIGINT/SIGTERM, cancel open orders, flush logs)
- [ ] Docker Compose setup for running on Mac Mini as a persistent service
- [ ] Health check endpoint or status CLI command showing system state
