# Polymarket Wallet Mirror Trading System

## Project Overview

Build a system that generates alpha by identifying consistently profitable Polymarket wallets (those holding contracts to expiration and winning), then mirroring their trades with controlled latency and slippage management.

The system has three core subsystems:
1. **Wallet Scanner** -- discover and rank wallets by historical profitability on resolved markets
2. **Trade Monitor** -- watch ranked wallets in real-time for new position entries
3. **Mirror Executor** -- replicate detected trades on our own account with latency-aware sizing and slippage controls

## Technical Stack

- **Language**: Python 3.11+
- **Framework**: asyncio throughout, no blocking I/O
- **Data**: Polymarket CLOB API (REST + WebSocket), Gamma Markets API for market metadata
- **Database**: SQLite for local wallet scoring and trade history (upgrade path to Postgres later)
- **Testing**: pytest with pytest-asyncio, plus a full backtest harness using historical resolved markets
- **Package management**: uv or pip, pyproject.toml based

## Architecture Principles

- Every module must be independently testable with mocked API responses
- Wallet scoring must be reproducible: given the same resolved market data, produce the same rankings
- The mirror executor must NEVER place a trade without checking current orderbook depth and bid-ask spread
- All API calls must have exponential backoff and rate limiting baked in
- Secrets (API keys, wallet credentials) loaded from environment variables, never hardcoded
- Logging via structlog with JSON output for every trade decision (entry, skip, slippage-reject)

## Wallet Scoring Model

Rank wallets by a composite score combining:
- **Win rate on resolved markets** (contracts held to expiration that resolved in their favor)
- **ROI per market** (profit relative to capital deployed per market)
- **Consistency** (low variance in returns across markets, prefer steady winners over lucky whales)
- **Recency weighting** (recent performance weighted more heavily, exponential decay)
- **Volume filter** (minimum number of resolved markets participated in, e.g. >= 20)
- **Hold-to-expiration ratio** (fraction of positions held to resolution vs. traded out early -- we want holders, not flippers)

The scanner should be able to re-score the entire wallet universe on a nightly cron and persist rankings to the database.

## Mirror Execution Logic

When a tracked wallet enters a new position:
1. Detect the trade via WebSocket or polling (configurable interval)
2. Look up current orderbook for that market on Polymarket CLOB
3. Compute expected slippage given our intended position size
4. If slippage exceeds configurable threshold (default: 2%), skip and log
5. If spread is too wide (configurable, default: 3%), skip and log
6. Place limit order at midpoint or better, with configurable aggression parameter
7. Set a timeout for fill (default: 60s), cancel and re-evaluate if unfilled
8. Log every decision with full context: wallet address, market, side, price, size, slippage estimate, outcome

## Slippage and Latency Considerations

- There WILL be latency between wallet detection and our execution. The system should model expected price impact as a function of time-since-detection and market liquidity.
- For thin markets (< $10k total liquidity), reduce position size proportionally or skip entirely.
- Implement a "stale signal" timeout: if more than N minutes pass between detection and execution readiness, skip the trade.
- Track realized slippage vs. estimated slippage over time as a key performance metric.

## Backtest Requirements

- Build a backtester that replays historical wallet activity against historical orderbook snapshots (or best-available price data)
- The backtester should simulate our mirror strategy with configurable latency (e.g., 30s, 60s, 120s, 300s delay)
- Output: PnL curve, Sharpe ratio, win rate, average slippage, number of trades, number of skipped trades by reason
- The backtester should share core logic with the live system (same scoring, same execution rules) to avoid train/test leakage between backtest and live code paths

## Non-Goals (Out of Scope)

- No web UI (CLI and logs are fine for v1)
- No multi-exchange support (Polymarket only for now)
- No MEV protection or onchain execution optimization (we trade via the CLOB API, not onchain)
- No portfolio optimization across wallets (simple independent mirroring per wallet for v1)

## Development Approach

- Build and test each subsystem independently before integration
- Wallet scanner first (can run against historical data immediately)
- Trade monitor second (requires WebSocket integration)
- Mirror executor last (requires both upstream systems working)
- Integration tests that wire all three together with mocked Polymarket responses
- Every PR-worthy chunk should include tests that pass

## RALPH_STATUS

At the end of each loop iteration, output a status block:

```
RALPH_STATUS:
  PROGRESS: [description of what was accomplished]
  NEXT: [what to work on next]
  BLOCKERS: [any issues encountered]
  EXIT_SIGNAL: false
```

Set `EXIT_SIGNAL: true` only when ALL of the following are true:
- All three subsystems (scanner, monitor, executor) are implemented
- Backtest harness is functional and produces output
- Integration tests pass
- CLI entry points work for: scoring wallets, starting live monitor, running backtest
