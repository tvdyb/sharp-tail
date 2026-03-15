"""Configuration management for polymir."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class APIConfig:
    """Polymarket API configuration."""

    clob_base_url: str = "https://clob.polymarket.com"
    gamma_base_url: str = "https://gamma-api.polymarket.com"
    ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = ""
    rate_limit_per_second: float = 5.0
    request_timeout: float = 30.0
    max_retries: int = 5
    backoff_base: float = 1.0
    backoff_max: float = 60.0

    @classmethod
    def from_env(cls) -> APIConfig:
        return cls(
            api_key=os.environ.get("POLYMARKET_API_KEY", ""),
            api_secret=os.environ.get("POLYMARKET_API_SECRET", ""),
            api_passphrase=os.environ.get("POLYMARKET_API_PASSPHRASE", ""),
        )


@dataclass(frozen=True)
class ScoringConfig:
    """Wallet scoring parameters."""

    min_resolved_markets: int = 20
    recency_half_life_days: float = 90.0
    win_rate_weight: float = 0.30
    roi_weight: float = 0.25
    consistency_weight: float = 0.20
    recency_weight: float = 0.15
    hold_ratio_weight: float = 0.10


@dataclass(frozen=True)
class ExecutionConfig:
    """Mirror execution parameters."""

    stale_signal_timeout_s: float = 300.0
    fill_timeout_s: float = 60.0
    aggression: float = 0.0  # 0 = midpoint, positive = more aggressive
    max_position_usd: float = 1_000.0
    poll_interval_s: float = 5.0
    fee_rate: float = 0.0  # Polymarket has no trading fees
    signal_sides: tuple[str, ...] = ("BUY",)  # sides to signal on; add "SELL" to include exits
    slippage_per_trade: float = 0.03  # flat 3 cent slippage from mid per trade
    # Live executor orderbook checks (not used in backtest)
    max_slippage_pct: float = 0.02
    max_spread_pct: float = 0.03
    min_liquidity_usd: float = 10_000.0


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    api: APIConfig = field(default_factory=APIConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    db_path: str = "polymir.db"
    top_wallets: int = 50
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> AppConfig:
        return cls(
            api=APIConfig.from_env(),
            db_path=os.environ.get("POLYMIR_DB_PATH", "polymir.db"),
            log_level=os.environ.get("POLYMIR_LOG_LEVEL", "INFO"),
        )
