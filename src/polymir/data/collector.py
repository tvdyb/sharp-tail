"""Data collector for alpha research — fetches markets, prices, and metadata."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta
from typing import Any

import aiohttp
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from polymir.api.rate_limiter import RateLimiter
from polymir.config import APIConfig
from polymir.research.models import (
    DataQualityReport,
    MarketCategory,
    PriceSnapshot,
    ResearchMarket,
)

logger = structlog.get_logger()

# Keywords for market categorization
_CATEGORY_KEYWORDS: dict[MarketCategory, list[str]] = {
    MarketCategory.POLITICS: [
        "president", "election", "congress", "senate", "democrat", "republican",
        "trump", "biden", "governor", "vote", "ballot", "impeach", "political",
        "gop", "dnc", "rnc", "primary", "caucus", "kamala", "harris", "nominee",
        "cabinet", "secretary", "speaker", "house", "mayor", "poll",
    ],
    MarketCategory.SPORTS: [
        "nfl", "nba", "mlb", "nhl", "fifa", "super bowl", "world series",
        "championship", "playoff", "mvp", "touchdown", "goal", "score",
        "game", "match", "league", "team", "player", "coach", "season",
        "tennis", "golf", "ufc", "boxing", "olympics", "medal",
    ],
    MarketCategory.CRYPTO: [
        "bitcoin", "btc", "ethereum", "eth", "crypto", "token", "blockchain",
        "solana", "sol", "dogecoin", "doge", "binance", "coinbase", "defi",
        "nft", "altcoin", "mining", "halving", "memecoin", "xrp", "ada",
    ],
    MarketCategory.WEATHER: [
        "temperature", "weather", "hurricane", "tornado", "rainfall",
        "snowfall", "heat", "cold", "storm", "flood", "drought", "climate",
        "celsius", "fahrenheit", "forecast", "noaa", "wildfire",
    ],
    MarketCategory.CULTURE: [
        "oscar", "emmy", "grammy", "movie", "film", "album", "song",
        "twitter", "tiktok", "instagram", "youtube", "celebrity", "kardashian",
        "taylor swift", "elon musk", "viral", "streaming", "netflix",
        "subscriber", "follower", "viewer", "audience", "award",
    ],
    MarketCategory.ECONOMICS: [
        "fed", "federal reserve", "interest rate", "inflation", "gdp",
        "unemployment", "cpi", "ppi", "fomc", "rate cut", "rate hike",
        "treasury", "bond", "yield", "recession", "jobs report", "nonfarm",
        "payroll", "housing", "consumer", "tariff", "trade deficit",
    ],
    MarketCategory.SCIENCE: [
        "spacex", "nasa", "launch", "rocket", "mars", "moon", "satellite",
        "ai", "artificial intelligence", "fda", "vaccine", "drug", "trial",
        "quantum", "fusion", "genome", "crispr", "research", "discovery",
    ],
}


def categorize_market(question: str, slug: str) -> MarketCategory:
    """Categorize a market based on its question and slug text."""
    text = f"{question} {slug}".lower()
    scores: dict[MarketCategory, int] = {}
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[cat] = score
    if not scores:
        return MarketCategory.OTHER
    return max(scores, key=scores.get)  # type: ignore[arg-type]


class DataCollector:
    """Fetches and stores all market data needed for alpha research."""

    def __init__(self, config: APIConfig | None = None, db_path: str = "research.db") -> None:
        self._config = config or APIConfig()
        self._limiter = RateLimiter(self._config.rate_limit_per_second)
        self._db_path = db_path
        self._session: aiohttp.ClientSession | None = None
        self._markets: list[ResearchMarket] = []
        self._prices: list[PriceSnapshot] = []

    async def __aenter__(self) -> DataCollector:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._config.request_timeout),
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("DataCollector must be used as async context manager")
        return self._session

    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, max=60),
        reraise=True,
    )
    async def _get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        await self._limiter.acquire()
        session = self._ensure_session()
        async with session.get(url, params=params) as resp:
            if resp.status == 429:
                await asyncio.sleep(5)
                raise aiohttp.ClientError("Rate limited")
            if resp.status != 200:
                text = await resp.text()
                raise aiohttp.ClientError(f"HTTP {resp.status}: {text[:200]}")
            return await resp.json()

    # ── Market Collection ──────────────────────────────────────────

    async def fetch_all_markets(self, max_markets: int = 2000) -> list[ResearchMarket]:
        """Fetch markets from Gamma API with pagination.

        Args:
            max_markets: Maximum total markets to fetch (default 2000).
        """
        all_markets: list[ResearchMarket] = []
        gamma_url = self._config.gamma_base_url

        for status in ["resolved", "active"]:
            if len(all_markets) >= max_markets:
                break
            offset = 0
            page_size = 100
            while len(all_markets) < max_markets:
                try:
                    data = await self._get(
                        f"{gamma_url}/markets",
                        params={
                            "limit": page_size,
                            "offset": offset,
                            "order": "volume",
                            "ascending": "false",
                            "closed": "true" if status == "resolved" else "false",
                        },
                    )
                except Exception as e:
                    logger.error("fetch_markets_error", status=status, offset=offset, error=str(e))
                    break

                if not data:
                    break

                parsed_count = 0
                for m in data:
                    market = self._parse_gamma_market(m)
                    if market:
                        all_markets.append(market)
                        parsed_count += 1

                logger.info("fetching_markets", offset=offset, parsed=parsed_count, total=len(all_markets))

                if len(data) < page_size:
                    break
                offset += page_size

        self._markets = all_markets
        logger.info("markets_fetched", total=len(all_markets))
        return all_markets

    def _parse_gamma_market(self, data: dict[str, Any]) -> ResearchMarket | None:
        """Parse a Gamma API market response into ResearchMarket."""
        try:
            condition_id = data.get("condition_id", data.get("conditionId", ""))
            if not condition_id:
                return None

            question = data.get("question", "")
            slug = data.get("slug", "")

            token_ids = []
            # Gamma API returns clobTokenIds as JSON string or list
            raw_clob = data.get("clobTokenIds", data.get("clob_token_ids"))
            if raw_clob:
                import json as _json
                if isinstance(raw_clob, str):
                    try:
                        token_ids = _json.loads(raw_clob)
                    except (ValueError, TypeError):
                        token_ids = [raw_clob]
                elif isinstance(raw_clob, list):
                    token_ids = raw_clob
            # Fallback to tokens array
            if not token_ids:
                for t in data.get("tokens", []):
                    tid = t.get("token_id", t.get("tokenId", ""))
                    if tid:
                        token_ids.append(tid)

            end_date = self._parse_dt(data.get("end_date") or data.get("endDate"))
            creation_date = self._parse_dt(
                data.get("created_at") or data.get("createdAt") or data.get("creation_date")
            )
            resolution_date = self._parse_dt(
                data.get("closedTime") or data.get("resolution_date") or data.get("resolutionDate")
            )

            # Determine outcome from outcomePrices: "1" means that outcome won
            outcome = ""
            outcome_prices = data.get("outcomePrices")
            outcomes_list = data.get("outcomes", [])
            if outcome_prices and outcomes_list:
                if isinstance(outcome_prices, str):
                    import json as _json2
                    try:
                        outcome_prices = _json2.loads(outcome_prices)
                    except (ValueError, TypeError):
                        outcome_prices = []
                for op, olbl in zip(outcome_prices, outcomes_list):
                    if str(op) == "1":
                        outcome = olbl
                        break

            volume = float(data.get("volumeNum", 0) or data.get("volume", 0) or 0)
            liquidity = float(data.get("liquidityNum", 0) or data.get("liquidity", 0) or 0)

            # neg_risk from events
            events = data.get("events", [])
            neg_risk = any(e.get("enableNegRisk") for e in events) if events else False
            event_id = events[0].get("id", "") if events else ""

            # Status from active/closed flags
            is_active = data.get("active", False)
            is_closed = data.get("closed", False)
            closed_time = data.get("closedTime")
            if is_closed and closed_time:
                status = "resolved"
            elif is_closed:
                status = "closed"
            elif is_active:
                status = "active"
            else:
                status = data.get("status", "")

            category = categorize_market(question, slug)

            lifetime = 0.0
            if creation_date and (resolution_date or end_date):
                ref = resolution_date or end_date
                lifetime = (ref - creation_date).total_seconds() / 86400.0

            return ResearchMarket(
                market_id=condition_id,
                condition_id=condition_id,
                question=question,
                slug=slug,
                category=category,
                creation_date=creation_date,
                end_date=end_date,
                resolution_date=resolution_date,
                outcome=outcome,
                total_volume=volume,
                liquidity=liquidity,
                token_ids=token_ids,
                neg_risk=neg_risk,
                event_id=event_id,
                status=status,
                total_lifetime_days=lifetime,
            )
        except Exception as e:
            logger.warning("parse_market_error", error=str(e), data_keys=list(data.keys()))
            return None

    # ── Price History Collection ────────────────────────────────────

    async def fetch_price_history(
        self,
        markets: list[ResearchMarket] | None = None,
        max_concurrent: int = 5,
    ) -> list[PriceSnapshot]:
        """Fetch price history for all markets via CLOB prices-history endpoint."""
        markets = markets or self._markets
        if not markets:
            logger.warning("no_markets_for_price_fetch")
            return []

        all_snapshots: list[PriceSnapshot] = []
        semaphore = asyncio.Semaphore(max_concurrent)
        clob_url = self._config.clob_base_url

        async def fetch_one(market: ResearchMarket) -> list[PriceSnapshot]:
            async with semaphore:
                snapshots = []
                for token_id in market.token_ids[:1]:  # YES token only
                    try:
                        # Resolved markets: only 12h+ granularity available
                        fidelity = 60 if market.status == "active" else 720
                        data = await self._get(
                            f"{clob_url}/prices-history",
                            params={
                                "market": token_id,
                                "interval": "max",
                                "fidelity": fidelity,
                            },
                        )
                        history = data.get("history", data) if isinstance(data, dict) else data
                        if isinstance(history, list):
                            for point in history:
                                ts = point.get("t", point.get("timestamp"))
                                price = point.get("p", point.get("price"))
                                if ts is not None and price is not None:
                                    timestamp = (
                                        datetime.utcfromtimestamp(int(ts))
                                        if isinstance(ts, (int, float)) and ts > 1e9
                                        else self._parse_dt(ts) or datetime.utcnow()
                                    )
                                    snapshots.append(PriceSnapshot(
                                        market_id=market.market_id,
                                        token_id=token_id,
                                        timestamp=timestamp,
                                        price=float(price),
                                    ))
                    except Exception as e:
                        logger.debug(
                            "price_fetch_error",
                            market_id=market.market_id,
                            error=str(e),
                        )
                return snapshots

        tasks = [fetch_one(m) for m in markets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_snapshots.extend(result)

        self._prices = all_snapshots
        logger.info("prices_fetched", total_snapshots=len(all_snapshots))
        return all_snapshots

    # ── Data Quality ────────────────────────────────────────────────

    def compute_quality_report(
        self,
        markets: list[ResearchMarket] | None = None,
        prices: list[PriceSnapshot] | None = None,
        min_volume: float = 1000.0,
        min_observations: int = 5,
    ) -> DataQualityReport:
        """Compute data quality metrics."""
        import numpy as np

        markets = markets or self._markets
        prices = prices or self._prices

        report = DataQualityReport(total_markets=len(markets))

        # Category counts
        for m in markets:
            cat = m.category.value
            report.markets_by_category[cat] = report.markets_by_category.get(cat, 0) + 1

        # Year counts
        for m in markets:
            if m.creation_date:
                yr = m.creation_date.year
                report.markets_by_year[yr] = report.markets_by_year.get(yr, 0) + 1

        # Status counts
        for m in markets:
            report.markets_by_status[m.status] = report.markets_by_status.get(m.status, 0) + 1

        # Price observations per market
        obs_per_market: dict[str, int] = {}
        for p in prices:
            obs_per_market[p.market_id] = obs_per_market.get(p.market_id, 0) + 1
        report.total_price_observations = len(prices)

        # Exclusions
        for m in markets:
            obs = obs_per_market.get(m.market_id, 0)
            if m.total_volume < min_volume:
                report.markets_excluded_low_volume += 1
            if obs < min_observations:
                report.markets_excluded_few_observations += 1

        # Volume distribution
        volumes = [m.total_volume for m in markets if m.total_volume > 0]
        if volumes:
            arr = np.array(volumes)
            report.volume_distribution = {
                "min": float(np.min(arr)),
                "p25": float(np.percentile(arr, 25)),
                "median": float(np.median(arr)),
                "p75": float(np.percentile(arr, 75)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
            }

        # Lifetime distribution
        lifetimes = [m.total_lifetime_days for m in markets if m.total_lifetime_days > 0]
        if lifetimes:
            arr = np.array(lifetimes)
            report.lifetime_distribution = {
                "min": float(np.min(arr)),
                "p25": float(np.percentile(arr, 25)),
                "median": float(np.median(arr)),
                "p75": float(np.percentile(arr, 75)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
            }

        # Observations per market distribution
        obs_vals = list(obs_per_market.values())
        if obs_vals:
            arr = np.array(obs_vals)
            report.observations_per_market = {
                "min": float(np.min(arr)),
                "p25": float(np.percentile(arr, 25)),
                "median": float(np.median(arr)),
                "p75": float(np.percentile(arr, 75)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
            }

        return report

    def filter_markets(
        self,
        markets: list[ResearchMarket] | None = None,
        prices: list[PriceSnapshot] | None = None,
        min_volume: float = 1000.0,
        min_observations: int = 5,
    ) -> tuple[list[ResearchMarket], list[PriceSnapshot]]:
        """Filter out low-quality markets."""
        markets = markets or self._markets
        prices = prices or self._prices

        obs_per_market: dict[str, int] = {}
        for p in prices:
            obs_per_market[p.market_id] = obs_per_market.get(p.market_id, 0) + 1

        valid_ids = set()
        for m in markets:
            if m.total_volume >= min_volume and obs_per_market.get(m.market_id, 0) >= min_observations:
                valid_ids.add(m.market_id)

        filtered_markets = [m for m in markets if m.market_id in valid_ids]
        filtered_prices = [p for p in prices if p.market_id in valid_ids]

        logger.info(
            "markets_filtered",
            total=len(markets),
            retained=len(filtered_markets),
            excluded=len(markets) - len(filtered_markets),
        )
        return filtered_markets, filtered_prices

    # ── Serialization ───────────────────────────────────────────────

    async def save_to_db(
        self,
        markets: list[ResearchMarket] | None = None,
        prices: list[PriceSnapshot] | None = None,
    ) -> None:
        """Save collected data to SQLite."""
        import aiosqlite

        markets = markets or self._markets
        prices = prices or self._prices

        async with aiosqlite.connect(self._db_path) as conn:
            await conn.executescript(_RESEARCH_SCHEMA)

            for m in markets:
                await conn.execute(
                    """INSERT OR REPLACE INTO research_markets
                       (market_id, condition_id, question, slug, category, creation_date,
                        end_date, resolution_date, outcome, total_volume, liquidity,
                        token_ids, neg_risk, event_id, status, total_lifetime_days)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        m.market_id, m.condition_id, m.question, m.slug,
                        m.category.value,
                        m.creation_date.isoformat() if m.creation_date else None,
                        m.end_date.isoformat() if m.end_date else None,
                        m.resolution_date.isoformat() if m.resolution_date else None,
                        m.outcome, m.total_volume, m.liquidity,
                        ",".join(m.token_ids), m.neg_risk, m.event_id,
                        m.status, m.total_lifetime_days,
                    ),
                )

            # Batch insert prices
            batch_size = 500
            for i in range(0, len(prices), batch_size):
                batch = prices[i:i + batch_size]
                await conn.executemany(
                    """INSERT OR IGNORE INTO price_snapshots
                       (market_id, token_id, timestamp, price, volume_bucket)
                       VALUES (?, ?, ?, ?, ?)""",
                    [
                        (p.market_id, p.token_id, p.timestamp.isoformat(), p.price, p.volume_bucket)
                        for p in batch
                    ],
                )

            await conn.commit()
        logger.info("data_saved", markets=len(markets), prices=len(prices))

    async def load_from_db(self) -> tuple[list[ResearchMarket], list[PriceSnapshot]]:
        """Load previously collected data from SQLite."""
        import aiosqlite

        markets: list[ResearchMarket] = []
        prices: list[PriceSnapshot] = []

        async with aiosqlite.connect(self._db_path) as conn:
            conn.row_factory = aiosqlite.Row

            cursor = await conn.execute("SELECT * FROM research_markets")
            rows = await cursor.fetchall()
            for r in rows:
                try:
                    markets.append(ResearchMarket(
                        market_id=r["market_id"],
                        condition_id=r["condition_id"],
                        question=r["question"],
                        slug=r["slug"],
                        category=MarketCategory(r["category"]),
                        creation_date=self._parse_dt(r["creation_date"]),
                        end_date=self._parse_dt(r["end_date"]),
                        resolution_date=self._parse_dt(r["resolution_date"]),
                        outcome=r["outcome"] or "",
                        total_volume=r["total_volume"],
                        liquidity=r["liquidity"],
                        token_ids=r["token_ids"].split(",") if r["token_ids"] else [],
                        neg_risk=bool(r["neg_risk"]),
                        event_id=r["event_id"] or "",
                        status=r["status"] or "",
                        total_lifetime_days=r["total_lifetime_days"] or 0.0,
                    ))
                except Exception as e:
                    logger.warning("load_market_error", error=str(e))

            cursor = await conn.execute("SELECT * FROM price_snapshots")
            rows = await cursor.fetchall()
            for r in rows:
                try:
                    prices.append(PriceSnapshot(
                        market_id=r["market_id"],
                        token_id=r["token_id"],
                        timestamp=datetime.fromisoformat(r["timestamp"]),
                        price=r["price"],
                        volume_bucket=r["volume_bucket"],
                    ))
                except Exception as e:
                    logger.warning("load_price_error", error=str(e))

        self._markets = markets
        self._prices = prices
        logger.info("data_loaded", markets=len(markets), prices=len(prices))
        return markets, prices

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _parse_dt(val: Any) -> datetime | None:
        """Parse various datetime formats into a naive UTC datetime."""
        if val is None:
            return None
        if isinstance(val, datetime):
            # Strip timezone info to keep everything naive-UTC
            return val.replace(tzinfo=None)
        if isinstance(val, (int, float)):
            if val > 1e12:
                return datetime.utcfromtimestamp(val / 1000)
            if val > 1e9:
                return datetime.utcfromtimestamp(val)
            return None
        if isinstance(val, str):
            val = val.strip()
            if not val:
                return None
            # Strip timezone suffixes so we always get naive datetimes
            import re as _re
            clean = val.replace("Z", "")
            # Remove trailing timezone offset like +00:00, +00, -05:30, etc.
            clean = _re.sub(r"[+-]\d{2}(:\d{2})?$", "", clean)
            for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.strptime(clean, fmt)
                except ValueError:
                    continue
            try:
                return datetime.fromisoformat(clean)
            except ValueError:
                return None
        return None

    async def run(self) -> tuple[list[ResearchMarket], list[PriceSnapshot], DataQualityReport]:
        """Full data collection pipeline."""
        logger.info("data_collection_starting")

        markets = await self.fetch_all_markets(max_markets=2000)
        logger.info("market_collection_done", total=len(markets))

        # Only fetch prices for markets that have token IDs
        with_tokens = [m for m in markets if m.token_ids]
        logger.info("markets_with_tokens", count=len(with_tokens))

        prices = await self.fetch_price_history(with_tokens)

        # Quality report before filtering
        report = self.compute_quality_report(markets, prices)
        logger.info(
            "data_quality",
            total_markets=report.total_markets,
            total_observations=report.total_price_observations,
            excluded_volume=report.markets_excluded_low_volume,
            excluded_obs=report.markets_excluded_few_observations,
            categories=report.markets_by_category,
        )

        # Save raw data
        await self.save_to_db(markets, prices)

        return markets, prices, report


_RESEARCH_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_markets (
    market_id TEXT PRIMARY KEY,
    condition_id TEXT,
    question TEXT,
    slug TEXT,
    category TEXT,
    creation_date TEXT,
    end_date TEXT,
    resolution_date TEXT,
    outcome TEXT,
    total_volume REAL DEFAULT 0,
    liquidity REAL DEFAULT 0,
    token_ids TEXT,
    neg_risk INTEGER DEFAULT 0,
    event_id TEXT,
    status TEXT,
    total_lifetime_days REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS price_snapshots (
    market_id TEXT,
    token_id TEXT,
    timestamp TEXT,
    price REAL,
    volume_bucket REAL DEFAULT 0,
    PRIMARY KEY (market_id, token_id, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_prices_market ON price_snapshots(market_id);
CREATE INDEX IF NOT EXISTS idx_prices_timestamp ON price_snapshots(timestamp);
CREATE INDEX IF NOT EXISTS idx_markets_status ON research_markets(status);
CREATE INDEX IF NOT EXISTS idx_markets_category ON research_markets(category);
CREATE INDEX IF NOT EXISTS idx_markets_event ON research_markets(event_id);
"""
