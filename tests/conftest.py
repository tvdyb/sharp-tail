"""Shared fixtures for polymir tests."""

from __future__ import annotations

import pytest
import pytest_asyncio

from polymir.db import Database


@pytest_asyncio.fixture
async def db(tmp_path):
    """Provide a fresh in-memory-like database for each test."""
    db_path = str(tmp_path / "test.db")
    async with Database(db_path) as database:
        yield database
