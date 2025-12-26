from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class PriceBar:
    """Simple OHLCV bar with status flag.

    status follows:
    - 1: intra-period data (not complete)
    - 2: latest live bar
    - 3: finalized bar (safe for state updates)
    """

    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    status: int


def normalize_prices(df):
    """Ensure price DataFrame has the expected schema and sorted index."""
    required = {"ts", "open", "high", "low", "close", "volume", "status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {missing}")
    out = df.copy()
    out = out.sort_values("ts").reset_index(drop=True)
    return out

