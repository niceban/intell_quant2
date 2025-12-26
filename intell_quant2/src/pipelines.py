from __future__ import annotations

from typing import Dict, Literal

import pandas as pd

from .aggregators import daily_to_weekly, weekly_snapshots_for_backtest
from .indicators import bbiboll, dma, macd, ma, skdj

Mode = Literal["batch", "stream"]


def compute_indicators(df: pd.DataFrame, mode: Mode = "batch") -> pd.DataFrame:
    """Compute all indicators on the provided DataFrame.

    - batch: vectorized calculations (use for daily data and intraday refresh)
    - stream: status-aware rolling updates (use for weekly snapshots in backtests)
    """
    if mode == "batch":
        out = ma.compute_batch(df)
        out = bbiboll.compute_batch(out)
        out = skdj.compute_batch(out)
        out = macd.compute_batch(out)
        out = dma.compute_batch(out)
        return out

    out = ma.compute_stream(df)
    out = bbiboll.compute_stream(out)
    out = skdj.compute_stream(out)
    out = macd.compute_stream(out)
    out = dma.compute_stream(out)
    return out


def build_base_frames(daily: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Prepare daily and weekly frames for backtesting.

    Returns a dict with:
    - daily: raw daily bars (status should be 3)
    - weekly: aggregated weekly bars (status 3 except most recent 2)
    - weekly_per_day: weekly snapshots aligned to each daily row with status marks
    """
    daily_sorted = daily.sort_values("ts").reset_index(drop=True)
    weekly = daily_to_weekly(daily_sorted)
    weekly_per_day = weekly_snapshots_for_backtest(daily_sorted)
    return {"daily": daily_sorted, "weekly": weekly, "weekly_per_day": weekly_per_day}

