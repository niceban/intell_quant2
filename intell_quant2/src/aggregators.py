from __future__ import annotations

import pandas as pd

from .config import STATUS_FINAL, STATUS_INTRA, STATUS_LATEST
from .data_models import normalize_prices


def daily_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily bars into weekly bars.

    The last (most recent) week is marked as status=2 to reflect that it may be
    incomplete; all other completed weeks carry status=3.
    """
    df = normalize_prices(daily)
    df["week"] = df["ts"].dt.to_period("W-FRI")

    agg = df.groupby("week").agg(
        ts=("ts", "max"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    agg = agg.reset_index(drop=True)
    agg["status"] = STATUS_FINAL
    if not agg.empty:
        agg.loc[agg.index[-1], "status"] = STATUS_LATEST
    return agg


def daily_to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily bars into monthly bars.

    The last (most recent) month is marked as status=2 to reflect that it may be
    incomplete; all other completed months carry status=3.
    """
    df = normalize_prices(daily)
    df["month"] = df["ts"].dt.to_period("M")

    agg = df.groupby("month").agg(
        ts=("ts", "max"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    agg = agg.reset_index(drop=True)
    agg["status"] = STATUS_FINAL
    if not agg.empty:
        agg.loc[agg.index[-1], "status"] = STATUS_LATEST
    return agg


def weekly_snapshots_for_backtest(daily: pd.DataFrame) -> pd.DataFrame:
    """Generate per-day weekly snapshots with status flags.

    Each daily row gets an accompanying weekly aggregate of its current week.
    - status=1 inside the week (not the last available day)
    - status=3 on the final day of a completed week (typically Friday)
    - status=2 on the most recent day if the week has not closed
    """
    df = normalize_prices(daily)
    df["week"] = df["ts"].dt.to_period("W-FRI")
    snapshots = []
    last_index = df.index[-1]

    for _, group in df.groupby("week"):
        group = group.sort_values("ts")
        status = [STATUS_INTRA] * len(group)
        last_pos = len(group) - 1
        status[last_pos] = STATUS_FINAL
        if group.index[last_pos] == last_index:
            status[last_pos] = STATUS_LATEST

        week_open = pd.Series([group["open"].iloc[0]] * len(group), index=group.index)
        week_high = group["high"].cummax()
        week_low = group["low"].cummin()
        week_close = group["close"]
        week_volume = group["volume"].cumsum()

        for i, (idx, row) in enumerate(group.iterrows()):
            snapshots.append(
                {
                    "ts": row["ts"],
                    "open": week_open.iloc[i],
                    "high": week_high.iloc[i],
                    "low": week_low.iloc[i],
                    "close": week_close.iloc[i],
                    "volume": float(week_volume.iloc[i]),
                    "status": status[i],
                }
            )

    result = pd.DataFrame(snapshots).sort_values("ts").reset_index(drop=True)
    return result


def monthly_snapshots_for_backtest(daily: pd.DataFrame) -> pd.DataFrame:
    """Generate per-day monthly snapshots with status flags.

    Each daily row gets an accompanying monthly aggregate of its current month.
    - status=1 inside the month (not the last available day)
    - status=3 on the final day of a completed month
    - status=2 on the most recent day if the month has not closed
    """
    df = normalize_prices(daily)
    df["month"] = df["ts"].dt.to_period("M")
    snapshots = []
    last_index = df.index[-1]

    for _, group in df.groupby("month"):
        group = group.sort_values("ts")
        status = [STATUS_INTRA] * len(group)
        last_pos = len(group) - 1
        status[last_pos] = STATUS_FINAL
        if group.index[last_pos] == last_index:
            status[last_pos] = STATUS_LATEST

        month_open = pd.Series([group["open"].iloc[0]] * len(group), index=group.index)
        month_high = group["high"].cummax()
        month_low = group["low"].cummin()
        month_close = group["close"]
        month_volume = group["volume"].cumsum()

        for i, (_, row) in enumerate(group.iterrows()):
            snapshots.append(
                {
                    "ts": row["ts"],
                    "open": month_open.iloc[i],
                    "high": month_high.iloc[i],
                    "low": month_low.iloc[i],
                    "close": month_close.iloc[i],
                    "volume": float(month_volume.iloc[i]),
                    "status": status[i],
                }
            )

    result = pd.DataFrame(snapshots).sort_values("ts").reset_index(drop=True)
    return result
