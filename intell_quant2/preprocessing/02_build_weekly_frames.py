"""Create weekly aggregated bars and per-day weekly snapshots from mock daily CSVs."""

from __future__ import annotations

import glob
import os

import duckdb
import pandas as pd

from src.aggregators import daily_to_weekly, weekly_snapshots_for_backtest

DDB_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "duckdb")
INPUT_GLOB = os.path.join(DDB_DIR, "*.duckdb")


def main():
    os.makedirs(DDB_DIR, exist_ok=True)
    for path in glob.glob(INPUT_GLOB):
        symbol = os.path.basename(path).split(".")[0]
        con = duckdb.connect(path)
        daily = con.execute("SELECT * FROM daily ORDER BY ts").df()
        con.close()
        daily["ts"] = pd.to_datetime(daily["ts"]).dt.normalize()

        weekly = daily_to_weekly(daily)
        weekly_per_day = weekly_snapshots_for_backtest(daily)

        con = duckdb.connect(path)
        con.register("weekly_df", weekly)
        con.execute(
            "CREATE TABLE IF NOT EXISTS weekly (ts DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE, status INTEGER)"
        )
        con.execute("DELETE FROM weekly")
        con.execute("INSERT INTO weekly SELECT DATE(ts) AS ts, * EXCLUDE(ts) FROM weekly_df")

        con.register("weekly_snap_df", weekly_per_day)
        con.execute(
            "CREATE TABLE IF NOT EXISTS weekly_snapshots (ts DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE, status INTEGER)"
        )
        con.execute("DELETE FROM weekly_snapshots")
        con.execute("INSERT INTO weekly_snapshots SELECT DATE(ts) AS ts, * EXCLUDE(ts) FROM weekly_snap_df")
        con.close()

        print(f"built weekly frames for {symbol}: {path} (tables=weekly, weekly_snapshots)")


if __name__ == "__main__":
    main()
