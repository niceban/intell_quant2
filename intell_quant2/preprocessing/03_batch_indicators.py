"""Batch compute all indicators on daily data (vectorized mode)."""

from __future__ import annotations

import glob
import os

import duckdb
import pandas as pd

from src.pipelines import compute_indicators

INPUT_GLOB = os.path.join(os.path.dirname(__file__), "..", "outputs", "duckdb", "*.duckdb")


def main():
    for path in glob.glob(INPUT_GLOB):
        symbol = os.path.basename(path).split(".")[0]
        con = duckdb.connect(path)
        daily = con.execute("SELECT * FROM daily ORDER BY ts").df()
        con.close()
        daily["ts"] = pd.to_datetime(daily["ts"]).dt.normalize()
        daily_ind = compute_indicators(daily, mode="batch")
        con = duckdb.connect(path)
        con.register("daily_ind_df", daily_ind)
        con.execute("CREATE OR REPLACE TABLE daily_ind AS SELECT DATE(ts) AS ts, * EXCLUDE(ts) FROM daily_ind_df")
        con.close()
        tail = daily_ind.tail(3)[["ts", "close", "ma5", "bbiboll", "skdj_k", "macd_bar", "dma"]]
        print(f"batch indicators {symbol}: stored in {path} (table=daily_ind)\n{tail}\n")


if __name__ == "__main__":
    main()
