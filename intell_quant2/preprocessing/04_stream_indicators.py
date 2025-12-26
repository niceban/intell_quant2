"""Stream (status-aware) compute indicators on weekly snapshots per day."""

from __future__ import annotations

import glob
import os

import duckdb
import pandas as pd

from src.config import STATUS_LATEST, STATUS_FINAL
from src.pipelines import compute_indicators

DDB_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "duckdb")
INPUT_GLOB = os.path.join(DDB_DIR, "*.duckdb")


def main():
    for path in glob.glob(INPUT_GLOB):
        symbol = os.path.basename(path).split(".")[0]
        con = duckdb.connect(path)
        weekly_snapshots = con.execute("SELECT * FROM weekly_snapshots ORDER BY ts").df()
        con.close()
        weekly_snapshots["ts"] = pd.to_datetime(weekly_snapshots["ts"]).dt.normalize()
        if weekly_snapshots.empty:
            continue
        # 仅在 status>=2 时推进指标状态，其他行承载当前状态
        mask_update = weekly_snapshots["status"] >= STATUS_LATEST
        weekly_ind = compute_indicators(weekly_snapshots.assign(update_flag=mask_update), mode="stream")
        con = duckdb.connect(path)
        con.register("weekly_ind_df", weekly_ind)
        con.execute("CREATE OR REPLACE TABLE weekly_ind_stream AS SELECT DATE(ts) AS ts, * EXCLUDE(ts) FROM weekly_ind_df")
        con.close()
        tail = weekly_ind.tail(6)[["ts", "status", "bbiboll", "skdj_k", "macd_bar", "dma"]]
        print(f"stream indicators {symbol}: stored in {path} (table=weekly_ind_stream)\n{tail}\n")


if __name__ == "__main__":
    main()
