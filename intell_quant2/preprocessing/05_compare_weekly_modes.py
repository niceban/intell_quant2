"""Compare weekly indicators: batch vs stream on status>=2 snapshots.

For each symbol DuckDB file:
- Load weekly_snapshots and keep rows with status>=2 (已确认或最新).
- Batch compute indicators on these rows.
- Load weekly_ind_stream (streamed indicators per day).
- Align on ts and report max absolute diff per indicator.
"""

from __future__ import annotations

import glob
import os
from typing import List

import duckdb
import pandas as pd

from src.pipelines import compute_indicators

DDB_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "duckdb")
INPUT_GLOB = os.path.join(DDB_DIR, "*.duckdb")
TOL = 1e-6


def indicator_columns(df: pd.DataFrame) -> List[str]:
    skip = {"ts", "status", "open", "high", "low", "close", "volume"}
    return [c for c in df.columns if c not in skip]


def main():
    files = glob.glob(INPUT_GLOB)
    if not files:
        print("no duckdb files found under outputs/duckdb")
        return

    for path in files:
        symbol = os.path.basename(path).split(".")[0]
        con = duckdb.connect(path)
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        if "weekly_snapshots" not in tables or "weekly_ind_stream" not in tables:
            con.close()
            print(f"[skip] {symbol}: missing weekly_snapshots or weekly_ind_stream")
            continue

        weekly_snap = con.execute("SELECT * FROM weekly_snapshots ORDER BY ts").df()
        stream_df = con.execute("SELECT * FROM weekly_ind_stream ORDER BY ts").df()
        con.close()

        weekly_snap["ts"] = pd.to_datetime(weekly_snap["ts"]).dt.normalize()
        stream_df["ts"] = pd.to_datetime(stream_df["ts"]).dt.normalize()

        # Status>=2 snapshots (已确认 or 最新)
        snap_ge2 = weekly_snap[weekly_snap["status"] >= 2].reset_index(drop=True)
        if snap_ge2.empty:
            print(f"[skip] {symbol}: no weekly_snapshots with status>=2")
            continue

        batch_df = compute_indicators(snap_ge2, mode="batch")
        # Align stream to same timestamps
        stream_aligned = stream_df[stream_df["ts"].isin(batch_df["ts"])].reset_index(drop=True)
        batch_aligned = batch_df[batch_df["ts"].isin(stream_aligned["ts"])].reset_index(drop=True)

        if stream_aligned.empty or batch_aligned.empty:
            print(f"[warn] {symbol}: no overlapping dates between batch and stream")
            continue

        cols_batch = set(indicator_columns(batch_aligned))
        cols_stream = set(indicator_columns(stream_aligned))
        cols = sorted(cols_batch & cols_stream)
        if not cols:
            print(f"[warn] {symbol}: no indicator columns found")
            continue

        diffs = {}
        for col in cols:
            b = batch_aligned[col].astype(float)
            s = stream_aligned[col].astype(float)
            if len(b) != len(s):
                diffs[col] = float("inf")
                continue
            max_diff = (b - s).abs().max()
            diffs[col] = max_diff

        max_overall = max(diffs.values()) if diffs else 0.0
        status = "ok" if max_overall <= TOL else "diff"
        print(f"[{status}] {symbol} max_diff={max_overall}")
        if status == "diff":
            top = sorted(diffs.items(), key=lambda x: x[1], reverse=True)[:5]
            print("  top diffs:", top)


if __name__ == "__main__":
    main()
