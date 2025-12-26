from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import duckdb
import pandas as pd

from .backtest import load_symbol_frames

INDICATORS: Sequence[str] = (
    "delta_low_bbiboll_dwn",
    "bbiboll",
    "bbiboll_std",
    "bbiboll_upr",
    "bbiboll_dwn",
    "skdj_k",
    "skdj_d",
    "macd_dif",
    "macd_dea",
    "macd_bar",
    "dma",
    "ama",
)

CROSS_PAIRS: Sequence[tuple[str, str]] = (
    ("ma10", "ma20"),
    ("ma20", "ma30"),
    ("ma30", "ma60"),
    ("ma60", "ma120"),
    ("ma120", "ma250"),
)

EXTRA_CROSS_PAIRS: Sequence[tuple[str, str]] = (
    ("macd_dif", "macd_dea"),
    ("skdj_k", "skdj_d"),
    ("dma", "ama"),
)


def _sign(v: float, tol: float = 1e-12) -> int:
    if v > tol:
        return 1
    if v < -tol:
        return -1
    return 0


@dataclass
class IndicatorFeatures:
    direction: List[int]
    turn: List[int]


def _annotate_indicator(df: pd.DataFrame, value_col: str, status_col: str = "status") -> IndicatorFeatures:
    direction_vals: List[int] = []
    turn_vals: List[int] = []

    last_confirm_val: Optional[float] = None
    last_state: Optional[int] = None  # 1 for rising, -1 for falling

    for _, row in df.iterrows():
        val = float(row[value_col])
        status = int(row[status_col])

        direction = 0
        if last_confirm_val is not None:
            direction = _sign(val - last_confirm_val)

        turn_flag = 0
        if direction != 0 and last_state is not None and direction != last_state:
            turn_flag = direction

        direction_vals.append(direction)
        turn_vals.append(turn_flag)

        if direction != 0:
            last_state = direction
        if status >= 2:
            last_confirm_val = val

    return IndicatorFeatures(
        direction=direction_vals,
        turn=turn_vals,
    )


def _annotate_cross_days(df: pd.DataFrame, fast_col: str, slow_col: str) -> List[int]:
    """Cross encoding: up-cross day=+1 then +2, down-cross day=-1 then -2, otherwise 0."""
    values: List[int] = []
    prev_sign: Optional[int] = None
    state: int = 0  # 0 none, 1/-1 cross today, 2/-2 after cross

    for _, row in df.iterrows():
        diff = float(row[fast_col]) - float(row[slow_col])
        sign = _sign(diff)

        if prev_sign is not None and sign != 0 and sign != prev_sign:
            state = 1 if sign > 0 else -1
        else:
            if state == 1:
                state = 2
            elif state == -1:
                state = -2
            # keep current state when no new cross

        values.append(state if sign != 0 or state != 0 else 0)

        if sign != 0:
            prev_sign = sign

    return values


def _build_frame_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Compute feature set for a single frequency (daily or weekly)."""
    work = df.copy().sort_values("ts").reset_index(drop=True)
    work["delta_low_bbiboll_dwn"] = work["low"] - work["bbiboll_dwn"]

    out = pd.DataFrame({"ts": pd.to_datetime(work["ts"]).dt.date})

    for ind in INDICATORS:
        feats = _annotate_indicator(work, ind, status_col="status")
        out[f"{prefix}{ind}_dir"] = feats.direction
        out[f"{prefix}{ind}_turn"] = feats.turn

    for fast, slow in CROSS_PAIRS:
        out[f"{prefix}cross_{fast}_{slow}_days"] = _annotate_cross_days(work, fast, slow)
    for fast, slow in EXTRA_CROSS_PAIRS:
        out[f"{prefix}cross_{fast}_{slow}_days"] = _annotate_cross_days(work, fast, slow)

    return out


def _output_columns(prefix: str) -> List[str]:
    cols: List[str] = []
    for ind in INDICATORS:
        cols.extend(
            [
                f"{prefix}{ind}_dir",
                f"{prefix}{ind}_turn",
            ]
        )
    for fast, slow in CROSS_PAIRS:
        cols.append(f"{prefix}cross_{fast}_{slow}_days")
    for fast, slow in EXTRA_CROSS_PAIRS:
        cols.append(f"{prefix}cross_{fast}_{slow}_days")
    return cols


def build_feature_table(symbol: str, duckdb_dir: Path, table_name: str = "feature_env") -> pd.DataFrame:
    daily, weekly = load_symbol_frames(symbol, duckdb_dir)

    daily_feat = _build_frame_features(daily, prefix="d_")
    daily_feat["close"] = daily["close"].reset_index(drop=True)
    weekly_feat = _build_frame_features(weekly, prefix="w_")

    merged = pd.merge(daily_feat, weekly_feat, on="ts", how="inner")
    merged = merged.sort_values("ts").reset_index(drop=True)

    # Ensure column order
    cols = ["ts", "close"] + _output_columns("d_") + _output_columns("w_")
    merged = merged[cols]

    path = Path(duckdb_dir) / f"{symbol}.duckdb"
    con = duckdb.connect(str(path))
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM merged")
    con.close()
    return merged


def run_all(duckdb_dir: str | Path = Path("outputs/duckdb"), table_name: str = "feature_env") -> dict[str, pd.DataFrame]:
    duckdb_dir = Path(duckdb_dir)
    outputs: dict[str, pd.DataFrame] = {}
    for db_path in sorted(duckdb_dir.glob("*.duckdb")):
        symbol = db_path.stem
        outputs[symbol] = build_feature_table(symbol, duckdb_dir, table_name=table_name)
    return outputs


if __name__ == "__main__":
    run_all()
