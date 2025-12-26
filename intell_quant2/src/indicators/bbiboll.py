from __future__ import annotations

from collections import deque
from math import sqrt
from typing import Tuple

import pandas as pd

from ..config import STATUS_FINAL


DEFAULT_MA_WINDOWS = (3, 6, 12, 24)
DEFAULT_STD_WINDOW = 10
DEFAULT_STD_MULTIPLIER = 3.0


def compute_batch(
    df: pd.DataFrame,
    price_col: str = "close",
    ma_windows: Tuple[int, int, int, int] = DEFAULT_MA_WINDOWS,
    std_window: int = DEFAULT_STD_WINDOW,
    std_multiplier: float = DEFAULT_STD_MULTIPLIER,
) -> pd.DataFrame:
    out = df.copy()
    ma_values = []
    for w in ma_windows:
        out[f"ma{w}"] = out[price_col].rolling(w, min_periods=1).mean()
        ma_values.append(out[f"ma{w}"])
    out["bbiboll"] = sum(ma_values) / len(ma_values)
    out["bbiboll_std"] = out["bbiboll"].rolling(std_window, min_periods=1).std(ddof=0)
    out["bbiboll_upr"] = out["bbiboll"] + std_multiplier * out["bbiboll_std"]
    out["bbiboll_dwn"] = out["bbiboll"] - std_multiplier * out["bbiboll_std"]
    return out


def compute_stream(
    df: pd.DataFrame,
    price_col: str = "close",
    ma_windows: Tuple[int, int, int, int] = DEFAULT_MA_WINDOWS,
    std_window: int = DEFAULT_STD_WINDOW,
    std_multiplier: float = DEFAULT_STD_MULTIPLIER,
) -> pd.DataFrame:
    ma_buffers = {w: deque(maxlen=w) for w in ma_windows}
    ma_sums = {w: 0.0 for w in ma_windows}
    bb_buffer = deque(maxlen=std_window)
    bb_sum = 0.0
    bb_sumsq = 0.0

    last_bbiboll = float("nan")
    last_std = float("nan")
    records = []

    for _, row in df.iterrows():
        price = float(row[price_col])
        do_update = bool(row.get("update_flag", row.get("status", STATUS_FINAL) >= STATUS_FINAL))

        # 临时计算当日 ma/BBIBOLL/std
        tmp_ma_vals = []
        tmp_ma_buffers = {w: deque(ma_buffers[w], maxlen=w) for w in ma_windows}
        tmp_ma_sums = dict(ma_sums)
        for w in ma_windows:
            buf = tmp_ma_buffers[w]
            if len(buf) == w:
                tmp_ma_sums[w] -= buf[0]
            buf.append(price)
            tmp_ma_sums[w] += price
            tmp_ma_vals.append(tmp_ma_sums[w] / len(buf))
        tmp_bbiboll = sum(tmp_ma_vals) / len(tmp_ma_vals)

        tmp_bb_buffer = deque(bb_buffer, maxlen=std_window)
        tmp_bb_buffer.append(tmp_bbiboll)
        tmp_std_series = pd.Series(tmp_bb_buffer)
        tmp_std = float(tmp_std_series.std(ddof=0))
        tmp_bbiboll = float(tmp_bbiboll)

        # 仅在允许更新时落地到缓存
        if do_update:
            for w in ma_windows:
                ma_buffers[w] = tmp_ma_buffers[w]
                ma_sums[w] = tmp_ma_sums[w]
            bb_buffer = tmp_bb_buffer
            last_bbiboll = tmp_bbiboll
            last_std = tmp_std
        else:
            last_bbiboll = tmp_bbiboll
            last_std = tmp_std

        records.append(
            {
                "bbiboll": last_bbiboll,
                "bbiboll_std": last_std,
                "bbiboll_upr": last_bbiboll + std_multiplier * last_std,
                "bbiboll_dwn": last_bbiboll - std_multiplier * last_std,
            }
        )

    bb_df = pd.DataFrame(records)
    return pd.concat([df.reset_index(drop=True), bb_df], axis=1)
