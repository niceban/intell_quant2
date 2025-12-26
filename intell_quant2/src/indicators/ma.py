from __future__ import annotations

from collections import deque
from typing import Iterable, List

import pandas as pd

from ..config import STATUS_FINAL


DEFAULT_WINDOWS = (5, 10, 20, 30, 60, 120, 250)


def _compute_ma_series(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def compute_batch(df: pd.DataFrame, price_col: str = "close", windows: Iterable[int] = DEFAULT_WINDOWS) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        out[f"ma{w}"] = _compute_ma_series(out[price_col], w)
    return out


def compute_stream(df: pd.DataFrame, price_col: str = "close", windows: Iterable[int] = DEFAULT_WINDOWS) -> pd.DataFrame:
    buffers = {w: deque(maxlen=w) for w in windows}
    sums = {w: 0.0 for w in windows}
    last_values: List[float] = [float("nan")] * len(windows)
    records = []

    for _, row in df.iterrows():
        price = float(row[price_col])
        do_update = bool(row.get("update_flag", row.get("status", STATUS_FINAL) >= STATUS_FINAL))

        # 临时插入当前价计算当日值
        tmp_values = []
        for i, w in enumerate(windows):
            buf = deque(buffers[w], maxlen=w)
            s = sums[w]
            if len(buf) == w:
                s -= buf[0]
            buf.append(price)
            s += price
            tmp_values.append(s / len(buf))

        # 仅在允许更新时落地到缓存
        if do_update:
            for i, w in enumerate(windows):
                buf = buffers[w]
                prev_len = len(buf)
                if prev_len == w:
                    sums[w] -= buf[0]
                buf.append(price)
                sums[w] += price
                last_values[i] = sums[w] / len(buf)
        else:
            last_values = tmp_values

        records.append({f"ma{w}": last_values[i] for i, w in enumerate(windows)})

    ma_df = pd.DataFrame(records)
    return pd.concat([df.reset_index(drop=True), ma_df], axis=1)
