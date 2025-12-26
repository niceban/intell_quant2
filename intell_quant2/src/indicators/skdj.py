from __future__ import annotations

from collections import deque

import pandas as pd

from ..config import STATUS_FINAL

DEFAULT_N = 9
DEFAULT_P1 = 3
DEFAULT_P2 = 3
DEFAULT_P3 = 5


def _sma_step(prev: float | None, x: float, n: int, m: int = 1) -> float:
    if prev is None:
        return x
    return (m * x + (n - m) * prev) / n


def _sma_series(series: pd.Series, n: int, m: int = 1) -> pd.Series:
    result = []
    prev = None
    for x in series:
        prev = _sma_step(prev, x, n, m)
        result.append(prev)
    return pd.Series(result, index=series.index)


def compute_batch(
    df: pd.DataFrame,
    n: int = DEFAULT_N,
    p1: int = DEFAULT_P1,
    p2: int = DEFAULT_P2,
    p3: int = DEFAULT_P3,
) -> pd.DataFrame:
    out = df.copy()
    lowv = out["low"].rolling(n, min_periods=1).min()
    highv = out["high"].rolling(n, min_periods=1).max()
    range_ = highv - lowv
    rsv_raw = (out["close"] - lowv) / range_
    rsv_raw = rsv_raw.replace([pd.NA, pd.NaT, float("inf"), -float("inf")], 0).fillna(0) * 100

    fastk = _sma_series(rsv_raw, p1, 1)
    k = _sma_series(fastk, p2, 1)
    d = _sma_series(k, p3, 1)

    out["skdj_rsv"] = fastk
    out["skdj_k"] = k
    out["skdj_d"] = d
    return out


def compute_stream(
    df: pd.DataFrame,
    n: int = DEFAULT_N,
    p1: int = DEFAULT_P1,
    p2: int = DEFAULT_P2,
    p3: int = DEFAULT_P3,
) -> pd.DataFrame:
    low_buf = deque(maxlen=n)
    high_buf = deque(maxlen=n)

    last_fastk = None
    last_k = None
    last_d = None
    records = []

    for _, row in df.iterrows():
        do_update = bool(row.get("update_flag", row.get("status", STATUS_FINAL) >= STATUS_FINAL))

        tmp_low = deque(low_buf, maxlen=n)
        tmp_high = deque(high_buf, maxlen=n)
        tmp_low.append(float(row["low"]))
        tmp_high.append(float(row["high"]))
        lowv = min(tmp_low)
        highv = max(tmp_high)
        range_v = highv - lowv
        raw_rsv = 0.0 if range_v == 0 else (float(row["close"]) - lowv) / range_v * 100

        tmp_fastk = _sma_step(last_fastk, raw_rsv, p1, 1)
        tmp_k = _sma_step(last_k, tmp_fastk, p2, 1)
        tmp_d = _sma_step(last_d, tmp_k, p3, 1)

        if do_update:
            low_buf = tmp_low
            high_buf = tmp_high
            last_fastk = tmp_fastk
            last_k = tmp_k
            last_d = tmp_d

        records.append({"skdj_rsv": tmp_fastk, "skdj_k": tmp_k, "skdj_d": tmp_d})

    skdj_df = pd.DataFrame(records)
    return pd.concat([df.reset_index(drop=True), skdj_df], axis=1)
