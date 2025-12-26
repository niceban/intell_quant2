from __future__ import annotations

from collections import deque

import pandas as pd

from ..config import STATUS_FINAL

DEFAULT_SHORT = 10
DEFAULT_LONG = 50
DEFAULT_AMA = 10


def compute_batch(
    df: pd.DataFrame,
    price_col: str = "close",
    short: int = DEFAULT_SHORT,
    long: int = DEFAULT_LONG,
    ama: int = DEFAULT_AMA,
) -> pd.DataFrame:
    out = df.copy()
    ma1 = out[price_col].rolling(short, min_periods=1).mean()
    ma2 = out[price_col].rolling(long, min_periods=1).mean()
    dma_val = ma1 - ma2
    out["dma"] = dma_val
    out["ama"] = dma_val.rolling(ama, min_periods=1).mean()
    return out


def compute_stream(
    df: pd.DataFrame,
    price_col: str = "close",
    short: int = DEFAULT_SHORT,
    long: int = DEFAULT_LONG,
    ama: int = DEFAULT_AMA,
) -> pd.DataFrame:
    short_buf = deque(maxlen=short)
    long_buf = deque(maxlen=long)
    ama_buf = deque(maxlen=ama)

    last_dma = float("nan")
    last_ama = float("nan")
    records = []

    for _, row in df.iterrows():
        do_update = bool(row.get("update_flag", row.get("status", STATUS_FINAL) >= STATUS_FINAL))

        price = float(row[price_col])
        tmp_short = deque(short_buf, maxlen=short)
        tmp_long = deque(long_buf, maxlen=long)
        tmp_ama = deque(ama_buf, maxlen=ama)

        tmp_short.append(price)
        tmp_long.append(price)
        ma1 = sum(tmp_short) / len(tmp_short)
        ma2 = sum(tmp_long) / len(tmp_long)
        tmp_dma = ma1 - ma2

        tmp_ama.append(tmp_dma)
        tmp_ama_val = sum(tmp_ama) / len(tmp_ama)

        if do_update:
            short_buf = tmp_short
            long_buf = tmp_long
            ama_buf = tmp_ama
            last_dma = tmp_dma
            last_ama = tmp_ama_val
        else:
            last_dma = tmp_dma
            last_ama = tmp_ama_val

        records.append({"dma": last_dma, "ama": last_ama})

    dma_df = pd.DataFrame(records)
    return pd.concat([df.reset_index(drop=True), dma_df], axis=1)
