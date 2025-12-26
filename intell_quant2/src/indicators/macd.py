from __future__ import annotations

import pandas as pd

from ..config import STATUS_FINAL

DEFAULT_FAST = 12
DEFAULT_SLOW = 26
DEFAULT_SIGNAL = 9


def compute_batch(
    df: pd.DataFrame,
    price_col: str = "close",
    fast: int = DEFAULT_FAST,
    slow: int = DEFAULT_SLOW,
    signal: int = DEFAULT_SIGNAL,
) -> pd.DataFrame:
    out = df.copy()
    ema_fast = out[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = out[price_col].ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    bar = (dif - dea) * 2
    out["macd_dif"] = dif
    out["macd_dea"] = dea
    out["macd_bar"] = bar
    return out


def compute_stream(
    df: pd.DataFrame,
    price_col: str = "close",
    fast: int = DEFAULT_FAST,
    slow: int = DEFAULT_SLOW,
    signal: int = DEFAULT_SIGNAL,
) -> pd.DataFrame:
    alpha_fast = 2 / (fast + 1)
    alpha_slow = 2 / (slow + 1)
    alpha_signal = 2 / (signal + 1)

    ema_fast = None
    ema_slow = None
    dea = None

    last_dif = float("nan")
    last_dea = float("nan")
    last_bar = float("nan")
    records = []

    for _, row in df.iterrows():
        do_update = bool(row.get("update_flag", row.get("status", STATUS_FINAL) >= STATUS_FINAL))

        price = float(row[price_col])
        tmp_ema_fast = price if ema_fast is None else ema_fast * (1 - alpha_fast) + price * alpha_fast
        tmp_ema_slow = price if ema_slow is None else ema_slow * (1 - alpha_slow) + price * alpha_slow
        tmp_dif = tmp_ema_fast - tmp_ema_slow
        tmp_dea = tmp_dif if dea is None else dea * (1 - alpha_signal) + tmp_dif * alpha_signal
        tmp_bar = (tmp_dif - tmp_dea) * 2

        if do_update:
            ema_fast = tmp_ema_fast
            ema_slow = tmp_ema_slow
            dea = tmp_dea
            last_dif = tmp_dif
            last_dea = tmp_dea
            last_bar = tmp_bar
        else:
            last_dif = tmp_dif
            last_dea = tmp_dea
            last_bar = tmp_bar

        records.append({"macd_dif": last_dif, "macd_dea": last_dea, "macd_bar": last_bar})

    macd_df = pd.DataFrame(records)
    return pd.concat([df.reset_index(drop=True), macd_df], axis=1)
