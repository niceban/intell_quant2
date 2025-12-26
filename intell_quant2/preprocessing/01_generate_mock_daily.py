"""Fetch real daily OHLCV (前复权) for all instruments.

- Primary source: Yahoo Finance via yfinance (auto_adjust 前复权)
- HK fallback: akshare(stock_hk_hist, qfq) when Yahoo returns empty
- Marks last bar status=2, others=3
- Outputs CSVs to outputs/daily_<symbol>.csv
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Dict

import pandas as pd
import duckdb
import yfinance as yf

from src.config import INSTRUMENTS

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
DDB_DIR = os.path.join(OUTPUT_DIR, "duckdb")
# Map to Yahoo tickers (HK codes need suffix .HK)
TICKER_MAP: Dict[str, str] = {
    "07226": "07226.HK",
    "07233": "07233.HK",
    "07234": "07234.HK",
    "FAS": "FAS",
    "FAZ": "FAZ",
    "SOXL": "SOXL",
    "SOXS": "SOXS",
    "TECL": "TECL",
    "TECS": "TECS",
    "TQQQ": "TQQQ",
    "SQQQ": "SQQQ",
}

# 全量拉取：从 2000-01-01 到今天
START_DATE = "2000-01-01"
# end 是右开区间，取到明天以覆盖最新一日
END_DATE = (datetime.utcnow().date() + timedelta(days=1)).isoformat()


def _set_status_flags(df: pd.DataFrame) -> pd.DataFrame:
    status = [3] * len(df)
    if status:
        status[-1] = 2
    df["status"] = status
    return df


def _normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dtypes are consistent before storage."""
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"]).dt.normalize()
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["ts", "open", "high", "low", "close"]).sort_values("ts")
    df["status"] = pd.Series(df["status"], dtype="int64")
    return df[["ts", "open", "high", "low", "close", "volume", "status"]]


def fetch_from_yahoo(symbol: str, ticker: str) -> pd.DataFrame | None:
    """Primary source: Yahoo Finance. Returns None when empty."""
    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
    # yfinance may return MultiIndex columns when group_by='ticker'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    out = (
        df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        .reset_index()
        .rename(columns={"Date": "ts"})
    )
    status = [3] * len(out)
    status[-1] = 2
    out["status"] = status
    required = ["ts", "open", "high", "low", "close", "volume", "status"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        return None
    return _normalize_types(out)


def fetch_from_akshare_hk(symbol: str) -> pd.DataFrame:
    """HK daily bars via akshare; try hist (东财) then daily (新浪) as fallback."""
    try:
        import akshare as ak
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("akshare is required for HK fallback; please install akshare") from exc

    ak_symbol = symbol.replace(".HK", "").zfill(5)
    start = START_DATE.replace("-", "")
    end = END_DATE.replace("-", "")
    providers = [
        ("daily", lambda: ak.stock_hk_daily(symbol=ak_symbol, adjust="qfq")),  # 新浪，常规可访问
        ("hist", lambda: ak.stock_hk_hist(symbol=ak_symbol, period="daily", start_date=start, end_date=end, adjust="qfq")),
    ]
    last_error = None
    for name, fn in providers:
        try:
            df = fn()
            if df is None or df.empty:
                raise RuntimeError("empty dataframe")
            rename_map = {
                "日期": "ts",
                "date": "ts",
                "开盘": "open",
                "open": "open",
                "最高": "high",
                "high": "high",
                "最低": "low",
                "low": "low",
                "收盘": "close",
                "close": "close",
                "成交量": "volume",
                "volume": "volume",
            }
            df = df.rename(columns=rename_map)
            required = ["ts", "open", "high", "low", "close", "volume"]
            missing = set(required) - set(df.columns)
            if missing:
                raise RuntimeError(f"akshare hk columns missing: {missing}")

            out = df[required].copy()
            out = _set_status_flags(out)
            out = _normalize_types(out)
            last_ts = out["ts"].iloc[-1].date()
            print(f"[ok] akshare {name} {symbol} rows={len(out)} last={last_ts}")
            return out[["ts", "open", "high", "low", "close", "volume", "status"]]
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[warn] akshare {name} failed for {symbol}: {exc}")
            continue

    raise RuntimeError(f"no data from akshare for {symbol}: {last_error}")


def fetch_one(symbol: str) -> pd.DataFrame:
    ticker = TICKER_MAP.get(symbol, symbol)
    is_hk = ticker.endswith(".HK") or symbol.isdigit()

    if is_hk:
        try:
            return fetch_from_akshare_hk(symbol)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] akshare failed for {symbol}, fallback to yahoo: {exc}")
        # fallback to yahoo if akshare unavailable
        yahoo_df = fetch_from_yahoo(symbol, ticker)
        if yahoo_df is not None:
            return yahoo_df
        raise RuntimeError(f"no data for {symbol} ({ticker}) from akshare/yahoo")

    yahoo_df = fetch_from_yahoo(symbol, ticker)
    if yahoo_df is not None:
        return yahoo_df

    raise RuntimeError(f"no data for {symbol} ({ticker}) from akshare/yahoo")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DDB_DIR, exist_ok=True)
    for sym in INSTRUMENTS:
        try:
            df = fetch_one(sym)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {sym}: {exc}")
            continue
        ddb_path = os.path.join(DDB_DIR, f"{sym}.duckdb")
        con = duckdb.connect(ddb_path)
        con.register("df", df)
        con.execute(
            "CREATE TABLE IF NOT EXISTS daily (ts DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE, status INTEGER)"
        )
        con.execute("DELETE FROM daily")
        con.execute("INSERT INTO daily SELECT * FROM df")
        con.close()
        print(f"[ok] {sym} rows={len(df)} saved to {ddb_path} (table=daily)")


if __name__ == "__main__":
    main()
