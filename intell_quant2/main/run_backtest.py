"""Monthly rule backtest on A-share symbols filtered by market cap."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to sys.path to allow importing from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.aggregators import monthly_snapshots_for_backtest
from src.config import STATUS_FINAL, STATUS_INTRA, STATUS_LATEST
from src.pipelines import compute_indicators

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "monthly_rule"
MIN_MARKET_CAP = 20_000_000_000  # 20B
INITIAL_CASH = 100_000.0
REQUEST_INTERVAL = 5.0
BACKTEST_YEARS = 20
_LOCAL_DUCKDB_DIR_ENV = os.environ.get("LOCAL_DUCKDB_DIR")
if _LOCAL_DUCKDB_DIR_ENV:
    LOCAL_DUCKDB_DIR = Path(_LOCAL_DUCKDB_DIR_ENV)
else:
    # Default to data/A
    LOCAL_DUCKDB_DIR = PROJECT_ROOT / "data" / "A"
_LAST_REQUEST_TS = 0.0


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce")


def _throttle_request() -> None:
    global _LAST_REQUEST_TS
    now = time.time()
    elapsed = now - _LAST_REQUEST_TS
    if elapsed < REQUEST_INTERVAL:
        time.sleep(REQUEST_INTERVAL - elapsed)
    _LAST_REQUEST_TS = time.time()


def fetch_market_cap_symbols(min_market_cap: float) -> pd.DataFrame:
    import akshare as ak

    if LOCAL_DUCKDB_DIR and LOCAL_DUCKDB_DIR.exists():
        symbols = _list_local_symbols(LOCAL_DUCKDB_DIR)
        records = []
        for sym in symbols:
            cap = _calc_market_cap_from_local(sym)
            if cap is None:
                continue
            records.append({"symbol": sym, "name": "", "market_cap": cap})
        df = pd.DataFrame(records)
        if df.empty:
            raise ValueError("No market cap records from local duckdb")
        # Trust local files: since we already filtered during download, 
        # we return all local symbols regardless of calculated cap (to avoid HFQ calc errors)
        # filtered = df.loc[df["market_cap"] >= min_market_cap].copy()
        return df.sort_values("market_cap", ascending=False).reset_index(drop=True)

    _throttle_request()
    spot = ak.stock_zh_a_spot_em()
    code_col = "代码" if "代码" in spot.columns else None
    name_col = "名称" if "名称" in spot.columns else None
    cap_col = "总市值" if "总市值" in spot.columns else None
    if code_col is None or cap_col is None:
        raise ValueError(f"Missing expected columns in akshare spot data: {spot.columns}")

    cap_vals = _to_numeric(spot[cap_col])
    keep_cols = [code_col, cap_col]
    if name_col is not None:
        keep_cols.insert(1, name_col)
    filtered = spot.loc[cap_vals >= min_market_cap, keep_cols].copy()
    filtered["market_cap"] = _to_numeric(filtered[cap_col])
    filtered = filtered.rename(columns={code_col: "symbol", name_col: "name"})
    if "name" not in filtered.columns:
        filtered["name"] = ""
    filtered = filtered.sort_values("market_cap", ascending=False).reset_index(drop=True)
    return filtered[["symbol", "name", "market_cap"]]


def fetch_daily_bars(symbol: str) -> pd.DataFrame:
    import akshare as ak

    if LOCAL_DUCKDB_DIR and LOCAL_DUCKDB_DIR.exists():
        df = _load_daily_from_local_duckdb(symbol)
    else:
        _throttle_request()
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="hfq")
    if df is None or df.empty:
        raise ValueError(f"No daily data for {symbol}")

    rename_map = {
        "日期": "ts",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
    }
    df = df.rename(columns=rename_map)
    required = ["ts", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol} missing columns: {missing}")

    out = df[required].copy()
    out["ts"] = pd.to_datetime(out["ts"]).dt.normalize()
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=BACKTEST_YEARS)
    out = out.loc[out["ts"] >= cutoff].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No daily data in last {BACKTEST_YEARS} years for {symbol}")
    out = out.dropna(subset=["ts", "open", "high", "low", "close"]).sort_values("ts").reset_index(drop=True)
    out["status"] = STATUS_FINAL
    if not out.empty:
        out.loc[out.index[-1], "status"] = STATUS_LATEST
    return out


def _list_local_symbols(duckdb_dir: Path) -> List[str]:
    symbols = []
    # Support both .db and .duckdb extensions
    files = sorted(list(duckdb_dir.glob("*.db")) + list(duckdb_dir.glob("*.duckdb")))
    for path in files:
        stem = path.stem
        if stem.startswith(("sh", "sz", "bj")) and len(stem) >= 8:
            symbols.append(stem[2:8])
        elif len(stem) >= 6:
            symbols.append(stem[:6])
    return sorted(set(symbols))


def _load_daily_from_local_duckdb(symbol: str) -> pd.DataFrame:
    import duckdb

    sym = str(symbol)
    prefix = ""
    stock_code = sym
    
    if sym.startswith(("sh", "sz", "bj")):
        stock_code = sym[2:]
        prefix = sym[:2]
        # Try finding file with prefix
        potential_names = [f"{sym}.db", f"{sym}.duckdb"]
    else:
        # Try inferring prefix
        if sym.startswith(("6", "9")):
            prefix = "sh"
        elif sym.startswith(("0", "3")):
            prefix = "sz"
        else:
            prefix = "bj"
        potential_names = [f"{prefix}{sym}.db", f"{prefix}{sym}.duckdb"]

    path = None
    for name in potential_names:
        p = LOCAL_DUCKDB_DIR / name
        if p.exists():
            path = p
            break
    
    if path is None or not path.exists():
        raise FileNotFoundError(f"Local DuckDB not found for {symbol} in {LOCAL_DUCKDB_DIR}")

    con = duckdb.connect(str(path), read_only=True)
    df = con.execute(
        "SELECT date AS ts, open, high, low, close, volume, turnover FROM daily_data WHERE stock_code = ? ORDER BY date",
        [stock_code],
    ).df()
    con.close()
    return df


def _calc_market_cap_from_local(symbol: str) -> Optional[float]:
    import duckdb

    sym = str(symbol)
    prefix = ""
    stock_code = sym
    
    if sym.startswith(("sh", "sz", "bj")):
        stock_code = sym[2:]
        prefix = sym[:2]
        potential_names = [f"{sym}.db", f"{sym}.duckdb"]
    else:
        if sym.startswith(("6", "9")):
            prefix = "sh"
        elif sym.startswith(("0", "3")):
            prefix = "sz"
        else:
            prefix = "bj"
        potential_names = [f"{prefix}{sym}.db", f"{prefix}{sym}.duckdb"]

    path = None
    for name in potential_names:
        p = LOCAL_DUCKDB_DIR / name
        if p.exists():
            path = p
            break
            
    if path is None or not path.exists():
        return None

    con = duckdb.connect(str(path), read_only=True)
    row = con.execute(
        "SELECT close, volume, turnover FROM daily_data WHERE stock_code = ? ORDER BY date DESC LIMIT 1",
        [stock_code],
    ).fetchone()
    con.close()
    if row is None:
        return None
    close, volume, turnover = row
    if turnover is None or turnover <= 0:
        return None
    shares = float(volume) / (float(turnover) / 100.0)
    return float(close) * shares


def _calc_market_cap_from_daily(daily: pd.DataFrame) -> float:
    if "outstanding_share" not in daily.columns:
        raise ValueError("Daily data missing outstanding_share for market cap")
    latest = daily.dropna(subset=["close", "outstanding_share"])
    if latest.empty:
        raise ValueError("No valid rows to compute market cap")
    row = latest.iloc[-1]
    return float(row["close"]) * float(row["outstanding_share"])


def _calc_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    return float(drawdown.min())


def _calc_metrics(equity_df: pd.DataFrame, trades: List[Dict], initial_cash: float, hold_days: int) -> Dict:
    if equity_df.empty:
        return {
            "final_value": initial_cash,
            "total_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "volatility": 0.0,
            "downside_vol": 0.0,
            "trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_win": 0.0,
            "max_loss": 0.0,
            "expectancy": 0.0,
            "payoff_ratio": 0.0,
            "avg_hold_days": 0.0,
            "max_hold_days": 0.0,
            "trades_per_year": 0.0,
            "profit_factor": 0.0,
            "exposure": 0.0,
            "total_pnl": 0.0,
        }

    equity = equity_df["equity"]
    final_value = float(equity.iloc[-1])
    total_return = final_value / initial_cash - 1.0
    start_ts = equity_df["ts"].iloc[0]
    end_ts = equity_df["ts"].iloc[-1]
    days = max((end_ts - start_ts).days, 1)
    cagr = (final_value / initial_cash) ** (365 / days) - 1.0
    max_dd = _calc_drawdown(equity)

    daily_ret = equity.pct_change().fillna(0.0)
    daily_std = float(daily_ret.std())
    sharpe = float(np.sqrt(252) * daily_ret.mean() / daily_std) if daily_std > 0 else 0.0
    downside = daily_ret[daily_ret < 0]
    downside_std = float(downside.std())
    sortino = float(np.sqrt(252) * daily_ret.mean() / downside_std) if downside_std > 0 else 0.0
    volatility = float(np.sqrt(252) * daily_std) if daily_std > 0 else 0.0
    downside_vol = float(np.sqrt(252) * downside_std) if downside_std > 0 else 0.0
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_rate = len(wins) / len(pnls) if pnls else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    max_win = float(np.max(wins)) if wins else 0.0
    max_loss = float(np.min(losses)) if losses else 0.0
    expectancy = float(np.mean(pnls)) if pnls else 0.0
    payoff_ratio = float(avg_win / abs(avg_loss)) if avg_loss < 0 else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else (float("inf") if wins else 0.0)
    hold_lengths = []
    for t in trades:
        entry_ts = t.get("entry_ts")
        exit_ts = t.get("exit_ts")
        if entry_ts is None or exit_ts is None:
            continue
        hold_lengths.append((exit_ts - entry_ts).days)
    avg_hold_days = float(np.mean(hold_lengths)) if hold_lengths else 0.0
    max_hold_days = float(np.max(hold_lengths)) if hold_lengths else 0.0
    trades_per_year = float(len(trades) / (days / 365)) if days > 0 else 0.0
    exposure = hold_days / len(equity_df) if len(equity_df) > 0 else 0.0
    total_pnl = float(sum(pnls)) if pnls else 0.0

    return {
        "final_value": final_value,
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "volatility": volatility,
        "downside_vol": downside_vol,
        "trades": len(pnls),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_win": max_win,
        "max_loss": max_loss,
        "expectancy": expectancy,
        "payoff_ratio": payoff_ratio,
        "avg_hold_days": avg_hold_days,
        "max_hold_days": max_hold_days,
        "trades_per_year": trades_per_year,
        "profit_factor": profit_factor,
        "exposure": exposure,
        "total_pnl": total_pnl,
    }


def _trend_sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def run_monthly_rule(symbol: str, daily: pd.DataFrame) -> tuple[pd.DataFrame, List[Dict], int]:
    monthly_snapshots = monthly_snapshots_for_backtest(daily)
    monthly_ind = compute_indicators(
        monthly_snapshots.assign(update_flag=monthly_snapshots["status"] >= STATUS_LATEST),
        mode="stream",
    )
    monthly_ind = monthly_ind.rename(columns={"status": "m_status"})
    monthly_ind = monthly_ind.drop(columns=["open", "high", "low", "close", "volume"], errors="ignore")

    merged = pd.merge(daily, monthly_ind, on="ts", how="left")
    merged = merged.sort_values("ts").reset_index(drop=True)

    cash = INITIAL_CASH
    shares = 0.0
    holding = False
    current_trade: Optional[Dict] = None
    pending: Optional[Dict] = None
    last_confirmed_skdj_d: Optional[float] = None
    prev_confirmed_skdj_d: Optional[float] = None
    last_confirmed_macd_dea: Optional[float] = None
    prev_confirmed_macd_dea: Optional[float] = None
    last_confirmed_macd_dif: Optional[float] = None
    prev_confirmed_macd_dif: Optional[float] = None
    last_confirmed_dma: Optional[float] = None
    prev_confirmed_dma: Optional[float] = None
    last_confirmed_ama: Optional[float] = None
    prev_confirmed_ama: Optional[float] = None
    macd_dea_trend: Optional[int] = None
    seen_up_to_down = False
    seen_down_to_up = False
    hold_days = 0

    trades: List[Dict] = []
    equity_rows = []

    for i in range(len(merged)):
        row = merged.iloc[i]
        ts = row["ts"]
        open_px = float(row["open"])
        close_px = float(row["close"])

        if pending is not None:
            if pending["action"] == "buy" and not holding and np.isfinite(open_px) and open_px > 0:
                shares = cash / open_px
                cash = 0.0
                holding = True
                current_trade = {
                    "symbol": symbol,
                    "entry_ts": ts,
                    "entry_price": open_px,
                    "entry_signal_ts": pending["signal_ts"],
                    "entry_reason": pending["reason"],
                    "qty": shares,
                }
            elif pending["action"] == "sell" and holding and np.isfinite(open_px) and open_px > 0:
                cash = shares * open_px
                pnl = cash - (current_trade["qty"] * current_trade["entry_price"])
                current_trade.update(
                    {
                        "exit_ts": ts,
                        "exit_price": open_px,
                        "exit_signal_ts": pending["signal_ts"],
                        "exit_reason": pending["reason"],
                        "pnl": pnl,
                        "return": pnl / (current_trade["qty"] * current_trade["entry_price"]),
                    }
                )
                trades.append(current_trade)
                current_trade = None
                shares = 0.0
                holding = False
            pending = None

        if holding:
            hold_days += 1
        equity = cash + shares * close_px
        equity_rows.append(
            {"ts": ts, "equity": equity, "cash": cash, "position": 1 if holding else 0}
        )

        skdj_d = row.get("skdj_d")
        macd_dea = row.get("macd_dea")
        macd_dif = row.get("macd_dif")
        dma = row.get("dma")
        ama = row.get("ama")
        m_status = row.get("m_status")

        indicators_ready = all(pd.notna(v) for v in (skdj_d, macd_dea, macd_dif, dma, ama))
        if pending is None and indicators_ready:
            skdj_d = float(skdj_d)
            macd_dea = float(macd_dea)
            macd_dif = float(macd_dif)
            dma = float(dma)
            ama = float(ama)
            trading_allowed = seen_up_to_down and seen_down_to_up

            def rising_two(curr: float, prev: Optional[float], prev_prev: Optional[float]) -> bool:
                return prev is not None and prev_prev is not None and curr > prev and prev > prev_prev

            def turns_up(curr: float, prev: Optional[float], prev_prev: Optional[float]) -> bool:
                return prev is not None and prev_prev is not None and curr > prev and prev < prev_prev

            macd_dea_rising_two = rising_two(macd_dea, last_confirmed_macd_dea, prev_confirmed_macd_dea)
            ama_rising_two = rising_two(ama, last_confirmed_ama, prev_confirmed_ama)
            skdj_rising_two = rising_two(skdj_d, last_confirmed_skdj_d, prev_confirmed_skdj_d)
            skdj_turns_up = turns_up(skdj_d, last_confirmed_skdj_d, prev_confirmed_skdj_d)
            macd_dea_turns_up = turns_up(macd_dea, last_confirmed_macd_dea, prev_confirmed_macd_dea)
            buy_condition = ((macd_dea_rising_two or ama_rising_two) and skdj_turns_up) or (
                skdj_rising_two and macd_dea_turns_up
            )

            macd_dif_down = last_confirmed_macd_dif is not None and macd_dif < last_confirmed_macd_dif
            dma_down = last_confirmed_dma is not None and dma < last_confirmed_dma

            if not holding:
                if trading_allowed and buy_condition:
                    pending = {
                        "action": "buy",
                        "signal_ts": ts,
                        "reason": "buy_condition",
                    }
            else:
                entry_signal_ts = current_trade.get("entry_signal_ts") if current_trade else None
                if entry_signal_ts is not None:
                    entry_month = pd.Timestamp(entry_signal_ts).to_period("M")
                    current_month = pd.Timestamp(ts).to_period("M")
                else:
                    entry_month = None
                    current_month = None

                if entry_month is not None and current_month is not None and entry_month == current_month:
                    if not buy_condition and pending is None:
                        pending = {
                            "action": "sell",
                            "signal_ts": ts,
                            "reason": "buy_condition_invalidated",
                        }
                if pending is None and macd_dif_down and dma_down:
                    pending = {
                        "action": "sell",
                        "signal_ts": ts,
                        "reason": "macd_dif_dma_down",
                    }

        if pd.notna(m_status) and int(m_status) == STATUS_FINAL and indicators_ready:
            macd_dea_val = float(macd_dea)
            if last_confirmed_macd_dea is not None:
                macd_trend = _trend_sign(macd_dea_val - last_confirmed_macd_dea)
                if macd_dea_trend is not None:
                    if macd_dea_trend > 0 and macd_trend < 0:
                        seen_up_to_down = True
                    if macd_dea_trend < 0 and macd_trend > 0:
                        seen_down_to_up = True
                if macd_trend != 0:
                    macd_dea_trend = macd_trend
            prev_confirmed_skdj_d = last_confirmed_skdj_d
            prev_confirmed_macd_dea = last_confirmed_macd_dea
            prev_confirmed_macd_dif = last_confirmed_macd_dif
            prev_confirmed_dma = last_confirmed_dma
            prev_confirmed_ama = last_confirmed_ama
            last_confirmed_skdj_d = float(skdj_d)
            last_confirmed_macd_dea = macd_dea_val
            last_confirmed_macd_dif = float(macd_dif)
            last_confirmed_dma = float(dma)
            last_confirmed_ama = float(ama)

    equity_df = pd.DataFrame(equity_rows)
    return equity_df, trades, hold_days


def main():
    for key in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
    ):
        os.environ.pop(key, None)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols_df = fetch_market_cap_symbols(MIN_MARKET_CAP)
    has_market_cap = "market_cap" in symbols_df.columns
    summaries = []

    for _, row in symbols_df.iterrows():
        symbol = str(row["symbol"])
        name = row.get("name", "")
        daily = None
        if has_market_cap:
            market_cap = float(row["market_cap"])
        else:
            daily = fetch_daily_bars(symbol)
            market_cap = _calc_market_cap_from_daily(daily)
            if market_cap < MIN_MARKET_CAP:
                continue

        if daily is None:
            daily = fetch_daily_bars(symbol)

        equity_df, trades, hold_days = run_monthly_rule(symbol, daily)
        metrics = _calc_metrics(equity_df, trades, INITIAL_CASH, hold_days)
        metrics.update(
            {
                "symbol": symbol,
                "name": name,
                "market_cap": market_cap,
                "start_ts": daily["ts"].iloc[0],
                "end_ts": daily["ts"].iloc[-1],
            }
        )
        summaries.append(metrics)

        trades_df = pd.DataFrame(trades)
        equity_path = OUTPUT_DIR / f"{symbol}_equity.csv"
        trades_path = OUTPUT_DIR / f"{symbol}_trades.csv"
        equity_df.to_csv(equity_path, index=False)
        trades_df.to_csv(trades_path, index=False)

        print(
            f"[ok] {symbol} {name} trades={metrics['trades']} "
            f"total_return={metrics['total_return']:.2%}"
        )

    summary_df = pd.DataFrame(summaries).sort_values("market_cap", ascending=False)
    summary_path = OUTPUT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[done] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
