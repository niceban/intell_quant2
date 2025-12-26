from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from ..config import INSTRUMENTS
from ..backtest import annotate_trends, build_base_dataframe, load_symbol_frames  # type: ignore


DEFAULT_DUCKDB_DIR = Path("outputs/duckdb")
DEFAULT_OUTPUT_DIR = Path("outputs/bbiboll_analysis")


@dataclass
class Band:
    symbol: str
    band_id: int
    dir: int  # 1=up, -1=down
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    length: int
    pct_change: float
    mean_range: float
    mean_volume: float

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["start_ts"] = self.start_ts.isoformat()
        data["end_ts"] = self.end_ts.isoformat()
        return data


@dataclass
class Trade:
    symbol: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: float
    entry_rule: str
    exit_rule: str
    pnl: float
    pnl_pct: float
    fees: float
    hold_days: int

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["entry_ts"] = self.entry_ts.isoformat()
        data["exit_ts"] = self.exit_ts.isoformat()
        return data


@dataclass
class BacktestResult:
    symbol: str
    trades: List[Trade]
    equity: pd.DataFrame
    bands: List[Band]
    metrics: Dict[str, float]
    base: pd.DataFrame


def _calc_metrics(symbol: str, trades: List[Trade], equity: pd.DataFrame, initial_cash: float) -> Dict[str, float]:
    if equity.empty:
        return {"symbol": symbol, "final_value": initial_cash}

    final_value = float(equity["equity"].iloc[-1])
    start_ts = equity["ts"].iloc[0]
    end_ts = equity["ts"].iloc[-1]
    days = max((end_ts - start_ts).days, 1)
    cagr = (final_value / initial_cash) ** (365 / days) - 1

    eq = equity["equity"]
    roll_max = eq.cummax()
    drawdown = eq / roll_max - 1
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    win_rate = len(wins) / len(trades) if trades else 0.0
    avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0.0
    profit_factor = (sum(t.pnl for t in wins) / abs(sum(t.pnl for t in losses))) if losses else (float("inf") if wins else 0.0)

    hold_days = sum(t.hold_days for t in trades)
    exposure = hold_days / days if days > 0 else 0.0
    fee_total = sum(t.fees for t in trades)
    turnover = sum(abs(t.entry_price * t.qty) + abs(t.exit_price * t.qty) for t in trades) / initial_cash if trades else 0.0

    return {
        "symbol": symbol,
        "final_value": final_value,
        "total_return": final_value / initial_cash - 1,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "exposure": exposure,
        "turnover": turnover,
        "fee_total": fee_total,
        "trades": len(trades),
    }


def _detect_bands(df: pd.DataFrame, min_len: int = 5, min_pct: float = 0.02, min_range: float = 0.003) -> Tuple[pd.DataFrame, List[Band]]:
    """Identify bbiboll-based bands using turn points."""
    df_sorted = df.sort_values("ts").reset_index(drop=True)
    bands: List[Band] = []
    band_ids = [-1] * len(df_sorted)

    last_dir: Optional[int] = None
    start_idx: Optional[int] = None
    band_counter = 0

    for idx, row in df_sorted.iterrows():
        turn = int(row.get("bbiboll_turn", 0))
        if last_dir is None:
            if turn != 0:
                last_dir = turn
                start_idx = idx
            continue

        if turn != 0 and turn != last_dir:
            end_idx = idx - 1
            if start_idx is not None:
                band_counter += 1
                band = _make_band(df_sorted, start_idx, end_idx, band_counter, last_dir, min_len, min_pct, min_range)
                if band:
                    bands.append(band)
                    for fill_idx in range(start_idx, end_idx + 1):
                        band_ids[fill_idx] = band.band_id
            start_idx = idx
            last_dir = turn

    # finalize last band
    if start_idx is not None and last_dir is not None:
        end_idx = len(df_sorted) - 1
        band_counter += 1
        band = _make_band(df_sorted, start_idx, end_idx, band_counter, last_dir, min_len, min_pct, min_range)
        if band:
            bands.append(band)
            for fill_idx in range(start_idx, end_idx + 1):
                band_ids[fill_idx] = band.band_id

    df_sorted["band_id"] = band_ids
    df_sorted["band_dir"] = [0] * len(df_sorted)
    if bands:
        band_map = {b.band_id: b.dir for b in bands}
        df_sorted["band_dir"] = [band_map.get(bid, 0) if bid is not None else 0 for bid in df_sorted["band_id"]]

    return df_sorted, bands


def _make_band(df: pd.DataFrame, start_idx: int, end_idx: int, band_id: int, dir_val: int, min_len: int, min_pct: float, min_range: float) -> Optional[Band]:
    window = df.iloc[start_idx : end_idx + 1]
    length = len(window)
    if length < min_len:
        return None

    start_price = float(window["close"].iloc[0])
    end_price = float(window["close"].iloc[-1])
    pct_change = (end_price / start_price) - 1 if start_price != 0 else 0.0
    if abs(pct_change) < min_pct:
        return None

    ranges = (window["bbiboll_upr"] - window["bbiboll_dwn"]) / window["close"].replace(0, pd.NA)
    mean_range = float(ranges.mean())
    if pd.notna(mean_range) and mean_range < min_range:
        return None

    mean_volume = float(window["volume"].mean()) if "volume" in window else 0.0
    return Band(
        symbol="",
        band_id=band_id,
        dir=dir_val,
        start_ts=pd.to_datetime(window["ts"].iloc[0]),
        end_ts=pd.to_datetime(window["ts"].iloc[-1]),
        length=length,
        pct_change=pct_change,
        mean_range=mean_range if pd.notna(mean_range) else 0.0,
        mean_volume=mean_volume,
    )


def _simulate_strategy(base: pd.DataFrame, bands: List[Band], initial_cash: float = 100_000.0, fee_rate: float = 0.001) -> Tuple[List[Trade], pd.DataFrame]:
    """Run bbiboll-band-gated strategy."""
    df = base.copy()
    cash = initial_cash
    qty = 0.0
    entry_price: Optional[float] = None
    entry_ts: Optional[pd.Timestamp] = None
    entry_rule: Optional[str] = None
    entry_fee: float = 0.0
    entry_notional: float = 0.0
    entry_idx: Optional[int] = None
    macd_up_after_entry = False

    low_touch_windows: List[Tuple[int, int]] = []
    skdj_up_windows: List[Tuple[int, int]] = []

    trades: List[Trade] = []
    equity_rows: List[Dict] = []
    peak_so_far = initial_cash

    band_map = {b.band_id: b for b in bands}
    epsilon = 0.005

    for idx, row in df.iterrows():
        ts = pd.to_datetime(row["ts"])
        price = float(row["close"])
        band_id = row.get("band_id", -1)
        band_dir = band_map.get(band_id, Band("", 0, 0, ts, ts, 0, 0.0, 0.0, 0.0)).dir if band_id in band_map else 0
        valid = bool(row.get("valid", False))

        # Maintain windows
        if row["low"] <= row["bbiboll_dwn"] * (1 + epsilon):
            low_touch_windows.append((idx, idx + 2))
        low_touch_windows = [(s, e) for s, e in low_touch_windows if e >= idx]

        skdj_turn_up = row.get("skdj_k_turn", 0) == 1
        macd_turn_up = row.get("macd_dif_turn", 0) == 1
        macd_turn_down = row.get("macd_dif_turn", 0) == -1
        skdj_d_turn_down = row.get("skdj_d_turn", 0) == -1
        skdj_k_dir_down = row.get("skdj_k_dir", 0) == -1
        macd_bar_dir_down = row.get("macd_bar_dir", 0) == -1
        macd_dir_down = row.get("macd_dif_dir", 0) == -1
        skdj_d_dir_down = row.get("skdj_d_dir", 0) == -1

        # Link skdj up to low-touch windows (for rule2)
        if skdj_turn_up and any(s <= idx <= e for s, e in low_touch_windows):
            skdj_up_windows.append((idx, idx + 2))
        skdj_up_windows = [(s, e) for s, e in skdj_up_windows if e >= idx]

        # Band exit if long while band not up
        if qty > 0 and band_dir != 1:
            cash, qty, entry_price, entry_ts, entry_rule, entry_fee, entry_notional, entry_idx, macd_up_after_entry, trades = _exit_trade(
                trades, ts, price, qty, entry_price, entry_ts, entry_rule, entry_fee, entry_notional, entry_idx, fee_rate, "band_exit", cash
            )

        # Forced sells
        if qty > 0:
            if macd_dir_down or skdj_d_dir_down:
                cash, qty, entry_price, entry_ts, entry_rule, entry_fee, entry_notional, entry_idx, macd_up_after_entry, trades = _exit_trade(
                    trades, ts, price, qty, entry_price, entry_ts, entry_rule, entry_fee, entry_notional, entry_idx, fee_rate, "force_down", cash
                )
            elif macd_up_after_entry and macd_turn_down:
                cash, qty, entry_price, entry_ts, entry_rule, entry_fee, entry_notional, entry_idx, macd_up_after_entry, trades = _exit_trade(
                    trades, ts, price, qty, entry_price, entry_ts, entry_rule, entry_fee, entry_notional, entry_idx, fee_rate, "macd_turn_down", cash
                )
            elif all(row.get(name, 0) == 1 for name in ["bbiboll_dir", "bbiboll_upr_dir", "bbiboll_std_dir", "bbiboll_dwn_dir"]) and macd_turn_down:
                cash, qty, entry_price, entry_ts, entry_rule, entry_fee, entry_notional, entry_idx, macd_up_after_entry, trades = _exit_trade(
                    trades, ts, price, qty, entry_price, entry_ts, entry_rule, entry_fee, entry_notional, entry_idx, fee_rate, "bbiboll_all_up_macd_down", cash
                )
            elif skdj_k_dir_down and macd_bar_dir_down:
                cash, qty, entry_price, entry_ts, entry_rule, entry_fee, entry_notional, entry_idx, macd_up_after_entry, trades = _exit_trade(
                    trades, ts, price, qty, entry_price, entry_ts, entry_rule, entry_fee, entry_notional, entry_idx, fee_rate, "momentum_sync_down", cash
                )

        # Buy logic
        if qty == 0 and valid and band_dir == 1:
            triggered_rule: Optional[str] = None
            # Rule A: low touch + skdj_k_turn_up, macd not turning up same day
            if skdj_turn_up and any(s <= idx <= e for s, e in low_touch_windows) and not macd_turn_up:
                triggered_rule = "A_low_touch_skdj_up"
            # Rule B: skdj_k_turn_up after low touch, then macd_turn_up within 3 days
            elif macd_turn_up and any(s <= idx <= e for s, e in skdj_up_windows):
                triggered_rule = "B_macd_confirms"
            else:
                # Rule C: momentum blocks align + skdj_k_turn_up
                macd_block = (row.get("macd_dif_dir", 0) == 1) or (row.get("macd_bar_dir", 0) == 1)
                dma_block = (row.get("dma_dir", 0) == 1) or (row.get("ama_dir", 0) == 1)
                if macd_block and dma_block and skdj_turn_up:
                    triggered_rule = "C_momentum_align"

            if triggered_rule:
                notional = cash / (1 + fee_rate)
                if notional > 0:
                    qty = notional / price
                    fee = notional * fee_rate
                    cash -= notional + fee
                    entry_price = price
                    entry_ts = ts
                    entry_rule = triggered_rule
                    entry_fee = fee
                    entry_notional = notional
                    entry_idx = idx
                    macd_up_after_entry = macd_turn_up
                    low_touch_windows = []
                    skdj_up_windows = []

        position_value = qty * price
        equity_val = cash + position_value
        peak_so_far = max(peak_so_far, equity_val)
        drawdown = equity_val / peak_so_far - 1 if peak_so_far else 0.0
        equity_rows.append({"ts": ts, "equity": equity_val, "drawdown": drawdown})

    equity_df = pd.DataFrame(equity_rows)
    return trades, equity_df


def _exit_trade(
    trades: List[Trade],
    ts: pd.Timestamp,
    price: float,
    qty: float,
    entry_price: Optional[float],
    entry_ts: Optional[pd.Timestamp],
    entry_rule: Optional[str],
    entry_fee: float,
    entry_notional: float,
    entry_idx: Optional[int],
    fee_rate: float,
    reason: str,
    cash: float,
):
    gross = qty * price
    fee = gross * fee_rate
    proceeds = gross - fee
    cash += proceeds
    trade_fees = fee + entry_fee
    pnl = proceeds - entry_notional - entry_fee
    pnl_pct = pnl / (entry_notional + entry_fee) if entry_notional > 0 else 0.0
    hold_days = (ts - entry_ts).days if entry_ts is not None else 0
    trades.append(
        Trade(
            symbol="",
            entry_ts=entry_ts if entry_ts is not None else ts,
            exit_ts=ts,
            entry_price=entry_price if entry_price is not None else price,
            exit_price=price,
            qty=qty,
            entry_rule=entry_rule or "",
            exit_rule=reason,
            pnl=pnl,
            pnl_pct=pnl_pct,
            fees=trade_fees,
            hold_days=hold_days,
        )
    )
    return cash, 0.0, None, None, None, 0.0, 0.0, None, False, trades


def run_symbol(symbol: str, duckdb_dir: Path = DEFAULT_DUCKDB_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR, initial_cash: float = 100_000.0, fee_rate: float = 0.001) -> BacktestResult:
    daily, weekly = load_symbol_frames(symbol, duckdb_dir)
    base = build_base_dataframe(daily, weekly)

    # Annotate bbiboll-specific directions/turns if not already present
    base = annotate_trends(base, ["bbiboll", "bbiboll_upr", "bbiboll_dwn", "bbiboll_std"])
    base_with_bands, bands = _detect_bands(base, min_len=5, min_pct=0.02, min_range=0.003)

    trades, equity = _simulate_strategy(base_with_bands, bands, initial_cash=initial_cash, fee_rate=fee_rate)

    # Attach symbol to bands/trades
    for b in bands:
        b.symbol = symbol
    for t in trades:
        t.symbol = symbol

    metrics = _calc_metrics(symbol, trades, equity, initial_cash)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([b.to_dict() for b in bands]).to_csv(output_dir / f"{symbol}_bands.csv", index=False)
    pd.DataFrame([t.to_dict() for t in trades]).to_csv(output_dir / f"{symbol}_trades.csv", index=False)
    equity.to_csv(output_dir / f"{symbol}_equity.csv", index=False)

    return BacktestResult(symbol=symbol, trades=trades, equity=equity, bands=bands, metrics=metrics, base=base_with_bands)


def run_all(symbols: Iterable[str] = INSTRUMENTS, duckdb_dir: Path = DEFAULT_DUCKDB_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR, initial_cash: float = 100_000.0, fee_rate: float = 0.001) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    for sym in symbols:
        try:
            result = run_symbol(sym, duckdb_dir=duckdb_dir, output_dir=output_dir, initial_cash=initial_cash, fee_rate=fee_rate)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {sym}: {exc}")
            continue
        metrics_rows.append(result.metrics)
    summary = pd.DataFrame(metrics_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    return summary
