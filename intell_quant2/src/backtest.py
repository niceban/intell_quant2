from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import duckdb
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import INSTRUMENTS


FEE_RATE = 0.01
INITIAL_CASH = 100_000.0

BUY_RULE_LABELS = {
    "A": "A: Low<bbiboll_dwn后4日内skdj_k转上，且当日macd_dif未转上",
    "B": "B: Low<bbiboll_dwn后4日内skdj_k转上，再4日内macd_dif转上",
    "C": "C: 前一日满足MACD/DMA上升块，当日skdj_k转上",
}

SELL_RULE_LABELS = {
    "F1": "F1: macd_dif 转为下降 且 skdj_k、macd_bar 均下行",
    "F2": "F2: 强制 skdj_d 转为下降",
    "F3": "F3: bbiboll三项下降 或 macd_dea 转下 或 ama 转下",
    "S1": "S1: 买入后 macd_dif 曾转上，现转下",
    "S2": "S2: bbiboll全升，macd_dif 转下",
    "S3": "S3: skdj_k 下降 且 macd_bar 下降",
}


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
        # 确保时间戳可序列化
        data["entry_ts"] = self.entry_ts.isoformat()
        data["exit_ts"] = self.exit_ts.isoformat()
        return data


@dataclass
class BacktestResult:
    symbol: str
    trades: List[Trade]
    equity: pd.DataFrame
    metrics: Dict[str, float]
    base: pd.DataFrame


def _sign(v: float, tol: float = 1e-12) -> int:
    if v > tol:
        return 1
    if v < -tol:
        return -1
    return 0


def annotate_trend(df: pd.DataFrame, value_col: str, status_col: str = "status") -> pd.DataFrame:
    """为单个指标追加趋势标记：dir/turn/last_turn_dir/last_turn_ts。"""
    dirs: List[int] = []
    turns: List[int] = []
    last_turn_dirs: List[int] = []
    last_turn_ts: List[pd.Timestamp] = []

    last_confirm_val: Optional[float] = None
    prev_dir: Optional[int] = None
    latest_turn_dir = 0
    latest_turn_time: Optional[pd.Timestamp] = None

    for _, row in df.iterrows():
        ts = pd.to_datetime(row["ts"])
        val = float(row[value_col])
        status = int(row[status_col])

        direction = 0
        if last_confirm_val is not None:
            direction = _sign(val - last_confirm_val)

        turn = 0
        if direction != 0 and prev_dir is not None and direction != prev_dir:
            turn = direction
            latest_turn_dir = direction
            latest_turn_time = ts

        if direction != 0:
            prev_dir = direction

        if status >= 2:
            last_confirm_val = val

        dirs.append(direction)
        turns.append(turn)
        last_turn_dirs.append(latest_turn_dir)
        last_turn_ts.append(latest_turn_time)

    out = df.copy()
    out[f"{value_col}_dir"] = dirs
    out[f"{value_col}_turn"] = turns
    out[f"{value_col}_last_turn_dir"] = last_turn_dirs
    out[f"{value_col}_last_turn_ts"] = last_turn_ts
    return out


def annotate_trends(df: pd.DataFrame, columns: Iterable[str], status_col: str = "status", prefix: str | None = None) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out = annotate_trend(out, col, status_col=status_col)
    if prefix:
        rename_map = {c: f"{prefix}{c}" for c in out.columns if c != "ts"}
        out = out.rename(columns=rename_map)
    return out


def latest_group_turn_dir(row: pd.Series, indicator_names: Iterable[str], prefix: str = "", suffix_dir: str = "_last_turn_dir", suffix_ts: str = "_last_turn_ts") -> int:
    latest_ts: Optional[pd.Timestamp] = None
    latest_dir: int = 0
    for name in indicator_names:
        dir_val = row.get(f"{prefix}{name}{suffix_dir}", 0)
        ts_val = row.get(f"{prefix}{name}{suffix_ts}", None)
        if dir_val == 0 or pd.isna(ts_val):
            continue
        ts_val = pd.to_datetime(ts_val)
        if latest_ts is None or ts_val > latest_ts:
            latest_ts = ts_val
            latest_dir = int(dir_val)
    return latest_dir


def load_symbol_frames(symbol: str, duckdb_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(duckdb_dir) / f"{symbol}.duckdb"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    con = duckdb.connect(str(path))
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    required = {"daily_ind", "weekly_ind_stream"}
    missing = required - tables
    if missing:
        con.close()
        raise RuntimeError(f"{symbol} missing tables: {missing}")
    daily = con.execute("SELECT * FROM daily_ind ORDER BY ts").df()
    weekly = con.execute("SELECT * FROM weekly_ind_stream ORDER BY ts").df()
    con.close()

    daily["ts"] = pd.to_datetime(daily["ts"]).dt.normalize()
    weekly["ts"] = pd.to_datetime(weekly["ts"]).dt.normalize()
    return daily, weekly


def compute_valid_flag(row: pd.Series) -> bool:
    cond1 = sum(row[f"w_{name}_dir"] == 1 for name in ["bbiboll", "bbiboll_dwn", "bbiboll_upr"]) >= 2
    cond2 = all(row[f"w_{name}_dir"] == 1 for name in ["skdj_d", "macd_dif", "dma"])
    cond3 = (
        latest_group_turn_dir(row, ["skdj_d", "skdj_k"], prefix="w_") == 1
        and latest_group_turn_dir(row, ["macd_dif", "macd_bar"], prefix="w_") == 1
        and latest_group_turn_dir(row, ["dma", "ama"], prefix="w_") == 1
    )
    return bool(cond1 or cond2 or cond3)


def build_base_dataframe(daily: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    daily_cols = ["bbiboll", "bbiboll_upr", "bbiboll_dwn", "bbiboll_std", "skdj_k", "skdj_d", "macd_dif", "macd_dea", "macd_bar", "dma", "ama"]
    weekly_cols = ["bbiboll", "bbiboll_upr", "bbiboll_dwn", "skdj_k", "skdj_d", "macd_dif", "macd_dea", "macd_bar", "dma", "ama"]

    daily_ann = annotate_trends(daily, daily_cols, status_col="status", prefix=None)
    weekly_ann = annotate_trends(weekly, weekly_cols, status_col="status", prefix="w_")

    merged = pd.merge(daily_ann, weekly_ann, on="ts", how="inner")
    merged = merged.sort_values("ts").reset_index(drop=True)
    merged["valid"] = merged.apply(compute_valid_flag, axis=1)
    return merged


def _calc_trade_notional(entry_price: float, qty: float) -> float:
    return entry_price * qty


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
    profit_factor = (sum(t.pnl for t in wins) / abs(sum(t.pnl for t in losses))) if losses else float("inf") if wins else 0.0

    hold_days = sum(t.hold_days for t in trades)
    exposure = hold_days / days if days > 0 else 0.0
    fee_total = sum(t.fees for t in trades)
    turnover = sum(abs(_calc_trade_notional(t.entry_price, t.qty)) + abs(_calc_trade_notional(t.exit_price, t.qty)) for t in trades) / initial_cash if trades else 0.0

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


def _make_plot(symbol: str, df: pd.DataFrame, equity: pd.DataFrame, trades: List[Trade], output_path: Path) -> None:
    buys_ts = [t.entry_ts for t in trades]
    buys_px = [t.entry_price for t in trades]
    buys_text = [f"Buy {t.entry_rule}" for t in trades]

    sells_ts = [t.exit_ts for t in trades]
    sells_px = [t.exit_price for t in trades]
    sells_text = [f"Sell {t.exit_rule}" for t in trades]
    sell_colors = ["red" if t.pnl < 0 else "green" for t in trades]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.6, 0.4])

    fig.add_trace(go.Scatter(x=df["ts"], y=df["close"], mode="lines", name="Close"), row=1, col=1)
    if buys_ts:
        fig.add_trace(go.Scatter(x=buys_ts, y=buys_px, mode="markers+text", name="Buys", marker=dict(color="blue", size=9, symbol="triangle-up"), text=buys_text, textposition="top center"), row=1, col=1)
    if sells_ts:
        fig.add_trace(go.Scatter(x=sells_ts, y=sells_px, mode="markers+text", name="Sells", marker=dict(color=sell_colors, size=9, symbol="x"), text=sells_text, textposition="bottom center"), row=1, col=1)

    fig.add_trace(go.Scatter(x=equity["ts"], y=equity["equity"], mode="lines", name="Equity"), row=2, col=1)
    fig.add_trace(go.Scatter(x=equity["ts"], y=equity["drawdown"], mode="lines", name="Drawdown", fill="tozeroy"), row=2, col=1)

    # 图例：买入/卖出规则说明
    for code, desc in BUY_RULE_LABELS.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="blue", symbol="triangle-up"), name=f"买入 {desc}"), row=1, col=1)
    for code, desc in SELL_RULE_LABELS.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="red", symbol="x"), name=f"卖出 {desc}"), row=1, col=1)

    fig.update_layout(title=f"{symbol} Backtest (rules annotated)", legend_orientation="h", height=900)
    fig.write_html(output_path)


def backtest_symbol(
    symbol: str,
    duckdb_dir: str | Path = Path("outputs/duckdb"),
    initial_cash: float = INITIAL_CASH,
    fee_rate: float = FEE_RATE,
    ignore_valid: bool = False,
    enabled_buy_rules: Optional[Iterable[str]] = None,
    enabled_sell_rules: Optional[Iterable[str]] = None,
    preloaded_base: Optional[pd.DataFrame] = None,
    use_skip_filters: bool = True,
    enabled_forced_rules: Optional[Iterable[str]] = None,
) -> BacktestResult:
    if preloaded_base is not None:
        base = preloaded_base.copy()
    else:
        daily, weekly = load_symbol_frames(symbol, duckdb_dir)
        base = build_base_dataframe(daily, weekly)
    buy_set = set(enabled_buy_rules) if enabled_buy_rules is not None else {"A", "B", "C"}
    sell_set = set(enabled_sell_rules) if enabled_sell_rules is not None else set(SELL_RULE_LABELS.keys()) - {"F1", "F2", "F3"}
    forced_set = set(enabled_forced_rules) if enabled_forced_rules is not None else {"F1", "F2", "F3"}

    cash = initial_cash
    qty = 0.0
    entry_price: Optional[float] = None
    entry_ts: Optional[pd.Timestamp] = None
    entry_rule: Optional[str] = None
    entry_fee: float = 0.0
    entry_notional: float = 0.0
    entry_idx: Optional[int] = None
    macd_up_after_entry = False
    # 跳过标记
    daily_short_skip = False
    daily_long_skip = False
    weekly_short_skip = False

    pending_low: List[Tuple[int, int]] = []
    pending_skdj_up: List[Tuple[int, int]] = []
    trades: List[Trade] = []
    equity_rows: List[Dict] = []
    peak_so_far = initial_cash

    for idx, row in base.iterrows():
        ts = row["ts"]
        price = float(row["close"])

        # 更新窗口：Low < bbiboll_dwn
        if row["low"] < row["bbiboll_dwn"]:
            pending_low.append((idx, idx + 3))  # 含触发日共 4 天
        pending_low = [(s, e) for s, e in pending_low if e >= idx]

        skdj_turn_up = row["skdj_k_turn"] == 1
        macd_turn_up = row["macd_dif_turn"] == 1
        macd_turn_down = row["macd_dif_turn"] == -1
        skdj_d_turn_down = row["skdj_d_turn"] == -1
        skdj_k_dir_down = row["skdj_k_dir"] == -1
        macd_bar_dir_down = row["macd_bar_dir"] == -1
        skdj_k_turn_down = row["skdj_k_turn"] == -1
        macd_dea_turn_up = row["macd_dea_turn"] == 1
        dma_turn_up = row["dma_turn"] == 1
        macd_dea_turn_down = row["macd_dea_turn"] == -1
        ama_turn_down = row["ama_turn"] == -1
        bbiboll_all_down = row["bbiboll_dir"] == -1 and row["bbiboll_upr_dir"] == -1 and row["bbiboll_dwn_dir"] == -1

        # 周级
        w_skdj_k_dir_down = row.get("w_skdj_k_dir", 0) == -1
        w_skdj_k_turn_down = row.get("w_skdj_k_turn", 0) == -1
        w_skdj_d_turn_down = row.get("w_skdj_d_turn", 0) == -1
        w_skdj_d_dir_up_prev = False
        w_macd_dea_turn_up = row.get("w_macd_dea_turn", 0) == 1
        w_dma_turn_up = row.get("w_dma_turn", 0) == 1

        # 跳过逻辑：需要前一根信息
        if use_skip_filters and idx > 0:
            prev = base.iloc[idx - 1]
            # 日短线：上一根 skdj_k_dir=1 当前 turn=-1
            if prev["skdj_k_dir"] == 1 and skdj_k_turn_down:
                daily_short_skip = True
            # 清零：上一根 skdj_d_dir=1 且当前 skdj_d_turn=-1，或当日 macd_dea_turn_up，或 dma_turn_up
            if (prev["skdj_d_dir"] == 1 and skdj_d_turn_down) or macd_dea_turn_up or dma_turn_up:
                daily_short_skip = False

            # 日长线：上一根 macd_dif_dir=1 当前 macd_dif_turn=-1
            if prev["macd_dif_dir"] == 1 and macd_turn_down:
                daily_long_skip = True
            # 清零条件同上
            if (prev["skdj_d_dir"] == 1 and skdj_d_turn_down) or macd_dea_turn_up or dma_turn_up:
                daily_long_skip = False

            # 周短线：上一周 skdj_k_dir=1 当前周 skdj_k_turn=-1
            if prev.get("w_skdj_k_dir", 0) == 1 and w_skdj_k_turn_down:
                weekly_short_skip = True
            w_skdj_d_dir_up_prev = prev.get("w_skdj_d_dir", 0) == 1
            if (w_skdj_d_dir_up_prev and w_skdj_d_turn_down) or w_macd_dea_turn_up or w_dma_turn_up or macd_dea_turn_up or dma_turn_up:
                weekly_short_skip = False

        # 记录 skdj_k 转上事件（需 low 窗口内）
        if skdj_turn_up and any(s <= idx <= e for s, e in pending_low):
            pending_skdj_up.append((idx, idx + 3))
        pending_skdj_up = [(s, e) for s, e in pending_skdj_up if e >= idx]

        # 卖出检查
        if qty > 0:
            # 买入后 macd_dif 是否曾转上
            if macd_turn_up and entry_ts is not None and ts > entry_ts:
                macd_up_after_entry = True

            sell_reason: Optional[str] = None
            if "F1" in forced_set and macd_turn_down and skdj_k_dir_down and macd_bar_dir_down:
                sell_reason = "F1"
            elif "F2" in forced_set and skdj_d_turn_down:
                sell_reason = "F2"
            elif "F3" in forced_set and (bbiboll_all_down or macd_dea_turn_down or ama_turn_down):
                sell_reason = "F3"
            elif "S1" in sell_set and macd_up_after_entry and macd_turn_down:
                sell_reason = "S1"
            elif "S2" in sell_set and all(row[name] == 1 for name in ["bbiboll_dir", "bbiboll_upr_dir", "bbiboll_std_dir", "bbiboll_dwn_dir"]) and macd_turn_down:
                sell_reason = "S2"
            elif "S3" in sell_set and skdj_k_dir_down and macd_bar_dir_down:
                sell_reason = "S3"

            if sell_reason:
                gross = qty * price
                fee = gross * fee_rate
                proceeds = gross - fee
                cash += proceeds
                trade_fees = fee + entry_fee
                pnl = proceeds - entry_notional - entry_fee
                pnl_pct = pnl / (entry_notional + entry_fee) if entry_notional > 0 else 0.0
                hold_days = idx - entry_idx if entry_idx is not None else 0
                trades.append(
                    Trade(
                        symbol=symbol,
                        entry_ts=entry_ts if entry_ts is not None else ts,
                        exit_ts=ts,
                        entry_price=entry_price if entry_price is not None else price,
                        exit_price=price,
                        qty=qty,
                        entry_rule=entry_rule or "",
                        exit_rule=sell_reason,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        fees=trade_fees,
                        hold_days=hold_days,
                    )
                )
                qty = 0.0
                entry_price = None
                entry_ts = None
                entry_rule = None
                entry_fee = 0.0
                entry_notional = 0.0
                entry_idx = None
                macd_up_after_entry = False
                pending_low = []
                pending_skdj_up = []

        # 买入检查（仅空仓）
        skip_buy = (daily_short_skip or daily_long_skip or weekly_short_skip) if use_skip_filters else False

        if qty == 0 and (ignore_valid or row["valid"]) and not skip_buy:
            triggered_rule: Optional[str] = None
            if "A" in buy_set and skdj_turn_up and any(s <= idx <= e for s, e in pending_low) and not macd_turn_up:
                triggered_rule = "A"
            elif "B" in buy_set and macd_turn_up and any(s <= idx <= e for s, e in pending_skdj_up):
                triggered_rule = "B"
            else:
                # Rule C
                if "C" in buy_set and idx > 0 and skdj_turn_up:
                    prev = base.iloc[idx - 1]
                    macd_block = (
                        latest_group_turn_dir(prev, ["macd_dif", "macd_bar"], prefix="") == 1
                        or prev["macd_dif_dir"] == 1
                    )
                    dma_block = (
                        latest_group_turn_dir(prev, ["dma", "ama"], prefix="") == 1
                        or prev["ama_dir"] == 1
                    )
                    if macd_block and dma_block:
                        triggered_rule = "C"

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
                    macd_up_after_entry = macd_turn_up  # 若当日已转上则视为已发生
                    pending_low = []
                    pending_skdj_up = []
                    # 买入后强制触发日长线跳过（直到清零条件出现）
                    if use_skip_filters:
                        daily_long_skip = True

        position_value = qty * price
        equity_val = cash + position_value
        peak_so_far = max(peak_so_far, equity_val)
        drawdown = equity_val / peak_so_far - 1 if peak_so_far else 0.0
        equity_rows.append({"ts": ts, "equity": equity_val, "drawdown": drawdown})

    equity_df = pd.DataFrame(equity_rows)
    metrics = _calc_metrics(symbol, trades, equity_df, initial_cash)
    return BacktestResult(symbol=symbol, trades=trades, equity=equity_df, metrics=metrics, base=base)


def run_backtests(
    symbols: Iterable[str] = INSTRUMENTS,
    duckdb_dir: str | Path = Path("outputs/duckdb"),
    output_dir: str | Path = Path("outputs/backtests"),
    ignore_valid: bool = False,
    enabled_buy_rules: Optional[Iterable[str]] = None,
    enabled_sell_rules: Optional[Iterable[str]] = None,
    write_outputs: bool = True,
    preloaded_bases: Optional[Dict[str, pd.DataFrame]] = None,
    use_skip_filters: bool = True,
    enabled_forced_rules: Optional[Iterable[str]] = None,
) -> Tuple[List[BacktestResult], pd.DataFrame]:
    output_dir = Path(output_dir)
    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)

    results: List[BacktestResult] = []
    metrics_rows = []
    for sym in symbols:
        try:
            result = backtest_symbol(
                sym,
                duckdb_dir=duckdb_dir,
                ignore_valid=ignore_valid,
                enabled_buy_rules=enabled_buy_rules,
                enabled_sell_rules=enabled_sell_rules,
                preloaded_base=preloaded_bases.get(sym) if preloaded_bases else None,
                use_skip_filters=use_skip_filters,
                enabled_forced_rules=enabled_forced_rules,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {sym}: {exc}")
            continue
        results.append(result)
        metrics_rows.append(result.metrics)

        # 保存交易与图
        if write_outputs:
            trades_df = pd.DataFrame([t.to_dict() for t in result.trades])
            trades_path = output_dir / f"{sym}_trades.csv"
            trades_df.to_csv(trades_path, index=False)

            equity_path = output_dir / f"{sym}_equity.csv"
            result.equity.to_csv(equity_path, index=False)

            plot_path = output_dir / f"{sym}_backtest.html"
            _make_plot(sym, result.base, result.equity, result.trades, plot_path)

    summary_df = pd.DataFrame(metrics_rows)
    if write_outputs:
        summary_path = output_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
    return results, summary_df
