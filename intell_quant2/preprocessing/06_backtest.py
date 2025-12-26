"""Run backtests using duckdb snapshots and plotly outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtest import run_backtests
from src.config import INSTRUMENTS


def main() -> None:
    results, summary = run_backtests(
        symbols=INSTRUMENTS,
        duckdb_dir=Path("outputs/duckdb"),
        output_dir=Path("outputs/backtests"),
        ignore_valid=True,
        enabled_buy_rules=["A", "B", "C"],
        enabled_sell_rules=["S2"],
    )
    if results:
        print("\n回测完成，指标摘要：")
        print(summary[["symbol", "total_return", "cagr", "max_drawdown", "win_rate", "fee_total", "trades"]].to_string(index=False))

        # 汇总亏损交易 TOP5
        all_trades = pd.concat([pd.DataFrame([t.to_dict() for t in res.trades]) for res in results if res.trades], ignore_index=True)
        if not all_trades.empty:
            losers = all_trades.sort_values("pnl").head(5)
            print("\n最差交易（按 pnl 排序）：")
            print(losers[["symbol", "entry_ts", "exit_ts", "entry_rule", "exit_rule", "pnl", "pnl_pct"]].to_string(index=False))
    else:
        print("没有可用的回测结果（可能缺少 duckdb 数据或指标表）。")


if __name__ == "__main__":
    main()
