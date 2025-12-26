"""遍历买/卖规则组合，寻找收益最佳的组合（平均总收益）。"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

from src.backtest import build_base_dataframe, load_symbol_frames, run_backtests
from src.config import INSTRUMENTS

BUY_RULES = ["A", "B", "C"]
SELL_RULES = ["S1", "S2", "S3"]
FORCED_RULES = ["F1", "F2", "F3"]


def iter_subsets(items: Iterable[str]) -> List[Tuple[str, ...]]:
    items = list(items)
    subsets: List[Tuple[str, ...]] = []
    for r in range(len(items) + 1):
        for comb in itertools.combinations(items, r):
            subsets.append(comb)
    return subsets


def main() -> None:
    # 预先加载并构建 base，避免重复读写 duckdb
    preloaded = {}
    for sym in INSTRUMENTS:
        try:
            daily, weekly = load_symbol_frames(sym, Path("outputs/duckdb"))
            preloaded[sym] = build_base_dataframe(daily, weekly)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip preload] {sym}: {exc}")
            continue

    output_dir = Path("outputs/backtests_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_path = output_dir / "rules_sweep_summary.csv"
    existing_records = []
    existing_keys = set()
    if existing_path.exists():
        try:
            df_exist = pd.read_csv(existing_path)
            existing_records = df_exist.to_dict("records")
            for r in existing_records:
                key = (r["buy_rules"], r["sell_rules"], r.get("forced_rules", "NONE"), r["skip_filters"])
                existing_keys.add(key)
            print(f"[resume] loaded {len(existing_records)} existing records, will skip duplicates")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to load existing summary: {exc}")

    records = []
    total_combos = (2 ** len(BUY_RULES) - 1) * (2 ** len(SELL_RULES) - 1) * (2 ** len(FORCED_RULES)) * 2  # 买非空、卖非空、强制可空、skip on/off
    idx = 0
    for buy_set in iter_subsets(BUY_RULES):
        if not buy_set:
            continue  # 无买点无意义
        for sell_set in iter_subsets(SELL_RULES):
            if not sell_set:
                continue  # 不考虑卖点为空
            for forced_set in iter_subsets(FORCED_RULES):
                for skip_flag in [True, False]:
                    idx += 1
                    key = (",".join(buy_set), ",".join(sell_set), ",".join(forced_set) if forced_set else "NONE", "on" if skip_flag else "off")
                    if key in existing_keys:
                        print(f"[skip existing] buy={key[0]}, sell={key[1]}, forced={key[2]}, skip={key[3]}")
                        continue
                    print(f"[{idx}/{total_combos}] buy={buy_set}, sell={sell_set}, forced={forced_set or 'NONE'}, skip={'ON' if skip_flag else 'OFF'}")
                    _, summary = run_backtests(
                        symbols=INSTRUMENTS,
                        duckdb_dir=Path("outputs/duckdb"),
                        output_dir=output_dir,
                        ignore_valid=True,
                        enabled_buy_rules=buy_set,
                        enabled_sell_rules=sell_set,
                        enabled_forced_rules=forced_set if forced_set else [],
                        write_outputs=False,
                        preloaded_bases=preloaded,
                        use_skip_filters=skip_flag,
                    )
                    if summary.empty:
                        continue
                    avg_total_return = summary["total_return"].mean()
                    avg_cagr = summary["cagr"].mean()
                    avg_dd = summary["max_drawdown"].mean()
                    trades = summary["trades"].sum()
                    rec = {
                        "buy_rules": ",".join(buy_set),
                        "sell_rules": ",".join(sell_set),
                        "forced_rules": ",".join(forced_set) if forced_set else "NONE",
                        "skip_filters": "on" if skip_flag else "off",
                        "avg_total_return": avg_total_return,
                        "avg_cagr": avg_cagr,
                        "avg_max_drawdown": avg_dd,
                        "total_trades": trades,
                    }
                    records.append(rec)
                    existing_keys.add(key)
                    # 每写一条就落地，便于中途恢复
                    merged = existing_records + records
                    pd.DataFrame(merged).to_csv(existing_path, index=False)

    if not records:
        print("无结果")
        return

    df = pd.DataFrame(records)
    df_sorted = df.sort_values("avg_total_return", ascending=False).reset_index(drop=True)
    df_sorted.to_csv(existing_path, index=False)

    print("\nTop 10 组合（按平均总收益）:")
    print(df_sorted.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
