#!/usr/bin/env python3
"""Reversal EV 全量 sweep — 3 模式对比 IM 909 天

模式：
  - off:        baseline，纯 score-based
  - exit_only:  reversal 触发时只平对向原仓（REVERSAL_EXIT），不反手开仓
  - full:       完整 reversal，既平原仓又反手开仓（当前 live 行为）

输出：
  - 总 PnL（spot + futures）per mode
  - Per-year breakdown
  - IS/OOS split（前半/后半）
  - Delta: full - off, exit_only - off
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _worker(args):
    sym, td, mode = args
    from data.storage.db_manager import get_db
    from scripts.backtest_signals_day import run_day
    db = get_db()
    try:
        trades = run_day(sym, td, db, verbose=False, slippage=0,
                         version="auto", reversal_mode=mode)
    except Exception as e:
        return (td, mode, 0.0, 0.0, 0, f"ERR:{e}")
    spot = sum(t["pnl_pts"] for t in trades)
    fut = sum(t.get("pnl_pts_fut", t["pnl_pts"]) for t in trades)
    n_rev_exit = sum(1 for t in trades if t.get("reason") == "REVERSAL_EXIT")
    n_rev_open = sum(1 for t in trades if t.get("source") == "reversal")
    return (td, mode, spot, fut, len(trades), f"rev_exit={n_rev_exit} rev_open={n_rev_open}")


def main():
    from data.storage.db_manager import get_db
    db = get_db()
    # All trading days from index_min (IM via 000852)
    dates_df = db.query_df(
        "SELECT DISTINCT substr(datetime,1,10) as d FROM index_min "
        "WHERE symbol='000852' AND period=300 ORDER BY d"
    )
    dates = [d.replace("-", "") for d in dates_df["d"].tolist()]
    print(f"Total trading days: {len(dates)}  range: {dates[0]} ~ {dates[-1]}")

    # IC reversal is disabled in REVERSAL_CONFIG → only IM produces differences
    sym = "IM"
    tasks = []
    for td in dates:
        for mode in ["off", "exit_only", "full"]:
            tasks.append((sym, td, mode))
    print(f"Tasks: {len(tasks)} ({len(dates)} days × 3 modes)")

    with Pool(8) as p:
        results = p.map(_worker, tasks)

    # Aggregate
    from collections import defaultdict
    by_mode = defaultdict(list)
    for td, mode, spot, fut, n, info in results:
        by_mode[mode].append({"td": td, "spot": spot, "fut": fut, "n": n, "info": info})

    print("\n" + "=" * 90)
    print(f" Reversal EV Sweep — {sym}, {len(dates)} days")
    print("=" * 90)
    print(f"{'Mode':<12} {'Trades':>8} {'Spot PnL':>12} {'Fut PnL':>12} {'Fut 元(1手)':>15}")
    print("-" * 90)
    for mode in ["off", "exit_only", "full"]:
        rows = by_mode[mode]
        tot_n = sum(r["n"] for r in rows)
        tot_spot = sum(r["spot"] for r in rows)
        tot_fut = sum(r["fut"] for r in rows)
        print(f"{mode:<12} {tot_n:>8} {tot_spot:>+12.0f} {tot_fut:>+12.0f} "
              f"{tot_fut * 200:>+15,.0f}")

    # Deltas vs baseline
    baseline = {r["td"]: r for r in by_mode["off"]}
    print("\n━━ Deltas (mode − off) ━━")
    for mode in ["exit_only", "full"]:
        d_spot = sum(r["spot"] - baseline[r["td"]]["spot"] for r in by_mode[mode])
        d_fut = sum(r["fut"] - baseline[r["td"]]["fut"] for r in by_mode[mode])
        print(f"  {mode:<12} Δspot={d_spot:+.0f}pt  Δfut={d_fut:+.0f}pt "
              f"Δfut元={d_fut*200:+,.0f}")

    # IS/OOS split (first half / second half)
    n_half = len(dates) // 2
    is_dates = set(dates[:n_half])
    print(f"\n━━ IS/OOS split (IS=first {n_half}, OOS=last {len(dates)-n_half}) ━━")
    for mode in ["off", "exit_only", "full"]:
        is_fut = sum(r["fut"] for r in by_mode[mode] if r["td"] in is_dates)
        oos_fut = sum(r["fut"] for r in by_mode[mode] if r["td"] not in is_dates)
        print(f"  {mode:<12} IS={is_fut:+.0f}pt  OOS={oos_fut:+.0f}pt")

    # Year breakdown
    print(f"\n━━ Per-year fut PnL ━━")
    print(f"{'Year':<6} {'off':>10} {'exit_only':>12} {'full':>10} "
          f"{'Δexit':>10} {'Δfull':>10}")
    years = sorted(set(td[:4] for td in dates))
    for y in years:
        by_mode_year = {}
        for mode in ["off", "exit_only", "full"]:
            by_mode_year[mode] = sum(r["fut"] for r in by_mode[mode] if r["td"].startswith(y))
        delta_exit = by_mode_year["exit_only"] - by_mode_year["off"]
        delta_full = by_mode_year["full"] - by_mode_year["off"]
        print(f"{y:<6} {by_mode_year['off']:>+10.0f} "
              f"{by_mode_year['exit_only']:>+12.0f} "
              f"{by_mode_year['full']:>+10.0f} "
              f"{delta_exit:>+10.0f} {delta_full:>+10.0f}")

    # Recent month (2026-04) breakdown — should match shadow period observation
    print(f"\n━━ 2026-04 breakdown (最近 shadow 观察期) ━━")
    for mode in ["off", "exit_only", "full"]:
        rows = [r for r in by_mode[mode] if r["td"].startswith("202604")]
        tot_fut = sum(r["fut"] for r in rows)
        tot_n = sum(r["n"] for r in rows)
        n_rev_e = sum(int(r["info"].split("rev_exit=")[1].split(" ")[0]) for r in rows)
        n_rev_o = sum(int(r["info"].split("rev_open=")[1]) for r in rows)
        print(f"  {mode:<12} {len(rows)}d  trades={tot_n}  "
              f"rev_exit={n_rev_e} rev_open={n_rev_o}  fut={tot_fut:+.0f}pt")


if __name__ == "__main__":
    main()
