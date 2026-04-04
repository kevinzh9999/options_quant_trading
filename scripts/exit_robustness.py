#!/usr/bin/env python3
"""
exit_robustness.py
------------------
G5 ME窄幅阈值 + G8 MID_BREAK bars 稳健性验证

复用 exit_sensitivity.py 的核心函数：
  - preload_day_data()
  - run_day_param()
  - _stats()

测试项目：
  G5 稳健性（IM+IC各3项）：
    1. 邻域平滑性：me_ratio = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    2. 时间分段：前半段 / 后半段，me_ratio=0.10 vs 0.20
    3. 每周表现：5周，me_ratio=0.10 vs 0.20，0.10至少4/5周不差于0.20

  G8 稳健性（IM+IC）：
    时间分段：mid_break_bars=2 vs 3，前后半段对比

  组合测试：
    Baseline    me_ratio=0.20, mid_break_bars=2
    G5 only     me_ratio=0.10, mid_break_bars=2
    G8 only     me_ratio=0.20, mid_break_bars=3
    G5+G8       me_ratio=0.10, mid_break_bars=3

使用方法：
    python scripts/exit_robustness.py
    python scripts/exit_robustness.py --slippage 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.storage.db_manager import DBManager
from config.config_loader import ConfigLoader

# Import core functions from exit_sensitivity
from scripts.exit_sensitivity import (
    preload_day_data,
    run_day_param,
    _stats,
    BASELINE,
)

# ─────────────────────────────────────────────────────────────
# 35-day date set
# ─────────────────────────────────────────────────────────────
ALL_DATES = (
    "20260204,20260205,20260206,20260209,20260210,20260211,20260212,20260213,"
    "20260225,20260226,20260227,20260302,20260303,20260304,20260305,20260306,"
    "20260309,20260310,20260311,20260312,20260313,20260316,20260317,20260318,"
    "20260319,20260320,20260323,20260324,20260325,20260326,20260327,20260328,"
    "20260401,20260402,20260403"
)

DATES = [d.strip() for d in ALL_DATES.split(",") if d.strip()]

# 5-week buckets
WEEKS = {
    "W1": DATES[0:5],
    "W2": DATES[5:10],
    "W3": DATES[10:17],
    "W4": DATES[17:24],
    "W5": DATES[24:],
}

HALF1 = DATES[:17]
HALF2 = DATES[17:]

# G5 sweep values
G5_SWEEP = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]

# Symbol configs
SYM_THRESHOLD = {"IM": 60, "IC": 65}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def run_dates(
    dates: List[str],
    all_bars,
    daily_all,
    gen,
    effective_threshold: int,
    per_day: Dict,
    slippage: float,
    **exit_params,
) -> Dict:
    """Run backtest on a subset of dates with given exit params. Returns _stats dict."""
    params = dict(BASELINE)
    params.update(exit_params)

    all_trades = []
    for td in dates:
        if td not in per_day:
            continue
        day = per_day[td]
        trades = run_day_param(
            sym="",  # not used inside
            td=td,
            db=None,
            all_bars=all_bars,
            daily_all=daily_all,
            ema20=day["ema20"],
            std20=day["std20"],
            is_high_vol=day["is_high_vol"],
            sentiment=day["sentiment"],
            d_override=day["d_override"],
            gen=gen,
            effective_threshold=effective_threshold,
            slippage=slippage,
            stop_loss_pct=params["stop_loss_pct"],
            trail_scale=params["trail_scale"],
            trail_bonus=params["trail_bonus"],
            me_hold_min=params["me_hold_min"],
            me_ratio=params["me_ratio"],
            time_stop_min=params["time_stop_min"],
            lunch_mode=params["lunch_mode"],
            mid_break_bars=params["mid_break_bars"],
        )
        all_trades.extend(trades)

    return _stats(all_trades)


def fmt_pnl(v: float) -> str:
    return f"{v:+.0f}"


def fmt_wr(v: float) -> str:
    return f"{v:.0f}%"


# ─────────────────────────────────────────────────────────────
# G5 tests
# ─────────────────────────────────────────────────────────────

def g5_neighborhood(sym: str, all_bars, daily_all, gen, thr, per_day, slippage):
    """G5 邻域平滑性：sweep me_ratio across 7 values."""
    print(f"\n--- {sym} 邻域平滑性 ---")
    print(f"{'me_ratio':>10} | {'PnL':>8} | {'Trades':>7} | {'WR':>6} | {'Avg/T':>7} | {'BE-slip':>7}")
    print("-" * 65)

    rows = []
    for ratio in G5_SWEEP:
        st = run_dates(DATES, all_bars, daily_all, gen, thr, per_day, slippage,
                       me_ratio=ratio)
        rows.append((ratio, st))
        marker = " ←BASE" if ratio == 0.20 else ""
        print(f"  {ratio:8.2f}   | {fmt_pnl(st['pnl']):>8} | {st['n']:>7} | "
              f"{fmt_wr(st['wr']):>6} | {st['avg']:>+7.1f} | {st['be_slip']:>6.1f}pt{marker}")

    return rows


def g5_time_split(sym: str, all_bars, daily_all, gen, thr, per_day, slippage):
    """G5 时间分段：前半 vs 后半，me_ratio=0.10 vs 0.20."""
    print(f"\n--- {sym} 时间分段 ---")
    print(f"{'':12} | {'前半段(17天)':>12} | {'后半段(18天)':>12} | {'全段(35天)':>12}")
    print("-" * 60)

    results = {}
    for ratio in [0.20, 0.10]:
        h1 = run_dates(HALF1, all_bars, daily_all, gen, thr, per_day, slippage, me_ratio=ratio)
        h2 = run_dates(HALF2, all_bars, daily_all, gen, thr, per_day, slippage, me_ratio=ratio)
        full = run_dates(DATES, all_bars, daily_all, gen, thr, per_day, slippage, me_ratio=ratio)
        results[ratio] = (h1, h2, full)
        label = f"me={ratio:.2f}"
        print(f"  {label:12} | {fmt_pnl(h1['pnl']):>12} | {fmt_pnl(h2['pnl']):>12} | {fmt_pnl(full['pnl']):>12}")

    return results


def g5_weekly(sym: str, all_bars, daily_all, gen, thr, per_day, slippage):
    """G5 每周表现：me_ratio=0.10 vs 0.20，0.10至少4/5周不差于0.20."""
    print(f"\n--- {sym} 每周 ---")
    week_names = list(WEEKS.keys())
    header = f"{'':12} | " + " | ".join(f"{w:>8}" for w in week_names)
    print(header)
    print("-" * (14 + 11 * len(week_names)))

    results = {}
    for ratio in [0.20, 0.10]:
        label = f"me={ratio:.2f}"
        week_pnls = []
        for wname in week_names:
            wdates = WEEKS[wname]
            st = run_dates(wdates, all_bars, daily_all, gen, thr, per_day, slippage,
                           me_ratio=ratio)
            week_pnls.append(st["pnl"])
        results[ratio] = week_pnls
        row = f"  {label:12} | " + " | ".join(f"{fmt_pnl(p):>8}" for p in week_pnls)
        print(row)

    # Count weeks where 0.10 >= 0.20
    wins = sum(1 for a, b in zip(results[0.10], results[0.20]) if a >= b)
    print(f"\n  me=0.10胜出周: {wins}/{len(week_names)}")
    if wins >= 4:
        print(f"  >> PASS (>= 4/5)")
    else:
        print(f"  >> FAIL (< 4/5), 结果不稳健")

    return results


# ─────────────────────────────────────────────────────────────
# G8 tests
# ─────────────────────────────────────────────────────────────

def g8_time_split(sym: str, all_bars, daily_all, gen, thr, per_day, slippage):
    """G8 时间分段：mid_break_bars=2 vs 3，前后半段对比."""
    print(f"\n--- {sym} G8时间分段 ---")
    print(f"{'':16} | {'前半段(17天)':>12} | {'后半段(18天)':>12} | {'全段(35天)':>12}")
    print("-" * 64)

    results = {}
    for bars in [2, 3]:
        h1 = run_dates(HALF1, all_bars, daily_all, gen, thr, per_day, slippage, mid_break_bars=bars)
        h2 = run_dates(HALF2, all_bars, daily_all, gen, thr, per_day, slippage, mid_break_bars=bars)
        full = run_dates(DATES, all_bars, daily_all, gen, thr, per_day, slippage, mid_break_bars=bars)
        results[bars] = (h1, h2, full)
        label = f"mid_bars={bars}"
        marker = " ←BASE" if bars == 2 else ""
        print(f"  {label:16} | {fmt_pnl(h1['pnl']):>12} | {fmt_pnl(h2['pnl']):>12} | {fmt_pnl(full['pnl']):>12}{marker}")

    return results


# ─────────────────────────────────────────────────────────────
# Combo test
# ─────────────────────────────────────────────────────────────

def combo_test(
    sym_data: Dict,  # {sym: (all_bars, daily_all, gen, thr, per_day)}
    slippage: float,
):
    """组合测试：2×2 grid of (me_ratio, mid_break_bars)."""
    combos = [
        ("Baseline", 0.20, 2),
        ("G5 only",  0.10, 2),
        ("G8 only",  0.20, 3),
        ("G5+G8",    0.10, 3),
    ]

    # Compute per-symbol stats
    combo_stats = {}  # combo_name -> {sym: stats}
    for name, ratio, bars in combos:
        combo_stats[name] = {}
        for sym, (all_bars, daily_all, gen, thr, per_day) in sym_data.items():
            st = run_dates(DATES, all_bars, daily_all, gen, thr, per_day, slippage,
                           me_ratio=ratio, mid_break_bars=bars)
            combo_stats[name][sym] = st

    # Display
    syms = list(sym_data.keys())
    print(f"\n=== 组合测试 ===")
    hdr = f"{'方案':12} | " + " | ".join(f"{s+' PnL':>10}" for s in syms) + f" | {'合计':>8} | {'vs baseline':>12}"
    print(hdr)
    print("-" * (14 + 13 * len(syms) + 24))

    base_total = sum(combo_stats["Baseline"][s]["pnl"] for s in syms)

    for name, ratio, bars in combos:
        total = sum(combo_stats[name][s]["pnl"] for s in syms)
        vs_base = total - base_total
        vs_str = f"{vs_base:+.0f}" if name != "Baseline" else "—"
        sym_cols = " | ".join(f"{fmt_pnl(combo_stats[name][s]['pnl']):>10}" for s in syms)
        print(f"  {name:12} | {sym_cols} | {fmt_pnl(total):>8} | {vs_str:>12}")

    return combo_stats


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="G5/G8稳健性验证")
    parser.add_argument("--slippage", type=float, default=0,
                        help="Slippage per side in points (default 0)")
    parser.add_argument("--symbol", default="IM,IC",
                        help="Comma-separated symbols (default IM,IC)")
    parser.add_argument("--version", choices=["v2", "v3", "auto"], default="auto")
    args = parser.parse_args()

    slippage = args.slippage
    symbols = [s.strip().upper() for s in args.symbol.split(",") if s.strip()]

    db = DBManager(ConfigLoader().get_db_path())

    # Preload data for each symbol
    print("=" * 70)
    print("  EXIT ROBUSTNESS TEST — G5 ME窄幅阈值 + G8 MID_BREAK bars")
    print(f"  Symbols: {symbols}  |  {len(DATES)} days  |  slippage={slippage}pt/side")
    print("=" * 70)

    sym_data = {}
    for sym in symbols:
        print(f"\n[Loading {sym}]")
        all_bars, daily_all, gen, thr, per_day = preload_day_data(
            sym, DATES, db, version=args.version
        )
        # Override threshold from our config
        thr = SYM_THRESHOLD.get(sym, thr)
        sym_data[sym] = (all_bars, daily_all, gen, thr, per_day)
        print(f"  threshold={thr}, bars={len(all_bars)}")

    # ── G5 Tests ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  === G5 ME窄幅阈值稳健性 ===")
    print(f"{'='*70}")

    g5_results = {}
    for sym in symbols:
        all_bars, daily_all, gen, thr, per_day = sym_data[sym]
        print(f"\n{'─'*60}")
        print(f"  [{sym}] (threshold={thr})")
        print(f"{'─'*60}")

        g5_results[sym] = {}
        g5_results[sym]["neighborhood"] = g5_neighborhood(sym, all_bars, daily_all, gen, thr, per_day, slippage)
        g5_results[sym]["time_split"] = g5_time_split(sym, all_bars, daily_all, gen, thr, per_day, slippage)
        g5_results[sym]["weekly"] = g5_weekly(sym, all_bars, daily_all, gen, thr, per_day, slippage)

    # G5 Summary
    print(f"\n{'─'*60}")
    print("  G5 总结")
    print(f"{'─'*60}")
    for sym in symbols:
        nb_rows = g5_results[sym]["neighborhood"]
        # PnL at 0.10 vs 0.20
        pnl_010 = next((st["pnl"] for r, st in nb_rows if abs(r - 0.10) < 0.001), 0)
        pnl_020 = next((st["pnl"] for r, st in nb_rows if abs(r - 0.20) < 0.001), 0)
        delta = pnl_010 - pnl_020
        wk_results = g5_results[sym]["weekly"]
        wins = sum(1 for a, b in zip(wk_results[0.10], wk_results[0.20]) if a >= b)
        verdict = "ROBUST" if (delta > 0 and wins >= 4) else ("PARTIAL" if (delta > 0 or wins >= 4) else "NOT ROBUST")
        print(f"  {sym}: 0.10 vs 0.20 全段Δ={delta:+.0f}pt  每周胜出={wins}/5  → {verdict}")

    # ── G8 Tests ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  === G8 MID_BREAK稳健性 ===")
    print(f"{'='*70}")

    g8_results = {}
    for sym in symbols:
        all_bars, daily_all, gen, thr, per_day = sym_data[sym]
        print(f"\n{'─'*60}")
        print(f"  [{sym}] (threshold={thr})")
        print(f"{'─'*60}")
        g8_results[sym] = g8_time_split(sym, all_bars, daily_all, gen, thr, per_day, slippage)

    # G8 Summary
    print(f"\n{'─'*60}")
    print("  G8 总结")
    print(f"{'─'*60}")
    for sym in symbols:
        res = g8_results[sym]
        h1_2, h2_2, full_2 = res[2]
        h1_3, h2_3, full_3 = res[3]
        delta_full = full_3["pnl"] - full_2["pnl"]
        h1_better = full_3["pnl"] > full_2["pnl"] or abs(delta_full) < 10
        delta_h1 = h1_3["pnl"] - h1_2["pnl"]
        delta_h2 = h2_3["pnl"] - h2_2["pnl"]
        both_halves = (delta_h1 >= 0 and delta_h2 >= 0)
        verdict = "ROBUST" if (delta_full >= 0 and both_halves) else ("PARTIAL" if delta_full >= 0 else "NOT ROBUST")
        print(f"  {sym}: bars=3 vs 2  全段Δ={delta_full:+.0f}pt  前半Δ={delta_h1:+.0f}pt  后半Δ={delta_h2:+.0f}pt  → {verdict}")

    # ── Combo Test ───────────────────────────────────────────────
    combo_test(sym_data, slippage)

    print(f"\n{'='*70}")
    print("  Done.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
