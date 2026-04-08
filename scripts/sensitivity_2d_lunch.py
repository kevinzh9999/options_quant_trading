#!/usr/bin/env python3
"""
sensitivity_2d_lunch.py — Phase 1.4
LUNCH_CLOSE时间 × 浮盈trailing宽度 联合2D优化。

优化：预计算entry信号（score_all只跑一次），replay only varies exit params。
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np, pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.storage.db_manager import get_db
from scripts.exit_sensitivity import (
    check_exit_param, preload_day_data, _stats,
    _build_15m_from_5m, _calc_minutes, COOLDOWN_MINUTES,
)
from scripts.sensitivity_215d import CURRENT_BASELINE, SYMBOL_BASELINE, get_all_dates
from scripts.sensitivity_2d_me_mid import precompute_signals, replay_exits

LUNCH_TIMES = [
    ("03:15", "11:15"),
    ("03:20", "11:20"),
    ("03:25", "11:25"),  # 当前
    ("never", "不关"),
]
LUNCH_TRAILS = [0.002, 0.003, 0.004, 0.005, 0.999]  # 0.999 = 不收紧


def replay_exits_lunch(sym, dates, day_signals, exit_params, threshold,
                       lunch_utc, lunch_trail, slippage=0):
    """replay_exits with custom lunch params."""
    stop_loss_pct = exit_params["stop_loss_pct"]

    def _make_stop(entry_p, direction):
        if direction == "LONG":
            return entry_p * (1 - stop_loss_pct)
        else:
            return entry_p * (1 + stop_loss_pct)

    all_trades = []
    lunch_mode = "never" if lunch_utc == "never" else "loss_only"
    lunch_close_utc = lunch_utc if lunch_utc != "never" else "99:99"

    for td in dates:
        bars = day_signals[td]
        if not bars:
            continue

        position = None
        last_exit_utc = ""
        last_exit_dir = ""

        for bd in bars:
            if bd is None:
                continue

            utc_hm    = bd["utc_hm"]
            price     = bd["price"]
            high      = bd["high"]
            low       = bd["low"]
            score     = bd["score"]
            direction = bd["direction"]

            if position is not None:
                stop_price = position.get("stop_loss", 0)
                bar_stopped = False
                if stop_price > 0:
                    if position["direction"] == "LONG" and low <= stop_price:
                        bar_stopped = True
                    elif position["direction"] == "SHORT" and high >= stop_price:
                        bar_stopped = True

                if bar_stopped:
                    entry_p = position["entry_price"]
                    exit_p = (stop_price - slippage if position["direction"] == "LONG"
                              else stop_price + slippage)
                    pnl = (exit_p - entry_p) if position["direction"] == "LONG" else (entry_p - exit_p)
                    all_trades.append({"pnl_pts": pnl, "direction": position["direction"],
                                       "reason": "STOP_LOSS"})
                    last_exit_utc, last_exit_dir = utc_hm, position["direction"]
                    position = None
                else:
                    if position["direction"] == "LONG":
                        position["highest_since"] = max(position["highest_since"], high)
                    else:
                        position["lowest_since"] = min(position["lowest_since"], low)

                    exit_info = check_exit_param(
                        position, price,
                        bd["bar_5m_signal"],
                        bd["bar_15m"] if len(bd["bar_15m"]) > 0 else None,
                        utc_hm, symbol=sym,
                        stop_loss_pct=exit_params["stop_loss_pct"],
                        trail_scale=exit_params["trail_scale"],
                        trail_bonus=exit_params["trail_bonus"],
                        me_hold_min=exit_params["me_hold_min"],
                        me_ratio=exit_params["me_ratio"],
                        time_stop_min=exit_params["time_stop_min"],
                        lunch_mode=lunch_mode,
                        mid_break_bars=exit_params["mid_break_bars"],
                        LUNCH_CLOSE_UTC=lunch_close_utc,
                        TRAILING_STOP_LUNCH=lunch_trail,
                    )

                    if exit_info["should_exit"]:
                        entry_p = position["entry_price"]
                        exit_p = (price - slippage if position["direction"] == "LONG"
                                  else price + slippage)
                        pnl = (exit_p - entry_p) if position["direction"] == "LONG" else (entry_p - exit_p)
                        all_trades.append({"pnl_pts": pnl, "direction": position["direction"],
                                           "reason": exit_info["exit_reason"]})
                        if exit_info["exit_volume"] >= position.get("volume", 1):
                            last_exit_utc, last_exit_dir = utc_hm, position["direction"]
                            position = None

            in_cooldown = False
            if last_exit_utc and direction == last_exit_dir:
                if 0 < _calc_minutes(last_exit_utc, utc_hm) < COOLDOWN_MINUTES:
                    in_cooldown = True

            if (position is None and not in_cooldown
                    and score >= threshold and direction and bd["can_open"]):
                entry_p = price + slippage if direction == "LONG" else price - slippage
                position = {
                    "entry_price": entry_p,
                    "entry_time_utc": utc_hm,
                    "direction": direction,
                    "volume": 1,
                    "highest_since": entry_p,
                    "lowest_since": entry_p,
                    "stop_loss": _make_stop(entry_p, direction),
                    "bars_below_mid": 0,
                }

        if position is not None and bars:
            last_bd = [b for b in reversed(bars) if b is not None]
            if last_bd:
                lp = last_bd[0]["price"]
                entry_p = position["entry_price"]
                exit_p = (lp - slippage if position["direction"] == "LONG" else lp + slippage)
                pnl = (exit_p - entry_p) if position["direction"] == "LONG" else (entry_p - exit_p)
                all_trades.append({"pnl_pts": pnl, "direction": position["direction"],
                                   "reason": "EOD_FORCE"})

    return _stats(all_trades)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slippage", type=float, default=0)
    args = parser.parse_args()

    db = get_db()
    dates = [d for d in get_all_dates(db) if d >= "20250516"]

    print(f"\n{'='*90}")
    print(f"  2D: LUNCH_CLOSE时间 × 浮盈trailing  |  {len(dates)} days")
    combos = len(LUNCH_TIMES) * len(LUNCH_TRAILS)
    print(f"  时间: {[t[1] for t in LUNCH_TIMES]}  Trail: {LUNCH_TRAILS}  ({combos} combos)")
    print(f"{'='*90}", flush=True)

    cached = {}
    for sym in ["IM", "IC"]:
        print(f"\n  Precomputing {sym} signals...", end="", flush=True)
        t0 = time.time()
        all_bars, daily_all, gen, thr, per_day = preload_day_data(sym, dates, db, version="auto")
        day_signals = precompute_signals(sym, dates, all_bars, daily_all, gen, per_day,
                                         SYMBOL_BASELINE[sym]["threshold"])
        cached[sym] = {"day_signals": day_signals, "threshold": SYMBOL_BASELINE[sym]["threshold"]}
        print(f" done ({time.time()-t0:.1f}s)", flush=True)

    print(f"\n  Replaying {combos} exit combos...", flush=True)
    results = {}
    t_start = time.time()
    combo_i = 0

    for lunch_utc, lunch_bj in LUNCH_TIMES:
        for lt in LUNCH_TRAILS:
            combo_i += 1
            combo_pnl = {}
            for sym in ["IM", "IC"]:
                exit_params = dict(CURRENT_BASELINE)
                exit_params["trail_scale"] = SYMBOL_BASELINE[sym]["trail_scale"]

                st = replay_exits_lunch(sym, dates, cached[sym]["day_signals"],
                                        exit_params, cached[sym]["threshold"],
                                        lunch_utc, lt, args.slippage)
                combo_pnl[sym] = st

            total_pnl = combo_pnl["IM"]["pnl"] + combo_pnl["IC"]["pnl"]
            results[(lunch_bj, lt)] = {**combo_pnl, "total": total_pnl}

            trail_str = f"{lt*100:.1f}%" if lt < 0.9 else "不收紧"
            elapsed = time.time() - t_start
            eta = elapsed / combo_i * (combos - combo_i) if combo_i > 0 else 0
            print(f"    [{combo_i:2d}/{combos}] {lunch_bj:>5} trail={trail_str:>5}"
                  f"  IM={combo_pnl['IM']['pnl']:+5.0f}  IC={combo_pnl['IC']['pnl']:+5.0f}"
                  f"  合计={total_pnl:+5.0f}  ({elapsed:.0f}s ETA {eta:.0f}s)", flush=True)

    # Heatmap
    baseline = results.get(("11:25", 0.003), {"total": 0})["total"]

    print(f"\n{'='*90}")
    print(f"  Heatmap: IM+IC PnL | 当前 11:25+0.3% = {baseline:+.0f}pt")
    print(f"{'='*90}")

    hdr = f"  {'Time':>6}"
    for lt in LUNCH_TRAILS:
        lbl = f"{lt*100:.1f}%" if lt < 0.9 else "不收紧"
        hdr += f" | {lbl:>9}"
    print(hdr)
    print(f"  {'─'*6}" + "─┼─".join(["─"*9]*len(LUNCH_TRAILS)))

    best_total, best_combo = -99999, None
    for _, lunch_bj in LUNCH_TIMES:
        row = f"  {lunch_bj:>6}"
        for lt in LUNCH_TRAILS:
            t = results[(lunch_bj, lt)]["total"]
            marker = " ◀" if (lunch_bj == "11:25" and lt == 0.003) else ""
            row += f" | {t:>+6.0f}{marker:3}"
            if t > best_total:
                best_total, best_combo = t, (lunch_bj, lt)
        print(row)

    best_trail_str = f"{best_combo[1]*100:.1f}%" if best_combo[1] < 0.9 else "不收紧"
    print(f"\n  最优: {best_combo[0]}+{best_trail_str}"
          f"  {best_total:+.0f}pt (vs当前 {best_total-baseline:+.0f})")

    print(f"\n  Delta vs baseline:")
    hdr = f"  {'Time':>6}"
    for lt in LUNCH_TRAILS:
        lbl = f"{lt*100:.1f}%" if lt < 0.9 else "不收紧"
        hdr += f" | {lbl:>9}"
    print(hdr)
    print(f"  {'─'*6}" + "─┼─".join(["─"*9]*len(LUNCH_TRAILS)))
    for _, lunch_bj in LUNCH_TIMES:
        row = f"  {lunch_bj:>6}"
        for lt in LUNCH_TRAILS:
            d = results[(lunch_bj, lt)]["total"] - baseline
            row += f" | {d:>+9.0f}"
        print(row)

    print(f"\n  总耗时: {time.time()-t_start:.0f}s (replay)")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
