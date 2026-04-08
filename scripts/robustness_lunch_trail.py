#!/usr/bin/env python3
"""
robustness_lunch_trail.py
LUNCH_CLOSE trailing=0.2% 稳健性三检：
  1. 时间分半（前半 vs 后半）
  2. 参数邻域（0.15~0.30%细化）
  3. 单日贡献（最大单日PnL占比）
"""
from __future__ import annotations
import sys, time
from pathlib import Path
from typing import Dict, List
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
from scripts.sensitivity_2d_me_mid import precompute_signals


def replay_lunch_perday(sym, dates, day_signals, exit_params, threshold,
                        lunch_utc, lunch_trail, slippage=0):
    """Replay with per-day PnL tracking."""
    stop_loss_pct = exit_params["stop_loss_pct"]
    lunch_mode = "never" if lunch_utc == "never" else "loss_only"
    lunch_close_utc = lunch_utc if lunch_utc != "never" else "99:99"

    def _make_stop(entry_p, direction):
        if direction == "LONG":
            return entry_p * (1 - stop_loss_pct)
        else:
            return entry_p * (1 + stop_loss_pct)

    day_pnls = {}

    for td in dates:
        bars = day_signals[td]
        if not bars:
            day_pnls[td] = 0
            continue

        position = None
        last_exit_utc = ""
        last_exit_dir = ""
        day_trades = []

        for bd in bars:
            if bd is None:
                continue

            utc_hm = bd["utc_hm"]
            price, high, low = bd["price"], bd["high"], bd["low"]
            score, direction = bd["score"], bd["direction"]

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
                    day_trades.append(pnl)
                    last_exit_utc, last_exit_dir = utc_hm, position["direction"]
                    position = None
                else:
                    if position["direction"] == "LONG":
                        position["highest_since"] = max(position["highest_since"], high)
                    else:
                        position["lowest_since"] = min(position["lowest_since"], low)

                    exit_info = check_exit_param(
                        position, price, bd["bar_5m_signal"],
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
                        day_trades.append(pnl)
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
                    "entry_price": entry_p, "entry_time_utc": utc_hm,
                    "direction": direction, "volume": 1,
                    "highest_since": entry_p, "lowest_since": entry_p,
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
                day_trades.append(pnl)

        day_pnls[td] = sum(day_trades)

    return day_pnls


def main():
    db = get_db()
    dates = [d for d in get_all_dates(db) if d >= "20250516"]
    mid = len(dates) // 2
    first_half = dates[:mid]
    second_half = dates[mid:]

    print(f"\n{'='*90}")
    print(f"  LUNCH trailing 0.2% 稳健性三检  |  {len(dates)} days")
    print(f"  前半: {first_half[0]}~{first_half[-1]} ({len(first_half)}d)")
    print(f"  后半: {second_half[0]}~{second_half[-1]} ({len(second_half)}d)")
    print(f"{'='*90}", flush=True)

    # Precompute signals
    cached = {}
    for sym in ["IM", "IC"]:
        print(f"  Precomputing {sym}...", end="", flush=True)
        t0 = time.time()
        all_bars, daily_all, gen, thr, per_day = preload_day_data(sym, dates, db, version="auto")
        day_signals = precompute_signals(sym, dates, all_bars, daily_all, gen, per_day,
                                         SYMBOL_BASELINE[sym]["threshold"])
        cached[sym] = day_signals
        print(f" {time.time()-t0:.1f}s", flush=True)

    # ── Test 1: 参数邻域（细化0.15~0.30%）────────────────────
    TRAILS = [0.0015, 0.0018, 0.0020, 0.0022, 0.0025, 0.0028, 0.0030, 0.0035]
    print(f"\n  ── 检查1: 参数邻域 ──", flush=True)
    print(f"  {'Trail':>8} | {'IM':>7} {'IC':>7} {'合计':>7} {'vs0.3%':>7}", flush=True)
    print(f"  {'─'*8}─┼─{'─'*31}", flush=True)

    baseline_03 = {}  # will store 0.3% result
    trail_results = {}

    for lt in TRAILS:
        total_pnl = 0
        sym_pnl = {}
        for sym in ["IM", "IC"]:
            ep = dict(CURRENT_BASELINE)
            ep["trail_scale"] = SYMBOL_BASELINE[sym]["trail_scale"]
            day_pnls = replay_lunch_perday(
                sym, dates, cached[sym], ep, SYMBOL_BASELINE[sym]["threshold"],
                "03:25", lt,
            )
            sp = sum(day_pnls.values())
            sym_pnl[sym] = sp
            total_pnl += sp

        trail_results[lt] = {"IM": sym_pnl["IM"], "IC": sym_pnl["IC"], "total": total_pnl}
        if lt == 0.003:
            baseline_03 = trail_results[lt]

        marker = " ◀" if lt == 0.003 else ("  ★" if lt == 0.002 else "")
        delta = total_pnl - baseline_03.get("total", total_pnl) if baseline_03 else 0
        print(f"  {lt*100:>7.2}% | {sym_pnl['IM']:>+7.0f} {sym_pnl['IC']:>+7.0f}"
              f" {total_pnl:>+7.0f} {delta:>+7.0f}{marker}", flush=True)

    # ── Test 2: 时间分半 ─────────────────────────────────────
    print(f"\n  ── 检查2: 时间分半 ──", flush=True)
    for label, subset in [("前半", first_half), ("后半", second_half)]:
        for lt_val, lt_name in [(0.002, "0.2%"), (0.003, "0.3%")]:
            total = 0
            for sym in ["IM", "IC"]:
                ep = dict(CURRENT_BASELINE)
                ep["trail_scale"] = SYMBOL_BASELINE[sym]["trail_scale"]
                day_pnls = replay_lunch_perday(
                    sym, subset, cached[sym], ep, SYMBOL_BASELINE[sym]["threshold"],
                    "03:25", lt_val,
                )
                total += sum(day_pnls.values())
            print(f"  {label} {lt_name}: {total:>+7.0f}pt", flush=True)

    # ── Test 3: 单日贡献 ─────────────────────────────────────
    print(f"\n  ── 检查3: 单日贡献 ──", flush=True)

    # Get per-day delta (0.2% - 0.3%)
    day_deltas = {}
    for sym in ["IM", "IC"]:
        ep = dict(CURRENT_BASELINE)
        ep["trail_scale"] = SYMBOL_BASELINE[sym]["trail_scale"]
        pnl_02 = replay_lunch_perday(sym, dates, cached[sym], ep,
                                      SYMBOL_BASELINE[sym]["threshold"], "03:25", 0.002)
        pnl_03 = replay_lunch_perday(sym, dates, cached[sym], ep,
                                      SYMBOL_BASELINE[sym]["threshold"], "03:25", 0.003)
        for td in dates:
            day_deltas[td] = day_deltas.get(td, 0) + (pnl_02.get(td, 0) - pnl_03.get(td, 0))

    total_delta = sum(day_deltas.values())
    sorted_days = sorted(day_deltas.items(), key=lambda x: -abs(x[1]))

    print(f"  总改善: {total_delta:+.0f}pt")
    print(f"  Top-5 单日贡献:")
    for td, d in sorted_days[:5]:
        pct = abs(d) / abs(total_delta) * 100 if total_delta != 0 else 0
        print(f"    {td}: {d:>+6.0f}pt ({pct:.0f}%)")

    max_day_pct = abs(sorted_days[0][1]) / abs(total_delta) * 100 if total_delta != 0 else 0
    print(f"  最大单日占比: {max_day_pct:.0f}%  ({'PASS' if max_day_pct < 30 else 'FAIL'}: <30%)")

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  稳健性总结:")

    # Neighborhood: is 0.2% on a plateau or a spike?
    vals = [(k, v["total"]) for k, v in trail_results.items()]
    vals.sort()
    neighbors_of_02 = [v for k, v in vals if 0.0015 <= k <= 0.0025]
    all_positive = all(v > baseline_03["total"] for v in neighbors_of_02)
    print(f"  1. 参数邻域: 0.15~0.25%区间全部优于0.3%? {'YES ✅' if all_positive else 'NO ❌'}")
    print(f"  2. 最大单日占比: {max_day_pct:.0f}% {'PASS ✅' if max_day_pct < 30 else 'FAIL ❌'}")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
