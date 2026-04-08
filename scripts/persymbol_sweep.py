#!/usr/bin/env python3
"""
persymbol_sweep.py — 品种独立参数优化
IM和IC分别找各自最优参数，对比统一参数。

Exit参数用precompute+replay（快），Entry参数用全量回测（慢但不可避免）。
"""
import sys, time, os
os.environ["PYTHONUNBUFFERED"] = "1"
from pathlib import Path
from typing import Dict, List
import numpy as np, pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.storage.db_manager import get_db
from scripts.exit_sensitivity import (
    check_exit_param, preload_day_data, run_day_param, _stats,
    _build_15m_from_5m, _calc_minutes, COOLDOWN_MINUTES,
)
from scripts.sensitivity_215d import CURRENT_BASELINE, SYMBOL_BASELINE, get_all_dates
from scripts.sensitivity_2d_me_mid import precompute_signals
import strategies.intraday.A_share_momentum_signal_v2 as sig_mod


def replay_with_exit_params(sym, dates, day_signals, exit_params, threshold, slippage=0):
    """Fast replay: only varies exit params, signals are cached."""
    stop_loss_pct = exit_params["stop_loss_pct"]

    def _make_stop(entry_p, direction):
        sl = stop_loss_pct
        return entry_p * (1 - sl) if direction == "LONG" else entry_p * (1 + sl)

    all_trades = []
    for td in dates:
        bars = day_signals.get(td, [])
        if not bars:
            continue
        position = None
        last_exit_utc = ""
        last_exit_dir = ""

        for bd in bars:
            if bd is None:
                continue
            utc_hm, price, high, low = bd["utc_hm"], bd["price"], bd["high"], bd["low"]
            score, direction = bd["score"], bd["direction"]

            if position is not None:
                sp = position.get("stop_loss", 0)
                bar_stopped = False
                if sp > 0:
                    if position["direction"] == "LONG" and low <= sp:
                        bar_stopped = True
                    elif position["direction"] == "SHORT" and high >= sp:
                        bar_stopped = True

                if bar_stopped:
                    ep = position["entry_price"]
                    xp = (sp - slippage if position["direction"] == "LONG" else sp + slippage)
                    pnl = (xp - ep) if position["direction"] == "LONG" else (ep - xp)
                    all_trades.append({"pnl_pts": pnl})
                    last_exit_utc, last_exit_dir = utc_hm, position["direction"]
                    position = None
                else:
                    if position["direction"] == "LONG":
                        position["highest_since"] = max(position["highest_since"], high)
                    else:
                        position["lowest_since"] = min(position["lowest_since"], low)

                    b15 = bd["bar_15m"]
                    exit_info = check_exit_param(
                        position, price, bd["bar_5m_signal"],
                        b15 if len(b15) > 0 else None,
                        utc_hm, symbol=sym, **exit_params,
                    )
                    if exit_info["should_exit"]:
                        ep = position["entry_price"]
                        xp = (price - slippage if position["direction"] == "LONG" else price + slippage)
                        pnl = (xp - ep) if position["direction"] == "LONG" else (ep - xp)
                        all_trades.append({"pnl_pts": pnl})
                        if exit_info["exit_volume"] >= position.get("volume", 1):
                            last_exit_utc, last_exit_dir = utc_hm, position["direction"]
                            position = None

            in_cooldown = False
            if last_exit_utc and direction == last_exit_dir:
                if 0 < _calc_minutes(last_exit_utc, utc_hm) < COOLDOWN_MINUTES:
                    in_cooldown = True

            if (position is None and not in_cooldown
                    and score >= threshold and direction and bd["can_open"]):
                ep = price + slippage if direction == "LONG" else price - slippage
                position = {
                    "entry_price": ep, "entry_time_utc": utc_hm,
                    "direction": direction, "volume": 1,
                    "highest_since": ep, "lowest_since": ep,
                    "stop_loss": _make_stop(ep, direction),
                    "bars_below_mid": 0,
                }

        if position is not None and bars:
            last_bd = [b for b in reversed(bars) if b is not None]
            if last_bd:
                lp = last_bd[0]["price"]
                ep = position["entry_price"]
                xp = (lp - slippage if position["direction"] == "LONG" else lp + slippage)
                pnl = (xp - ep) if position["direction"] == "LONG" else (ep - xp)
                all_trades.append({"pnl_pts": pnl})

    return sum(t["pnl_pts"] for t in all_trades), len(all_trades)


def main():
    db = get_db()
    dates = [d for d in get_all_dates(db) if d >= "20250516"]

    print(f"{'='*90}")
    print(f"  品种独立参数sweep  |  {len(dates)} days")
    print(f"{'='*90}", flush=True)

    # Exit params to sweep (precompute+replay, fast)
    EXIT_PARAMS = {
        "trail_scale":   [0.8, 1.0, 1.5, 2.0, 2.5, 3.0],
        "stop_loss_pct": [0.003, 0.004, 0.005, 0.006, 0.007],
        "me_ratio":      [0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
        "mid_break_bars":[2, 3, 4, 5],
    }

    # Entry params (need full recompute per value)
    ENTRY_PARAMS = {
        "threshold": [50, 55, 60, 65, 70],
        "dm": [(1.0, 1.0), (1.1, 0.9), (1.2, 0.8)],
    }

    results = {}  # sym -> param -> [(val, pnl, n), ...]

    for sym in ["IM", "IC"]:
        print(f"\n  ── {sym} ──", flush=True)
        results[sym] = {}

        # 1. Precompute signals for default entry params
        t0 = time.time()
        all_bars, daily_all, gen, _, per_day = preload_day_data(sym, dates, db, version="auto")
        base_thr = SYMBOL_BASELINE[sym]["threshold"]
        day_signals = precompute_signals(sym, dates, all_bars, daily_all, gen, per_day, base_thr)
        print(f"  Precompute: {time.time()-t0:.0f}s", flush=True)

        # 2. Exit params: fast replay
        for param, values in EXIT_PARAMS.items():
            param_results = []
            for val in values:
                ep = dict(CURRENT_BASELINE)
                ep["trail_scale"] = SYMBOL_BASELINE[sym]["trail_scale"]
                ep[param] = val
                pnl, n = replay_with_exit_params(sym, dates, day_signals, ep, base_thr)
                param_results.append((val, pnl, n))
            results[sym][param] = param_results

            best = max(param_results, key=lambda x: x[1])
            cur_val = SYMBOL_BASELINE[sym].get(param, CURRENT_BASELINE.get(param))
            cur_pnl = next((p for v, p, n in param_results if v == cur_val), 0)
            delta = best[1] - cur_pnl
            vals_str = " ".join(f"{v if not (isinstance(v,float) and v<0.1) else f'{v*100:.1f}%'}:{p:+.0f}{'◀' if v==cur_val else '★' if p==best[1] else ''}" for v, p, n in param_results)
            print(f"  {param:>14}: best={best[0]} ({delta:>+4.0f}pt)  {vals_str}", flush=True)

        # 3. Entry params: full recompute for each value
        for param, values in ENTRY_PARAMS.items():
            param_results = []
            for val in values:
                saved_dm = None
                thr = base_thr

                if param == "threshold":
                    thr = val
                    # Need to re-precompute signals at this threshold
                    sigs = precompute_signals(sym, dates, all_bars, daily_all, gen, per_day, thr)
                elif param == "dm":
                    prof = sig_mod.SYMBOL_PROFILES.get(sym, sig_mod._DEFAULT_PROFILE)
                    saved_dm = (prof.get("dm_trend"), prof.get("dm_contrarian"))
                    prof["dm_trend"], prof["dm_contrarian"] = val[0], val[1]
                    # Re-precompute with new dm
                    sigs = precompute_signals(sym, dates, all_bars, daily_all, gen, per_day, base_thr)
                    prof["dm_trend"], prof["dm_contrarian"] = saved_dm[0], saved_dm[1]
                else:
                    sigs = day_signals

                ep = dict(CURRENT_BASELINE)
                ep["trail_scale"] = SYMBOL_BASELINE[sym]["trail_scale"]
                pnl, n = replay_with_exit_params(sym, dates, sigs, ep, thr)
                param_results.append((val, pnl, n))

            results[sym][param] = param_results

            best = max(param_results, key=lambda x: x[1])
            if param == "threshold":
                cur_val = base_thr
            elif param == "dm":
                cur_val = (SYMBOL_BASELINE[sym].get("dm_trend", 1.1),
                           SYMBOL_BASELINE[sym].get("dm_contrarian", 0.9))
            else:
                cur_val = None
            cur_pnl = next((p for v, p, n in param_results if v == cur_val), 0)
            delta = best[1] - cur_pnl

            def _fmt(v):
                if isinstance(v, tuple): return f"{v[0]}/{v[1]}"
                if isinstance(v, float) and v < 0.1: return f"{v*100:.1f}%"
                return str(v)

            vals_str = " ".join(f"{_fmt(v)}:{p:+.0f}{'◀' if v==cur_val else '★' if p==best[1] else ''}" for v, p, n in param_results)
            print(f"  {param:>14}: best={_fmt(best[0])} ({delta:>+4.0f}pt)  {vals_str}", flush=True)

        print(f"  Total: {time.time()-t0:.0f}s", flush=True)

    # ── Summary table ────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  品种独立最优 vs 当前参数")
    print(f"{'='*90}")
    all_params = list(EXIT_PARAMS.keys()) + list(ENTRY_PARAMS.keys())

    print(f"  {'参数':>14} | {'IM最优':>10} {'IM当前':>10} {'Δ':>6} | {'IC最优':>10} {'IC当前':>10} {'Δ':>6} | {'分化?':>4}")
    print(f"  {'─'*14}─┼─{'─'*28}─┼─{'─'*28}─┼─{'─'*4}")

    def _fmt(v):
        if isinstance(v, tuple): return f"{v[0]}/{v[1]}"
        if isinstance(v, float) and v < 0.1: return f"{v*100:.1f}%"
        return str(v)

    for p in all_params:
        im_res = results["IM"][p]
        ic_res = results["IC"][p]
        im_best = max(im_res, key=lambda x: x[1])
        ic_best = max(ic_res, key=lambda x: x[1])

        if p == "threshold":
            im_cur = SYMBOL_BASELINE["IM"]["threshold"]
            ic_cur = SYMBOL_BASELINE["IC"]["threshold"]
        elif p == "trail_scale":
            im_cur = SYMBOL_BASELINE["IM"]["trail_scale"]
            ic_cur = SYMBOL_BASELINE["IC"]["trail_scale"]
        elif p == "dm":
            im_cur = (1.1, 0.9)
            ic_cur = (1.1, 0.9)
        else:
            im_cur = ic_cur = CURRENT_BASELINE.get(p)

        im_cur_pnl = next((pnl for v, pnl, n in im_res if v == im_cur), 0)
        ic_cur_pnl = next((pnl for v, pnl, n in ic_res if v == ic_cur), 0)
        im_d = im_best[1] - im_cur_pnl
        ic_d = ic_best[1] - ic_cur_pnl
        diff = "YES" if im_best[0] != ic_best[0] else "no"

        print(f"  {p:>14} | {_fmt(im_best[0]):>10} {_fmt(im_cur):>10} {im_d:>+6.0f}"
              f" | {_fmt(ic_best[0]):>10} {_fmt(ic_cur):>10} {ic_d:>+6.0f} | {diff:>4}")

    # Total potential
    im_total_cur = sum(next((pnl for v, pnl, n in results["IM"][p] if v == (SYMBOL_BASELINE["IM"].get(p, CURRENT_BASELINE.get(p)))), 0) for p in ["threshold"])
    # Actually just show the per-param best sums as upper bound
    print(f"\n  注意：以上是逐参数独立最优，联合最优需要做多维grid（参数间可能有交互）")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
