#!/usr/bin/env python3
"""
sensitivity_2d_me_mid.py — Phase 1.3
ME narrow_ratio × MID_BREAK bars 联合2D优化。

优化：预计算所有entry信号（score_all只跑一次），然后对每个exit参数组合只做replay。
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
import strategies.intraday.A_share_momentum_signal_v2 as sig_mod

ME_RATIOS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
MID_BARS  = [2, 3, 4, 5]


def precompute_signals(sym, dates, all_bars, daily_all, gen, per_day, threshold):
    """Run score_all once for all bars across all days, cache results."""
    from strategies.intraday.A_share_momentum_signal_v2 import is_open_allowed

    # per-day list of bar-level data: [(utc_hm, price, high, low, bar_5m_signal, bar_15m, score, direction, is_open), ...]
    day_signals = {}

    for td in dates:
        day = per_day[td]
        date_dash = f"{td[:4]}-{td[4:6]}-{td[6:]}"
        today_mask = all_bars["datetime"].str.startswith(date_dash)
        today_indices = all_bars.index[today_mask].tolist()
        if not today_indices:
            day_signals[td] = []
            continue

        daily_df = None
        if daily_all is not None:
            daily_df = daily_all[daily_all["trade_date"] < td].tail(30).reset_index(drop=True)
            if daily_df.empty:
                daily_df = None

        ema20, std20 = day["ema20"], day["std20"]

        from strategies.intraday.A_share_momentum_signal_v2 import compute_volume_profile
        _vol_profile = compute_volume_profile(all_bars, before_date=td, lookback_days=20)

        bars_data = []

        for idx in today_indices:
            bar_5m = all_bars.loc[:idx].tail(200).copy()
            if len(bar_5m) < 15:
                bars_data.append(None)
                continue
            bar_5m_signal = bar_5m.iloc[:-1]
            if len(bar_5m_signal) < 15:
                bars_data.append(None)
                continue

            price = float(bar_5m.iloc[-1]["close"])
            high  = float(bar_5m.iloc[-1]["high"])
            low   = float(bar_5m.iloc[-1]["low"])
            dt_str = str(all_bars.loc[idx, "datetime"])
            utc_hm = dt_str[11:16]

            signal_price = float(bar_5m_signal.iloc[-1]["close"])
            z_val = (signal_price - ema20) / std20 if std20 > 0 else None
            bar_15m_full = _build_15m_from_5m(bar_5m)
            bar_15m = bar_15m_full.iloc[:-1] if len(bar_15m_full) > 1 else bar_15m_full

            result = gen.score_all(
                sym, bar_5m_signal, bar_15m, daily_df, None, day["sentiment"],
                zscore=z_val, is_high_vol=day["is_high_vol"], d_override=day["d_override"],
                vol_profile=_vol_profile,
            )

            score     = result["total"]     if result else 0
            direction = result["direction"] if result else ""
            can_open  = is_open_allowed(utc_hm)

            bars_data.append({
                "utc_hm": utc_hm, "price": price, "high": high, "low": low,
                "bar_5m_signal": bar_5m_signal, "bar_15m": bar_15m,
                "score": score, "direction": direction, "can_open": can_open,
            })

        day_signals[td] = bars_data

    return day_signals


def replay_exits(sym, dates, day_signals, exit_params, threshold, slippage=0):
    """Replay with cached signals, only varying exit params."""
    stop_loss_pct = exit_params["stop_loss_pct"]

    def _make_stop(entry_p, direction):
        if direction == "LONG":
            return entry_p * (1 - stop_loss_pct)
        else:
            return entry_p * (1 + stop_loss_pct)

    all_trades = []

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

            # ── exit ──
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
                        **exit_params,
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

            # ── entry ──
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

        # Force close
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
    print(f"  2D联合优化: ME narrow_ratio × MID_BREAK bars  |  {len(dates)} days")
    print(f"  ME: {ME_RATIOS}  MID: {MID_BARS}  ({len(ME_RATIOS)*len(MID_BARS)} combos)")
    print(f"{'='*90}", flush=True)

    # Phase 1: precompute signals (slow, but only once per symbol)
    cached = {}
    for sym in ["IM", "IC"]:
        print(f"\n  Precomputing {sym} signals...", end="", flush=True)
        t0 = time.time()
        all_bars, daily_all, gen, thr, per_day = preload_day_data(sym, dates, db, version="auto")
        day_signals = precompute_signals(sym, dates, all_bars, daily_all, gen, per_day,
                                         SYMBOL_BASELINE[sym]["threshold"])
        cached[sym] = {"day_signals": day_signals, "threshold": SYMBOL_BASELINE[sym]["threshold"]}
        print(f" done ({time.time()-t0:.1f}s)", flush=True)

    # Phase 2: replay exits (fast)
    print(f"\n  Replaying {len(ME_RATIOS)*len(MID_BARS)} exit combos...", flush=True)
    results = {}
    t_start = time.time()
    combo_i = 0
    total = len(ME_RATIOS) * len(MID_BARS)

    for me_r in ME_RATIOS:
        for mid_b in MID_BARS:
            combo_i += 1
            combo_pnl = {}
            for sym in ["IM", "IC"]:
                exit_params = dict(CURRENT_BASELINE)
                exit_params["trail_scale"] = SYMBOL_BASELINE[sym]["trail_scale"]
                exit_params["me_ratio"] = me_r
                exit_params["mid_break_bars"] = mid_b

                st = replay_exits(sym, dates, cached[sym]["day_signals"],
                                  exit_params, cached[sym]["threshold"], args.slippage)
                combo_pnl[sym] = st

            total_pnl = combo_pnl["IM"]["pnl"] + combo_pnl["IC"]["pnl"]
            results[(me_r, mid_b)] = {**combo_pnl, "total": total_pnl}

            elapsed = time.time() - t_start
            eta = elapsed / combo_i * (total - combo_i) if combo_i > 0 else 0
            print(f"    [{combo_i:2d}/{total}] ME={me_r:.2f} MID={mid_b}"
                  f"  IM={combo_pnl['IM']['pnl']:+5.0f}  IC={combo_pnl['IC']['pnl']:+5.0f}"
                  f"  合计={total_pnl:+5.0f}  ({elapsed:.0f}s ETA {eta:.0f}s)", flush=True)

    # ── Heatmap ──
    cur_me = CURRENT_BASELINE["me_ratio"]
    cur_mid = CURRENT_BASELINE["mid_break_bars"]
    baseline = results.get((cur_me, cur_mid), {"total": 0})["total"]

    print(f"\n{'='*90}")
    print(f"  Heatmap: IM+IC PnL (pt) | 当前 ME={cur_me} MID={cur_mid} = {baseline:+.0f}pt")
    print(f"{'='*90}")

    hdr = f"  {'ME':>6}"
    for mid_b in MID_BARS:
        hdr += f" | {'MID='+str(mid_b):>9}"
    print(hdr)
    print(f"  {'─'*6}" + "─┼─".join(["─"*9]*len(MID_BARS)))

    best_total, best_combo = -99999, None
    for me_r in ME_RATIOS:
        row = f"  {me_r:>6.2f}"
        for mid_b in MID_BARS:
            t = results[(me_r, mid_b)]["total"]
            marker = " ◀" if (me_r == cur_me and mid_b == cur_mid) else ""
            row += f" | {t:>+6.0f}{marker:3}"
            if t > best_total:
                best_total, best_combo = t, (me_r, mid_b)
        print(row)

    print(f"\n  最优: ME={best_combo[0]:.2f} MID={best_combo[1]}"
          f"  {best_total:+.0f}pt (vs当前 {best_total-baseline:+.0f})")

    # Delta table
    print(f"\n  Delta vs baseline:")
    hdr = f"  {'ME':>6}"
    for mid_b in MID_BARS:
        hdr += f" | {'MID='+str(mid_b):>9}"
    print(hdr)
    print(f"  {'─'*6}" + "─┼─".join(["─"*9]*len(MID_BARS)))
    for me_r in ME_RATIOS:
        row = f"  {me_r:>6.2f}"
        for mid_b in MID_BARS:
            d = results[(me_r, mid_b)]["total"] - baseline
            row += f" | {d:>+9.0f}"
        print(row)

    # Best detail
    b = results[best_combo]
    c = results.get((cur_me, cur_mid), {"IM": {"pnl":0,"n":0,"wr":0,"avg":0},
                                         "IC": {"pnl":0,"n":0,"wr":0,"avg":0}})
    print(f"\n  {'':12} | {'IM PnL':>7} {'N':>4} {'WR':>5} {'avg':>6}"
          f" | {'IC PnL':>7} {'N':>4} {'WR':>5} {'avg':>6}")
    for label, r in [("当前", c), ("最优", b)]:
        print(f"  {label:>12} | {r['IM']['pnl']:>+7.0f} {r['IM']['n']:>4}"
              f" {r['IM']['wr']:>4.0f}% {r['IM']['avg']:>+6.1f}"
              f" | {r['IC']['pnl']:>+7.0f} {r['IC']['n']:>4}"
              f" {r['IC']['wr']:>4.0f}% {r['IC']['avg']:>+6.1f}")

    print(f"\n  总耗时: {time.time()-t_start:.0f}s (replay only)")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
