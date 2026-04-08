#!/usr/bin/env python3
"""
sensitivity_215d.py
-------------------
全量215天敏感分析。基于exit_sensitivity.py框架，扩展覆盖7个核心参数。

每次只变一个参数，其他保持当前值。同时对IM和IC跑，输出并排对比表。

参数清单（方案E baseline, 2026-04-08）：
  P1  ME窄幅阈值     me_ratio          当前0.10
  P2  MID_BREAK bars mid_break_bars    当前3
  P3  trailing scale  trail_scale       当前IM=1.5, IC=2.0
  P4  信号阈值       threshold         当前IM=55, IC=60
  P5  dm顺势/逆势    dm_trend/contra   当前1.1/0.9
  P6  止损幅度       stop_loss_pct     当前0.5%
  P7  午后session_wt session_afternoon 当前1.0
  P8  动态trailing   trail_atr_mult    当前0(禁用)  ← Phase 1.1
  P9  动态止损       sl_atr_mult       当前0(禁用)  ← Phase 1.2

用法：
    python scripts/sensitivity_215d.py                    # 全部参数
    python scripts/sensitivity_215d.py --param P1         # 单个参数
    python scripts/sensitivity_215d.py --param P1,P4      # 多个参数
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.storage.db_manager import DBManager, get_db
from config.config_loader import ConfigLoader

# Import exit_sensitivity's core functions
from scripts.exit_sensitivity import (
    check_exit_param, run_day_param, preload_day_data,
    _stats, _reason_breakdown, _build_15m_from_5m,
    BASELINE, IM_MULT, COOLDOWN_MINUTES,
)

import strategies.intraday.A_share_momentum_signal_v2 as sig_mod

# ─────────────────────────────────────────────────────────────
# 当前值（baseline）— 反映04-08方案E最新参数
# ─────────────────────────────────────────────────────────────
CURRENT_BASELINE = {
    "stop_loss_pct":     0.005,
    "trail_scale":       1.0,      # IM=1.5, IC=2.0（per-symbol覆盖）
    "trail_bonus":       0.002,
    "me_hold_min":       20,
    "me_ratio":          0.10,     # 当前值（从0.20改过来）
    "time_stop_min":     60,
    "lunch_mode":        "loss_only",
    "mid_break_bars":    3,        # 当前值（从2改过来）
    "trail_atr_mult":    0,        # 0=禁用ATR trailing（固定阶梯）
    "sl_atr_mult":       0,        # 0=禁用ATR stop loss（固定0.5%）
}

# Per-symbol overrides — 方案E参数
SYMBOL_BASELINE = {
    "IM": {"trail_scale": 1.5, "threshold": 55, "dm_trend": 1.1, "dm_contrarian": 0.9},
    "IC": {"trail_scale": 2.0, "threshold": 60, "dm_trend": 1.1, "dm_contrarian": 0.9},
}

# ─────────────────────────────────────────────────────────────
# 参数网格
# ─────────────────────────────────────────────────────────────
PARAM_DEFS = {
    "P1": {
        "name": "ME窄幅阈值",
        "key": "me_ratio",
        "values": [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30],
        "type": "exit",
    },
    "P2": {
        "name": "MID_BREAK bars",
        "key": "mid_break_bars",
        "values": [1, 2, 3, 4],
        "type": "exit",
    },
    "P3": {
        "name": "Trailing Stop scale",
        "key": "trail_scale",
        "values": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5],
        "type": "exit",
    },
    "P4": {
        "name": "信号阈值",
        "key": "threshold",
        "values": [55, 60, 65, 70],
        "type": "entry",
    },
    "P5": {
        "name": "dm顺势/逆势",
        "key": "dm",
        "values": [(1.0, 1.0), (1.1, 0.9), (1.2, 0.8), (1.3, 0.7)],
        "type": "entry",
    },
    "P6": {
        "name": "止损幅度",
        "key": "stop_loss_pct",
        "values": [0.003, 0.004, 0.005, 0.006, 0.007],
        "type": "exit",
    },
    "P7": {
        "name": "午后session_weight",
        "key": "session_afternoon",
        "values": [0.7, 0.8, 0.9, 1.0, 1.1],
        "type": "entry",
    },
    "P8": {
        "name": "动态trailing(ATR mult)",
        "key": "trail_atr_mult",
        "values": [0, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5],
        "type": "exit",
    },
    "P9": {
        "name": "动态止损(ATR mult)",
        "key": "sl_atr_mult",
        "values": [0, 1.5, 2.0, 2.5, 3.0, 4.0],
        "type": "exit",
    },
}


# ─────────────────────────────────────────────────────────────
# 获取全量日期
# ─────────────────────────────────────────────────────────────
def get_all_dates(db: DBManager) -> List[str]:
    r = db.query_df(
        "SELECT DISTINCT substr(datetime,1,10) as d "
        "FROM index_min WHERE symbol='000852' AND period=300 ORDER BY d"
    )
    return [d.replace('-', '') for d in r['d'].tolist()]


# ─────────────────────────────────────────────────────────────
# Entry参数：通过monkey-patch SYMBOL_PROFILES实现
# ─────────────────────────────────────────────────────────────
def _patch_entry_param(sym: str, key: str, val: Any) -> dict:
    """Patch entry-side param, return original values for restore."""
    prof = sig_mod.SYMBOL_PROFILES.get(sym, sig_mod._DEFAULT_PROFILE)
    saved = {}
    if key == "threshold":
        # threshold不在profile里直接控制score_all，而是在run_day_param里比较
        # 无需patch profile
        pass
    elif key == "dm":
        saved["dm_trend"] = prof.get("dm_trend", 1.2)
        saved["dm_contrarian"] = prof.get("dm_contrarian", 0.8)
        prof["dm_trend"] = val[0]
        prof["dm_contrarian"] = val[1]
    elif key == "session_afternoon":
        saved["session_multiplier"] = dict(prof.get("session_multiplier", {}))
        # 修改午后时段权重（1300~1430）
        sm = prof.get("session_multiplier", {})
        for k in list(sm.keys()):
            if k.startswith("13") or k == "1330-1430":
                sm[k] = val
    return saved


def _restore_entry_param(sym: str, key: str, saved: dict):
    """Restore patched values."""
    prof = sig_mod.SYMBOL_PROFILES.get(sym, sig_mod._DEFAULT_PROFILE)
    if key == "dm":
        if "dm_trend" in saved:
            prof["dm_trend"] = saved["dm_trend"]
        if "dm_contrarian" in saved:
            prof["dm_contrarian"] = saved["dm_contrarian"]
    elif key == "session_afternoon":
        if "session_multiplier" in saved:
            prof["session_multiplier"] = saved["session_multiplier"]


# ─────────────────────────────────────────────────────────────
# 跑一个参数值的完整回测
# ─────────────────────────────────────────────────────────────
def run_one_config(
    sym: str,
    dates: List[str],
    all_bars: pd.DataFrame,
    daily_all: Optional[pd.DataFrame],
    gen,
    per_day: Dict,
    # 参数
    exit_params: Dict,
    threshold: int,
    slippage: float = 0,
) -> Dict:
    """Run backtest with given params, return stats."""
    all_trades = []
    day_pnls = {}

    for td in dates:
        day = per_day[td]
        trades = run_day_param(
            sym=sym, td=td, db=None,
            all_bars=all_bars, daily_all=daily_all,
            ema20=day["ema20"], std20=day["std20"],
            is_high_vol=day["is_high_vol"],
            sentiment=day["sentiment"],
            d_override=day["d_override"],
            gen=gen,
            effective_threshold=threshold,
            slippage=slippage,
            **exit_params,
        )
        full = [t for t in trades if not t.get("partial")]
        pnl = sum(t["pnl_pts"] for t in full)
        all_trades.extend(full)
        day_pnls[td] = pnl

    st = _stats(all_trades)
    return {
        **st,
        "day_pnls": day_pnls,
        "trades": all_trades,
    }


# ─────────────────────────────────────────────────────────────
# 一个参数的完整sweep（IM+IC并排）
# ─────────────────────────────────────────────────────────────
def sweep_param(
    param_id: str,
    dates: List[str],
    preloaded: Dict[str, Tuple],  # sym -> (all_bars, daily_all, gen, thr, per_day)
    slippage: float = 0,
):
    pdef = PARAM_DEFS[param_id]
    values = pdef["values"]
    key = pdef["key"]
    ptype = pdef["type"]

    print(f"\n{'═'*95}")
    print(f"  {param_id}: {pdef['name']}  |  {len(dates)} days  |  Grid: {values}")
    print(f"{'═'*95}")

    results = {}  # (sym, val) -> stats

    for sym in ["IM", "IC"]:
        all_bars, daily_all, gen, default_thr, per_day = preloaded[sym]
        sym_base = SYMBOL_BASELINE[sym]

        for val in values:
            # Build exit params from current baseline
            exit_params = dict(CURRENT_BASELINE)

            # Apply per-symbol baseline (trail_scale)
            exit_params["trail_scale"] = sym_base["trail_scale"]

            # The threshold to use
            threshold = sym_base["threshold"]

            # Now override the specific param being tested
            if ptype == "exit":
                exit_params[key] = val
            elif key == "threshold":
                threshold = val
            elif key == "dm":
                saved = _patch_entry_param(sym, "dm", val)
            elif key == "session_afternoon":
                saved = _patch_entry_param(sym, "session_afternoon", val)

            stats = run_one_config(
                sym, dates, all_bars, daily_all, gen, per_day,
                exit_params=exit_params,
                threshold=threshold,
                slippage=slippage,
            )

            # Restore entry patches
            if ptype == "entry" and key in ("dm", "session_afternoon"):
                _restore_entry_param(sym, key, saved)

            results[(sym, str(val))] = stats

    # ── Print table ──────────────────────────────────────────
    print(f"\n  {'Value':>12} | {'IM PnL':>8} {'IM N':>5} {'IM WR':>6} {'IM avg':>7}"
          f" | {'IC PnL':>8} {'IC N':>5} {'IC WR':>6} {'IC avg':>7}"
          f" | {'合计':>8} {'vs当前':>8}")
    print(f"  {'─'*12}─┼─{'─'*28}─┼─{'─'*28}─┼─{'─'*17}")

    # Determine current value for each symbol
    if key == "threshold":
        im_cur, ic_cur = str(SYMBOL_BASELINE["IM"]["threshold"]), str(SYMBOL_BASELINE["IC"]["threshold"])
    elif key == "trail_scale":
        im_cur, ic_cur = str(SYMBOL_BASELINE["IM"]["trail_scale"]), str(SYMBOL_BASELINE["IC"]["trail_scale"])
    elif key == "dm":
        im_cur = str((SYMBOL_BASELINE["IM"]["dm_trend"], SYMBOL_BASELINE["IM"]["dm_contrarian"]))
        ic_cur = im_cur
    elif key == "session_afternoon":
        im_cur = ic_cur = "1.0"
    else:
        im_cur = ic_cur = str(CURRENT_BASELINE.get(key, "?"))

    # For "合计" baseline
    baseline_im = results.get(("IM", im_cur), {}).get("pnl", 0)
    baseline_ic = results.get(("IC", ic_cur), {}).get("pnl", 0)
    baseline_total = baseline_im + baseline_ic

    best_total = -99999
    best_val = None

    for val in values:
        val_str = str(val)
        im = results.get(("IM", val_str), {"pnl": 0, "n": 0, "wr": 0, "avg": 0})
        ic = results.get(("IC", val_str), {"pnl": 0, "n": 0, "wr": 0, "avg": 0})
        total = im["pnl"] + ic["pnl"]

        if total > best_total:
            best_total = total
            best_val = val

        # Marker for current value
        is_im_cur = (val_str == im_cur)
        is_ic_cur = (val_str == ic_cur)
        marker = ""
        if is_im_cur and is_ic_cur:
            marker = " ◀"
        elif is_im_cur:
            marker = " ◀IM"
        elif is_ic_cur:
            marker = " ◀IC"

        delta = total - baseline_total if baseline_total != 0 else 0
        delta_str = f"{delta:>+8.0f}" if not (is_im_cur and is_ic_cur) else "     ---"

        # Format value
        if isinstance(val, tuple):
            fmt_val = f"{val[0]}/{val[1]}"
        elif isinstance(val, float) and val < 0.1:
            fmt_val = f"{val*100:.1f}%"
        else:
            fmt_val = f"{val}"

        print(f"  {fmt_val:>12}{marker:4}"
              f" | {im['pnl']:>+8.0f} {im['n']:>5} {im['wr']:>5.0f}% {im['avg']:>+7.1f}"
              f" | {ic['pnl']:>+8.0f} {ic['n']:>5} {ic['wr']:>5.0f}% {ic['avg']:>+7.1f}"
              f" | {total:>+8.0f} {delta_str}")

    print(f"  {'─'*12}─┴─{'─'*28}─┴─{'─'*28}─┴─{'─'*17}")

    # Best
    if isinstance(best_val, tuple):
        fmt_best = f"{best_val[0]}/{best_val[1]}"
    elif isinstance(best_val, float) and best_val < 0.1:
        fmt_best = f"{best_val*100:.1f}%"
    else:
        fmt_best = f"{best_val}"
    print(f"  215天最优: {fmt_best}  (合计{best_total:+.0f}pt)")

    # Return summary for final comparison table
    return {
        "param_id": param_id,
        "name": pdef["name"],
        "best_val": best_val,
        "best_total": best_total,
        "baseline_total": baseline_total,
        "results": results,
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="全量215天敏感分析")
    parser.add_argument("--param", default="ALL",
                        help="P1,P2,...,P7 or ALL")
    parser.add_argument("--slippage", type=float, default=0)
    parser.add_argument("--version", default="auto", choices=["v2", "v3", "auto"])
    args = parser.parse_args()

    db = get_db()
    dates = get_all_dates(db)

    if args.param.upper() == "ALL":
        param_ids = list(PARAM_DEFS.keys())
    else:
        param_ids = [p.strip().upper() for p in args.param.split(",")]

    print(f"\n{'═'*95}")
    print(f"  全量敏感分析 | {len(dates)} days ({dates[0]}~{dates[-1]})")
    print(f"  参数: {param_ids}  |  Slippage: {args.slippage}pt")
    print(f"{'═'*95}")

    # Preload data for both symbols
    preloaded = {}
    for sym in ["IM", "IC"]:
        print(f"\nPreloading {sym}...")
        t0 = time.time()
        all_bars, daily_all, gen, thr, per_day = preload_day_data(
            sym, dates, db, version=args.version
        )
        print(f"  {sym}: {len(all_bars)} bars, {len(dates)} days preloaded ({time.time()-t0:.1f}s)")
        preloaded[sym] = (all_bars, daily_all, gen, thr, per_day)

    # Run sweeps
    all_summaries = []
    for pid in param_ids:
        if pid not in PARAM_DEFS:
            print(f"\n  Unknown param: {pid}. Available: {list(PARAM_DEFS.keys())}")
            continue
        t0 = time.time()
        summary = sweep_param(pid, dates, preloaded, slippage=args.slippage)
        elapsed = time.time() - t0
        print(f"  ({elapsed:.0f}s)")
        all_summaries.append(summary)

    # ── Final comparison table ────────────────────────────────
    if len(all_summaries) > 1:
        print(f"\n{'═'*85}")
        print(f"  总结: 215天最优 vs 34天最优（当前值）")
        print(f"{'─'*85}")
        print(f"  {'Param':5} {'Name':>18} {'当前值':>12} {'215d最优':>12}"
              f" {'当前合计':>9} {'最优合计':>9} {'Δ':>7} {'一致?':>5}")
        print(f"{'─'*85}")

        for s in all_summaries:
            pid = s["param_id"]
            key = PARAM_DEFS[pid]["key"]

            # Current value
            if key == "threshold":
                cur_str = "IM55/IC60"
            elif key == "trail_scale":
                cur_str = "IM1.5/IC2.0"
            elif key == "dm":
                cur_str = "1.1/0.9"
            elif key == "session_afternoon":
                cur_str = "1.0"
            elif key == "stop_loss_pct":
                cur_str = "0.5%"
            elif key == "me_ratio":
                cur_str = "0.10"
            elif key == "mid_break_bars":
                cur_str = "3"
            elif key == "trail_atr_mult":
                cur_str = "0(off)"
            elif key == "sl_atr_mult":
                cur_str = "0(off)"
            else:
                cur_str = "?"

            bv = s["best_val"]
            if isinstance(bv, tuple):
                best_str = f"{bv[0]}/{bv[1]}"
            elif isinstance(bv, float) and bv < 0.1:
                best_str = f"{bv*100:.1f}%"
            else:
                best_str = str(bv)

            delta = s["best_total"] - s["baseline_total"]
            # Check consistency
            consistent = "YES" if abs(delta) < 50 else "NO"

            print(f"  {pid:5} {s['name']:>18} {cur_str:>12} {best_str:>12}"
                  f" {s['baseline_total']:>+9.0f} {s['best_total']:>+9.0f}"
                  f" {delta:>+7.0f} {consistent:>5}")

        print(f"{'═'*85}")


if __name__ == "__main__":
    main()
