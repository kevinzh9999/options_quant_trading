#!/usr/bin/env python3
"""
sensitivity_full.py
-------------------
全参数敏感分析：对215天数据做grid search，按时间段拆分检验稳定性。

用法:
    python scripts/sensitivity_full.py --symbol IM --param threshold
    python scripts/sensitivity_full.py --symbol IC --param trailing_stop_scale
    python scripts/sensitivity_full.py --symbol IM --param all   # 逐个跑全部参数
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.storage.db_manager import DBManager, get_db
from config.config_loader import ConfigLoader
from scripts.backtest_signals_day import run_day

import strategies.intraday.A_share_momentum_signal_v2 as sig_mod


# ---------------------------------------------------------------------------
# 日期分段：按时间拆分检验稳定性
# ---------------------------------------------------------------------------
def _get_all_dates(db: DBManager) -> List[str]:
    r = db.query_df(
        "SELECT DISTINCT substr(datetime,1,10) as d "
        "FROM index_min WHERE symbol='000852' AND period=300 ORDER BY d"
    )
    return [d.replace('-', '') for d in r['d'].tolist()]


def _split_periods(dates: List[str]) -> Dict[str, List[str]]:
    """按时间段拆分"""
    periods = {}
    periods["ALL"] = dates

    # 按半年拆
    h1 = [d for d in dates if d < "20260101"]  # 2025-H2
    h2 = [d for d in dates if d >= "20260101"]  # 2026-H1
    if h1:
        periods["2025H2"] = h1
    if h2:
        periods["2026H1"] = h2

    # 按季度拆
    for q_name, q_start, q_end in [
        ("25Q2", "20250401", "20250701"),
        ("25Q3", "20250701", "20251001"),
        ("25Q4", "20251001", "20260101"),
        ("26Q1", "20260101", "20260401"),
        ("26Q2", "20260401", "20260701"),
    ]:
        q_dates = [d for d in dates if q_start <= d < q_end]
        if q_dates:
            periods[q_name] = q_dates

    return periods


# ---------------------------------------------------------------------------
# 参数patch工具
# ---------------------------------------------------------------------------
def _patch_and_run(sym: str, dates: List[str], db: DBManager,
                   patches: Dict[str, Any], version: str = "auto") -> Dict:
    """Apply patches, run backtest, return stats."""
    # Save originals
    originals = {}
    prof = sig_mod.SYMBOL_PROFILES.get(sym, sig_mod._DEFAULT_PROFILE)

    for key, val in patches.items():
        if key == "threshold":
            originals["threshold"] = getattr(sig_mod, '_SIGNAL_THRESHOLD', 60)
            sig_mod._SIGNAL_THRESHOLD = val
            # Also patch per-symbol threshold
            if "signal_threshold" in prof:
                originals["signal_threshold"] = prof["signal_threshold"]
                prof["signal_threshold"] = val
        elif key == "trailing_stop_scale":
            originals["trailing_stop_scale"] = prof.get("trailing_stop_scale", 1.0)
            prof["trailing_stop_scale"] = val
        elif key == "me_narrow_ratio":
            # Patch the hardcoded 0.10 in check_exit — need monkey-patch
            originals["me_narrow_ratio"] = val  # just track
            _patch_me_narrow(val)
        elif key == "mid_break_bars":
            originals["mid_break_bars"] = val
            _patch_mid_break(val)
        elif key == "dm_trend":
            originals["dm_trend"] = prof.get("dm_trend", 1.2)
            prof["dm_trend"] = val
        elif key == "dm_contrarian":
            originals["dm_contrarian"] = prof.get("dm_contrarian", 0.8)
            prof["dm_contrarian"] = val
        elif key == "stop_loss_pct":
            originals["stop_loss_pct"] = sig_mod.STOP_LOSS_PCT
            sig_mod.STOP_LOSS_PCT = val

    # Run
    all_trades = []
    day_pnls = {}
    for td in dates:
        trades = run_day(sym, td, db, verbose=False, slippage=0, version=version)
        full = [t for t in trades if not t.get("partial")]
        pnl = sum(t["pnl_pts"] for t in full)
        all_trades.extend(full)
        day_pnls[td] = pnl

    # Restore
    for key, orig in originals.items():
        if key == "threshold":
            sig_mod._SIGNAL_THRESHOLD = orig
        elif key == "signal_threshold" and "signal_threshold" in prof:
            prof["signal_threshold"] = orig
        elif key == "trailing_stop_scale":
            prof["trailing_stop_scale"] = orig
        elif key == "me_narrow_ratio":
            _patch_me_narrow(0.10)  # restore default
        elif key == "mid_break_bars":
            _patch_mid_break(3)  # restore default
        elif key == "dm_trend":
            prof["dm_trend"] = orig
        elif key == "dm_contrarian":
            prof["dm_contrarian"] = orig
        elif key == "stop_loss_pct":
            sig_mod.STOP_LOSS_PCT = orig

    # Stats
    n = len(all_trades)
    total_pnl = sum(t["pnl_pts"] for t in all_trades)
    wins = sum(1 for t in all_trades if t["pnl_pts"] > 0)
    win_days = sum(1 for p in day_pnls.values() if p > 0)
    active_days = sum(1 for p in day_pnls.values() if p != 0 or True)  # days with any result

    return {
        "pnl": total_pnl,
        "n": n,
        "wr": wins / n * 100 if n else 0,
        "days": len(dates),
        "win_days": win_days,
        "avg_per_day": total_pnl / len(dates) if dates else 0,
        "avg_per_trade": total_pnl / n if n else 0,
        "day_pnls": day_pnls,
    }


# ---------------------------------------------------------------------------
# Monkey-patch helpers for ME narrow ratio and MID_BREAK bars
# ---------------------------------------------------------------------------
_original_check_exit = None


def _patch_me_narrow(ratio: float):
    """Patch the ME narrow_range ratio in check_exit."""
    import types
    # We patch the module-level constant approach:
    # The value 0.10 is hardcoded at line 496. We use a module-level var.
    sig_mod._ME_NARROW_RATIO = ratio


def _patch_mid_break(bars: int):
    """Patch the MID_BREAK required bars."""
    sig_mod._MID_BREAK_BARS = bars


# Check if the signal module supports these patches
def _check_patchability():
    """Verify the signal module reads our patched values."""
    # We need to check if check_exit reads _ME_NARROW_RATIO and _MID_BREAK_BARS
    # If not, these params won't actually change behavior
    source = Path(ROOT) / "strategies/intraday/A_share_momentum_signal_v2.py"
    code = source.read_text()
    can_patch_me = "_ME_NARROW_RATIO" in code or "0.10" in code
    can_patch_mid = "_MID_BREAK_BARS" in code or ">= 3" in code
    return can_patch_me, can_patch_mid


# ---------------------------------------------------------------------------
# 参数定义
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "threshold": {
        "desc": "信号阈值",
        "values": [50, 55, 58, 60, 62, 65, 68, 70],
        "patch_key": "threshold",
        "current_im": 60,
        "current_ic": 65,
    },
    "trailing_stop_scale": {
        "desc": "Trailing Stop宽度倍数",
        "values": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "patch_key": "trailing_stop_scale",
        "current_im": 1.0,
        "current_ic": 2.0,
    },
    "stop_loss_pct": {
        "desc": "止损百���比",
        "values": [0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.010],
        "patch_key": "stop_loss_pct",
        "current_im": 0.005,
        "current_ic": 0.005,
    },
    "dm": {
        "desc": "Daily Mult (顺势/逆势)",
        "values": [
            (1.0, 1.0),   # 中性
            (1.1, 0.9),   # 轻度
            (1.2, 0.8),   # 当前IM/IC
            (1.3, 0.7),   # 激进
            (1.5, 0.5),   # 极端
        ],
        "patch_key": "dm",
        "current_im": (1.2, 0.8),
        "current_ic": (1.2, 0.8),
    },
}


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------
def run_param_sensitivity(sym: str, param_name: str, db: DBManager,
                          version: str = "auto"):
    """Run sensitivity for one parameter."""
    all_dates = _get_all_dates(db)
    periods = _split_periods(all_dates)

    if param_name not in PARAM_GRID:
        print(f"Unknown param: {param_name}. Available: {list(PARAM_GRID.keys())}")
        return

    cfg = PARAM_GRID[param_name]
    values = cfg["values"]
    current = cfg.get(f"current_{sym.lower()}", cfg.get("current_im"))

    print(f"\n{'='*90}")
    print(f" SENSITIVITY: {cfg['desc']} ({param_name}) | {sym} | {len(all_dates)} days")
    print(f" Current value: {current}")
    print(f" Grid: {values}")
    print(f"{'='*90}")

    # Run for each value × each period
    results = {}
    for val in values:
        is_current = (val == current)
        marker = " ◀ CURRENT" if is_current else ""
        print(f"\n  Testing {param_name}={val}{marker} ...")

        for period_name, period_dates in periods.items():
            if param_name == "dm":
                patches = {"dm_trend": val[0], "dm_contrarian": val[1]}
            else:
                patches = {cfg["patch_key"]: val}

            stats = _patch_and_run(sym, period_dates, db, patches, version)
            results[(val, period_name)] = stats

            if period_name == "ALL":
                print(f"    ALL: {stats['pnl']:>+7.0f}pt  "
                      f"{stats['n']:>3d}T  WR={stats['wr']:.0f}%  "
                      f"avg/day={stats['avg_per_day']:>+.1f}")

    # Print summary table
    period_names = [p for p in periods.keys()]

    print(f"\n{'─'*90}")
    print(f" {param_name:>20} |", end="")
    for pn in period_names:
        print(f" {pn:>10}", end="")
    print(f" | {'avg/day':>8} {'n_trades':>8} {'WR':>6}")
    print(f"{'─'*90}")

    for val in values:
        is_current = (val == current)
        marker = " ◀" if is_current else "  "
        label = f"{val}" if not isinstance(val, tuple) else f"{val[0]}/{val[1]}"
        print(f" {label:>20}{marker}|", end="")
        for pn in period_names:
            r = results[(val, pn)]
            print(f" {r['pnl']:>+10.0f}", end="")
        r_all = results[(val, "ALL")]
        print(f" | {r_all['avg_per_day']:>+8.1f} {r_all['n']:>8} {r_all['wr']:>5.0f}%")

    print(f"{'─'*90}")

    # Stability score: coefficient of variation across periods (excluding ALL)
    print(f"\n  --- 稳定性分析 (跨期一致性) ---")
    for val in values:
        is_current = (val == current)
        marker = " ◀" if is_current else "  "
        label = f"{val}" if not isinstance(val, tuple) else f"{val[0]}/{val[1]}"
        period_avgs = []
        for pn in period_names:
            if pn == "ALL":
                continue
            r = results[(val, pn)]
            period_avgs.append(r['avg_per_day'])

        if period_avgs:
            import statistics
            mean = statistics.mean(period_avgs)
            positive_periods = sum(1 for x in period_avgs if x > 0)
            all_periods = len(period_avgs)
            # How many periods profitable?
            print(f"  {label:>20}{marker}  "
                  f"盈利期数={positive_periods}/{all_periods}  "
                  f"均值={mean:>+.1f}/天  "
                  f"最差={min(period_avgs):>+.1f}  最好={max(period_avgs):>+.1f}")


def main():
    parser = argparse.ArgumentParser(description="全参数敏���分析")
    parser.add_argument("--symbol", default="IM")
    parser.add_argument("--param", default="threshold",
                        help="Parameter to test: threshold, trailing_stop_scale, "
                             "stop_loss_pct, dm, all")
    parser.add_argument("--version", default="auto", choices=["v2", "v3", "auto"])
    args = parser.parse_args()

    db = get_db()

    if args.param == "all":
        for param_name in PARAM_GRID:
            run_param_sensitivity(args.symbol, param_name, db, args.version)
    else:
        run_param_sensitivity(args.symbol, args.param, db, args.version)


if __name__ == "__main__":
    main()
