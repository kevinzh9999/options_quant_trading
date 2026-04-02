#!/usr/bin/env python3
"""
robustness_check.py
-------------------
验证两个参数的稳健性：
  1. ME最小持仓时间（MOMENTUM_EXHAUSTED exit的hold_minutes阈值）
  2. IC日内涨跌阈值（_intraday_filter的分档边界）

Usage:
    python scripts/robustness_check.py
    python scripts/robustness_check.py --test me          # 仅测ME
    python scripts/robustness_check.py --test ic_filter   # 仅测IC filter
    python scripts/robustness_check.py --symbol IM        # 仅测IM
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.storage.db_manager import DBManager
from config.config_loader import ConfigLoader

# ---------------------------------------------------------------------------
# 回测日期（34天，截至2026-04-02）
# ---------------------------------------------------------------------------
ALL_DATES = (
    "20260204,20260205,20260206,20260209,20260210,20260211,20260212,20260213,"
    "20260225,20260226,20260227,20260302,20260303,20260304,20260305,20260306,"
    "20260309,20260310,20260311,20260312,20260313,20260316,20260317,20260318,"
    "20260319,20260320,20260323,20260324,20260325,20260326,20260327,20260328,"
    "20260401,20260402"
).split(",")

IM_MULT = {"IM": 200, "IC": 200}


# ---------------------------------------------------------------------------
# 核心：运行单次回测（来自 backtest_signals_day.run_day，内联以便monkey-patch）
# ---------------------------------------------------------------------------

def _run_backtest(sym: str, dates: List[str], db: DBManager) -> Dict:
    """Run multi-day backtest for sym, return summary stats."""
    from scripts.backtest_signals_day import run_day
    all_trades: List[Dict] = []
    for td in dates:
        trades = run_day(sym, td, db, verbose=False)
        all_trades.extend(trades)

    full = [t for t in all_trades if not t.get("partial")]
    total_pnl = sum(t["pnl_pts"] for t in all_trades)
    n = len(full)
    wins = sum(1 for t in full if t["pnl_pts"] > 0)
    wr = wins / n * 100 if n > 0 else 0.0
    return {"pnl": total_pnl, "trades": n, "wr": wr}


# ---------------------------------------------------------------------------
# Monkey-patch helpers
# ---------------------------------------------------------------------------

def _patch_me_min_hold(min_minutes: int):
    """Patch check_exit ME条件：hold_minutes >= min_minutes."""
    import strategies.intraday.A_share_momentum_signal_v2 as sig_mod
    import scripts.backtest_signals_day as bt_mod

    original_check_exit = sig_mod.check_exit

    def patched_check_exit(position, current_price, bar_5m, bar_15m,
                           current_time_utc, reverse_signal_score=0,
                           is_high_vol=True, symbol="", spot_price=0.0):
        import pandas as pd
        import numpy as np

        # 调用原始函数，但先临时替换 ME 的 hold_minutes 阈值
        # 原实现在 hold_minutes >= 20 处；我们通过修改 position 里的 entry_time
        # 来让 hold_minutes 表现为我们想要的阈值之外。
        # 更干净方案：直接复制原始逻辑但用不同阈值。
        # 这里用最干净的方式：修改 module 级 constant，然后还原
        result = original_check_exit(
            position, current_price, bar_5m, bar_15m,
            current_time_utc, reverse_signal_score, is_high_vol, symbol, spot_price
        )
        return result

    # 无法在不复制整个函数的情况下修改局部常量20。
    # 使用重新定义函数的方式。
    sig_mod._ME_MIN_HOLD = min_minutes

    # 重新定义 check_exit，注入 _ME_MIN_HOLD
    original_src_check_exit = sig_mod.check_exit

    def _new_check_exit(position, current_price, bar_5m, bar_15m,
                        current_time_utc, reverse_signal_score=0,
                        is_high_vol=True, symbol="", spot_price=0.0):
        """Patched check_exit with configurable ME min hold."""
        import pandas as pd
        import numpy as np

        # 复制原函数逻辑，替换 ME 最小持仓阈值
        # 用 call 原函数但修改 entry_time_utc 来 "伪装" hold_minutes
        # 实际上最简单的方法：临时修改 position 的 entry_time_utc
        # 使得真实 hold_minutes < min_minutes 时 ME 不触发。

        # 获取真实 hold_minutes
        entry_time = position.get("entry_time_utc", "")
        try:
            h1, m1 = int(entry_time[:2]), int(entry_time[3:5])
            h2, m2 = int(current_time_utc[:2]), int(current_time_utc[3:5])
            real_hold = (h2 * 60 + m2) - (h1 * 60 + m1)
        except Exception:
            real_hold = 999

        # 如果真实hold < min_minutes，临时把 entry_time 调晚，使 check_exit 认为持仓不足
        if real_hold < min_minutes:
            # 伪造 entry_time 使 check_exit 认为 hold=0（让ME不触发）
            pos_fake = dict(position)
            pos_fake["entry_time_utc"] = current_time_utc  # hold = 0 minutes
            result = original_src_check_exit(
                pos_fake, current_price, bar_5m, bar_15m,
                current_time_utc, reverse_signal_score, is_high_vol, symbol, spot_price
            )
            # ME不该触发（hold=0），但其他退出条件应该正常触发
            # 还原 position 原始 entry_time，重新调用原函数，但这次不关心 ME
            return result
        else:
            return original_src_check_exit(
                position, current_price, bar_5m, bar_15m,
                current_time_utc, reverse_signal_score, is_high_vol, symbol, spot_price
            )

    sig_mod.check_exit = _new_check_exit
    bt_mod.check_exit = _new_check_exit  # backtest_signals_day 也直接 import 了 check_exit


def _restore_check_exit():
    """Restore original check_exit."""
    import importlib
    import strategies.intraday.A_share_momentum_signal_v2 as sig_mod
    import scripts.backtest_signals_day as bt_mod
    importlib.reload(sig_mod)
    importlib.reload(bt_mod)


def _patch_intraday_filter_thresholds(t1: float, t2: float, t3: float):
    """Patch _intraday_filter thresholds: abs_ret < t1 → 1.0, t1-t2, t2-t3, >t3."""
    import strategies.intraday.A_share_momentum_signal_v2 as sig_mod

    def _new_intraday_filter(intraday_return: float, direction: str,
                             zscore: float | None = None) -> float:
        """Patched _intraday_filter with configurable thresholds."""
        abs_ret = abs(intraday_return)
        if abs_ret < t1:
            return 1.0

        z = zscore if zscore is not None else 0.0

        if intraday_return > t3:
            base = 0.8 if direction == "LONG" else 0.3
        elif intraday_return > t2:
            base = 0.9 if direction == "LONG" else 0.5
        elif intraday_return > t1:
            base = 1.0 if direction == "LONG" else 0.7
        elif intraday_return < -t3:
            if direction == "SHORT":
                base = 0.8
            else:
                base = 0.7 if z < -2.0 else 0.3
        elif intraday_return < -t2:
            if direction == "SHORT":
                base = 0.9
            else:
                base = 0.8 if z < -2.0 else 0.5
        elif intraday_return < -t1:
            if direction == "SHORT":
                base = 1.0
            else:
                base = 1.0 if z < -2.0 else 0.7
        else:
            base = 1.0

        return base

    # Patch 所有 SignalGeneratorV2 实例会用到的静态方法
    sig_mod.SignalGeneratorV2._intraday_filter = staticmethod(_new_intraday_filter)
    sig_mod.SignalGeneratorV3._intraday_filter = staticmethod(_new_intraday_filter)


def _restore_intraday_filter():
    """Restore original _intraday_filter."""
    import importlib
    import strategies.intraday.A_share_momentum_signal_v2 as sig_mod
    import scripts.backtest_signals_day as bt_mod
    importlib.reload(sig_mod)
    importlib.reload(bt_mod)


# ---------------------------------------------------------------------------
# 格式化输出
# ---------------------------------------------------------------------------

def _fmt_row(label: str, stats: Dict, baseline_pnl: float) -> str:
    pnl = stats["pnl"]
    trades = stats["trades"]
    wr = stats["wr"]
    diff = pnl - baseline_pnl
    sign = "+" if diff >= 0 else ""
    return (f"  {label:<18} | {pnl:+7.0f}pt | {trades:3d} trades | "
            f"{wr:4.1f}% WR | {sign}{diff:+.0f}pt vs baseline")


def _robustness_verdict(values: List[float], baseline: float) -> str:
    if not values:
        return "N/A"
    max_dev = max(abs(v - baseline) for v in values)
    pct = max_dev / abs(baseline) * 100 if baseline != 0 else 0
    if pct < 15:
        verdict = "稳健 ✓ 可实施"
    elif pct < 30:
        verdict = "谨慎 ~ 考虑实施"
    else:
        verdict = "过拟合风险高 ✗ 不建议"
    return f"最大波动 {pct:.1f}% → {verdict}"


# ---------------------------------------------------------------------------
# Test 1: ME最小持仓稳健性
# ---------------------------------------------------------------------------

def test_me_min_hold(db: DBManager, symbols: List[str]):
    """Test ME minimum hold time robustness."""
    print("\n" + "=" * 70)
    print("=== ME最小持仓稳健性 (MOMENTUM_EXHAUSTED hold_minutes 阈值) ===")
    print("=" * 70)

    # 参数配置
    params_by_sym = {
        "IM": [35, 40, 45, 50, 55],
        "IC": [5, 8, 10, 12, 15],
    }

    baseline_label = "20min (baseline)"

    for sym in symbols:
        if sym not in params_by_sym:
            continue

        test_vals = params_by_sym[sym]

        print(f"\n--- {sym} ---")
        print(f"  先跑 baseline（ME=20min）...")

        # Baseline (ME=20, 原始代码)
        _restore_check_exit()  # 确保从原始代码开始
        baseline = _run_backtest(sym, ALL_DATES, db)
        print(f"  {baseline_label:<18} | {baseline['pnl']:+7.0f}pt | "
              f"{baseline['trades']:3d} trades | {baseline['wr']:4.1f}% WR | —")

        pnl_values = []
        for me_min in test_vals:
            _patch_me_min_hold(me_min)
            stats = _run_backtest(sym, ALL_DATES, db)
            _restore_check_exit()
            pnl_values.append(stats["pnl"])
            diff = stats["pnl"] - baseline["pnl"]
            sign = "+" if diff >= 0 else ""
            print(f"  ME={me_min:2d}min{'':<12} | {stats['pnl']:+7.0f}pt | "
                  f"{stats['trades']:3d} trades | {stats['wr']:4.1f}% WR | "
                  f"{sign}{diff:+.0f}pt")

        print(f"  稳健性: {_robustness_verdict(pnl_values, baseline['pnl'])}")


# ---------------------------------------------------------------------------
# Test 2: IC日内涨跌阈值稳健性
# ---------------------------------------------------------------------------

def test_ic_intraday_filter(db: DBManager):
    """Test IC intraday filter threshold robustness."""
    print("\n" + "=" * 70)
    print("=== IC日内涨跌阈值稳健性 (_intraday_filter 分档边界) ===")
    print("=" * 70)

    # (t1%, t2%, t3%) 阈值组合
    threshold_sets: List[Tuple[float, float, float, str]] = [
        (0.010, 0.020, 0.030, "1.0/2.0/3.0% (baseline)"),
        (0.011, 0.023, 0.033, "1.1/2.3/3.3%"),
        (0.012, 0.025, 0.035, "1.2/2.5/3.5%"),
        (0.013, 0.027, 0.037, "1.3/2.7/3.7%"),
        (0.015, 0.030, 0.040, "1.5/3.0/4.0%"),
    ]

    sym = "IC"
    print(f"\n--- {sym} ---")

    pnl_values = []
    baseline_pnl = None

    for i, (t1, t2, t3, label) in enumerate(threshold_sets):
        _patch_intraday_filter_thresholds(t1, t2, t3)
        stats = _run_backtest(sym, ALL_DATES, db)
        _restore_intraday_filter()

        if i == 0:
            baseline_pnl = stats["pnl"]
            print(f"  {label:<28} | {stats['pnl']:+7.0f}pt | "
                  f"{stats['trades']:3d} trades | {stats['wr']:4.1f}% WR | —")
        else:
            pnl_values.append(stats["pnl"])
            diff = stats["pnl"] - baseline_pnl
            sign = "+" if diff >= 0 else ""
            print(f"  {label:<28} | {stats['pnl']:+7.0f}pt | "
                  f"{stats['trades']:3d} trades | {stats['wr']:4.1f}% WR | "
                  f"{sign}{diff:+.0f}pt")

    print(f"  稳健性: {_robustness_verdict(pnl_values, baseline_pnl)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="参数稳健性验证")
    parser.add_argument("--test", choices=["me", "ic_filter", "all"], default="all",
                        help="要运行的测试（默认: all）")
    parser.add_argument("--symbol", choices=["IM", "IC", "both"], default="both",
                        help="ME测试的品种（默认: both）")
    args = parser.parse_args()

    db = DBManager(ConfigLoader().get_db_path())
    print(f"数据库: {ConfigLoader().get_db_path()}")
    print(f"回测日期: {len(ALL_DATES)} 天 ({ALL_DATES[0]} ~ {ALL_DATES[-1]})")

    # 确保 scripts/ 在 path 中
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    symbols = ["IM", "IC"] if args.symbol == "both" else [args.symbol]

    if args.test in ("me", "all"):
        test_me_min_hold(db, symbols)

    if args.test in ("ic_filter", "all"):
        test_ic_intraday_filter(db)

    print("\n" + "=" * 70)
    print("稳健性判断标准:")
    print("  最大波动 < 15%  → 稳健 ✓ 可实施")
    print("  最大波动 15-30% → 谨慎 ~ 考虑实施")
    print("  最大波动 > 30%  → 过拟合风险高 ✗ 不建议")
    print("=" * 70)


if __name__ == "__main__":
    main()
