#!/usr/bin/env python3
"""
score_exit_robustness.py
--------------------------
ME/TC Score确认退出 稳健性验证（5项测试）

复用 score_confirmed_exit_research.py 的核心函数，对以下维度验证：
  测试1 - 时间分段（前半段/后半段）
  测试2 - 每周表现（5周各自delta）
  测试3 - 阈值系数敏感性（0.6/0.8/1.0/1.2）
  测试4 - 单日集中度（每天delta，集中性分析）
  测试5 - 仅ME vs ME+TC vs ME+TC+MID_BREAK

用法：
    python scripts/score_exit_robustness.py
    python scripts/score_exit_robustness.py --symbol IM
    python scripts/score_exit_robustness.py --tests 1,3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.storage.db_manager import DBManager
from config.config_loader import ConfigLoader

# Import everything from the research script
from scripts.score_confirmed_exit_research import (
    run_day_both, run_multi, _summarize,
    SCORE_CONFIRMABLE_REASONS, UNCONDITIONAL_EXIT_REASONS,
    IM_MULT, IC_MULT,
)

# ─── Constants ────────────────────────────────────────────────────────────────

ALL_DATES = (
    "20260204,20260205,20260206,20260209,20260210,20260211,20260212,20260213,"
    "20260225,20260226,20260227,20260302,20260303,20260304,20260305,20260306,"
    "20260309,20260310,20260311,20260312,20260313,20260316,20260317,20260318,"
    "20260319,20260320,20260323,20260324,20260325,20260326,20260327,20260328,"
    "20260401,20260402"
).split(",")

# date boundaries for weekly splits
WEEK_SLICES = [
    (0, 5),    # W1: dates[0:5]
    (5, 10),   # W2: dates[5:10]
    (10, 17),  # W3: dates[10:17]
    (17, 24),  # W4: dates[17:24]
    (24, 34),  # W5: dates[24:]
]

SYM_BASE_THRESHOLD = {"IM": 60, "IC": 65}
THRESHOLD_MULTIPLIERS = [0.6, 0.8, 1.0, 1.2]


# ─── Helper: run with custom confirmable reasons and threshold ─────────────────

def run_day_custom(sym: str, td: str, db: DBManager,
                   confirmable_reasons: set,
                   override_threshold: Optional[int] = None) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Like run_day_both but with custom SCORE_CONFIRMABLE_REASONS and threshold.
    We monkey-patch the module-level constant, run, then restore.
    """
    import scripts.score_confirmed_exit_research as research_mod
    from strategies.intraday.A_share_momentum_signal_v2 import SYMBOL_PROFILES, _DEFAULT_PROFILE

    # If no customization needed, use the standard function
    if confirmable_reasons == SCORE_CONFIRMABLE_REASONS and override_threshold is None:
        return run_day_both(sym, td, db)

    # We need to re-implement the score gate with custom params.
    # Simplest approach: patch the constant before calling, restore after.
    original_reasons = research_mod.SCORE_CONFIRMABLE_REASONS

    # Patch effective threshold by temporarily modifying SYMBOL_PROFILES
    _sym_prof = SYMBOL_PROFILES.get(sym, _DEFAULT_PROFILE)
    original_threshold = _sym_prof.get("signal_threshold", 60)

    try:
        research_mod.SCORE_CONFIRMABLE_REASONS = confirmable_reasons
        if override_threshold is not None:
            _sym_prof["signal_threshold"] = override_threshold
        return run_day_both(sym, td, db)
    finally:
        research_mod.SCORE_CONFIRMABLE_REASONS = original_reasons
        if override_threshold is not None:
            _sym_prof["signal_threshold"] = original_threshold


def run_multi_custom(sym: str, dates: List[str], db: DBManager,
                     confirmable_reasons: set,
                     override_threshold: Optional[int] = None,
                     quiet: bool = True) -> Tuple[Dict, Dict, List[Dict]]:
    """Run multiple dates with custom params."""
    all_trades_b: List[Dict] = []
    all_trades_c: List[Dict] = []
    all_blocked: List[Dict] = []

    for td in dates:
        tb, tc, blocked = run_day_custom(sym, td, db, confirmable_reasons, override_threshold)
        all_trades_b.extend(tb)
        all_trades_c.extend(tc)
        all_blocked.extend(blocked)

    sum_b = _summarize(all_trades_b, "Baseline", sym)
    sum_c = _summarize(all_trades_c, "Score确认", sym)
    return sum_b, sum_c, all_blocked


def _pnl(trades: List[Dict]) -> float:
    return sum(t["pnl_pts"] for t in trades if not t.get("partial"))


def _n(trades: List[Dict]) -> int:
    return len([t for t in trades if not t.get("partial")])


# ─── Test 1: 时间分段 ──────────────────────────────────────────────────────────

def test_time_split(syms: List[str], db: DBManager):
    print(f"\n{'='*80}")
    print(" 测试1: 时间分段（前17天 vs 后17天）")
    print(f"{'='*80}")

    half1 = ALL_DATES[:17]   # 20260204~20260306
    half2 = ALL_DATES[17:]   # 20260309~20260402
    assert len(half1) == 17 and len(half2) == 17, f"Expected 17+17, got {len(half1)}+{len(half2)}"

    segments = [
        ("前半段(17天)", half1),
        ("后半段(17天)", half2),
        ("全段(34天)", ALL_DATES),
    ]

    print(f"\n  {'时间分段':<15} {'Baseline':>10} {'Score确认':>10} {'Delta':>8} {'方向'}")
    print(f"  {'─'*60}")

    results_by_seg = {}
    for seg_name, dates in segments:
        grand_b = 0
        grand_c = 0
        for sym in syms:
            print(f"  运行 {sym} {seg_name} ({len(dates)}天)...", end="", flush=True)
            sb, sc, _ = run_multi_custom(sym, dates, db, SCORE_CONFIRMABLE_REASONS)
            grand_b += sb["pnl"]
            grand_c += sc["pnl"]
            print(f" base={sb['pnl']:+.0f} conf={sc['pnl']:+.0f}")
        delta = grand_c - grand_b
        ok = "OK" if delta >= 0 else "FAIL"
        results_by_seg[seg_name] = (grand_b, grand_c, delta)
        print(f"  {seg_name:<15} {grand_b:>+10.0f} {grand_c:>+10.0f} {delta:>+8.0f} [{ok}]")

    print(f"\n  结论:")
    all_ok = all(v[2] >= 0 for v in results_by_seg.values())
    for seg_name, (b, c, d) in results_by_seg.items():
        print(f"    {seg_name}: delta={d:+.0f}pt ({'改善' if d>=0 else '恶化'})")
    if all_ok:
        print(f"  ** 两个半段均改善，时间稳定性确认 **")
    else:
        fails = [k for k, v in results_by_seg.items() if v[2] < 0]
        print(f"  ** 以下分段恶化: {fails} **")


# ─── Test 2: 每周表现 ──────────────────────────────────────────────────────────

def test_weekly(syms: List[str], db: DBManager):
    print(f"\n{'='*80}")
    print(" 测试2: 每周表现（5周）")
    print(f"{'='*80}")

    week_names = ["W1", "W2", "W3", "W4", "W5"]
    week_dates = [ALL_DATES[s:e] for s, e in WEEK_SLICES]

    print(f"\n  {'周次':<6} {'日期范围':<22} {'Baseline':>10} {'Score确认':>10} {'Delta':>8} {'方向'}")
    print(f"  {'─'*70}")

    weekly_deltas = []
    for wname, wdates in zip(week_names, week_dates):
        grand_b = 0
        grand_c = 0
        for sym in syms:
            sb, sc, _ = run_multi_custom(sym, wdates, db, SCORE_CONFIRMABLE_REASONS)
            grand_b += sb["pnl"]
            grand_c += sc["pnl"]
        delta = grand_c - grand_b
        weekly_deltas.append(delta)
        date_range = f"{wdates[0]}~{wdates[-1]}"
        ok = "OK" if delta >= 0 else "FAIL"
        print(f"  {wname:<6} {date_range:<22} {grand_b:>+10.0f} {grand_c:>+10.0f} {delta:>+8.0f} [{ok}]")

    # Full period
    grand_b_all = 0
    grand_c_all = 0
    for sym in syms:
        sb, sc, _ = run_multi_custom(sym, ALL_DATES, db, SCORE_CONFIRMABLE_REASONS)
        grand_b_all += sb["pnl"]
        grand_c_all += sc["pnl"]
    delta_all = grand_c_all - grand_b_all

    print(f"  {'全段':<6} {'':22} {grand_b_all:>+10.0f} {grand_c_all:>+10.0f} {delta_all:>+8.0f}")

    weeks_win = sum(1 for d in weekly_deltas if d >= 0)
    print(f"\n  Score确认胜出周: {weeks_win}/5")
    if weeks_win >= 3:
        print(f"  ** 超过3/5周改善，每周稳定性确认 **")
    else:
        print(f"  ** 仅{weeks_win}/5周改善，稳定性存疑 **")


# ─── Test 3: 阈值敏感性 ────────────────────────────────────────────────────────

def test_threshold_sensitivity(syms: List[str], db: DBManager):
    print(f"\n{'='*80}")
    print(" 测试3: 阈值系数敏感性（0.6/0.8/1.0/1.2）")
    print(f"{'='*80}")
    print(f"\n  注：系数×基础阈值(IM=60,IC=65)得到实际阈值")

    # First compute baseline (once, threshold doesn't affect baseline)
    grand_baseline = 0
    for sym in syms:
        sb, _, _ = run_multi_custom(sym, ALL_DATES, db, SCORE_CONFIRMABLE_REASONS)
        grand_baseline += sb["pnl"]

    print(f"\n  {'阈值系数':<10} {'IM阈值':<8} {'IC阈值':<8} {'IM+IC PnL':>12} {'vs Baseline':>12} {'阻止次数':>10}")
    print(f"  {'─'*70}")

    for mult in THRESHOLD_MULTIPLIERS:
        grand_c = 0
        total_blocked = 0
        for sym in syms:
            base_thr = SYM_BASE_THRESHOLD[sym]
            thr = int(round(base_thr * mult))
            sb, sc, blocked = run_multi_custom(sym, ALL_DATES, db,
                                               SCORE_CONFIRMABLE_REASONS,
                                               override_threshold=thr)
            grand_c += sc["pnl"]
            total_blocked += len(blocked)

        delta = grand_c - grand_baseline
        im_thr = int(round(SYM_BASE_THRESHOLD["IM"] * mult))
        ic_thr = int(round(SYM_BASE_THRESHOLD["IC"] * mult))
        marker = " << 当前" if abs(mult - 1.0) < 0.01 else ""
        print(f"  {mult:<10.1f} {im_thr:<8} {ic_thr:<8} {grand_c:>+12.0f} {delta:>+12.0f} {total_blocked:>10}{marker}")

    print(f"\n  Baseline: {grand_baseline:+.0f}pt")
    print(f"\n  稳健性判断：")
    print(f"  - 若0.8~1.2区间内delta均>0，说明阈值选择不敏感")
    print(f"  - 若0.6时delta最大，说明更宽松更好（考虑降低阈值）")
    print(f"  - 若1.2时delta最大，说明更严格更好（考虑提高阈值）")


# ─── Test 4: 单日集中度 ────────────────────────────────────────────────────────

def test_daily_concentration(syms: List[str], db: DBManager):
    print(f"\n{'='*80}")
    print(" 测试4: 单日集中度（每天delta分布）")
    print(f"{'='*80}")

    print(f"\n  {'日期':<12} {'Baseline':>10} {'Score确认':>10} {'Delta':>8}")
    print(f"  {'─'*45}")

    daily_results = []
    for td in ALL_DATES:
        day_b = 0
        day_c = 0
        for sym in syms:
            tb, tc, _ = run_day_both(sym, td, db)
            day_b += _pnl(tb)
            day_c += _pnl(tc)
        delta = day_c - day_b
        daily_results.append((td, day_b, day_c, delta))
        marker = " *" if abs(delta) > 50 else ""
        print(f"  {td:<12} {day_b:>+10.0f} {day_c:>+10.0f} {delta:>+8.0f}{marker}")

    total_b = sum(r[1] for r in daily_results)
    total_c = sum(r[2] for r in daily_results)
    total_delta = total_c - total_b
    print(f"  {'─'*45}")
    print(f"  {'合计':<12} {total_b:>+10.0f} {total_c:>+10.0f} {total_delta:>+8.0f}")

    deltas = [r[3] for r in daily_results]
    positive_days = sum(1 for d in deltas if d > 0)
    negative_days = sum(1 for d in deltas if d < 0)
    neutral_days = sum(1 for d in deltas if d == 0)

    # Top 3 contributors
    sorted_by_delta = sorted(daily_results, key=lambda x: x[3], reverse=True)
    top3 = sorted_by_delta[:3]
    top3_sum = sum(r[3] for r in top3)

    # Max single day delta
    max_delta = max(deltas)
    max_delta_day = sorted_by_delta[0][0]
    max_pct = max_delta / abs(total_delta) * 100 if total_delta != 0 else 0

    print(f"\n  统计摘要:")
    print(f"  Delta>0天数: {positive_days}/34  Delta<0天数: {negative_days}/34  Delta=0: {neutral_days}/34")
    print(f"  最大单日delta: {max_delta:+.0f}pt ({max_delta_day}, 占总改善{max_pct:.0f}%)")
    top3_pct = top3_sum / abs(total_delta) * 100 if total_delta != 0 else 0
    print(f"  Top3天合计delta: {top3_sum:+.0f}pt (占{top3_pct:.0f}%)")
    print(f"  Top3天: {', '.join(f'{r[0]}({r[3]:+.0f}pt)' for r in top3)}")

    if max_pct > 50:
        print(f"\n  ** 警告：最大单日贡献{max_pct:.0f}%>50%，结果可能集中 **")
    else:
        print(f"\n  ** 最大单日贡献{max_pct:.0f}%<50%，改善较分散 **")

    if positive_days >= 17:  # >= 50%
        print(f"  ** {positive_days}/34天改善（{positive_days/34*100:.0f}%），每日稳定性{'良好' if positive_days >= 20 else '一般'} **")
    else:
        print(f"  ** 仅{positive_days}/34天改善，警惕少数天驱动 **")


# ─── Test 5: 仅ME vs ME+TC vs ME+TC+MID_BREAK ────────────────────────────────

def test_reason_decomposition(syms: List[str], db: DBManager):
    print(f"\n{'='*80}")
    print(" 测试5: 阻止范围分解（只ME vs ME+TC vs ME+TC+MID）")
    print(f"{'='*80}")

    configs = [
        ("只ME",            {"MOMENTUM_EXHAUSTED"}),
        ("ME+TC",           {"MOMENTUM_EXHAUSTED", "TREND_COMPLETE"}),
        ("ME+TC+MID",       {"MOMENTUM_EXHAUSTED", "TREND_COMPLETE", "MID_BREAK"}),
    ]

    # Baseline (computed once)
    grand_baseline = 0
    for sym in syms:
        sb, _, _ = run_multi_custom(sym, ALL_DATES, db, set())  # empty = no gating
        grand_baseline += sb["pnl"]
    # Actually baseline means no gating at all, use run_day_both which uses the standard confirmable reasons
    # Let's recompute correctly: baseline = trades_b from run_day_both
    grand_baseline = 0
    for sym in syms:
        sb, sc, _ = run_multi_custom(sym, ALL_DATES, db, SCORE_CONFIRMABLE_REASONS)
        grand_baseline += sb["pnl"]

    print(f"\n  Baseline (无score确认): {grand_baseline:+.0f}pt")
    print(f"\n  {'方案':<20} {'IM+IC PnL':>12} {'vs Baseline':>12} {'阻止次数':>10}")
    print(f"  {'─'*60}")

    for label, reasons in configs:
        grand_c = 0
        total_blocked = 0
        for sym in syms:
            _, sc, blocked = run_multi_custom(sym, ALL_DATES, db, reasons)
            grand_c += sc["pnl"]
            total_blocked += len(blocked)
        delta = grand_c - grand_baseline
        marker = " << 当前" if label == "ME+TC+MID" else ""
        print(f"  {label:<20} {grand_c:>+12.0f} {delta:>+12.0f} {total_blocked:>10}{marker}")

    print(f"\n  判断:")
    print(f"  - 若'只ME'已有大部分改善，说明ME是主要贡献者")
    print(f"  - 若逐步增加到MID改善逐渐增大，说明每类都有贡献")
    print(f"  - 若某步添加后改善下降，说明该类阻止有害")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ME/TC score确认退出 稳健性验证")
    parser.add_argument("--symbol", default="ALL", help="IM / IC / ALL (default: ALL)")
    parser.add_argument("--tests", default="1,2,3,4,5",
                        help="逗号分隔的测试编号 (default: 1,2,3,4,5)")
    args = parser.parse_args()

    db = DBManager(ConfigLoader().get_db_path())

    syms = ["IM", "IC"] if args.symbol == "ALL" else [args.symbol.upper()]
    tests = [int(x.strip()) for x in args.tests.split(",")]

    print(f"\n{'#'*80}")
    print(f"# ME/TC Score确认退出 稳健性验证")
    print(f"# 品种: {'+'.join(syms)}")
    print(f"# 日期: {ALL_DATES[0]}~{ALL_DATES[-1]} ({len(ALL_DATES)}天)")
    print(f"# 测试: {tests}")
    print(f"{'#'*80}")

    if 1 in tests:
        test_time_split(syms, db)

    if 2 in tests:
        test_weekly(syms, db)

    if 3 in tests:
        test_threshold_sensitivity(syms, db)

    if 4 in tests:
        test_daily_concentration(syms, db)

    if 5 in tests:
        test_reason_decomposition(syms, db)

    print(f"\n{'='*80}")
    print(" 稳健性验证完成")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
