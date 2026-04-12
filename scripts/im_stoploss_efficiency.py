#!/usr/bin/env python3
"""IM止损效率精确分析：不同止损宽度的全策略PnL影响 + IS/OOS双窗口。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from data.storage.db_manager import get_db
from scripts.backtest_signals_day import run_day
from strategies.intraday.A_share_momentum_signal_v2 import SYMBOL_PROFILES

SYM = 'IM'
SPOT = '000905'  # IM对应000852
SPOT_IM = '000852'
STOP_LEVELS = [0.002, 0.003, 0.004, 0.005, 0.007, 0.010]  # 0.2% ~ 1.0%


def get_dates(db):
    df = db.query_df(
        f"SELECT DISTINCT substr(datetime,1,10) as d FROM index_min "
        f"WHERE symbol='{SPOT_IM}' AND period=300 ORDER BY d"
    )
    dates = [d.replace('-', '') for d in df['d'].tolist()]
    return dates[-900:] if len(dates) > 900 else dates


def _run_one_day(args):
    td, sl_pct = args
    SYMBOL_PROFILES[SYM]["signal_threshold"] = 55
    SYMBOL_PROFILES[SYM]["stop_loss_pct"] = sl_pct
    db = get_db()
    trades = run_day(SYM, td, db, verbose=False)
    full = [t for t in trades if not t.get("partial")]
    for t in full:
        t['trade_date'] = td
        t['symbol'] = SYM
        t['sl_pct'] = sl_pct
        # hold_bars
        eb = t.get('entry_time', '00:00')
        xb = t.get('exit_time', '00:00')
        try:
            eh, em = int(eb[:2]), int(eb[3:5])
            xh, xm = int(xb[:2]), int(xb[3:5])
            t['hold_bars'] = max(1, ((xh - eh) * 60 + (xm - em)) // 5)
        except Exception:
            t['hold_bars'] = 0
    return full


def main():
    print("=" * 60)
    print("  IM 止损效率精确分析")
    print("=" * 60)

    db = get_db()
    dates = get_dates(db)
    n = len(dates)
    is_dates = set(dates[-219:])
    oos_dates = set(dates[:-219])
    n_workers = min(cpu_count(), 8)

    doc = ["# IM 止损效率精确分析\n"]
    doc.append(f"数据: IM {n}天 ({dates[0]}~{dates[-1]})")
    doc.append(f"IS(训练窗口): 最近219天, OOS: 早期681天")
    doc.append(f"测试止损: {[f'{s*100:.1f}%' for s in STOP_LEVELS]}\n")

    # ═══════════════════════════════════════════════
    # B.3+B.5: 全样本不同止损的PnL
    # ═══════════════════════════════════════════════
    all_results = {}

    for sl in STOP_LEVELS:
        print(f"\n[IM] 止损={sl*100:.1f}%...")
        args = [(td, sl) for td in dates]
        with Pool(n_workers) as pool:
            day_results = pool.map(_run_one_day, args)
        trades = [t for day in day_results for t in day]
        all_results[sl] = trades
        total_pnl = sum(t['pnl_pts'] for t in trades)
        n_trades = len(trades)
        reason_col = 'reason' if trades and 'reason' in trades[0] else 'exit_reason'
        n_sl = sum(1 for t in trades if 'STOP' in str(t.get(reason_col, '')).upper())
        sl_pnl = sum(t['pnl_pts'] for t in trades if 'STOP' in str(t.get(reason_col, '')).upper())
        print(f"  {n_trades}笔, PnL={total_pnl:+.0f}, 止损{n_sl}笔({n_sl/n_trades*100:.0f}%), 止损PnL={sl_pnl:+.0f}")

    # B.5 全策略PnL影响
    doc.append("## B.5 全策略PnL影响（900天全样本）\n")
    doc.append("| 止损宽度 | 总PnL | 笔数 | 止损笔数 | 止损率 | 止损PnL | 非止损PnL | vs 0.3%差异 |")
    doc.append("|---------|-------|------|---------|-------|--------|---------|-----------|")

    baseline_pnl = None
    for sl in STOP_LEVELS:
        trades = all_results[sl]
        total = sum(t['pnl_pts'] for t in trades)
        n_t = len(trades)
        reason_col = 'reason' if trades and 'reason' in trades[0] else 'exit_reason'
        sl_trades = [t for t in trades if 'STOP' in str(t.get(reason_col, '')).upper()]
        non_sl = [t for t in trades if 'STOP' not in str(t.get(reason_col, '')).upper()]
        n_sl = len(sl_trades)
        sl_pnl = sum(t['pnl_pts'] for t in sl_trades)
        non_sl_pnl = sum(t['pnl_pts'] for t in non_sl)

        if sl == 0.003:
            baseline_pnl = total
        diff = total - baseline_pnl if baseline_pnl is not None else 0

        doc.append(f"| {sl*100:.1f}% | {total:+.0f} | {n_t} | {n_sl} ({n_sl/n_t*100:.0f}%) | "
                   f"{n_sl/n_t*100:.0f}% | {sl_pnl:+.0f} | {non_sl_pnl:+.0f} | {diff:+.0f} |")
    doc.append("")

    # ═══════════════════════════════════════════════
    # B.4: 被0.3%止损的trade在其他宽度下的表现
    # ═══════════════════════════════════════════════
    doc.append("## B.4 被0.3%止损的trade在不同宽度下的命运\n")

    # 找到0.3%下被止损的trade的(trade_date, entry_time)
    trades_03 = all_results[0.003]
    reason_col = 'reason' if trades_03 and 'reason' in trades_03[0] else 'exit_reason'
    stopped_03 = [(t['trade_date'], t.get('entry_time', '')) for t in trades_03
                  if 'STOP' in str(t.get(reason_col, '')).upper()]
    stopped_03_set = set(stopped_03)

    doc.append(f"0.3%下被止损: {len(stopped_03)}笔\n")
    doc.append("这些trade在不同止损宽度下的表现:\n")
    doc.append("| 止损宽度 | 仍被止损 | 被其他原因出场 | 平均PnL | 总PnL | vs实际(-25.2)差异 |")
    doc.append("|---------|---------|-------------|--------|-------|-----------------|")

    for sl in STOP_LEVELS:
        trades = all_results[sl]
        # 匹配同一笔trade
        matched = []
        for t in trades:
            key = (t['trade_date'], t.get('entry_time', ''))
            if key in stopped_03_set:
                matched.append(t)

        if not matched:
            continue

        still_stopped = sum(1 for t in matched if 'STOP' in str(t.get(reason_col, '')).upper())
        other_exit = len(matched) - still_stopped
        avg_pnl = np.mean([t['pnl_pts'] for t in matched])
        total_pnl = sum(t['pnl_pts'] for t in matched)
        diff_avg = avg_pnl - (-25.2)

        doc.append(f"| {sl*100:.1f}% | {still_stopped} | {other_exit} | {avg_pnl:+.1f} | "
                   f"{total_pnl:+.0f} | {diff_avg:+.1f} |")
    doc.append("")

    # ═══════════════════════════════════════════════
    # B.6: IS/OOS双窗口
    # ═══════════════════════════════════════════════
    doc.append("## B.6 IS/OOS双窗口分析\n")

    doc.append("### OOS (681天)\n")
    doc.append("| 止损宽度 | PnL | 笔数 | 日均PnL |")
    doc.append("|---------|-----|------|--------|")
    oos_baseline = None
    for sl in STOP_LEVELS:
        trades = [t for t in all_results[sl] if t['trade_date'] in oos_dates]
        pnl = sum(t['pnl_pts'] for t in trades)
        daily = pnl / 681
        if sl == 0.003:
            oos_baseline = pnl
        doc.append(f"| {sl*100:.1f}% | {pnl:+.0f} | {len(trades)} | {daily:+.1f} |")

    doc.append(f"\n### IS (219天)\n")
    doc.append("| 止损宽度 | PnL | 笔数 | 日均PnL |")
    doc.append("|---------|-----|------|--------|")
    is_baseline = None
    for sl in STOP_LEVELS:
        trades = [t for t in all_results[sl] if t['trade_date'] in is_dates]
        pnl = sum(t['pnl_pts'] for t in trades)
        daily = pnl / 219
        if sl == 0.003:
            is_baseline = pnl
        doc.append(f"| {sl*100:.1f}% | {pnl:+.0f} | {len(trades)} | {daily:+.1f} |")

    doc.append(f"\n### IS/OOS最优对比\n")
    doc.append("| 止损宽度 | OOS PnL | IS PnL | IS/OOS效率比 |")
    doc.append("|---------|---------|--------|------------|")
    for sl in STOP_LEVELS:
        oos_t = [t for t in all_results[sl] if t['trade_date'] in oos_dates]
        is_t = [t for t in all_results[sl] if t['trade_date'] in is_dates]
        oos_daily = sum(t['pnl_pts'] for t in oos_t) / 681
        is_daily = sum(t['pnl_pts'] for t in is_t) / 219
        ratio = is_daily / oos_daily if oos_daily > 0 else 99
        doc.append(f"| {sl*100:.1f}% | {sum(t['pnl_pts'] for t in oos_t):+.0f} | "
                   f"{sum(t['pnl_pts'] for t in is_t):+.0f} | {ratio:.2f} |")
    doc.append("")

    # ═══════════════════════════════════════════════
    # B.7 综合判定
    # ═══════════════════════════════════════════════
    doc.append("## B.7 综合判定\n")

    # 找全样本最优
    best_sl = max(STOP_LEVELS, key=lambda sl: sum(t['pnl_pts'] for t in all_results[sl]))
    best_pnl = sum(t['pnl_pts'] for t in all_results[best_sl])
    current_pnl = sum(t['pnl_pts'] for t in all_results[0.003])

    doc.append(f"全样本最优: {best_sl*100:.1f}% ({best_pnl:+.0f}pt)")
    doc.append(f"当前0.3%: {current_pnl:+.0f}pt")
    doc.append(f"差异: {best_pnl - current_pnl:+.0f}pt\n")

    # 找OOS最优
    oos_best_sl = max(STOP_LEVELS, key=lambda sl: sum(t['pnl_pts'] for t in all_results[sl] if t['trade_date'] in oos_dates))
    oos_best = sum(t['pnl_pts'] for t in all_results[oos_best_sl] if t['trade_date'] in oos_dates)
    oos_current = sum(t['pnl_pts'] for t in all_results[0.003] if t['trade_date'] in oos_dates)

    doc.append(f"OOS最优: {oos_best_sl*100:.1f}% ({oos_best:+.0f}pt)")
    doc.append(f"OOS当前0.3%: {oos_current:+.0f}pt")
    doc.append(f"OOS差异: {oos_best - oos_current:+.0f}pt\n")

    if best_sl == 0.003 and oos_best_sl == 0.003:
        doc.append("**判定SL1: 0.3%是稳健最优** ✓")
    elif best_sl != 0.003 and oos_best_sl != 0.003 and best_sl == oos_best_sl:
        doc.append(f"**判定SL2: {best_sl*100:.1f}%更优（全样本+OOS一致）**")
    elif best_sl == 0.003 and oos_best_sl != 0.003:
        doc.append(f"**判定SL3: 0.3%只在IS最优，OOS上{oos_best_sl*100:.1f}%更优**")
    elif oos_best_sl == 0.003 and best_sl != 0.003:
        doc.append(f"**判定SL1(变体): OOS上0.3%最优，全样本上{best_sl*100:.1f}%更优（IS拉高）**")
    else:
        doc.append(f"**判定SL4: 全样本最优{best_sl*100:.1f}%，OOS最优{oos_best_sl*100:.1f}%，不一致**")

    report = "\n".join(doc)
    path = Path("tmp") / "principle25_and_im_stoploss_efficiency.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
