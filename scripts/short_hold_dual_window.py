#!/usr/bin/env python3
"""短持仓现象681/219双窗口验证。复用现有backtest数据，按日期切片。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from data.storage.db_manager import get_db
from scripts.backtest_signals_day import run_day
from strategies.intraday.A_share_momentum_signal_v2 import SYMBOL_PROFILES

SPOTS = {'IC': '000905', 'IM': '000852'}


def get_dates(db, spot):
    df = db.query_df(
        f"SELECT DISTINCT substr(datetime,1,10) as d FROM index_min "
        f"WHERE symbol='{spot}' AND period=300 ORDER BY d"
    )
    dates = [d.replace('-', '') for d in df['d'].tolist()]
    return dates[-900:] if len(dates) > 900 else dates


def _run_one_day(args):
    td, sym, thr = args
    SYMBOL_PROFILES[sym]["signal_threshold"] = thr
    db = get_db()
    trades = run_day(sym, td, db, verbose=False)
    full = [t for t in trades if not t.get("partial")]
    for t in full:
        t['trade_date'] = td
        t['symbol'] = sym
        eb = t.get('entry_time', '00:00')
        xb = t.get('exit_time', '00:00')
        try:
            eh, em = int(eb[:2]), int(eb[3:5])
            xh, xm = int(xb[:2]), int(xb[3:5])
            t['hold_minutes'] = (xh - eh) * 60 + (xm - em)
            t['hold_bars'] = max(1, t['hold_minutes'] // 5)
        except Exception:
            t['hold_minutes'] = 0
            t['hold_bars'] = 0
    return full


def hold_group(bars):
    if bars <= 2: return '极短(<=2bar)'
    if bars <= 4: return '短(3-4bar)'
    if bars <= 12: return '中(5-12bar)'
    return '长(>12bar)'


def window_stats(tdf, label, doc):
    """对一个时间窗口的trade做持仓分组统计。"""
    groups = ['极短(<=2bar)', '短(3-4bar)', '中(5-12bar)', '长(>12bar)']
    tdf = tdf.copy()
    tdf['hold_group'] = tdf['hold_bars'].apply(hold_group)
    total = len(tdf)

    for g in groups:
        sub = tdf[tdf['hold_group'] == g]
        if len(sub) == 0:
            continue
        wr = (sub['pnl_pts'] > 0).sum() / len(sub) * 100
        doc.append(f"| {label} | {g} | {len(sub)} | {len(sub)/total*100:.0f}% | "
                   f"{sub['pnl_pts'].mean():+.1f} | {sub['pnl_pts'].sum():+.0f} | {wr:.0f}% |")


def exit_reason_stats(tdf, label, doc):
    """短持仓(<=4bar)的exit_reason分布。"""
    short = tdf[tdf['hold_bars'] <= 4]
    if len(short) == 0:
        return
    reason_col = 'reason' if 'reason' in short.columns else 'exit_reason'
    if reason_col not in short.columns:
        return
    for reason in short[reason_col].value_counts().index:
        sub = short[short[reason_col] == reason]
        if len(sub) >= 5:
            doc.append(f"| {label} | {reason} | {len(sub)} | {len(sub)/len(short)*100:.0f}% | "
                       f"{sub['pnl_pts'].mean():+.1f} |")


def main():
    print("=" * 60)
    print("  短持仓现象 681/219 双窗口验证")
    print("=" * 60)

    db = get_db()
    n_workers = min(cpu_count(), 8)
    doc = ["# 短持仓现象 681/219 双窗口验证\n"]

    all_data = {}
    all_dates = {}
    for sym in ['IM', 'IC']:
        dates = get_dates(db, SPOTS[sym])
        thr = SYMBOL_PROFILES[sym].get("signal_threshold", 60)
        print(f"\n[{sym}] 收集交易数据 (thr={thr})...")
        args = [(td, sym, thr) for td in dates]
        with Pool(n_workers) as pool:
            day_results = pool.map(_run_one_day, args)
        trades = [t for day in day_results for t in day]
        tdf = pd.DataFrame(trades)
        all_data[sym] = tdf
        all_dates[sym] = dates
        print(f"  {len(tdf)}笔, {len(dates)}天")

    # ═══════════════════════════════════════════════
    # B.1 时间切片
    # ═══════════════════════════════════════════════
    doc.append("## B.1 时间切片确认\n")
    splits = {}
    for sym in ['IM', 'IC']:
        dates = all_dates[sym]
        is_dates = set(dates[-219:])  # 最近219天 = IS(训练窗口)
        oos_dates = set(dates[:-219])  # 更早的681天 = OOS
        splits[sym] = {'is': is_dates, 'oos': oos_dates}
        doc.append(f"- {sym}: 总{len(dates)}天, IS(最近219天)={len(is_dates)}天 "
                   f"({sorted(is_dates)[0]}~{sorted(is_dates)[-1]}), "
                   f"OOS(早期681天)={len(oos_dates)}天 ({sorted(oos_dates)[0]}~{sorted(oos_dates)[-1]})")
    doc.append("")

    # ═══════════════════════════════════════════════
    # B.2 持仓时长分组（双窗口）
    # ═══════════════════════════════════════════════
    doc.append("## B.2 持仓时长分组（双窗口）\n")

    for sym in ['IM', 'IC']:
        tdf = all_data[sym]
        is_tdf = tdf[tdf['trade_date'].isin(splits[sym]['is'])]
        oos_tdf = tdf[tdf['trade_date'].isin(splits[sym]['oos'])]

        doc.append(f"### {sym}\n")
        doc.append("| 窗口 | 持仓时长 | 笔数 | 占比 | AvgPnL | 累计PnL | 胜率 |")
        doc.append("|------|---------|------|------|--------|---------|------|")
        window_stats(oos_tdf, f'OOS681', doc)
        window_stats(is_tdf, f'IS219', doc)
        doc.append("")

    # ═══════════════════════════════════════════════
    # B.3 exit_reason分布（双窗口，短持仓）
    # ═══════════════════════════════════════════════
    doc.append("## B.3 短持仓(<=4bar) Exit Reason（双窗口）\n")

    for sym in ['IM', 'IC']:
        tdf = all_data[sym]
        is_tdf = tdf[tdf['trade_date'].isin(splits[sym]['is'])]
        oos_tdf = tdf[tdf['trade_date'].isin(splits[sym]['oos'])]

        doc.append(f"### {sym}\n")
        doc.append("| 窗口 | exit_reason | 笔数 | 占比 | AvgPnL |")
        doc.append("|------|-----------|------|------|--------|")
        exit_reason_stats(oos_tdf, 'OOS681', doc)
        exit_reason_stats(is_tdf, 'IS219', doc)
        doc.append("")

    # ═══════════════════════════════════════════════
    # B.4 四个关键现象对比
    # ═══════════════════════════════════════════════
    doc.append("## B.4 四个关键现象双窗口对比\n")

    for sym in ['IM', 'IC']:
        tdf = all_data[sym]
        is_tdf = tdf[tdf['trade_date'].isin(splits[sym]['is'])]
        oos_tdf = tdf[tdf['trade_date'].isin(splits[sym]['oos'])]

        # 现象1: IM短持仓止损率
        doc.append(f"### 现象1: {sym}短持仓(<=4bar)止损率\n")
        doc.append("| 窗口 | 短持仓总笔 | STOP_LOSS笔 | 止损率 | STOP_LOSS总亏损 |")
        doc.append("|------|---------|-----------|------|-------------|")
        reason_col = 'reason' if 'reason' in tdf.columns else 'exit_reason'
        for label, wdf in [('900天', tdf), ('OOS681', oos_tdf), ('IS219', is_tdf)]:
            short = wdf[wdf['hold_bars'] <= 4]
            if reason_col in short.columns:
                sl = short[short[reason_col].str.contains('STOP|stop', na=False)]
                rate = len(sl) / len(short) * 100 if len(short) > 0 else 0
                doc.append(f"| {label} | {len(short)} | {len(sl)} | {rate:.0f}% | {sl['pnl_pts'].sum():+.0f} |")
        doc.append("")

    # 现象2: IC极短持仓盈利
    doc.append("### 现象2: IC极短(<=2bar)盈利\n")
    doc.append("| 窗口 | 笔数 | AvgPnL | 累计PnL | 胜率 |")
    doc.append("|------|------|--------|---------|------|")
    ic_tdf = all_data['IC']
    ic_is = ic_tdf[ic_tdf['trade_date'].isin(splits['IC']['is'])]
    ic_oos = ic_tdf[ic_tdf['trade_date'].isin(splits['IC']['oos'])]
    for label, wdf in [('900天', ic_tdf), ('OOS681', ic_oos), ('IS219', ic_is)]:
        sub = wdf[wdf['hold_bars'] <= 2]
        if len(sub) > 0:
            wr = (sub['pnl_pts'] > 0).sum() / len(sub) * 100
            doc.append(f"| {label} | {len(sub)} | {sub['pnl_pts'].mean():+.1f} | {sub['pnl_pts'].sum():+.0f} | {wr:.0f}% |")
    doc.append("")

    # 现象3: IM只有长持仓盈利
    doc.append("### 现象3: IM长持仓vs其他\n")
    doc.append("| 窗口 | 长(>12bar)PnL | 其他PnL | 长/其他比 |")
    doc.append("|------|------------|---------|---------|")
    im_tdf = all_data['IM']
    im_is = im_tdf[im_tdf['trade_date'].isin(splits['IM']['is'])]
    im_oos = im_tdf[im_tdf['trade_date'].isin(splits['IM']['oos'])]
    for label, wdf in [('900天', im_tdf), ('OOS681', im_oos), ('IS219', im_is)]:
        long_ = wdf[wdf['hold_bars'] > 12]['pnl_pts'].sum()
        other = wdf[wdf['hold_bars'] <= 12]['pnl_pts'].sum()
        ratio = long_ / abs(other) if other != 0 else 99
        doc.append(f"| {label} | {long_:+.0f} | {other:+.0f} | {ratio:.2f} |")
    doc.append("")

    # 现象4: IM/IC止损率对比
    doc.append("### 现象4: IM vs IC短持仓止损率对比\n")
    doc.append("| 窗口 | IM止损率 | IC止损率 | 比值 |")
    doc.append("|------|--------|--------|------|")
    for label in ['900天', 'OOS681', 'IS219']:
        im_rates = {}
        ic_rates = {}
        for sym in ['IM', 'IC']:
            tdf_w = all_data[sym]
            if label == 'OOS681':
                tdf_w = tdf_w[tdf_w['trade_date'].isin(splits[sym]['oos'])]
            elif label == 'IS219':
                tdf_w = tdf_w[tdf_w['trade_date'].isin(splits[sym]['is'])]
            short = tdf_w[tdf_w['hold_bars'] <= 4]
            reason_col = 'reason' if 'reason' in short.columns else 'exit_reason'
            if reason_col in short.columns:
                sl = short[short[reason_col].str.contains('STOP|stop', na=False)]
                rate = len(sl) / len(short) * 100 if len(short) > 0 else 0
            else:
                rate = 0
            if sym == 'IM':
                im_rates[label] = rate
            else:
                ic_rates[label] = rate
        im_r = im_rates.get(label, 0)
        ic_r = ic_rates.get(label, 0)
        ratio = im_r / ic_r if ic_r > 0 else 99
        doc.append(f"| {label} | {im_r:.0f}% | {ic_r:.0f}% | {ratio:.1f}x |")
    doc.append("")

    # ═══════════════════════════════════════════════
    # B.5 综合判定
    # ═══════════════════════════════════════════════
    doc.append("## B.5 综合判定\n")
    doc.append("(根据以上数据对4个现象分别判定W1/W2/W3/W4)")

    report = "\n".join(doc)
    path = Path("tmp") / "short_hold_dual_window_validation.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
