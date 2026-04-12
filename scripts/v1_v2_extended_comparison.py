#!/usr/bin/env python3
"""v2 vs v1 最近60/90/120天对比。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from data.storage.db_manager import get_db
from scripts.backtest_signals_day import run_day
from strategies.intraday.A_share_momentum_signal_v2 import SYMBOL_PROFILES
from strategies.intraday.experimental.signal_new_mapping_v1_im import score as score_im, THRESHOLD as THR_IM
from strategies.intraday.experimental.signal_new_mapping_v1_ic import score as score_ic, THRESHOLD as THR_IC

SPOTS = {'IC': '000905', 'IM': '000852'}
V2_THR = {'IM': 60, 'IC': 55}


def get_recent_dates(db, n):
    df = db.query_df(
        "SELECT DISTINCT substr(datetime,1,10) as d FROM index_min "
        "WHERE symbol='000852' AND period=300 ORDER BY d DESC LIMIT ?",
        params=(n,))
    return sorted([d.replace('-', '') for d in df['d'].tolist()])


def _run_one_day(args):
    td, sym = args
    SYMBOL_PROFILES[sym]["signal_threshold"] = 20
    db = get_db()
    trades = run_day(sym, td, db, verbose=False)
    full = [t for t in trades if not t.get("partial")]
    for t in full:
        t['trade_date'] = td
        t['symbol'] = sym
        raw_mom = abs(t.get('raw_mom_5m', 0.0))
        raw_atr = t.get('raw_atr_ratio', 0.0)
        raw_vpct = t.get('raw_vol_pct', -1.0)
        raw_vratio = t.get('raw_vol_ratio', -1.0)
        try:
            hour_bj = int(t.get('entry_time', '13:00')[:2])
        except:
            hour_bj = 13
        gap_aligned = t.get('entry_gap_aligned', False)
        if sym == 'IM':
            v1r = score_im(raw_mom, raw_atr, raw_vpct, raw_vratio, hour_bj, gap_aligned)
            t['v1_thr'] = THR_IM
        else:
            v1r = score_ic(raw_mom, raw_atr, raw_vpct, raw_vratio, hour_bj, gap_aligned)
            t['v1_thr'] = THR_IC
        t['v1_total'] = v1r['total_score']
    return full


def main():
    print("=" * 60)
    print("  v2 vs v1 扩展时间窗口对比")
    print("=" * 60)

    db = get_db()
    dates_120 = get_recent_dates(db, 120)
    n_workers = min(cpu_count(), 8)

    doc = ["# v2 vs v1 扩展对比 (60/90/120天)\n"]

    for sym in ['IM', 'IC']:
        print(f"\n[{sym}] 收集数据 ({len(dates_120)}天)...")
        args = [(td, sym) for td in dates_120]
        with Pool(n_workers) as pool:
            day_results = pool.map(_run_one_day, args)
        all_trades = [t for day in day_results for t in day]
        tdf = pd.DataFrame(all_trades)

        v2_thr = V2_THR[sym]
        v1_thr = THR_IM if sym == 'IM' else THR_IC

        doc.append(f"## {sym}\n")

        # 多窗口对比
        for n_days in [20, 40, 60, 90, 120]:
            recent = dates_120[-n_days:]
            sub = tdf[tdf['trade_date'].isin(set(recent))]
            v2 = sub[sub['entry_score'] >= v2_thr]
            v1 = sub[sub['v1_total'] >= v1_thr]

            v2_pnl = v2['pnl_pts'].sum()
            v1_pnl = v1['pnl_pts'].sum()
            v2_avg = v2['pnl_pts'].mean() if len(v2) > 0 else 0
            v1_avg = v1['pnl_pts'].mean() if len(v1) > 0 else 0
            v2_wr = (v2['pnl_pts'] > 0).sum() / len(v2) * 100 if len(v2) > 0 else 0
            v1_wr = (v1['pnl_pts'] > 0).sum() / len(v1) * 100 if len(v1) > 0 else 0

            if n_days == 20:
                doc.append(f"### 多窗口总览\n")
                doc.append(f"| 窗口 | v2笔 | v2总PnL | v2单笔 | v2WR | v1笔 | v1总PnL | v1单笔 | v1WR | v1-v2总 | v1/v2单笔 |")
                doc.append(f"|------|------|--------|-------|------|------|--------|-------|------|---------|---------|")

            diff_total = v1_pnl - v2_pnl
            eff_ratio = v1_avg / v2_avg if v2_avg != 0 else 0
            doc.append(f"| {n_days}天 | {len(v2)} | {v2_pnl:+.0f} | {v2_avg:+.2f} | {v2_wr:.0f}% | "
                       f"{len(v1)} | {v1_pnl:+.0f} | {v1_avg:+.2f} | {v1_wr:.0f}% | "
                       f"{diff_total:+.0f} | {eff_ratio:.2f}x |")

        doc.append("")

        # 月度拆解（120天约6个月）
        doc.append(f"### 月度v1 vs v2\n")
        doc.append(f"| 月份 | v2笔 | v2 PnL | v1笔 | v1 PnL | v1-v2 |")
        doc.append(f"|------|------|--------|------|--------|-------|")

        tdf_full = tdf.copy()
        tdf_full['month'] = tdf_full['trade_date'].apply(lambda x: x[:6])
        for month in sorted(tdf_full['month'].unique()):
            m_sub = tdf_full[tdf_full['month'] == month]
            v2_m = m_sub[m_sub['entry_score'] >= v2_thr]
            v1_m = m_sub[m_sub['v1_total'] >= v1_thr]
            v2_p = v2_m['pnl_pts'].sum()
            v1_p = v1_m['pnl_pts'].sum()
            doc.append(f"| {month} | {len(v2_m)} | {v2_p:+.0f} | {len(v1_m)} | {v1_p:+.0f} | {v1_p-v2_p:+.0f} |")
        doc.append("")

        # 被v1过滤的大赚trade统计
        v2_all = tdf[tdf['entry_score'] >= v2_thr]
        missed = v2_all[(v2_all['v1_total'] < v1_thr) & (v2_all['pnl_pts'] > 20)]
        doc.append(f"### 被v1过滤的大赚trade(pnl>20pt)\n")
        doc.append(f"总计: {len(missed)}笔, 累计{missed['pnl_pts'].sum():+.0f}pt\n")
        if len(missed) > 0:
            # 按入场时段分组
            doc.append(f"| 入场小时(BJ) | 笔数 | 累计PnL |")
            doc.append(f"|------------|------|---------|")
            for _, t in missed.iterrows():
                try:
                    t['_h'] = int(t['entry_time'][:2])
                except:
                    t['_h'] = 0
            missed_copy = missed.copy()
            missed_copy['_h'] = missed_copy['entry_time'].apply(lambda x: int(x[:2]) if isinstance(x, str) and len(x) >= 2 else 0)
            for h in sorted(missed_copy['_h'].unique()):
                h_sub = missed_copy[missed_copy['_h'] == h]
                doc.append(f"| {h:02d}:00 | {len(h_sub)} | {h_sub['pnl_pts'].sum():+.0f} |")
        doc.append("")

    doc.append("## 综合判定\n")
    doc.append("(根据60-120天数据判断v1的退化是20天噪音还是结构性问题)")

    report = "\n".join(doc)
    path = Path("tmp") / "v1_v2_extended_comparison.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
