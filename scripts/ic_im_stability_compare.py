#!/usr/bin/env python3
"""IC vs IM：为什么同样方法IM成功IC失败？诊断PnL-vs-raw关系的时间稳定性。"""
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
    td, sym = args
    SYMBOL_PROFILES[sym]["signal_threshold"] = 20
    db = get_db()
    trades = run_day(sym, td, db, verbose=False)
    full = [t for t in trades if not t.get("partial")]
    for t in full:
        t['trade_date'] = td; t['symbol'] = sym
        t['_raw_mom'] = abs(t.get('raw_mom_5m', 0.0))
        t['_raw_atr'] = t.get('raw_atr_ratio', 0.0)
        t['_raw_vpct'] = t.get('raw_vol_pct', -1.0)
        try: t['_hour_bj'] = int(t.get('entry_time', '13:00')[:2])
        except: t['_hour_bj'] = 13
        t['_gap_aligned'] = t.get('entry_gap_aligned', False)
    return full


def bucket_pnl_by_window(tdf, col, n_buckets, dates, windows):
    """对一个raw值列，在不同时间窗口上分桶看PnL，检查稳定性。"""
    valid = tdf[(tdf[col].notna()) & (tdf[col] != 0) & (tdf[col] > -0.5)].copy()
    if len(valid) < n_buckets * 20:
        return None

    # 用全样本确定桶边界（保证各窗口可比）
    try:
        valid['bucket'] = pd.qcut(valid[col], n_buckets, duplicates='drop')
    except:
        valid['bucket'] = pd.cut(valid[col], n_buckets, duplicates='drop')

    buckets = sorted(valid['bucket'].unique())

    results = {}
    for wname, wset in windows:
        w_data = valid[valid['trade_date'].isin(wset)]
        bucket_pnl = []
        for b in buckets:
            sub = w_data[w_data['bucket'] == b]
            bucket_pnl.append(sub['pnl_pts'].mean() if len(sub) >= 10 else np.nan)
        results[wname] = bucket_pnl

    return buckets, results


def main():
    print("=" * 60)
    print("  IC vs IM: PnL-vs-raw 时间稳定性诊断")
    print("=" * 60)

    db = get_db()
    n_workers = min(cpu_count(), 8)

    all_data = {}
    for sym in ['IM', 'IC']:
        print(f"\n[{sym}] 收集数据...")
        dates = get_dates(db, SPOTS[sym])
        args = [(td, sym) for td in dates]
        with Pool(n_workers) as pool:
            day_results = pool.map(_run_one_day, args)
        trades = [t for day in day_results for t in day]
        tdf = pd.DataFrame(trades)
        all_data[sym] = {'tdf': tdf, 'dates': dates}
        print(f"  {len(tdf)}笔")

    doc = ["# IC vs IM: 为什么同样方法IM成功IC失败？\n"]

    for sym in ['IM', 'IC']:
        tdf = all_data[sym]['tdf']
        dates = all_data[sym]['dates']

        oos_set = set(dates[:-219])
        is_set = set(dates[-219:])
        r120 = set(dates[-120:])
        r450 = set(dates[-450:])
        e450 = set(dates[:450])

        windows = [
            ('全900', set(dates)),
            ('早450', e450),
            ('近450', r450),
            ('OOS681', oos_set),
            ('IS219', is_set),
            ('近120', r120),
        ]

        doc.append(f"## {sym}\n")

        # 对M分和V分分别做时间稳定性检查
        for col, label, n_b in [('_raw_mom', 'M(动量)', 8), ('_raw_atr', 'V(ATR ratio)', 8)]:
            result = bucket_pnl_by_window(tdf, col, n_b, dates, windows)
            if result is None:
                doc.append(f"### {label}: 数据不足\n")
                continue

            buckets, w_results = result
            doc.append(f"### {label} 分桶PnL (各时间窗口)\n")

            # 表头
            header = f"| 桶 |"
            for wn, _ in windows:
                header += f" {wn} |"
            doc.append(header)
            doc.append("|---" + "|---" * len(windows) + "|")

            for i, b in enumerate(buckets):
                row = f"| {b} |"
                for wn, _ in windows:
                    val = w_results[wn][i]
                    row += f" {val:+.1f} |" if not np.isnan(val) else " - |"
                doc.append(row)

            # 计算各窗口之间的Pearson相关（桶排序一致性）
            doc.append(f"\n**桶排序一致性（Pearson相关）:**")
            full_rank = w_results['全900']
            for wn in ['早450', '近450', 'IS219', '近120']:
                w_rank = w_results[wn]
                valid_pairs = [(a, b) for a, b in zip(full_rank, w_rank) if not np.isnan(a) and not np.isnan(b)]
                if len(valid_pairs) >= 4:
                    from scipy.stats import pearsonr
                    r, p = pearsonr([x[0] for x in valid_pairs], [x[1] for x in valid_pairs])
                    doc.append(f"  全900 vs {wn}: r={r:.3f} (p={p:.3f})")
            doc.append("")

        # 时段PnL稳定性
        doc.append(f"### 时段PnL稳定性\n")
        doc.append(f"| 时段 | 全900 | 早450 | 近450 | IS219 | 近120 |")
        doc.append(f"|------|-------|-------|-------|-------|-------|")
        for h in [9, 10, 11, 13, 14]:
            row = f"| {h:02d}:00 |"
            for wn, wset in windows[:1] + windows[1:3] + windows[4:6]:
                sub = tdf[(tdf['_hour_bj'] == h) & (tdf['trade_date'].isin(wset))]
                if len(sub) >= 20:
                    row += f" {sub['pnl_pts'].mean():+.1f} |"
                else:
                    row += " - |"
            doc.append(row)
        doc.append("")

        # v2 score分布的时间稳定性
        doc.append(f"### v2 Score分布的时间稳定性\n")
        doc.append(f"| 窗口 | 笔数 | v2 score均值 | v2 score中位数 | 单笔PnL |")
        doc.append(f"|------|------|-----------|------------|---------|")
        for wn, wset in windows:
            sub = tdf[tdf['trade_date'].isin(wset)]
            if len(sub) > 0:
                doc.append(f"| {wn} | {len(sub)} | {sub['entry_score'].mean():.1f} | "
                           f"{sub['entry_score'].median():.0f} | {sub['pnl_pts'].mean():+.2f} |")
        doc.append("")

    # 综合对比
    doc.append("## 综合对比\n")
    doc.append("核心问题：IM的PnL-vs-raw关系在不同窗口上是否比IC更稳定？\n")

    report = "\n".join(doc)
    path = Path("tmp") / "ic_im_stability_compare.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
