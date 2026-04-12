#!/usr/bin/env python3
"""把v1_im的参数集直接套用到IC，看效果。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from data.storage.db_manager import get_db
from scripts.backtest_signals_day import run_day
from strategies.intraday.A_share_momentum_signal_v2 import SYMBOL_PROFILES
from strategies.intraday.experimental.signal_new_mapping_v1_im import (
    score as score_im, THRESHOLD as THR_IM,
)
from strategies.intraday.experimental.score_components_new import compute_gap_bonus

SPOT = '000905'
SYM = 'IC'
V2_THR = 55


def get_dates(db):
    df = db.query_df(
        f"SELECT DISTINCT substr(datetime,1,10) as d FROM index_min "
        f"WHERE symbol='{SPOT}' AND period=300 ORDER BY d"
    )
    dates = [d.replace('-', '') for d in df['d'].tolist()]
    return dates[-900:] if len(dates) > 900 else dates


def _run_one_day(args):
    td, = args
    SYMBOL_PROFILES[SYM]["signal_threshold"] = 20
    db = get_db()
    trades = run_day(SYM, td, db, verbose=False)
    full = [t for t in trades if not t.get("partial")]
    for t in full:
        t['trade_date'] = td; t['symbol'] = SYM
        raw_mom = abs(t.get('raw_mom_5m', 0.0))
        raw_atr = t.get('raw_atr_ratio', 0.0)
        raw_vpct = t.get('raw_vol_pct', -1.0)
        raw_vratio = t.get('raw_vol_ratio', -1.0)
        try: hour_bj = int(t.get('entry_time', '13:00')[:2])
        except: hour_bj = 13
        gap_aligned = t.get('entry_gap_aligned', False)
        # 用IM的v1参数给IC打分
        v1r = score_im(raw_mom, raw_atr, raw_vpct, raw_vratio, hour_bj, gap_aligned)
        t['v1_total'] = v1r['total_score']
    return full


def main():
    print("=" * 60)
    print("  IC用v1_im参数集的效果")
    print("=" * 60)

    db = get_db()
    n_workers = min(cpu_count(), 8)
    dates = get_dates(db)

    print(f"\n收集IC {len(dates)}天数据...")
    args = [(td,) for td in dates]
    with Pool(n_workers) as pool:
        day_results = pool.map(_run_one_day, args)
    all_trades = [t for day in day_results for t in day]
    tdf = pd.DataFrame(all_trades)
    print(f"  {len(tdf)}笔")

    v2_all = tdf[tdf['entry_score'] >= V2_THR]
    v2_n = len(v2_all)
    v2_total = v2_all['pnl_pts'].sum()
    v2_avg = v2_all['pnl_pts'].mean()

    doc = ["# IC 用 v1_im 参数集的效果\n"]
    doc.append(f"v2基线(thr={V2_THR}): {v2_n}笔, 总{v2_total:+.0f}, 单笔{v2_avg:+.2f}\n")
    doc.append(f"v1_im参数: threshold={THR_IM}\n")

    # Threshold扫描
    doc.append("## Threshold扫描\n")
    doc.append("| thr | 笔数 | 总PnL | 单笔 | 单笔vs v2 | 总vs v2 | 频率比 | WR |")
    doc.append("|-----|------|-------|------|----------|---------|-------|-----|")
    for thr in [30, 35, 40, 45, 50, 55, 60, 65, 70]:
        v1 = tdf[tdf['v1_total'] >= thr]
        n = len(v1)
        if n == 0: continue
        total = v1['pnl_pts'].sum()
        avg = v1['pnl_pts'].mean()
        wr = (v1['pnl_pts'] > 0).sum() / n * 100
        doc.append(f"| {thr} | {n} | {total:+.0f} | {avg:+.2f} | "
                   f"{(avg-v2_avg)/abs(v2_avg)*100:+.1f}% | {(total-v2_total)/abs(v2_total)*100:+.1f}% | "
                   f"{n/v2_n:.2f} | {wr:.0f}% |")

    # 扩展窗口对比（用IM的threshold=60）
    doc.append(f"\n## 扩展窗口对比 (thr={THR_IM})\n")
    doc.append("| 窗口 | v2笔 | v2总PnL | v2单笔 | v1笔 | v1总PnL | v1单笔 | v1-v2总 | v1/v2单笔 |")
    doc.append("|------|------|--------|-------|------|--------|-------|---------|---------|")
    for n_days in [20, 60, 120, 219, 450, 681, 900]:
        if n_days > len(dates): continue
        recent = set(dates[-n_days:])
        v2_r = v2_all[v2_all['trade_date'].isin(recent)]
        v1_r = tdf[(tdf['v1_total'] >= THR_IM) & (tdf['trade_date'].isin(recent))]
        v2_pnl = v2_r['pnl_pts'].sum()
        v1_pnl = v1_r['pnl_pts'].sum()
        v2_a = v2_r['pnl_pts'].mean() if len(v2_r) > 0 else 0
        v1_a = v1_r['pnl_pts'].mean() if len(v1_r) > 0 else 0
        ratio = v1_a / v2_a if v2_a != 0 else 0
        doc.append(f"| {n_days}天 | {len(v2_r)} | {v2_pnl:+.0f} | {v2_a:+.2f} | "
                   f"{len(v1_r)} | {v1_pnl:+.0f} | {v1_a:+.2f} | {v1_pnl-v2_pnl:+.0f} | {ratio:.2f}x |")

    # 也试几个其他threshold
    doc.append(f"\n## 多threshold扩展窗口\n")
    for thr in [45, 50, 55]:
        doc.append(f"\n### thr={thr}\n")
        doc.append("| 窗口 | v1笔 | v1总PnL | v1单笔 | v1-v2总 |")
        doc.append("|------|------|--------|-------|---------|")
        for n_days in [20, 60, 120, 450, 900]:
            if n_days > len(dates): continue
            recent = set(dates[-n_days:])
            v2_r = v2_all[v2_all['trade_date'].isin(recent)]
            v1_r = tdf[(tdf['v1_total'] >= thr) & (tdf['trade_date'].isin(recent))]
            v2_pnl = v2_r['pnl_pts'].sum()
            v1_pnl = v1_r['pnl_pts'].sum()
            v1_a = v1_r['pnl_pts'].mean() if len(v1_r) > 0 else 0
            doc.append(f"| {n_days}天 | {len(v1_r)} | {v1_pnl:+.0f} | {v1_a:+.2f} | {v1_pnl-v2_pnl:+.0f} |")

    report = "\n".join(doc)
    path = Path("tmp") / "ic_with_im_v1_params.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
