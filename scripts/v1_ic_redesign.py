#!/usr/bin/env python3
"""v1_ic per-symbol数据驱动重设计：完全对称v1_im的方法。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from data.storage.db_manager import get_db
from scripts.backtest_signals_day import run_day
from strategies.intraday.A_share_momentum_signal_v2 import SYMBOL_PROFILES
from strategies.intraday.experimental.score_components_new import compute_gap_bonus

SPOT = '000905'
SYM = 'IC'
V2_THR = 55
SS_SIGNAL_REDUCE = 0.25
SS_PNL_LOSS = 0.05
SS_EFFICIENCY = 0.30


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
        t['_raw_mom'] = abs(t.get('raw_mom_5m', 0.0))
        t['_raw_atr'] = t.get('raw_atr_ratio', 0.0)
        t['_raw_vpct'] = t.get('raw_vol_pct', -1.0)
        t['_raw_vratio'] = t.get('raw_vol_ratio', -1.0)
        try: t['_hour_bj'] = int(t.get('entry_time', '13:00')[:2])
        except: t['_hour_bj'] = 13
        t['_gap_aligned'] = t.get('entry_gap_aligned', False)
    return full


def data_driven_mapping(tdf, col, n_buckets, max_score):
    valid = tdf[tdf[col].notna() & (tdf[col] != 0) & (tdf[col] > -0.5)].copy()
    if len(valid) < n_buckets * 15:
        return None, None
    try:
        valid['bucket'] = pd.qcut(valid[col], n_buckets, duplicates='drop')
    except ValueError:
        valid['bucket'] = pd.cut(valid[col], n_buckets, duplicates='drop')
    stats = valid.groupby('bucket', observed=True).agg(
        n=('pnl_pts', 'count'), avg_pnl=('pnl_pts', 'mean'),
        wr=('pnl_pts', lambda x: (x > 0).mean()),
    ).reset_index()
    stats = stats.sort_values('avg_pnl')
    n = len(stats)
    stats['score'] = [int(round(i / (n - 1) * max_score)) if n > 1 else max_score // 2 for i in range(n)]
    thresholds = []
    for _, r in stats.sort_values('bucket').iterrows():
        thresholds.append((r['bucket'].left, r['bucket'].right, int(r['score'])))
    return thresholds, stats


def data_driven_session(tdf):
    stats = []
    for h in [9, 10, 11, 13, 14]:
        sub = tdf[tdf['_hour_bj'] == h]
        if len(sub) >= 30:
            stats.append({'hour': h, 'n': len(sub), 'avg_pnl': sub['pnl_pts'].mean()})
    if not stats:
        return {h: 0 for h in [9, 10, 11, 13, 14]}
    sdf = pd.DataFrame(stats)
    mn, mx = sdf['avg_pnl'].min(), sdf['avg_pnl'].max()
    if mx - mn > 0:
        sdf['bonus'] = ((sdf['avg_pnl'] - mn) / (mx - mn) * 20 - 10).round().astype(int)
    else:
        sdf['bonus'] = 0
    return dict(zip(sdf['hour'], sdf['bonus']))


def data_driven_q(tdf, n_buckets=5, max_score=15):
    """Q分数据驱动（跟M/V相同方法）。"""
    # 优先用percentile
    col = '_raw_vpct'
    valid = tdf[(tdf[col] >= 0) & (tdf[col] <= 1)].copy()
    if len(valid) < n_buckets * 15:
        col = '_raw_vratio'
        valid = tdf[(tdf[col] > 0)].copy()
    if len(valid) < n_buckets * 15:
        return None, None, col
    return (*data_driven_mapping(valid, col, n_buckets, max_score), col)


def make_lookup_fn(thresholds):
    def fn(val):
        for lo, hi, s in thresholds:
            if lo <= val < hi:
                return s
        if val >= thresholds[-1][1]:
            return thresholds[-1][2]
        return 0
    return fn


def apply_scoring(tdf, m_thr, v_thr, q_thr, q_col, session_map):
    tdf = tdf.copy()
    m_fn = make_lookup_fn(m_thr)
    v_fn = make_lookup_fn(v_thr)
    q_fn = make_lookup_fn(q_thr) if q_thr else lambda x: 8
    tdf['v1_m'] = tdf['_raw_mom'].apply(m_fn)
    tdf['v1_v'] = tdf['_raw_atr'].apply(v_fn)
    tdf['v1_q'] = tdf[q_col].apply(q_fn) if q_thr else 8
    tdf['v1_session'] = tdf['_hour_bj'].apply(lambda h: session_map.get(h, 0))
    tdf['v1_gap'] = tdf['_gap_aligned'].apply(compute_gap_bonus)
    tdf['v1_total'] = tdf['v1_m'] + tdf['v1_v'] + tdf['v1_q'] + tdf['v1_session'] + tdf['v1_gap']
    return tdf


def main():
    print("=" * 60)
    print("  v1_ic Per-Symbol 数据驱动重设计")
    print("=" * 60)

    db = get_db()
    n_workers = min(cpu_count(), 8)
    dates = get_dates(db)
    print(f"IC {len(dates)}天")

    # Step 1: 收集全量trade
    print("\n[Step 1] 收集IC 900天trade...")
    args = [(td,) for td in dates]
    with Pool(n_workers) as pool:
        day_results = pool.map(_run_one_day, args)
    all_trades = [t for day in day_results for t in day]
    tdf = pd.DataFrame(all_trades)
    print(f"  {len(tdf)}笔")

    # v2基线
    v2_all = tdf[tdf['entry_score'] >= V2_THR]
    v2_n = len(v2_all)
    v2_total = v2_all['pnl_pts'].sum()
    v2_avg = v2_all['pnl_pts'].mean()

    doc = ["# v1_ic Per-Symbol 数据驱动重设计\n"]
    doc.append(f"方法: 完全对称v1_im——IC全样本900天数据驱动映射")
    doc.append(f"v2基线(thr={V2_THR}): {v2_n}笔, 总{v2_total:+.0f}, 单笔{v2_avg:+.2f}\n")

    # Step 2: 数据驱动设计
    print("\n[Step 2] 数据驱动映射设计...")
    doc.append("## Step 2: IC 900天数据驱动映射\n")

    # M分
    m_thr, m_stats = data_driven_mapping(tdf, '_raw_mom', 10, 50)
    doc.append("### M分 (10桶按PnL排序)\n")
    doc.append("| 动量范围 | 笔数 | AvgPnL | WR | 分数 |")
    doc.append("|---------|------|--------|-----|------|")
    for _, r in m_stats.sort_values('bucket').iterrows():
        doc.append(f"| {r['bucket']} | {int(r['n'])} | {r['avg_pnl']:+.1f} | {r['wr']*100:.0f}% | {int(r['score'])} |")

    # V分
    v_thr, v_stats = data_driven_mapping(tdf, '_raw_atr', 10, 30)
    doc.append("\n### V分 (10桶按PnL排序)\n")
    doc.append("| ATR ratio范围 | 笔数 | AvgPnL | WR | 分数 |")
    doc.append("|-------------|------|--------|-----|------|")
    for _, r in v_stats.sort_values('bucket').iterrows():
        doc.append(f"| {r['bucket']} | {int(r['n'])} | {r['avg_pnl']:+.1f} | {r['wr']*100:.0f}% | {int(r['score'])} |")

    # Q分
    q_result = data_driven_q(tdf)
    q_thr, q_stats, q_col = q_result if q_result[0] else (None, None, '_raw_vpct')
    doc.append(f"\n### Q分 (col={q_col})\n")
    if q_stats is not None:
        doc.append("| 范围 | 笔数 | AvgPnL | 分数 |")
        doc.append("|------|------|--------|------|")
        for _, r in q_stats.sort_values('bucket').iterrows():
            doc.append(f"| {r['bucket']} | {int(r['n'])} | {r['avg_pnl']:+.1f} | {int(r['score'])} |")
    else:
        doc.append("Q分数据不足，使用默认值8\n")

    # Session
    session_map = data_driven_session(tdf)
    doc.append(f"\n### Session加分: {session_map}\n")

    # Gap
    gap_true = tdf[tdf['_gap_aligned'] == True]['pnl_pts'].mean()
    gap_false = tdf[tdf['_gap_aligned'] == False]['pnl_pts'].mean()
    doc.append(f"### Gap: aligned={gap_true:+.1f}pt, not={gap_false:+.1f}pt → 保持+5/0\n")

    # Step 3: Threshold扫描
    print("\n[Step 3] Threshold扫描...")
    tdf_scored = apply_scoring(tdf, m_thr, v_thr, q_thr, q_col, session_map)

    doc.append(f"## Step 3: Threshold扫描\n")
    doc.append(f"v1_ic_new score分布: mean={tdf_scored['v1_total'].mean():.1f}, "
               f"median={tdf_scored['v1_total'].median():.0f}, "
               f"[{tdf_scored['v1_total'].min()}, {tdf_scored['v1_total'].max()}]\n")

    thresholds = list(range(20, 85, 5))
    doc.append("| thr | 笔数 | 总PnL | 单笔 | 单笔vs | 总vs | 频率比 | WR | 信号↓ | PnL✓ | 效率✓ |")
    doc.append("|-----|------|-------|------|-------|------|-------|-----|------|------|------|")

    sweet = None
    for thr in thresholds:
        v1 = tdf_scored[tdf_scored['v1_total'] >= thr]
        n = len(v1)
        if n == 0: continue
        total = v1['pnl_pts'].sum()
        avg = v1['pnl_pts'].mean()
        wr = (v1['pnl_pts'] > 0).sum() / n * 100
        freq = n / v2_n
        avg_vs = (avg - v2_avg) / abs(v2_avg) * 100
        total_vs = (total - v2_total) / abs(v2_total) * 100
        sig = (1 - freq) >= SS_SIGNAL_REDUCE
        pnl = total >= v2_total * (1 - SS_PNL_LOSS)
        eff = avg >= v2_avg * (1 + SS_EFFICIENCY)
        is_sw = sig and pnl and eff
        if is_sw and sweet is None: sweet = thr
        mark = " ★" if is_sw else ""
        doc.append(f"| {thr} | {n} | {total:+.0f} | {avg:+.2f} | {avg_vs:+.1f}% | "
                   f"{total_vs:+.1f}% | {freq:.2f} | {wr:.0f}% | "
                   f"{'✓' if sig else '✗'} | {'✓' if pnl else '✗'} | {'✓' if eff else '✗'} |{mark}")

    if sweet:
        doc.append(f"\n**Sweet Spot: thr={sweet}**\n")
    else:
        doc.append(f"\n**无Sweet Spot**\n")

    # Step 5: 扩展对比验证
    print("\n[Step 5] 扩展对比验证...")
    best_thr = sweet or 55  # 如果没sweet spot用55

    doc.append(f"## Step 5: 扩展对比验证 (thr={best_thr})\n")
    doc.append("| 窗口 | v2笔 | v2总PnL | v2单笔 | v1笔 | v1总PnL | v1单笔 | v1-v2总 | v1/v2单笔 |")
    doc.append("|------|------|--------|-------|------|--------|-------|---------|---------|")

    for n_days in [20, 60, 120, 219, 450, 681, 900]:
        if n_days > len(dates): continue
        recent = set(dates[-n_days:])
        v2_r = v2_all[v2_all['trade_date'].isin(recent)]
        v1_r = tdf_scored[(tdf_scored['v1_total'] >= best_thr) & (tdf_scored['trade_date'].isin(recent))]
        v2_pnl = v2_r['pnl_pts'].sum()
        v1_pnl = v1_r['pnl_pts'].sum()
        v2_avg_r = v2_r['pnl_pts'].mean() if len(v2_r) > 0 else 0
        v1_avg_r = v1_r['pnl_pts'].mean() if len(v1_r) > 0 else 0
        eff_ratio = v1_avg_r / v2_avg_r if v2_avg_r != 0 else 0
        doc.append(f"| {n_days}天 | {len(v2_r)} | {v2_pnl:+.0f} | {v2_avg_r:+.2f} | "
                   f"{len(v1_r)} | {v1_pnl:+.0f} | {v1_avg_r:+.2f} | {v1_pnl-v2_pnl:+.0f} | {eff_ratio:.2f}x |")

    # 如果没找到sweet spot，也试几个备选threshold
    if not sweet:
        doc.append(f"\n### 备选threshold对比\n")
        for alt_thr in [40, 45, 50, 55, 60]:
            doc.append(f"\n**thr={alt_thr}:**\n")
            doc.append("| 窗口 | v1笔 | v1总PnL | v1单笔 | v1-v2总 |")
            doc.append("|------|------|--------|-------|---------|")
            for n_days in [20, 60, 120, 450, 900]:
                if n_days > len(dates): continue
                recent = set(dates[-n_days:])
                v2_r = v2_all[v2_all['trade_date'].isin(recent)]
                v1_r = tdf_scored[(tdf_scored['v1_total'] >= alt_thr) & (tdf_scored['trade_date'].isin(recent))]
                v2_pnl = v2_r['pnl_pts'].sum()
                v1_pnl = v1_r['pnl_pts'].sum()
                v1_avg_r = v1_r['pnl_pts'].mean() if len(v1_r) > 0 else 0
                doc.append(f"| {n_days}天 | {len(v1_r)} | {v1_pnl:+.0f} | {v1_avg_r:+.2f} | {v1_pnl-v2_pnl:+.0f} |")

    doc.append("\n## 综合判定\n")
    doc.append("(根据Step 5扩展对比判定v1_ic_new是否在所有窗口都有效)")

    report = "\n".join(doc)
    path = Path("tmp") / "v1_ic_redesign.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
