#!/usr/bin/env python3
"""Phase C: IM per-symbol v1参数设计——基于IM实际PnL数据驱动的映射。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from data.storage.db_manager import get_db
from scripts.backtest_signals_day import run_day
from strategies.intraday.A_share_momentum_signal_v2 import SYMBOL_PROFILES
from strategies.intraday.experimental.score_components_new import (
    compute_q_score, compute_gap_bonus,
)

SPOTS = {'IM': '000852', 'IC': '000905'}
SS_SIGNAL_REDUCE = 0.25
SS_PNL_LOSS = 0.05
SS_EFFICIENCY = 0.30


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
        t['_raw_vratio'] = t.get('raw_vol_ratio', -1.0)
        try: t['_hour_bj'] = int(t.get('entry_time', '13:00')[:2])
        except: t['_hour_bj'] = 13
        t['_gap_aligned'] = t.get('entry_gap_aligned', False)
    return full


def data_driven_m(tdf, n_buckets=10):
    """基于IM数据设计M分映射。"""
    valid = tdf[tdf['_raw_mom'] > 0.0003].copy()
    valid['m_bucket'] = pd.qcut(valid['_raw_mom'], n_buckets, duplicates='drop')
    bucket_stats = valid.groupby('m_bucket', observed=True).agg(
        n=('pnl_pts', 'count'),
        avg_pnl=('pnl_pts', 'mean'),
        wr=('pnl_pts', lambda x: (x > 0).mean()),
    ).reset_index()
    # 按avg_pnl排序分配分数: 最差桶0分，最好桶50分
    bucket_stats = bucket_stats.sort_values('avg_pnl')
    n = len(bucket_stats)
    bucket_stats['score'] = [int(round(i / (n - 1) * 50)) if n > 1 else 25 for i in range(n)]
    return bucket_stats


def data_driven_v(tdf, n_buckets=10):
    """基于IM数据设计V分映射。"""
    valid = tdf[tdf['_raw_atr'] > 0].copy()
    valid['v_bucket'] = pd.qcut(valid['_raw_atr'], n_buckets, duplicates='drop')
    bucket_stats = valid.groupby('v_bucket', observed=True).agg(
        n=('pnl_pts', 'count'),
        avg_pnl=('pnl_pts', 'mean'),
        wr=('pnl_pts', lambda x: (x > 0).mean()),
    ).reset_index()
    bucket_stats = bucket_stats.sort_values('avg_pnl')
    n = len(bucket_stats)
    bucket_stats['score'] = [int(round(i / (n - 1) * 30)) if n > 1 else 15 for i in range(n)]
    return bucket_stats


def data_driven_session(tdf):
    """基于IM数据设计时段加分。"""
    stats = []
    for h in [9, 10, 11, 13, 14]:
        sub = tdf[tdf['_hour_bj'] == h]
        if len(sub) >= 50:
            stats.append({'hour': h, 'n': len(sub), 'avg_pnl': sub['pnl_pts'].mean()})
    sdf = pd.DataFrame(stats)
    if len(sdf) == 0:
        return {}
    # 归一化到-10~+10
    min_pnl = sdf['avg_pnl'].min()
    max_pnl = sdf['avg_pnl'].max()
    if max_pnl - min_pnl > 0:
        sdf['bonus'] = ((sdf['avg_pnl'] - min_pnl) / (max_pnl - min_pnl) * 20 - 10).round().astype(int)
    else:
        sdf['bonus'] = 0
    return dict(zip(sdf['hour'], sdf['bonus']))


def create_im_specific_scoring(m_buckets, v_buckets, session_map):
    """创建IM专属的评分函数。"""
    # M分: 从bucket stats构建lookup
    m_thresholds = []
    for _, row in m_buckets.iterrows():
        interval = row['m_bucket']
        m_thresholds.append((interval.left, interval.right, int(row['score'])))

    def m_fn(raw_mom):
        for lo, hi, score in m_thresholds:
            if lo <= raw_mom < hi:
                return score
        # 超出范围: 最高或最低
        if raw_mom >= m_thresholds[-1][1]:
            return m_thresholds[-1][2]
        return 0

    v_thresholds = []
    for _, row in v_buckets.iterrows():
        interval = row['v_bucket']
        v_thresholds.append((interval.left, interval.right, int(row['score'])))

    def v_fn(raw_atr):
        for lo, hi, score in v_thresholds:
            if lo <= raw_atr < hi:
                return score
        if raw_atr >= v_thresholds[-1][1]:
            return v_thresholds[-1][2]
        return 0

    def session_fn(h):
        return session_map.get(h, 0)

    return m_fn, v_fn, session_fn


def apply_scoring(tdf, m_fn, v_fn, session_fn):
    tdf = tdf.copy()
    tdf['v1_m'] = tdf['_raw_mom'].apply(m_fn)
    tdf['v1_v'] = tdf['_raw_atr'].apply(v_fn)
    tdf['v1_q'] = tdf.apply(lambda r: compute_q_score(r['_raw_vpct'], r['_raw_vratio']), axis=1)
    tdf['v1_session'] = tdf['_hour_bj'].apply(session_fn)
    tdf['v1_gap'] = tdf['_gap_aligned'].apply(compute_gap_bonus)
    tdf['v1_total'] = tdf['v1_m'] + tdf['v1_v'] + tdf['v1_q'] + tdf['v1_session'] + tdf['v1_gap']
    return tdf


def scan_thresholds(tdf, thresholds, v2_n, v2_total, v2_avg, dates):
    is_set = set(dates[-219:])
    oos_set = set(dates[:-219])
    rows = []
    sweet = None
    for thr in thresholds:
        v1 = tdf[tdf['v1_total'] >= thr]
        n = len(v1)
        if n == 0: continue
        total = v1['pnl_pts'].sum()
        avg = v1['pnl_pts'].mean()
        wr = (v1['pnl_pts'] > 0).sum() / n * 100
        freq = n / v2_n if v2_n > 0 else 99
        avg_vs = (avg - v2_avg) / abs(v2_avg) * 100 if v2_avg != 0 else 0
        total_vs = (total - v2_total) / abs(v2_total) * 100 if v2_total != 0 else 0
        is_pnl = v1[v1['trade_date'].isin(is_set)]['pnl_pts'].sum()
        oos_pnl = v1[v1['trade_date'].isin(oos_set)]['pnl_pts'].sum()
        ratio = (is_pnl / 219) / (oos_pnl / 681) if oos_pnl > 0 else 99

        sig_red = (1 - freq) >= SS_SIGNAL_REDUCE
        pnl_ok = total >= v2_total * (1 - SS_PNL_LOSS)
        eff_ok = avg >= v2_avg * (1 + SS_EFFICIENCY)
        is_sweet = sig_red and pnl_ok and eff_ok

        rows.append({'thr': thr, 'n': n, 'total': total, 'avg': avg,
                     'avg_vs': avg_vs, 'total_vs': total_vs, 'freq': freq,
                     'wr': wr, 'ratio': ratio, 'sweet': is_sweet,
                     'signal_reduced': sig_red, 'pnl_ok': pnl_ok, 'eff_ok': eff_ok})
        if is_sweet and sweet is None:
            sweet = thr
    return pd.DataFrame(rows), sweet


def main():
    print("=" * 60)
    print("  Phase C: IM per-symbol v1参数设计")
    print("=" * 60)

    db = get_db()
    n_workers = min(cpu_count(), 8)
    doc = ["# Phase C: IM Per-Symbol v1参数设计\n"]

    # 收集数据
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

    # v2基线
    V2_THR = {'IM': 60, 'IC': 55}
    v2_stats = {}
    for sym in ['IM', 'IC']:
        v2 = all_data[sym]['tdf'][all_data[sym]['tdf']['entry_score'] >= V2_THR[sym]]
        v2_stats[sym] = {'n': len(v2), 'total': v2['pnl_pts'].sum(), 'avg': v2['pnl_pts'].mean()}

    # ═══════════════════════════════════════════════
    # C.1: 数据驱动设计
    # ═══════════════════════════════════════════════
    im_tdf = all_data['IM']['tdf']

    doc.append("## C.1 IM数据驱动的映射设计\n")

    # M分
    m_buckets = data_driven_m(im_tdf)
    doc.append("### IM M分数据驱动映射\n")
    doc.append("| 动量范围 | 笔数 | AvgPnL | WR | 分配分数 |")
    doc.append("|---------|------|--------|-----|---------|")
    for _, r in m_buckets.sort_values('m_bucket').iterrows():
        doc.append(f"| {r['m_bucket']} | {int(r['n'])} | {r['avg_pnl']:+.1f} | {r['wr']*100:.0f}% | {int(r['score'])} |")

    # V分
    v_buckets = data_driven_v(im_tdf)
    doc.append("\n### IM V分数据驱动映射\n")
    doc.append("| ATR ratio范围 | 笔数 | AvgPnL | WR | 分配分数 |")
    doc.append("|-------------|------|--------|-----|---------|")
    for _, r in v_buckets.sort_values('v_bucket').iterrows():
        doc.append(f"| {r['v_bucket']} | {int(r['n'])} | {r['avg_pnl']:+.1f} | {r['wr']*100:.0f}% | {int(r['score'])} |")

    # Session
    session_map = data_driven_session(im_tdf)
    doc.append(f"\n### IM时段加分(数据驱动)\n")
    for h in [9, 10, 11, 13, 14]:
        doc.append(f"  {h:02d}:00 → {session_map.get(h, 0):+d}")
    doc.append("")

    # ═══════════════════════════════════════════════
    # C.2: 用IM专属映射扫描
    # ═══════════════════════════════════════════════
    doc.append("## C.2 IM专属映射的Threshold扫描\n")

    m_fn, v_fn, session_fn = create_im_specific_scoring(m_buckets, v_buckets, session_map)
    im_scored = apply_scoring(im_tdf, m_fn, v_fn, session_fn)

    doc.append(f"v1_im score分布: mean={im_scored['v1_total'].mean():.1f}, "
               f"median={im_scored['v1_total'].median():.0f}, "
               f"[{im_scored['v1_total'].min()}, {im_scored['v1_total'].max()}]\n")

    thresholds = list(range(20, 85, 5))
    rdf, sweet = scan_thresholds(
        im_scored, thresholds,
        v2_stats['IM']['n'], v2_stats['IM']['total'], v2_stats['IM']['avg'],
        all_data['IM']['dates'])

    doc.append(f"v2基线: {v2_stats['IM']['n']}笔, 总{v2_stats['IM']['total']:+.0f}, 单笔{v2_stats['IM']['avg']:+.2f}\n")
    doc.append("| thr | 笔数 | 总PnL | 单笔 | 单笔vs | 总vs | 频率比 | WR | IS/OOS | 信号↓ | PnL✓ | 效率✓ |")
    doc.append("|-----|------|-------|------|-------|------|-------|-----|--------|------|------|------|")
    for _, r in rdf.iterrows():
        mark = " ★" if r['sweet'] else ""
        doc.append(f"| {r['thr']} | {int(r['n'])} | {r['total']:+.0f} | {r['avg']:+.2f} | "
                   f"{r['avg_vs']:+.1f}% | {r['total_vs']:+.1f}% | {r['freq']:.2f} | "
                   f"{r['wr']:.1f}% | {r['ratio']:.2f} | "
                   f"{'✓' if r['signal_reduced'] else '✗'} | "
                   f"{'✓' if r['pnl_ok'] else '✗'} | "
                   f"{'✓' if r['eff_ok'] else '✗'} |{mark}")

    if sweet:
        doc.append(f"\n**Phase C Sweet Spot找到: thr={sweet}** ✓")
        # IC影响
        ic_scored = apply_scoring(all_data['IC']['tdf'], m_fn, v_fn, session_fn)
        ic_55 = ic_scored[ic_scored['v1_total'] >= 55]
        doc.append(f"\nIC影响(IM专属映射,thr=55): {len(ic_55)}笔, 总{ic_55['pnl_pts'].sum():+.0f}, "
                   f"单笔{ic_55['pnl_pts'].mean():+.2f}")
    else:
        doc.append(f"\n**Phase C 无Sweet Spot**")
        rdf['cond'] = rdf['signal_reduced'].astype(int) + rdf['pnl_ok'].astype(int) + rdf['eff_ok'].astype(int)
        best = rdf.loc[rdf['cond'].idxmax()]
        doc.append(f"最接近: thr={best['thr']} ({int(best['cond'])}/3)")
        doc.append(f"  单笔{best['avg']:+.2f}({best['avg_vs']:+.1f}%), 总{best['total']:+.0f}({best['total_vs']:+.1f}%), 频率{best['freq']:.2f}")

    doc.append("")

    # ═══════════════════════════════════════════════
    # 综合判定
    # ═══════════════════════════════════════════════
    doc.append("## 综合判定\n")
    if sweet:
        doc.append(f"**Phase C成功: IM per-symbol映射在thr={sweet}找到Sweet Spot**")
    else:
        doc.append("**Phase C失败: IM per-symbol映射也无法找到Sweet Spot**")
        doc.append("\nIM的物理结构（PnL均匀分散）决定了评分系统过滤方向的天花板。")
        doc.append("IM的优化方向应该不是'过滤低质量信号'，而是其他维度（出场优化、入场时机等）。")

    report = "\n".join(doc)
    path = Path("tmp") / "im_sweet_spot_phase_c.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
