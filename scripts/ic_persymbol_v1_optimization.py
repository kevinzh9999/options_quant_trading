#!/usr/bin/env python3
"""IC per-symbol v1优化：两个方向的数据驱动映射设计+交叉验证。

方向A: 在681天(早期OOS)上设计映射 → 在219天(近期IS)上验证
方向B: 在219天(近期IS)上设计映射 → 在681天(早期OOS)上验证
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from data.storage.db_manager import get_db
from scripts.backtest_signals_day import run_day
from strategies.intraday.A_share_momentum_signal_v2 import SYMBOL_PROFILES
from strategies.intraday.experimental.score_components_new import compute_q_score, compute_gap_bonus

SPOTS = {'IC': '000905', 'IM': '000852'}
V2_THR = {'IC': 55, 'IM': 60}
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


def data_driven_mapping(tdf, col, n_buckets, max_score):
    """基于PnL数据驱动的分桶映射。"""
    valid = tdf[tdf[col].notna() & (tdf[col] != 0)].copy()
    if len(valid) < n_buckets * 20:
        return None, None
    try:
        valid['bucket'] = pd.qcut(valid[col], n_buckets, duplicates='drop')
    except ValueError:
        valid['bucket'] = pd.cut(valid[col], n_buckets, duplicates='drop')

    stats = valid.groupby('bucket', observed=True).agg(
        n=('pnl_pts', 'count'),
        avg_pnl=('pnl_pts', 'mean'),
    ).reset_index()
    stats = stats.sort_values('avg_pnl')
    n = len(stats)
    stats['score'] = [int(round(i / (n - 1) * max_score)) if n > 1 else max_score // 2 for i in range(n)]

    # 构建lookup: [(lo, hi, score), ...]
    thresholds = []
    for _, r in stats.sort_values('bucket').iterrows():
        thresholds.append((r['bucket'].left, r['bucket'].right, int(r['score'])))
    return thresholds, stats


def data_driven_session(tdf):
    """基于PnL的时段加分。"""
    stats = []
    for h in [9, 10, 11, 13, 14]:
        sub = tdf[tdf['_hour_bj'] == h]
        if len(sub) >= 30:
            stats.append({'hour': h, 'avg_pnl': sub['pnl_pts'].mean()})
    if not stats:
        return {9: 0, 10: 0, 11: 0, 13: 0, 14: 0}
    sdf = pd.DataFrame(stats)
    mn, mx = sdf['avg_pnl'].min(), sdf['avg_pnl'].max()
    if mx - mn > 0:
        sdf['bonus'] = ((sdf['avg_pnl'] - mn) / (mx - mn) * 20 - 10).round().astype(int)
    else:
        sdf['bonus'] = 0
    return dict(zip(sdf['hour'], sdf['bonus']))


def make_lookup_fn(thresholds):
    def fn(val):
        for lo, hi, s in thresholds:
            if lo <= val < hi:
                return s
        if val >= thresholds[-1][1]:
            return thresholds[-1][2]
        return 0
    return fn


def apply_custom_scoring(tdf, m_thresholds, v_thresholds, session_map):
    tdf = tdf.copy()
    m_fn = make_lookup_fn(m_thresholds)
    v_fn = make_lookup_fn(v_thresholds)
    tdf['v1_m'] = tdf['_raw_mom'].apply(m_fn)
    tdf['v1_v'] = tdf['_raw_atr'].apply(v_fn)
    tdf['v1_q'] = tdf.apply(lambda r: compute_q_score(r['_raw_vpct'], r['_raw_vratio']), axis=1)
    tdf['v1_session'] = tdf['_hour_bj'].apply(lambda h: session_map.get(h, 0))
    tdf['v1_gap'] = tdf['_gap_aligned'].apply(compute_gap_bonus)
    tdf['v1_total'] = tdf['v1_m'] + tdf['v1_v'] + tdf['v1_q'] + tdf['v1_session'] + tdf['v1_gap']
    return tdf


def scan_and_report(tdf, dates, v2_n, v2_total, v2_avg, thresholds_list, doc):
    """扫描threshold并输出表格+sweet spot判定。"""
    is_set = set(dates[-219:])
    oos_set = set(dates[:-219])
    sweet = None

    doc.append("| thr | 笔数 | 总PnL | 单笔 | 单笔vs | 总vs | 频率比 | WR | 信号↓ | PnL✓ | 效率✓ |")
    doc.append("|-----|------|-------|------|-------|------|-------|-----|------|------|------|")

    for thr in thresholds_list:
        v1 = tdf[tdf['v1_total'] >= thr]
        n = len(v1)
        if n == 0: continue
        total = v1['pnl_pts'].sum()
        avg = v1['pnl_pts'].mean()
        wr = (v1['pnl_pts'] > 0).sum() / n * 100
        freq = n / v2_n if v2_n > 0 else 99
        avg_vs = (avg - v2_avg) / abs(v2_avg) * 100 if v2_avg != 0 else 0
        total_vs = (total - v2_total) / abs(v2_total) * 100 if v2_total != 0 else 0
        sig = (1 - freq) >= SS_SIGNAL_REDUCE
        pnl = total >= v2_total * (1 - SS_PNL_LOSS)
        eff = avg >= v2_avg * (1 + SS_EFFICIENCY)
        is_sw = sig and pnl and eff
        if is_sw and sweet is None: sweet = thr
        mark = " ★" if is_sw else ""
        doc.append(f"| {thr} | {n} | {total:+.0f} | {avg:+.2f} | {avg_vs:+.1f}% | "
                   f"{total_vs:+.1f}% | {freq:.2f} | {wr:.0f}% | "
                   f"{'✓' if sig else '✗'} | {'✓' if pnl else '✗'} | {'✓' if eff else '✗'} |{mark}")

    return sweet


def main():
    print("=" * 60)
    print("  IC Per-Symbol v1 优化（双向交叉验证）")
    print("=" * 60)

    db = get_db()
    n_workers = min(cpu_count(), 8)

    # 收集IC全量数据
    print("\n[IC] 收集900天数据...")
    dates = get_dates(db, SPOTS['IC'])
    args = [(td, 'IC') for td in dates]
    with Pool(n_workers) as pool:
        day_results = pool.map(_run_one_day, args)
    all_trades = [t for day in day_results for t in day]
    tdf = pd.DataFrame(all_trades)
    print(f"  {len(tdf)}笔")

    oos_dates = dates[:-219]
    is_dates = dates[-219:]
    oos_set = set(oos_dates)
    is_set = set(is_dates)
    tdf_oos = tdf[tdf['trade_date'].isin(oos_set)]
    tdf_is = tdf[tdf['trade_date'].isin(is_set)]

    # v2基线
    v2_all = tdf[tdf['entry_score'] >= V2_THR['IC']]
    v2_n = len(v2_all)
    v2_total = v2_all['pnl_pts'].sum()
    v2_avg = v2_all['pnl_pts'].mean()

    v2_oos = v2_all[v2_all['trade_date'].isin(oos_set)]
    v2_is = v2_all[v2_all['trade_date'].isin(is_set)]

    doc = ["# IC Per-Symbol v1 优化（双向交叉验证）\n"]
    doc.append(f"数据: IC {len(dates)}天, OOS={len(oos_dates)}天, IS={len(is_dates)}天")
    doc.append(f"v2基线(thr=55): {v2_n}笔, 总{v2_total:+.0f}, 单笔{v2_avg:+.2f}\n")

    thresholds_scan = list(range(20, 85, 5))

    # ═══════════════════════════════════════════════
    # 方向A: 在681天(OOS)上设计 → 在219天(IS)上验证
    # ═══════════════════════════════════════════════
    doc.append("# 方向A: 在681天(早期OOS)上设计 → 在219天(近期IS)上验证\n")

    # 设计映射
    m_thr_a, m_stats_a = data_driven_mapping(tdf_oos, '_raw_mom', 10, 50)
    v_thr_a, v_stats_a = data_driven_mapping(tdf_oos, '_raw_atr', 10, 30)
    session_a = data_driven_session(tdf_oos)

    doc.append("## A.1 OOS 681天数据驱动映射\n")
    doc.append("### M分\n")
    if m_stats_a is not None:
        doc.append("| 动量范围 | 笔数 | AvgPnL | 分数 |")
        doc.append("|---------|------|--------|------|")
        for _, r in m_stats_a.sort_values('bucket').iterrows():
            doc.append(f"| {r['bucket']} | {int(r['n'])} | {r['avg_pnl']:+.1f} | {int(r['score'])} |")

    doc.append("\n### V分\n")
    if v_stats_a is not None:
        doc.append("| ATR ratio范围 | 笔数 | AvgPnL | 分数 |")
        doc.append("|-------------|------|--------|------|")
        for _, r in v_stats_a.sort_values('bucket').iterrows():
            doc.append(f"| {r['bucket']} | {int(r['n'])} | {r['avg_pnl']:+.1f} | {int(r['score'])} |")

    doc.append(f"\n### Session: {session_a}\n")

    # 在全样本上扫描（看总体表现）
    if m_thr_a and v_thr_a:
        tdf_scored_a = apply_custom_scoring(tdf, m_thr_a, v_thr_a, session_a)

        doc.append("## A.2 全样本Threshold扫描\n")
        sweet_a_full = scan_and_report(tdf_scored_a, dates, v2_n, v2_total, v2_avg, thresholds_scan, doc)

        # 在219天(验证集)上检查
        doc.append(f"\n## A.3 验证: 在219天(近期IS)上的表现\n")
        v2_is_n = len(v2_is)
        v2_is_total = v2_is['pnl_pts'].sum()
        v2_is_avg = v2_is['pnl_pts'].mean() if v2_is_n > 0 else 0
        doc.append(f"v2基线(IS 219天): {v2_is_n}笔, 总{v2_is_total:+.0f}, 单笔{v2_is_avg:+.2f}\n")
        sweet_a_is = scan_and_report(
            tdf_scored_a[tdf_scored_a['trade_date'].isin(is_set)],
            is_dates, v2_is_n, v2_is_total, v2_is_avg, thresholds_scan, doc)

        # 120天验证
        recent_120 = set(dates[-120:])
        v2_120 = v2_all[v2_all['trade_date'].isin(recent_120)]
        doc.append(f"\n## A.4 验证: 最近120天\n")
        v2_120_n = len(v2_120)
        v2_120_total = v2_120['pnl_pts'].sum()
        v2_120_avg = v2_120['pnl_pts'].mean() if v2_120_n > 0 else 0
        doc.append(f"v2基线(120天): {v2_120_n}笔, 总{v2_120_total:+.0f}, 单笔{v2_120_avg:+.2f}\n")
        sweet_a_120 = scan_and_report(
            tdf_scored_a[tdf_scored_a['trade_date'].isin(recent_120)],
            list(recent_120), v2_120_n, v2_120_total, v2_120_avg, thresholds_scan, doc)

        if sweet_a_full:
            doc.append(f"\n**方向A全样本Sweet Spot: thr={sweet_a_full}**")
        doc.append("")

    # ═══════════════════════════════════════════════
    # 方向B: 在219天(IS)上设计 → 在681天(OOS)上验证
    # ═══════════════════════════════════════════════
    doc.append("# 方向B: 在219天(近期IS)上设计 → 在681天(早期OOS)上验证\n")

    m_thr_b, m_stats_b = data_driven_mapping(tdf_is, '_raw_mom', 8, 50)  # IS样本少用8桶
    v_thr_b, v_stats_b = data_driven_mapping(tdf_is, '_raw_atr', 8, 30)
    session_b = data_driven_session(tdf_is)

    doc.append("## B.1 IS 219天数据驱动映射\n")
    doc.append("### M分\n")
    if m_stats_b is not None:
        doc.append("| 动量范围 | 笔数 | AvgPnL | 分数 |")
        doc.append("|---------|------|--------|------|")
        for _, r in m_stats_b.sort_values('bucket').iterrows():
            doc.append(f"| {r['bucket']} | {int(r['n'])} | {r['avg_pnl']:+.1f} | {int(r['score'])} |")

    doc.append("\n### V分\n")
    if v_stats_b is not None:
        doc.append("| ATR ratio范围 | 笔数 | AvgPnL | 分数 |")
        doc.append("|-------------|------|--------|------|")
        for _, r in v_stats_b.sort_values('bucket').iterrows():
            doc.append(f"| {r['bucket']} | {int(r['n'])} | {r['avg_pnl']:+.1f} | {int(r['score'])} |")

    doc.append(f"\n### Session: {session_b}\n")

    if m_thr_b and v_thr_b:
        tdf_scored_b = apply_custom_scoring(tdf, m_thr_b, v_thr_b, session_b)

        doc.append("## B.2 全样本Threshold扫描\n")
        sweet_b_full = scan_and_report(tdf_scored_b, dates, v2_n, v2_total, v2_avg, thresholds_scan, doc)

        # 在681天(验证集)上检查
        doc.append(f"\n## B.3 验证: 在681天(早期OOS)上的表现\n")
        v2_oos_n = len(v2_oos)
        v2_oos_total = v2_oos['pnl_pts'].sum()
        v2_oos_avg = v2_oos['pnl_pts'].mean() if v2_oos_n > 0 else 0
        doc.append(f"v2基线(OOS 681天): {v2_oos_n}笔, 总{v2_oos_total:+.0f}, 单笔{v2_oos_avg:+.2f}\n")
        sweet_b_oos = scan_and_report(
            tdf_scored_b[tdf_scored_b['trade_date'].isin(oos_set)],
            oos_dates, v2_oos_n, v2_oos_total, v2_oos_avg, thresholds_scan, doc)

        # 120天验证
        doc.append(f"\n## B.4 验证: 最近120天\n")
        doc.append(f"v2基线(120天): {v2_120_n}笔, 总{v2_120_total:+.0f}, 单笔{v2_120_avg:+.2f}\n")
        sweet_b_120 = scan_and_report(
            tdf_scored_b[tdf_scored_b['trade_date'].isin(recent_120)],
            list(recent_120), v2_120_n, v2_120_total, v2_120_avg, thresholds_scan, doc)

        if sweet_b_full:
            doc.append(f"\n**方向B全样本Sweet Spot: thr={sweet_b_full}**")
        doc.append("")

    # ═══════════════════════════════════════════════
    # 综合判定
    # ═══════════════════════════════════════════════
    doc.append("# 综合判定\n")
    doc.append("| 方向 | 设计数据 | 验证数据 | 全样本Sweet Spot | 验证集通过? | 120天通过? |")
    doc.append("|------|---------|---------|---------------|----------|---------|")
    doc.append(f"| A | OOS681天 | IS219天 | {sweet_a_full or '无'} | {sweet_a_is or '无'} | {sweet_a_120 or '无'} |")
    if m_thr_b and v_thr_b:
        doc.append(f"| B | IS219天 | OOS681天 | {sweet_b_full or '无'} | {sweet_b_oos or '无'} | {sweet_b_120 or '无'} |")

    report = "\n".join(doc)
    path = Path("tmp") / "ic_persymbol_v1_optimization.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
