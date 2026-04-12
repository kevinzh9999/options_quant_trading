#!/usr/bin/env python3
"""IM sweet spot搜索：Phase A精细扫描 + Phase B组件调整 + Phase D物理差异。"""
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
    compute_m_score, compute_v_score, compute_q_score,
    compute_session_bonus, compute_gap_bonus,
)

SPOTS = {'IC': '000905', 'IM': '000852'}

# Sweet spot criteria
SS_SIGNAL_REDUCE = 0.25   # 信号减少>=25%
SS_PNL_LOSS = 0.05        # 总PnL损失<5%
SS_EFFICIENCY = 0.30       # 单笔效率提升>=30%


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
        t['trade_date'] = td
        t['symbol'] = sym
        t['_raw_mom'] = abs(t.get('raw_mom_5m', 0.0))
        t['_raw_atr'] = t.get('raw_atr_ratio', 0.0)
        t['_raw_vpct'] = t.get('raw_vol_pct', -1.0)
        t['_raw_vratio'] = t.get('raw_vol_ratio', -1.0)
        try:
            t['_hour_bj'] = int(t.get('entry_time', '13:00')[:2])
        except (ValueError, IndexError):
            t['_hour_bj'] = 13
        t['_gap_aligned'] = t.get('entry_gap_aligned', False)
    return full


def apply_scoring(tdf, m_fn=None, v_fn=None, q_fn=None, session_fn=None, gap_fn=None):
    """对DataFrame应用自定义评分函数，返回v1_total列。"""
    if m_fn is None: m_fn = compute_m_score
    if v_fn is None: v_fn = compute_v_score
    if q_fn is None: q_fn = lambda vpct, vratio: compute_q_score(vpct, vratio)
    if session_fn is None: session_fn = compute_session_bonus
    if gap_fn is None: gap_fn = compute_gap_bonus

    tdf = tdf.copy()
    tdf['v1_m'] = tdf['_raw_mom'].apply(m_fn)
    tdf['v1_v'] = tdf['_raw_atr'].apply(v_fn)
    tdf['v1_q'] = tdf.apply(lambda r: q_fn(r['_raw_vpct'], r['_raw_vratio']), axis=1)
    tdf['v1_session'] = tdf['_hour_bj'].apply(session_fn)
    tdf['v1_gap'] = tdf['_gap_aligned'].apply(gap_fn)
    tdf['v1_total'] = tdf['v1_m'] + tdf['v1_v'] + tdf['v1_q'] + tdf['v1_session'] + tdf['v1_gap']
    return tdf


def scan_thresholds(tdf, thresholds, v2_n, v2_total, v2_avg, dates, label=""):
    """扫描多个threshold，返回结果和sweet_spot。"""
    is_set = set(dates[-219:])
    oos_set = set(dates[:-219])
    rows = []
    sweet = None

    for thr in thresholds:
        v1 = tdf[tdf['v1_total'] >= thr]
        n = len(v1)
        if n == 0:
            continue
        total = v1['pnl_pts'].sum()
        avg = v1['pnl_pts'].mean()
        wr = (v1['pnl_pts'] > 0).sum() / n * 100
        freq_ratio = n / v2_n if v2_n > 0 else 99
        avg_vs = (avg - v2_avg) / abs(v2_avg) * 100 if v2_avg != 0 else 0
        total_vs = (total - v2_total) / abs(v2_total) * 100 if v2_total != 0 else 0

        is_pnl = v1[v1['trade_date'].isin(is_set)]['pnl_pts'].sum()
        oos_pnl = v1[v1['trade_date'].isin(oos_set)]['pnl_pts'].sum()
        ratio = (is_pnl / 219) / (oos_pnl / 681) if oos_pnl > 0 else 99

        # Sweet spot check
        signal_reduced = (1 - freq_ratio) >= SS_SIGNAL_REDUCE
        pnl_ok = total >= v2_total * (1 - SS_PNL_LOSS)
        eff_ok = avg >= v2_avg * (1 + SS_EFFICIENCY)
        is_sweet = signal_reduced and pnl_ok and eff_ok

        rows.append({
            'thr': thr, 'n': n, 'total': total, 'avg': avg,
            'avg_vs': avg_vs, 'total_vs': total_vs, 'freq': freq_ratio,
            'wr': wr, 'ratio': ratio, 'sweet': is_sweet,
            'signal_reduced': signal_reduced, 'pnl_ok': pnl_ok, 'eff_ok': eff_ok,
        })
        if is_sweet and sweet is None:
            sweet = thr

    return pd.DataFrame(rows), sweet


def format_scan_table(rdf, doc):
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


def main():
    print("=" * 60)
    print("  IM Sweet Spot 搜索")
    print("=" * 60)

    db = get_db()
    n_workers = min(cpu_count(), 8)
    doc = ["# IM Sweet Spot 搜索\n"]
    doc.append(f"Sweet spot三条件: 信号减少>={SS_SIGNAL_REDUCE*100:.0f}%, "
               f"总PnL损失<{SS_PNL_LOSS*100:.0f}%, 单笔效率提升>={SS_EFFICIENCY*100:.0f}%\n")

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
        tdf = all_data[sym]['tdf']
        v2 = tdf[tdf['entry_score'] >= V2_THR[sym]]
        v2_stats[sym] = {'n': len(v2), 'total': v2['pnl_pts'].sum(), 'avg': v2['pnl_pts'].mean()}

    # ═══════════════════════════════════════════════
    # Phase A: 精细threshold扫描
    # ═══════════════════════════════════════════════
    doc.append("# Phase A: IM精细Threshold扫描\n")
    thresholds_fine = [25, 28, 30, 33, 35, 38, 40, 43, 45, 48, 50, 53, 55, 58, 60]

    tdf_im = apply_scoring(all_data['IM']['tdf'])
    rdf_a, sweet_a = scan_thresholds(
        tdf_im, thresholds_fine,
        v2_stats['IM']['n'], v2_stats['IM']['total'], v2_stats['IM']['avg'],
        all_data['IM']['dates'])

    doc.append(f"v2基线: {v2_stats['IM']['n']}笔, 总{v2_stats['IM']['total']:+.0f}, 单笔{v2_stats['IM']['avg']:+.2f}\n")
    format_scan_table(rdf_a, doc)

    if sweet_a:
        doc.append(f"\n**Phase A Sweet Spot找到: thr={sweet_a}** ✓\n")
        doc.append("后续阶段跳过。")
    else:
        doc.append(f"\n**Phase A 无Sweet Spot**\n")
        # 找最接近的
        rdf_a['conditions_met'] = rdf_a['signal_reduced'].astype(int) + rdf_a['pnl_ok'].astype(int) + rdf_a['eff_ok'].astype(int)
        best_a = rdf_a.loc[rdf_a['conditions_met'].idxmax()]
        doc.append(f"最接近: thr={best_a['thr']} (满足{int(best_a['conditions_met'])}/3条件)\n")

    # ═══════════════════════════════════════════════
    # Phase B: 组件调整（仅在Phase A未找到时）
    # ═══════════════════════════════════════════════
    if sweet_a is None:
        doc.append("# Phase B: v1组件局部调整\n")

        variants = {}

        # B1: 时段加分力度减半
        def session_b1(h):
            return {9: 5, 10: -5, 11: -3, 13: 0, 14: 5}.get(h, 0)

        # B2: M分高动量合并到35
        def m_b2(raw):
            if raw < 0.0005: return 0
            elif raw < 0.001: return 5
            elif raw < 0.002: return 15
            elif raw < 0.003: return 25
            else: return 35  # 合并40/50到35

        # B3: V分平缓U型
        def v_b3(ratio):
            if ratio < 0.6: return 30
            elif ratio < 0.8: return 25
            elif ratio < 1.0: return 15   # 从10提高
            elif ratio < 1.2: return 10   # 从5提高
            elif ratio < 1.5: return 20
            elif ratio < 2.0: return 25
            else: return 30

        variant_configs = {
            'B1(时段减半)': {'session_fn': session_b1},
            'B2(M封顶35)': {'m_fn': m_b2},
            'B3(V平缓U)': {'v_fn': v_b3},
            'B1+B2': {'session_fn': session_b1, 'm_fn': m_b2},
            'B1+B3': {'session_fn': session_b1, 'v_fn': v_b3},
            'B2+B3': {'m_fn': m_b2, 'v_fn': v_b3},
            'B1+B2+B3': {'session_fn': session_b1, 'm_fn': m_b2, 'v_fn': v_b3},
        }

        found_sweet = False
        for vname, cfg in variant_configs.items():
            tdf_v = apply_scoring(all_data['IM']['tdf'], **cfg)
            rdf_v, sweet_v = scan_thresholds(
                tdf_v, thresholds_fine,
                v2_stats['IM']['n'], v2_stats['IM']['total'], v2_stats['IM']['avg'],
                all_data['IM']['dates'])

            doc.append(f"## {vname}\n")
            format_scan_table(rdf_v, doc)

            if sweet_v:
                doc.append(f"\n**Sweet Spot找到: {vname} thr={sweet_v}** ✓\n")

                # 同时跑IC看影响
                tdf_ic_v = apply_scoring(all_data['IC']['tdf'], **cfg)
                ic_v1 = tdf_ic_v[tdf_ic_v['v1_total'] >= 55]  # IC用thr=55
                ic_avg = ic_v1['pnl_pts'].mean() if len(ic_v1) > 0 else 0
                ic_total = ic_v1['pnl_pts'].sum()
                doc.append(f"IC影响(thr=55): {len(ic_v1)}笔, 总{ic_total:+.0f}, 单笔{ic_avg:+.2f} "
                           f"(vs v2 {v2_stats['IC']['total']:+.0f})")

                found_sweet = True
                break
            else:
                rdf_v['cond'] = rdf_v['signal_reduced'].astype(int) + rdf_v['pnl_ok'].astype(int) + rdf_v['eff_ok'].astype(int)
                best = rdf_v.loc[rdf_v['cond'].idxmax()]
                doc.append(f"  无Sweet Spot. 最接近: thr={best['thr']} ({int(best['cond'])}/3)\n")

        if not found_sweet:
            doc.append("\n**Phase B 全部变体无Sweet Spot**\n")

    # ═══════════════════════════════════════════════
    # Phase D: 物理差异分析（如果A/B都失败）
    # ═══════════════════════════════════════════════
    if sweet_a is None and not found_sweet:
        doc.append("# Phase D: IM/IC物理差异分析\n")

        for sym in ['IM', 'IC']:
            tdf = apply_scoring(all_data[sym]['tdf'])
            v2_thr = V2_THR[sym]
            v2_trades = tdf[tdf['entry_score'] >= v2_thr]
            pnls = v2_trades['pnl_pts'].values

            doc.append(f"## {sym}\n")

            # PnL集中度
            sorted_pnl = np.sort(pnls)[::-1]  # 降序
            total = pnls.sum()
            for pct in [10, 20, 30, 50]:
                top_n = int(len(sorted_pnl) * pct / 100)
                top_sum = sorted_pnl[:top_n].sum()
                contrib = top_sum / total * 100 if total != 0 else 0
                doc.append(f"Top {pct}% trade贡献: {contrib:.0f}%")

            # PnL分布
            doc.append(f"\nPnL分布:")
            for lo, hi in [(-999, -20), (-20, -10), (-10, 0), (0, 10), (10, 20), (20, 999)]:
                sub = v2_trades[(v2_trades['pnl_pts'] >= lo) & (v2_trades['pnl_pts'] < hi)]
                doc.append(f"  [{lo},{hi}): {len(sub)}笔 ({len(sub)/len(v2_trades)*100:.0f}%)")

            doc.append(f"\n单笔PnL标准差: {pnls.std():.1f}")

            # v2高分trade在v1上的分布
            doc.append(f"\nv2 score>={v2_thr}的trade在v1上的分布:")
            v1_scores = v2_trades['v1_total']
            doc.append(f"  v1 mean={v1_scores.mean():.1f}, std={v1_scores.std():.1f}")
            doc.append(f"  v1 [<40]={( v1_scores<40).sum()}, [40-60]={((v1_scores>=40)&(v1_scores<60)).sum()}, "
                       f"[60-80]={((v1_scores>=60)&(v1_scores<80)).sum()}, [80+]={(v1_scores>=80).sum()}")
            doc.append("")

    # 综合结论
    doc.append("# 综合结论\n")
    if sweet_a:
        doc.append(f"**Phase A找到Sweet Spot: thr={sweet_a}**")
    elif found_sweet:
        doc.append(f"**Phase B找到Sweet Spot**")
    else:
        doc.append("**A+B均未找到Sweet Spot，进入Phase D物理分析**")

    report = "\n".join(doc)
    path = Path("tmp") / "im_sweet_spot_search.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
