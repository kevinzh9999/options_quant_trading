#!/usr/bin/env python3
"""v1精细threshold扫描 + v1/v2重叠分析。复用阶段1的trade数据(thr=30收集)。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from data.storage.db_manager import get_db
from scripts.backtest_signals_day import run_day
from strategies.intraday.A_share_momentum_signal_v2 import SYMBOL_PROFILES
from strategies.intraday.experimental.score_components_new import compute_total_score

SPOTS = {'IC': '000905', 'IM': '000852'}
V2_THR = {'IM': 60, 'IC': 55}
V1_FINE_THR = [25, 30, 35, 40, 45, 50, 55, 60]


def get_dates(db, spot):
    df = db.query_df(
        f"SELECT DISTINCT substr(datetime,1,10) as d FROM index_min "
        f"WHERE symbol='{spot}' AND period=300 ORDER BY d"
    )
    dates = [d.replace('-', '') for d in df['d'].tolist()]
    return dates[-900:] if len(dates) > 900 else dates


def _run_one_day(args):
    td, sym = args
    SYMBOL_PROFILES[sym]["signal_threshold"] = 20  # 极低收集全量
    db = get_db()
    trades = run_day(sym, td, db, verbose=False)
    full = [t for t in trades if not t.get("partial")]
    for t in full:
        t['trade_date'] = td
        t['symbol'] = sym
        # v1 score
        raw_mom = abs(t.get('raw_mom_5m', 0.0))
        raw_atr = t.get('raw_atr_ratio', 0.0)
        raw_vpct = t.get('raw_vol_pct', -1.0)
        raw_vratio = t.get('raw_vol_ratio', -1.0)
        entry_time = t.get('entry_time', '13:00')
        try:
            hour_bj = int(entry_time[:2])
        except (ValueError, IndexError):
            hour_bj = 13
        gap_aligned = t.get('entry_gap_aligned', False)
        r = compute_total_score(raw_mom, raw_atr, raw_vpct, raw_vratio, hour_bj, gap_aligned)
        t['v1_total'] = r['total_score']
        t['v1_m'] = r['m_score']
        t['v1_v'] = r['v_score']
        t['v1_q'] = r['q_score']
        t['v1_session'] = r['session_bonus']
        t['v1_gap'] = r['gap_bonus']
    return full


def main():
    print("=" * 60)
    print("  v1 精细Threshold扫描 + 重叠分析")
    print("=" * 60)

    db = get_db()
    n_workers = min(cpu_count(), 8)
    doc = ["# v1 精细Threshold扫描 + 重叠分析\n"]

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

    # ═══════════════════════════════════════════════
    # Part A: 精细threshold扫描
    # ═══════════════════════════════════════════════
    doc.append("# Part A: 精细Threshold扫描\n")

    for sym in ['IM', 'IC']:
        tdf = all_data[sym]['tdf']
        dates = all_data[sym]['dates']
        is_set = set(dates[-219:])
        oos_set = set(dates[:-219])
        v2_thr = V2_THR[sym]

        # v2基线
        v2_trades = tdf[tdf['entry_score'] >= v2_thr]
        v2_n = len(v2_trades)
        v2_total = v2_trades['pnl_pts'].sum()
        v2_avg = v2_trades['pnl_pts'].mean()
        v2_wr = (v2_trades['pnl_pts'] > 0).sum() / v2_n * 100 if v2_n > 0 else 0
        v2_is = v2_trades[v2_trades['trade_date'].isin(is_set)]['pnl_pts'].sum()
        v2_oos = v2_trades[v2_trades['trade_date'].isin(oos_set)]['pnl_pts'].sum()
        v2_ratio = (v2_is / 219) / (v2_oos / 681) if v2_oos > 0 else 99

        doc.append(f"## {sym}\n")
        doc.append(f"v2基线(thr={v2_thr}): {v2_n}笔, 单笔{v2_avg:+.2f}, 总{v2_total:+.0f}, WR={v2_wr:.1f}%, IS/OOS={v2_ratio:.2f}\n")
        doc.append("| thr | 笔数 | 总PnL | 单笔PnL | 单笔vs v2 | 总vs v2 | 频率比 | WR | IS/OOS |")
        doc.append("|-----|------|-------|---------|----------|---------|-------|-----|--------|")

        sweet_spot = None
        for thr in V1_FINE_THR:
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

            mark = ""
            if avg_vs > 0 and total_vs >= 0 and freq_ratio >= 0.5:
                mark = " ★sweet"
                if sweet_spot is None:
                    sweet_spot = thr

            doc.append(f"| {thr} | {n} | {total:+.0f} | {avg:+.2f} | {avg_vs:+.1f}% | "
                       f"{total_vs:+.1f}% | {freq_ratio:.2f} | {wr:.1f}% | {ratio:.2f} |{mark}")

        if sweet_spot:
            doc.append(f"\n**Sweet spot: thr={sweet_spot}** (单笔>v2 + 总PnL>=v2 + 频率>=0.5x)")
        else:
            doc.append(f"\n**无sweet spot** (没有threshold同时满足三个条件)")
        doc.append("")

    # ═══════════════════════════════════════════════
    # Part B: 重叠分析
    # ═══════════════════════════════════════════════
    doc.append("# Part B: v1/v2 重叠分析\n")

    for sym in ['IM', 'IC']:
        tdf = all_data[sym]['tdf']
        v2_thr = V2_THR[sym]

        # 集合定义
        set_v2 = tdf[tdf['entry_score'] >= v2_thr]                    # v2接受
        set_v1 = tdf[tdf['v1_total'] >= 20]                           # v1接受(thr=20)
        set_overlap = tdf[(tdf['v1_total'] >= 20) & (tdf['entry_score'] >= v2_thr)]  # 两者都接受
        set_v1_only = tdf[(tdf['v1_total'] >= 20) & (tdf['entry_score'] < v2_thr)]   # v1独有
        set_v2_only = tdf[(tdf['entry_score'] >= v2_thr) & (tdf['v1_total'] < 20)]   # v2独有

        doc.append(f"## {sym}\n")
        doc.append("| 集合 | 笔数 | 单笔PnL | 总PnL | 胜率 |")
        doc.append("|------|------|---------|-------|------|")

        for label, subset in [
            ('V2全部', set_v2),
            ('V1全部(thr=20)', set_v1),
            ('**V1∩V2(重叠)**', set_overlap),
            ('V1\\V2(v1独有)', set_v1_only),
            ('V2\\V1(v2独有)', set_v2_only),
        ]:
            n = len(subset)
            if n > 0:
                avg = subset['pnl_pts'].mean()
                total = subset['pnl_pts'].sum()
                wr = (subset['pnl_pts'] > 0).sum() / n * 100
                doc.append(f"| {label} | {n} | {avg:+.2f} | {total:+.0f} | {wr:.1f}% |")
            else:
                doc.append(f"| {label} | 0 | - | - | - |")

        # 判定B1
        v2_avg = set_v2['pnl_pts'].mean() if len(set_v2) > 0 else 0
        overlap_avg = set_overlap['pnl_pts'].mean() if len(set_overlap) > 0 else 0
        diff_pct = (overlap_avg - v2_avg) / abs(v2_avg) * 100 if v2_avg != 0 else 0

        doc.append(f"\n**重叠集合单笔 vs V2全部**: {overlap_avg:+.2f} vs {v2_avg:+.2f} ({diff_pct:+.1f}%)")
        if diff_pct > 5:
            doc.append("→ **v1评分在共同信号上真正更准** ✓")
        elif diff_pct > -5:
            doc.append("→ v1跟v2在共同信号上差不多")
        else:
            doc.append("→ **v1在共同信号上更差** ✗")

        # 判定B2
        v1_only_avg = set_v1_only['pnl_pts'].mean() if len(set_v1_only) > 0 else 0
        doc.append(f"**V1独有信号单笔**: {v1_only_avg:+.2f}")
        if v1_only_avg > 0.5:
            doc.append("→ v1找到了v2漏掉的好信号 ✓")
        elif v1_only_avg > -0.5:
            doc.append("→ v1额外信号是中性的")
        else:
            doc.append("→ v1额外信号质量差 ✗")
        doc.append("")

        # 补充：重叠集合按v1 score段拆解
        doc.append(f"### {sym} 重叠集合按v1_total段拆解\n")
        doc.append("| v1段 | 笔数 | 单笔PnL | WR |")
        doc.append("|------|------|---------|-----|")
        for lo in range(20, 101, 10):
            hi = lo + 10
            sub = set_overlap[(set_overlap['v1_total'] >= lo) & (set_overlap['v1_total'] < hi)]
            if len(sub) >= 30:
                wr = (sub['pnl_pts'] > 0).sum() / len(sub) * 100
                doc.append(f"| [{lo},{hi}) | {len(sub)} | {sub['pnl_pts'].mean():+.2f} | {wr:.0f}% |")
        doc.append("")

    # ═══════════════════════════════════════════════
    # Part C: 综合结论
    # ═══════════════════════════════════════════════
    doc.append("# Part C: 综合结论\n")
    doc.append("(根据Part A和Part B的数据判定情况1-5)")

    report = "\n".join(doc)
    path = Path("tmp") / "new_mapping_v1_fine_scan_and_overlap.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
