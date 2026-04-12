#!/usr/bin/env python3
"""new_mapping_v1 补充分析：单笔效率/信号频率/胜率/IC IS-OOS归因。"""
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


def get_dates(db, spot):
    df = db.query_df(
        f"SELECT DISTINCT substr(datetime,1,10) as d FROM index_min "
        f"WHERE symbol='{spot}' AND period=300 ORDER BY d"
    )
    dates = [d.replace('-', '') for d in df['d'].tolist()]
    return dates[-900:] if len(dates) > 900 else dates


def _run_one_day(args):
    td, sym = args
    SYMBOL_PROFILES[sym]["signal_threshold"] = 30
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
        t['hour_bj'] = hour_bj
    return full


def main():
    print("=" * 60)
    print("  new_mapping_v1 补充分析")
    print("=" * 60)

    db = get_db()
    n_workers = min(cpu_count(), 8)
    doc = ["# new_mapping_v1 补充分析\n"]

    v2_thresholds = {'IM': 60, 'IC': 55}
    v1_threshold = 20

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
    # 问题1: 单笔效率对比
    # ═══════════════════════════════════════════════
    doc.append("## 问题1: 单笔效率对比\n")
    doc.append("| 品种 | v2笔数 | v2单笔PnL | v1笔数 | v1单笔PnL | 单笔效率改善 |")
    doc.append("|------|--------|---------|--------|---------|-----------|")

    for sym in ['IM', 'IC']:
        tdf = all_data[sym]['tdf']
        v2_thr = v2_thresholds[sym]
        v2_trades = tdf[tdf['entry_score'] >= v2_thr]
        v1_trades = tdf[tdf['v1_total'] >= v1_threshold]

        v2_n = len(v2_trades)
        v1_n = len(v1_trades)
        v2_avg = v2_trades['pnl_pts'].mean() if v2_n > 0 else 0
        v1_avg = v1_trades['pnl_pts'].mean() if v1_n > 0 else 0
        improve = (v1_avg - v2_avg) / abs(v2_avg) * 100 if v2_avg != 0 else 0

        doc.append(f"| {sym} | {v2_n} | {v2_avg:+.2f} | {v1_n} | {v1_avg:+.2f} | {improve:+.1f}% |")

    doc.append("")

    # ═══════════════════════════════════════════════
    # 问题2: 信号频率对比
    # ═══════════════════════════════════════════════
    doc.append("## 问题2: 信号频率对比\n")
    doc.append("| 品种 | v2日均信号 | v1日均信号 | 比值(v1/v2) | 在[0.5,2.0]? |")
    doc.append("|------|---------|---------|----------|-----------|")

    for sym in ['IM', 'IC']:
        tdf = all_data[sym]['tdf']
        dates = all_data[sym]['dates']
        n_days = len(dates)
        v2_thr = v2_thresholds[sym]

        v2_daily = len(tdf[tdf['entry_score'] >= v2_thr]) / n_days
        v1_daily = len(tdf[tdf['v1_total'] >= v1_threshold]) / n_days
        ratio = v1_daily / v2_daily if v2_daily > 0 else 99
        in_range = "✓" if 0.5 <= ratio <= 2.0 else "✗"

        doc.append(f"| {sym} | {v2_daily:.1f} | {v1_daily:.1f} | {ratio:.2f} | {in_range} |")

    doc.append("")

    # ═══════════════════════════════════════════════
    # 问题3: 胜率对比
    # ═══════════════════════════════════════════════
    doc.append("## 问题3: 胜率对比\n")
    doc.append("| 品种 | v2胜率 | v1胜率 | 差异(v1-v2) | 满足>=-3%? |")
    doc.append("|------|-------|-------|-----------|---------|")

    for sym in ['IM', 'IC']:
        tdf = all_data[sym]['tdf']
        v2_thr = v2_thresholds[sym]
        v2_trades = tdf[tdf['entry_score'] >= v2_thr]
        v1_trades = tdf[tdf['v1_total'] >= v1_threshold]

        v2_wr = (v2_trades['pnl_pts'] > 0).sum() / len(v2_trades) * 100 if len(v2_trades) > 0 else 0
        v1_wr = (v1_trades['pnl_pts'] > 0).sum() / len(v1_trades) * 100 if len(v1_trades) > 0 else 0
        diff = v1_wr - v2_wr
        ok = "✓" if diff >= -3 else "✗"

        doc.append(f"| {sym} | {v2_wr:.1f}% | {v1_wr:.1f}% | {diff:+.1f}% | {ok} |")

    doc.append("")

    # ═══════════════════════════════════════════════
    # 问题4: IC IS/OOS偏高的组件归因
    # ═══════════════════════════════════════════════
    doc.append("## 问题4: IC IS/OOS组件归因\n")

    for sym in ['IC', 'IM']:
        tdf = all_data[sym]['tdf']
        dates = all_data[sym]['dates']
        is_set = set(dates[-219:])
        oos_set = set(dates[:-219])

        v1_trades = tdf[tdf['v1_total'] >= v1_threshold]
        is_t = v1_trades[v1_trades['trade_date'].isin(is_set)]
        oos_t = v1_trades[v1_trades['trade_date'].isin(oos_set)]

        doc.append(f"### {sym} (IS={len(is_t)}笔, OOS={len(oos_t)}笔)\n")

        # 组件IS/OOS平均值对比
        doc.append("#### 评分组件IS/OOS对比\n")
        doc.append("| 组件 | IS平均值 | OOS平均值 | 差异 |")
        doc.append("|------|---------|---------|------|")

        for col, label in [('v1_m', 'M分'), ('v1_v', 'V分'), ('v1_q', 'Q分'),
                           ('v1_session', 'Session'), ('v1_gap', 'Gap'), ('v1_total', '总score')]:
            is_avg = is_t[col].mean() if len(is_t) > 0 else 0
            oos_avg = oos_t[col].mean() if len(oos_t) > 0 else 0
            doc.append(f"| {label} | {is_avg:.1f} | {oos_avg:.1f} | {is_avg-oos_avg:+.1f} |")

        # PnL的IS/OOS对比
        is_pnl = is_t['pnl_pts'].mean() if len(is_t) > 0 else 0
        oos_pnl = oos_t['pnl_pts'].mean() if len(oos_t) > 0 else 0
        doc.append(f"| **单笔PnL** | **{is_pnl:+.1f}** | **{oos_pnl:+.1f}** | **{is_pnl-oos_pnl:+.1f}** |")
        doc.append("")

        # 时段分布IS/OOS对比
        doc.append("#### 时段分布IS/OOS对比\n")
        doc.append("| 时段 | 加分 | IS笔数 | IS占比 | OOS笔数 | OOS占比 | IS AvgPnL | OOS AvgPnL |")
        doc.append("|------|------|-------|-------|---------|---------|----------|----------|")

        sessions = [(9, '+10'), (10, '-10'), (11, '-5'), (13, '0'), (14, '+10')]
        for hour, bonus in sessions:
            is_sub = is_t[is_t['hour_bj'] == hour]
            oos_sub = oos_t[oos_t['hour_bj'] == hour]
            is_pct = len(is_sub) / len(is_t) * 100 if len(is_t) > 0 else 0
            oos_pct = len(oos_sub) / len(oos_t) * 100 if len(oos_t) > 0 else 0
            is_avg = is_sub['pnl_pts'].mean() if len(is_sub) > 0 else 0
            oos_avg = oos_sub['pnl_pts'].mean() if len(oos_sub) > 0 else 0
            doc.append(f"| {hour:02d}:00 | {bonus} | {len(is_sub)} | {is_pct:.0f}% | "
                       f"{len(oos_sub)} | {oos_pct:.0f}% | {is_avg:+.1f} | {oos_avg:+.1f} |")
        doc.append("")

    # ═══════════════════════════════════════════════
    # 综合判定
    # ═══════════════════════════════════════════════
    doc.append("## 综合判定\n")
    doc.append("(根据以上4个问题的数据给出v1的真实质量改善程度)")

    report = "\n".join(doc)
    path = Path("tmp") / "new_mapping_v1_supplementary_analysis.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
