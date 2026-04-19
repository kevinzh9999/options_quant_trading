#!/usr/bin/env python3
"""用monitor overlay的评分逻辑跑900天全量，验证跟之前的backtest结果是否一致。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from data.storage.db_manager import get_db
from scripts.backtest_signals_day import run_day
from strategies.intraday.A_share_momentum_signal_v2 import SYMBOL_PROFILES

# 直接从overlay模块导入评分函数
from strategies.intraday.experimental.monitor_v1_overlay import MonitorV1Overlay
from strategies.intraday.experimental.signal_new_mapping_v1_im import (
    score as score_im, THRESHOLD as THR_IM, _im_m_score, _im_v_score, _im_session_bonus,
)
from strategies.intraday.experimental.signal_new_mapping_v1_ic import (
    score as score_ic, THRESHOLD as THR_IC,
)
from strategies.intraday.experimental.score_components_new import (
    compute_m_score, compute_v_score, compute_q_score,
    compute_session_bonus, compute_gap_bonus,
)

SPOTS = {'IC': '000905', 'IM': '000852'}
V2_THR = {'IM': 60, 'IC': 55}


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
        raw_mom = abs(t.get('raw_mom_5m', 0.0))
        raw_atr = t.get('raw_atr_ratio', 0.0)
        raw_vpct = t.get('raw_vol_pct', -1.0)
        raw_vratio = t.get('raw_vol_ratio', -1.0)
        try:
            hour_bj = int(t.get('entry_time', '13:00')[:2])
        except:
            hour_bj = 13
        gap_aligned = t.get('entry_gap_aligned', False)

        # 两种评分方式都算
        if sym == 'IM':
            v1r = score_im(raw_mom, raw_atr, raw_vpct, raw_vratio, hour_bj, gap_aligned)
        else:
            v1r = score_ic(raw_mom, raw_atr, raw_vpct, raw_vratio, hour_bj, gap_aligned)

        t['v1_total'] = v1r['total_score']
        t['v1_m'] = v1r['m_score']
        t['v1_v'] = v1r['v_score']
        t['v1_q'] = v1r['q_score']
        t['v1_session'] = v1r['session_bonus']
        t['v1_gap'] = v1r['gap_bonus']

        # 也记录overlay会用的原始数据
        t['_hour_bj'] = hour_bj
        t['_gap_aligned'] = gap_aligned
    return full


def main():
    print("=" * 60)
    print("  v1 overlay 900天全量验证")
    print("=" * 60)

    db = get_db()
    n_workers = min(cpu_count(), 8)
    doc = ["# v1 Overlay 900天全量验证\n"]
    doc.append("目的：验证overlay用的评分逻辑跟之前backtest验证是否一致\n")

    for sym in ['IM', 'IC']:
        print(f"\n[{sym}] 收集900天数据...")
        dates = get_dates(db, SPOTS[sym])
        args = [(td, sym) for td in dates]
        with Pool(n_workers) as pool:
            day_results = pool.map(_run_one_day, args)
        all_trades = [t for day in day_results for t in day]
        tdf = pd.DataFrame(all_trades)
        print(f"  {len(tdf)}笔")

        v2_thr = V2_THR[sym]
        v1_thr = THR_IM if sym == 'IM' else THR_IC

        # 681/219切分
        is_set = set(dates[-219:])
        oos_set = set(dates[:-219])

        v2_all = tdf[tdf['entry_score'] >= v2_thr]
        v1_all = tdf[tdf['v1_total'] >= v1_thr]

        doc.append(f"## {sym} (v1 thr={v1_thr})\n")

        # 全样本
        doc.append(f"### 全样本 ({len(dates)}天)\n")
        doc.append(f"| 系统 | 笔数 | 总PnL | 单笔PnL | 胜率 |")
        doc.append(f"|------|------|-------|---------|------|")
        for label, sub in [('v2', v2_all), ('v1', v1_all)]:
            n = len(sub)
            if n > 0:
                doc.append(f"| {label} | {n} | {sub['pnl_pts'].sum():+.0f} | "
                           f"{sub['pnl_pts'].mean():+.2f} | {(sub['pnl_pts']>0).sum()/n*100:.0f}% |")

        # 681/219双窗口
        doc.append(f"\n### 双窗口 (OOS 681天 / IS 219天)\n")
        doc.append(f"| 系统 | 窗口 | 笔数 | PnL | 单笔PnL |")
        doc.append(f"|------|------|------|-----|---------|")
        for label, sub_all in [('v2', v2_all), ('v1', v1_all)]:
            for wname, wset in [('OOS681', oos_set), ('IS219', is_set)]:
                sub = sub_all[sub_all['trade_date'].isin(wset)]
                n = len(sub)
                if n > 0:
                    doc.append(f"| {label} | {wname} | {n} | {sub['pnl_pts'].sum():+.0f} | "
                               f"{sub['pnl_pts'].mean():+.2f} |")

        # 多时间窗口（跟之前120天对比一致的方式）
        doc.append(f"\n### 多窗口对比（最近N天）\n")
        doc.append(f"| 窗口 | v2笔 | v2总PnL | v2单笔 | v1笔 | v1总PnL | v1单笔 | v1-v2总 |")
        doc.append(f"|------|------|--------|-------|------|--------|-------|---------|")
        for n_days in [20, 60, 120, 219, 450, 681, 900]:
            if n_days > len(dates):
                continue
            recent = set(dates[-n_days:])
            v2_r = v2_all[v2_all['trade_date'].isin(recent)]
            v1_r = v1_all[v1_all['trade_date'].isin(recent)]
            v2_pnl = v2_r['pnl_pts'].sum()
            v1_pnl = v1_r['pnl_pts'].sum()
            v2_avg = v2_r['pnl_pts'].mean() if len(v2_r) > 0 else 0
            v1_avg = v1_r['pnl_pts'].mean() if len(v1_r) > 0 else 0
            doc.append(f"| {n_days}天 | {len(v2_r)} | {v2_pnl:+.0f} | {v2_avg:+.2f} | "
                       f"{len(v1_r)} | {v1_pnl:+.0f} | {v1_avg:+.2f} | {v1_pnl-v2_pnl:+.0f} |")

        # v1 score分布验证
        doc.append(f"\n### v1 Score分布\n")
        doc.append(f"均值={tdf['v1_total'].mean():.1f}, 中位数={tdf['v1_total'].median():.0f}, "
                   f"[{tdf['v1_total'].min()}, {tdf['v1_total'].max()}]")
        doc.append(f"M均值={tdf['v1_m'].mean():.1f}, V均值={tdf['v1_v'].mean():.1f}, "
                   f"Q均值={tdf['v1_q'].mean():.1f}, Session均值={tdf['v1_session'].mean():.1f}")
        doc.append("")

    doc.append("## 跟之前报告的数据对账\n")
    doc.append("之前的数据(阶段1 backtest + 双窗口验证):")
    doc.append("- IC v1(thr=55) 全样本: +4758pt, OOS681: +2840pt")
    doc.append("- IM v1(thr=60) 全样本: +6116pt, OOS681: +4144pt")
    doc.append("\n如果本次数字跟上面不一致，说明代码或数据有变化。")

    report = "\n".join(doc)
    path = Path("tmp") / "v1_overlay_900day_test.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
