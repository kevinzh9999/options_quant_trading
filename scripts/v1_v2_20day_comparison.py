#!/usr/bin/env python3
"""v2 vs v1 最近20天对比：自研backtest + TqBacktest验证。"""
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


def get_recent_dates(db, n=20):
    """取最近N个交易日。"""
    df = db.query_df(
        "SELECT DISTINCT substr(datetime,1,10) as d FROM index_min "
        "WHERE symbol='000852' AND period=300 ORDER BY d DESC LIMIT ?"
    , params=(n,))
    return sorted([d.replace('-', '') for d in df['d'].tolist()])


def _run_one_day(args):
    td, sym = args
    SYMBOL_PROFILES[sym]["signal_threshold"] = 20  # 极低收集全量
    db = get_db()
    trades = run_day(sym, td, db, verbose=False)
    full = [t for t in trades if not t.get("partial")]
    for t in full:
        t['trade_date'] = td
        t['symbol'] = sym
        # v1评分
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
        t['v1_m'] = v1r['m_score']
        t['v1_v'] = v1r['v_score']
        t['v1_q'] = v1r['q_score']
        t['v1_session'] = v1r['session_bonus']
        t['v1_gap'] = v1r['gap_bonus']
    return full


def main():
    print("=" * 60)
    print("  v2 vs v1 最近20天自研Backtest对比")
    print("=" * 60)

    db = get_db()
    dates = get_recent_dates(db, 20)
    print(f"日期范围: {dates[0]} ~ {dates[-1]} ({len(dates)}天)\n")

    n_workers = min(cpu_count(), 8)
    doc = ["# v2 vs v1 最近20天对比\n"]
    doc.append(f"日期: {dates[0]} ~ {dates[-1]} ({len(dates)}天)\n")

    for sym in ['IM', 'IC']:
        print(f"\n[{sym}] 收集交易数据...")
        args = [(td, sym) for td in dates]
        with Pool(n_workers) as pool:
            day_results = pool.map(_run_one_day, args)
        all_trades = [t for day in day_results for t in day]
        tdf = pd.DataFrame(all_trades)

        v2_thr = V2_THR[sym]
        v1_thr = THR_IM if sym == 'IM' else THR_IC

        v2_trades = tdf[tdf['entry_score'] >= v2_thr]
        v1_trades = tdf[tdf['v1_total'] >= v1_thr]

        doc.append(f"## {sym}\n")
        doc.append(f"### 总览\n")
        doc.append(f"| 系统 | 笔数 | 总PnL | 单笔PnL | 胜率 |")
        doc.append(f"|------|------|-------|---------|------|")
        for label, sub in [('v2', v2_trades), ('v1', v1_trades)]:
            n = len(sub)
            if n > 0:
                pnl = sub['pnl_pts'].sum()
                avg = sub['pnl_pts'].mean()
                wr = (sub['pnl_pts'] > 0).sum() / n * 100
                doc.append(f"| {label} | {n} | {pnl:+.0f} | {avg:+.2f} | {wr:.0f}% |")
        doc.append("")

        # 日度对比
        doc.append(f"### 日度对比\n")
        doc.append(f"| 日期 | v2笔数 | v2 PnL | v1笔数 | v1 PnL | v1-v2 |")
        doc.append(f"|------|--------|--------|--------|--------|-------|")

        for td in dates:
            v2_day = v2_trades[v2_trades['trade_date'] == td]
            v1_day = v1_trades[v1_trades['trade_date'] == td]
            v2_pnl = v2_day['pnl_pts'].sum() if len(v2_day) > 0 else 0
            v1_pnl = v1_day['pnl_pts'].sum() if len(v1_day) > 0 else 0
            doc.append(f"| {td} | {len(v2_day)} | {v2_pnl:+.0f} | {len(v1_day)} | {v1_pnl:+.0f} | {v1_pnl-v2_pnl:+.0f} |")
        doc.append("")

        # 逐笔明细（v2的每笔交易对应的v1状态）
        doc.append(f"### v2逐笔 + v1评分\n")
        doc.append(f"| 日期 | 方向 | 入场 | 出场 | v2score | v1score | v1通过? | v2pnl | 出场原因 |")
        doc.append(f"|------|------|------|------|---------|---------|--------|-------|---------|")
        for _, t in v2_trades.iterrows():
            v1_pass = "✓" if t['v1_total'] >= v1_thr else "✗"
            reason = t.get('reason', t.get('exit_reason', '?'))
            doc.append(f"| {t['trade_date']} | {t['direction']:5s} | {t.get('entry_time','?')} | "
                       f"{t.get('exit_time','?')} | {int(t['entry_score'])} | {int(t['v1_total'])} | "
                       f"{v1_pass} | {t['pnl_pts']:+.1f} | {reason} |")
        doc.append("")

        # v1独有信号（v1通过但v2不通过）
        v1_only = tdf[(tdf['v1_total'] >= v1_thr) & (tdf['entry_score'] < v2_thr)]
        if len(v1_only) > 0:
            doc.append(f"### v1独有信号（v2未触发）\n")
            doc.append(f"| 日期 | 方向 | 入场 | 出场 | v2score | v1score | pnl | 出场原因 |")
            doc.append(f"|------|------|------|------|---------|---------|-----|---------|")
            for _, t in v1_only.iterrows():
                reason = t.get('reason', t.get('exit_reason', '?'))
                doc.append(f"| {t['trade_date']} | {t['direction']:5s} | {t.get('entry_time','?')} | "
                           f"{t.get('exit_time','?')} | {int(t['entry_score'])} | {int(t['v1_total'])} | "
                           f"{t['pnl_pts']:+.1f} | {reason} |")
            doc.append("")

    report = "\n".join(doc)
    path = Path("tmp") / "v1_v2_20day_comparison.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
