#!/usr/bin/env python3
"""new_mapping_v1 backtest验证：用v2跑全量trade(thr=0)，然后用v1评分重映射+threshold过滤。"""
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
V1_THRESHOLDS = [20, 30, 40, 50, 60, 70, 80]


def get_dates(db, spot):
    df = db.query_df(
        f"SELECT DISTINCT substr(datetime,1,10) as d FROM index_min "
        f"WHERE symbol='{spot}' AND period=300 ORDER BY d"
    )
    dates = [d.replace('-', '') for d in df['d'].tolist()]
    return dates[-900:] if len(dates) > 900 else dates


def _run_one_day(args):
    """用极低threshold跑v2，收集全量trade+raw字段。"""
    td, sym = args
    SYMBOL_PROFILES[sym]["signal_threshold"] = 30  # 极低，收集几乎所有信号
    db = get_db()
    trades = run_day(sym, td, db, verbose=False)
    full = [t for t in trades if not t.get("partial")]
    for t in full:
        t['trade_date'] = td
        t['symbol'] = sym
    return full


def apply_v1_score(trade):
    """对一笔trade应用v1评分。"""
    raw_mom = abs(trade.get('raw_mom_5m', 0.0))
    raw_atr = trade.get('raw_atr_ratio', 0.0)
    raw_vpct = trade.get('raw_vol_pct', -1.0)
    raw_vratio = trade.get('raw_vol_ratio', -1.0)

    # 入场时间(BJ)解析
    entry_time = trade.get('entry_time', '13:00')
    try:
        hour_bj = int(entry_time[:2])
    except (ValueError, IndexError):
        hour_bj = 13

    gap_aligned = trade.get('entry_gap_aligned', False)

    result = compute_total_score(
        abs_mom_5m=raw_mom,
        raw_atr_ratio=raw_atr,
        raw_vol_pct=raw_vpct,
        raw_vol_ratio=raw_vratio,
        entry_hour_bj=hour_bj,
        gap_aligned=gap_aligned,
    )

    trade['v1_total'] = result['total_score']
    trade['v1_m'] = result['m_score']
    trade['v1_v'] = result['v_score']
    trade['v1_q'] = result['q_score']
    trade['v1_session'] = result['session_bonus']
    trade['v1_gap'] = result['gap_bonus']
    return trade


def main():
    print("=" * 60)
    print("  new_mapping_v1 Backtest 验证")
    print("=" * 60)

    db = get_db()
    n_workers = min(cpu_count(), 8)
    doc = ["# new_mapping_v1 Backtest 验证报告\n"]

    all_trades = {}
    all_dates = {}

    for sym in ['IM', 'IC']:
        print(f"\n[{sym}] 收集全量trade (thr=30)...")
        dates = get_dates(db, SPOTS[sym])
        args = [(td, sym) for td in dates]
        with Pool(n_workers) as pool:
            day_results = pool.map(_run_one_day, args)
        trades = [t for day in day_results for t in day]
        # 应用v1评分
        trades = [apply_v1_score(t) for t in trades]
        tdf = pd.DataFrame(trades)
        all_trades[sym] = tdf
        all_dates[sym] = dates
        print(f"  {len(tdf)}笔, v1_total范围: [{tdf['v1_total'].min()}, {tdf['v1_total'].max()}]")
        print(f"  v1_total分布: mean={tdf['v1_total'].mean():.1f}, median={tdf['v1_total'].median():.0f}")

    # ═══════════════════════════════════════════════
    # v2基线（用当前参数）
    # ═══════════════════════════════════════════════
    doc.append("## v2 基线\n")
    v2_thresholds = {'IM': 60, 'IC': 55}

    for sym in ['IM', 'IC']:
        tdf = all_trades[sym]
        dates = all_dates[sym]
        is_set = set(dates[-219:])
        oos_set = set(dates[:-219])
        v2_thr = v2_thresholds[sym]

        v2_trades = tdf[tdf['entry_score'] >= v2_thr]
        is_pnl = v2_trades[v2_trades['trade_date'].isin(is_set)]['pnl_pts'].sum()
        oos_pnl = v2_trades[v2_trades['trade_date'].isin(oos_set)]['pnl_pts'].sum()
        total = is_pnl + oos_pnl
        n = len(v2_trades)
        wr = (v2_trades['pnl_pts'] > 0).sum() / n * 100 if n > 0 else 0

        doc.append(f"### {sym} v2基线 (thr={v2_thr})")
        doc.append(f"  笔数={n}, IS={is_pnl:+.0f}, OOS={oos_pnl:+.0f}, 合计={total:+.0f}, WR={wr:.0f}%\n")

    # ═══════════════════════════════════════════════
    # v1 score分布诊断
    # ═══════════════════════════════════════════════
    doc.append("## v1 Score分布诊断\n")
    for sym in ['IM', 'IC']:
        tdf = all_trades[sym]
        doc.append(f"### {sym} v1_total分布")
        doc.append(f"  范围: [{tdf['v1_total'].min()}, {tdf['v1_total'].max()}]")
        doc.append(f"  均值: {tdf['v1_total'].mean():.1f}, 中位数: {tdf['v1_total'].median():.0f}")
        doc.append(f"  25%: {tdf['v1_total'].quantile(0.25):.0f}, 75%: {tdf['v1_total'].quantile(0.75):.0f}")

        # 各子分量分布
        for col, label in [('v1_m', 'M'), ('v1_v', 'V'), ('v1_q', 'Q'),
                           ('v1_session', 'Session'), ('v1_gap', 'Gap')]:
            doc.append(f"  {label}: mean={tdf[col].mean():.1f}, "
                       f"[{tdf[col].min()}, {tdf[col].max()}]")
        doc.append("")

    # ═══════════════════════════════════════════════
    # v1 threshold扫描
    # ═══════════════════════════════════════════════
    doc.append("## v1 Threshold扫描 (681/219双窗口)\n")

    for sym in ['IM', 'IC']:
        tdf = all_trades[sym]
        dates = all_dates[sym]
        is_set = set(dates[-219:])
        oos_set = set(dates[:-219])
        v2_thr = v2_thresholds[sym]

        # v2基线
        v2_is = tdf[(tdf['entry_score'] >= v2_thr) & (tdf['trade_date'].isin(is_set))]['pnl_pts'].sum()
        v2_oos = tdf[(tdf['entry_score'] >= v2_thr) & (tdf['trade_date'].isin(oos_set))]['pnl_pts'].sum()
        v2_total = v2_is + v2_oos

        doc.append(f"### {sym}\n")
        doc.append(f"v2基线(thr={v2_thr}): IS={v2_is:+.0f}, OOS={v2_oos:+.0f}, 合计={v2_total:+.0f}\n")
        doc.append("| thr | 笔数 | IS_PnL | OOS_PnL | 合计 | IS/OOS比 | WR | vs v2 |")
        doc.append("|-----|------|--------|---------|------|---------|-----|-------|")

        best_total = -99999
        best_thr = 0
        for thr in V1_THRESHOLDS:
            filtered = tdf[tdf['v1_total'] >= thr]
            n = len(filtered)
            if n == 0:
                doc.append(f"| {thr} | 0 | - | - | - | - | - | - |")
                continue
            is_trades = filtered[filtered['trade_date'].isin(is_set)]
            oos_trades = filtered[filtered['trade_date'].isin(oos_set)]
            is_pnl = is_trades['pnl_pts'].sum()
            oos_pnl = oos_trades['pnl_pts'].sum()
            total = is_pnl + oos_pnl
            is_daily = is_pnl / 219
            oos_daily = oos_pnl / 681
            ratio = is_daily / oos_daily if oos_daily > 0 else 99
            wr = (filtered['pnl_pts'] > 0).sum() / n * 100
            diff = total - v2_total

            if total > best_total:
                best_total = total
                best_thr = thr

            doc.append(f"| {thr} | {n} | {is_pnl:+.0f} | {oos_pnl:+.0f} | {total:+.0f} | "
                       f"{ratio:.2f} | {wr:.0f}% | {diff:+.0f} |")

        doc.append(f"\n**v1最优: thr={best_thr}** (合计{best_total:+.0f}, vs v2 {best_total-v2_total:+.0f}pt)\n")

        # OOS最优
        oos_best_thr = 0
        oos_best_pnl = -99999
        for thr in V1_THRESHOLDS:
            filtered = tdf[tdf['v1_total'] >= thr]
            oos_pnl = filtered[filtered['trade_date'].isin(oos_set)]['pnl_pts'].sum()
            if oos_pnl > oos_best_pnl:
                oos_best_pnl = oos_pnl
                oos_best_thr = thr
        doc.append(f"**v1 OOS最优: thr={oos_best_thr}** (OOS={oos_best_pnl:+.0f})\n")

    # ═══════════════════════════════════════════════
    # v1 vs v2 Score段PnL对比
    # ═══════════════════════════════════════════════
    doc.append("## v1 Score段PnL贡献\n")
    for sym in ['IM', 'IC']:
        tdf = all_trades[sym]
        total_pnl = tdf['pnl_pts'].sum()
        doc.append(f"### {sym}\n")
        doc.append("| v1_total段 | 笔数 | AvgPnL | 累计PnL | 贡献% | WR |")
        doc.append("|-----------|------|--------|---------|-------|-----|")
        bins = list(range(-10, 111, 10))
        for i in range(len(bins)-1):
            lo, hi = bins[i], bins[i+1]
            sub = tdf[(tdf['v1_total'] >= lo) & (tdf['v1_total'] < hi)]
            if len(sub) < 10:
                continue
            contrib = sub['pnl_pts'].sum() / total_pnl * 100 if total_pnl != 0 else 0
            wr = (sub['pnl_pts'] > 0).sum() / len(sub) * 100
            doc.append(f"| [{lo},{hi}) | {len(sub)} | {sub['pnl_pts'].mean():+.1f} | "
                       f"{sub['pnl_pts'].sum():+.0f} | {contrib:+.0f}% | {wr:.0f}% |")
        doc.append("")

    # ═══════════════════════════════════════════════
    # 关键判定
    # ═══════════════════════════════════════════════
    doc.append("## 关键判定\n")

    # 判定1: 合理性
    for sym in ['IM', 'IC']:
        tdf = all_trades[sym]
        n = len(tdf)
        wr = (tdf['pnl_pts'] > 0).sum() / n * 100
        doc.append(f"{sym}: 全量{n}笔, v1_total均值={tdf['v1_total'].mean():.1f}, WR={wr:.0f}%")

    doc.append("\n(根据以上数据判定情况A/B/C)")

    report = "\n".join(doc)
    path = Path("tmp") / "new_mapping_v1_backtest_validation.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
