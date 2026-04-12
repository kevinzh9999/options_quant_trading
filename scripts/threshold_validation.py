#!/usr/bin/env python3
"""Threshold策略的全量数据稳健性验证。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd
from pathlib import Path
from data.storage.db_manager import get_db
from scripts.backtest_signals_day import run_day
from strategies.intraday.A_share_momentum_signal_v2 import SYMBOL_PROFILES


def get_dates(db, spot, start=None, end=None):
    df = db.query_df(
        f"SELECT DISTINCT substr(datetime,1,10) as d FROM index_min "
        f"WHERE symbol='{spot}' AND period=300 ORDER BY d"
    )
    dates = [d.replace('-','') for d in df['d'].tolist()]
    if start: dates = [d for d in dates if d >= start]
    if end: dates = [d for d in dates if d <= end]
    return dates


def run_threshold_scan(sym, dates, thresholds, db):
    """跑多个threshold，返回汇总结果。"""
    orig_thr = SYMBOL_PROFILES[sym].get("signal_threshold", 60)
    results = []
    for thr in thresholds:
        SYMBOL_PROFILES[sym]["signal_threshold"] = thr
        daily_pnl = []; total_trades = 0; total_wins = 0; total_pnl = 0
        for td in dates:
            trades = run_day(sym, td, db, verbose=False)
            full = [t for t in trades if not t.get("partial")]
            pnl = sum(t["pnl_pts"] for t in full)
            daily_pnl.append(pnl)
            total_trades += len(full)
            total_wins += sum(1 for t in full if t["pnl_pts"] > 0)
            total_pnl += pnl
        wr = total_wins / total_trades * 100 if total_trades > 0 else 0
        avg = total_pnl / len(dates) if dates else 0
        std = np.std(daily_pnl) if daily_pnl else 1
        sharpe = avg / std * np.sqrt(252) if std > 0 else 0
        results.append({
            'thr': thr, 'pnl': total_pnl, 'trades': total_trades,
            'wr': wr, 'avg': avg, 'sharpe': sharpe, 'days': len(dates),
        })
        print(f"  {sym} thr={thr} [{dates[0]}~{dates[-1]}]: {total_pnl:+.0f}pt {total_trades}笔 WR={wr:.1f}% Sharpe={sharpe:.2f}")
    SYMBOL_PROFILES[sym]["signal_threshold"] = orig_thr
    return pd.DataFrame(results)


def run_profiling_summary(sym, dates, db, bin_width=5, bin_min=50):
    """跑一次profiling的简化版，只返回per-bin的MFE/MAE统计。"""
    SYMBOL_PROFILES[sym]["signal_threshold"] = bin_min  # 允许最低分交易
    all_trades = []
    for td in dates:
        trades = run_day(sym, td, db, verbose=False)
        full = [t for t in trades if not t.get("partial")]
        for t in full:
            t['trade_date'] = td; t['symbol'] = sym
        all_trades.extend(full)
    SYMBOL_PROFILES[sym]["signal_threshold"] = 60 if sym == 'IC' else 55  # restore

    if not all_trades:
        return pd.DataFrame()

    # 加载ohlcv做MFE/MAE
    from scripts.entry_score_profiling import load_ohlcv, enrich_trades
    ohlcv = load_ohlcv(None, sym, dates[0], dates[-1])
    tdf = pd.DataFrame(all_trades)
    enriched = enrich_trades(tdf, ohlcv, n_bars_list=[24], amp_threshold=0.004)

    if len(enriched) == 0:
        return pd.DataFrame()

    # 分箱
    bins = list(range(bin_min, 101, bin_width))
    if bins[-1] != 100: bins.append(100)
    labels = [f'[{bins[i]},{bins[i+1]})' for i in range(len(bins)-2)] + [f'[{bins[-2]},{bins[-1]}]']
    enriched['score_bin'] = pd.cut(enriched['entry_score'], bins=bins, labels=labels, right=False, include_lowest=True)

    # 聚合
    agg = enriched.groupby('score_bin', observed=True).agg(
        n=('entry_score', 'count'),
        med_mfe=('mfe_fixed_24', 'median'),
        med_mae=('mae_fixed_24', 'median'),
        wr=('actual_win', 'mean'),
    ).reset_index()
    agg['mfe_mae'] = agg['med_mfe'] / agg['med_mae'].replace(0, np.nan)
    agg = agg[agg['n'] > 0]
    return agg


if __name__ == "__main__":
    db = get_db()
    output_dir = Path("tmp/threshold_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = ["# Threshold 策略全量数据稳健性验证\n"]

    SPOT = {"IM": "000852", "IC": "000905"}
    THRESHOLDS = [50, 55, 60, 65]

    # ═══════════════════════════════════════════════════════════
    # IM: 900天
    # ═══════════════════════════════════════════════════════════
    print("=" * 60)
    print("  IM 全量验证 (900天)")
    print("=" * 60)

    im_dates = get_dates(db, SPOT['IM'])
    doc.append(f"## IM 全量验证\n")
    doc.append(f"数据窗口: {im_dates[0]}~{im_dates[-1]} ({len(im_dates)}天)\n")

    # 第二步：全量profiling
    print("\n[IM] 全量细分箱profiling...")
    im_prof = run_profiling_summary('IM', im_dates, db)
    doc.append("### 全量细分箱 profiling (900天)\n")
    doc.append(f"{'Bin':<12s} {'N':>5s} {'Med_MFE':>9s} {'Med_MAE':>9s} {'MFE/MAE':>8s} {'WR':>6s}")
    doc.append("-" * 55)
    for _, r in im_prof.iterrows():
        mark = " ◀" if '[75,80)' in str(r['score_bin']) else (" ★" if '[50,55)' in str(r['score_bin']) else "")
        doc.append(f"{str(r['score_bin']):<12s} {int(r['n']):>5d} {r['med_mfe']:>9.1f} {r['med_mae']:>9.1f} {r['mfe_mae']:>8.2f} {r['wr']*100:>5.1f}%{mark}")

    # 第三步：时间分段
    n_im = len(im_dates)
    split = int(n_im * 2 / 3)  # ~600天 in-sample, ~300天 out-of-sample
    im_is = im_dates[:split]
    im_oos = im_dates[split:]
    doc.append(f"\n### 时间分段验证\n")
    doc.append(f"In-sample: {im_is[0]}~{im_is[-1]} ({len(im_is)}天)")
    doc.append(f"Out-of-sample: {im_oos[0]}~{im_oos[-1]} ({len(im_oos)}天)\n")

    print(f"\n[IM] In-sample ({len(im_is)}天) threshold扫描...")
    im_is_results = run_threshold_scan('IM', im_is, THRESHOLDS, db)
    is_best = im_is_results.loc[im_is_results['pnl'].idxmax(), 'thr']

    doc.append("**In-sample结果:**\n")
    doc.append(f"{'thr':>5s} {'PnL':>8s} {'笔数':>6s} {'WR':>6s} {'Sharpe':>7s}")
    for _, r in im_is_results.iterrows():
        mark = " ◀最优" if r['thr'] == is_best else ""
        doc.append(f"{int(r['thr']):>5d} {r['pnl']:>+8.0f} {int(r['trades']):>6d} {r['wr']:>5.1f}% {r['sharpe']:>7.2f}{mark}")

    print(f"\n[IM] Out-of-sample ({len(im_oos)}天) 验证 thr=50 vs thr=55...")
    oos_thrs = [50, 55]
    if is_best not in oos_thrs:
        oos_thrs.append(int(is_best))
    im_oos_results = run_threshold_scan('IM', im_oos, sorted(oos_thrs), db)

    doc.append(f"\n**Out-of-sample结果 (in-sample最优={is_best}):**\n")
    doc.append(f"{'thr':>5s} {'PnL':>8s} {'笔数':>6s} {'WR':>6s} {'Sharpe':>7s}")
    for _, r in im_oos_results.iterrows():
        doc.append(f"{int(r['thr']):>5d} {r['pnl']:>+8.0f} {int(r['trades']):>6d} {r['wr']:>5.1f}% {r['sharpe']:>7.2f}")

    oos_50 = im_oos_results[im_oos_results['thr']==50]
    oos_55 = im_oos_results[im_oos_results['thr']==55]
    if len(oos_50) > 0 and len(oos_55) > 0:
        pnl_diff = float(oos_55['pnl'].iloc[0] - oos_50['pnl'].iloc[0])
        sharpe_diff = float(oos_55['sharpe'].iloc[0] - oos_50['sharpe'].iloc[0])
        doc.append(f"\nthr=55 vs thr=50 OOS差: PnL={pnl_diff:+.0f}pt Sharpe={sharpe_diff:+.2f}")
        if pnl_diff > 0 and sharpe_diff > 0:
            doc.append("**→ OOS验证通过：thr=55优于thr=50，建议上线**")
        elif pnl_diff <= 0:
            doc.append("**→ OOS验证失败：thr=55不如thr=50，不建议上线**")
        else:
            doc.append("**→ OOS结果混合：PnL/Sharpe方向不一致，谨慎决策**")

    # ═══════════════════════════════════════════════════════════
    # IC: 2000天，三段验证
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  IC 三段验证 (2000天)")
    print("=" * 60)

    ic_dates = get_dates(db, SPOT['IC'])
    # 三段切分
    ic_seg1 = [d for d in ic_dates if d < '20200701']
    ic_seg2 = [d for d in ic_dates if '20200701' <= d < '20221001']
    ic_seg3 = [d for d in ic_dates if d >= '20221001']

    doc.append(f"\n---\n## IC 三段验证\n")
    doc.append(f"数据窗口: {ic_dates[0]}~{ic_dates[-1]} ({len(ic_dates)}天)")
    doc.append(f"Seg1: {ic_seg1[0]}~{ic_seg1[-1]} ({len(ic_seg1)}天)")
    doc.append(f"Seg2: {ic_seg2[0]}~{ic_seg2[-1]} ({len(ic_seg2)}天)")
    doc.append(f"Seg3: {ic_seg3[0]}~{ic_seg3[-1]} ({len(ic_seg3)}天)\n")

    for seg_name, seg_dates in [("Seg1(2018~mid2020)", ic_seg1),
                                 ("Seg2(mid2020~2022)", ic_seg2),
                                 ("Seg3(2022~now)", ic_seg3)]:
        if not seg_dates:
            continue
        print(f"\n[IC] {seg_name} ({len(seg_dates)}天) threshold扫描...")
        ic_seg_results = run_threshold_scan('IC', seg_dates, THRESHOLDS, db)
        seg_best = ic_seg_results.loc[ic_seg_results['pnl'].idxmax(), 'thr']

        doc.append(f"\n### {seg_name}\n")
        doc.append(f"{'thr':>5s} {'PnL':>8s} {'笔数':>6s} {'WR':>6s} {'Sharpe':>7s}")
        for _, r in ic_seg_results.iterrows():
            mark = " ◀最优" if r['thr'] == seg_best else ""
            doc.append(f"{int(r['thr']):>5d} {r['pnl']:>+8.0f} {int(r['trades']):>6d} {r['wr']:>5.1f}% {r['sharpe']:>7.2f}{mark}")

    # IC全量profiling（只跑最近段以节省时间）
    print(f"\n[IC] Seg3 细分箱profiling...")
    ic_prof = run_profiling_summary('IC', ic_seg3, db)
    doc.append(f"\n### IC Seg3 细分箱 profiling ({len(ic_seg3)}天)\n")
    if len(ic_prof) > 0:
        doc.append(f"{'Bin':<12s} {'N':>5s} {'Med_MFE':>9s} {'Med_MAE':>9s} {'MFE/MAE':>8s} {'WR':>6s}")
        doc.append("-" * 55)
        for _, r in ic_prof.iterrows():
            mark = " ◀" if '[75,80)' in str(r['score_bin']) else ""
            doc.append(f"{str(r['score_bin']):<12s} {int(r['n']):>5d} {r['med_mfe']:>9.1f} {r['med_mae']:>9.1f} {r['mfe_mae']:>8.2f} {r['wr']*100:>5.1f}%{mark}")

    # ═══════════════════════════════════════════════════════════
    # 最终结论
    # ═══════════════════════════════════════════════════════════
    doc.append("\n---\n## 最终决策建议\n")
    doc.append("（基于以上数据自动填充，人工复核后上线）")

    report = "\n".join(doc)
    path = output_dir / "threshold_validation_full_window.md"
    with open(path, 'w') as f:
        f.write(report)
    print(f"\n报告已保存: {path}")
