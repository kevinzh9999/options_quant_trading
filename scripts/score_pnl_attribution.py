#!/usr/bin/env python3
"""Part 1: Signal_v2 Score段PnL贡献分析。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from data.storage.db_manager import get_db
from scripts.backtest_signals_day import run_day
from strategies.intraday.A_share_momentum_signal_v2 import SYMBOL_PROFILES

SPOTS = {'IC': '000905', 'IM': '000852'}


def get_dates(db, spot):
    df = db.query_df(
        f"SELECT DISTINCT substr(datetime,1,10) as d FROM index_min "
        f"WHERE symbol='{spot}' AND period=300 ORDER BY d"
    )
    dates = [d.replace('-', '') for d in df['d'].tolist()]
    return dates[-900:] if len(dates) > 900 else dates


def _run_one_day(args):
    td, sym, thr = args
    SYMBOL_PROFILES[sym]["signal_threshold"] = thr
    db = get_db()
    trades = run_day(sym, td, db, verbose=False)
    full = [t for t in trades if not t.get("partial")]
    for t in full:
        t['trade_date'] = td
        t['symbol'] = sym
    return full


def collect_and_enrich(sym, dates, thr=45):
    n_workers = min(cpu_count(), 8)
    args_list = [(td, sym, thr) for td in dates]
    with Pool(n_workers) as pool:
        day_results = pool.map(_run_one_day, args_list)
    all_trades = [t for day in day_results for t in day]
    if not all_trades:
        return pd.DataFrame()
    from scripts.entry_score_profiling import load_ohlcv, enrich_trades
    ohlcv = load_ohlcv(None, sym, dates[0], dates[-1])
    tdf = pd.DataFrame(all_trades)
    return enrich_trades(tdf, ohlcv, n_bars_list=[24], amp_threshold=0.004)


def score_bin_analysis(enriched, sym):
    """按score段做PnL贡献分析。"""
    lines = []
    if len(enriched) == 0:
        lines.append("无数据")
        return lines

    df = enriched.copy()
    total_pnl = df['pnl_pts'].sum()

    bins = list(range(55, 101, 5))
    if bins[-1] != 100:
        bins.append(100)
    labels = [f'[{bins[i]},{bins[i+1]})' for i in range(len(bins) - 2)] + [f'[{bins[-2]},{bins[-1]}]']
    df['score_bin'] = pd.cut(df['entry_score'], bins=bins, labels=labels, right=False, include_lowest=True)

    lines.append(f"| Score段 | 笔数 | 占比 | AvgPnL | 累计PnL | PnL贡献% | 胜率 | MFE/MAE |")
    lines.append(f"|---------|------|------|--------|---------|---------|------|---------|")

    bin_stats = []
    for label in labels:
        sub = df[df['score_bin'] == label]
        if len(sub) == 0:
            continue
        n = len(sub)
        pct = n / len(df) * 100
        avg = sub['pnl_pts'].mean()
        cum = sub['pnl_pts'].sum()
        contrib = cum / total_pnl * 100 if total_pnl != 0 else 0
        wr = (sub['pnl_pts'] > 0).sum() / n * 100
        mfe = sub['mfe_fixed_24'].median()
        mae = sub['mae_fixed_24'].median()
        mm = mfe / mae if mae > 0 else np.nan
        mm_s = f"{mm:.2f}" if not np.isnan(mm) else "-"

        mark = ""
        if cum < 0:
            mark = " ◀负贡献"
        elif contrib > 20:
            mark = " ★主力"

        lines.append(f"| {label} | {n} | {pct:.0f}% | {avg:+.1f} | {cum:+.0f} | {contrib:+.0f}% | {wr:.0f}% | {mm_s} |{mark}")
        bin_stats.append({'bin': label, 'n': n, 'avg': avg, 'cum': cum, 'contrib': contrib, 'wr': wr, 'mfe_mae': mm})

    lines.append(f"\n总计: {len(df)}笔, 累计PnL={total_pnl:+.0f}pt\n")

    bdf = pd.DataFrame(bin_stats)
    if len(bdf) == 0:
        return lines

    # 问题A: 哪个段贡献最多
    lines.append("### 问题A: 利润主要来源\n")
    top_contrib = bdf.loc[bdf['contrib'].idxmax()]
    lines.append(f"- 贡献最多: **{top_contrib['bin']}** ({top_contrib['contrib']:+.0f}%, 累计{top_contrib['cum']:+.0f}pt)")

    # 60-75段的合计贡献
    mid_range = bdf[(bdf['bin'].str.contains('60|65|70'))]
    if len(mid_range) > 0:
        mid_cum = mid_range['cum'].sum()
        mid_contrib = mid_cum / total_pnl * 100 if total_pnl != 0 else 0
        lines.append(f"- [60,75) 合计贡献: {mid_contrib:+.0f}% ({mid_cum:+.0f}pt)")

    # 85+段的合计贡献
    hi_range = bdf[bdf['bin'].str.contains('85|90|95')]
    if len(hi_range) > 0:
        hi_cum = hi_range['cum'].sum()
        hi_contrib = hi_cum / total_pnl * 100 if total_pnl != 0 else 0
        lines.append(f"- [85,100] 合计贡献: {hi_contrib:+.0f}% ({hi_cum:+.0f}pt)")

    # 问题B: 负贡献段
    lines.append("\n### 问题B: 负贡献段\n")
    neg_bins = bdf[bdf['cum'] < 0]
    if len(neg_bins) > 0:
        neg_total = neg_bins['cum'].sum()
        lines.append(f"- 负贡献段: {', '.join(neg_bins['bin'].tolist())}")
        lines.append(f"- 负贡献合计: {neg_total:+.0f}pt")
        lines.append(f"- 剔除后总PnL: {total_pnl - neg_total:+.0f}pt (提升{-neg_total:+.0f}pt, {-neg_total/total_pnl*100:+.0f}%)")
    else:
        lines.append("- 无负贡献段")

    # 问题C: 每笔效率
    lines.append("\n### 问题C: 每笔效率\n")
    valid = bdf[bdf['n'] >= 30]
    if len(valid) > 0:
        best_eff = valid.loc[valid['avg'].idxmax()]
        lines.append(f"- 每笔效率最高(N>=30): **{best_eff['bin']}** (avg={best_eff['avg']:+.1f}pt)")
        worst_eff = valid.loc[valid['avg'].idxmin()]
        lines.append(f"- 每笔效率最低(N>=30): **{worst_eff['bin']}** (avg={worst_eff['avg']:+.1f}pt)")

    return lines, bdf


def monthly_cumulative(enriched, bin_label):
    """某个score段的月度累计PnL轨迹。"""
    df = enriched.copy()
    bins = list(range(55, 101, 5))
    if bins[-1] != 100:
        bins.append(100)
    labels = [f'[{bins[i]},{bins[i+1]})' for i in range(len(bins) - 2)] + [f'[{bins[-2]},{bins[-1]}]']
    df['score_bin'] = pd.cut(df['entry_score'], bins=bins, labels=labels, right=False, include_lowest=True)
    sub = df[df['score_bin'] == bin_label]
    if len(sub) == 0:
        return "无数据"

    sub = sub.copy()
    sub['month'] = sub['trade_date'].apply(lambda x: str(x)[:6])
    monthly = sub.groupby('month')['pnl_pts'].sum()
    cum = monthly.cumsum()

    # 简化描述
    months = cum.index.tolist()
    vals = cum.values
    if len(months) == 0:
        return "无数据"

    peak = np.max(vals)
    peak_month = months[np.argmax(vals)]
    final = vals[-1]
    trough = np.min(vals)
    trough_month = months[np.argmin(vals)]

    monotonic = all(vals[i] >= vals[i-1] for i in range(1, len(vals)))

    desc = f"最终={final:+.0f}pt, 峰值={peak:+.0f}({peak_month}), 谷值={trough:+.0f}({trough_month})"
    if monotonic:
        desc += ", 单调上升 ✓"
    else:
        desc += f", 非单调(最大回撤={peak-trough:.0f}pt)"
    return desc


def main():
    print("=" * 60)
    print("  Score段PnL贡献分析")
    print("=" * 60)

    db = get_db()
    doc = ["# Signal_v2 Score段PnL贡献分析\n"]

    all_bin_stats = {}

    for sym in ['IM', 'IC']:
        print(f"\n[{sym}] 收集数据...")
        dates = get_dates(db, SPOTS[sym])
        enriched = collect_and_enrich(sym, dates, thr=45)
        print(f"  {len(enriched)}笔")

        doc.append(f"## {sym} Score段PnL贡献\n")
        doc.append(f"数据: {dates[0]}~{dates[-1]} ({len(dates)}天, {len(enriched)}笔)\n")

        lines, bdf = score_bin_analysis(enriched, sym)
        doc.extend(lines)
        all_bin_stats[sym] = bdf

        # 累计PnL曲线描述
        doc.append("\n### 关键段累计PnL轨迹\n")
        for bl in ['[60,65)', '[65,70)', '[70,75)', '[85,90)', '[90,95)', '[95,100]']:
            desc = monthly_cumulative(enriched, bl)
            doc.append(f"- {bl}: {desc}")
        doc.append("")

    # ═══════════════════════════════════════════════
    # IM vs IC 对比
    # ═══════════════════════════════════════════════
    doc.append("## IM vs IC 对比\n")
    if 'IM' in all_bin_stats and 'IC' in all_bin_stats:
        im_bdf = all_bin_stats['IM']
        ic_bdf = all_bin_stats['IC']
        doc.append("| Score段 | IM_AvgPnL | IM_贡献% | IC_AvgPnL | IC_贡献% |")
        doc.append("|---------|----------|---------|----------|---------|")
        all_bins_set = sorted(set(im_bdf['bin'].tolist() + ic_bdf['bin'].tolist()))
        for b in all_bins_set:
            im_r = im_bdf[im_bdf['bin'] == b]
            ic_r = ic_bdf[ic_bdf['bin'] == b]
            im_avg = f"{im_r['avg'].iloc[0]:+.1f}" if len(im_r) > 0 else "-"
            im_c = f"{im_r['contrib'].iloc[0]:+.0f}%" if len(im_r) > 0 else "-"
            ic_avg = f"{ic_r['avg'].iloc[0]:+.1f}" if len(ic_r) > 0 else "-"
            ic_c = f"{ic_r['contrib'].iloc[0]:+.0f}%" if len(ic_r) > 0 else "-"
            doc.append(f"| {b} | {im_avg} | {im_c} | {ic_avg} | {ic_c} |")
    doc.append("")

    # ═══════════════════════════════════════════════
    # 综合判定
    # ═══════════════════════════════════════════════
    doc.append("## Part 1 综合判定\n")
    doc.append("(根据以上数据回答: 利润来自哪个分数段? 高分是优势还是劣势?)")

    report = "\n".join(doc)
    path = Path("tmp") / "score_pnl_attribution_and_decay.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
