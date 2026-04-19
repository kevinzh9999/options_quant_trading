#!/usr/bin/env python3
"""高分段崩塌时间定位：按月切片分析IM/IC的[85+]表现。"""
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


def collect_trades(sym, dates, thr=45):
    n_workers = min(cpu_count(), 8)
    args_list = [(td, sym, thr) for td in dates]
    with Pool(n_workers) as pool:
        day_results = pool.map(_run_one_day, args_list)
    return [t for day in day_results for t in day]


def enrich_all(sym, trades, dates):
    if not trades:
        return pd.DataFrame()
    from scripts.entry_score_profiling import load_ohlcv, enrich_trades
    ohlcv = load_ohlcv(None, sym, dates[0], dates[-1])
    tdf = pd.DataFrame(trades)
    return enrich_trades(tdf, ohlcv, n_bars_list=[24], amp_threshold=0.004)


def monthly_stats(enriched, high_threshold=85):
    """按月统计高分段和全体的表现。"""
    if len(enriched) == 0:
        return pd.DataFrame()
    df = enriched.copy()
    df['month'] = df['trade_date'].apply(lambda x: str(x)[:6] if isinstance(x, str) else x.strftime('%Y%m') if hasattr(x, 'strftime') else str(x)[:6])
    df['is_high'] = df['entry_score'] >= high_threshold

    rows = []
    for month in sorted(df['month'].unique()):
        m_all = df[df['month'] == month]
        m_hi = m_all[m_all['is_high']]

        all_pnl = m_all['pnl_pts']
        hi_pnl = m_hi['pnl_pts']

        row = {
            'month': month,
            'n_total': len(m_all),
            'n_high': len(m_hi),
            'all_avg_pnl': all_pnl.mean() if len(all_pnl) > 0 else 0,
            'all_total_pnl': all_pnl.sum() if len(all_pnl) > 0 else 0,
        }
        if len(m_hi) >= 5:
            hi_mfe = m_hi['mfe_fixed_24'].median()
            hi_mae = m_hi['mae_fixed_24'].median()
            row['high_avg_pnl'] = hi_pnl.mean()
            row['high_total_pnl'] = hi_pnl.sum()
            row['high_mfe_mae'] = hi_mfe / hi_mae if hi_mae > 0 else np.nan
        else:
            row['high_avg_pnl'] = np.nan
            row['high_total_pnl'] = np.nan
            row['high_mfe_mae'] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("  高分段崩塌时间定位")
    print("=" * 60)

    db = get_db()
    doc = ["# 高分段崩塌时间定位\n"]

    for sym in ['IM', 'IC']:
        print(f"\n[{sym}] 收集交易数据...")
        dates = get_dates(db, SPOTS[sym])
        trades = collect_trades(sym, dates, thr=45)
        print(f"  {len(trades)}笔交易, enriching...")
        enriched = enrich_all(sym, trades, dates)
        print(f"  enriched: {len(enriched)}笔")

        stats = monthly_stats(enriched)

        doc.append(f"## {sym} 月度高分段表现\n")
        doc.append(f"数据: {dates[0]}~{dates[-1]} ({len(dates)}天, {len(enriched)}笔)\n")

        doc.append("| 月份 | 总笔数 | 高分笔数 | 高分AvgPnL | 高分MFE/MAE | 全体AvgPnL | 高分-全体 |")
        doc.append("|------|-------|---------|-----------|-----------|----------|---------|")

        valid_months = []
        for _, r in stats.iterrows():
            hi_avg = f"{r['high_avg_pnl']:+.1f}" if not np.isnan(r['high_avg_pnl']) else "N/A(<5)"
            hi_mm = f"{r['high_mfe_mae']:.2f}" if not np.isnan(r['high_mfe_mae']) else "-"
            diff = r['high_avg_pnl'] - r['all_avg_pnl'] if not np.isnan(r['high_avg_pnl']) else np.nan
            diff_s = f"{diff:+.1f}" if not np.isnan(diff) else "-"
            doc.append(f"| {r['month']} | {int(r['n_total'])} | {int(r['n_high'])} | "
                       f"{hi_avg} | {hi_mm} | {r['all_avg_pnl']:+.1f} | {diff_s} |")
            if not np.isnan(r['high_avg_pnl']):
                valid_months.append(r)

        if not valid_months:
            doc.append("\n无有效月份(高分笔数>=5)\n")
            continue

        vdf = pd.DataFrame(valid_months)

        # 趋势描述
        doc.append(f"\n### {sym} 趋势描述\n")
        neg_months = vdf[vdf['high_avg_pnl'] < 0]
        pos_months = vdf[vdf['high_avg_pnl'] > 0]
        doc.append(f"- 有效月份: {len(vdf)}个 (高分笔数>=5)")
        doc.append(f"- 高分段正PnL月份: {len(pos_months)}个")
        doc.append(f"- 高分段负PnL月份: {len(neg_months)}个")
        if len(neg_months) > 0:
            doc.append(f"- 负PnL月份列表: {', '.join(neg_months['month'].tolist())}")

        # 转折点识别: 滑动窗口3个月
        doc.append(f"\n### {sym} 转折点分析\n")
        if len(vdf) >= 6:
            vals = vdf['high_avg_pnl'].values
            months_list = vdf['month'].tolist()
            best_split = None
            best_diff = 0
            for i in range(3, len(vals) - 3):
                before = np.mean(vals[:i])
                after = np.mean(vals[i:])
                diff = before - after
                if diff > best_diff:
                    best_diff = diff
                    best_split = i

            if best_split is not None and best_diff >= 5:
                before_avg = np.mean(vals[:best_split])
                after_avg = np.mean(vals[best_split:])
                doc.append(f"**转折点: {months_list[best_split]}**")
                doc.append(f"- 转折前 ({months_list[0]}~{months_list[best_split-1]}): avg={before_avg:+.1f}pt ({best_split}个月)")
                doc.append(f"- 转折后 ({months_list[best_split]}~{months_list[-1]}): avg={after_avg:+.1f}pt ({len(vals)-best_split}个月)")
                doc.append(f"- 差距: {best_diff:.1f}pt")
            else:
                doc.append(f"无清晰转折点 (最大前后差距={best_diff:.1f}pt < 5pt)")

            # 线性趋势
            from scipy import stats as sp_stats
            x = np.arange(len(vals))
            slope, intercept, r, p, se = sp_stats.linregress(x, vals)
            doc.append(f"\n线性趋势: 斜率={slope:+.2f}pt/月, R²={r**2:.3f}, p={p:.4f}")
            if slope < -0.5 and p < 0.1:
                doc.append("→ 有统计显著的下降趋势")
            elif slope < 0:
                doc.append("→ 轻微下降但不显著")
            else:
                doc.append("→ 无下降趋势")

        # 高分-全体差异趋势
        doc.append(f"\n### {sym} 高分段 vs 全体对比\n")
        vdf_diff = vdf.copy()
        vdf_diff['diff'] = vdf_diff['high_avg_pnl'] - vdf_diff['all_avg_pnl']
        neg_diff = vdf_diff[vdf_diff['diff'] < 0]
        doc.append(f"- 高分>全体的月份: {(vdf_diff['diff'] >= 0).sum()}个")
        doc.append(f"- 高分<全体的月份: {(vdf_diff['diff'] < 0).sum()}个")
        doc.append(f"- 差异均值: {vdf_diff['diff'].mean():+.1f}pt")
        doc.append("")

    # ═══════════════════════════════════════════════
    # 综合判定
    # ═══════════════════════════════════════════════
    doc.append("## 综合判定\n")
    doc.append("(根据以上月度数据，选择T1/T2/T3/T4之一)")
    # 这里根据数据动态判定，但需要看完整结果

    report = "\n".join(doc)
    path = Path("tmp") / "high_score_collapse_timing.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
