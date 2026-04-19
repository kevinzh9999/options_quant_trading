#!/usr/bin/env python3
"""IC信号质量综合诊断: 子分量分析 + IM对照。"""
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
    """收集交易并做MFE/MAE enrichment。"""
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
    enriched = enrich_trades(tdf, ohlcv, n_bars_list=[24], amp_threshold=0.004)
    return enriched


def bin_profiling(enriched, bin_width=5, bin_min=50):
    """按score分箱做MFE/MAE profiling。"""
    if len(enriched) == 0:
        return pd.DataFrame()
    bins = list(range(bin_min, 101, bin_width))
    if bins[-1] != 100:
        bins.append(100)
    labels = [f'[{bins[i]},{bins[i+1]})' for i in range(len(bins) - 2)] + [f'[{bins[-2]},{bins[-1]}]']
    enriched = enriched.copy()
    enriched['score_bin'] = pd.cut(
        enriched['entry_score'], bins=bins, labels=labels,
        right=False, include_lowest=True
    )
    agg = enriched.groupby('score_bin', observed=True).agg(
        n=('entry_score', 'count'),
        med_mfe=('mfe_fixed_24', 'median'),
        med_mae=('mae_fixed_24', 'median'),
        avg_pnl=('pnl_pts', 'mean'),
        wr=('actual_win', 'mean'),
    ).reset_index()
    agg['mfe_mae'] = agg['med_mfe'] / agg['med_mae'].replace(0, np.nan)
    return agg[agg['n'] > 0]


def sub_component_table(data, label, full_data):
    """生成子分量分布表。"""
    lines = []
    cols = ['entry_m', 'entry_v', 'entry_q', 'entry_b', 'entry_s']
    available = [c for c in cols if c in data.columns and c in full_data.columns]
    if not available:
        lines.append("子分量列不存在")
        return lines
    lines.append(f"| 子分量 | {label}均值 | {label}中位数 | 全样本均值 | 差异 |")
    lines.append(f"|--------|-----------|------------|----------|------|")
    for col in available:
        t_mean = data[col].mean()
        t_med = data[col].median()
        f_mean = full_data[col].mean()
        lines.append(f"| {col.replace('entry_','')} | {t_mean:.1f} | {t_med:.1f} | {f_mean:.1f} | {t_mean-f_mean:+.1f} |")
    return lines


def main():
    print("=" * 60)
    print("  IC 信号质量综合诊断 + IM 对照")
    print("=" * 60)

    db = get_db()
    doc = ["# IC 信号质量综合诊断\n"]

    # ═══════════════════════════════════════════════
    # Step 1: 确认子分量数据可用
    # ═══════════════════════════════════════════════
    doc.append("## Step 1: 子分量数据确认\n")
    doc.append("backtest_signals_day.py 已记录 entry_m/v/q/b/s_score。")
    doc.append("enrich_trades() 已保留为 entry_m/v/q/b/s 列。无需修改代码。\n")

    # ═══════════════════════════════════════════════
    # Step 2: IC 子分量分析
    # ═══════════════════════════════════════════════
    print("\n[Step 2] 收集IC交易数据...")
    ic_dates = get_dates(db, SPOTS['IC'])
    n_ic = len(ic_dates)
    split_ic = int(n_ic * 2 / 3)
    ic_is_dates = ic_dates[:split_ic]
    ic_oos_dates = ic_dates[split_ic:]

    print(f"  IC: {n_ic}天, IS={len(ic_is_dates)}, OOS={len(ic_oos_dates)}")

    print("  收集IS段...")
    ic_is = collect_and_enrich('IC', ic_is_dates, thr=45)
    print(f"  IS: {len(ic_is)}笔")
    print("  收集OOS段...")
    ic_oos = collect_and_enrich('IC', ic_oos_dates, thr=45)
    print(f"  OOS: {len(ic_oos)}笔")

    ic_all = pd.concat([ic_is, ic_oos], ignore_index=True)

    doc.append("## Step 2: IC 子分量分析\n")
    doc.append(f"IC {n_ic}天, IS={len(ic_is_dates)}天({len(ic_is)}笔), OOS={len(ic_oos_dates)}天({len(ic_oos)}笔)\n")

    # 2.1 IS vs OOS profiling对比
    doc.append("### 2.1 IS vs OOS Profiling对比\n")
    ic_is_prof = bin_profiling(ic_is)
    ic_oos_prof = bin_profiling(ic_oos)

    is_dict = {str(r['score_bin']): r for _, r in ic_is_prof.iterrows()}
    oos_dict = {str(r['score_bin']): r for _, r in ic_oos_prof.iterrows()}
    all_bins = sorted(set(list(is_dict.keys()) + list(oos_dict.keys())))

    doc.append("| Bin | IS_N | IS_MFE/MAE | IS_PnL | OOS_N | OOS_MFE/MAE | OOS_PnL |")
    doc.append("|-----|------|-----------|--------|-------|-------------|---------|")
    for b in all_bins:
        is_r = is_dict.get(b)
        oos_r = oos_dict.get(b)
        is_n = int(is_r['n']) if is_r is not None else 0
        is_mm = f"{is_r['mfe_mae']:.2f}" if is_r is not None else "-"
        is_pnl = f"{is_r['avg_pnl']:+.1f}" if is_r is not None else "-"
        oos_n = int(oos_r['n']) if oos_r is not None else 0
        oos_mm = f"{oos_r['mfe_mae']:.2f}" if oos_r is not None else "-"
        oos_pnl = f"{oos_r['avg_pnl']:+.1f}" if oos_r is not None else "-"
        mark = ""
        if '[55,60)' in b: mark = " ◀中分段"
        elif '[85,' in b or '[90,' in b or '[95,' in b: mark = " ◀高分段"
        doc.append(f"| {b} | {is_n} | {is_mm} | {is_pnl} | {oos_n} | {oos_mm} | {oos_pnl} |{mark}")
    doc.append("")

    # 2.2 [55,60) 子分量分布
    doc.append("### 2.2 [55,60) 子分量分布\n")
    target_55_60 = ic_all[(ic_all['entry_score'] >= 55) & (ic_all['entry_score'] < 60)]
    doc.append(f"样本: {len(target_55_60)}笔\n")

    # IS vs OOS分开
    t_is = ic_is[(ic_is['entry_score'] >= 55) & (ic_is['entry_score'] < 60)]
    t_oos = ic_oos[(ic_oos['entry_score'] >= 55) & (ic_oos['entry_score'] < 60)]

    doc.append("**IS段 [55,60):**\n")
    doc.extend(sub_component_table(t_is, 'IS[55,60)', ic_is))
    doc.append(f"\n**OOS段 [55,60):**\n")
    doc.extend(sub_component_table(t_oos, 'OOS[55,60)', ic_oos))
    doc.append("")

    # 2.3 [85,100] 子分量分布 IS vs OOS
    doc.append("### 2.3 [85,100] 高分段子分量分布 (IS vs OOS)\n")
    hi_is = ic_is[ic_is['entry_score'] >= 85]
    hi_oos = ic_oos[ic_oos['entry_score'] >= 85]
    doc.append(f"IS段[85,100]: {len(hi_is)}笔, OOS段[85,100]: {len(hi_oos)}笔\n")

    doc.append("**IS段 [85,100]:**\n")
    doc.extend(sub_component_table(hi_is, 'IS[85+]', ic_is))
    doc.append(f"\n**OOS段 [85,100]:**\n")
    doc.extend(sub_component_table(hi_oos, 'OOS[85+]', ic_oos))
    doc.append("")

    # 2.4 LONG vs SHORT 拆解
    doc.append("### 2.4 LONG vs SHORT 拆解\n")
    if 'direction' in ic_all.columns:
        for bin_label, bin_data in [('[55,60)', target_55_60), ('[85,100]', ic_all[ic_all['entry_score'] >= 85])]:
            doc.append(f"**{bin_label}:**\n")
            doc.append(f"| 方向 | N | AvgPnL | WR | M均值 | V均值 | Q均值 |")
            doc.append(f"|------|---|--------|-----|-------|-------|-------|")
            for d in ['LONG', 'SHORT']:
                sub = bin_data[bin_data['direction'] == d]
                if len(sub) >= 10:
                    wr = (sub['pnl_pts'] > 0).sum() / len(sub) * 100
                    m = sub['entry_m'].mean() if 'entry_m' in sub.columns else 0
                    v = sub['entry_v'].mean() if 'entry_v' in sub.columns else 0
                    q = sub['entry_q'].mean() if 'entry_q' in sub.columns else 0
                    doc.append(f"| {d} | {len(sub)} | {sub['pnl_pts'].mean():+.1f} | {wr:.0f}% | {m:.1f} | {v:.1f} | {q:.1f} |")
            doc.append("")

    # 2.5 V分分组（[55,60)内部）
    doc.append("### 2.5 [55,60) 内部按V分分组\n")
    if 'entry_v' in target_55_60.columns:
        doc.append("| V分组 | N | AvgPnL | WR | M均值 |")
        doc.append("|-------|---|--------|-----|-------|")
        for lo, hi, label in [(0, 10, 'V<10'), (10, 20, 'V 10-20'), (20, 31, 'V 20+')]:
            sub = target_55_60[(target_55_60['entry_v'] >= lo) & (target_55_60['entry_v'] < hi)]
            if len(sub) >= 30:
                wr = (sub['pnl_pts'] > 0).sum() / len(sub) * 100
                m = sub['entry_m'].mean() if 'entry_m' in sub.columns else 0
                doc.append(f"| {label} | {len(sub)} | {sub['pnl_pts'].mean():+.1f} | {wr:.0f}% | {m:.1f} |")
            elif len(sub) > 0:
                doc.append(f"| {label} | {len(sub)} | (N<30) | | |")
        doc.append("")

    # ═══════════════════════════════════════════════
    # Step 3: IM 对照
    # ═══════════════════════════════════════════════
    print("\n[Step 3] 收集IM交易数据...")
    im_dates = get_dates(db, SPOTS['IM'])
    n_im = len(im_dates)
    split_im = int(n_im * 2 / 3)
    im_is_dates = im_dates[:split_im]
    im_oos_dates = im_dates[split_im:]

    print(f"  IM: {n_im}天, IS={len(im_is_dates)}, OOS={len(im_oos_dates)}")

    print("  收集IS段...")
    im_is = collect_and_enrich('IM', im_is_dates, thr=45)
    print(f"  IS: {len(im_is)}笔")
    print("  收集OOS段...")
    im_oos = collect_and_enrich('IM', im_oos_dates, thr=45)
    print(f"  OOS: {len(im_oos)}笔")

    doc.append("## Step 3: IM 对照\n")
    doc.append(f"IM {n_im}天, IS={len(im_is_dates)}天({len(im_is)}笔), OOS={len(im_oos_dates)}天({len(im_oos)}笔)\n")

    # 3.1 IM profiling
    doc.append("### 3.1 IM IS vs OOS Profiling\n")
    im_is_prof = bin_profiling(im_is)
    im_oos_prof = bin_profiling(im_oos)

    im_is_dict = {str(r['score_bin']): r for _, r in im_is_prof.iterrows()}
    im_oos_dict = {str(r['score_bin']): r for _, r in im_oos_prof.iterrows()}
    im_all_bins = sorted(set(list(im_is_dict.keys()) + list(im_oos_dict.keys())))

    doc.append("| Bin | IS_N | IS_MFE/MAE | IS_PnL | OOS_N | OOS_MFE/MAE | OOS_PnL |")
    doc.append("|-----|------|-----------|--------|-------|-------------|---------|")
    for b in im_all_bins:
        is_r = im_is_dict.get(b)
        oos_r = im_oos_dict.get(b)
        is_n = int(is_r['n']) if is_r is not None else 0
        is_mm = f"{is_r['mfe_mae']:.2f}" if is_r is not None else "-"
        is_pnl = f"{is_r['avg_pnl']:+.1f}" if is_r is not None else "-"
        oos_n = int(oos_r['n']) if oos_r is not None else 0
        oos_mm = f"{oos_r['mfe_mae']:.2f}" if oos_r is not None else "-"
        oos_pnl = f"{oos_r['avg_pnl']:+.1f}" if oos_r is not None else "-"
        doc.append(f"| {b} | {is_n} | {is_mm} | {is_pnl} | {oos_n} | {oos_mm} | {oos_pnl} |")
    doc.append("")

    # 3.2 IM vs IC 对比表
    doc.append("### 3.2 IM vs IC 对比 (OOS段)\n")
    doc.append("| Bin | IM_OOS_MFE/MAE | IM_OOS_PnL | IC_OOS_MFE/MAE | IC_OOS_PnL |")
    doc.append("|-----|---------------|-----------|---------------|-----------|")
    for b in sorted(set(list(im_oos_dict.keys()) + list(oos_dict.keys()))):
        im_r = im_oos_dict.get(b)
        ic_r = oos_dict.get(b)
        im_mm = f"{im_r['mfe_mae']:.2f}" if im_r is not None else "-"
        im_pnl = f"{im_r['avg_pnl']:+.1f}" if im_r is not None else "-"
        ic_mm = f"{ic_r['mfe_mae']:.2f}" if ic_r is not None else "-"
        ic_pnl = f"{ic_r['avg_pnl']:+.1f}" if ic_r is not None else "-"
        doc.append(f"| {b} | {im_mm} | {im_pnl} | {ic_mm} | {ic_pnl} |")
    doc.append("")

    # 3.3 IM高分段子分量（如果OOS也崩塌）
    im_hi_oos = im_oos[im_oos['entry_score'] >= 85]
    im_hi_oos_prof = im_oos_prof[im_oos_prof['score_bin'].astype(str).str.contains('85|90|95')]
    im_hi_collapse = False
    if len(im_hi_oos_prof) > 0:
        avg_mm = im_hi_oos_prof['mfe_mae'].mean()
        if avg_mm < 1.0:
            im_hi_collapse = True

    doc.append("### 3.3 IM高分段OOS状况\n")
    if im_hi_collapse:
        doc.append(f"**IM也有OOS高分段崩塌** (avg MFE/MAE={avg_mm:.2f})")
        doc.append("\nIM [85+] OOS段子分量:\n")
        doc.extend(sub_component_table(im_hi_oos, 'IM_OOS[85+]', im_oos))
    else:
        doc.append(f"**IM OOS高分段正常** — IC的高分段崩塌是IC特异问题")
    doc.append("")

    # ═══════════════════════════════════════════════
    # Step 4: 综合判定
    # ═══════════════════════════════════════════════
    doc.append("## Step 4: 综合判定\n")

    # 检查IC [55,60) 在IM上是否也存在
    im_is_55 = im_is_dict.get('[55,60)')
    im_oos_55 = im_oos_dict.get('[55,60)')
    im_mid_collapse = False
    if im_is_55 is not None and im_oos_55 is not None:
        if im_is_55['mfe_mae'] < 1.0 and im_oos_55['mfe_mae'] < 1.0:
            im_mid_collapse = True

    doc.append("| 现象 | IC | IM | 结论 |")
    doc.append("|------|-----|-----|------|")
    ic_is_mm = is_dict.get('[55,60)', {}).get('mfe_mae', 0)
    ic_oos_mm = oos_dict.get('[55,60)', {}).get('mfe_mae', 0)
    im_is_mm_55 = im_is_55['mfe_mae'] if im_is_55 is not None else 0
    im_oos_mm_55 = im_oos_55['mfe_mae'] if im_oos_55 is not None else 0
    doc.append(f"| [55,60)中分段塌陷 | IS={ic_is_mm:.2f} OOS={ic_oos_mm:.2f} | "
               f"IS={im_is_mm_55:.2f} OOS={im_oos_mm_55:.2f} | "
               f"{'两品种都有' if im_mid_collapse else 'IC特异'} |")
    doc.append(f"| OOS高分段[85+]崩塌 | 是(0.70-0.93) | {'是' if im_hi_collapse else '否'} | "
               f"{'两品种都有' if im_hi_collapse else 'IC特异'} |")

    # 最终判定
    if im_hi_collapse and im_mid_collapse:
        doc.append(f"\n**判定D2: 市场普遍问题，所有品种都受影响**")
        doc.append("IM和IC都有中分段塌陷+OOS高分段崩塌。")
        doc.append("signal_v2评分系统在OOS段整体校准失效。")
        doc.append("\n**建议**: 暂停per-symbol参数优化，重新评估评分系统的稳定性。")
    elif im_hi_collapse and not im_mid_collapse:
        doc.append(f"\n**判定D2(部分): 高分段崩塌是普遍问题**")
        doc.append("两个品种OOS高分段都崩塌，但[55,60)中分段塌陷是IC特异。")
        doc.append("\n**建议**: 高分段问题需要系统级诊断，中分段可做IC专项处理。")
    elif not im_hi_collapse and im_mid_collapse:
        doc.append(f"\n**判定D3: 评分系统在[55,60)有结构性问题**")
        doc.append("两个品种都有中分段塌陷，跟品种无关。")
        doc.append("\n**建议**: 研究评分函数在55-60分段的具体构成。")
    elif not im_hi_collapse and not im_mid_collapse:
        doc.append(f"\n**判定D1: IC特异问题，IM健康**")
        doc.append("IM没有中分段塌陷也没有高分段崩塌。问题集中在IC。")
        doc.append("\n**建议**: 可以继续IM的优化。IC需要专项诊断（可能跟IC市场结构变化有关）。")
    doc.append("")

    # IC 2.5综合解读
    doc.append("### 子分量综合解读\n")
    if 'entry_m' in target_55_60.columns and len(target_55_60) > 0:
        m_55 = target_55_60['entry_m'].mean()
        v_55 = target_55_60['entry_v'].mean()
        m_all = ic_all['entry_m'].mean()
        v_all = ic_all['entry_v'].mean()
        doc.append(f"[55,60)区间: M均值={m_55:.1f}(全样本{m_all:.1f}), V均值={v_55:.1f}(全样本{v_all:.1f})")
        if v_55 < v_all - 3:
            doc.append(f"\n**主要特征: V分偏低** ({v_55:.1f} vs {v_all:.1f})，regime不友好信号集中在此区间")
        elif abs(m_55 - 25) < 5:
            doc.append(f"\n**主要特征: M分踩线** (M均值{m_55:.1f}接近M分中档阈值25)")
        else:
            doc.append(f"\n**无明显单一特征**")

    report = "\n".join(doc)
    path = Path("tmp") / "ic_signal_quality_comprehensive_diagnostic.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
