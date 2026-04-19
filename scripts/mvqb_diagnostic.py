#!/usr/bin/env python3
"""MVQB评分系统诊断 - 6任务一次性完成。"""
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


def collect_trades(sym, dates):
    thr = SYMBOL_PROFILES[sym].get("signal_threshold", 60)
    n_workers = min(cpu_count(), 8)
    args = [(td, sym, thr) for td in dates]
    with Pool(n_workers) as pool:
        day_results = pool.map(_run_one_day, args)
    return [t for day in day_results for t in day]


def bucket_analysis(tdf, col, n_buckets=12, label="", doc=None):
    """对连续值分桶做PnL分析。"""
    valid = tdf[tdf[col].notna() & (tdf[col] != 0) & (tdf[col] != -1.0)].copy()
    if len(valid) < 100:
        doc.append(f"  (有效样本{len(valid)}<100，跳过)\n")
        return

    try:
        valid['bucket'] = pd.qcut(valid[col], n_buckets, duplicates='drop')
    except ValueError:
        valid['bucket'] = pd.cut(valid[col], n_buckets, duplicates='drop')

    doc.append(f"| 桶 | 数值范围 | 笔数 | AvgPnL | 胜率 | 累计PnL |")
    doc.append(f"|---|---------|------|--------|------|---------|")
    for i, (bucket, group) in enumerate(valid.groupby('bucket', observed=True)):
        if len(group) < 10:
            continue
        wr = (group['pnl_pts'] > 0).sum() / len(group) * 100
        doc.append(f"| {i+1} | {bucket} | {len(group)} | {group['pnl_pts'].mean():+.1f} | {wr:.0f}% | {group['pnl_pts'].sum():+.0f} |")
    doc.append("")


def score_bin_table(tdf, score_col, label, doc):
    """按分数段(每5分)做PnL分析。"""
    bins = list(range(55, 101, 5))
    bins.append(101)
    labels_list = [f'[{bins[i]},{bins[i+1]})' for i in range(len(bins)-1)]
    tdf = tdf.copy()
    tdf['sbin'] = pd.cut(tdf[score_col], bins=bins, labels=labels_list, right=False)
    total_pnl = tdf['pnl_pts'].sum()

    doc.append(f"| Score段 | 笔数 | 占比 | AvgPnL | 累计PnL | PnL贡献% | 胜率 |")
    doc.append(f"|---------|------|------|--------|---------|---------|------|")
    for label_s in labels_list:
        sub = tdf[tdf['sbin'] == label_s]
        if len(sub) == 0:
            continue
        contrib = sub['pnl_pts'].sum() / total_pnl * 100 if total_pnl != 0 else 0
        wr = (sub['pnl_pts'] > 0).sum() / len(sub) * 100
        doc.append(f"| {label_s} | {len(sub)} | {len(sub)/len(tdf)*100:.0f}% | "
                   f"{sub['pnl_pts'].mean():+.1f} | {sub['pnl_pts'].sum():+.0f} | {contrib:+.0f}% | {wr:.0f}% |")
    doc.append("")


def boundary_analysis(tdf, raw_col, score_col, boundaries, window, sym, doc):
    """分档边界跳变分析。"""
    valid = tdf[tdf[raw_col].notna() & (tdf[raw_col] != 0)].copy()
    if len(valid) < 100:
        doc.append(f"  (有效样本不足，跳过)\n")
        return

    for boundary_val, left_score, right_score in boundaries:
        left = valid[(valid[raw_col] >= boundary_val - window) & (valid[raw_col] < boundary_val)]
        right = valid[(valid[raw_col] >= boundary_val) & (valid[raw_col] < boundary_val + window)]
        if len(left) < 20 or len(right) < 20:
            doc.append(f"**边界 {boundary_val} ({left_score}→{right_score})**: 样本不足(左{len(left)}/右{len(right)})\n")
            continue
        l_avg = left['pnl_pts'].mean()
        r_avg = right['pnl_pts'].mean()
        gap = r_avg - l_avg
        real = "**真实**" if abs(gap) > 5 else ("边缘" if abs(gap) > 2 else "不真实")
        doc.append(f"**边界 {boundary_val} ({left_score}→{right_score})**: "
                   f"左[{boundary_val-window:.4f},{boundary_val:.4f}) {len(left)}笔 avg={l_avg:+.1f} | "
                   f"右[{boundary_val:.4f},{boundary_val+window:.4f}) {len(right)}笔 avg={r_avg:+.1f} | "
                   f"差距={gap:+.1f} → {real}\n")


def interaction_matrix(tdf, col1, col2, bins1, bins2, labels1, labels2, doc):
    """两个子分量的2D交互矩阵。"""
    valid = tdf.copy()
    valid['g1'] = pd.cut(valid[col1], bins=bins1, labels=labels1, right=False)
    valid['g2'] = pd.cut(valid[col2], bins=bins2, labels=labels2, right=False)

    # PnL矩阵
    doc.append("**平均PnL矩阵:**\n")
    header = f"| {col2}→ |"
    for l2 in labels2:
        header += f" {l2} |"
    doc.append(header)
    doc.append("|---" + "|---" * len(labels2) + "|")

    for l1 in labels1:
        row = f"| {l1} |"
        for l2 in labels2:
            cell = valid[(valid['g1'] == l1) & (valid['g2'] == l2)]
            if len(cell) >= 30:
                row += f" {cell['pnl_pts'].mean():+.1f}({len(cell)}) |"
            elif len(cell) > 0:
                row += f" ({len(cell)}) |"
            else:
                row += " - |"
        doc.append(row)

    # 笔数矩阵
    doc.append("\n**笔数矩阵:**\n")
    header = f"| {col2}→ |"
    for l2 in labels2:
        header += f" {l2} |"
    doc.append(header)
    doc.append("|---" + "|---" * len(labels2) + "|")

    for l1 in labels1:
        row = f"| {l1} |"
        for l2 in labels2:
            cell = valid[(valid['g1'] == l1) & (valid['g2'] == l2)]
            row += f" {len(cell)} |"
        doc.append(row)
    doc.append("")


def main():
    print("=" * 60)
    print("  MVQB 评分系统完整诊断")
    print("=" * 60)

    db = get_db()
    doc = ["# MVQB 评分系统诊断报告\n"]

    all_data = {}
    for sym in ['IM', 'IC']:
        print(f"\n[{sym}] 收集交易数据...")
        dates = get_dates(db, SPOTS[sym])
        trades = collect_trades(sym, dates)
        tdf = pd.DataFrame(trades)
        all_data[sym] = tdf
        print(f"  {len(tdf)}笔")
        doc.append(f"- {sym}: {len(dates)}天, {len(tdf)}笔")

    doc.append("\n## 任务1: 脚本扩展说明\n")
    doc.append("原始连续数值定义:")
    doc.append("- raw_mom_5m: 5分钟动量百分比(连续值,如0.0024=0.24%)")
    doc.append("- raw_atr_ratio: ATR(5)/ATR(40)比值(连续值,如0.85)")
    doc.append("- raw_vol_pct: 成交量百分位(0-1,percentile法)或raw_vol_ratio(ratio法)")
    doc.append("- B分和S分是离散状态(0或有值),无连续原始值\n")

    # ═══════════════════════════════════════════════
    # 任务2: 单子分量原始值 vs PnL
    # ═══════════════════════════════════════════════
    doc.append("## 任务2: 单子分量原始值 vs PnL\n")

    for sym in ['IM', 'IC']:
        tdf = all_data[sym]

        # M分
        doc.append(f"### M分原始值 vs PnL ({sym})\n")
        tdf['abs_raw_mom'] = tdf['raw_mom_5m'].abs()
        bucket_analysis(tdf, 'abs_raw_mom', n_buckets=12, label=f'M_{sym}', doc=doc)

        # 跟当前分档对应
        if 'abs_raw_mom' in tdf.columns:
            for lo, hi, score in [(0, 0.001, 0), (0.001, 0.002, 15), (0.002, 0.003, 25), (0.003, 1.0, 35)]:
                sub = tdf[(tdf['abs_raw_mom'] >= lo) & (tdf['abs_raw_mom'] < hi)]
                if len(sub) >= 30:
                    wr = (sub['pnl_pts'] > 0).sum() / len(sub) * 100
                    doc.append(f"当前分档[{lo:.3f},{hi:.3f})→M={score}: {len(sub)}笔, avg={sub['pnl_pts'].mean():+.1f}, WR={wr:.0f}%")
            doc.append("")

        # V分
        doc.append(f"### V分原始值(ATR ratio) vs PnL ({sym})\n")
        bucket_analysis(tdf, 'raw_atr_ratio', n_buckets=12, label=f'V_{sym}', doc=doc)

        for lo, hi, score in [(0, 0.7, 30), (0.7, 0.9, 25), (0.9, 1.1, 15), (1.1, 1.5, 5), (1.5, 99, 0)]:
            sub = tdf[(tdf['raw_atr_ratio'] >= lo) & (tdf['raw_atr_ratio'] < hi)]
            if len(sub) >= 30:
                wr = (sub['pnl_pts'] > 0).sum() / len(sub) * 100
                doc.append(f"当前分档[{lo:.1f},{hi:.1f})→V={score}: {len(sub)}笔, avg={sub['pnl_pts'].mean():+.1f}, WR={wr:.0f}%")
        doc.append("")

        # Q分
        doc.append(f"### Q分原始值 vs PnL ({sym})\n")
        # 用vol_pct（如果有）或vol_ratio
        q_col = 'raw_vol_pct' if tdf['raw_vol_pct'].notna().sum() > tdf['raw_vol_ratio'].apply(lambda x: x > 0).sum() else 'raw_vol_ratio'
        valid_q = tdf[(tdf[q_col].notna()) & (tdf[q_col] > -0.5)]
        if len(valid_q) >= 100:
            bucket_analysis(valid_q, q_col, n_buckets=10, label=f'Q_{sym}', doc=doc)
        else:
            doc.append(f"  Q分有效样本不足({len(valid_q)}), 跳过\n")

        # B分 (离散)
        doc.append(f"### B分 vs PnL ({sym}, 离散)\n")
        doc.append("| B分 | 笔数 | AvgPnL | 胜率 |")
        doc.append("|-----|------|--------|------|")
        for b_val in sorted(tdf['entry_b_score'].unique()):
            sub = tdf[tdf['entry_b_score'] == b_val]
            if len(sub) >= 30:
                wr = (sub['pnl_pts'] > 0).sum() / len(sub) * 100
                doc.append(f"| {int(b_val)} | {len(sub)} | {sub['pnl_pts'].mean():+.1f} | {wr:.0f}% |")
        doc.append("")

        # S分 (离散)
        doc.append(f"### S分 vs PnL ({sym}, 离散)\n")
        doc.append("| S分 | 笔数 | AvgPnL | 胜率 |")
        doc.append("|-----|------|--------|------|")
        for s_val in sorted(tdf['entry_s_score'].unique()):
            sub = tdf[tdf['entry_s_score'] == s_val]
            if len(sub) >= 30:
                wr = (sub['pnl_pts'] > 0).sum() / len(sub) * 100
                doc.append(f"| {int(s_val)} | {len(sub)} | {sub['pnl_pts'].mean():+.1f} | {wr:.0f}% |")
        doc.append("")

    # ═══════════════════════════════════════════════
    # 任务3: 分档边界跳变分析
    # ═══════════════════════════════════════════════
    doc.append("## 任务3: 分档边界跳变分析\n")

    for sym in ['IM', 'IC']:
        tdf = all_data[sym]
        tdf['abs_raw_mom'] = tdf['raw_mom_5m'].abs()

        doc.append(f"### M分边界 ({sym})\n")
        boundary_analysis(tdf, 'abs_raw_mom', 'entry_m_score',
                          [(0.001, 0, 15), (0.002, 15, 25), (0.003, 25, 35)],
                          0.0005, sym, doc)

        doc.append(f"### V分边界 ({sym})\n")
        boundary_analysis(tdf, 'raw_atr_ratio', 'entry_v_score',
                          [(0.7, 30, 25), (0.9, 25, 15), (1.1, 15, 5), (1.5, 5, 0)],
                          0.1, sym, doc)

    # ═══════════════════════════════════════════════
    # 任务4: 子分量交互二维矩阵
    # ═══════════════════════════════════════════════
    doc.append("## 任务4: 子分量交互二维矩阵\n")

    m_bins = [0, 15, 25, 35, 51]
    m_labels = ['M<15', 'M[15,25)', 'M[25,35)', 'M[35,50]']
    v_bins = [0, 10, 15, 20, 31]
    v_labels = ['V<10', 'V[10,15)', 'V[15,20)', 'V[20,30]']
    q_bins = [0, 10, 21]
    q_labels = ['Q<10', 'Q[10,20]']

    for sym in ['IM', 'IC']:
        tdf = all_data[sym]

        doc.append(f"### M × V ({sym})\n")
        interaction_matrix(tdf, 'entry_m_score', 'entry_v_score',
                          m_bins, v_bins, m_labels, v_labels, doc)

        doc.append(f"### M × Q ({sym})\n")
        interaction_matrix(tdf, 'entry_m_score', 'entry_q_score',
                          m_bins, q_bins, m_labels, q_labels, doc)

        doc.append(f"### V × Q ({sym})\n")
        interaction_matrix(tdf, 'entry_v_score', 'entry_q_score',
                          v_bins, q_bins, v_labels, q_labels, doc)

    # ═══════════════════════════════════════════════
    # 任务5: Score真实曲线
    # ═══════════════════════════════════════════════
    doc.append("## 任务5: Score真实曲线\n")

    for sym in ['IM', 'IC']:
        tdf = all_data[sym]
        doc.append(f"### {sym}\n")
        score_bin_table(tdf, 'entry_score', sym, doc)

    # ═══════════════════════════════════════════════
    # 任务6: 未利用特征探索
    # ═══════════════════════════════════════════════
    doc.append("## 任务6: 未利用特征探索\n")

    for sym in ['IM', 'IC']:
        tdf = all_data[sym]

        # 特征1: 入场时段
        doc.append(f"### 入场时段 ({sym})\n")
        if 'entry_time' in tdf.columns:
            def get_session(t):
                try:
                    h = int(t[:2])
                    if h < 12: return f"{h:02d}:00-{h+1:02d}:00"
                    else: return f"{h:02d}:00-{h+1:02d}:00"
                except: return 'unknown'
            tdf['session'] = tdf['entry_time'].apply(get_session)
            doc.append("| 时段 | 笔数 | AvgPnL | 胜率 |")
            doc.append("|------|------|--------|------|")
            for s in sorted(tdf['session'].unique()):
                sub = tdf[tdf['session'] == s]
                if len(sub) >= 50:
                    wr = (sub['pnl_pts'] > 0).sum() / len(sub) * 100
                    doc.append(f"| {s} | {len(sub)} | {sub['pnl_pts'].mean():+.1f} | {wr:.0f}% |")
            doc.append("")

        # 特征2: 入场价格相对位置 (用entry_rebound_pct代理)
        if 'entry_rebound_pct' in tdf.columns:
            doc.append(f"### 入场反弹/回撤幅度 ({sym})\n")
            bucket_analysis(tdf, 'entry_rebound_pct', n_buckets=8, label=f'rebound_{sym}', doc=doc)

        # 特征3: gap对齐
        if 'entry_gap_aligned' in tdf.columns:
            doc.append(f"### Gap对齐 ({sym})\n")
            doc.append("| Gap对齐 | 笔数 | AvgPnL | 胜率 |")
            doc.append("|---------|------|--------|------|")
            for aligned in [True, False]:
                sub = tdf[tdf['entry_gap_aligned'] == aligned]
                if len(sub) >= 50:
                    wr = (sub['pnl_pts'] > 0).sum() / len(sub) * 100
                    doc.append(f"| {aligned} | {len(sub)} | {sub['pnl_pts'].mean():+.1f} | {wr:.0f}% |")
            doc.append("")

        # 特征4: gap大小
        if 'entry_gap_pct' in tdf.columns:
            doc.append(f"### Gap大小 ({sym})\n")
            bucket_analysis(tdf, 'entry_gap_pct', n_buckets=8, label=f'gap_{sym}', doc=doc)

        # 特征5: 方向
        if 'direction' in tdf.columns:
            doc.append(f"### 多空方向 ({sym})\n")
            doc.append("| 方向 | 笔数 | AvgPnL | 胜率 |")
            doc.append("|------|------|--------|------|")
            for d in ['LONG', 'SHORT']:
                sub = tdf[tdf['direction'] == d]
                if len(sub) >= 50:
                    wr = (sub['pnl_pts'] > 0).sum() / len(sub) * 100
                    doc.append(f"| {d} | {len(sub)} | {sub['pnl_pts'].mean():+.1f} | {wr:.0f}% |")
            doc.append("")

    # ═══════════════════════════════════════════════
    # 整体诊断结论
    # ═══════════════════════════════════════════════
    doc.append("## 整体诊断结论\n")
    doc.append("(根据以上6个任务的数据，回答5组核心问题)")

    report = "\n".join(doc)
    path = Path("tmp") / "mvqb_diagnostic_report.md"
    with open(path, 'w') as f:
        f.write(report)
    print(f"\n报告已保存: {path}")
    print(f"报告长度: {len(doc)}行")


if __name__ == "__main__":
    main()
