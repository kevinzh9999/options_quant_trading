#!/usr/bin/env python3
"""分数维度构成分析：定位[75,80)陷阱区的维度成因。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_data(sym):
    path = f"tmp/dim_analysis_{sym.lower()}/entry_score_profile_data.csv"
    df = pd.read_csv(path)
    return df


def layer1_descriptive(df, sym):
    """第一层：描述性统计。"""
    dims = ['entry_m', 'entry_v', 'entry_q', 'entry_b', 'entry_s']
    bins = ['[60,65)', '[65,70)', '[70,75)', '[75,80)', '[80,85)', '[85,90)']
    if sym == 'IM':
        bins = ['[55,60)'] + bins

    lines = [f"\n### {sym} 维度构成统计\n"]
    lines.append(f"{'Bin':<12s} {'N':>4s} | {'M_avg':>6s} {'V_avg':>6s} {'Q_avg':>6s} {'B_avg':>6s} {'S_avg':>6s} | {'M%':>5s} {'V%':>5s} {'Q%':>5s} {'MFE/MAE':>8s}")
    lines.append("-" * 85)

    rows = []
    for b in bins:
        sub = df[df['score_bin'] == b]
        if len(sub) == 0:
            continue
        raw = sub['entry_raw'].mean()
        r = {'bin': b, 'n': len(sub)}
        for d in dims:
            r[f'{d}_avg'] = sub[d].mean()
            r[f'{d}_med'] = sub[d].median()
            r[f'{d}_std'] = sub[d].std()
            r[f'{d}_pct'] = sub[d].mean() / raw * 100 if raw > 0 else 0
        r['mfe_mae'] = sub['mfe_fixed_24'].median() / sub['mae_fixed_24'].median() if sub['mae_fixed_24'].median() > 0 else 0
        rows.append(r)

        m_pct = f"{r['entry_m_pct']:.0f}%"
        v_pct = f"{r['entry_v_pct']:.0f}%"
        q_pct = f"{r['entry_q_pct']:.0f}%"
        mark = " ◀陷阱" if b == '[75,80)' else ""
        lines.append(
            f"{b:<12s} {r['n']:>4d} | {r['entry_m_avg']:>6.1f} {r['entry_v_avg']:>6.1f} "
            f"{r['entry_q_avg']:>6.1f} {r['entry_b_avg']:>6.1f} {r['entry_s_avg']:>6.1f} | "
            f"{m_pct:>5s} {v_pct:>5s} {q_pct:>5s} {r['mfe_mae']:>8.2f}{mark}"
        )

    return "\n".join(lines), pd.DataFrame(rows)


def layer2_comparison(df, sym, output_dir):
    """第二层：[70,75) vs [75,80) vs [80,85) 维度对比图。"""
    dims = ['entry_m', 'entry_v', 'entry_q', 'entry_b', 'entry_s']
    dim_names = ['M (Momentum)', 'V (Volatility)', 'Q (Volume)', 'B (Breakout)', 'S (Startup)']
    target_bins = ['[70,75)', '[75,80)', '[80,85)']
    colors = ['#2ecc71', '#e74c3c', '#3498db']

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(f'{sym} — Dimension Distribution: [70,75) vs [75,80) vs [80,85)', fontsize=13)

    observations = []
    for i, (dim, name) in enumerate(zip(dims, dim_names)):
        ax = axes[i]
        data_by_bin = []
        for b in target_bins:
            sub = df[df['score_bin'] == b]
            data_by_bin.append(sub[dim].values)

        bp = ax.boxplot(data_by_bin, labels=['70-75', '75-80', '80-85'],
                        patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(name, fontsize=10)
        ax.set_ylabel('Score')

        # 记录观察
        avgs = [np.mean(d) for d in data_by_bin]
        stds = [np.std(d) for d in data_by_bin]
        if len(data_by_bin[1]) > 0:
            trap_avg = avgs[1]
            neighbor_avg = (avgs[0] + avgs[2]) / 2
            diff = trap_avg - neighbor_avg
            if abs(diff) > 3:
                observations.append(f"  - {name}: 陷阱区均值={trap_avg:.1f} vs 邻居均值={neighbor_avg:.1f} (差{diff:+.1f})")

    plt.tight_layout()
    path = output_dir / f"dim_comparison_{sym.lower()}.png"
    plt.savefig(str(path), dpi=150)
    plt.close()

    return observations, str(path)


def layer3_subclass(df, sym):
    """第三层：[75,80)内部子类分解。"""
    trap = df[df['score_bin'] == '[75,80)'].copy()
    if len(trap) == 0:
        return "无数据", {}

    dims = ['entry_m', 'entry_v', 'entry_q', 'entry_b', 'entry_s']
    dim_labels = {'entry_m': 'M', 'entry_v': 'V', 'entry_q': 'Q', 'entry_b': 'B', 'entry_s': 'S'}

    results = {}
    lines = [f"\n### {sym} [75,80) 子类分解 (N={len(trap)})\n"]

    # === 分类A：按主导维度 ===
    trap['dominant'] = trap[dims].idxmax(axis=1).map(dim_labels)
    lines.append("**分类A：按主导维度**")
    lines.append(f"{'主导':>6s} {'N':>4s} {'Med_MFE':>9s} {'Med_MAE':>9s} {'MFE/MAE':>8s} {'WR':>6s}")

    best_a = worst_a = None
    for dom in sorted(trap['dominant'].unique()):
        sub = trap[trap['dominant'] == dom]
        if len(sub) < 5:
            continue
        mfe = sub['mfe_fixed_24'].median()
        mae = sub['mae_fixed_24'].median()
        ratio = mfe / mae if mae > 0 else 0
        wr = (sub['actual_win'] > 0).mean() * 100
        lines.append(f"{dom:>6s} {len(sub):>4d} {mfe:>9.1f} {mae:>9.1f} {ratio:>8.2f} {wr:>5.1f}%")
        if best_a is None or ratio > best_a[1]:
            best_a = (dom, ratio, len(sub))
        if worst_a is None or ratio < worst_a[1]:
            worst_a = (dom, ratio, len(sub))

    if best_a and worst_a and best_a[0] != worst_a[0]:
        gap_a = best_a[1] - worst_a[1]
        lines.append(f"  → 最佳={best_a[0]}({best_a[1]:.2f}), 最差={worst_a[0]}({worst_a[1]:.2f}), 差距={gap_a:.2f}")
        results['A'] = {'best': best_a, 'worst': worst_a, 'gap': gap_a}

    # === 分类B：按集中度 ===
    def concentration(row):
        vals = [row[d] for d in dims if row[d] > 0]
        if not vals or sum(vals) == 0:
            return 0
        return max(vals) / sum(vals)

    trap['concentration'] = trap.apply(concentration, axis=1)
    median_conc = trap['concentration'].median()
    trap['conc_type'] = trap['concentration'].apply(lambda x: 'concentrated' if x > median_conc else 'distributed')

    lines.append("\n**分类B：按集中度（主导维度占比 vs 中位数切分）**")
    lines.append(f"{'类型':>14s} {'N':>4s} {'Med_MFE':>9s} {'Med_MAE':>9s} {'MFE/MAE':>8s} {'WR':>6s}")

    best_b = worst_b = None
    for ct in ['concentrated', 'distributed']:
        sub = trap[trap['conc_type'] == ct]
        if len(sub) < 5:
            continue
        mfe = sub['mfe_fixed_24'].median()
        mae = sub['mae_fixed_24'].median()
        ratio = mfe / mae if mae > 0 else 0
        wr = (sub['actual_win'] > 0).mean() * 100
        lines.append(f"{ct:>14s} {len(sub):>4d} {mfe:>9.1f} {mae:>9.1f} {ratio:>8.2f} {wr:>5.1f}%")
        if best_b is None or ratio > best_b[1]:
            best_b = (ct, ratio, len(sub))
        if worst_b is None or ratio < worst_b[1]:
            worst_b = (ct, ratio, len(sub))

    if best_b and worst_b and best_b[0] != worst_b[0]:
        gap_b = best_b[1] - worst_b[1]
        lines.append(f"  → 最佳={best_b[0]}({best_b[1]:.2f}), 最差={worst_b[0]}({worst_b[1]:.2f}), 差距={gap_b:.2f}")
        results['B'] = {'best': best_b, 'worst': worst_b, 'gap': gap_b}

    # === 分类C：按是否包含特定维度 ===
    lines.append("\n**分类C：按维度达标二元分类**")
    lines.append(f"{'条件':>16s} {'Y_N':>6s} {'Y_MFE/MAE':>10s} {'N_MFE/MAE':>10s} {'差距':>6s}")

    criteria = [
        ('M>=35', lambda r: r['entry_m'] >= 35),
        ('M>=50', lambda r: r['entry_m'] >= 50),
        ('V>=25', lambda r: r['entry_v'] >= 25),
        ('V=30', lambda r: r['entry_v'] == 30),
        ('Q>=10', lambda r: r['entry_q'] >= 10),
        ('Q=20', lambda r: r['entry_q'] == 20),
        ('B>0', lambda r: r['entry_b'] > 0),
        ('S>0', lambda r: r['entry_s'] > 0),
    ]

    best_c = None
    for label, crit in criteria:
        y = trap[trap.apply(crit, axis=1)]
        n = trap[~trap.apply(crit, axis=1)]
        if len(y) < 5 or len(n) < 5:
            continue
        y_mfe = y['mfe_fixed_24'].median()
        y_mae = y['mae_fixed_24'].median()
        n_mfe = n['mfe_fixed_24'].median()
        n_mae = n['mae_fixed_24'].median()
        y_ratio = y_mfe / y_mae if y_mae > 0 else 0
        n_ratio = n_mfe / n_mae if n_mae > 0 else 0
        gap = y_ratio - n_ratio
        lines.append(f"{label:>16s} {len(y):>2d}/{len(n):<3d} {y_ratio:>10.2f} {n_ratio:>10.2f} {gap:>+6.2f}")
        if best_c is None or abs(gap) > abs(best_c[2]):
            best_c = (label, y_ratio, gap, n_ratio, len(y), len(n))

    if best_c:
        lines.append(f"  → 最强区分: {best_c[0]} (Y={best_c[1]:.2f} vs N={best_c[3]:.2f}, 差距={best_c[2]:+.2f})")
        results['C'] = {'criterion': best_c[0], 'y_ratio': best_c[1], 'n_ratio': best_c[3],
                        'gap': best_c[2], 'y_n': f"{best_c[4]}/{best_c[5]}"}

    return "\n".join(lines), results


def main():
    output_dir = Path("tmp/dimension_analysis_preview")
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = ["# 分数维度构成分析（219天预览）\n"]

    all_results = {}

    for sym in ['IM', 'IC']:
        df = load_data(sym)
        doc.append(f"\n## {sym} ({len(df)}笔)\n")

        # 第一层
        desc_text, desc_df = layer1_descriptive(df, sym)
        doc.append("### 第一层：描述性统计")
        doc.append(desc_text)

        # 第二层
        obs, fig_path = layer2_comparison(df, sym, output_dir)
        doc.append(f"\n### 第二层：[70-75) vs [75-80) vs [80-85) 对比")
        doc.append(f"图: {fig_path}")
        if obs:
            doc.append("维度差异观察：")
            doc.extend(obs)
        else:
            doc.append("未发现显著维度差异")

        # 第三层
        sub_text, sub_results = layer3_subclass(df, sym)
        doc.append(f"\n### 第三层：[75,80)内部子类分解")
        doc.append(sub_text)
        all_results[sym] = sub_results

    # 第四层：跨品种对比
    doc.append("\n## 跨品种一致性检查\n")

    for method in ['A', 'B', 'C']:
        im_r = all_results.get('IM', {}).get(method)
        ic_r = all_results.get('IC', {}).get(method)
        if im_r and ic_r:
            if method == 'A':
                im_worst = im_r['worst'][0]
                ic_worst = ic_r['worst'][0]
                consistent = im_worst == ic_worst
                doc.append(f"**分类{method}（主导维度）**: IM最差={im_worst} IC最差={ic_worst} → {'一致 ✓' if consistent else '不一致 ✗'}")
            elif method == 'B':
                im_worst = im_r['worst'][0]
                ic_worst = ic_r['worst'][0]
                consistent = im_worst == ic_worst
                doc.append(f"**分类{method}（集中度）**: IM最差={im_worst} IC最差={ic_worst} → {'一致 ✓' if consistent else '不一致 ✗'}")
            elif method == 'C':
                im_crit = im_r['criterion']
                ic_crit = ic_r['criterion']
                consistent = im_crit == ic_crit
                doc.append(f"**分类{method}（达标二元）**: IM最强={im_crit}(gap={im_r['gap']:+.2f}) IC最强={ic_crit}(gap={ic_r['gap']:+.2f}) → {'一致 ✓' if consistent else '不一致 ✗'}")

    # 结论
    doc.append("\n## 结论\n")

    # 自动判断最有解释力的分类
    best_method = None
    best_gap = 0
    for sym in ['IM', 'IC']:
        for method in ['A', 'B', 'C']:
            r = all_results.get(sym, {}).get(method)
            if r and abs(r.get('gap', 0)) > best_gap:
                best_gap = abs(r.get('gap', 0))
                best_method = method

    if best_gap >= 0.4:
        doc.append(f"- **有效信号**: 分类{best_method}的MFE/MAE差距={best_gap:.2f}（>0.4阈值），值得在全量数据上验证")
        doc.append(f"- 建议在900/2000天数据上验证此发现的稳定性")
    elif best_gap >= 0.2:
        doc.append(f"- **弱信号**: 分类{best_method}的MFE/MAE差距={best_gap:.2f}（0.2-0.4），方向有指示性但不够强")
        doc.append(f"- 建议在全量数据上做初步检查，但不急于上线")
    else:
        doc.append(f"- **无明显信号**: 最大MFE/MAE差距={best_gap:.2f}（<0.2），维度构成可能不是陷阱的主因")
        doc.append(f"- 建议换角度调查（时段、regime、方向等）")

    # 写文件
    report = "\n".join(doc)
    report_path = output_dir / "dimension_analysis_preview.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(report)
    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    main()
