#!/usr/bin/env python3
"""维度构成分析（修正版）：用归一化分数+isolated_m假设验证[75,80)陷阱成因。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_data(sym):
    return pd.read_csv(f"tmp/dim_analysis_{sym.lower()}/entry_score_profile_data.csv")


# ═══════════════════════════════════════════════════════════════
# 第一步：归一化分类
# ═══════════════════════════════════════════════════════════════

MAX_SCORES = {'entry_m': 50, 'entry_v': 30, 'entry_q': 20, 'entry_b': 20, 'entry_s': 15}
DIM_LABELS = {'entry_m': 'M', 'entry_v': 'V', 'entry_q': 'Q', 'entry_b': 'B', 'entry_s': 'S'}
FOCUS_BINS = ['[65,70)', '[70,75)', '[75,80)', '[80,85)', '[85,90)']


def add_normalized(df):
    for dim, mx in MAX_SCORES.items():
        df[f'{dim}_norm'] = df[dim] / mx
    norm_cols = [f'{d}_norm' for d in MAX_SCORES]
    df['norm_dominant'] = df[norm_cols].idxmax(axis=1).str.replace('_norm', '').map(
        {'entry_m': 'M', 'entry_v': 'V', 'entry_q': 'Q', 'entry_b': 'B', 'entry_s': 'S'})
    # 极端维度计数：>=80%满分
    df['n_extreme'] = sum((df[f'{d}_norm'] >= 0.8).astype(int) for d in MAX_SCORES)
    return df


def step1_normalized_classification(df, sym):
    lines = [f"\n### {sym} 归一化主导维度分布\n"]
    lines.append(f"{'Bin':<12s} {'N':>4s} | {'M%':>5s} {'V%':>5s} {'Q%':>5s} {'B%':>5s} {'S%':>5s} | {'0极端':>5s} {'1极端':>5s} {'2+极端':>5s}")
    lines.append("-" * 78)

    for b in FOCUS_BINS:
        sub = df[df['score_bin'] == b]
        if len(sub) == 0:
            continue
        n = len(sub)
        dom_pcts = sub['norm_dominant'].value_counts(normalize=True) * 100
        ext_counts = sub['n_extreme'].value_counts(normalize=True) * 100

        m_pct = dom_pcts.get('M', 0)
        v_pct = dom_pcts.get('V', 0)
        q_pct = dom_pcts.get('Q', 0)
        b_pct = dom_pcts.get('B', 0)
        s_pct = dom_pcts.get('S', 0)
        e0 = ext_counts.get(0, 0)
        e1 = ext_counts.get(1, 0)
        e2p = sum(ext_counts.get(i, 0) for i in range(2, 6))

        mark = " ◀" if b == '[75,80)' else ""
        lines.append(f"{b:<12s} {n:>4d} | {m_pct:>4.0f}% {v_pct:>4.0f}% {q_pct:>4.0f}% {b_pct:>4.0f}% {s_pct:>4.0f}% | {e0:>4.0f}% {e1:>4.0f}% {e2p:>4.0f}%{mark}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 第二步：isolated_m 假设验证
# ═══════════════════════════════════════════════════════════════

def add_isolated_m(df, m_thr=40, v_thr=25, b_thr=10):
    df['is_isolated_m'] = (df['entry_m'] >= m_thr) & (df['entry_v'] < v_thr) & (df['entry_b'] < b_thr)
    return df


def step2_isolated_m_verification(df, sym):
    lines = [f"\n### {sym} isolated_m 验证 (M>=40, V<25, B<10)\n"]
    lines.append(f"{'Bin':<12s} {'N':>4s} | {'iso_N':>5s} {'iso%':>5s} | {'iso_MFE/MAE':>12s} {'non_MFE/MAE':>12s} {'差距':>6s}")
    lines.append("-" * 70)

    results = []
    for b in FOCUS_BINS:
        sub = df[df['score_bin'] == b]
        if len(sub) == 0:
            continue
        iso = sub[sub['is_isolated_m']]
        non = sub[~sub['is_isolated_m']]
        iso_pct = len(iso) / len(sub) * 100

        iso_ratio = iso['mfe_fixed_24'].median() / iso['mae_fixed_24'].median() if len(iso) >= 5 and iso['mae_fixed_24'].median() > 0 else np.nan
        non_ratio = non['mfe_fixed_24'].median() / non['mae_fixed_24'].median() if len(non) >= 5 and non['mae_fixed_24'].median() > 0 else np.nan
        gap = iso_ratio - non_ratio if pd.notna(iso_ratio) and pd.notna(non_ratio) else np.nan

        iso_str = f"{iso_ratio:.2f}" if pd.notna(iso_ratio) else f"N/A({len(iso)}笔)"
        non_str = f"{non_ratio:.2f}" if pd.notna(non_ratio) else f"N/A({len(non)}笔)"
        gap_str = f"{gap:+.2f}" if pd.notna(gap) else "N/A"
        mark = " ◀" if b == '[75,80)' else ""

        lines.append(f"{b:<12s} {len(sub):>4d} | {len(iso):>5d} {iso_pct:>4.0f}% | {iso_str:>12s} {non_str:>12s} {gap_str:>6s}{mark}")
        results.append({'bin': b, 'n': len(sub), 'iso_n': len(iso), 'iso_pct': iso_pct,
                        'iso_ratio': iso_ratio, 'non_ratio': non_ratio, 'gap': gap})

    return "\n".join(lines), pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# 第三步：跨bin占比分布图
# ═══════════════════════════════════════════════════════════════

def step3_plot(im_results, ic_results, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    bins_x = list(range(len(FOCUS_BINS)))

    if len(im_results) > 0:
        ax.plot(bins_x[:len(im_results)], im_results['iso_pct'].values, 'o-', label='IM', linewidth=2, markersize=8)
    if len(ic_results) > 0:
        ax.plot(bins_x[:len(ic_results)], ic_results['iso_pct'].values, 's-', label='IC', linewidth=2, markersize=8)

    ax.set_xticks(bins_x)
    ax.set_xticklabels(FOCUS_BINS, rotation=30)
    ax.set_ylabel('isolated_m 占比 (%)')
    ax.set_title('isolated_m (M>=40, V<25, B<10) 占比 by Score Bin')
    ax.legend()
    ax.axvline(x=2, color='red', linestyle='--', alpha=0.5, label='[75,80) 陷阱区')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = output_dir / "isolated_m_ratio_by_bin.png"
    plt.savefig(str(path), dpi=150)
    plt.close()
    return str(path)


# ═══════════════════════════════════════════════════════════════
# 第四步：阈值敏感性
# ═══════════════════════════════════════════════════════════════

def step4_sensitivity(df, sym):
    lines = [f"\n### {sym} 阈值敏感性\n"]
    trap = df[df['score_bin'] == '[75,80)']
    if len(trap) == 0:
        return "无数据"

    configs = [
        ("M>=35,V<25,B<10", 35, 25, 10),
        ("M>=40,V<25,B<10", 40, 25, 10),
        ("M>=45,V<25,B<10", 45, 25, 10),
        ("M>=40,V<20,B<10", 40, 20, 10),
        ("M>=40,V<30,B<10", 40, 30, 10),
    ]
    lines.append(f"{'条件':>20s} {'iso_N':>5s} {'iso%':>5s} {'iso_MFE/MAE':>12s} {'non_MFE/MAE':>12s} {'差距':>6s}")
    lines.append("-" * 70)

    for label, m_thr, v_thr, b_thr in configs:
        iso = trap[(trap['entry_m'] >= m_thr) & (trap['entry_v'] < v_thr) & (trap['entry_b'] < b_thr)]
        non = trap[~((trap['entry_m'] >= m_thr) & (trap['entry_v'] < v_thr) & (trap['entry_b'] < b_thr))]
        iso_pct = len(iso) / len(trap) * 100
        iso_r = iso['mfe_fixed_24'].median() / iso['mae_fixed_24'].median() if len(iso) >= 5 and iso['mae_fixed_24'].median() > 0 else np.nan
        non_r = non['mfe_fixed_24'].median() / non['mae_fixed_24'].median() if len(non) >= 5 and non['mae_fixed_24'].median() > 0 else np.nan
        gap = iso_r - non_r if pd.notna(iso_r) and pd.notna(non_r) else np.nan
        iso_s = f"{iso_r:.2f}" if pd.notna(iso_r) else "N/A"
        non_s = f"{non_r:.2f}" if pd.notna(non_r) else "N/A"
        gap_s = f"{gap:+.2f}" if pd.notna(gap) else "N/A"
        mark = " ◀默认" if label == "M>=40,V<25,B<10" else ""
        lines.append(f"{label:>20s} {len(iso):>5d} {iso_pct:>4.0f}% {iso_s:>12s} {non_s:>12s} {gap_s:>6s}{mark}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# V=30 排除法验证
# ═══════════════════════════════════════════════════════════════

def v30_check(df, sym):
    lines = [f"\n### {sym} V=30 观察\n"]
    lines.append(f"{'Bin':<12s} {'V30_N':>5s} {'V30%':>5s} {'V30_MFE/MAE':>12s} {'其他_MFE/MAE':>12s}")
    lines.append("-" * 55)

    for b in FOCUS_BINS:
        sub = df[df['score_bin'] == b]
        if len(sub) == 0:
            continue
        v30 = sub[sub['entry_v'] == 30]
        other = sub[sub['entry_v'] != 30]
        v30_r = v30['mfe_fixed_24'].median() / v30['mae_fixed_24'].median() if len(v30) >= 3 and v30['mae_fixed_24'].median() > 0 else np.nan
        oth_r = other['mfe_fixed_24'].median() / other['mae_fixed_24'].median() if len(other) >= 5 and other['mae_fixed_24'].median() > 0 else np.nan
        v30_s = f"{v30_r:.2f}" if pd.notna(v30_r) else f"N/A({len(v30)})"
        oth_s = f"{oth_r:.2f}" if pd.notna(oth_r) else f"N/A({len(other)})"
        mark = " ◀" if b == '[75,80)' else ""
        lines.append(f"{b:<12s} {len(v30):>5d} {len(v30)/len(sub)*100:>4.0f}% {v30_s:>12s} {oth_s:>12s}{mark}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    output_dir = Path("tmp/dimension_analysis_preview")
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = ["# 维度构成分析（修正版）\n"]
    doc.append("## 方法论修正\n")
    doc.append("- 归一化分数：M/50, V/30, Q/20, B/20, S/15（消除满分天花板差异）")
    doc.append("- 核心假设：isolated_m = (M>=40) AND (V<25) AND (B<10)")
    doc.append("- 理论：纯靠M凑到75-80的信号是追已跑趋势，需要V/B确认才是真突破\n")

    im_results = ic_results = None

    for sym in ['IM', 'IC']:
        df = load_data(sym)
        df = add_normalized(df)
        df = add_isolated_m(df)

        doc.append(f"\n---\n## {sym} ({len(df)}笔)\n")

        # 第一步
        doc.append("### 第一步：归一化分类")
        doc.append(step1_normalized_classification(df, sym))

        # 第二步
        doc.append("\n### 第二步：isolated_m 假设验证")
        text, results = step2_isolated_m_verification(df, sym)
        doc.append(text)
        if sym == 'IM':
            im_results = results
        else:
            ic_results = results

        # 第四步
        doc.append("\n### 第四步：阈值敏感性")
        doc.append(step4_sensitivity(df, sym))

        # V=30
        doc.append(v30_check(df, sym))

    # 第三步：跨bin图
    doc.append("\n---\n## 第三步：isolated_m 占比分布图\n")
    fig_path = step3_plot(im_results, ic_results, output_dir)
    doc.append(f"![isolated_m ratio]({fig_path})\n")

    # 第五步：跨品种一致性
    doc.append("\n## 第五步：跨品种一致性\n")
    if im_results is not None and ic_results is not None:
        im_trap = im_results[im_results['bin'] == '[75,80)']
        ic_trap = ic_results[ic_results['bin'] == '[75,80)']
        if len(im_trap) > 0 and len(ic_trap) > 0:
            im_iso_pct = float(im_trap['iso_pct'].iloc[0])
            ic_iso_pct = float(ic_trap['iso_pct'].iloc[0])
            # 跟邻居比
            im_neighbor_pct = im_results[im_results['bin'].isin(['[70,75)', '[80,85)'])]['iso_pct'].mean()
            ic_neighbor_pct = ic_results[ic_results['bin'].isin(['[70,75)', '[80,85)'])]['iso_pct'].mean()

            doc.append(f"| 指标 | IM | IC |")
            doc.append(f"|------|-----|-----|")
            doc.append(f"| [75,80) iso占比 | {im_iso_pct:.0f}% | {ic_iso_pct:.0f}% |")
            doc.append(f"| 邻居平均iso占比 | {im_neighbor_pct:.0f}% | {ic_neighbor_pct:.0f}% |")
            doc.append(f"| [75,80)是否集中 | {'是 ✓' if im_iso_pct > im_neighbor_pct + 10 else '否 ✗'} | {'是 ✓' if ic_iso_pct > ic_neighbor_pct + 10 else '否 ✗'} |")

            im_gap = im_trap['gap'].iloc[0] if pd.notna(im_trap['gap'].iloc[0]) else 0
            ic_gap = ic_trap['gap'].iloc[0] if pd.notna(ic_trap['gap'].iloc[0]) else 0
            doc.append(f"| iso vs non差距 | {im_gap:+.2f} | {ic_gap:+.2f} |")

    # 结论
    doc.append("\n## 结论\n")

    # 自动判定
    im_concentrated = False
    ic_concentrated = False
    if im_results is not None:
        im_trap_row = im_results[im_results['bin'] == '[75,80)']
        im_neighbor = im_results[im_results['bin'].isin(['[70,75)', '[80,85)'])]
        if len(im_trap_row) > 0 and len(im_neighbor) > 0:
            im_concentrated = float(im_trap_row['iso_pct'].iloc[0]) > im_neighbor['iso_pct'].mean() + 10
    if ic_results is not None:
        ic_trap_row = ic_results[ic_results['bin'] == '[75,80)']
        ic_neighbor = ic_results[ic_results['bin'].isin(['[70,75)', '[80,85)'])]
        if len(ic_trap_row) > 0 and len(ic_neighbor) > 0:
            ic_concentrated = float(ic_trap_row['iso_pct'].iloc[0]) > ic_neighbor['iso_pct'].mean() + 10

    both = im_concentrated and ic_concentrated
    one = im_concentrated or ic_concentrated

    if both:
        doc.append("**假设高度成立**：两品种都在[75,80)陷阱区集中了isolated_m信号")
        doc.append("→ 全量验证加入isolated_m规则稳定性测试")
    elif one:
        sym_ok = 'IM' if im_concentrated else 'IC'
        doc.append(f"**假设部分成立**：{sym_ok}在[75,80)集中了isolated_m，另一品种不明显")
        doc.append("→ 维度分析降级为候选观察，全量验证只测threshold")
    else:
        doc.append("**假设不成立**：isolated_m占比在陷阱区未显著集中")
        doc.append("→ 维度分析路径可能不适合解释陷阱现象，考虑其他角度")

    report = "\n".join(doc)
    path = output_dir / "dimension_analysis_revised.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)
    print(f"\n报告已保存: {path}")


if __name__ == "__main__":
    main()
