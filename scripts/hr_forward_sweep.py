#!/usr/bin/env python3
"""简化版横盘反转因子 Forward Periods 扫描。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data.storage.db_manager import get_db
from models.factors.catalog_structure import HorizontalReversalSimple

SPOT = {"IM": "000852", "IC": "000905"}
AMP_THR = 0.4
FWD_PERIODS = [2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 24]
output_dir = Path("tmp")


def load_bars(sym):
    db = get_db()
    df = db.query_df(
        f"SELECT datetime, open, high, low, close, volume FROM index_min "
        f"WHERE symbol='{SPOT[sym]}' AND period=300 "
        f"AND datetime >= '2025-05-16' AND datetime <= '2026-04-09' ORDER BY datetime"
    )
    for c in ['open','high','low','close','volume']:
        df[c] = df[c].astype(float)
    df.index = pd.to_datetime(df['datetime'])
    df['date'] = df.index.date
    return df


def calc_amp(bar_5m):
    daily_amp = {}
    for date, day_bars in bar_5m.groupby('date'):
        if len(day_bars) < 6: continue
        ob = day_bars.iloc[:6]
        daily_amp[date] = (ob['high'].max() - ob['low'].min()) / ob['open'].iloc[0] * 100
    s = bar_5m['date'].map(daily_amp)
    s.index = bar_5m.index
    return s


def sweep_one_symbol(sym, bar_5m, factor_values, amp):
    """对一个品种做完整forward sweep。"""
    results = []
    for n in FWD_PERIODS:
        fwd = bar_5m['close'].pct_change(n).shift(-n) * 10000
        df = pd.DataFrame({'f': factor_values, 'fwd': fwd, 'amp': amp}).dropna()

        for regime, cond, label in [
            ('all', pd.Series(True, index=df.index), '全局'),
            ('high', df['amp'] >= AMP_THR, '高振幅'),
            ('low', df['amp'] < AMP_THR, '低振幅'),
        ]:
            sub = df[cond]
            pos = sub[sub['f'] > 0]
            neg = sub[sub['f'] < 0]

            if len(pos) < 10 or len(neg) < 10:
                results.append({'fwd': n, 'regime': regime, 'label': label,
                                'pos_n': len(pos), 'neg_n': len(neg),
                                'pos_mean': np.nan, 'neg_mean': np.nan,
                                'spread': np.nan, 'rank_ic': np.nan})
                continue

            pos_m = pos['fwd'].mean()
            neg_m = neg['fwd'].mean()
            spread = pos_m - neg_m

            # Rank IC
            from scipy.stats import spearmanr
            signal_bars = sub[sub['f'] != 0]
            if len(signal_bars) >= 30:
                ric, _ = spearmanr(signal_bars['f'], signal_bars['fwd'])
            else:
                ric = np.nan

            results.append({
                'fwd': n, 'regime': regime, 'label': label,
                'pos_n': len(pos), 'neg_n': len(neg),
                'pos_mean': pos_m, 'neg_mean': neg_m,
                'spread': spread, 'rank_ic': ric,
            })

    return pd.DataFrame(results)


def plot_spread_curves(im_df, ic_df, output_dir):
    """画两张收益差曲线图。"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, sym, df in [(axes[0], 'IM', im_df), (axes[1], 'IC', ic_df)]:
        for regime, color, ls in [('all', 'black', '-'), ('high', 'red', '--'), ('low', 'blue', ':')]:
            sub = df[df['regime'] == regime]
            ax.plot(sub['fwd'], sub['spread'], color=color, linestyle=ls,
                    marker='o', markersize=4, label=sub['label'].iloc[0] if len(sub) > 0 else regime)
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.axhline(y=10, color='green', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axhline(y=-10, color='green', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.set_xlabel('Forward Period (bars)')
        ax.set_ylabel('Pos-Neg Spread (bps)')
        ax.set_title(f'{sym} — Horizontal Reversal Spread by Forward Period')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = output_dir / "hr_forward_sweep.png"
    plt.savefig(str(path), dpi=150)
    plt.close()
    return str(path)


def main():
    doc = ["# 简化版横盘反转因子 Forward Periods 扫描\n"]
    doc.append("因子: HorizontalReversalSimple(K=12, N=3)")
    doc.append(f"Forward periods: {FWD_PERIODS}")
    doc.append(f"Regime阈值: 开盘30min振幅 {AMP_THR}%\n")

    all_results = {}
    for sym in ['IM', 'IC']:
        bar_5m = load_bars(sym)
        amp = calc_amp(bar_5m)
        hr = HorizontalReversalSimple(12, 3)
        fv = hr.compute_series(bar_5m)

        results = sweep_one_symbol(sym, bar_5m, fv, amp)
        all_results[sym] = results

        doc.append(f"\n## {sym}\n")
        doc.append(f"{'Fwd':>4s} | {'Regime':>6s} | {'Pos_N':>5s} {'Neg_N':>5s} | {'Pos均值':>8s} {'Neg均值':>8s} | {'差距':>8s} | {'RankIC':>7s}")
        doc.append("-" * 75)
        for _, r in results.iterrows():
            sp = f"{r['spread']:+.1f}" if pd.notna(r['spread']) else "N/A"
            ric = f"{r['rank_ic']:.3f}" if pd.notna(r['rank_ic']) else "N/A"
            pm = f"{r['pos_mean']:+.1f}" if pd.notna(r['pos_mean']) else "N/A"
            nm = f"{r['neg_mean']:+.1f}" if pd.notna(r['neg_mean']) else "N/A"
            mark = " ★" if pd.notna(r['spread']) and abs(r['spread']) >= 10 else ""
            doc.append(f"{int(r['fwd']):>4d} | {r['label']:>6s} | {int(r['pos_n']):>5d} {int(r['neg_n']):>5d} | {pm:>8s} {nm:>8s} | {sp:>8s} | {ric:>7s}{mark}")

    # 图
    fig_path = plot_spread_curves(all_results['IM'], all_results['IC'], output_dir)
    doc.append(f"\n## 曲线图\n![spread curves]({fig_path})\n")

    # 峰值识别
    doc.append("## 峰值识别\n")
    for sym in ['IM', 'IC']:
        df = all_results[sym]
        valid = df.dropna(subset=['spread'])
        if len(valid) == 0:
            continue

        # 找绝对值最大的
        idx_max = valid['spread'].abs().idxmax()
        peak = valid.loc[idx_max]
        doc.append(f"**{sym}** 最强信号: fwd={int(peak['fwd'])}bar {peak['label']} spread={peak['spread']:+.1f}bps")

        # 检查是否稳定（相邻period）
        regime_df = valid[valid['regime'] == peak['regime']]
        peak_fwd = int(peak['fwd'])
        neighbors = regime_df[regime_df['fwd'].isin([peak_fwd-1, peak_fwd, peak_fwd+1, peak_fwd+2, peak_fwd-2])]
        same_sign = (neighbors['spread'] * peak['spread'] > 0).sum()
        doc.append(f"  邻域一致性: {same_sign}/{len(neighbors)}个邻居同向 → {'稳定高原' if same_sign >= 2 else '可能尖峰'}")

        # 10bps达标的组合数
        strong = valid[valid['spread'].abs() >= 10]
        doc.append(f"  |spread|>=10bps的组合: {len(strong)}个")

    # 结论
    doc.append("\n## 结论\n")

    report = "\n".join(doc)
    path = output_dir / "horizontal_reversal_forward_sweep.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
