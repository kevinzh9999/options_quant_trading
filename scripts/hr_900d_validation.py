#!/usr/bin/env python3
"""横盘反转因子900天Forward Sweep验证。"""
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

AMP_THR = 0.4
FWD_PERIODS = [2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 24]


def load_im_bars(start=None, end=None):
    db = get_db()
    df = db.query_df(
        "SELECT datetime, open, high, low, close, volume FROM index_min "
        "WHERE symbol='000852' AND period=300 ORDER BY datetime"
    )
    for c in ['open','high','low','close','volume']:
        df[c] = df[c].astype(float)
    df.index = pd.to_datetime(df['datetime'])
    df['date'] = df.index.date
    df['date_str'] = df.index.strftime('%Y%m%d')
    if start: df = df[df['date_str'] >= start]
    if end: df = df[df['date_str'] <= end]
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


def sweep(bar_5m, factor_values, amp):
    results = []
    for n in FWD_PERIODS:
        fwd = bar_5m['close'].pct_change(n).shift(-n) * 10000
        df = pd.DataFrame({'f': factor_values, 'fwd': fwd, 'amp': amp}).dropna()

        for regime, cond, label in [
            ('all', pd.Series(True, index=df.index), 'all'),
            ('high', df['amp'] >= AMP_THR, 'high_amp'),
            ('low', df['amp'] < AMP_THR, 'low_amp'),
        ]:
            sub = df[cond]
            pos = sub[sub['f'] > 0]
            neg = sub[sub['f'] < 0]
            if len(pos) < 10 or len(neg) < 10:
                results.append({'fwd': n, 'regime': label, 'pos_n': len(pos), 'neg_n': len(neg),
                                'pos_mean': np.nan, 'neg_mean': np.nan, 'spread': np.nan})
                continue
            pm, nm = pos['fwd'].mean(), neg['fwd'].mean()
            results.append({'fwd': n, 'regime': label, 'pos_n': len(pos), 'neg_n': len(neg),
                            'pos_mean': pm, 'neg_mean': nm, 'spread': pm - nm})
    return pd.DataFrame(results)


def main():
    output_dir = Path("tmp")
    doc = ["# 横盘反转因子 900天 Forward Sweep 验证\n"]

    # 900天
    print("加载IM 900天数据...")
    bar_900 = load_im_bars()
    amp_900 = calc_amp(bar_900)
    hr = HorizontalReversalSimple(12, 3)
    fv_900 = hr.compute_series(bar_900)

    n_total = len(bar_900)
    n_high = (amp_900 >= AMP_THR).sum()
    n_low = (amp_900 < AMP_THR).sum()
    n_sig = (fv_900 != 0).sum()
    doc.append(f"数据: {bar_900.index[0].strftime('%Y-%m-%d')} ~ {bar_900.index[-1].strftime('%Y-%m-%d')}")
    doc.append(f"总bar: {n_total}, 高振幅: {n_high} ({n_high/n_total*100:.0f}%), 低振幅: {n_low} ({n_low/n_total*100:.0f}%)")
    doc.append(f"信号bar: {n_sig} ({n_sig/n_total*100:.1f}%)\n")

    print("900天 sweep...")
    r900 = sweep(bar_900, fv_900, amp_900)

    doc.append("## 900天完整扫描结果\n")
    doc.append(f"{'Fwd':>4s} | {'Regime':>8s} | {'Pos_N':>5s} {'Neg_N':>5s} | {'Pos均值':>8s} {'Neg均值':>8s} | {'差距':>8s}")
    doc.append("-" * 65)
    for _, r in r900.iterrows():
        sp = f"{r['spread']:+.1f}" if pd.notna(r['spread']) else "N/A"
        pm = f"{r['pos_mean']:+.1f}" if pd.notna(r['pos_mean']) else "N/A"
        nm = f"{r['neg_mean']:+.1f}" if pd.notna(r['neg_mean']) else "N/A"
        mark = " ★" if pd.notna(r['spread']) and abs(r['spread']) >= 10 else ""
        doc.append(f"{int(r['fwd']):>4d} | {r['regime']:>8s} | {int(r['pos_n']):>5d} {int(r['neg_n']):>5d} | {pm:>8s} {nm:>8s} | {sp:>8s}{mark}")

    # 219天对比
    print("219天 sweep...")
    bar_219 = load_im_bars('20250516', '20260409')
    amp_219 = calc_amp(bar_219)
    fv_219 = hr.compute_series(bar_219)
    r219 = sweep(bar_219, fv_219, amp_219)

    doc.append("\n## 219天 vs 900天对比（IM低振幅日）\n")
    doc.append(f"{'Fwd':>4s} | {'219天差距':>10s} | {'900天差距':>10s} | {'变化':>8s}")
    doc.append("-" * 45)

    r219_low = r219[r219['regime'] == 'low_amp'].set_index('fwd')
    r900_low = r900[r900['regime'] == 'low_amp'].set_index('fwd')

    for n in FWD_PERIODS:
        s219 = r219_low.loc[n, 'spread'] if n in r219_low.index else np.nan
        s900 = r900_low.loc[n, 'spread'] if n in r900_low.index else np.nan
        s219_s = f"{s219:+.1f}" if pd.notna(s219) else "N/A"
        s900_s = f"{s900:+.1f}" if pd.notna(s900) else "N/A"
        if pd.notna(s219) and pd.notna(s900):
            change = s900 - s219
            ch_s = f"{change:+.1f}"
        else:
            ch_s = "—"
        doc.append(f"{n:>4d} | {s219_s:>10s} | {s900_s:>10s} | {ch_s:>8s}")

    # 曲线图
    fig, ax = plt.subplots(figsize=(10, 5))
    low219 = r219[r219['regime'] == 'low_amp']
    low900 = r900[r900['regime'] == 'low_amp']
    ax.plot(low219['fwd'], low219['spread'], 'o--', label='219 days', markersize=6)
    ax.plot(low900['fwd'], low900['spread'], 's-', label='900 days', markersize=6, linewidth=2)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axhline(y=10, color='green', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axhline(y=-10, color='green', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_xlabel('Forward Period (bars)')
    ax.set_ylabel('Pos-Neg Spread (bps)')
    ax.set_title('IM Low Amplitude Days — Horizontal Reversal Spread: 219d vs 900d')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = output_dir / "hr_900d_vs_219d.png"
    plt.savefig(str(fig_path), dpi=150)
    plt.close()
    doc.append(f"\n## 曲线图\n![comparison]({fig_path})\n")

    # 判定
    doc.append("## 判定\n")
    low900_spreads = r900_low['spread'].dropna()
    above10 = (low900_spreads.abs() >= 10).sum()
    peak = low900_spreads.abs().max() if len(low900_spreads) > 0 else 0
    peak_fwd = low900_spreads.abs().idxmax() if len(low900_spreads) > 0 else 0

    if above10 >= 3 and peak >= 15:
        doc.append("**结果A：900天上高原仍然存在** ✓")
        doc.append(f"  {above10}个forward period |spread|>=10bps, 峰值={peak:.1f}bps @ fwd={peak_fwd}")
        doc.append("→ 信号是真实的，可以考虑集成方式")
    elif above10 == 0 or peak < 5:
        doc.append("**结果B：900天上完全消失** ✗")
        doc.append(f"  最大|spread|={peak:.1f}bps, 达标数={above10}")
        doc.append("→ 跟[75,80)陷阱一样是小样本噪音，归档放弃")
    else:
        doc.append(f"**结果C：900天上信号弱化**")
        doc.append(f"  {above10}个forward period达标, 峰值={peak:.1f}bps @ fwd={peak_fwd}")
        doc.append("→ 规律部分成立但边际价值降低")

    # 样本数
    doc.append("\n## 样本数\n")
    for n in [5, 8, 12]:
        if n in r900_low.index:
            r = r900_low.loc[n]
            concern = " ⚠sample concern" if r['pos_n'] < 100 or r['neg_n'] < 100 else ""
            doc.append(f"fwd={n}: Pos={int(r['pos_n'])}, Neg={int(r['neg_n'])}{concern}")

    report = "\n".join(doc)
    path = output_dir / "horizontal_reversal_900d_validation.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
