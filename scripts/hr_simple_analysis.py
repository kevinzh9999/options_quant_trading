#!/usr/bin/env python3
"""简化版横盘反转因子评估。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np, pandas as pd
from pathlib import Path
from data.storage.db_manager import get_db
from models.factors.catalog_structure import HorizontalReversalSimple, HorizontalReversalFactor, BollBreakout
from models.factors.catalog_price import MomSimple
from models.factors.catalog_volume import QtyRatio
from models.factors.evaluator import FactorEvaluator

SPOT = {"IM": "000852", "IC": "000905"}
AMP_THR = 0.4


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


def calc_daily_amp(bar_5m):
    daily_amp = {}
    for date, day_bars in bar_5m.groupby('date'):
        if len(day_bars) < 6: continue
        ob = day_bars.iloc[:6]
        daily_amp[date] = (ob['high'].max() - ob['low'].min()) / ob['open'].iloc[0] * 100
    s = bar_5m['date'].map(daily_amp)
    s.index = bar_5m.index
    return s


def regime_decomp(bar_5m, factor_values, fwd12, amp_series, sym):
    """Regime分解：高/低振幅日 × Pos/Neg组。"""
    df = pd.DataFrame({
        'factor': factor_values, 'fwd12': fwd12, 'amp': amp_series
    }).dropna()

    lines = []
    for regime, amp_cond, label in [
        ('high', df['amp'] >= AMP_THR, '高振幅(>=0.4%)'),
        ('low', df['amp'] < AMP_THR, '低振幅(<0.4%)'),
    ]:
        sub = df[amp_cond]
        pos = sub[sub['factor'] > 0]
        neg = sub[sub['factor'] < 0]
        zero = sub[sub['factor'] == 0]

        lines.append(f"\n**{label}** (总{len(sub)}bar, signal={len(pos)+len(neg)}, 无信号={len(zero)})")
        lines.append(f"| 组 | N | fwd12均值(bps) | 中位数 | 正收益% |")
        lines.append(f"|---|---|---|---|---|")

        for grp_label, grp in [("Pos(+1,预测多)", pos), ("Neg(-1,预测空)", neg)]:
            if len(grp) < 10:
                lines.append(f"| {sym} {grp_label} | {len(grp)} | N/A(<10) | | |")
                continue
            m = grp['fwd12'].mean()
            med = grp['fwd12'].median()
            ppos = (grp['fwd12'] > 0).mean() * 100
            lines.append(f"| {sym} {grp_label} | {len(grp)} | {m:+.1f} | {med:+.1f} | {ppos:.0f}% |")

        if len(pos) >= 10 and len(neg) >= 10:
            spread = pos['fwd12'].mean() - neg['fwd12'].mean()
            strength = "达标(>=10)" if abs(spread) >= 10 else ("边缘(5-10)" if abs(spread) >= 5 else "不足(<5)")
            lines.append(f"  收益差(Pos-Neg): **{spread:+.1f}bps** ({strength})")
    return "\n".join(lines)


def main():
    output_dir = Path("tmp")
    doc = ["# 简化版横盘反转因子评估\n"]
    doc.append("## 因子定义\n")
    doc.append("简化版：`prior_mom(K) > 0 且 recent_N_bar_high <= ref_high` → -1（预测反转下跌）")
    doc.append("复杂版额外加了：布林宽度归一化 × 低波动确认 × 连续值强度。简化版只输出+1/-1/0。\n")

    for sym in ['IM', 'IC']:
        bar_5m = load_bars(sym)
        amp = calc_daily_amp(bar_5m)
        fwd12 = bar_5m['close'].pct_change(12).shift(-12) * 10000

        doc.append(f"\n---\n## {sym}\n")

        # 默认因子
        hr = HorizontalReversalSimple(12, 3)
        fv = hr.compute_series(bar_5m)

        # 信号稀疏度
        n_signal = (fv != 0).sum()
        n_pos = (fv > 0).sum()
        n_neg = (fv < 0).sum()
        doc.append(f"### 信号稀疏度")
        doc.append(f"总bar: {len(fv)}, 有信号: {n_signal} ({n_signal/len(fv)*100:.1f}%), "
                   f"+1={n_pos}, -1={n_neg}\n")

        # FactorEvaluator
        evaluator = FactorEvaluator(bar_5m, forward_periods=[3, 6, 12], daily_range=amp)
        hr_complex = HorizontalReversalFactor(12, 3, 3)
        mom = MomSimple(12)
        boll = BollBreakout(20)
        vol = QtyRatio(20)

        results, corr_df = evaluator.batch_evaluate([hr, hr_complex, mom, boll, vol])
        r_simple = results[0]
        r_complex = results[1]

        # Daily IC
        doc.append("### Daily IC\n")
        doc.append(f"| Forward | 简化版 dIC | 复杂版 dIC |")
        doc.append(f"|---------|-----------|-----------|")
        for n in [3, 6, 12]:
            s_ic = r_simple['ic'].get(f'dIC_{n}bar', np.nan)
            c_ic = r_complex['ic'].get(f'dIC_{n}bar', np.nan)
            s_s = f"{s_ic:.4f}" if not np.isnan(s_ic) else "N/A"
            c_s = f"{c_ic:.4f}" if not np.isnan(c_ic) else "N/A"
            doc.append(f"| {n}bar | {s_s} | {c_s} |")

        # Regime IC
        if 'regime_ic' in r_simple and r_simple['regime_ic']:
            doc.append(f"\n### Regime IC")
            for k, v in sorted(r_simple['regime_ic'].items()):
                doc.append(f"  {k}: {v}")

        # 相关性
        doc.append(f"\n### 跟现有因子相关性")
        hr_corr = corr_df[hr.name]
        for fname in [hr_complex.name, mom.name, boll.name, vol.name]:
            if fname in hr_corr:
                doc.append(f"  vs {fname}: {hr_corr[fname]:.3f}")

        # Regime分解
        doc.append(f"\n### Regime分解验证")
        doc.append(regime_decomp(bar_5m, fv, fwd12, amp, sym))

    # 参数敏感性（只用IM）
    doc.append(f"\n---\n## 参数敏感性 (IM)\n")
    bar_im = load_bars('IM')
    ev_im = FactorEvaluator(bar_im, forward_periods=[6])
    doc.append(f"{'K':>3s} {'N':>3s} | {'dIC_6bar':>9s} | {'信号率':>6s}")
    doc.append("-" * 30)
    for K in [6, 12, 18]:
        for N in [2, 3, 5]:
            f = HorizontalReversalSimple(K, N)
            fv = f.compute_series(bar_im)
            r = ev_im.evaluate(f)
            dic = r['ic'].get('dIC_6bar', np.nan)
            sig_rate = (fv != 0).sum() / len(fv) * 100
            dic_s = f"{dic:.4f}" if not np.isnan(dic) else "N/A"
            doc.append(f"{K:>3d} {N:>3d} | {dic_s:>9s} | {sig_rate:>5.1f}%")

    # 结论
    doc.append(f"\n---\n## 结论\n")

    report = "\n".join(doc)
    path = output_dir / "horizontal_reversal_simple_analysis.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
