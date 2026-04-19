#!/usr/bin/env python3
"""低振幅日市场结构验证：6个主观假设的数据检验。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd
from pathlib import Path
from data.storage.db_manager import get_db

AMP_THR = 0.4  # 开盘30min振幅阈值(%)


def load_data():
    db = get_db()
    df = db.query_df(
        "SELECT datetime, open, high, low, close, volume FROM index_min "
        "WHERE symbol='000852' AND period=300 ORDER BY datetime"
    )
    for c in ['open','high','low','close','volume']:
        df[c] = df[c].astype(float)
    df.index = pd.to_datetime(df['datetime'])
    df['date'] = df.index.date

    # prev_close (上一交易日最后一根bar的close)
    daily_close = df.groupby('date')['close'].last()
    df['prev_close'] = df['date'].map(lambda d: daily_close.shift(1).get(d, np.nan))

    return df


def identify_low_amp_days(df):
    """筛选低振幅日，返回每日统计。"""
    days = []
    for date, day_bars in df.groupby('date'):
        if len(day_bars) < 10:
            continue
        ob = day_bars.iloc[:6]  # 开盘30min
        open_amp = (ob['high'].max() - ob['low'].min()) / ob['open'].iloc[0] * 100
        if open_amp >= AMP_THR:
            continue

        open_price = float(day_bars.iloc[0]['open'])
        first_bar_close = float(day_bars.iloc[0]['close'])
        prev_close = float(day_bars.iloc[0]['prev_close']) if pd.notna(day_bars.iloc[0]['prev_close']) else open_price
        gap_pct = (open_price - prev_close) / prev_close * 100 if prev_close > 0 else 0

        day_high = day_bars['high'].max()
        day_low = day_bars['low'].min()

        # post_open = 第一根bar之后
        post_open = day_bars.iloc[1:]
        po_high = post_open['high'].max() if len(post_open) > 0 else day_high
        po_low = post_open['low'].min() if len(post_open) > 0 else day_low

        day_amp_full = (day_high - day_low) / open_price * 100
        day_amp_post = (po_high - po_low) / first_bar_close * 100 if first_bar_close > 0 else 0

        # 上午/下午分段
        # UTC: 上午 01:45~03:30 = BJ 09:45~11:30, 下午 05:00~06:30 = BJ 13:00~14:30
        am_bars = day_bars[(day_bars.index.hour >= 1) & (day_bars.index.hour < 4)]
        pm_bars = day_bars[(day_bars.index.hour >= 5) & (day_bars.index.hour < 7)]
        am_amp = (am_bars['high'].max() - am_bars['low'].min()) / open_price * 100 if len(am_bars) > 0 else 0
        pm_amp = (pm_bars['high'].max() - pm_bars['low'].min()) / open_price * 100 if len(pm_bars) > 0 else 0

        # EMA20
        ema20 = day_bars['close'].ewm(span=20).mean()
        dev = (day_bars['close'] - ema20) / ema20 * 100

        max_dev_above = dev.max()
        max_dev_below = dev.min()
        mean_abs_dev = dev.abs().mean()

        # 穿越次数
        sign_series = np.sign(day_bars['close'].values - ema20.values)
        crossings = sum(sign_series[i] != sign_series[i-1] and sign_series[i] != 0 and sign_series[i-1] != 0
                        for i in range(1, len(sign_series)))

        # EMA drift
        ema_drift = (float(ema20.iloc[-1]) - float(ema20.iloc[0])) / float(ema20.iloc[0]) * 100 if float(ema20.iloc[0]) > 0 else 0

        days.append({
            'date': date, 'open_price': open_price, 'first_bar_close': first_bar_close,
            'gap_pct': gap_pct, 'day_high': day_high, 'day_low': day_low,
            'day_amp_full': day_amp_full, 'day_amp_post': day_amp_post,
            'am_amp': am_amp, 'pm_amp': pm_amp,
            'max_dev_above': max_dev_above, 'max_dev_below': max_dev_below,
            'mean_abs_dev': mean_abs_dev, 'crossings': crossings, 'ema_drift': ema_drift,
        })

    return pd.DataFrame(days)


def main():
    df = load_data()
    days = identify_low_amp_days(df)
    n = len(days)

    doc = ["# 低振幅日市场结构验证\n"]
    doc.append(f"数据: IM 900天 ({df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')})")
    doc.append(f"低振幅日(<{AMP_THR}%): **{n}天** ({n/900*100:.0f}%)\n")

    # ══════════════════════════════════════════════
    # 假设1：日内振幅上界
    # ══════════════════════════════════════════════
    doc.append("## 假设1：日内振幅上界 (1.25%)\n")
    amp = days['day_amp_post']
    amp_full = days['day_amp_full']
    pct_below = (amp < 1.25).mean() * 100
    doc.append(f"| 指标 | 含跳空 | 去跳空 |")
    doc.append(f"|------|--------|--------|")
    doc.append(f"| 均值 | {amp_full.mean():.2f}% | {amp.mean():.2f}% |")
    doc.append(f"| 中位数 | {amp_full.median():.2f}% | {amp.median():.2f}% |")
    doc.append(f"| 75分位 | {amp_full.quantile(0.75):.2f}% | {amp.quantile(0.75):.2f}% |")
    doc.append(f"| 90分位 | {amp_full.quantile(0.90):.2f}% | {amp.quantile(0.90):.2f}% |")
    doc.append(f"| 95分位 | {amp_full.quantile(0.95):.2f}% | {amp.quantile(0.95):.2f}% |")
    doc.append(f"| <1.25% | {(amp_full<1.25).mean()*100:.0f}% | **{pct_below:.0f}%** |")

    if pct_below >= 90:
        doc.append(f"\n**假设1：成立** ✓ ({pct_below:.0f}%的低振幅日post-open振幅<1.25%)")
    elif pct_below >= 70:
        doc.append(f"\n**假设1：部分成立** ({pct_below:.0f}%<1.25%)")
    else:
        doc.append(f"\n**假设1：不成立** ✗ (仅{pct_below:.0f}%<1.25%)")

    # ══════════════════════════════════════════════
    # 假设2：均线振荡对称性
    # ══════════════════════════════════════════════
    doc.append("\n## 假设2：均线振荡对称性\n")
    symmetry = days['max_dev_above'].abs() / days['max_dev_below'].abs().replace(0, np.nan)
    doc.append(f"| 指标 | 值 |")
    doc.append(f"|------|-----|")
    doc.append(f"| 对称比中位数 | {symmetry.median():.2f} (1.0=完美对称) |")
    doc.append(f"| 25/75分位 | [{symmetry.quantile(0.25):.2f}, {symmetry.quantile(0.75):.2f}] |")
    doc.append(f"| mean_abs_dev中位数 | {days['mean_abs_dev'].median():.3f}% ({days['mean_abs_dev'].median()*80:.0f}pt@IM8000) |")
    doc.append(f"| max_dev_above中位数 | +{days['max_dev_above'].median():.3f}% |")
    doc.append(f"| max_dev_below中位数 | {days['max_dev_below'].median():.3f}% |")

    sym_med = symmetry.median()
    if 0.7 <= sym_med <= 1.4:
        doc.append(f"\n**假设2：成立** ✓ (对称比{sym_med:.2f}在0.7-1.4范围内)")
    else:
        doc.append(f"\n**假设2：不成立** ✗ (对称比{sym_med:.2f})")

    # ══════════════════════════════════════════════
    # 假设3：均线穿越频率
    # ══════════════════════════════════════════════
    doc.append("\n## 假设3：均线穿越频率\n")
    cx = days['crossings']
    doc.append(f"| 指标 | 值 |")
    doc.append(f"|------|-----|")
    doc.append(f"| 均值 | {cx.mean():.1f}次/天 |")
    doc.append(f"| 中位数 | {cx.median():.0f}次/天 |")
    doc.append(f"| 分布 | 0次:{(cx==0).sum()}天, 1次:{(cx==1).sum()}, 2次:{(cx==2).sum()}, 3次:{(cx==3).sum()}, 4次:{(cx==4).sum()}, 5+次:{(cx>=5).sum()} |")

    if 2 <= cx.median() <= 4:
        doc.append(f"\n**假设3：成立** ✓ (中位数{cx.median():.0f}次/天)")
    elif cx.median() < 2:
        doc.append(f"\n**假设3：不成立** ✗ (中位数{cx.median():.0f}次，大部分日子不是振荡)")
    else:
        doc.append(f"\n**假设3：部分成立** (中位数{cx.median():.0f}次，振荡比预期更频繁)")

    # ══════════════════════════════════════════════
    # 假设4：跳空后的振荡
    # ══════════════════════════════════════════════
    doc.append("\n## 假设4：跳空后的振荡\n")
    gap_days = days[days['gap_pct'].abs() > 0.3]
    no_gap = days[days['gap_pct'].abs() <= 0.3]
    doc.append(f"显著跳空(|gap|>0.3%)的低振幅日: {len(gap_days)}天 / {n}天")
    if len(gap_days) >= 5:
        gap_low = (gap_days['day_amp_post'] < 1.25).mean() * 100
        doc.append(f"跳空日post-open振幅<1.25%: {gap_low:.0f}%")
        doc.append(f"跳空日均线穿越中位数: {gap_days['crossings'].median():.0f}次")
        if gap_low >= 70:
            doc.append(f"\n**假设4：成立** ✓ (跳空不影响日内振荡)")
        else:
            doc.append(f"\n**假设4：不成立** ✗ (跳空日日内不再低振荡)")
    else:
        doc.append(f"\n**假设4：样本不足** (仅{len(gap_days)}天有显著跳空)")

    # ══════════════════════════════════════════════
    # 假设5：时段振幅差异
    # ══════════════════════════════════════════════
    doc.append("\n## 假设5：时段振幅差异\n")
    am_med = days['am_amp'].median()
    pm_med = days['pm_amp'].median()
    ratio = am_med / pm_med if pm_med > 0 else np.inf
    am_gt_pm = (days['am_amp'] > days['pm_amp']).mean() * 100

    doc.append(f"| 指标 | 上午 | 下午 |")
    doc.append(f"|------|------|------|")
    doc.append(f"| 振幅中位数 | {am_med:.2f}% | {pm_med:.2f}% |")
    doc.append(f"| 振幅均值 | {days['am_amp'].mean():.2f}% | {days['pm_amp'].mean():.2f}% |")
    doc.append(f"| AM/PM比值 | {ratio:.2f} | |")
    doc.append(f"| AM>PM天数 | {am_gt_pm:.0f}% | |")

    if ratio > 1.5:
        doc.append(f"\n**假设5：成立** ✓ (上午振幅{ratio:.1f}x下午)")
    elif ratio > 1.1:
        doc.append(f"\n**假设5：部分成立** (上午略大{ratio:.1f}x)")
    else:
        doc.append(f"\n**假设5：不成立** ✗ (上下午接近)")

    # ══════════════════════════════════════════════
    # 假设6：均线漂移+振荡叠加
    # ══════════════════════════════════════════════
    doc.append("\n## 假设6：均线漂移+振荡叠加\n")
    drift = days['ema_drift']
    drift_sig = days[drift.abs() > 0.3]

    doc.append(f"| 指标 | 值 |")
    doc.append(f"|------|-----|")
    doc.append(f"| EMA drift中位数 | {drift.median():+.3f}% |")
    doc.append(f"| drift均值 | {drift.mean():+.3f}% |")
    doc.append(f"| |drift|>0.3%的天数 | {len(drift_sig)}天 ({len(drift_sig)/n*100:.0f}%) |")
    doc.append(f"| |drift|>0.5%的天数 | {(drift.abs()>0.5).sum()}天 |")

    if len(drift_sig) >= 5:
        doc.append(f"\n有漂移的低振幅日(|drift|>0.3%)振幅: {drift_sig['day_amp_post'].median():.2f}%")
        doc.append(f"无漂移的低振幅日振幅: {days[drift.abs()<=0.3]['day_amp_post'].median():.2f}%")

    if len(drift_sig) / n >= 0.3:
        doc.append(f"\n**假设6：成立** ✓ ({len(drift_sig)/n*100:.0f}%的日子有显著均线漂移)")
    elif len(drift_sig) / n >= 0.1:
        doc.append(f"\n**假设6：部分成立** ({len(drift_sig)/n*100:.0f}%有漂移)")
    else:
        doc.append(f"\n**假设6：不成立** ✗ (仅{len(drift_sig)/n*100:.0f}%有漂移)")

    # ══════════════════════════════════════════════
    # 综合判定
    # ══════════════════════════════════════════════
    doc.append("\n## 综合判定\n")

    report = "\n".join(doc)
    path = Path("tmp") / "low_amp_day_structure_verification.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
