#!/usr/bin/env python3
"""节奏摆动日定义与筛选。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd
from pathlib import Path
from data.storage.db_manager import get_db


def load_im():
    db = get_db()
    df = db.query_df(
        "SELECT datetime, open, high, low, close, volume FROM index_min "
        "WHERE symbol='000852' AND period=300 ORDER BY datetime"
    )
    for c in ['open','high','low','close','volume']:
        df[c] = df[c].astype(float)
    df.index = pd.to_datetime(df['datetime'])
    df['date'] = df.index.date
    return df


def compute_day_stats(df):
    """计算每日统计。"""
    daily_close = df.groupby('date')['close'].last()
    rows = []
    for date, day in df.groupby('date'):
        if len(day) < 10:
            continue
        open_p = float(day.iloc[0]['open'])
        close_p = float(day.iloc[-1]['close'])
        prev_dates = [d for d in daily_close.index if d < date]
        prev_c = float(daily_close[prev_dates[-1]]) if prev_dates else open_p

        ob = day.iloc[:6]
        o30_amp = (ob['high'].max() - ob['low'].min()) / ob['open'].iloc[0] * 100

        dh, dl = day['high'].max(), day['low'].min()
        full_amp = (dh - dl) / open_p * 100

        # 上午/下午 (UTC: am=01:45~03:30, pm=05:00~06:30)
        am = day[(day.index.hour >= 1) & (day.index.hour < 4)]
        pm = day[(day.index.hour >= 5) & (day.index.hour < 7)]
        am_amp = (am['high'].max() - am['low'].min()) / open_p * 100 if len(am) > 0 else 0
        pm_amp = (pm['high'].max() - pm['low'].min()) / open_p * 100 if len(pm) > 0 else 0

        ratio = full_amp / o30_amp if o30_amp > 0 else 99

        rows.append({
            'date': date, 'open': open_p, 'close': close_p, 'prev_close': prev_c,
            'o30_amp': o30_amp, 'full_amp': full_amp, 'ratio': ratio,
            'am_amp': am_amp, 'pm_amp': pm_amp,
            'day_high': dh, 'day_low': dl,
        })
    return pd.DataFrame(rows)


def classify(row, o30_lo=0.4, o30_hi=1.2, ratio_max=1.8, full_max=1.5):
    """分类：节奏摆动日 / 趋势日 / 窄幅日 / 其他。"""
    if row['o30_amp'] < o30_lo:
        return 'narrow'
    if (o30_lo <= row['o30_amp'] <= o30_hi and
        row['ratio'] < ratio_max and
        row['full_amp'] < full_max and
        row['am_amp'] >= row['pm_amp'] * 0.8):
        return 'rhythm_swing'
    if row['ratio'] >= ratio_max or row['full_amp'] >= 2.0:
        return 'trend'
    return 'other'


def main():
    df = load_im()
    stats = compute_day_stats(df)

    doc = ["# 节奏摆动日（Rhythm-Swing Day）定义与筛选\n"]

    # ══════════════════════════════════════════════
    # 第一步：候选定义
    # ══════════════════════════════════════════════
    doc.append("## 候选定义\n")
    doc.append("```")
    doc.append("open30_amp ∈ [0.4%, 1.2%]")
    doc.append("AND full_amp / open30_amp < 1.8")
    doc.append("AND full_amp < 1.5%")
    doc.append("AND am_amp >= pm_amp × 0.8")
    doc.append("```\n")

    # ══════════════════════════════════════════════
    # 第二步：4/7和4/10验证
    # ══════════════════════════════════════════════
    doc.append("## 4/7 和 4/10 逐条件验证\n")
    import datetime
    targets = [datetime.date(2026, 4, 7), datetime.date(2026, 4, 10)]
    doc.append(f"| 条件 | 4/7 | 通过? | 4/10 | 通过? |")
    doc.append(f"|------|-----|-------|------|-------|")

    for td in targets:
        row = stats[stats['date'] == td]
        if len(row) == 0:
            doc.append(f"| {td} 无数据 | | | | |")

    t7 = stats[stats['date'] == targets[0]].iloc[0] if len(stats[stats['date'] == targets[0]]) > 0 else None
    t10 = stats[stats['date'] == targets[1]].iloc[0] if len(stats[stats['date'] == targets[1]]) > 0 else None

    if t7 is not None and t10 is not None:
        checks = [
            ("open30 ∈ [0.4,1.2]%",
             f"{t7['o30_amp']:.2f}%", 0.4 <= t7['o30_amp'] <= 1.2,
             f"{t10['o30_amp']:.2f}%", 0.4 <= t10['o30_amp'] <= 1.2),
            ("full/open30 < 1.8",
             f"{t7['ratio']:.2f}", t7['ratio'] < 1.8,
             f"{t10['ratio']:.2f}", t10['ratio'] < 1.8),
            ("full_amp < 1.5%",
             f"{t7['full_amp']:.2f}%", t7['full_amp'] < 1.5,
             f"{t10['full_amp']:.2f}%", t10['full_amp'] < 1.5),
            ("am >= pm×0.8",
             f"{t7['am_amp']:.2f}/{t7['pm_amp']:.2f}", t7['am_amp'] >= t7['pm_amp'] * 0.8,
             f"{t10['am_amp']:.2f}/{t10['pm_amp']:.2f}", t10['am_amp'] >= t10['pm_amp'] * 0.8),
        ]
        for label, v7, p7, v10, p10 in checks:
            doc.append(f"| {label} | {v7} | {'✓' if p7 else '✗'} | {v10} | {'✓' if p10 else '✗'} |")

        both_pass = all(p7 and p10 for _, _, p7, _, p10 in checks)
        doc.append(f"\n**两天都通过: {'✓' if both_pass else '✗ 需要调整定义'}**\n")

    # ══════════════════════════════════════════════
    # 第三步：最近2个月筛选
    # ══════════════════════════════════════════════
    recent = stats[stats['date'] >= datetime.date(2026, 2, 15)]
    recent['type'] = recent.apply(classify, axis=1)

    doc.append("## 最近2个月统计\n")
    doc.append(f"时间范围: {recent['date'].min()} ~ {recent['date'].max()} ({len(recent)}天)\n")

    type_counts = recent['type'].value_counts()
    for t in ['rhythm_swing', 'trend', 'narrow', 'other']:
        n = type_counts.get(t, 0)
        doc.append(f"- **{t}**: {n}天 ({n/len(recent)*100:.0f}%)")

    # 节奏摆动日列表
    swing_days = recent[recent['type'] == 'rhythm_swing'].sort_values('date')
    doc.append(f"\n## 节奏摆动日列表 ({len(swing_days)}天)\n")
    doc.append(f"| date | open30_amp% | full_amp% | full/open30 | am_amp% | pm_amp% | 备注 |")
    doc.append(f"|------|-------------|-----------|-------------|---------|---------|------|")
    for _, r in swing_days.iterrows():
        note = ""
        if r['date'] == datetime.date(2026, 4, 7): note = "✓目标日"
        elif r['date'] == datetime.date(2026, 4, 10): note = "✓目标日"
        doc.append(f"| {r['date']} | {r['o30_amp']:.2f} | {r['full_amp']:.2f} | {r['ratio']:.2f} | "
                   f"{r['am_amp']:.2f} | {r['pm_amp']:.2f} | {note} |")

    # ══════════════════════════════════════════════
    # 第四步：对照样本
    # ══════════════════════════════════════════════
    doc.append(f"\n## 对照样本（非摆动日）\n")
    doc.append(f"| date | type | open30_amp% | full_amp% | full/open30 | am% | pm% |")
    doc.append(f"|------|------|-------------|-----------|-------------|-----|-----|")
    for t in ['trend', 'narrow', 'other']:
        sub = recent[recent['type'] == t].sort_values('date')
        for _, r in sub.tail(2).iterrows():
            doc.append(f"| {r['date']} | {t} | {r['o30_amp']:.2f} | {r['full_amp']:.2f} | {r['ratio']:.2f} | "
                       f"{r['am_amp']:.2f} | {r['pm_amp']:.2f} |")

    # ══════════════════════════════════════════════
    # 第五步：每个摆动日画像
    # ══════════════════════════════════════════════
    doc.append(f"\n## 节奏摆动日画像\n")
    for _, r in swing_days.iterrows():
        day = df[df['date'] == r['date']]
        if len(day) < 10: continue

        # 时段数据
        ob = day.iloc[:6]
        am = day[(day.index.hour >= 1) & (day.index.hour < 4)]
        pm = day[(day.index.hour >= 5) & (day.index.hour < 7)]

        doc.append(f"**{r['date']}**:")
        doc.append(f"  - 开盘 {r['open']:.0f}，前收 {r['prev_close']:.0f}")
        doc.append(f"  - 09:30-10:00: 振幅 {r['o30_amp']:.2f}%（{ob['high'].max()-ob['low'].min():.0f}pt），"
                   f"从 {ob['low'].min():.0f} 到 {ob['high'].max():.0f}")
        if len(am) > 0:
            doc.append(f"  - 10:00-11:30: max={am['high'].max():.0f}, min={am['low'].min():.0f}")
        if len(pm) > 0:
            doc.append(f"  - 13:00-14:30: max={pm['high'].max():.0f}, min={pm['low'].min():.0f}")
        doc.append(f"  - 全天 max={r['day_high']:.0f}, min={r['day_low']:.0f}, 振幅 {r['full_amp']:.2f}%")
        doc.append(f"  - 收盘 {r['close']:.0f}\n")

    report = "\n".join(doc)
    path = Path("tmp") / "rhythm_swing_day_filter.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
