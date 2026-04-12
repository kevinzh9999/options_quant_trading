#!/usr/bin/env python3
"""节奏摆动日"早盘高点回落空间"基本面分析——纯上帝视角统计，不涉及任何策略。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

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


def is_rhythm_swing(day, open_p):
    if len(day) < 10: return False
    ob = day.iloc[:6]
    o30 = (ob['high'].max() - ob['low'].min()) / ob['open'].iloc[0] * 100
    dh, dl = day['high'].max(), day['low'].min()
    full = (dh - dl) / open_p * 100
    ratio = full / o30 if o30 > 0 else 99
    am = day[(day.index.hour >= 1) & (day.index.hour < 4)]
    pm = day[(day.index.hour >= 5) & (day.index.hour < 7)]
    am_a = (am['high'].max() - am['low'].min()) / open_p * 100 if len(am) > 0 else 0
    pm_a = (pm['high'].max() - pm['low'].min()) / open_p * 100 if len(pm) > 0 else 0
    return 0.4 <= o30 <= 1.2 and ratio < 1.8 and full < 1.5 and am_a >= pm_a * 0.8


def find_candidates(df):
    daily_close = df.groupby('date')['close'].last()
    all_dates = sorted(df['date'].unique())
    candidates = []
    for i, date in enumerate(all_dates):
        day = df[df['date'] == date]
        if len(day) < 10: continue
        open_p = float(day.iloc[0]['open'])
        prev_c = float(daily_close.get(all_dates[i-1], open_p)) if i > 0 else open_p
        if not is_rhythm_swing(day, open_p): continue
        gap = (open_p - prev_c) / prev_c * 100 if prev_c > 0 else 0
        if gap < -0.2: continue
        early = day[(day.index.hour == 1) | (day.index.hour == 2)]
        if len(early) < 4: continue
        eh, el = early['high'].max(), early['low'].min()
        if (eh - el) / open_p * 100 < 0.4: continue
        peak_time = early['high'].idxmax()

        # 找实际入场点（3 bar无新高后下一根bar的open）
        after = day[day.index > peak_time]
        count = 0
        entry_price = None
        for j, (idx, bar) in enumerate(after.iterrows()):
            if float(bar['high']) <= eh:
                count += 1
            else:
                count = 0
            if count >= 3:
                remaining = after.iloc[j+1:]
                if len(remaining) > 0:
                    entry_price = float(remaining.iloc[0]['open'])
                break

        candidates.append({
            'date': date, 'day_bars': day,
            'early_peak': eh, 'early_low': el, 'early_amp': eh - el,
            'peak_time': peak_time, 'entry_price': entry_price,
            'day_close': float(day.iloc[-1]['close']),
            'day_low': day['low'].min(), 'day_high': day['high'].max(),
        })
    return candidates


def find_first_local_min(day, peak_time, peak_price):
    """找早盘高点之后的第一个local minimum。

    定义：bar的low比前后各2根bar的low都低（5-bar局部最低）。
    只看peak之后到收盘的bar。
    """
    after = day[day.index > peak_time]
    if len(after) < 5:
        return float(after['low'].min()) if len(after) > 0 else peak_price

    lows = after['low'].values.astype(float)
    # 找5-bar窗口的局部最低
    for i in range(2, len(lows) - 2):
        if lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and \
           lows[i] <= lows[i+1] and lows[i] <= lows[i+2]:
            return lows[i]

    # 没找到严格局部最低，取peak之后的全局最低
    return float(after['low'].min())


def compute_metrics(candidates):
    """对每个候选日计算5个回落空间指标。"""
    rows = []
    for c in candidates:
        day = c['day_bars']
        ep = c['early_peak']
        day_low = float(c['day_low'])
        day_close = c['day_close']
        entry_p = c['entry_price']

        # 第一个local min
        first_lm = find_first_local_min(day, c['peak_time'], ep)

        row = {
            'date': c['date'],
            'early_amp': c['early_amp'],
            'early_peak': ep,
            # 指标1: 早盘高点到全天最低
            'peak_to_eod_low': ep - day_low,
            # 指标2: 早盘高点到收盘
            'peak_to_eod_close': ep - day_close,
            # 指标3: 早盘高点到第一个local min
            'peak_to_first_lm': ep - first_lm,
            # 指标4: 实际入场点到第一个local min
            'entry_to_first_lm': (entry_p - first_lm) if entry_p is not None else np.nan,
            # 指标5: 第一个local min到收盘（反弹幅度）
            'first_lm_to_eod': day_close - first_lm,
            # 附加
            'first_lm': first_lm,
            'entry_price': entry_p,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def stat_table(data, cols, labels):
    """生成统计表。"""
    lines = []
    lines.append(f"| 指标 | 均值 | 中位数 | 25% | 75% | 最大 |")
    lines.append(f"|------|------|--------|-----|-----|------|")
    for col, label in zip(cols, labels):
        s = data[col].dropna()
        if len(s) == 0:
            lines.append(f"| {label} | N/A | N/A | N/A | N/A | N/A |")
            continue
        lines.append(f"| {label} | {s.mean():.1f} | {s.median():.1f} | "
                     f"{s.quantile(0.25):.1f} | {s.quantile(0.75):.1f} | {s.max():.1f} |")
    return lines


def main():
    print("加载数据...")
    df = load_im()
    print("筛选候选日...")
    candidates = find_candidates(df)
    print(f"候选日: {len(candidates)}天")

    metrics = compute_metrics(candidates)
    n = len(metrics)

    # 三段切分
    is_n = 130
    oos_mid = is_n + (n - is_n) // 2

    is_data = metrics.iloc[:is_n]
    oos_first = metrics.iloc[is_n:oos_mid]
    oos_second = metrics.iloc[oos_mid:]

    cols = ['peak_to_eod_low', 'peak_to_eod_close', 'peak_to_first_lm',
            'entry_to_first_lm', 'first_lm_to_eod']
    labels = ['peak→全天最低', 'peak→收盘', 'peak→第一波低点',
              '入场→第一波低点', '第一波低点→收盘(反弹)']

    doc = ["# 节奏摆动日 早盘高点回落空间 基本面分析\n"]
    doc.append(f"数据: IM {n}天节奏摆动日（冲高型）")
    doc.append(f"- IS段: 前{len(is_data)}天 ({is_data['date'].iloc[0]}~{is_data['date'].iloc[-1]})")
    doc.append(f"- OOS前半: {len(oos_first)}天 ({oos_first['date'].iloc[0]}~{oos_first['date'].iloc[-1]})")
    doc.append(f"- OOS后半: {len(oos_second)}天 ({oos_second['date'].iloc[0]}~{oos_second['date'].iloc[-1]})\n")

    # ═══════════ 三段统计表 ═══════════
    for label, data in [("IS段", is_data), ("OOS前半", oos_first), ("OOS后半", oos_second)]:
        doc.append(f"## {label}（{len(data)}天）\n")
        doc.extend(stat_table(data, cols, labels))
        doc.append("")

    # ═══════════ 三段对比 ═══════════
    doc.append("## 三段对比（中位数）\n")
    doc.append("| 指标 | IS | OOS前半 | OOS后半 | IS→OOS后半变化 |")
    doc.append("|------|-----|---------|---------|---------------|")
    for col, label in zip(cols, labels):
        is_med = is_data[col].dropna().median()
        oos1_med = oos_first[col].dropna().median()
        oos2_med = oos_second[col].dropna().median()
        change = oos2_med - is_med
        doc.append(f"| {label} | {is_med:.1f} | {oos1_med:.1f} | {oos2_med:.1f} | {change:+.1f} |")
    doc.append("")

    # ═══════════ 策略实际抓取率 ═══════════
    doc.append("## 策略抓取率分析\n")
    doc.append("对比策略实际avg pnl vs 理论空间:\n")

    # 从OOS decay诊断已知: 基线IS avg=+8.5, OOS avg=+5.6, OOS前半~+7.9, OOS后半~+3.3
    strategy_pnl = {'IS': 8.5, 'OOS前半': 7.9, 'OOS后半': 3.3}

    doc.append("| 段 | 策略avg | peak→低点(中位数) | 入场→低点(中位数) | 抓取率(vs peak) | 抓取率(vs 入场) |")
    doc.append("|-----|---------|-------------------|-------------------|-----------------|-----------------|")
    for label, data, key in [("IS", is_data, "IS"), ("OOS前半", oos_first, "OOS前半"), ("OOS后半", oos_second, "OOS后半")]:
        p2lm = data['peak_to_first_lm'].dropna().median()
        e2lm = data['entry_to_first_lm'].dropna().median()
        sp = strategy_pnl[key]
        cap_p = sp / p2lm * 100 if p2lm > 0 else 0
        cap_e = sp / e2lm * 100 if e2lm > 0 else 0
        doc.append(f"| {label} | {sp:+.1f}pt | {p2lm:.1f}pt | {e2lm:.1f}pt | {cap_p:.0f}% | {cap_e:.0f}% |")
    doc.append("")

    # ═══════════ 入场时机损失 ═══════════
    doc.append("## 入场时机损失\n")
    doc.append("| 段 | peak→低点(中位数) | 入场→低点(中位数) | 入场滞后损失 | 损失占比 |")
    doc.append("|-----|-------------------|-------------------|-------------|---------|")
    for label, data in [("IS", is_data), ("OOS前半", oos_first), ("OOS后半", oos_second)]:
        p2lm = data['peak_to_first_lm'].dropna().median()
        e2lm = data['entry_to_first_lm'].dropna().median()
        lag = p2lm - e2lm
        pct = lag / p2lm * 100 if p2lm > 0 else 0
        doc.append(f"| {label} | {p2lm:.1f}pt | {e2lm:.1f}pt | {lag:.1f}pt | {pct:.0f}% |")
    doc.append("")

    # ═══════════ 持有到EOD分析 ═══════════
    doc.append("## 持有到EOD分析\n")
    doc.append("| 段 | peak→低点(中位数) | peak→收盘(中位数) | 低点后反弹(中位数) | 持有到EOD好? |")
    doc.append("|-----|-------------------|-------------------|-------------------|-------------|")
    for label, data in [("IS", is_data), ("OOS前半", oos_first), ("OOS后半", oos_second)]:
        p2lm = data['peak_to_first_lm'].dropna().median()
        p2c = data['peak_to_eod_close'].dropna().median()
        bounce = data['first_lm_to_eod'].dropna().median()
        good = "✓ 差异小" if abs(p2c - p2lm) < p2lm * 0.3 else "✗ 反弹大"
        doc.append(f"| {label} | {p2lm:.1f}pt | {p2c:.1f}pt | {bounce:+.1f}pt | {good} |")
    doc.append("")

    # ═══════════ 按振幅分组 ═══════════
    doc.append("## 按早盘振幅分组\n")
    bins = [(0, 50, '0-50pt'), (50, 80, '50-80pt'), (80, 200, '80+pt')]
    doc.append("| 振幅组 | N | peak→低点均值 | peak→低点中位数 | peak→收盘中位数 | 回落/振幅比 |")
    doc.append("|--------|---|--------------|----------------|----------------|-----------|")
    for lo, hi, label in bins:
        sub = metrics[(metrics['early_amp'] >= lo) & (metrics['early_amp'] < hi)]
        if len(sub) == 0: continue
        p2lm_mean = sub['peak_to_first_lm'].dropna().mean()
        p2lm_med = sub['peak_to_first_lm'].dropna().median()
        p2c_med = sub['peak_to_eod_close'].dropna().median()
        amp_med = sub['early_amp'].median()
        ratio = p2lm_med / amp_med * 100 if amp_med > 0 else 0
        doc.append(f"| {label} | {len(sub)} | {p2lm_mean:.1f}pt | {p2lm_med:.1f}pt | {p2c_med:.1f}pt | {ratio:.0f}% |")
    doc.append("")

    # 按振幅分组 × 三段
    doc.append("### 振幅组 × 时段交叉\n")
    doc.append("| 振幅组 | 段 | N | peak→低点中位数 | peak→收盘中位数 |")
    doc.append("|--------|-----|---|----------------|----------------|")
    for lo, hi, amp_label in bins:
        for seg_label, seg_data in [("IS", is_data), ("OOS前", oos_first), ("OOS后", oos_second)]:
            sub = seg_data[(seg_data['early_amp'] >= lo) & (seg_data['early_amp'] < hi)]
            if len(sub) < 2: continue
            p2lm = sub['peak_to_first_lm'].dropna().median()
            p2c = sub['peak_to_eod_close'].dropna().median()
            doc.append(f"| {amp_label} | {seg_label} | {len(sub)} | {p2lm:.1f}pt | {p2c:.1f}pt |")
    doc.append("")

    # ═══════════ 回答四个问题 ═══════════
    doc.append("## 四个核心问题\n")

    # Q1
    is_p2low = is_data['peak_to_eod_low'].median()
    oos2_p2low = oos_second['peak_to_eod_low'].median()
    is_p2lm = is_data['peak_to_first_lm'].median()
    oos2_p2lm = oos_second['peak_to_first_lm'].median()

    doc.append(f"### Q1: 理论最大回落空间\n")
    doc.append(f"- IS中位数: peak→全天最低 = **{is_p2low:.1f}pt**，peak→第一波低点 = **{is_p2lm:.1f}pt**")
    doc.append(f"- OOS后半中位数: peak→全天最低 = **{oos2_p2low:.1f}pt**，peak→第一波低点 = **{oos2_p2lm:.1f}pt**")
    if is_p2low >= 40:
        doc.append(f"- **市场有充足空间**（{is_p2low:.0f}pt），问题在策略抓取效率")
    elif is_p2low >= 20:
        doc.append(f"- **市场有中等空间**（{is_p2low:.0f}pt），策略改善余地有限")
    else:
        doc.append(f"- **市场空间很小**（{is_p2low:.0f}pt），策略再优化也没用")
    doc.append("")

    # Q2
    is_e2lm = is_data['entry_to_first_lm'].dropna().median()
    oos2_e2lm = oos_second['entry_to_first_lm'].dropna().median()
    is_lag = is_p2lm - is_e2lm

    doc.append(f"### Q2: 策略时机损失\n")
    doc.append(f"- IS: peak→低点={is_p2lm:.1f}pt，入场→低点={is_e2lm:.1f}pt，滞后损失={is_lag:.1f}pt ({is_lag/is_p2lm*100:.0f}%)")
    oos2_lag = oos2_p2lm - oos2_e2lm
    doc.append(f"- OOS后半: peak→低点={oos2_p2lm:.1f}pt，入场→低点={oos2_e2lm:.1f}pt，滞后损失={oos2_lag:.1f}pt")
    if is_lag > 15:
        doc.append(f"- **入场时机损失大**（{is_lag:.0f}pt），3bar无新高等待太久")
    else:
        doc.append(f"- **入场时机损失可接受**（{is_lag:.0f}pt）")
    doc.append("")

    # Q3
    is_p2c = is_data['peak_to_eod_close'].median()
    is_bounce = is_data['first_lm_to_eod'].median()
    oos2_p2c = oos_second['peak_to_eod_close'].median()
    oos2_bounce = oos_second['first_lm_to_eod'].median()

    doc.append(f"### Q3: 持有到EOD好不好\n")
    doc.append(f"- IS: peak→收盘={is_p2c:.1f}pt，低点后反弹={is_bounce:+.1f}pt")
    doc.append(f"- OOS后半: peak→收盘={oos2_p2c:.1f}pt，低点后反弹={oos2_bounce:+.1f}pt")
    if is_bounce > 10:
        doc.append(f"- **反弹明显**（{is_bounce:.0f}pt），应在第一波低点附近止盈而非持有到EOD")
    else:
        doc.append(f"- **反弹有限**（{is_bounce:.0f}pt），持有到EOD可行")
    doc.append("")

    # Q4
    doc.append(f"### Q4: OOS后半衰退原因\n")
    doc.append(f"- 理论空间变化: IS peak→低点={is_p2lm:.1f}pt → OOS后半={oos2_p2lm:.1f}pt (变化{oos2_p2lm-is_p2lm:+.1f})")
    doc.append(f"- 理论空间变化: IS peak→全天低={is_p2low:.1f}pt → OOS后半={oos2_p2low:.1f}pt (变化{oos2_p2low-is_p2low:+.1f})")
    shrink_pct = (oos2_p2lm - is_p2lm) / is_p2lm * 100 if is_p2lm > 0 else 0
    if abs(shrink_pct) < 20:
        doc.append(f"- **理论空间变化不大**（{shrink_pct:+.0f}%），衰退更可能是策略时机问题")
    else:
        doc.append(f"- **理论空间明显缩小**（{shrink_pct:+.0f}%），是市场结构变化")
    doc.append("")

    # ═══════════ 核心结论 ═══════════
    doc.append("## 核心结论\n")

    # 策略抓取率
    cap_rate_is = 8.5 / is_p2lm * 100 if is_p2lm > 0 else 0
    cap_rate_oos2 = 3.3 / oos2_p2lm * 100 if oos2_p2lm > 0 else 0

    doc.append(f"1. **理论最大空间**: IS中位数 peak→低点={is_p2lm:.0f}pt, peak→全天低={is_p2low:.0f}pt")
    doc.append(f"2. **策略抓取率**: IS {cap_rate_is:.0f}%（{8.5:.1f}/{is_p2lm:.1f}pt），OOS后半 {cap_rate_oos2:.0f}%（{3.3:.1f}/{oos2_p2lm:.1f}pt）")
    doc.append(f"3. **入场滞后损失**: {is_lag:.0f}pt（{is_lag/is_p2lm*100:.0f}%的理论空间）")

    if oos2_p2lm < is_p2lm * 0.7:
        doc.append(f"4. **衰退根因**: 市场结构变化——OOS后半回落空间缩小{shrink_pct:.0f}%")
        doc.append(f"5. **建议**: 搁置策略，等待市场环境恢复（回落空间>={is_p2lm:.0f}pt时重新启用）")
    elif cap_rate_is < 30:
        doc.append(f"4. **衰退根因**: 策略时机问题——理论空间足够但只抓到{cap_rate_is:.0f}%")
        doc.append(f"5. **改进方向**: ① 更早入场（减少3bar等待）② 更宽目标位 ③ 分段止盈")
    else:
        doc.append(f"4. **衰退根因**: 需综合以上数据判断")
        doc.append(f"5. **建议**: 根据以上具体数字制定下一步")

    report = "\n".join(doc)
    path = Path("tmp") / "rhythm_swing_drawdown_fundamental_analysis.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
