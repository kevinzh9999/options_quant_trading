#!/usr/bin/env python3
"""任务A: 新Target定义IS/OOS验证 + 任务B: Gap对回撤幅度的基本面分析。"""
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
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = df[c].astype(float)
    df.index = pd.to_datetime(df['datetime'])
    df['date'] = df.index.date
    return df


def is_rhythm_swing(day, open_p):
    if len(day) < 10:
        return False
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


def find_first_local_min(day, peak_time):
    after = day[day.index > peak_time]
    if len(after) < 5:
        return float(after['low'].min()) if len(after) > 0 else np.nan
    lows = after['low'].values.astype(float)
    for i in range(2, len(lows) - 2):
        if (lows[i] <= lows[i - 1] and lows[i] <= lows[i - 2] and
                lows[i] <= lows[i + 1] and lows[i] <= lows[i + 2]):
            return lows[i]
    return float(after['low'].min())


def find_candidates(df):
    daily_close = df.groupby('date')['close'].last()
    all_dates = sorted(df['date'].unique())
    candidates = []
    for i, date in enumerate(all_dates):
        day = df[df['date'] == date]
        if len(day) < 10:
            continue
        open_p = float(day.iloc[0]['open'])
        prev_c = float(daily_close.get(all_dates[i - 1], open_p)) if i > 0 else open_p
        if not is_rhythm_swing(day, open_p):
            continue
        gap = (open_p - prev_c) / prev_c * 100 if prev_c > 0 else 0
        if gap < -0.2:
            continue
        early = day[(day.index.hour == 1) | (day.index.hour == 2)]
        if len(early) < 4:
            continue
        eh, el = early['high'].max(), early['low'].min()
        if (eh - el) / open_p * 100 < 0.4:
            continue
        peak_time = early['high'].idxmax()

        # 基线入场
        after = day[day.index > peak_time]
        count = 0
        entry_price = None
        entry_idx = None
        for j, (idx, bar) in enumerate(after.iterrows()):
            if float(bar['high']) <= eh:
                count += 1
            else:
                count = 0
            if count >= 3:
                remaining = after.iloc[j + 1:]
                if len(remaining) > 0:
                    entry_price = float(remaining.iloc[0]['open'])
                    entry_idx = remaining.index[0]
                break

        first_lm = find_first_local_min(day, peak_time)

        candidates.append({
            'date': date, 'day_bars': day,
            'early_peak': eh, 'early_low': el,
            'early_amp': eh - el, 'peak_time': peak_time,
            'entry_price': entry_price, 'entry_idx': entry_idx,
            'open_price': open_p, 'prev_close': prev_c,
            'gap_pct': gap, 'gap_pts': open_p - prev_c,
            'day_low': float(day['low'].min()),
            'day_close': float(day.iloc[-1]['close']),
            'first_lm': first_lm,
        })
    return candidates


def backtest_new_target(cand, target_pct):
    """新定义: target = entry_price - amp * target_pct。"""
    if cand['entry_price'] is None:
        return None
    day = cand['day_bars']
    peak = cand['early_peak']
    amp = cand['early_amp']
    entry_p = cand['entry_price']
    entry_idx = cand['entry_idx']
    stop = peak + 3
    target = entry_p - amp * target_pct

    for idx, bar in day[day.index >= entry_idx].iterrows():
        bh, bl = float(bar['high']), float(bar['low'])
        if bh >= stop:
            return {'pnl': entry_p - stop, 'reason': 'stop', 'date': cand['date'],
                    'entry_p': entry_p, 'exit_p': stop}
        if bl <= target:
            return {'pnl': entry_p - target, 'reason': 'target', 'date': cand['date'],
                    'entry_p': entry_p, 'exit_p': target}
        if idx.hour >= 6:
            exit_p = float(bar['open'])
            return {'pnl': entry_p - exit_p, 'reason': 'time', 'date': cand['date'],
                    'entry_p': entry_p, 'exit_p': exit_p}
    last = day.iloc[-1]
    return {'pnl': entry_p - float(last['close']), 'reason': 'eod', 'date': cand['date'],
            'entry_p': entry_p, 'exit_p': float(last['close'])}


def backtest_old_target(cand, target_pct=0.8):
    """旧定义基线: target = peak - amp * target_pct。"""
    if cand['entry_price'] is None:
        return None
    day = cand['day_bars']
    peak = cand['early_peak']
    amp = cand['early_amp']
    entry_p = cand['entry_price']
    entry_idx = cand['entry_idx']
    stop = peak + 3
    target = peak - amp * target_pct

    for idx, bar in day[day.index >= entry_idx].iterrows():
        bh, bl = float(bar['high']), float(bar['low'])
        if bh >= stop:
            return {'pnl': entry_p - stop, 'reason': 'stop', 'date': cand['date'],
                    'entry_p': entry_p, 'exit_p': stop}
        if bl <= target:
            return {'pnl': entry_p - target, 'reason': 'target', 'date': cand['date'],
                    'entry_p': entry_p, 'exit_p': target}
        if idx.hour >= 6:
            exit_p = float(bar['open'])
            return {'pnl': entry_p - exit_p, 'reason': 'time', 'date': cand['date'],
                    'entry_p': entry_p, 'exit_p': exit_p}
    last = day.iloc[-1]
    return {'pnl': entry_p - float(last['close']), 'reason': 'eod', 'date': cand['date'],
            'entry_p': entry_p, 'exit_p': float(last['close'])}


def calc_stats(trades):
    if not trades:
        return {'n': 0, 'wr': 0, 'avg': 0, 'total': 0, 'tgt_rate': 0, 'stop_rate': 0, 'time_rate': 0}
    pnls = [t['pnl'] for t in trades]
    n = len(pnls)
    return {
        'n': n,
        'wr': sum(1 for p in pnls if p > 0) / n * 100,
        'avg': np.mean(pnls),
        'total': sum(pnls),
        'tgt_rate': sum(1 for t in trades if t['reason'] == 'target') / n * 100,
        'stop_rate': sum(1 for t in trades if t['reason'] == 'stop') / n * 100,
        'time_rate': sum(1 for t in trades if t['reason'] in ('time', 'eod')) / n * 100,
    }


def main():
    print("加载数据...")
    df = load_im()
    print("筛选候选日...")
    candidates = find_candidates(df)
    n = len(candidates)
    print(f"候选日: {n}天")

    is_n = 130
    is_cands = candidates[:is_n]
    oos_cands = candidates[is_n:]
    oos_mid = len(oos_cands) // 2
    oos_first = oos_cands[:oos_mid]
    oos_second = oos_cands[oos_mid:]

    doc = ["# 新Target定义IS/OOS验证 + Gap基本面分析\n"]
    doc.append(f"数据: {n}天节奏摆动日(冲高型)")
    doc.append(f"- IS: 前{len(is_cands)}天 ({is_cands[0]['date']}~{is_cands[-1]['date']})")
    doc.append(f"- OOS: 后{len(oos_cands)}天 ({oos_cands[0]['date']}~{oos_cands[-1]['date']})")
    doc.append(f"  - OOS前半: {len(oos_first)}天, OOS后半: {len(oos_second)}天\n")

    # ═══════════════════════════════════════════════
    # Part A: 新Target定义 IS/OOS验证
    # ═══════════════════════════════════════════════
    print("\n=== 任务A: 新Target定义IS/OOS验证 ===")
    doc.append("---\n# Part A: 新Target定义 IS/OOS验证\n")
    doc.append("新定义: target = entry_price - early_amp x target_pct\n")

    target_pcts = [0.3, 0.4, 0.5]
    segments = [
        ('IS', is_cands),
        ('OOS', oos_cands),
        ('OOS前半', oos_first),
        ('OOS后半', oos_second),
    ]

    # 完整表格
    doc.append("## 完整IS/OOS结果\n")
    doc.append("| tgt | 段 | N | WR | Avg | Total | Tgt% | Stp% | Time% |")
    doc.append("|-----|-----|---|-----|-----|-------|------|------|-------|")

    all_stats = {}  # (tgt, seg_name) -> stats
    for tgt in target_pcts:
        for seg_name, seg_cands in segments:
            trades = [backtest_new_target(c, tgt) for c in seg_cands]
            trades = [t for t in trades if t is not None]
            s = calc_stats(trades)
            all_stats[(tgt, seg_name)] = s
            doc.append(f"| {tgt:.1f} | {seg_name} | {s['n']} | {s['wr']:.0f}% | "
                       f"{s['avg']:+.1f} | {s['total']:+.0f} | {s['tgt_rate']:.0f}% | "
                       f"{s['stop_rate']:.0f}% | {s['time_rate']:.0f}% |")

    # 旧定义基线
    doc.append("\n## 旧定义基线(target=0.8)对照\n")
    doc.append("| 段 | N | WR | Avg | Total | Tgt% | Stp% | Time% |")
    doc.append("|-----|---|-----|-----|-------|------|------|-------|")
    old_stats = {}
    for seg_name, seg_cands in segments:
        trades = [backtest_old_target(c, 0.8) for c in seg_cands]
        trades = [t for t in trades if t is not None]
        s = calc_stats(trades)
        old_stats[seg_name] = s
        doc.append(f"| {seg_name} | {s['n']} | {s['wr']:.0f}% | "
                   f"{s['avg']:+.1f} | {s['total']:+.0f} | {s['tgt_rate']:.0f}% | "
                   f"{s['stop_rate']:.0f}% | {s['time_rate']:.0f}% |")

    # 新旧对比
    doc.append("\n## 新旧定义OOS对比\n")
    doc.append("| 参数 | OOS avg | OOS后半 avg | OOS WR | OOS后半 WR |")
    doc.append("|------|---------|------------|--------|-----------|")
    doc.append(f"| 旧tgt=0.8 | {old_stats['OOS']['avg']:+.1f} | {old_stats['OOS后半']['avg']:+.1f} | "
               f"{old_stats['OOS']['wr']:.0f}% | {old_stats['OOS后半']['wr']:.0f}% |")
    for tgt in target_pcts:
        s_oos = all_stats[(tgt, 'OOS')]
        s_oos2 = all_stats[(tgt, 'OOS后半')]
        doc.append(f"| 新tgt={tgt:.1f} | {s_oos['avg']:+.1f} | {s_oos2['avg']:+.1f} | "
                   f"{s_oos['wr']:.0f}% | {s_oos2['wr']:.0f}% |")

    # 判定
    doc.append("\n## Part A 判定\n")
    best_a_tgt = None
    best_a_oos_avg = -999
    for tgt in target_pcts:
        s_oos = all_stats[(tgt, 'OOS')]
        s_oos2 = all_stats[(tgt, 'OOS后半')]
        if s_oos['avg'] >= 7.0 and s_oos2['avg'] >= 5.0 and s_oos['wr'] >= 60:
            if s_oos['avg'] > best_a_oos_avg:
                best_a_oos_avg = s_oos['avg']
                best_a_tgt = tgt

    if best_a_tgt is not None:
        s = all_stats[(best_a_tgt, 'OOS')]
        s2 = all_stats[(best_a_tgt, 'OOS后半')]
        doc.append(f"**判定A1: 新定义显著优于旧基线** ✓")
        doc.append(f"最优: target={best_a_tgt:.1f}")
        doc.append(f"  OOS avg={s['avg']:+.1f}>=7.0, OOS后半avg={s2['avg']:+.1f}>=5.0, OOS WR={s['wr']:.0f}%>=60%")
        doc.append(f"  vs 旧基线OOS: avg={old_stats['OOS']['avg']:+.1f}, 后半avg={old_stats['OOS后半']['avg']:+.1f}")
    else:
        # 检查A2 vs A3
        any_close = False
        for tgt in target_pcts:
            s_oos = all_stats[(tgt, 'OOS')]
            if abs(s_oos['avg'] - old_stats['OOS']['avg']) < 1.0:
                any_close = True
        if any_close:
            doc.append(f"**判定A3: 新定义和旧定义OOS差不多**")
            doc.append(f"Target定义修复只是代码清洁，不是效果提升。")
        else:
            doc.append(f"**判定A2: 新定义IS好但OOS退化**")
            for tgt in target_pcts:
                s_is = all_stats[(tgt, 'IS')]
                s_oos = all_stats[(tgt, 'OOS')]
                doc.append(f"  tgt={tgt:.1f}: IS avg={s_is['avg']:+.1f}, OOS avg={s_oos['avg']:+.1f}")

    # ═══════════════════════════════════════════════
    # Part B: Gap基本面分析
    # ═══════════════════════════════════════════════
    print("\n=== 任务B: Gap基本面分析 ===")
    doc.append("\n---\n# Part B: Gap对回撤幅度的基本面分析\n")

    # 构建gap数据
    gap_data = []
    for c in candidates:
        peak = c['early_peak']
        extended_amp = peak - min(c['early_low'], c['prev_close'])
        peak_to_prev = peak - c['prev_close']
        first_lm = c['first_lm']
        lm_vs_prev = first_lm - c['prev_close'] if not np.isnan(first_lm) else np.nan

        # 旧定义基线策略表现
        t = backtest_old_target(c, 0.8)

        gap_data.append({
            'date': c['date'],
            'gap_pct': c['gap_pct'],
            'gap_pts': c['gap_pts'],
            'early_amp': c['early_amp'],
            'extended_amp': extended_amp,
            'peak_to_day_low': peak - c['day_low'],
            'peak_to_prev': peak_to_prev,
            'peak_to_close': peak - c['day_close'],
            'first_lm': first_lm,
            'lm_vs_prev': lm_vs_prev,  # 负=低点在前收之下(缺口回补)
            'strategy_pnl': t['pnl'] if t else np.nan,
            'strategy_reason': t['reason'] if t else None,
        })
    gdf = pd.DataFrame(gap_data)

    # 分组
    def gap_group(row):
        g = row['gap_pct']
        if abs(g) <= 0.1:
            return '无gap(<=0.1%)'
        elif g > 0.1 and g <= 0.5:
            return '小gap(0.1-0.5%)'
        elif g > 0.5 and g <= 1.0:
            return '中gap(0.5-1.0%)'
        elif g > 1.0:
            return '大gap(>1.0%)'
        else:
            return '跳低(<-0.1%)'

    gdf['gap_group'] = gdf.apply(gap_group, axis=1)

    # 基本面统计表
    group_order = ['跳低(<-0.1%)', '无gap(<=0.1%)', '小gap(0.1-0.5%)', '中gap(0.5-1.0%)', '大gap(>1.0%)']
    doc.append("## 各Gap组基本面统计（中位数）\n")
    doc.append("| gap组 | N | early_amp | extended_amp | peak->低点 | peak->prev | peak->close | 低点vs前收 |")
    doc.append("|-------|---|-----------|-------------|-----------|-----------|------------|----------|")

    for grp in group_order:
        sub = gdf[gdf['gap_group'] == grp]
        if len(sub) < 2:
            continue
        doc.append(f"| {grp} | {len(sub)} | {sub['early_amp'].median():.1f} | "
                   f"{sub['extended_amp'].median():.1f} | {sub['peak_to_day_low'].median():.1f} | "
                   f"{sub['peak_to_prev'].median():.1f} | {sub['peak_to_close'].median():.1f} | "
                   f"{sub['lm_vs_prev'].median():+.1f} |")
    doc.append("")

    # 问题1: gap越大回撤越大?
    doc.append("## Q1: Gap越大回撤空间越大?\n")
    for grp in group_order:
        sub = gdf[gdf['gap_group'] == grp]
        if len(sub) < 2:
            continue
        doc.append(f"- {grp} (N={len(sub)}): peak->低点中位数={sub['peak_to_day_low'].median():.1f}pt, "
                   f"peak->close中位数={sub['peak_to_close'].median():.1f}pt")
    doc.append("")

    # 问题2: 大gap日低点会不会跌破前收?
    doc.append("## Q2: 大Gap日第一波低点是否跌破前收?\n")
    for grp in group_order:
        sub = gdf[gdf['gap_group'] == grp]
        if len(sub) < 2:
            continue
        below_prev = (sub['lm_vs_prev'] < 0).sum()
        below_pct = below_prev / len(sub) * 100
        doc.append(f"- {grp}: {below_prev}/{len(sub)}天低点跌破前收 ({below_pct:.0f}%)")
    doc.append("")

    # 问题3: extended_amp vs early_amp
    doc.append("## Q3: Extended_amp vs Early_amp\n")
    doc.append("| gap组 | early_amp中位数 | extended_amp中位数 | 差值 | peak->低点中位数 |")
    doc.append("|-------|--------------|-----------------|------|-------------|")
    for grp in group_order:
        sub = gdf[gdf['gap_group'] == grp]
        if len(sub) < 2:
            continue
        ea = sub['early_amp'].median()
        xa = sub['extended_amp'].median()
        p2l = sub['peak_to_day_low'].median()
        doc.append(f"| {grp} | {ea:.1f} | {xa:.1f} | {xa - ea:+.1f} | {p2l:.1f} |")
    doc.append("")

    # 按gap组做策略表现拆解
    doc.append("## 按Gap组的策略表现（旧基线tgt=0.8）\n")
    doc.append("| gap组 | N | WR | Avg | Total |")
    doc.append("|-------|---|----|-----|-------|")
    for grp in group_order:
        sub = gdf[gdf['gap_group'] == grp]
        if len(sub) < 2:
            continue
        pnls = sub['strategy_pnl'].dropna()
        if len(pnls) == 0:
            continue
        wr = (pnls > 0).sum() / len(pnls) * 100
        doc.append(f"| {grp} | {len(pnls)} | {wr:.0f}% | {pnls.mean():+.1f} | {pnls.sum():+.0f} |")
    doc.append("")

    # Part B判定
    doc.append("## Part B 判定\n")
    # 检查各组peak->低点的差异
    group_medians = {}
    for grp in group_order:
        sub = gdf[gdf['gap_group'] == grp]
        if len(sub) >= 2:
            group_medians[grp] = sub['peak_to_day_low'].median()
    if group_medians:
        range_val = max(group_medians.values()) - min(group_medians.values())
    else:
        range_val = 0

    # 检查策略表现差异
    group_avg_pnl = {}
    for grp in group_order:
        sub = gdf[gdf['gap_group'] == grp]
        if len(sub) >= 2:
            pnls = sub['strategy_pnl'].dropna()
            if len(pnls) > 0:
                group_avg_pnl[grp] = pnls.mean()
    if group_avg_pnl:
        pnl_range = max(group_avg_pnl.values()) - min(group_avg_pnl.values())
    else:
        pnl_range = 0

    if range_val >= 15 and pnl_range >= 4:
        doc.append(f"**判定B1: Gap是重要维度** ✓")
        doc.append(f"- peak->低点极差: {range_val:.1f}pt (>= 15)")
        doc.append(f"- 策略表现极差: {pnl_range:.1f}pt (>= 4)")
    elif range_val >= 15 or pnl_range >= 4:
        doc.append(f"**判定B1(部分): Gap有一定影响**")
        doc.append(f"- peak->低点极差: {range_val:.1f}pt")
        doc.append(f"- 策略表现极差: {pnl_range:.1f}pt")
    else:
        doc.append(f"**判定B2: Gap影响有限**")
        doc.append(f"- peak->低点极差: {range_val:.1f}pt (< 15)")
        doc.append(f"- 策略表现极差: {pnl_range:.1f}pt (< 4)")

    # ═══════════════════════════════════════════════
    # 综合建议
    # ═══════════════════════════════════════════════
    doc.append("\n---\n# 综合建议\n")

    a_success = best_a_tgt is not None
    b_success = range_val >= 15 and pnl_range >= 4

    if a_success and b_success:
        doc.append("**A1 + B1**: 新target定义 + gap纳入，做组合优化")
    elif a_success and not b_success:
        doc.append("**A1 + B2**: 新target定义上线，gap暂缓")
    elif not a_success and b_success:
        doc.append("**A2/A3 + B1**: target定义不是关键，gap是新方向")
    else:
        doc.append("**A2/A3 + B2**: 两个方向效果有限")
        doc.append(f"接受旧基线+5.6pt/笔的OOS天花板，或转向其他策略方向")

    report = "\n".join(doc)
    path = Path("tmp") / "new_target_isoos_and_gap_analysis.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
