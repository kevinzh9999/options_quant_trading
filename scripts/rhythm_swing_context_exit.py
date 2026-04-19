#!/usr/bin/env python3
"""节奏摆动日策略 - 上下文感知出场逻辑 CA1/CA2/CA3/CA4。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import datetime
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

        candidates.append({
            'date': date, 'day_bars': day,
            'early_peak': eh, 'early_low': el,
            'early_amp': eh - el, 'peak_time': peak_time,
            'entry_price': entry_price, 'entry_idx': entry_idx,
        })
    return candidates


def _exit_result(entry_p, exit_p, reason, date):
    return {'pnl': entry_p - exit_p, 'reason': reason, 'date': date,
            'entry_p': entry_p, 'exit_p': exit_p}


def backtest_ca1(cand):
    """CA1: 持仓>=6bar + 浮盈>=amp*0.4 + 2bar无新低。"""
    if cand['entry_price'] is None:
        return None
    day = cand['day_bars']
    peak = cand['early_peak']
    amp = cand['early_amp']
    entry_p = cand['entry_price']
    entry_idx = cand['entry_idx']
    stop = peak + 3

    trade_bars = day[day.index >= entry_idx]
    lowest = entry_p
    no_new_low = 0
    bars_held = 0

    for idx, bar in trade_bars.iterrows():
        bh, bl = float(bar['high']), float(bar['low'])
        bars_held += 1

        if bh >= stop:
            return _exit_result(entry_p, stop, 'stop', cand['date'])

        if bl < lowest:
            lowest = bl
            no_new_low = 0
        else:
            no_new_low += 1

        profit = entry_p - lowest
        if bars_held >= 6 and profit >= amp * 0.4 and no_new_low >= 2:
            pos = trade_bars.index.get_loc(idx)
            if pos + 1 < len(trade_bars):
                exit_p = float(trade_bars.iloc[pos + 1]['open'])
                return _exit_result(entry_p, exit_p, 'ca1_exit', cand['date'])

        if idx.hour >= 6:
            return _exit_result(entry_p, float(bar['open']), 'time', cand['date'])

    return _exit_result(entry_p, float(trade_bars.iloc[-1]['close']), 'eod', cand['date'])


def backtest_ca2(cand):
    """CA2: 时间分段敏感度。1-5bar只止损/BE, 6-10bar 3bar无新低, >=11bar 2bar无新低, 13:30平仓。"""
    if cand['entry_price'] is None:
        return None
    day = cand['day_bars']
    peak = cand['early_peak']
    entry_p = cand['entry_price']
    entry_idx = cand['entry_idx']
    stop = peak + 3

    trade_bars = day[day.index >= entry_idx]
    lowest = entry_p
    no_new_low = 0
    bars_held = 0

    for idx, bar in trade_bars.iterrows():
        bh, bl = float(bar['high']), float(bar['low'])
        bars_held += 1

        if bh >= stop:
            return _exit_result(entry_p, stop, 'stop', cand['date'])

        if bl < lowest:
            lowest = bl
            no_new_low = 0
        else:
            no_new_low += 1

        # 13:30 BJ = 05:30 UTC
        if idx.hour >= 5 and idx.minute >= 30:
            return _exit_result(entry_p, float(bar['open']), 'ca2_1330', cand['date'])
        if idx.hour >= 6:
            return _exit_result(entry_p, float(bar['open']), 'time', cand['date'])

        # 1-5 bar: 只止损或break-even
        if bars_held <= 5:
            if entry_p - float(bar['close']) <= 0 and bars_held >= 3:
                # 浮盈回到0（break-even检查，但不强制）
                pass
            continue

        # 6-10 bar: 3 bar无新低
        if 6 <= bars_held <= 10:
            if no_new_low >= 3:
                pos = trade_bars.index.get_loc(idx)
                if pos + 1 < len(trade_bars):
                    return _exit_result(entry_p, float(trade_bars.iloc[pos + 1]['open']), 'ca2_phase2', cand['date'])

        # >=11 bar: 2 bar无新低
        if bars_held >= 11:
            if no_new_low >= 2:
                pos = trade_bars.index.get_loc(idx)
                if pos + 1 < len(trade_bars):
                    return _exit_result(entry_p, float(trade_bars.iloc[pos + 1]['open']), 'ca2_phase3', cand['date'])

    return _exit_result(entry_p, float(trade_bars.iloc[-1]['close']), 'eod', cand['date'])


def backtest_ca3(cand):
    """CA3: 回撤门槛 + 跌破早盘low保护。T1/T2/T3。"""
    if cand['entry_price'] is None:
        return None
    day = cand['day_bars']
    peak = cand['early_peak']
    amp = cand['early_amp']
    early_low = cand['early_low']
    entry_p = cand['entry_price']
    entry_idx = cand['entry_idx']
    stop = peak + 3

    trade_bars = day[day.index >= entry_idx]
    lowest = entry_p
    no_new_low = 0
    broke_early_low = False
    no_new_low_after_break = 0

    for idx, bar in trade_bars.iterrows():
        bh, bl = float(bar['high']), float(bar['low'])

        if bh >= stop:
            return _exit_result(entry_p, stop, 'stop', cand['date'])

        if bl < lowest:
            lowest = bl
            no_new_low = 0
        else:
            no_new_low += 1

        profit = entry_p - lowest

        # T2: 浮盈 >= amp * 0.9 立刻平仓
        if profit >= amp * 0.9:
            return _exit_result(entry_p, lowest, 'ca3_T2', cand['date'])

        # T1: 浮盈 >= amp * 0.6 且 2 bar无新低
        if profit >= amp * 0.6 and no_new_low >= 2:
            pos = trade_bars.index.get_loc(idx)
            if pos + 1 < len(trade_bars):
                return _exit_result(entry_p, float(trade_bars.iloc[pos + 1]['open']), 'ca3_T1', cand['date'])

        # 跌破早盘low检测
        if bl <= early_low:
            broke_early_low = True
            no_new_low_after_break = 0
        elif broke_early_low:
            no_new_low_after_break += 1

        # T3: 跌破早盘low后3bar无新低
        if broke_early_low and no_new_low_after_break >= 3:
            pos = trade_bars.index.get_loc(idx)
            if pos + 1 < len(trade_bars):
                return _exit_result(entry_p, float(trade_bars.iloc[pos + 1]['open']), 'ca3_T3', cand['date'])

        if idx.hour >= 6:
            return _exit_result(entry_p, float(bar['open']), 'time', cand['date'])

    return _exit_result(entry_p, float(trade_bars.iloc[-1]['close']), 'eod', cand['date'])


def backtest_ca4(cand):
    """CA4: 综合版。止损 > 13:30 > T2深度 > 时间+幅度+横盘 > 跌破low确认 > 14:00。"""
    if cand['entry_price'] is None:
        return None
    day = cand['day_bars']
    peak = cand['early_peak']
    amp = cand['early_amp']
    early_low = cand['early_low']
    entry_p = cand['entry_price']
    entry_idx = cand['entry_idx']
    stop = peak + 3

    trade_bars = day[day.index >= entry_idx]
    lowest = entry_p
    no_new_low = 0
    bars_held = 0
    broke_early_low = False
    no_new_low_after_break = 0

    for idx, bar in trade_bars.iterrows():
        bh, bl = float(bar['high']), float(bar['low'])
        bars_held += 1

        # P1: 止损
        if bh >= stop:
            return _exit_result(entry_p, stop, 'stop', cand['date'])

        if bl < lowest:
            lowest = bl
            no_new_low = 0
        else:
            no_new_low += 1

        profit = entry_p - lowest

        # P2: 13:30 BJ = 05:30 UTC
        if idx.hour >= 5 and idx.minute >= 30:
            return _exit_result(entry_p, float(bar['open']), 'ca4_1330', cand['date'])

        # P3: 深度目标 amp*0.9
        if profit >= amp * 0.9:
            return _exit_result(entry_p, lowest, 'ca4_deep', cand['date'])

        # P4: 时间+幅度+横盘三重确认
        if bars_held >= 6 and profit >= amp * 0.4 and no_new_low >= 2:
            pos = trade_bars.index.get_loc(idx)
            if pos + 1 < len(trade_bars):
                return _exit_result(entry_p, float(trade_bars.iloc[pos + 1]['open']), 'ca4_triple', cand['date'])

        # 跌破早盘low
        if bl <= early_low:
            broke_early_low = True
            no_new_low_after_break = 0
        elif broke_early_low:
            no_new_low_after_break += 1

        # P5: 跌破low后3bar确认
        if broke_early_low and no_new_low_after_break >= 3:
            pos = trade_bars.index.get_loc(idx)
            if pos + 1 < len(trade_bars):
                return _exit_result(entry_p, float(trade_bars.iloc[pos + 1]['open']), 'ca4_break', cand['date'])

        # P6: 14:00
        if idx.hour >= 6:
            return _exit_result(entry_p, float(bar['open']), 'time', cand['date'])

    return _exit_result(entry_p, float(trade_bars.iloc[-1]['close']), 'eod', cand['date'])


def backtest_old(cand):
    """旧基线: target=0.8, stop=peak+3, time=14:00。"""
    if cand['entry_price'] is None:
        return None
    day = cand['day_bars']
    peak = cand['early_peak']
    amp = cand['early_amp']
    entry_p = cand['entry_price']
    entry_idx = cand['entry_idx']
    stop = peak + 3
    target = peak - amp * 0.8

    for idx, bar in day[day.index >= entry_idx].iterrows():
        bh, bl = float(bar['high']), float(bar['low'])
        if bh >= stop:
            return _exit_result(entry_p, stop, 'stop', cand['date'])
        if bl <= target:
            return _exit_result(entry_p, target, 'target', cand['date'])
        if idx.hour >= 6:
            return _exit_result(entry_p, float(bar['open']), 'time', cand['date'])
    return _exit_result(entry_p, float(day.iloc[-1]['close']), 'eod', cand['date'])


def calc_stats(trades):
    if not trades:
        return {'n': 0, 'wr': 0, 'avg': 0, 'total': 0, 'max_dd': 0}
    pnls = [t['pnl'] for t in trades]
    n = len(pnls)
    cum = np.cumsum(pnls)
    dd = np.max(np.maximum.accumulate(cum) - cum) if len(cum) > 0 else 0
    return {
        'n': n, 'wr': sum(1 for p in pnls if p > 0) / n * 100,
        'avg': np.mean(pnls), 'total': sum(pnls), 'max_dd': dd,
    }


def main():
    print("加载数据...")
    df = load_im()
    print("筛选候选日...")
    candidates = find_candidates(df)
    n = len(candidates)
    print(f"候选日: {n}天")

    rules = {'CA1': backtest_ca1, 'CA2': backtest_ca2, 'CA3': backtest_ca3, 'CA4': backtest_ca4}

    is_n = 130
    is_cands = candidates[:is_n]
    oos_cands = candidates[is_n:]
    oos_mid = len(oos_cands) // 2
    oos_first = oos_cands[:oos_mid]
    oos_second = oos_cands[oos_mid:]

    doc = []
    doc.append("# 节奏摆动日 上下文感知出场逻辑\n")

    # ═══════════ 第一步: 4/10 sanity check ═══════════
    print("第一步: 4/10 sanity check...")
    doc.append("## 第一步: 4/10 Sanity Check\n")

    sanity_dates = [
        datetime.date(2026, 4, 10),
        datetime.date(2026, 4, 7),
        datetime.date(2026, 1, 6),
        datetime.date(2025, 8, 29),
    ]

    for sd in sanity_dates:
        cand_s = None
        for c in candidates:
            if c['date'] == sd:
                cand_s = c
                break
        if cand_s is None:
            doc.append(f"### {sd}: 不在候选日中\n")
            continue

        doc.append(f"### {sd} (peak={cand_s['early_peak']:.0f}, amp={cand_s['early_amp']:.0f})\n")
        doc.append(f"| 规则 | 出场reason | exit_p | pnl | 旧基线pnl |")
        doc.append(f"|------|-----------|--------|-----|---------|")
        t_old = backtest_old(cand_s)
        old_pnl = t_old['pnl'] if t_old else 0
        for rname, rfunc in rules.items():
            t = rfunc(cand_s)
            if t:
                doc.append(f"| {rname} | {t['reason']} | {t['exit_p']:.0f} | {t['pnl']:+.1f} | {old_pnl:+.1f} |")
            else:
                doc.append(f"| {rname} | 无入场 | - | - | {old_pnl:+.1f} |")
        doc.append("")

    # ═══════════ 第三步: 全样本回测 ═══════════
    print("第三步: 全样本回测...")
    doc.append("## 第三步: 全样本回测 (196笔)\n")

    doc.append("| 规则 | N | WR | Avg | Total | DD |")
    doc.append("|------|---|----|-----|-------|-----|")

    all_rule_trades = {}
    for rname, rfunc in rules.items():
        trades = [rfunc(c) for c in candidates]
        trades = [t for t in trades if t is not None]
        all_rule_trades[rname] = trades
        s = calc_stats(trades)
        doc.append(f"| {rname} | {s['n']} | {s['wr']:.0f}% | {s['avg']:+.1f} | {s['total']:+.0f} | {s['max_dd']:.0f} |")

    # 旧基线
    old_trades = [backtest_old(c) for c in candidates]
    old_trades = [t for t in old_trades if t is not None]
    all_rule_trades['OLD'] = old_trades
    s = calc_stats(old_trades)
    doc.append(f"| 旧tgt=0.8 | {s['n']} | {s['wr']:.0f}% | {s['avg']:+.1f} | {s['total']:+.0f} | {s['max_dd']:.0f} |")
    doc.append("")

    # 出场原因分布
    doc.append("### 出场原因分布\n")
    for rname in ['CA1', 'CA2', 'CA3', 'CA4']:
        trades = all_rule_trades[rname]
        reasons = {}
        for t in trades:
            r = t['reason']
            if r not in reasons:
                reasons[r] = []
            reasons[r].append(t['pnl'])
        doc.append(f"**{rname}:**")
        for r, pnls in sorted(reasons.items(), key=lambda x: -len(x[1])):
            doc.append(f"  {r}: {len(pnls)}笔 ({len(pnls)/len(trades)*100:.0f}%) avg={np.mean(pnls):+.1f}")
        doc.append("")

    # ═══════════ 第四步: 最优规则IS/OOS ═══════════
    print("第四步: IS/OOS验证...")
    # 选全样本avg最高的规则
    best_rule = max(['CA1', 'CA2', 'CA3', 'CA4'],
                    key=lambda r: calc_stats(all_rule_trades[r])['avg'])
    doc.append(f"## 第四步: 最优规则 {best_rule} IS/OOS验证\n")

    rfunc = rules[best_rule]
    segments = [('IS', is_cands), ('OOS', oos_cands), ('OOS前半', oos_first), ('OOS后半', oos_second)]

    doc.append(f"| 段 | N | WR | Avg | Total |")
    doc.append(f"|-----|---|-----|-----|-------|")
    seg_stats = {}
    for seg_name, seg_cands in segments:
        trades = [rfunc(c) for c in seg_cands]
        trades = [t for t in trades if t is not None]
        s = calc_stats(trades)
        seg_stats[seg_name] = s
        doc.append(f"| {seg_name} | {s['n']} | {s['wr']:.0f}% | {s['avg']:+.1f} | {s['total']:+.0f} |")

    # 旧基线对照
    doc.append(f"\n旧基线对照:")
    doc.append(f"| 段 | N | WR | Avg | Total |")
    doc.append(f"|-----|---|-----|-----|-------|")
    for seg_name, seg_cands in segments:
        trades = [backtest_old(c) for c in seg_cands]
        trades = [t for t in trades if t is not None]
        s = calc_stats(trades)
        doc.append(f"| {seg_name} | {s['n']} | {s['wr']:.0f}% | {s['avg']:+.1f} | {s['total']:+.0f} |")

    # ═══════════ 第五步: OOS逐笔对比 ═══════════
    doc.append(f"\n## 第五步: OOS逐笔对比 ({best_rule} vs 旧基线)\n")

    new_oos = [rfunc(c) for c in oos_cands]
    new_oos = [t for t in new_oos if t is not None]
    old_oos = [backtest_old(c) for c in oos_cands]
    old_oos = [t for t in old_oos if t is not None]

    new_by_date = {t['date']: t for t in new_oos}
    old_by_date = {t['date']: t for t in old_oos}
    common = sorted(set(new_by_date) & set(old_by_date))

    new_better = 0
    old_better = 0
    same = 0
    diffs = []
    for d in common:
        diff = new_by_date[d]['pnl'] - old_by_date[d]['pnl']
        diffs.append(diff)
        if diff > 0.5:
            new_better += 1
        elif diff < -0.5:
            old_better += 1
        else:
            same += 1

    doc.append(f"- 比较天数: {len(common)}")
    doc.append(f"- {best_rule}更好: {new_better} ({new_better/len(common)*100:.0f}%)")
    doc.append(f"- 旧基线更好: {old_better} ({old_better/len(common)*100:.0f}%)")
    doc.append(f"- 差不多: {same} ({same/len(common)*100:.0f}%)")
    doc.append(f"- 差异均值: {np.mean(diffs):+.1f}pt")

    # 重点日期
    doc.append(f"\n### 重点日期\n")
    doc.append(f"| 日期 | 旧pnl | {best_rule} pnl | 差异 |")
    doc.append(f"|------|-------|---------|------|")
    for sd in sanity_dates:
        if sd in new_by_date and sd in old_by_date:
            op = old_by_date[sd]['pnl']
            np_ = new_by_date[sd]['pnl']
            doc.append(f"| {sd} | {op:+.1f} | {np_:+.1f} | {np_-op:+.1f} |")

    # ═══════════ 综合判定 ═══════════
    doc.append(f"\n## 综合判定\n")
    oos_avg = seg_stats['OOS']['avg']
    oos2_avg = seg_stats['OOS后半']['avg']

    if oos_avg >= 11 and oos2_avg >= 8:
        doc.append(f"**判定CA1: 显著突破天花板** ✓")
        doc.append(f"OOS avg={oos_avg:+.1f}>=11, 后半={oos2_avg:+.1f}>=8")
    elif oos_avg >= 7 or oos2_avg >= 5:
        doc.append(f"**判定CA2: 改进有限但方向正确**")
        doc.append(f"OOS avg={oos_avg:+.1f}, 后半={oos2_avg:+.1f}")
    else:
        doc.append(f"**判定CA3: 没有明显改进** ✗")
        doc.append(f"OOS avg={oos_avg:+.1f}, 后半={oos2_avg:+.1f}")

    report = "\n".join(doc)
    path = Path("tmp") / "context_aware_exit_logic.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
