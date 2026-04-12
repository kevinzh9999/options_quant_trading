#!/usr/bin/env python3
"""Target机制Bug诊断：旧定义(基于peak) vs 新定义(基于entry_price)。"""
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


def backtest_one(cand, target_pct, use_new_target=False):
    """回测单笔。use_new_target=True时target基于entry_price而非peak。"""
    if cand['entry_price'] is None:
        return None

    day = cand['day_bars']
    peak = cand['early_peak']
    amp = cand['early_amp']
    entry_p = cand['entry_price']
    entry_idx = cand['entry_idx']
    stop = peak + 3

    if use_new_target:
        target = entry_p - amp * target_pct
    else:
        target = peak - amp * target_pct

    for idx, bar in day[day.index >= entry_idx].iterrows():
        bh, bl = float(bar['high']), float(bar['low'])
        if bh >= stop:
            return {'pnl': entry_p - stop, 'reason': 'stop', 'date': cand['date'],
                    'entry_p': entry_p, 'exit_p': stop, 'target_price': target}
        if bl <= target:
            return {'pnl': entry_p - target, 'reason': 'target', 'date': cand['date'],
                    'entry_p': entry_p, 'exit_p': target, 'target_price': target}
        if idx.hour >= 6:
            exit_p = float(bar['open'])
            return {'pnl': entry_p - exit_p, 'reason': 'time', 'date': cand['date'],
                    'entry_p': entry_p, 'exit_p': exit_p, 'target_price': target}

    last = day.iloc[-1]
    return {'pnl': entry_p - float(last['close']), 'reason': 'eod', 'date': cand['date'],
            'entry_p': entry_p, 'exit_p': float(last['close']), 'target_price': target}


def main():
    print("加载数据...")
    df = load_im()
    print("筛选候选日...")
    candidates = find_candidates(df)
    n = len(candidates)
    print(f"候选日: {n}天")

    doc = ["# Target 机制 Bug 诊断\n"]

    # ═══════════════════════════════════════════════
    # 第一步：broken 占比统计
    # ═══════════════════════════════════════════════
    doc.append("## 第一步: Broken 占比统计\n")
    doc.append("旧定义: target_price = early_peak - early_amp x target_pct")
    doc.append("broken = 入场价 <= target_price (target在入场价上方，做空触发target=亏钱)\n")

    target_pcts = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

    doc.append("| target_pct | 总笔数 | valid | broken | broken% | broken时avg_loss |")
    doc.append("|-----------|--------|-------|--------|---------|----------------|")

    for tgt in target_pcts:
        valid_n = 0
        broken_n = 0
        broken_losses = []
        for c in candidates:
            if c['entry_price'] is None:
                continue
            target_price = c['early_peak'] - c['early_amp'] * tgt
            if c['entry_price'] > target_price:
                valid_n += 1
            else:
                broken_n += 1
                # 做空入场在entry_p，target在target_price(高于entry_p)
                # 如果触发target，pnl = entry_p - target_price < 0
                broken_losses.append(c['entry_price'] - target_price)
        total = valid_n + broken_n
        broken_pct = broken_n / total * 100 if total > 0 else 0
        avg_loss = np.mean(broken_losses) if broken_losses else 0
        doc.append(f"| {tgt:.1f} | {total} | {valid_n} | {broken_n} | "
                   f"**{broken_pct:.0f}%** | {avg_loss:+.1f}pt |")

    doc.append("")

    # 诊断结论
    # 检查target=0.3的broken率
    broken_03 = 0
    for c in candidates:
        if c['entry_price'] is None:
            continue
        tp = c['early_peak'] - c['early_amp'] * 0.3
        if c['entry_price'] <= tp:
            broken_03 += 1
    total_valid = sum(1 for c in candidates if c['entry_price'] is not None)
    broken_03_pct = broken_03 / total_valid * 100

    if broken_03_pct >= 40:
        doc.append(f"**第一步结论: Target定义确认有Bug** ✓")
        doc.append(f"target=0.3时{broken_03_pct:.0f}%的交易入场价已在target之下，触发target=亏钱。")
    elif broken_03_pct >= 15:
        doc.append(f"**第一步结论: Target定义有部分Bug**")
        doc.append(f"target=0.3时{broken_03_pct:.0f}%的交易broken。")
    else:
        doc.append(f"**第一步结论: Target定义无明显Bug**")
        doc.append(f"target=0.3时broken仅{broken_03_pct:.0f}%。")

    # ═══════════════════════════════════════════════
    # 第二步：新旧定义对比
    # ═══════════════════════════════════════════════
    doc.append(f"\n## 第二步: 新旧定义对比\n")
    doc.append("旧: target = peak - amp x pct")
    doc.append("新: target = entry_price - amp x pct\n")

    doc.append("| tgt | 旧Avg | 新Avg | 旧WR | 新WR | 旧Tgt% | 新Tgt% | 旧Total | 新Total |")
    doc.append("|-----|-------|-------|------|------|--------|--------|---------|---------|")

    best_new_avg = -999
    best_new_tgt = None
    new_results = {}

    for tgt in target_pcts:
        # 旧定义
        old_trades = [backtest_one(c, tgt, use_new_target=False) for c in candidates]
        old_trades = [t for t in old_trades if t is not None]
        # 新定义
        new_trades = [backtest_one(c, tgt, use_new_target=True) for c in candidates]
        new_trades = [t for t in new_trades if t is not None]

        old_pnls = [t['pnl'] for t in old_trades]
        new_pnls = [t['pnl'] for t in new_trades]

        old_avg = np.mean(old_pnls) if old_pnls else 0
        new_avg = np.mean(new_pnls) if new_pnls else 0
        old_wr = sum(1 for p in old_pnls if p > 0) / len(old_pnls) * 100 if old_pnls else 0
        new_wr = sum(1 for p in new_pnls if p > 0) / len(new_pnls) * 100 if new_pnls else 0
        old_tgt_rate = sum(1 for t in old_trades if t['reason'] == 'target') / len(old_trades) * 100 if old_trades else 0
        new_tgt_rate = sum(1 for t in new_trades if t['reason'] == 'target') / len(new_trades) * 100 if new_trades else 0
        old_total = sum(old_pnls)
        new_total = sum(new_pnls)

        doc.append(f"| {tgt:.1f} | {old_avg:+.1f} | {new_avg:+.1f} | "
                   f"{old_wr:.0f}% | {new_wr:.0f}% | "
                   f"{old_tgt_rate:.0f}% | {new_tgt_rate:.0f}% | "
                   f"{old_total:+.0f} | {new_total:+.0f} |")

        new_results[tgt] = {
            'trades': new_trades, 'avg': new_avg, 'wr': new_wr,
            'total': new_total, 'tgt_rate': new_tgt_rate,
        }
        if new_avg > best_new_avg:
            best_new_avg = new_avg
            best_new_tgt = tgt

    doc.append(f"\n**新定义最优: target_pct={best_new_tgt:.1f}, avg={best_new_avg:+.1f}pt**")
    improvement = best_new_avg - 8.5  # vs 旧定义基线(tgt=0.8)
    doc.append(f"vs 旧定义基线(tgt=0.8, avg=+8.5): {improvement:+.1f}pt")

    # ═══════════════════════════════════════════════
    # 第三步：IS/OOS验证（如果改善>=1.5pt）
    # ═══════════════════════════════════════════════
    if improvement >= 1.5:
        doc.append(f"\n## 第三步: IS/OOS验证 (新定义 target={best_new_tgt:.1f})\n")
        doc.append(f"改善{improvement:+.1f}pt >= 1.5pt阈值，执行IS/OOS验证。\n")

        is_n = 130
        is_cands = candidates[:is_n]
        oos_cands = candidates[is_n:]
        oos_mid = len(oos_cands) // 2
        oos_first_cands = oos_cands[:oos_mid]
        oos_second_cands = oos_cands[oos_mid:]

        segments = {
            'IS': is_cands,
            'OOS全部': oos_cands,
            'OOS前半': oos_first_cands,
            'OOS后半': oos_second_cands,
        }

        doc.append(f"| 指标 | IS ({len(is_cands)}笔) | OOS全部 ({len(oos_cands)}笔) | "
                   f"OOS前半 ({len(oos_first_cands)}笔) | OOS后半 ({len(oos_second_cands)}笔) |")
        doc.append(f"|------|-----|-----|-----|-----|")

        seg_stats = {}
        for seg_name, seg_cands in segments.items():
            trades = [backtest_one(c, best_new_tgt, use_new_target=True) for c in seg_cands]
            trades = [t for t in trades if t is not None]
            pnls = [t['pnl'] for t in trades]
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100 if pnls else 0
            avg = np.mean(pnls) if pnls else 0
            total = sum(pnls) if pnls else 0
            tgt_rate = sum(1 for t in trades if t['reason'] == 'target') / len(trades) * 100 if trades else 0
            seg_stats[seg_name] = {'n': len(trades), 'wr': wr, 'avg': avg, 'total': total, 'tgt_rate': tgt_rate}

        for metric, fmt in [('n', '{:.0f}'), ('wr', '{:.0f}%'), ('avg', '{:+.1f}pt'),
                            ('total', '{:+.0f}pt'), ('tgt_rate', '{:.0f}%')]:
            row = f"| {metric} |"
            for seg_name in ['IS', 'OOS全部', 'OOS前半', 'OOS后半']:
                val = seg_stats[seg_name][metric]
                if metric == 'wr' or metric == 'tgt_rate':
                    row += f" {val:.0f}% |"
                elif metric == 'n':
                    row += f" {val:.0f} |"
                elif metric == 'avg':
                    row += f" {val:+.1f}pt |"
                else:
                    row += f" {val:+.0f}pt |"
            doc.append(row)

        # 旧基线对照
        doc.append(f"\n### 旧基线(tgt=0.8, 旧定义)对照\n")
        doc.append(f"| 指标 | IS | OOS全部 | OOS前半 | OOS后半 |")
        doc.append(f"|------|-----|-----|-----|-----|")
        old_seg_stats = {}
        for seg_name, seg_cands in segments.items():
            trades = [backtest_one(c, 0.8, use_new_target=False) for c in seg_cands]
            trades = [t for t in trades if t is not None]
            pnls = [t['pnl'] for t in trades]
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100 if pnls else 0
            avg = np.mean(pnls) if pnls else 0
            total = sum(pnls) if pnls else 0
            old_seg_stats[seg_name] = {'n': len(trades), 'wr': wr, 'avg': avg, 'total': total}

        for metric in ['avg', 'wr', 'total']:
            row = f"| {metric} |"
            for seg_name in ['IS', 'OOS全部', 'OOS前半', 'OOS后半']:
                val = old_seg_stats[seg_name][metric]
                if metric == 'wr':
                    row += f" {val:.0f}% |"
                elif metric == 'avg':
                    row += f" {val:+.1f}pt |"
                else:
                    row += f" {val:+.0f}pt |"
            doc.append(row)

        # 判定
        doc.append(f"\n### IS/OOS验证判定\n")
        oos_avg = seg_stats['OOS全部']['avg']
        oos_wr = seg_stats['OOS全部']['wr']
        oos2_avg = seg_stats['OOS后半']['avg']
        is_avg = seg_stats['IS']['avg']
        old_oos_avg = old_seg_stats['OOS全部']['avg']

        if oos_avg >= 8.0 and oos_wr >= 60 and oos2_avg >= 6.0:
            doc.append(f"**OOS验证通过** ✓")
            doc.append(f"OOS avg={oos_avg:+.1f}>=8.0, WR={oos_wr:.0f}%>=60%, OOS后半={oos2_avg:+.1f}>=6.0")
        else:
            fails = []
            if oos_avg < 8.0: fails.append(f"OOS avg={oos_avg:+.1f}<8.0")
            if oos_wr < 60: fails.append(f"OOS WR={oos_wr:.0f}%<60%")
            if oos2_avg < 6.0: fails.append(f"OOS后半={oos2_avg:+.1f}<6.0")
            doc.append(f"**OOS验证未通过**: {', '.join(fails)}")

        # 改善幅度
        doc.append(f"\n改善幅度:")
        doc.append(f"- IS: 旧{old_seg_stats['IS']['avg']:+.1f} → 新{is_avg:+.1f} ({is_avg - old_seg_stats['IS']['avg']:+.1f})")
        doc.append(f"- OOS: 旧{old_oos_avg:+.1f} → 新{oos_avg:+.1f} ({oos_avg - old_oos_avg:+.1f})")
        doc.append(f"- OOS后半: 旧{old_seg_stats['OOS后半']['avg']:+.1f} → 新{oos2_avg:+.1f} ({oos2_avg - old_seg_stats['OOS后半']['avg']:+.1f})")

    else:
        doc.append(f"\n## 第三步: 跳过\n")
        doc.append(f"改善{improvement:+.1f}pt < 1.5pt阈值，不做IS/OOS验证。")

    # ═══════════════════════════════════════════════
    # 综合判定
    # ═══════════════════════════════════════════════
    doc.append(f"\n## 综合判定\n")

    if broken_03_pct >= 40 and improvement >= 2.0:
        doc.append(f"**判定X1: Target定义确认有Bug，新定义显著改善** ✓")
        doc.append(f"- broken率: target=0.3时{broken_03_pct:.0f}%")
        doc.append(f"- 新定义最优: target={best_new_tgt:.1f}, 全样本avg={best_new_avg:+.1f} (vs旧+8.5)")
        if improvement >= 1.5:
            doc.append(f"- IS/OOS验证结果见上方")
    elif broken_03_pct >= 15:
        doc.append(f"**判定X2: Target定义有Bug，但新定义改善有限**")
        doc.append(f"- broken率: target=0.3时{broken_03_pct:.0f}%")
        doc.append(f"- 新定义最优改善仅{improvement:+.1f}pt")
        doc.append(f"- Bug是真的，但不是策略效果不好的根本原因")
    else:
        doc.append(f"**判定X3: Target定义无明显Bug**")
        doc.append(f"- broken率低于预期")

    # 方法论教训
    doc.append(f"\n## 方法论教训\n")
    doc.append("**先假设是Bug，再假设是市场。**")
    doc.append("任何反直觉的实证结果（如target越紧越差），应先检查代码逻辑/定义是否有误，")
    doc.append("而非直接寻找复杂的市场解释。Occam's razor适用于量化研究。")

    report = "\n".join(doc)
    path = Path("tmp") / "target_mechanism_bug_diagnostic.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
