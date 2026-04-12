#!/usr/bin/env python3
"""节奏摆动日策略：研究A(出场优化) + 研究B(早入场基本面分析)。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np, pandas as pd
from pathlib import Path
from itertools import product
from data.storage.db_manager import get_db


# ═══════════════════════════════════════════════
# 数据加载 & 候选日筛选（复用现有逻辑）
# ═══════════════════════════════════════════════

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
        candidates.append({
            'date': date, 'day_bars': day,
            'early_peak': eh, 'early_low': el,
            'early_amp': eh - el, 'peak_time': peak_time,
        })
    return candidates


def find_first_local_min(day, peak_time):
    """peak之后的第一个5-bar局部最低点。"""
    after = day[day.index > peak_time]
    if len(after) < 5:
        return float(after['low'].min()) if len(after) > 0 else np.nan
    lows = after['low'].values.astype(float)
    for i in range(2, len(lows) - 2):
        if (lows[i] <= lows[i - 1] and lows[i] <= lows[i - 2] and
                lows[i] <= lows[i + 1] and lows[i] <= lows[i + 2]):
            return lows[i]
    return float(after['low'].min())


# ═══════════════════════════════════════════════
# 研究 A：出场优化
# ═══════════════════════════════════════════════

def backtest_exit(cand, target_pct, time_close_utc_h):
    """固定入场(基线)，变化出场参数。

    time_close_utc_h: UTC小时数，对应BJ时间:
      11:30 BJ = 03:30 UTC → h=3, m=30
      13:00 BJ = 05:00 UTC → h=5
      14:00 BJ = 06:00 UTC → h=6
    """
    day = cand['day_bars']
    peak = cand['early_peak']
    low = cand['early_low']
    pt = cand['peak_time']
    stop = peak + 3
    target = peak - (peak - low) * target_pct

    # 基线入场：3 bar无新高后下一根bar open
    after = day[day.index > pt]
    count = 0
    entry_bar = None
    entry_idx = None
    for j, (idx, bar) in enumerate(after.iterrows()):
        if float(bar['high']) <= peak:
            count += 1
        else:
            count = 0
        if count >= 3:
            remaining = after.iloc[j + 1:]
            if len(remaining) > 0:
                entry_bar = remaining.iloc[0]
                entry_idx = remaining.index[0]
            break

    if entry_bar is None:
        return None

    entry_p = float(entry_bar['open'])

    for idx, bar in day[day.index >= entry_idx].iterrows():
        bh, bl = float(bar['high']), float(bar['low'])
        if bh >= stop:
            return {'pnl': entry_p - stop, 'reason': 'stop', 'date': cand['date'],
                    'entry_p': entry_p, 'exit_p': stop}
        if bl <= target:
            return {'pnl': entry_p - target, 'reason': 'target', 'date': cand['date'],
                    'entry_p': entry_p, 'exit_p': target}
        # 时间平仓检查
        if time_close_utc_h == 3:
            # 11:30 BJ = 03:30 UTC
            if idx.hour > 3 or (idx.hour == 3 and idx.minute >= 30):
                exit_p = float(bar['open'])
                return {'pnl': entry_p - exit_p, 'reason': 'time', 'date': cand['date'],
                        'entry_p': entry_p, 'exit_p': exit_p}
        else:
            if idx.hour >= time_close_utc_h:
                exit_p = float(bar['open'])
                return {'pnl': entry_p - exit_p, 'reason': 'time', 'date': cand['date'],
                        'entry_p': entry_p, 'exit_p': exit_p}

    last = day.iloc[-1]
    return {'pnl': entry_p - float(last['close']), 'reason': 'eod', 'date': cand['date'],
            'entry_p': entry_p, 'exit_p': float(last['close'])}


def run_exit_sweep(candidates, target_pcts, time_closes):
    """研究A: 12组合扫描。"""
    # time_close映射: BJ → UTC hour
    tc_map = {'11:30': 3, '13:00': 5, '14:00': 6}

    results = []
    for tgt, tc_bj in product(target_pcts, time_closes):
        tc_utc = tc_map[tc_bj]
        trades = []
        for c in candidates:
            t = backtest_exit(c, tgt, tc_utc)
            if t is not None:
                trades.append(t)
        if not trades:
            results.append({'target': tgt, 'time_close': tc_bj,
                            'n': 0, 'wr': 0, 'avg': 0, 'total': 0,
                            'max_dd': 0, 'pf': 0, 'tgt_rate': 0,
                            'stop_rate': 0, 'time_rate': 0})
            continue
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        cum = np.cumsum(pnls)
        dd = np.max(np.maximum.accumulate(cum) - cum) if len(cum) > 0 else 0
        pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 99
        tgt_n = sum(1 for t in trades if t['reason'] == 'target')
        stp_n = sum(1 for t in trades if t['reason'] == 'stop')
        tm_n = sum(1 for t in trades if t['reason'] == 'time')
        results.append({
            'target': tgt, 'time_close': tc_bj,
            'n': len(trades), 'wr': len(wins) / len(trades) * 100,
            'avg': np.mean(pnls), 'total': sum(pnls),
            'max_dd': dd, 'pf': pf,
            'tgt_rate': tgt_n / len(trades) * 100,
            'stop_rate': stp_n / len(trades) * 100,
            'time_rate': tm_n / len(trades) * 100,
        })
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════
# 研究 B：早入场基本面分析
# ═══════════════════════════════════════════════

def compute_early_entry_points(cand):
    """计算5个入场时机的价格和空间。"""
    day = cand['day_bars']
    peak = cand['early_peak']
    pt = cand['peak_time']
    stop = peak + 3

    after = day[day.index > pt]
    first_lm = find_first_local_min(day, pt)

    # 点0: peak bar的收盘价
    peak_bar = day[day.index == pt]
    p0_price = float(peak_bar.iloc[0]['close']) if len(peak_bar) > 0 else peak

    # 点1/2/3: peak之后第1/2/3根bar的开盘价
    points = {'p0': p0_price}
    for k in range(1, 4):
        if k <= len(after):
            points[f'p{k}'] = float(after.iloc[k - 1]['open'])
        else:
            points[f'p{k}'] = np.nan

    # 点N: 基线入场（3 bar无新高后下一根bar open）
    count = 0
    pn_price = np.nan
    for j, (idx, bar) in enumerate(after.iterrows()):
        if float(bar['high']) <= peak:
            count += 1
        else:
            count = 0
        if count >= 3:
            remaining = after.iloc[j + 1:]
            if len(remaining) > 0:
                pn_price = float(remaining.iloc[0]['open'])
            break
    points['pN'] = pn_price

    # 计算每个入场点到first_local_min的空间
    spaces = {}
    for key, price in points.items():
        if np.isnan(price) or np.isnan(first_lm):
            spaces[key] = np.nan
        else:
            spaces[key] = price - first_lm

    # 检查每个入场点是否会被止损（入场后价格是否突破peak+3）
    stopped = {}
    for key, price in points.items():
        if np.isnan(price):
            stopped[key] = np.nan
            continue
        # 确定入场bar的index
        if key == 'p0':
            entry_idx = pt
        elif key == 'pN':
            # 找到pN对应的index
            count2 = 0
            entry_idx = None
            for j, (idx, bar) in enumerate(after.iterrows()):
                if float(bar['high']) <= peak:
                    count2 += 1
                else:
                    count2 = 0
                if count2 >= 3:
                    remaining = after.iloc[j + 1:]
                    if len(remaining) > 0:
                        entry_idx = remaining.index[0]
                    break
            if entry_idx is None:
                stopped[key] = np.nan
                continue
        else:
            k = int(key[1])
            if k <= len(after):
                entry_idx = after.index[k - 1]
            else:
                stopped[key] = np.nan
                continue

        # 从入场bar开始检查是否触发止损
        post_entry = day[day.index >= entry_idx]
        hit_stop = False
        for idx, bar in post_entry.iterrows():
            if float(bar['high']) >= stop:
                hit_stop = True
                break
        stopped[key] = hit_stop

    return {
        'date': cand['date'],
        'peak': peak, 'early_low': cand['early_low'],
        'early_amp': cand['early_amp'],
        'first_lm': first_lm,
        **{f'{k}_price': v for k, v in points.items()},
        **{f'{k}_space': v for k, v in spaces.items()},
        **{f'{k}_stopped': v for k, v in stopped.items()},
    }


# ═══════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════

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

    doc = ["# 节奏摆动日策略: 出场优化 + 早入场基本面分析\n"]
    doc.append(f"数据: {n}天节奏摆动日(冲高型)")
    doc.append(f"- IS: 前{len(is_cands)}天 ({is_cands[0]['date']}~{is_cands[-1]['date']})")
    doc.append(f"- OOS: 后{len(oos_cands)}天 ({oos_cands[0]['date']}~{oos_cands[-1]['date']})")
    doc.append(f"  - OOS前半: {len(oos_first)}天, OOS后半: {len(oos_second)}天\n")

    # ═══════════════════════════════════════════════
    # Part A: 出场优化
    # ═══════════════════════════════════════════════
    print("\n=== 研究A: 出场优化 ===")
    doc.append("---\n# Part A: 出场优化\n")
    doc.append("固定入场(基线: 3bar无新高, stop=peak+3)，优化target_pct和time_close。\n")

    target_pcts = [0.3, 0.4, 0.5, 0.6]
    time_closes = ['11:30', '13:00', '14:00']

    # IS sweep
    print("IS段扫描...")
    is_results = run_exit_sweep(is_cands, target_pcts, time_closes)

    doc.append("## IS段结果 (12组合)\n")
    doc.append(f"| tgt | time | N | WR | Avg | Total | DD | PF | Tgt% | Stp% | Time% |")
    doc.append(f"|-----|------|---|-----|-----|-------|-----|-----|------|------|-------|")
    for _, r in is_results.sort_values('total', ascending=False).iterrows():
        doc.append(f"| {r['target']:.1f} | {r['time_close']} | {int(r['n'])} | "
                   f"{r['wr']:.0f}% | {r['avg']:+.1f} | {r['total']:+.0f} | "
                   f"{r['max_dd']:.0f} | {r['pf']:.1f} | {r['tgt_rate']:.0f}% | "
                   f"{r['stop_rate']:.0f}% | {r['time_rate']:.0f}% |")

    # IS最优选择
    doc.append(f"\n## IS最优选择\n")
    qualified = is_results[
        (is_results['n'] >= 100) &
        (is_results['wr'] >= 65) &
        (is_results['avg'] >= 9.0) &
        (is_results['max_dd'] < 100)
    ].copy()

    if len(qualified) == 0:
        doc.append("**无组合同时满足 N>=100, WR>=65%, Avg>=9.0, DD<100**\n")
        # 展示接近达标的
        near = is_results[is_results['n'] >= 100].copy()
        if len(near) > 0:
            near['score'] = near['total'] * near['wr'] / near['max_dd'].replace(0, 1)
            best_row = near.loc[near['score'].idxmax()]
            doc.append(f"次优（综合评分最高）: target={best_row['target']:.1f}, time={best_row['time_close']}")
            doc.append(f"  N={int(best_row['n'])}, WR={best_row['wr']:.0f}%, Avg={best_row['avg']:+.1f}, "
                       f"Total={best_row['total']:+.0f}, DD={best_row['max_dd']:.0f}, PF={best_row['pf']:.1f}")
            # 说明哪些条件未达标
            fails = []
            if best_row['wr'] < 65: fails.append(f"WR={best_row['wr']:.0f}%<65%")
            if best_row['avg'] < 9.0: fails.append(f"Avg={best_row['avg']:+.1f}<9.0")
            if best_row['max_dd'] >= 100: fails.append(f"DD={best_row['max_dd']:.0f}>=100")
            doc.append(f"  未达标: {', '.join(fails)}")
            best_tgt = best_row['target']
            best_tc = best_row['time_close']
        else:
            doc.append("无有效组合")
            best_tgt, best_tc = 0.5, '14:00'
    else:
        qualified['score'] = qualified['total'] * qualified['wr'] / qualified['max_dd'].replace(0, 1)
        best_row = qualified.loc[qualified['score'].idxmax()]
        best_tgt = best_row['target']
        best_tc = best_row['time_close']
        doc.append(f"**IS最优: target={best_tgt:.1f}, time={best_tc}**")
        doc.append(f"  N={int(best_row['n'])}, WR={best_row['wr']:.0f}%, Avg={best_row['avg']:+.1f}, "
                   f"Total={best_row['total']:+.0f}, DD={best_row['max_dd']:.0f}, PF={best_row['pf']:.1f}")

    # OOS验证
    doc.append(f"\n## OOS验证 (target={best_tgt:.1f}, time={best_tc})\n")
    tc_map = {'11:30': 3, '13:00': 5, '14:00': 6}
    tc_utc = tc_map[best_tc]

    # IS trades
    is_trades = []
    for c in is_cands:
        t = backtest_exit(c, best_tgt, tc_utc)
        if t: is_trades.append(t)

    # OOS trades
    oos_trades = []
    for c in oos_cands:
        t = backtest_exit(c, best_tgt, tc_utc)
        if t: oos_trades.append(t)

    if is_trades and oos_trades:
        is_pnls = [t['pnl'] for t in is_trades]
        oos_pnls = [t['pnl'] for t in oos_trades]
        is_wr = sum(1 for p in is_pnls if p > 0) / len(is_pnls) * 100
        oos_wr = sum(1 for p in oos_pnls if p > 0) / len(oos_pnls) * 100
        is_avg = np.mean(is_pnls)
        oos_avg = np.mean(oos_pnls)
        is_tgt = sum(1 for t in is_trades if t['reason'] == 'target') / len(is_trades) * 100
        oos_tgt = sum(1 for t in oos_trades if t['reason'] == 'target') / len(oos_trades) * 100

        doc.append(f"| 指标 | IS ({len(is_trades)}笔) | OOS ({len(oos_trades)}笔) | 衰减 |")
        doc.append(f"|------|-----|-----|------|")
        doc.append(f"| n_trades | {len(is_trades)} | {len(oos_trades)} | |")
        doc.append(f"| win_rate | {is_wr:.0f}% | {oos_wr:.0f}% | {oos_wr - is_wr:+.0f}% |")
        doc.append(f"| avg_pnl | {is_avg:+.1f}pt | {oos_avg:+.1f}pt | {oos_avg - is_avg:+.1f} |")
        doc.append(f"| total_pnl | {sum(is_pnls):+.0f}pt | {sum(oos_pnls):+.0f}pt | |")
        doc.append(f"| target_rate | {is_tgt:.0f}% | {oos_tgt:.0f}% | {oos_tgt - is_tgt:+.0f}% |")

        # OOS后半段单独检查
        doc.append(f"\n## OOS内部时间分组\n")
        oos_first_trades = []
        for c in oos_first:
            t = backtest_exit(c, best_tgt, tc_utc)
            if t: oos_first_trades.append(t)
        oos_second_trades = []
        for c in oos_second:
            t = backtest_exit(c, best_tgt, tc_utc)
            if t: oos_second_trades.append(t)

        doc.append(f"| 段 | 笔数 | win_rate | avg_pnl | total_pnl |")
        doc.append(f"|----|------|----------|---------|-----------|")
        for label, trades in [("OOS前半", oos_first_trades), ("OOS后半", oos_second_trades)]:
            if trades:
                pnls = [t['pnl'] for t in trades]
                wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
                doc.append(f"| {label} | {len(trades)} | {wr:.0f}% | {np.mean(pnls):+.1f}pt | {sum(pnls):+.0f}pt |")

        # 与旧基线对比
        doc.append(f"\n## 与旧基线(stop=3,tgt=0.8,time=14:00)对比\n")
        old_oos_trades = []
        for c in oos_cands:
            t = backtest_exit(c, 0.8, 6)
            if t: old_oos_trades.append(t)
        if old_oos_trades:
            old_pnls = [t['pnl'] for t in old_oos_trades]
            old_wr = sum(1 for p in old_pnls if p > 0) / len(old_pnls) * 100
            doc.append(f"| 指标 | 旧基线OOS | 新参数OOS | 改善 |")
            doc.append(f"|------|----------|----------|------|")
            doc.append(f"| avg_pnl | {np.mean(old_pnls):+.1f}pt | {oos_avg:+.1f}pt | {oos_avg - np.mean(old_pnls):+.1f} |")
            doc.append(f"| win_rate | {old_wr:.0f}% | {oos_wr:.0f}% | {oos_wr - old_wr:+.0f}% |")
            doc.append(f"| total_pnl | {sum(old_pnls):+.0f}pt | {sum(oos_pnls):+.0f}pt | {sum(oos_pnls) - sum(old_pnls):+.0f} |")

        # 判定
        doc.append(f"\n## 研究A判定\n")
        oos2_avg = np.mean([t['pnl'] for t in oos_second_trades]) if oos_second_trades else 0
        if oos_avg >= 8.0 and oos_wr >= 60 and oos2_avg >= 6.0:
            doc.append(f"**判定A1: 出场优化成功** ✓")
            doc.append(f"OOS avg={oos_avg:+.1f}>=8.0, WR={oos_wr:.0f}%>=60%, OOS后半avg={oos2_avg:+.1f}>=6.0")
        else:
            doc.append(f"**判定A2: 出场优化无效** ✗")
            fails = []
            if oos_avg < 8.0: fails.append(f"OOS avg={oos_avg:+.1f}<8.0")
            if oos_wr < 60: fails.append(f"OOS WR={oos_wr:.0f}%<60%")
            if oos2_avg < 6.0: fails.append(f"OOS后半avg={oos2_avg:+.1f}<6.0")
            doc.append(f"未达标: {', '.join(fails)}")

    # ═══════════════════════════════════════════════
    # Part B: 早入场基本面分析
    # ═══════════════════════════════════════════════
    print("\n=== 研究B: 早入场基本面分析 ===")
    doc.append(f"\n---\n# Part B: 早入场基本面分析\n")
    doc.append("不做策略测试，只分析不同入场时机的理论空间和止损率。\n")

    # 计算所有候选日的早入场数据
    entry_data = []
    for c in candidates:
        row = compute_early_entry_points(c)
        entry_data.append(row)
    edf = pd.DataFrame(entry_data)

    is_edf = edf.iloc[:is_n]
    oos_first_edf = edf.iloc[is_n:is_n + oos_mid]
    oos_second_edf = edf.iloc[is_n + oos_mid:]

    # 空间统计表
    point_keys = ['p0', 'p1', 'p2', 'p3', 'pN']
    point_labels = ['点0(peak收盘)', '点1(peak+1bar)', '点2(peak+2bar)',
                    '点3(peak+3bar)', '点N(基线3bar无新高)']

    doc.append("## 各入场时机到第一波低点的空间\n")
    for seg_label, seg_data in [("IS", is_edf), ("OOS前半", oos_first_edf), ("OOS后半", oos_second_edf)]:
        doc.append(f"### {seg_label} ({len(seg_data)}天)\n")
        doc.append(f"| 入场时机 | 均值 | 中位数 | 25% | 75% |")
        doc.append(f"|---------|------|--------|-----|-----|")
        for key, label in zip(point_keys, point_labels):
            col = f'{key}_space'
            s = seg_data[col].dropna()
            if len(s) == 0:
                doc.append(f"| {label} | N/A | N/A | N/A | N/A |")
            else:
                doc.append(f"| {label} | {s.mean():.1f}pt | {s.median():.1f}pt | "
                           f"{s.quantile(0.25):.1f}pt | {s.quantile(0.75):.1f}pt |")
        doc.append("")

    # 三段中位数对比
    doc.append("### 三段中位数对比\n")
    doc.append(f"| 入场时机 | IS | OOS前半 | OOS后半 |")
    doc.append(f"|---------|-----|---------|---------|")
    for key, label in zip(point_keys, point_labels):
        col = f'{key}_space'
        is_med = is_edf[col].dropna().median()
        oos1_med = oos_first_edf[col].dropna().median()
        oos2_med = oos_second_edf[col].dropna().median()
        doc.append(f"| {label} | {is_med:.1f}pt | {oos1_med:.1f}pt | {oos2_med:.1f}pt |")
    doc.append("")

    # 止损率检查
    doc.append("## 止损率检查\n")
    doc.append("入场后价格是否突破peak+3(全天剩余bar内):\n")
    doc.append(f"| 入场时机 | 总样本 | 被止损 | 未被止损 | 未止损率 | 未止损空间均值 | 未止损空间中位数 |")
    doc.append(f"|---------|-------|-------|---------|---------|-------------|--------------|")

    net_expectations = {}
    for key, label in zip(point_keys, point_labels):
        stop_col = f'{key}_stopped'
        space_col = f'{key}_space'
        valid = edf[edf[stop_col].notna()].copy()
        if len(valid) == 0:
            doc.append(f"| {label} | 0 | - | - | - | - | - |")
            continue
        stopped = valid[valid[stop_col] == True]
        not_stopped = valid[valid[stop_col] == False]
        n_total = len(valid)
        n_stopped = len(stopped)
        n_ok = len(not_stopped)
        ok_rate = n_ok / n_total * 100
        ok_space_mean = not_stopped[space_col].dropna().mean() if n_ok > 0 else 0
        ok_space_med = not_stopped[space_col].dropna().median() if n_ok > 0 else 0
        doc.append(f"| {label} | {n_total} | {n_stopped} | {n_ok} | {ok_rate:.0f}% | "
                   f"{ok_space_mean:.1f}pt | {ok_space_med:.1f}pt |")

        # 净期望
        stop_rate = n_stopped / n_total
        # 止损损失: 对于p0, 入场在peak收盘, 止损在peak+3, 所以损失不一定正好3
        # 简化: 用3pt作为止损固定损失
        net_exp = (1 - stop_rate) * ok_space_mean - stop_rate * 3
        net_expectations[key] = {
            'label': label, 'stop_rate': stop_rate * 100,
            'space_mean': ok_space_mean, 'net_exp': net_exp,
        }

    # 净期望估算
    doc.append(f"\n## 净期望估算\n")
    doc.append(f"expected = (1-stop_rate) x avg_space - stop_rate x 3pt\n")
    doc.append(f"| 入场时机 | 止损率 | 空间均值 | 净期望 | vs 基线 |")
    doc.append(f"|---------|-------|---------|-------|--------|")
    baseline_exp = net_expectations.get('pN', {}).get('net_exp', 0)
    for key, label in zip(point_keys, point_labels):
        if key in net_expectations:
            ne = net_expectations[key]
            diff = ne['net_exp'] - baseline_exp
            doc.append(f"| {label} | {ne['stop_rate']:.0f}% | {ne['space_mean']:.1f}pt | "
                       f"{ne['net_exp']:+.1f}pt | {diff:+.1f} |")

    # 研究B判定
    doc.append(f"\n## 研究B判定\n")
    best_early = None
    best_early_exp = baseline_exp
    for key in ['p0', 'p1', 'p2', 'p3']:
        if key in net_expectations:
            ne = net_expectations[key]
            ok_rate = 100 - ne['stop_rate']
            if ok_rate >= 65 and ne['space_mean'] >= 25 and ne['net_exp'] > best_early_exp:
                best_early = key
                best_early_exp = ne['net_exp']

    if best_early is not None:
        ne = net_expectations[best_early]
        doc.append(f"**判定B1: 早入场显著优于基线** ✓")
        doc.append(f"最优时机: {ne['label']}")
        doc.append(f"  未止损率={100 - ne['stop_rate']:.0f}%, 空间均值={ne['space_mean']:.1f}pt, "
                   f"净期望={ne['net_exp']:+.1f}pt (基线{baseline_exp:+.1f}pt)")
    else:
        doc.append(f"**判定B2: 早入场不可行** ✗")
        # 解释原因
        for key in ['p0', 'p1', 'p2', 'p3']:
            if key in net_expectations:
                ne = net_expectations[key]
                ok_rate = 100 - ne['stop_rate']
                reasons = []
                if ok_rate < 65: reasons.append(f"未止损率{ok_rate:.0f}%<65%")
                if ne['space_mean'] < 25: reasons.append(f"空间{ne['space_mean']:.1f}<25pt")
                if ne['net_exp'] <= baseline_exp: reasons.append(f"净期望{ne['net_exp']:+.1f}<=基线{baseline_exp:+.1f}")
                doc.append(f"  {ne['label']}: {', '.join(reasons)}")

    # ═══════════════════════════════════════════════
    # 综合判断
    # ═══════════════════════════════════════════════
    doc.append(f"\n---\n# 综合判断\n")

    # 研究A改善
    if oos_trades:
        old_avg = np.mean([t['pnl'] for t in old_oos_trades]) if old_oos_trades else 5.6
        doc.append(f"## 研究A: 出场优化\n")
        doc.append(f"- 旧基线OOS: {old_avg:+.1f}pt/笔")
        doc.append(f"- 新参数OOS: {oos_avg:+.1f}pt/笔")
        doc.append(f"- 改善: {oos_avg - old_avg:+.1f}pt/笔")

    doc.append(f"\n## 研究B: 早入场\n")
    doc.append(f"- 基线净期望: {baseline_exp:+.1f}pt/笔")
    if best_early:
        doc.append(f"- 最优早入场净期望: {best_early_exp:+.1f}pt/笔")
        doc.append(f"- 改善: {best_early_exp - baseline_exp:+.1f}pt/笔")
    else:
        doc.append(f"- 早入场不可行（止损率过高或空间不足）")

    doc.append(f"\n## 下一步建议\n")
    a_success = oos_trades and oos_avg >= 8.0 and oos_wr >= 60
    b_success = best_early is not None

    if a_success and b_success:
        doc.append("**情况3: A和B都成功** — 两个改进方向都有价值")
        doc.append("下一步: 谨慎考虑组合优化（早入场+紧target），但需严格OOS验证")
    elif a_success and not b_success:
        doc.append("**情况1: A成功，B失败** — 改进target就够了")
        doc.append("下一步: 用新target参数推进regime实时判定研究")
    elif not a_success and b_success:
        doc.append("**情况2: A失败，B成功** — 改进入场时机更关键")
        doc.append("下一步: 完整回测验证早入场策略")
    else:
        doc.append("**情况4: A和B都失败** — 策略改进空间有限")
        doc.append("建议: 用基线参数(+5.6pt/笔)继续运行，不再追求大幅改善")
        doc.append("或: 转向其他策略方向")

    report = "\n".join(doc)
    path = Path("tmp") / "rhythm_swing_exit_optimization_and_early_entry.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
