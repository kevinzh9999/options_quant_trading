#!/usr/bin/env python3
"""节奏摆动日冲高反转策略 IS/OOS 参数优化。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np, pandas as pd
from pathlib import Path
from itertools import product
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
    """找所有节奏摆动日+冲高型的候选。"""
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
        amp = eh - el
        if (eh - el) / open_p * 100 < 0.4: continue
        peak_time = early['high'].idxmax()
        candidates.append({
            'date': date, 'day_bars': day, 'peak_price': eh,
            'peak_low': el, 'amp_pts': amp, 'peak_time': peak_time,
        })
    return candidates


def backtest_one(cand, no_new_high_bars=3, stop_offset=3, target_pct=0.8,
                 time_close_h=6, min_amp=0):
    """回测单个候选日。"""
    if cand['amp_pts'] < min_amp:
        return None
    day = cand['day_bars']
    peak = cand['peak_price']
    low = cand['peak_low']
    pt = cand['peak_time']
    stop = peak + stop_offset
    target = peak - (peak - low) * target_pct

    after = day[day.index > pt]
    count = 0
    entry_bar = None
    for i, (idx, bar) in enumerate(after.iterrows()):
        if float(bar['high']) <= peak:
            count += 1
        else:
            count = 0
        if count >= no_new_high_bars:
            remaining = after.iloc[i+1:]
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
                    'entry_p': entry_p, 'exit_p': stop, 'amp': cand['amp_pts']}
        if bl <= target:
            return {'pnl': entry_p - target, 'reason': 'target', 'date': cand['date'],
                    'entry_p': entry_p, 'exit_p': target, 'amp': cand['amp_pts']}
        if idx.hour >= time_close_h:
            exit_p = float(bar['open'])
            return {'pnl': entry_p - exit_p, 'reason': 'time', 'date': cand['date'],
                    'entry_p': entry_p, 'exit_p': exit_p, 'amp': cand['amp_pts']}

    last = day.iloc[-1]
    return {'pnl': entry_p - float(last['close']), 'reason': 'eod', 'date': cand['date'],
            'entry_p': entry_p, 'exit_p': float(last['close']), 'amp': cand['amp_pts']}


def run_sweep(candidates, params_grid):
    """在候选日上跑参数sweep。"""
    results = []
    for stop, tgt, amp in params_grid:
        trades = []
        for c in candidates:
            t = backtest_one(c, stop_offset=stop, target_pct=tgt, min_amp=amp)
            if t is not None:
                trades.append(t)
        if not trades:
            results.append({'stop': stop, 'target': tgt, 'min_amp': amp,
                            'n': 0, 'wr': 0, 'avg': 0, 'total': 0, 'max_dd': 0,
                            'pf': 0, 'tgt_rate': 0, 'stop_rate': 0})
            continue
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        cum = np.cumsum(pnls)
        dd = np.max(np.maximum.accumulate(cum) - cum) if len(cum) > 0 else 0
        pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 99
        tgt_n = sum(1 for t in trades if t['reason'] == 'target')
        stp_n = sum(1 for t in trades if t['reason'] == 'stop')
        results.append({
            'stop': stop, 'target': tgt, 'min_amp': amp,
            'n': len(trades), 'wr': len(wins)/len(trades)*100,
            'avg': np.mean(pnls), 'total': sum(pnls),
            'max_dd': dd, 'pf': pf,
            'tgt_rate': tgt_n/len(trades)*100, 'stop_rate': stp_n/len(trades)*100,
        })
    return pd.DataFrame(results)


def main():
    print("加载数据...")
    df = load_im()
    print("筛选候选日...")
    candidates = find_candidates(df)
    print(f"候选日: {len(candidates)}天")

    # IS/OOS切分
    n = len(candidates)
    is_n = n * 2 // 3
    is_cands = candidates[:is_n]
    oos_cands = candidates[is_n:]
    print(f"IS: {len(is_cands)}天, OOS: {len(oos_cands)}天")

    doc = ["# 节奏摆动日冲高反转策略 IS/OOS 参数优化\n"]
    doc.append(f"总候选日: {n}天")
    doc.append(f"IS: 前{len(is_cands)}天 ({is_cands[0]['date']}~{is_cands[-1]['date']})")
    doc.append(f"OOS: 后{len(oos_cands)}天 ({oos_cands[0]['date']}~{oos_cands[-1]['date']})\n")

    # 参数网格
    stops = [3, 5, 8, 12]
    targets = [0.5, 0.65, 0.8, 1.0]
    amps = [0, 30, 50, 70]
    grid = list(product(stops, targets, amps))
    print(f"参数组合: {len(grid)}个")

    # IS sweep
    print("IS段参数搜索...")
    is_results = run_sweep(is_cands, grid)

    doc.append("## IS段结果 (64组合)\n")
    doc.append(f"{'stop':>5s} {'tgt':>5s} {'amp':>5s} | {'N':>4s} {'WR':>5s} {'Avg':>7s} {'Total':>7s} {'DD':>6s} {'PF':>5s} {'Tgt%':>5s} {'Stp%':>5s}")
    doc.append("-" * 72)
    for _, r in is_results.sort_values('total', ascending=False).iterrows():
        doc.append(f"{int(r['stop']):>5d} {r['target']:>5.2f} {int(r['min_amp']):>5d} | "
                   f"{int(r['n']):>4d} {r['wr']:>4.0f}% {r['avg']:>+7.1f} {r['total']:>+7.0f} "
                   f"{r['max_dd']:>6.0f} {r['pf']:>5.1f} {r['tgt_rate']:>4.0f}% {r['stop_rate']:>4.0f}%")

    # 选最优
    doc.append(f"\n## IS最优选择\n")
    qualified = is_results[
        (is_results['n'] >= 80) &
        (is_results['wr'] >= 60) &
        (is_results['avg'] >= 8.0) &
        (is_results['max_dd'] < 100)
    ].copy()

    if len(qualified) == 0:
        # 降低标准
        qualified = is_results[
            (is_results['n'] >= 60) &
            (is_results['wr'] >= 55) &
            (is_results['avg'] >= 7.5)
        ].copy()
        doc.append("标准1(N>=80,WR>=60%,Avg>=8,DD<100)无合格组合，降低标准...")

    if len(qualified) == 0:
        doc.append("降低标准后仍无合格组合，使用基线参数")
        best_stop, best_tgt, best_amp = 3, 0.8, 0
    else:
        qualified['score'] = qualified['total'] * qualified['wr'] / qualified['max_dd'].replace(0, 1)
        best = qualified.loc[qualified['score'].idxmax()]
        best_stop = int(best['stop'])
        best_tgt = best['target']
        best_amp = int(best['min_amp'])
        doc.append(f"最优组合: stop={best_stop} target={best_tgt} min_amp={best_amp}")
        doc.append(f"IS指标: N={int(best['n'])} WR={best['wr']:.0f}% Avg={best['avg']:+.1f} "
                   f"Total={best['total']:+.0f} DD={best['max_dd']:.0f} PF={best['pf']:.1f}")

    # IS最优诊断
    doc.append(f"\n## IS最优诊断\n")
    is_trades = []
    for c in is_cands:
        t = backtest_one(c, stop_offset=best_stop, target_pct=best_tgt, min_amp=best_amp)
        if t: is_trades.append(t)

    if is_trades:
        is_tdf = pd.DataFrame(is_trades)
        # Top5/Bottom5
        doc.append("最赚5笔:")
        for _, t in is_tdf.nlargest(5, 'pnl').iterrows():
            doc.append(f"  {t['date']} {t['reason']} {t['pnl']:+.0f}pt (amp={t['amp']:.0f})")
        doc.append("最亏5笔:")
        for _, t in is_tdf.nsmallest(5, 'pnl').iterrows():
            doc.append(f"  {t['date']} {t['reason']} {t['pnl']:+.0f}pt (amp={t['amp']:.0f})")

    # OOS验证
    doc.append(f"\n## OOS验证\n")
    doc.append(f"参数: stop={best_stop} target={best_tgt} min_amp={best_amp}")

    oos_trades = []
    for c in oos_cands:
        t = backtest_one(c, stop_offset=best_stop, target_pct=best_tgt, min_amp=best_amp)
        if t: oos_trades.append(t)

    if not oos_trades:
        doc.append("OOS段无交易")
    else:
        oos_tdf = pd.DataFrame(oos_trades)
        oos_pnls = oos_tdf['pnl']
        oos_wins = oos_pnls[oos_pnls > 0]
        oos_wr = len(oos_wins) / len(oos_tdf) * 100
        oos_avg = oos_pnls.mean()
        oos_total = oos_pnls.sum()

        # IS对照
        is_pnls = pd.Series([t['pnl'] for t in is_trades]) if is_trades else pd.Series([0])
        is_wr = (is_pnls > 0).mean() * 100
        is_avg = is_pnls.mean()
        is_total = is_pnls.sum()

        doc.append(f"\n| 指标 | IS ({len(is_trades)}笔) | OOS ({len(oos_trades)}笔) | 变化 |")
        doc.append(f"|------|-----|-----|------|")
        doc.append(f"| n_trades | {len(is_trades)} | {len(oos_trades)} | |")
        doc.append(f"| win_rate | {is_wr:.0f}% | {oos_wr:.0f}% | {oos_wr-is_wr:+.0f}% |")
        doc.append(f"| avg_pnl | {is_avg:+.1f}pt | {oos_avg:+.1f}pt | {oos_avg-is_avg:+.1f} |")
        doc.append(f"| total_pnl | {is_total:+.0f}pt | {oos_total:+.0f}pt | |")
        tgt_rate = sum(1 for t in oos_trades if t['reason']=='target') / len(oos_trades) * 100
        stp_rate = sum(1 for t in oos_trades if t['reason']=='stop') / len(oos_trades) * 100
        doc.append(f"| target_rate | | {tgt_rate:.0f}% | |")
        doc.append(f"| stop_rate | | {stp_rate:.0f}% | |")

        # 判定
        doc.append(f"\n## 判定\n")
        if oos_wr >= 55 and oos_avg >= 6.5 and oos_total > 0:
            doc.append(f"**判定A：参数优化成功** ✓")
            doc.append(f"最终参数: stop={best_stop} target={best_tgt} min_amp={best_amp}")
        elif oos_wr < 50 or oos_total < 0 or oos_avg < 4:
            doc.append(f"**判定B：优化无效，OOS退化** ✗")
            doc.append(f"退回基线: stop=3 target=0.8 min_amp=0")
        else:
            doc.append(f"**判定C：边缘**")
            doc.append(f"记录IS最优但不采用，用基线推进")

    report = "\n".join(doc)
    path = Path("tmp") / "rhythm_swing_short_param_optimization.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
