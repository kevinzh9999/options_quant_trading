#!/usr/bin/env python3
"""节奏摆动日对称入场/出场逻辑：横盘衰竭 + 急速反转。"""
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


def backtest_symmetric(cand, entry_x_pct, exit_y_pct, exit_n_bars):
    """对称入场/出场回测。

    入场路径A: 3 bar无新高 → 下一根bar open
    入场路径B: close <= peak * (1 - entry_x_pct/100) → 该bar close
    出场路径A: N bar无新低 → 下一根bar open
    出场路径B: close >= lowest * (1 + exit_y_pct/100) → 该bar close
    """
    day = cand['day_bars']
    peak = cand['early_peak']
    pt = cand['peak_time']
    stop = peak + 3

    after = day[day.index > pt]
    if len(after) < 3:
        return None

    # === 入场判定 ===
    entry_price = None
    entry_idx = None
    entry_path = None

    # 同时扫描路径A和B，取先触发的
    no_new_high_count = 0
    path_a_entry = None
    path_a_idx = None
    path_b_entry = None
    path_b_idx = None

    for j, (idx, bar) in enumerate(after.iterrows()):
        # 路径B: 急反转
        if path_b_entry is None:
            if float(bar['close']) <= peak * (1 - entry_x_pct / 100):
                path_b_entry = float(bar['close'])
                path_b_idx = idx

        # 路径A: 横盘衰竭
        if path_a_entry is None:
            if float(bar['high']) <= peak:
                no_new_high_count += 1
            else:
                no_new_high_count = 0
            if no_new_high_count >= 3:
                remaining = after.iloc[j + 1:]
                if len(remaining) > 0:
                    path_a_entry = float(remaining.iloc[0]['open'])
                    path_a_idx = remaining.index[0]

        # 两个都找到了就停
        if path_a_entry is not None and path_b_entry is not None:
            break

    # 取先触发的
    if path_a_idx is not None and path_b_idx is not None:
        if path_b_idx <= path_a_idx:
            entry_price, entry_idx, entry_path = path_b_entry, path_b_idx, 'B'
        else:
            entry_price, entry_idx, entry_path = path_a_entry, path_a_idx, 'A'
    elif path_a_idx is not None:
        entry_price, entry_idx, entry_path = path_a_entry, path_a_idx, 'A'
    elif path_b_idx is not None:
        entry_price, entry_idx, entry_path = path_b_entry, path_b_idx, 'B'
    else:
        return None

    # === 出场判定 ===
    trade_bars = day[day.index >= entry_idx]
    lowest_since = entry_price
    no_new_low_count = 0
    bars_held = 0
    exit_path = None

    for idx, bar in trade_bars.iterrows():
        bh, bl, bc = float(bar['high']), float(bar['low']), float(bar['close'])
        bars_held += 1

        # 止损优先
        if bh >= stop:
            return {'pnl': entry_price - stop, 'reason': 'stop', 'date': cand['date'],
                    'entry_p': entry_price, 'exit_p': stop,
                    'entry_path': entry_path, 'exit_path': 'stop'}

        # 更新最低
        if bl < lowest_since:
            lowest_since = bl
            no_new_low_count = 0
        else:
            no_new_low_count += 1

        # 至少持仓2根bar才检查出场
        if bars_held <= 2:
            continue

        # 出场路径B: 急反弹
        if lowest_since < entry_price and bc >= lowest_since * (1 + exit_y_pct / 100):
            return {'pnl': entry_price - bc, 'reason': 'exit_B', 'date': cand['date'],
                    'entry_p': entry_price, 'exit_p': bc,
                    'entry_path': entry_path, 'exit_path': 'B'}

        # 出场路径A: N bar无新低
        if no_new_low_count >= exit_n_bars:
            # 下一根bar的open
            bar_pos = trade_bars.index.get_loc(idx)
            if bar_pos + 1 < len(trade_bars):
                next_bar = trade_bars.iloc[bar_pos + 1]
                exit_p = float(next_bar['open'])
                return {'pnl': entry_price - exit_p, 'reason': 'exit_A', 'date': cand['date'],
                        'entry_p': entry_price, 'exit_p': exit_p,
                        'entry_path': entry_path, 'exit_path': 'A'}

        # 时间平仓
        if idx.hour >= 6:
            exit_p = float(bar['open'])
            return {'pnl': entry_price - exit_p, 'reason': 'time', 'date': cand['date'],
                    'entry_p': entry_price, 'exit_p': exit_p,
                    'entry_path': entry_path, 'exit_path': 'time'}

    # EOD
    last = trade_bars.iloc[-1]
    return {'pnl': entry_price - float(last['close']), 'reason': 'eod', 'date': cand['date'],
            'entry_p': entry_price, 'exit_p': float(last['close']),
            'entry_path': entry_path, 'exit_path': 'eod'}


def backtest_old(cand):
    """旧基线: 3bar无新高, target=0.8, stop=peak+3, time=14:00。"""
    day = cand['day_bars']
    peak = cand['early_peak']
    amp = cand['early_amp']
    pt = cand['peak_time']
    stop = peak + 3
    target = peak - amp * 0.8

    after = day[day.index > pt]
    count = 0
    entry_price = None
    entry_idx = None
    for j, (idx, bar) in enumerate(after.iterrows()):
        if float(bar['high']) <= peak:
            count += 1
        else:
            count = 0
        if count >= 3:
            remaining = after.iloc[j + 1:]
            if len(remaining) > 0:
                entry_price = float(remaining.iloc[0]['open'])
                entry_idx = remaining.index[0]
            break
    if entry_price is None:
        return None

    for idx, bar in day[day.index >= entry_idx].iterrows():
        bh, bl = float(bar['high']), float(bar['low'])
        if bh >= stop:
            return {'pnl': entry_price - stop, 'reason': 'stop', 'date': cand['date'],
                    'entry_p': entry_price, 'exit_p': stop}
        if bl <= target:
            return {'pnl': entry_price - target, 'reason': 'target', 'date': cand['date'],
                    'entry_p': entry_price, 'exit_p': target}
        if idx.hour >= 6:
            exit_p = float(bar['open'])
            return {'pnl': entry_price - exit_p, 'reason': 'time', 'date': cand['date'],
                    'entry_p': entry_price, 'exit_p': exit_p}
    last = day.iloc[-1]
    return {'pnl': entry_price - float(last['close']), 'reason': 'eod', 'date': cand['date'],
            'entry_p': entry_price, 'exit_p': float(last['close'])}


def calc_stats(trades):
    if not trades:
        return None
    pnls = [t['pnl'] for t in trades]
    n = len(pnls)
    wins = [p for p in pnls if p > 0]
    cum = np.cumsum(pnls)
    dd = np.max(np.maximum.accumulate(cum) - cum) if len(cum) > 0 else 0
    return {
        'n': n, 'wr': len(wins) / n * 100, 'avg': np.mean(pnls),
        'total': sum(pnls), 'max_dd': dd,
    }


def main():
    import datetime
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

    doc = []
    doc.append("# 节奏摆动日对称入场/出场逻辑\n")

    # ═══════════════════════════════════════════════
    # 第一步: 4/10 sanity check
    # ═══════════════════════════════════════════════
    doc.append("## 第一步: 4/10 Sanity Check\n")
    target_date = datetime.date(2026, 4, 10)
    cand_410 = None
    for c in candidates:
        if c['date'] == target_date:
            cand_410 = c
            break

    if cand_410 is not None:
        day = cand_410['day_bars']
        peak = cand_410['early_peak']
        doc.append(f"peak={peak:.0f}, early_low={cand_410['early_low']:.0f}, amp={cand_410['early_amp']:.0f}\n")

        # 列出peak之后的bar
        after_peak = day[day.index > cand_410['peak_time']]
        doc.append("| bar_time(BJ) | open | high | low | close |")
        doc.append("|-------------|------|------|-----|-------|")
        for idx, bar in after_peak.head(20).iterrows():
            bj = (idx + pd.Timedelta(hours=8)).strftime('%H:%M')
            doc.append(f"| {bj} | {bar['open']:.0f} | {bar['high']:.0f} | {bar['low']:.0f} | {bar['close']:.0f} |")

        # 新逻辑
        t_new = backtest_symmetric(cand_410, 0.2, 0.15, 3)
        t_old = backtest_old(cand_410)
        doc.append(f"\n旧基线: pnl={t_old['pnl']:+.1f}pt, reason={t_old['reason']}")
        if t_new:
            doc.append(f"新对称: pnl={t_new['pnl']:+.1f}pt, entry_path={t_new['entry_path']}, "
                       f"exit_path={t_new['exit_path']}, entry={t_new['entry_p']:.0f}, exit={t_new['exit_p']:.0f}")
        else:
            doc.append("新对称: 无入场")
        doc.append("")

    # ═══════════════════════════════════════════════
    # 第二步: IS段 24组合sweep
    # ═══════════════════════════════════════════════
    print("IS段sweep...")
    doc.append("## 第二步: IS段24组合Sweep\n")

    entry_xs = [0.15, 0.2, 0.25, 0.3]
    exit_ys = [0.15, 0.2, 0.25]
    exit_ns = [2, 3]
    grid = list(product(entry_xs, exit_ys, exit_ns))

    results = []
    for x, y, nn in grid:
        trades = [backtest_symmetric(c, x, y, nn) for c in is_cands]
        trades = [t for t in trades if t is not None]
        if not trades:
            continue
        s = calc_stats(trades)
        pA_entry = sum(1 for t in trades if t['entry_path'] == 'A') / len(trades) * 100
        pB_entry = sum(1 for t in trades if t['entry_path'] == 'B') / len(trades) * 100
        pA_exit = sum(1 for t in trades if t['exit_path'] == 'A') / len(trades) * 100
        pB_exit = sum(1 for t in trades if t['exit_path'] == 'B') / len(trades) * 100
        time_exit = sum(1 for t in trades if t['exit_path'] in ('time', 'eod')) / len(trades) * 100
        stop_r = sum(1 for t in trades if t['exit_path'] == 'stop') / len(trades) * 100
        results.append({
            'X': x, 'Y': y, 'N': nn, **s,
            'pA_entry': pA_entry, 'pB_entry': pB_entry,
            'pA_exit': pA_exit, 'pB_exit': pB_exit,
            'time_exit': time_exit, 'stop_rate': stop_r,
        })

    rdf = pd.DataFrame(results).sort_values('total', ascending=False)

    doc.append("| X% | Y% | N | n | WR | Avg | Total | DD | entA% | entB% | exA% | exB% | time% | stp% |")
    doc.append("|-----|-----|---|---|-----|-----|-------|-----|-------|-------|------|------|-------|------|")
    for _, r in rdf.iterrows():
        doc.append(f"| {r['X']:.2f} | {r['Y']:.2f} | {int(r['N'])} | {int(r['n'])} | "
                   f"{r['wr']:.0f}% | {r['avg']:+.1f} | {r['total']:+.0f} | {r['max_dd']:.0f} | "
                   f"{r['pA_entry']:.0f}% | {r['pB_entry']:.0f}% | {r['pA_exit']:.0f}% | {r['pB_exit']:.0f}% | "
                   f"{r['time_exit']:.0f}% | {r['stop_rate']:.0f}% |")

    # ═══════════════════════════════════════════════
    # 第三步: IS最优选择
    # ═══════════════════════════════════════════════
    doc.append(f"\n## 第三步: IS最优选择\n")
    qualified = rdf[
        (rdf['n'] >= 120) &
        (rdf['wr'] >= 65) &
        (rdf['avg'] >= 10.0) &
        (rdf['max_dd'] < 100) &
        (rdf['pA_entry'] >= 20) & (rdf['pB_entry'] >= 20)  # 两路径都有贡献
    ].copy()

    # 也试放宽路径要求（入场或出场有一方双路径即可）
    if len(qualified) == 0:
        qualified2 = rdf[
            (rdf['n'] >= 120) &
            (rdf['wr'] >= 65) &
            (rdf['avg'] >= 10.0) &
            (rdf['max_dd'] < 100)
        ].copy()
        if len(qualified2) > 0:
            doc.append("严格条件(双路径各>=20%入场)无合格，放宽路径要求:")
            qualified = qualified2

    if len(qualified) == 0:
        doc.append("**无组合满足 N>=120, WR>=65%, Avg>=10, DD<100**\n")
        # 次优
        near = rdf[rdf['n'] >= 100].copy()
        if len(near) > 0:
            near['score'] = near['total'] * near['wr'] / near['max_dd'].replace(0, 1)
            best = near.loc[near['score'].idxmax()]
            doc.append(f"次优: X={best['X']:.2f}, Y={best['Y']:.2f}, N={int(best['N'])}")
            doc.append(f"  N={int(best['n'])}, WR={best['wr']:.0f}%, Avg={best['avg']:+.1f}, "
                       f"Total={best['total']:+.0f}, DD={best['max_dd']:.0f}")
            fails = []
            if best['wr'] < 65: fails.append(f"WR={best['wr']:.0f}%<65%")
            if best['avg'] < 10: fails.append(f"Avg={best['avg']:+.1f}<10")
            if best['max_dd'] >= 100: fails.append(f"DD={best['max_dd']:.0f}>=100")
            doc.append(f"  未达标: {', '.join(fails)}")
            best_x, best_y, best_nn = best['X'], best['Y'], int(best['N'])
        else:
            best_x, best_y, best_nn = 0.2, 0.15, 3
    else:
        qualified['score'] = qualified['total'] * qualified['wr'] / qualified['max_dd'].replace(0, 1)
        best = qualified.loc[qualified['score'].idxmax()]
        best_x, best_y, best_nn = best['X'], best['Y'], int(best['N'])
        doc.append(f"**IS最优: X={best_x:.2f}%, Y={best_y:.2f}%, N={best_nn}**")
        doc.append(f"  N={int(best['n'])}, WR={best['wr']:.0f}%, Avg={best['avg']:+.1f}, "
                   f"Total={best['total']:+.0f}, DD={best['max_dd']:.0f}")

    # ═══════════════════════════════════════════════
    # 第四步: OOS验证
    # ═══════════════════════════════════════════════
    doc.append(f"\n## 第四步: OOS验证 (X={best_x:.2f}, Y={best_y:.2f}, N={best_nn})\n")

    segments = [
        ('IS', is_cands), ('OOS', oos_cands),
        ('OOS前半', oos_first), ('OOS后半', oos_second),
    ]

    doc.append("| 段 | N | WR | Avg | Total | entA% | entB% | exA% | exB% |")
    doc.append("|-----|---|-----|-----|-------|-------|-------|------|------|")
    seg_trades = {}
    for seg_name, seg_cands in segments:
        trades = [backtest_symmetric(c, best_x, best_y, best_nn) for c in seg_cands]
        trades = [t for t in trades if t is not None]
        seg_trades[seg_name] = trades
        if not trades:
            doc.append(f"| {seg_name} | 0 | - | - | - | - | - | - | - |")
            continue
        s = calc_stats(trades)
        pA_e = sum(1 for t in trades if t['entry_path'] == 'A') / len(trades) * 100
        pB_e = sum(1 for t in trades if t['entry_path'] == 'B') / len(trades) * 100
        pA_x = sum(1 for t in trades if t['exit_path'] == 'A') / len(trades) * 100
        pB_x = sum(1 for t in trades if t['exit_path'] == 'B') / len(trades) * 100
        doc.append(f"| {seg_name} | {s['n']} | {s['wr']:.0f}% | {s['avg']:+.1f} | {s['total']:+.0f} | "
                   f"{pA_e:.0f}% | {pB_e:.0f}% | {pA_x:.0f}% | {pB_x:.0f}% |")

    # ═══════════════════════════════════════════════
    # 第五步: 跟旧基线对比
    # ═══════════════════════════════════════════════
    doc.append(f"\n## 第五步: 跟旧基线对比\n")

    old_oos_trades = [backtest_old(c) for c in oos_cands]
    old_oos_trades = [t for t in old_oos_trades if t is not None]
    old_s = calc_stats(old_oos_trades)

    new_oos = seg_trades.get('OOS', [])
    new_s = calc_stats(new_oos) if new_oos else {'avg': 0, 'wr': 0, 'total': 0}

    old_oos2 = [backtest_old(c) for c in oos_second]
    old_oos2 = [t for t in old_oos2 if t is not None]
    old_s2 = calc_stats(old_oos2) if old_oos2 else {'avg': 0, 'wr': 0}

    new_oos2 = seg_trades.get('OOS后半', [])
    new_s2 = calc_stats(new_oos2) if new_oos2 else {'avg': 0, 'wr': 0}

    doc.append("| 指标 | 旧基线OOS | 新对称OOS | 改进 |")
    doc.append("|------|---------|---------|------|")
    doc.append(f"| avg_pnl | {old_s['avg']:+.1f} | {new_s['avg']:+.1f} | {new_s['avg']-old_s['avg']:+.1f} |")
    doc.append(f"| OOS后半avg | {old_s2['avg']:+.1f} | {new_s2['avg']:+.1f} | {new_s2['avg']-old_s2['avg']:+.1f} |")
    doc.append(f"| 胜率 | {old_s['wr']:.0f}% | {new_s['wr']:.0f}% | {new_s['wr']-old_s['wr']:+.0f}% |")
    doc.append(f"| total | {old_s['total']:+.0f} | {new_s['total']:+.0f} | {new_s['total']-old_s['total']:+.0f} |")

    # 重点日期逐笔
    doc.append(f"\n### 重点日期逐笔\n")
    key_dates = [datetime.date(2026, 4, 10), datetime.date(2026, 4, 7),
                 datetime.date(2025, 9, 17)]
    doc.append("| 日期 | 旧pnl | 新pnl | 新entry_path | 新exit_path |")
    doc.append("|------|-------|-------|-------------|-------------|")
    for kd in key_dates:
        old_t = None
        new_t = None
        for c in candidates:
            if c['date'] == kd:
                old_t = backtest_old(c)
                new_t = backtest_symmetric(c, best_x, best_y, best_nn)
                break
        if old_t and new_t:
            doc.append(f"| {kd} | {old_t['pnl']:+.1f} | {new_t['pnl']:+.1f} | {new_t['entry_path']} | {new_t['exit_path']} |")

    # ═══════════════════════════════════════════════
    # 第六步: 入场路径拆解
    # ═══════════════════════════════════════════════
    doc.append(f"\n## 第六步: 入场路径拆解（全196笔）\n")
    all_trades = [backtest_symmetric(c, best_x, best_y, best_nn) for c in candidates]
    all_trades = [t for t in all_trades if t is not None]

    doc.append("| 入场路径 | N | WR | Avg | Total |")
    doc.append("|---------|---|----|-----|-------|")
    for path in ['A', 'B']:
        sub = [t for t in all_trades if t['entry_path'] == path]
        if sub:
            pnls = [t['pnl'] for t in sub]
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            doc.append(f"| 路径{path} | {len(sub)} | {wr:.0f}% | {np.mean(pnls):+.1f} | {sum(pnls):+.0f} |")

    # ═══════════════════════════════════════════════
    # 第七步: 出场路径拆解
    # ═══════════════════════════════════════════════
    doc.append(f"\n## 第七步: 出场路径拆解（全196笔）\n")
    doc.append("| 出场路径 | N | Avg |")
    doc.append("|---------|---|----|")
    for path in ['A', 'B', 'stop', 'time', 'eod']:
        sub = [t for t in all_trades if t['exit_path'] == path]
        if sub:
            pnls = [t['pnl'] for t in sub]
            doc.append(f"| {path} | {len(sub)} | {np.mean(pnls):+.1f} |")

    # ═══════════════════════════════════════════════
    # 综合判定
    # ═══════════════════════════════════════════════
    doc.append(f"\n## 综合判定\n")
    oos_avg = new_s['avg'] if new_s else 0
    oos2_avg = new_s2['avg'] if new_s2 else 0
    oos_wr = new_s['wr'] if new_s else 0

    if oos_avg >= 10 and oos2_avg >= 7 and oos_wr >= 65:
        doc.append(f"**判定S1: 对称逻辑显著优于旧基线** ✓")
        doc.append(f"OOS avg={oos_avg:+.1f}>=10, 后半={oos2_avg:+.1f}>=7, WR={oos_wr:.0f}%>=65%")
    elif oos_avg >= 7 or oos2_avg >= 5:
        doc.append(f"**判定S2: 改进有限但方向正确**")
        doc.append(f"OOS avg={oos_avg:+.1f}, 后半={oos2_avg:+.1f}, WR={oos_wr:.0f}%")
    else:
        doc.append(f"**判定S3: 对称逻辑没有显著改进** ✗")
        doc.append(f"OOS avg={oos_avg:+.1f}, 后半={oos2_avg:+.1f}")

    report = "\n".join(doc)
    path = Path("tmp") / "symmetric_entry_exit_logic.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
