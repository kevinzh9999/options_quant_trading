#!/usr/bin/env python3
"""节奏摆动日"冲高回落"反向交易理想化测试。"""
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


def is_rhythm_swing(day_bars, open_p, prev_c):
    if len(day_bars) < 10:
        return False
    ob = day_bars.iloc[:6]
    o30_amp = (ob['high'].max() - ob['low'].min()) / ob['open'].iloc[0] * 100
    dh, dl = day_bars['high'].max(), day_bars['low'].min()
    full_amp = (dh - dl) / open_p * 100
    ratio = full_amp / o30_amp if o30_amp > 0 else 99
    am = day_bars[(day_bars.index.hour >= 1) & (day_bars.index.hour < 4)]
    pm = day_bars[(day_bars.index.hour >= 5) & (day_bars.index.hour < 7)]
    am_amp = (am['high'].max() - am['low'].min()) / open_p * 100 if len(am) > 0 else 0
    pm_amp = (pm['high'].max() - pm['low'].min()) / open_p * 100 if len(pm) > 0 else 0
    return (0.4 <= o30_amp <= 1.2 and ratio < 1.8 and full_amp < 1.5 and am_amp >= pm_amp * 0.8)


def is_early_rally(day_bars, open_p, prev_c):
    gap = (open_p - prev_c) / prev_c * 100 if prev_c > 0 else 0
    if gap < -0.2:
        return False, 0, 0, 0, None
    # 09:30-10:30 = UTC 01:30-02:30
    early = day_bars[(day_bars.index.hour == 1) | (day_bars.index.hour == 2)]
    if len(early) < 4:
        return False, 0, 0, 0, None
    eh = early['high'].max()
    el = early['low'].min()
    amp = (eh - el) / open_p * 100
    if amp < 0.4:
        return False, 0, 0, 0, None
    peak_idx = early['high'].idxmax()
    return True, eh, el, eh - el, peak_idx


def simulate_trade(day_bars, peak_price, peak_low, peak_time):
    """模拟反向做空交易。"""
    stop = peak_price + 3
    target_80 = peak_price - (peak_price - peak_low) * 0.8

    # 找3 bar无新高
    bars_after_peak = day_bars[day_bars.index > peak_time]
    no_new_high_count = 0
    entry_bar = None
    for i, (idx, bar) in enumerate(bars_after_peak.iterrows()):
        if float(bar['high']) <= peak_price:
            no_new_high_count += 1
        else:
            no_new_high_count = 0
        if no_new_high_count >= 3:
            # 下一根bar入场
            remaining = bars_after_peak.iloc[i+1:]
            if len(remaining) > 0:
                entry_bar = remaining.iloc[0]
                entry_idx = remaining.index[0]
                break
            break

    if entry_bar is None:
        return None

    entry_price = float(entry_bar['open'])
    entry_time = entry_idx

    # 从entry之后逐bar检查
    trade_bars = day_bars[day_bars.index >= entry_idx]
    # 14:00 UTC = 06:00
    eod_cutoff_h = 6

    for idx, bar in trade_bars.iterrows():
        bh = float(bar['high'])
        bl = float(bar['low'])

        # 止损
        if bh >= stop:
            return {
                'entry_time': entry_time, 'entry_price': entry_price,
                'exit_time': idx, 'exit_price': stop,
                'exit_reason': 'stop', 'pnl_pts': entry_price - stop,
            }
        # target_80
        if bl <= target_80:
            return {
                'entry_time': entry_time, 'entry_price': entry_price,
                'exit_time': idx, 'exit_price': target_80,
                'exit_reason': 'target_80', 'pnl_pts': entry_price - target_80,
            }
        # 14:00 时间平仓
        if idx.hour >= eod_cutoff_h:
            exit_p = float(bar['open'])
            return {
                'entry_time': entry_time, 'entry_price': entry_price,
                'exit_time': idx, 'exit_price': exit_p,
                'exit_reason': 'time_close', 'pnl_pts': entry_price - exit_p,
            }

    # 持仓到最后
    last = trade_bars.iloc[-1]
    return {
        'entry_time': entry_time, 'entry_price': entry_price,
        'exit_time': trade_bars.index[-1], 'exit_price': float(last['close']),
        'exit_reason': 'eod', 'pnl_pts': entry_price - float(last['close']),
    }


def main():
    df = load_im()
    daily_close = df.groupby('date')['close'].last()
    all_dates = sorted(df['date'].unique())

    trades = []
    n_swing = 0
    n_rally = 0
    n_no_entry = 0

    for i, date in enumerate(all_dates):
        day = df[df['date'] == date]
        if len(day) < 10:
            continue
        open_p = float(day.iloc[0]['open'])
        prev_c = float(daily_close.get(all_dates[i-1], open_p)) if i > 0 else open_p

        if not is_rhythm_swing(day, open_p, prev_c):
            continue
        n_swing += 1

        ok, peak_price, peak_low, amp_pts, peak_time = is_early_rally(day, open_p, prev_c)
        if not ok:
            continue
        n_rally += 1

        result = simulate_trade(day, peak_price, peak_low, peak_time)
        if result is None:
            n_no_entry += 1
            continue

        result['trade_date'] = date
        result['early_peak_price'] = peak_price
        result['early_low'] = peak_low
        result['early_amp_pts'] = amp_pts
        result['pnl_pct'] = result['pnl_pts'] / result['entry_price'] * 100
        # BJ times for display
        result['entry_bj'] = (result['entry_time'] + pd.Timedelta(hours=8)).strftime('%H:%M')
        result['exit_bj'] = (result['exit_time'] + pd.Timedelta(hours=8)).strftime('%H:%M')
        result['peak_bj'] = (peak_time + pd.Timedelta(hours=8)).strftime('%H:%M')
        trades.append(result)

    tdf = pd.DataFrame(trades)
    doc = ["# 节奏摆动日 冲高回落 反向交易理想化测试\n"]
    doc.append(f"数据: IM 900天 ({df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')})")
    doc.append(f"节奏摆动日: {n_swing}天")
    doc.append(f"冲高型: {n_rally}天")
    doc.append(f"未入场(无3bar无新高): {n_no_entry}天")
    doc.append(f"有效交易: **{len(tdf)}笔**\n")

    if len(tdf) == 0:
        doc.append("无交易，终止")
        report = "\n".join(doc)
        Path("tmp").mkdir(exist_ok=True)
        with open("tmp/rhythm_swing_short_idealized_test.md", 'w') as f:
            f.write(report)
        print(report)
        return

    # 交易明细
    doc.append("## 交易明细\n")
    doc.append(f"| date | peak_bj | peak | early_low | amp | entry_bj | entry | exit_bj | exit | reason | pnl_pts |")
    doc.append(f"|------|---------|------|-----------|-----|----------|-------|---------|------|--------|---------|")
    for _, t in tdf.iterrows():
        doc.append(f"| {t['trade_date']} | {t['peak_bj']} | {t['early_peak_price']:.0f} | {t['early_low']:.0f} | "
                   f"{t['early_amp_pts']:.0f} | {t['entry_bj']} | {t['entry_price']:.0f} | "
                   f"{t['exit_bj']} | {t['exit_price']:.0f} | {t['exit_reason']} | {t['pnl_pts']:+.0f} |")

    # 整体统计
    doc.append(f"\n## 整体统计\n")
    pnl = tdf['pnl_pts']
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    doc.append(f"| 指标 | 值 |")
    doc.append(f"|------|-----|")
    doc.append(f"| 总交易数 | {len(tdf)} |")
    doc.append(f"| **胜率** | **{len(wins)/len(tdf)*100:.0f}%** |")
    doc.append(f"| **平均盈利** | **{pnl.mean():+.1f}pt** |")
    doc.append(f"| 中位数盈利 | {pnl.median():+.1f}pt |")
    doc.append(f"| 最大单笔盈利 | {pnl.max():+.0f}pt |")
    doc.append(f"| 最大单笔亏损 | {pnl.min():+.0f}pt |")
    doc.append(f"| 标准差 | {pnl.std():.1f}pt |")
    doc.append(f"| **累计PnL** | **{pnl.sum():+.0f}pt** |")
    if len(wins) > 0 and len(losses) > 0:
        doc.append(f"| 盈亏比 | {wins.mean()/abs(losses.mean()):.2f} |")

    # 按exit_reason分组
    doc.append(f"\n## 按退出原因分组\n")
    doc.append(f"| exit_reason | 笔数 | 平均pnl | 占比 |")
    doc.append(f"|-------------|------|---------|------|")
    for reason in ['target_80', 'stop', 'time_close', 'eod']:
        sub = tdf[tdf['exit_reason'] == reason]
        if len(sub) > 0:
            doc.append(f"| {reason} | {len(sub)} | {sub['pnl_pts'].mean():+.1f}pt | {len(sub)/len(tdf)*100:.0f}% |")

    # 按early_amp分组
    doc.append(f"\n## 按早盘振幅分组\n")
    doc.append(f"| early_amp区间 | 笔数 | 平均pnl |")
    doc.append(f"|--------------|------|---------|")
    for lo, hi, label in [(0, 50, '0-50pt'), (50, 80, '50-80pt'), (80, 200, '80+pt')]:
        sub = tdf[(tdf['early_amp_pts'] >= lo) & (tdf['early_amp_pts'] < hi)]
        if len(sub) > 0:
            doc.append(f"| {label} | {len(sub)} | {sub['pnl_pts'].mean():+.1f}pt |")

    # 时间稳定性
    doc.append(f"\n## 时间稳定性\n")
    mid_date = all_dates[len(all_dates)//2]
    first_half = tdf[tdf['trade_date'] < mid_date]
    second_half = tdf[tdf['trade_date'] >= mid_date]
    doc.append(f"| 段 | 笔数 | 平均pnl | 累计pnl |")
    doc.append(f"|---|------|---------|---------|")
    doc.append(f"| 前半 | {len(first_half)} | {first_half['pnl_pts'].mean():+.1f}pt | {first_half['pnl_pts'].sum():+.0f}pt |")
    doc.append(f"| 后半 | {len(second_half)} | {second_half['pnl_pts'].mean():+.1f}pt | {second_half['pnl_pts'].sum():+.0f}pt |")

    # 判定
    avg_pnl = pnl.mean()
    win_rate = len(wins) / len(tdf) * 100
    both_positive = first_half['pnl_pts'].sum() > 0 and second_half['pnl_pts'].sum() > 0
    target_rate = len(tdf[tdf['exit_reason'] == 'target_80']) / len(tdf) * 100

    doc.append(f"\n## 判定\n")
    doc.append(f"| 条件 | 值 | 达标? |")
    doc.append(f"|------|-----|-------|")
    doc.append(f"| 平均盈利>=30pt | {avg_pnl:+.1f}pt | {'✓' if avg_pnl >= 30 else '✗'} |")
    doc.append(f"| 胜率>=55% | {win_rate:.0f}% | {'✓' if win_rate >= 55 else '✗'} |")
    doc.append(f"| 时间稳定 | 前{first_half['pnl_pts'].sum():+.0f}/后{second_half['pnl_pts'].sum():+.0f} | {'✓' if both_positive else '✗'} |")
    doc.append(f"| target达成>=30% | {target_rate:.0f}% | {'✓' if target_rate >= 30 else '✗'} |")

    if avg_pnl >= 30 and win_rate >= 55 and both_positive and target_rate >= 30:
        doc.append(f"\n**判定A：值得继续研究** ✓")
    elif avg_pnl >= 15 or win_rate >= 50:
        doc.append(f"\n**判定B：边缘值得**")
    else:
        doc.append(f"\n**判定C：不值得继续** ✗")

    # Top5 / Bottom5
    doc.append(f"\n## 最赚5笔\n")
    for _, t in tdf.nlargest(5, 'pnl_pts').iterrows():
        doc.append(f"  {t['trade_date']} {t['entry_bj']}→{t['exit_bj']} {t['exit_reason']} {t['pnl_pts']:+.0f}pt (amp={t['early_amp_pts']:.0f})")
    doc.append(f"\n## 最亏5笔\n")
    for _, t in tdf.nsmallest(5, 'pnl_pts').iterrows():
        doc.append(f"  {t['trade_date']} {t['entry_bj']}→{t['exit_bj']} {t['exit_reason']} {t['pnl_pts']:+.0f}pt (amp={t['early_amp_pts']:.0f})")

    report = "\n".join(doc)
    path = Path("tmp") / "rhythm_swing_short_idealized_test.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
