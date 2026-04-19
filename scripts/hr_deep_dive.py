#!/usr/bin/env python3
"""横盘反转因子深度分析：多空不对称 + 首次触发。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np, pandas as pd
from pathlib import Path
from data.storage.db_manager import get_db
from models.factors.catalog_structure import HorizontalReversalSimple

AMP_THR = 0.4
FWD_PERIODS = [5, 8, 12, 15, 18, 24]


def load_im_900d():
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


def calc_amp(bar_5m):
    daily_amp = {}
    for date, day_bars in bar_5m.groupby('date'):
        if len(day_bars) < 6: continue
        ob = day_bars.iloc[:6]
        daily_amp[date] = (ob['high'].max() - ob['low'].min()) / ob['open'].iloc[0] * 100
    s = bar_5m['date'].map(daily_amp)
    s.index = bar_5m.index
    return s


def main():
    bar_5m = load_im_900d()
    amp = calc_amp(bar_5m)
    hr = HorizontalReversalSimple(12, 3)
    fv = hr.compute_series(bar_5m)

    # Forward returns
    fwd_returns = {}
    for n in FWD_PERIODS:
        fwd_returns[n] = bar_5m['close'].pct_change(n).shift(-n) * 10000

    # 低振幅日信号
    low_amp_mask = amp < AMP_THR
    signals = pd.DataFrame({
        'factor': fv, 'amp': amp, 'date': bar_5m['date'], 'low_amp': low_amp_mask,
    })
    for n in FWD_PERIODS:
        signals[f'fwd{n}'] = fwd_returns[n]

    low_sig = signals[signals['low_amp'] & (signals['factor'] != 0)].copy()

    # 触发顺序编号
    low_sig['trigger_order'] = low_sig.groupby('date').cumcount() + 1
    low_sig['trigger_cat'] = low_sig['trigger_order'].apply(
        lambda x: 'first' if x == 1 else ('second' if x == 2 else 'third+'))

    n_low_days = low_sig['date'].nunique()
    n_signals = len(low_sig)
    n_pos = (low_sig['factor'] > 0).sum()
    n_neg = (low_sig['factor'] < 0).sum()
    n_first = (low_sig['trigger_cat'] == 'first').sum()
    n_second = (low_sig['trigger_cat'] == 'second').sum()
    n_third = (low_sig['trigger_cat'] == 'third+').sum()

    doc = ["# 横盘反转因子深度分析 — 多空不对称 + 首次触发\n"]
    doc.append(f"数据: IM 900天 ({bar_5m.index[0].strftime('%Y-%m-%d')} ~ {bar_5m.index[-1].strftime('%Y-%m-%d')})")
    doc.append(f"低振幅日(<{AMP_THR}%): {n_low_days}天")
    doc.append(f"低振幅日信号: {n_signals}个 (+1={n_pos}, -1={n_neg})")
    doc.append(f"触发分布: first={n_first}, second={n_second}, third+={n_third}\n")

    # ══════════════════════════════════════════════════════════
    # 维度1：多空不对称
    # ══════════════════════════════════════════════════════════
    doc.append("## 维度1：多空不对称\n")
    doc.append("调整后收益：+1信号直接用forward return，-1信号用forward return×(-1)")
    doc.append("正值=预测方向正确\n")

    pos_sig = low_sig[low_sig['factor'] > 0]
    neg_sig = low_sig[low_sig['factor'] < 0]

    doc.append(f"{'Fwd':>4s} | {'+1(看多) N':>10s} {'调整收益':>9s} | {'-1(看空) N':>10s} {'调整收益':>9s} | {'差距':>6s} {'哪边强':>8s}")
    doc.append("-" * 75)

    for n in FWD_PERIODS:
        col = f'fwd{n}'
        # +1信号：调整后=原始收益（正=预测对）
        pos_adj = pos_sig[col].dropna()
        # -1信号：调整后=原始收益×(-1)（正=预测对）
        neg_adj = (-neg_sig[col]).dropna()

        pos_m = pos_adj.mean() if len(pos_adj) >= 10 else np.nan
        neg_m = neg_adj.mean() if len(neg_adj) >= 10 else np.nan
        diff = pos_m - neg_m if pd.notna(pos_m) and pd.notna(neg_m) else np.nan

        if pd.notna(diff):
            if diff > 5: stronger = "+1更强"
            elif diff < -5: stronger = "-1更强"
            else: stronger = "相当"
        else:
            stronger = "N/A"

        pos_s = f"{pos_m:+.1f}" if pd.notna(pos_m) else "N/A"
        neg_s = f"{neg_m:+.1f}" if pd.notna(neg_m) else "N/A"
        diff_s = f"{diff:+.1f}" if pd.notna(diff) else "N/A"
        doc.append(f"{n:>4d} | {len(pos_adj):>10d} {pos_s:>9s} | {len(neg_adj):>10d} {neg_s:>9s} | {diff_s:>6s} {stronger:>8s}")

    # ══════════════════════════════════════════════════════════
    # 维度2：首次触发 vs 后续
    # ══════════════════════════════════════════════════════════
    doc.append(f"\n## 维度2：首次触发 vs 后续\n")
    doc.append(f"{'Fwd':>4s} | {'first N':>7s} {'first差':>8s} | {'second N':>8s} {'second差':>9s} | {'third+ N':>8s} {'third+差':>9s}")
    doc.append("-" * 75)

    for n in FWD_PERIODS:
        col = f'fwd{n}'
        parts = []
        for cat in ['first', 'second', 'third+']:
            sub = low_sig[low_sig['trigger_cat'] == cat]
            pos = sub[sub['factor'] > 0][col].dropna()
            neg = sub[sub['factor'] < 0][col].dropna()
            if len(pos) >= 5 and len(neg) >= 5:
                spread = pos.mean() - neg.mean()
                parts.append((len(sub), spread))
            else:
                parts.append((len(sub), np.nan))

        f_n, f_sp = parts[0]
        s_n, s_sp = parts[1]
        t_n, t_sp = parts[2]
        f_s = f"{f_sp:+.1f}" if pd.notna(f_sp) else "N/A"
        s_s = f"{s_sp:+.1f}" if pd.notna(s_sp) else "N/A"
        t_s = f"{t_sp:+.1f}" if pd.notna(t_sp) else "N/A"
        doc.append(f"{n:>4d} | {f_n:>7d} {f_s:>8s} | {s_n:>8d} {s_s:>9s} | {t_n:>8d} {t_s:>9s}")

    # ══════════════════════════════════════════════════════════
    # 维度1+2交叉
    # ══════════════════════════════════════════════════════════
    doc.append(f"\n## 交叉分析（forward=15, 调整后收益）\n")
    doc.append(f"{'子组':>20s} | {'N':>5s} | {'调整后收益':>10s}")
    doc.append("-" * 45)

    n = 15
    col = f'fwd{n}'
    for cat in ['first', 'second', 'third+']:
        for direction, label, sign in [(1, '+1看多', 1), (-1, '-1看空', -1)]:
            sub = low_sig[(low_sig['trigger_cat'] == cat) & (low_sig['factor'] == direction)]
            adj = (sub[col] * sign).dropna()  # +1直接用, -1翻转
            if len(adj) >= 10:
                m = adj.mean()
                doc.append(f"{cat+' '+label:>20s} | {len(adj):>5d} | {m:+10.1f}")
            else:
                doc.append(f"{cat+' '+label:>20s} | {len(adj):>5d} | {'N/A(<10)':>10s}")

    # ══════════════════════════════════════════════════════════
    # 稳健性：时间分半
    # ══════════════════════════════════════════════════════════
    doc.append(f"\n## 稳健性：时间分半\n")
    all_dates = sorted(low_sig['date'].unique())
    mid = len(all_dates) // 2
    first_half_dates = set(all_dates[:mid])
    second_half_dates = set(all_dates[mid:])

    doc.append(f"前半: {len(first_half_dates)}天, 后半: {len(second_half_dates)}天\n")
    doc.append(f"{'半段':>6s} | {'Fwd':>4s} | {'Pos-Neg差':>10s} | {'N':>5s}")
    doc.append("-" * 35)

    for half_name, date_set in [('前半', first_half_dates), ('后半', second_half_dates)]:
        half = low_sig[low_sig['date'].isin(date_set)]
        for n in [12, 15, 18]:
            col = f'fwd{n}'
            pos = half[half['factor'] > 0][col].dropna()
            neg = half[half['factor'] < 0][col].dropna()
            if len(pos) >= 10 and len(neg) >= 10:
                sp = pos.mean() - neg.mean()
                doc.append(f"{half_name:>6s} | {n:>4d} | {sp:+10.1f} | {len(pos)+len(neg):>5d}")
            else:
                doc.append(f"{half_name:>6s} | {n:>4d} | {'N/A':>10s} | {len(pos)+len(neg):>5d}")

    # ══════════════════════════════════════════════════════════
    # 结论
    # ══════════════════════════════════════════════════════════
    doc.append(f"\n## 结论\n")
    doc.append("（基于数据自动填充）")

    report = "\n".join(doc)
    path = Path("tmp") / "horizontal_reversal_deep_dive.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
