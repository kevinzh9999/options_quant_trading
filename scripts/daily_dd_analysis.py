#!/usr/bin/env python3
"""Max DD 来源分析

加载 strict-OOS 的预测 + 跑 backtest，然后细看：
  1. equity curve 时序，找 DD 阶段
  2. 每个 DD 阶段的 trade list
  3. 那段是 LONG/SHORT 哪边占比？
  4. 模型预测分数（pred）是否极端？
  5. 是否一致性失败（连续 N 笔亏）还是单次大亏？
  6. 那段 IV regime 怎么样？(iv level, rv level)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from strategies.daily.factors import (
    build_default_pipeline, load_default_context,
)

DB_PATH = "/Users/kevinzhao/Documents/options_quant_trading/data/storage/trading.db"
HOLD_DAYS = 5
CONTRACT_MULT = 200
SLIPPAGE_PCT = 0.0008
TOP_PCT = 0.20
BOT_PCT = 0.20
TRAIN_END = "20241231"


def main():
    ctx = load_default_context(DB_PATH)
    px = ctx.px_df.copy()
    iv_dates = set(ctx.iv_df["trade_date"].tolist())
    px_e = px[px["trade_date"].isin(iv_dates)].reset_index(drop=True)
    dates = px_e["trade_date"].tolist()
    closes = px_e["close"].astype(float).values

    train_end_idx = next((i for i, d in enumerate(dates) if d > TRAIN_END), len(dates))

    pipeline = build_default_pipeline()
    X_full = pipeline.features_matrix(dates, ctx)
    fwd_ret = np.array([
        closes[i + HOLD_DAYS] / closes[i] - 1 if i + HOLD_DAYS < len(closes) else np.nan
        for i in range(len(closes))
    ])

    train_cutoff = train_end_idx - HOLD_DAYS
    valid = ~(np.isnan(X_full[:train_cutoff]).any(axis=1) | np.isnan(fwd_ret[:train_cutoff]))
    pipeline.train(X_full[:train_cutoff][valid], fwd_ret[:train_cutoff][valid])
    pred_tr = pipeline.predict(X_full[:train_cutoff][valid])
    top_thr = float(np.quantile(pred_tr, 1 - TOP_PCT))
    bot_thr = float(np.quantile(pred_tr, BOT_PCT))

    predictions = np.full(len(dates), np.nan)
    valid_full = ~np.isnan(X_full).any(axis=1)
    predictions[valid_full] = pipeline.predict(X_full[valid_full])

    # Backtest
    trades = []
    daily_pnl = np.zeros(len(dates))
    for td_idx in range(len(dates) - HOLD_DAYS):
        pred = predictions[td_idx]
        if np.isnan(pred):
            continue
        if pred >= top_thr:
            direction = "LONG"
        elif pred <= bot_thr:
            direction = "SHORT"
        else:
            continue
        entry_close = closes[td_idx]
        exit_idx = td_idx + HOLD_DAYS
        exit_close = closes[exit_idx]
        gross_ret = (exit_close - entry_close) / entry_close
        if direction == "SHORT":
            gross_ret = -gross_ret
        net_ret = gross_ret - SLIPPAGE_PCT
        pnl_pts = net_ret * entry_close
        daily_pnl[exit_idx] += pnl_pts
        trades.append({
            "entry_idx": td_idx,
            "exit_idx": exit_idx,
            "entry_date": dates[td_idx],
            "exit_date": dates[exit_idx],
            "direction": direction,
            "pred": pred,
            "entry_close": entry_close,
            "exit_close": exit_close,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "pnl_pts": pnl_pts,
            "pnl_yuan": pnl_pts * CONTRACT_MULT,
        })
    trades_df = pd.DataFrame(trades)
    print(f"Total trades: {len(trades_df)}")

    # Equity curve
    cum_yuan = np.cumsum(daily_pnl) * CONTRACT_MULT
    peak = np.maximum.accumulate(cum_yuan)
    dd = cum_yuan - peak

    # ════════════════════════════════════════════════════════════
    # 1. Equity / DD timeline
    # ════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(" 1. Equity & DD timeline (key milestones)")
    print(f"{'=' * 70}")
    eq_df = pd.DataFrame({
        "date": dates, "daily_pnl_pts": daily_pnl,
        "cum_yuan": cum_yuan, "peak_yuan": peak, "dd_yuan": dd,
    })
    # Only test period (after train end)
    test_eq = eq_df.iloc[train_end_idx:].reset_index(drop=True)
    print(f"  Test 起始日: {test_eq['date'].iloc[0]}, "
          f"final cum: {test_eq['cum_yuan'].iloc[-1]:+,.0f}元")

    # Find drawdown periods (DD 阶段)
    # 定义：DD 开始 = peak 之后第一日 PnL < 0；结束 = 创新高那天
    dd_periods = []
    in_dd = False
    dd_start_idx = None
    dd_min_idx = None
    dd_min_val = 0
    test_dd_arr = test_eq["dd_yuan"].values
    test_cum_arr = test_eq["cum_yuan"].values
    test_dates_arr = test_eq["date"].values
    for i in range(len(test_eq)):
        if test_dd_arr[i] < 0 and not in_dd:
            in_dd = True
            dd_start_idx = i
            dd_min_idx = i
            dd_min_val = test_dd_arr[i]
        elif test_dd_arr[i] < 0 and in_dd:
            if test_dd_arr[i] < dd_min_val:
                dd_min_val = test_dd_arr[i]
                dd_min_idx = i
        elif test_dd_arr[i] >= 0 and in_dd:
            in_dd = False
            dd_periods.append({
                "start_date": test_dates_arr[dd_start_idx],
                "trough_date": test_dates_arr[dd_min_idx],
                "end_date": test_dates_arr[i],
                "depth_yuan": dd_min_val,
                "n_days": i - dd_start_idx + 1,
            })
    if in_dd:
        dd_periods.append({
            "start_date": test_dates_arr[dd_start_idx],
            "trough_date": test_dates_arr[dd_min_idx],
            "end_date": test_dates_arr[-1] + " (ongoing)",
            "depth_yuan": dd_min_val,
            "n_days": len(test_eq) - dd_start_idx,
        })

    dd_df = pd.DataFrame(dd_periods).sort_values("depth_yuan")
    print(f"\n  DD 阶段（按深度排序，前 5）:")
    print(f"  {'start':<12} {'trough':<12} {'end':<22} {'depth (元)':>14} {'days':>5}")
    for _, r in dd_df.head(5).iterrows():
        print(f"  {r['start_date']:<12} {r['trough_date']:<12} {str(r['end_date']):<22} "
              f"{r['depth_yuan']:>+14,.0f} {r['n_days']:>5}")

    # ════════════════════════════════════════════════════════════
    # 2. Top 3 worst DD periods - drill down
    # ════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(" 2. 最深的 3 个 DD 阶段细节")
    print(f"{'=' * 70}")
    top3 = dd_df.head(3)
    for _, dd_row in top3.iterrows():
        s_date = dd_row["start_date"]
        e_date = str(dd_row["end_date"]).split(" ")[0]  # remove "(ongoing)"
        period_trades = trades_df[
            (trades_df["entry_date"] >= s_date) & (trades_df["entry_date"] <= e_date)
        ].copy()
        print(f"\n  ─ {s_date} ~ {e_date} (depth {dd_row['depth_yuan']:+,.0f}元) ─")
        print(f"    DD 期间 trades: {len(period_trades)}")
        if not period_trades.empty:
            n_long = (period_trades["direction"] == "LONG").sum()
            n_short = (period_trades["direction"] == "SHORT").sum()
            wr = (period_trades["net_ret"] > 0).mean() * 100
            tot_pnl = period_trades["pnl_yuan"].sum()
            avg_ret = period_trades["net_ret"].mean() * 100
            print(f"    LONG/SHORT: {n_long}/{n_short}  WR: {wr:.1f}%  "
                  f"Avg: {avg_ret:+.3f}%  PnL: {tot_pnl:+,.0f}")

            # 最大 5 笔亏
            worst5 = period_trades.nsmallest(5, "pnl_yuan")
            print(f"    最大 5 笔亏:")
            print(f"    {'entry':<12} {'dir':<6} {'pred':>9} {'gross':>9} {'net':>9} {'PnL元':>11}")
            for _, t in worst5.iterrows():
                print(f"    {t['entry_date']:<12} {t['direction']:<6} "
                      f"{t['pred']*100:>+8.2f}% {t['gross_ret']*100:>+8.2f}% "
                      f"{t['net_ret']*100:>+8.2f}% {t['pnl_yuan']:>+11,.0f}")

            # IV regime during this period
            iv_during = ctx.iv_df[
                (ctx.iv_df["trade_date"] >= s_date) & (ctx.iv_df["trade_date"] <= e_date)
            ]
            if not iv_during.empty:
                print(f"    Regime: avg IV={iv_during['atm_iv_market'].mean()*100:.1f}%  "
                      f"avg RV20={iv_during['realized_vol_20d'].mean()*100:.1f}%  "
                      f"avg RR={iv_during['rr_25d'].mean()*100:+.2f}pp  "
                      f"avg VRP={iv_during['vrp'].mean()*100:+.2f}pp")

    # ════════════════════════════════════════════════════════════
    # 3. By direction split
    # ════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(" 3. LONG vs SHORT 分组统计（test period）")
    print(f"{'=' * 70}")
    test_trades = trades_df[trades_df["entry_date"] >= dates[train_end_idx]]
    for d in ["LONG", "SHORT"]:
        sub = test_trades[test_trades["direction"] == d]
        if sub.empty:
            continue
        # Compute per-direction equity
        d_daily = np.zeros(len(dates))
        for _, t in sub.iterrows():
            d_daily[int(t["exit_idx"])] += t["pnl_pts"]
        d_cum = np.cumsum(d_daily) * CONTRACT_MULT
        d_peak = np.maximum.accumulate(d_cum)
        d_dd = d_cum - d_peak
        max_dd_d = d_dd.min()
        final_d = d_cum[-1]
        wr = (sub["net_ret"] > 0).mean() * 100
        print(f"  {d}:  N={len(sub)}  WR={wr:.1f}%  "
              f"Final={final_d:+,.0f}元  MaxDD={max_dd_d:+,.0f}元  "
              f"DD/Final={abs(max_dd_d)/max(abs(final_d),1)*100:.0f}%")
        # Worst 5 trades
        worst = sub.nsmallest(5, "pnl_yuan")
        print(f"    最大 5 笔亏:")
        for _, t in worst.iterrows():
            print(f"    {t['entry_date']} {t['direction']} pred={t['pred']*100:+.2f}% "
                  f"gross={t['gross_ret']*100:+.2f}% PnL={t['pnl_yuan']:+,.0f}")

    # ════════════════════════════════════════════════════════════
    # 4. Win/loss distribution
    # ════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(" 4. PnL 分布")
    print(f"{'=' * 70}")
    test_pnl_arr = test_trades["pnl_yuan"].values
    print(f"  N={len(test_pnl_arr)}")
    print(f"  Mean: {test_pnl_arr.mean():+,.0f}  Median: {np.median(test_pnl_arr):+,.0f}")
    print(f"  Std:  {test_pnl_arr.std():,.0f}")
    print(f"  Max win:  {test_pnl_arr.max():+,.0f}")
    print(f"  Max loss: {test_pnl_arr.min():+,.0f}")
    print(f"  P5/P25/P50/P75/P95:")
    for p in [5, 25, 50, 75, 95]:
        v = np.percentile(test_pnl_arr, p)
        print(f"    P{p:<3}: {v:+,.0f}")

    # 最坏单日 PnL
    print(f"\n  最坏 5 个单日 PnL（accumulated 多笔）:")
    test_daily_pnl = daily_pnl[train_end_idx:] * CONTRACT_MULT
    test_dates_arr2 = np.array(dates[train_end_idx:])
    bad_days = np.argsort(test_daily_pnl)[:5]
    for i in bad_days:
        if test_daily_pnl[i] < 0:
            print(f"    {test_dates_arr2[i]}: {test_daily_pnl[i]:+,.0f}元")


if __name__ == "__main__":
    main()
