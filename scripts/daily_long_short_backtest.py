#!/usr/bin/env python3
"""Daily Long-Short Backtest — 因子化 + walk-forward retrain

策略：
  1. 每个 td 用截至 td-1 的所有数据训 pipeline（rolling retrain，每 N 天一次以省时间）
  2. 预测 next-5d return
  3. 当日预测分数：高分→LONG, 低分→SHORT, 中间→空仓
  4. 阈值：滚动训练集的 P80 / P20
  5. 持仓 5 天，每日开新仓 + 关到期仓
  6. 现货 close-to-close 计 PnL，乘 contract mult，扣滑点

输出:
  - 累计 PnL 曲线（CSV）
  - 年化收益、Sharpe、Max DD
  - Long-only / Short-only / Long-Short 三种模式对比
  - 月度 PnL breakdown
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
    build_default_pipeline, load_default_context, add_forward_returns,
)

DB_PATH = "/Users/kevinzhao/Documents/options_quant_trading/data/storage/trading.db"

# Backtest config
HOLD_DAYS = 5             # 持仓天数（与最佳 horizon 匹配）
RETRAIN_EVERY = 20        # rolling retrain 间隔（天）
INITIAL_TRAIN_DAYS = 200  # 初始训练样本数
TOP_PCT = 0.20            # top 20% LONG
BOT_PCT = 0.20            # bot 20% SHORT
CONTRACT_MULT = 200       # IM 乘数（用于换算元）
SLIPPAGE_PCT = 0.0008     # 0.08% per round trip (~5-7pt 在 8000 价位)


def main():
    ctx = load_default_context(DB_PATH)
    px = ctx.px_df.copy()

    # 只用有 IV 数据的日期
    iv_dates = set(ctx.iv_df["trade_date"].tolist())
    eligible_mask = px["trade_date"].isin(iv_dates)
    px_e = px[eligible_mask].reset_index(drop=True)
    print(f"Eligible days: {len(px_e)}")
    print(f"Date range: {px_e['trade_date'].iloc[0]} ~ {px_e['trade_date'].iloc[-1]}")

    dates = px_e["trade_date"].tolist()
    closes = px_e["close"].astype(float).values

    # 预先算 features matrix (一次)
    pipeline_proto = build_default_pipeline()
    print(f"\nFactors: {len(pipeline_proto.factors)}")
    print("Computing feature matrix for all dates...")
    X_full = pipeline_proto.features_matrix(dates, ctx)
    print(f"  shape={X_full.shape}")

    # 每日 fwd ret target (训练用，只能用 t-HOLD_DAYS 之前的数据)
    fwd_ret = np.array([
        closes[i + HOLD_DAYS] / closes[i] - 1 if i + HOLD_DAYS < len(closes) else np.nan
        for i in range(len(closes))
    ])

    # Walk-forward backtest
    # 在 td_idx 时，训练数据是 [0, td_idx - HOLD_DAYS - 1] 的 features+target
    # （-HOLD_DAYS 是为了保证 target 已经实现）
    # Predict 时用 td_idx 的 features，预测 next-5d return
    # 然后从 td_idx 开始持仓 5 天

    positions = []  # [(entry_idx, exit_idx, direction)]
    daily_pnl = np.zeros(len(dates))  # PnL 在 exit_idx 那天 realize

    last_train_idx = -RETRAIN_EVERY  # force first train
    cached_pipe = None
    cached_thresholds = None

    print(f"\nBacktest config: HOLD={HOLD_DAYS}d  retrain_every={RETRAIN_EVERY}d  "
          f"init_train={INITIAL_TRAIN_DAYS}  top/bot={TOP_PCT*100:.0f}%/{BOT_PCT*100:.0f}%")

    n_long, n_short = 0, 0
    for td_idx in range(INITIAL_TRAIN_DAYS, len(dates) - HOLD_DAYS):
        td = dates[td_idx]

        # Retrain？
        if td_idx - last_train_idx >= RETRAIN_EVERY:
            train_end = td_idx - HOLD_DAYS  # 最后一个有 realized target 的样本
            if train_end < INITIAL_TRAIN_DAYS:
                continue
            X_tr = X_full[:train_end]
            y_tr = fwd_ret[:train_end]
            valid = ~(np.isnan(X_tr).any(axis=1) | np.isnan(y_tr))
            if valid.sum() < INITIAL_TRAIN_DAYS:
                continue
            X_tr_v = X_tr[valid]
            y_tr_v = y_tr[valid]

            cached_pipe = build_default_pipeline()
            cached_pipe.train(X_tr_v, y_tr_v)
            # In-sample preds → percentile thresholds
            pred_in = cached_pipe.predict(X_tr_v)
            cached_thresholds = (
                np.quantile(pred_in, 1 - TOP_PCT),
                np.quantile(pred_in, BOT_PCT),
            )
            last_train_idx = td_idx

        # Predict for today
        x = X_full[td_idx:td_idx+1]
        if np.isnan(x).any():
            continue
        pred = float(cached_pipe.predict(x)[0])
        top_thr, bot_thr = cached_thresholds

        # Decide direction
        direction = None
        if pred >= top_thr:
            direction = "LONG"
        elif pred <= bot_thr:
            direction = "SHORT"
        else:
            continue

        # Open position at today's close (entry), exit at td_idx + HOLD_DAYS close
        entry_close = closes[td_idx]
        exit_idx = td_idx + HOLD_DAYS
        exit_close = closes[exit_idx]
        gross_ret = (exit_close - entry_close) / entry_close
        if direction == "SHORT":
            gross_ret = -gross_ret
        net_ret = gross_ret - SLIPPAGE_PCT  # 单边滑点视作来回总成本

        # Convert to points (% × close → points × mult to convert to 元)
        pnl_pts = net_ret * entry_close
        # Realized on exit day
        daily_pnl[exit_idx] += pnl_pts

        positions.append({
            "entry_date": td,
            "exit_date": dates[exit_idx],
            "direction": direction,
            "entry": entry_close,
            "exit": exit_close,
            "pred": pred,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "pnl_pts": pnl_pts,
            "pnl_yuan": pnl_pts * CONTRACT_MULT,
        })
        if direction == "LONG":
            n_long += 1
        else:
            n_short += 1

    pos_df = pd.DataFrame(positions)
    if pos_df.empty:
        print("No trades")
        return

    print(f"\n=== Trade summary ===")
    print(f"  Total: {len(pos_df)}  LONG: {n_long}  SHORT: {n_short}")
    print(f"  WR (net): {(pos_df['net_ret'] > 0).mean() * 100:.1f}%")
    print(f"  Avg net return per trade: {pos_df['net_ret'].mean()*100:+.3f}%")
    print(f"  Total gross PnL pts: {pos_df['pnl_pts'].sum()/(1-len(pos_df)*SLIPPAGE_PCT/pos_df['gross_ret'].abs().sum()):+.0f}")
    print(f"  Total NET PnL pts: {pos_df['pnl_pts'].sum():+.0f}  "
          f"= {pos_df['pnl_yuan'].sum():+,.0f} 元 (1手 / no compound)")

    # Long-only / Short-only / Long-Short
    print(f"\n=== Mode breakdown ===")
    for label, sub in [("LONG only", pos_df[pos_df["direction"] == "LONG"]),
                        ("SHORT only", pos_df[pos_df["direction"] == "SHORT"]),
                        ("Long-Short", pos_df)]:
        if len(sub) == 0:
            continue
        wr = (sub["net_ret"] > 0).mean() * 100
        avg = sub["net_ret"].mean() * 100
        tot_pnl = sub["pnl_yuan"].sum()
        # Annualized: total period
        first_d = sub["entry_date"].iloc[0]
        last_d = sub["exit_date"].iloc[-1]
        n_years = (pd.Timestamp(last_d) - pd.Timestamp(first_d)).days / 365.0
        ann_pnl = tot_pnl / max(n_years, 0.1)
        print(f"  {label:<12}  N={len(sub):>4}  WR={wr:.1f}%  "
              f"avg={avg:+.3f}%  total={tot_pnl:+,.0f}元  年化={ann_pnl:+,.0f}元")

    # Equity curve & drawdown (按 exit_date 聚合)
    eq_df = pd.DataFrame({"date": dates, "daily_pnl": daily_pnl})
    eq_df["cum_pnl_pts"] = eq_df["daily_pnl"].cumsum()
    eq_df["cum_pnl_yuan"] = eq_df["cum_pnl_pts"] * CONTRACT_MULT
    eq_df["peak"] = eq_df["cum_pnl_yuan"].cummax()
    eq_df["dd"] = eq_df["cum_pnl_yuan"] - eq_df["peak"]

    print(f"\n=== Equity 指标 ===")
    final = eq_df["cum_pnl_yuan"].iloc[-1]
    n_trade_days = (eq_df["daily_pnl"] != 0).sum()
    n_calendar_yrs = (
        pd.Timestamp(eq_df["date"].iloc[-1]) - pd.Timestamp(eq_df["date"].iloc[INITIAL_TRAIN_DAYS])
    ).days / 365.0
    ann_pnl = final / max(n_calendar_yrs, 0.1)
    max_dd = eq_df["dd"].min()
    # Sharpe（粗略）：daily PnL / std × sqrt(252)
    daily_arr = eq_df["daily_pnl"].iloc[INITIAL_TRAIN_DAYS:].values
    daily_arr = daily_arr[daily_arr != 0] if len(daily_arr[daily_arr != 0]) > 0 else daily_arr
    if len(daily_arr) > 1 and daily_arr.std() > 0:
        sharpe = daily_arr.mean() / daily_arr.std() * np.sqrt(252)
    else:
        sharpe = 0
    print(f"  最终 PnL: {final:+,.0f} 元 (1手)")
    print(f"  年化: {ann_pnl:+,.0f} 元/年 ({n_calendar_yrs:.1f} 年期间)")
    print(f"  Max DD: {max_dd:+,.0f} 元")
    print(f"  Sharpe（粗略）: {sharpe:.2f}")
    print(f"  Trade days: {n_trade_days}")

    # 月度
    print(f"\n=== 月度 PnL ===")
    pos_df["month"] = pd.to_datetime(pos_df["exit_date"]).dt.to_period("M").astype(str)
    monthly = pos_df.groupby("month").agg(
        n=("pnl_yuan", "count"),
        pnl_yuan=("pnl_yuan", "sum"),
        wr=("net_ret", lambda x: (x > 0).mean() * 100),
    )
    print(f"{'month':<10} {'N':>4} {'WR':>7} {'PnL 元':>12}")
    for m, r in monthly.iterrows():
        print(f"  {m:<10} {int(r['n']):>4} {r['wr']:>6.0f}% {r['pnl_yuan']:>+12,.0f}")

    # Save outputs
    pos_df.to_csv("/tmp/daily_ls_trades.csv", index=False)
    eq_df.to_csv("/tmp/daily_ls_equity.csv", index=False)
    print(f"\n  Saved: /tmp/daily_ls_trades.csv, /tmp/daily_ls_equity.csv")


if __name__ == "__main__":
    main()
