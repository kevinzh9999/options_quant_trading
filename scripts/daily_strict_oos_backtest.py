#!/usr/bin/env python3
"""Strict OOS Backtest + Slippage 敏感性

A. Strict OOS:
   - Train ONCE on 2022-07 to 2024-12 (factor design + hyperparams 全 frozen)
   - Predict on 2025-01 onwards, no retrain
   - 验证 walk-forward 结果 robust

B. Slippage sweep:
   - 在 strict-OOS 设定下，跑 slippage = {0.05, 0.08, 0.12, 0.16, 0.20, 0.25}%
   - 找到 break-even slippage（策略归零的临界点）

Hypothesis check:
   - 如果 strict-OOS IC ≈ walk-forward IC → 信号 robust
   - 如果 strict-OOS IC <<<< walk-forward → 之前 partial overfit
   - 如果 0.20% 滑点下仍正 → 实盘 viable
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

HOLD_DAYS = 5
TOP_PCT = 0.20
BOT_PCT = 0.20
CONTRACT_MULT = 200

TRAIN_END = "20241231"   # train 截止
TEST_START = "20250101"  # test 起始


def run_backtest(predictions: np.ndarray, dates: list, closes: np.ndarray,
                 thresholds: tuple, slippage_pct: float,
                 hold_days: int = HOLD_DAYS):
    """给定预测序列 + 阈值 + slippage，跑 long-short backtest。返回 trades + equity."""
    top_thr, bot_thr = thresholds
    trades = []
    daily_pnl = np.zeros(len(dates))

    for td_idx in range(len(dates) - hold_days):
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
        exit_idx = td_idx + hold_days
        exit_close = closes[exit_idx]
        gross_ret = (exit_close - entry_close) / entry_close
        if direction == "SHORT":
            gross_ret = -gross_ret
        net_ret = gross_ret - slippage_pct
        pnl_pts = net_ret * entry_close
        daily_pnl[exit_idx] += pnl_pts
        trades.append({
            "entry_date": dates[td_idx],
            "exit_date": dates[exit_idx],
            "direction": direction,
            "pred": pred,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "pnl_pts": pnl_pts,
            "pnl_yuan": pnl_pts * CONTRACT_MULT,
        })

    return pd.DataFrame(trades), daily_pnl


def equity_metrics(daily_pnl, dates, train_end_idx):
    """Compute Sharpe / MaxDD / annualized PnL on test segment."""
    test_pnl = daily_pnl[train_end_idx:]
    cum = np.cumsum(test_pnl) * CONTRACT_MULT
    if len(cum) == 0:
        return {}
    final = cum[-1]
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = dd.min()

    days_arr = test_pnl[test_pnl != 0]
    sharpe = (days_arr.mean() / days_arr.std() * np.sqrt(252)
              if len(days_arr) > 1 and days_arr.std() > 0 else 0)
    n_yrs = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[train_end_idx])).days / 365.0
    ann = final / max(n_yrs, 0.1)
    return {
        "final_yuan": final,
        "ann_yuan": ann,
        "max_dd_yuan": max_dd,
        "sharpe": sharpe,
        "n_yrs": n_yrs,
    }


def main():
    ctx = load_default_context(DB_PATH)
    px = ctx.px_df.copy()
    iv_dates = set(ctx.iv_df["trade_date"].tolist())
    px_e = px[px["trade_date"].isin(iv_dates)].reset_index(drop=True)
    dates = px_e["trade_date"].tolist()
    closes = px_e["close"].astype(float).values

    print(f"Eligible days: {len(px_e)}")
    print(f"Range: {dates[0]} ~ {dates[-1]}")

    # Boundary indices
    train_end_idx = next(
        (i for i, d in enumerate(dates) if d > TRAIN_END), len(dates))
    test_start_idx = next(
        (i for i, d in enumerate(dates) if d >= TEST_START), len(dates))
    print(f"Train end (last train day): {dates[train_end_idx-1]} (idx {train_end_idx-1})")
    print(f"Test start: {dates[test_start_idx]} (idx {test_start_idx})")
    print(f"Train days: {train_end_idx}  Test days: {len(dates) - test_start_idx}")

    # Compute features once
    pipeline = build_default_pipeline()
    print(f"\nFactors: {len(pipeline.factors)}")
    print("Computing features...")
    X_full = pipeline.features_matrix(dates, ctx)

    # Forward returns target
    fwd_ret = np.array([
        closes[i + HOLD_DAYS] / closes[i] - 1 if i + HOLD_DAYS < len(closes) else np.nan
        for i in range(len(closes))
    ])

    # ── A. Strict OOS train ──
    # Train data: indices [0, train_end_idx - HOLD_DAYS) (target must have realized)
    train_cutoff = train_end_idx - HOLD_DAYS
    X_tr = X_full[:train_cutoff]
    y_tr = fwd_ret[:train_cutoff]
    valid = ~(np.isnan(X_tr).any(axis=1) | np.isnan(y_tr))
    X_tr_v = X_tr[valid]
    y_tr_v = y_tr[valid]
    print(f"\n=== Train (frozen) ===")
    print(f"  Train samples: {len(X_tr_v)} (clean) / {train_cutoff} (raw)")
    print(f"  y mean: {y_tr_v.mean():+.4f}  std: {y_tr_v.std():.4f}")

    pipeline.train(X_tr_v, y_tr_v)
    pred_tr = pipeline.predict(X_tr_v)
    ic_tr = np.corrcoef(pred_tr, y_tr_v)[0, 1]
    print(f"  Train IC: {ic_tr:+.4f}")

    # Thresholds from in-sample (frozen)
    top_thr = float(np.quantile(pred_tr, 1 - TOP_PCT))
    bot_thr = float(np.quantile(pred_tr, BOT_PCT))
    print(f"  Thresholds (in-sample P{(1-TOP_PCT)*100:.0f} / P{BOT_PCT*100:.0f}): "
          f"{top_thr:+.4f} / {bot_thr:+.4f}")

    # Predict on full series (we'll evaluate test period)
    predictions = np.full(len(dates), np.nan)
    valid_full = ~np.isnan(X_full).any(axis=1)
    predictions[valid_full] = pipeline.predict(X_full[valid_full])

    # ── OOS IC ──
    test_pred = predictions[test_start_idx:]
    test_y = fwd_ret[test_start_idx:]
    valid_te = ~(np.isnan(test_pred) | np.isnan(test_y))
    if valid_te.sum() > 0:
        ic_te = np.corrcoef(test_pred[valid_te], test_y[valid_te])[0, 1]
        print(f"\n=== Strict OOS IC ===")
        print(f"  Test samples: {valid_te.sum()}")
        print(f"  OOS IC: {ic_te:+.4f}")
        print(f"  vs Train IC: {ic_tr:+.4f}  (gap: {ic_tr - ic_te:+.4f})")

        # Decile breakdown
        df_t = pd.DataFrame({
            "pred": test_pred[valid_te],
            "y": test_y[valid_te],
        })
        df_t["dec"] = pd.qcut(df_t["pred"], 10, labels=False, duplicates="drop")
        print(f"\n  Test decile breakdown:")
        print(f"  {'Decile':<6} {'N':>4} {'pred mean':>12} {'actual mean':>13} {'win %':>7}")
        for d, sub in df_t.groupby("dec", observed=True):
            wr = (sub["y"] > 0).mean() * 100
            print(f"  D{int(d)+1:<5} {len(sub):>4} {sub['pred'].mean():>+12.4f} "
                  f"{sub['y'].mean():>+13.4f} {wr:>6.0f}%")

        top_y = df_t[df_t["dec"] == 9]["y"].mean()
        bot_y = df_t[df_t["dec"] == 0]["y"].mean()
        print(f"\n  Long-Short spread: {(top_y - bot_y)*100:+.3f}pp")

    # ── B. Slippage sweep ──
    print(f"\n{'=' * 75}")
    print(f" Slippage Sensitivity（在 strict-OOS test 上跑 backtest）")
    print(f"{'=' * 75}")
    print(f"{'Slippage':>10} {'N trades':>10} {'WR':>6} {'Avg/笔':>9} "
          f"{'OOS Total':>12} {'年化':>10} {'Sharpe':>7} {'MaxDD':>10}")
    print("-" * 90)
    for slp_bps in [5, 8, 12, 16, 20, 25, 30, 40]:
        slp = slp_bps / 10000
        trades_df, daily_pnl = run_backtest(
            predictions, dates, closes,
            (top_thr, bot_thr), slp, HOLD_DAYS,
        )
        # Filter to test period only
        test_trades = trades_df[trades_df["entry_date"] >= TEST_START]
        if test_trades.empty:
            continue
        m = equity_metrics(daily_pnl, dates, test_start_idx)
        wr = (test_trades["net_ret"] > 0).mean() * 100
        avg_ret = test_trades["net_ret"].mean() * 100
        print(f"  {slp*100:.2f}%   {len(test_trades):>10} {wr:>5.1f}% "
              f"{avg_ret:>+8.3f}% {m['final_yuan']:>+12,.0f} "
              f"{m['ann_yuan']:>+10,.0f} {m['sharpe']:>7.2f} "
              f"{m['max_dd_yuan']:>+10,.0f}")

    # ── Feature importance（frozen model）──
    print(f"\n=== Frozen model feature importance ===")
    imp = pipeline.feature_importance()
    for _, r in imp.iterrows():
        bar = "█" * int(r["importance"] * 80)
        print(f"  [{r['category']:<5}] {r['feature']:<25} {r['importance']:.4f}  {bar}")


if __name__ == "__main__":
    main()
