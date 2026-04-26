#!/usr/bin/env python3
"""Walk-forward 验证 A+D 组合 (ATR-scaled SL + Regime features)

每 RETRAIN_EVERY 天用至 td-1 的全部数据 retrain 含 regime 因子的 pipeline。
ATR(20)×k 作 per-trade 止损。

输出:
  - Walk-forward 整段 PnL/MaxDD/Sharpe/Calmar
  - 与 strict-OOS A+D 对比 (检查 retrain 是否带来 robustness 或反而 overfit)
  - 与 baseline walk-forward 对比 (检查 A+D 增量)
  - 月度 PnL breakdown
  - SHORT/LONG 分解
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
    DailyFactor, DailyFactorPipeline,
    build_default_pipeline, load_default_context,
)
from scripts.daily_robust_methods_compare import (
    CloseSma60Factor, Slope60dFactor, VolRegimeFactor,
    compute_atr20, simulate_trade,
)

DB_PATH = "/Users/kevinzhao/Documents/options_quant_trading/data/storage/trading.db"

HOLD_DAYS = 5
RETRAIN_EVERY = 20
INITIAL_TRAIN_DAYS = 200
TOP_PCT = 0.20
BOT_PCT = 0.20
ATR_K = 1.5
CONTRACT_MULT = 200
SLIPPAGE_PCT = 0.0008


def build_pipeline_with_regime():
    base = build_default_pipeline()
    base.factors.extend([CloseSma60Factor(), Slope60dFactor(), VolRegimeFactor()])
    base.feature_names = [f.name for f in base.factors]
    return base


def run_walkforward(label: str, pipe_builder, ctx, dates, closes, highs, lows,
                     atr20, use_atr_sl: bool):
    n = len(dates)
    fwd = np.array([
        closes[i + HOLD_DAYS] / closes[i] - 1 if i + HOLD_DAYS < n else np.nan
        for i in range(n)
    ])
    pipe_proto = pipe_builder()
    print(f"  Computing X_full ({pipe_proto.feature_names}) ...")
    X_full = pipe_proto.features_matrix(dates, ctx)
    print(f"  X_full shape: {X_full.shape}")

    cached_pipe = None
    cached_thr = None
    last_train = -RETRAIN_EVERY
    trades = []
    daily_pnl = np.zeros(n)
    n_retrains = 0
    train_ic_history = []

    for td_idx in range(INITIAL_TRAIN_DAYS, n - HOLD_DAYS):
        # Retrain?
        if td_idx - last_train >= RETRAIN_EVERY:
            train_end = td_idx - HOLD_DAYS
            X_tr = X_full[:train_end]
            y_tr = fwd[:train_end]
            valid = ~(np.isnan(X_tr).any(axis=1) | np.isnan(y_tr))
            if valid.sum() < INITIAL_TRAIN_DAYS:
                continue
            X_tr_v = X_tr[valid]
            y_tr_v = y_tr[valid]
            cached_pipe = pipe_builder()
            cached_pipe.train(X_tr_v, y_tr_v)
            pred_in = cached_pipe.predict(X_tr_v)
            cached_thr = (
                float(np.quantile(pred_in, 1 - TOP_PCT)),
                float(np.quantile(pred_in, BOT_PCT)),
            )
            train_ic_history.append(np.corrcoef(pred_in, y_tr_v)[0, 1])
            last_train = td_idx
            n_retrains += 1

        if cached_pipe is None:
            continue

        x = X_full[td_idx:td_idx+1]
        if np.isnan(x).any():
            continue
        pred = float(cached_pipe.predict(x)[0])
        top_thr, bot_thr = cached_thr
        if pred >= top_thr:
            direction = "LONG"
        elif pred <= bot_thr:
            direction = "SHORT"
        else:
            continue

        sl = ATR_K * atr20[td_idx] if (use_atr_sl and not np.isnan(atr20[td_idx])) else None

        exit_idx, gross_ret, reason = simulate_trade(
            direction, td_idx, closes, highs, lows,
            hold_days=HOLD_DAYS, sl_pct=sl,
            slippage=SLIPPAGE_PCT,
        )
        net_ret = gross_ret - SLIPPAGE_PCT
        pnl_pts = net_ret * closes[td_idx]
        daily_pnl[exit_idx] += pnl_pts
        trades.append({
            "entry_date": dates[td_idx], "exit_date": dates[exit_idx],
            "direction": direction, "pred": pred,
            "gross_ret": gross_ret, "net_ret": net_ret,
            "pnl_pts": pnl_pts, "pnl_yuan": pnl_pts * CONTRACT_MULT,
            "reason": reason, "hold": exit_idx - td_idx,
        })

    return _summarize(label, trades, daily_pnl, dates, n_retrains, train_ic_history)


def _summarize(label, trades, daily_pnl, dates, n_retrains, train_ics):
    if not trades:
        return {"label": label, "n": 0}
    df = pd.DataFrame(trades)

    eval_start = INITIAL_TRAIN_DAYS
    cum = np.cumsum(daily_pnl[eval_start:]) * CONTRACT_MULT
    final = cum[-1]
    peak = np.maximum.accumulate(cum)
    max_dd = (cum - peak).min()

    days_arr = daily_pnl[eval_start:]
    days_arr = days_arr[days_arr != 0]
    sharpe = (days_arr.mean() / days_arr.std() * np.sqrt(252)
              if len(days_arr) > 1 and days_arr.std() > 0 else 0)
    n_yrs = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[eval_start])).days / 365.0
    ann = final / max(n_yrs, 0.1)

    return {
        "label": label,
        "n": len(df),
        "n_retrains": n_retrains,
        "train_ic_mean": np.mean(train_ics) if train_ics else 0,
        "train_ic_std": np.std(train_ics) if train_ics else 0,
        "wr": (df["net_ret"] > 0).mean() * 100,
        "avg_pct": df["net_ret"].mean() * 100,
        "worst_pct": df["net_ret"].min() * 100,
        "best_pct": df["net_ret"].max() * 100,
        "final_yuan": final,
        "ann_yuan": ann,
        "max_dd_yuan": max_dd,
        "sharpe": sharpe,
        "n_yrs": n_yrs,
        "long_n": (df["direction"] == "LONG").sum(),
        "short_n": (df["direction"] == "SHORT").sum(),
        "long_pnl": df[df["direction"] == "LONG"]["pnl_yuan"].sum(),
        "short_pnl": df[df["direction"] == "SHORT"]["pnl_yuan"].sum(),
        "sl_n": (df["reason"] == "SL").sum(),
        "df": df,
        "daily_pnl": daily_pnl,
    }


def print_metrics(m):
    print(f"\n  ── {m['label']} ──")
    print(f"  Trades:        {m['n']}  (LONG {m['long_n']}, SHORT {m['short_n']})")
    print(f"  Retrains:      {m['n_retrains']}  Train IC: {m['train_ic_mean']:+.4f} ± {m['train_ic_std']:.4f}")
    print(f"  WR:            {m['wr']:.1f}%")
    print(f"  Avg / trade:   {m['avg_pct']:+.3f}%   worst {m['worst_pct']:+.2f}%   best {m['best_pct']:+.2f}%")
    print(f"  Total PnL:     {m['final_yuan']:+,.0f} 元 ({m['n_yrs']:.1f} 年, 1手)")
    print(f"  年化:          {m['ann_yuan']:+,.0f} 元/年")
    print(f"  Max DD:        {m['max_dd_yuan']:+,.0f} 元")
    print(f"  Sharpe:        {m['sharpe']:.2f}")
    if m["max_dd_yuan"] < 0:
        print(f"  Calmar:        {m['ann_yuan'] / abs(m['max_dd_yuan']):.2f}")
    print(f"  LONG PnL:      {m['long_pnl']:+,.0f}")
    print(f"  SHORT PnL:     {m['short_pnl']:+,.0f}")
    print(f"  SL hits:       {m['sl_n']} / {m['n']}  ({m['sl_n']/m['n']*100:.1f}%)")


def main():
    print("Loading data + computing features...")
    ctx = load_default_context(DB_PATH)
    px = ctx.px_df.copy()
    iv_dates = set(ctx.iv_df["trade_date"].tolist())
    px_e = px[px["trade_date"].isin(iv_dates)].reset_index(drop=True)
    dates = px_e["trade_date"].tolist()
    closes = px_e["close"].astype(float).values
    highs = px_e["high"].astype(float).values
    lows = px_e["low"].astype(float).values
    atr20 = compute_atr20(px_e)
    print(f"Range: {dates[0]} ~ {dates[-1]}  ({len(dates)} days)")
    print(f"Eval starts at idx {INITIAL_TRAIN_DAYS}: {dates[INITIAL_TRAIN_DAYS]}")

    print(f"\n{'═' * 75}")
    print(f" Walk-forward retrain every {RETRAIN_EVERY}d, hold {HOLD_DAYS}d")
    print(f"{'═' * 75}")

    print("\n[1/4] Baseline (no regime, no ATR-SL)...")
    m_base = run_walkforward(
        "Baseline (no enhancements)", build_default_pipeline,
        ctx, dates, closes, highs, lows, atr20, use_atr_sl=False)

    print("\n[2/4] A only (ATR×1.5 SL)...")
    m_a = run_walkforward(
        "A only (ATR×1.5 SL)", build_default_pipeline,
        ctx, dates, closes, highs, lows, atr20, use_atr_sl=True)

    print("\n[3/4] D only (regime feats)...")
    m_d = run_walkforward(
        "D only (regime feats)", build_pipeline_with_regime,
        ctx, dates, closes, highs, lows, atr20, use_atr_sl=False)

    print("\n[4/4] A+D (ATR + Regime)...")
    m_ad = run_walkforward(
        "A+D (ATR + Regime)", build_pipeline_with_regime,
        ctx, dates, closes, highs, lows, atr20, use_atr_sl=True)

    # ── 输出 ──
    print(f"\n{'═' * 90}")
    print(" Walk-forward 完整结果")
    print(f"{'═' * 90}")
    for m in [m_base, m_a, m_d, m_ad]:
        print_metrics(m)

    # ── 对比表 ──
    print(f"\n{'═' * 100}")
    print(" 对比 (walk-forward retrain every 20d)")
    print(f"{'═' * 100}")
    print(f"  {'变体':<28} {'N':>4} {'WR':>5} {'年化¥':>11} {'MaxDD¥':>11} "
          f"{'Calmar':>7} {'Sharpe':>7} {'LONG¥':>11} {'SHORT¥':>11}")
    print("  " + "-" * 100)
    for m in [m_base, m_a, m_d, m_ad]:
        if m["n"] == 0:
            continue
        cal = m["ann_yuan"] / abs(m["max_dd_yuan"]) if m["max_dd_yuan"] < 0 else 999
        print(f"  {m['label']:<28} {m['n']:>4} {m['wr']:>4.0f}% "
              f"{m['ann_yuan']:>+11,.0f} {m['max_dd_yuan']:>+11,.0f} "
              f"{cal:>6.2f} {m['sharpe']:>6.2f} "
              f"{m['long_pnl']:>+11,.0f} {m['short_pnl']:>+11,.0f}")

    # ── A+D 月度 PnL ──
    print(f"\n=== A+D 月度 PnL breakdown ===")
    df_ad = m_ad["df"]
    df_ad["month"] = pd.to_datetime(df_ad["exit_date"]).dt.to_period("M").astype(str)
    monthly = df_ad.groupby("month").agg(
        n=("pnl_yuan", "count"),
        wr=("net_ret", lambda x: (x > 0).mean() * 100),
        pnl=("pnl_yuan", "sum"),
    )
    print(f"  {'month':<10} {'N':>4} {'WR':>6} {'PnL¥':>12}")
    for mo, r in monthly.iterrows():
        print(f"  {mo:<10} {int(r['n']):>4} {r['wr']:>5.0f}% {r['pnl']:>+12,.0f}")

    # ── A+D 年度 PnL ──
    print(f"\n=== A+D 年度 PnL ===")
    df_ad["year"] = pd.to_datetime(df_ad["exit_date"]).dt.year
    yearly = df_ad.groupby("year").agg(
        n=("pnl_yuan", "count"),
        wr=("net_ret", lambda x: (x > 0).mean() * 100),
        pnl=("pnl_yuan", "sum"),
        long_pnl=("pnl_yuan", lambda x: x[df_ad.loc[x.index, "direction"] == "LONG"].sum()),
        short_pnl=("pnl_yuan", lambda x: x[df_ad.loc[x.index, "direction"] == "SHORT"].sum()),
    )
    print(f"  {'year':<6} {'N':>4} {'WR':>6} {'PnL¥':>12} {'LONG¥':>12} {'SHORT¥':>12}")
    for y, r in yearly.iterrows():
        print(f"  {int(y):<6} {int(r['n']):>4} {r['wr']:>5.0f}% "
              f"{r['pnl']:>+12,.0f} {r['long_pnl']:>+12,.0f} {r['short_pnl']:>+12,.0f}")

    # ── 与 strict-OOS A+D 对比 ──
    test_start_2025 = next((i for i, d in enumerate(dates) if d >= "20250101"), len(dates))
    daily_oos = m_ad["daily_pnl"][test_start_2025:]
    cum_oos = np.cumsum(daily_oos) * CONTRACT_MULT
    final_oos = cum_oos[-1]
    peak_oos = np.maximum.accumulate(cum_oos)
    dd_oos = (cum_oos - peak_oos).min()
    n_yrs_oos = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[test_start_2025])).days / 365.0
    print(f"\n=== A+D 在 2025-01+ test 段 (与 strict-OOS 同窗口对比) ===")
    print(f"  Walk-forward A+D:  +{final_oos:,.0f} 元  ({final_oos/n_yrs_oos:+,.0f} 年化)")
    print(f"                     MaxDD {dd_oos:+,.0f}")
    print(f"  Strict-OOS A+D:    +887,032 元  (+678,756 年化, MaxDD -788,692)  ← 之前结果")

    df_ad.to_csv("/tmp/daily_walkforward_AD_trades.csv", index=False)
    print(f"\n  Saved: /tmp/daily_walkforward_AD_trades.csv")


if __name__ == "__main__":
    main()
