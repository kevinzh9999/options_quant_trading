#!/usr/bin/env python3
"""Walk-forward A+D + hard regime gates: 测多个候选 gate 设计

候选 gate (only suppress SHORT in uptrend; LONG in downtrend):
  G0: 不加 gate (= A+D baseline, ann +918K, 2025 +238元)
  G1: SHORT 禁 if close/sma60 > 1.04
  G2: SHORT 禁 if close/sma200 > 1.10
  G3: SHORT 禁 if close/sma60 > 1.04 AND close/sma200 > 1.05  (双 horizon AND)
  G4: asymmetric thresh — SHORT 用 P10 阈值, LONG 用 P20
  G5: G3 + asymmetric (双重保险)
  G6: SHORT 禁 if close/sma60 > 1.03 OR close/sma200 > 1.08 (OR 版)

每个 gate 输出：
  - 总年化 / MaxDD / Sharpe / Calmar
  - 每年 PnL (LONG/SHORT 分解)
  - 与 G0 baseline 对比
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from typing import Optional, Callable

import numpy as np
import pandas as pd

from strategies.daily.factors import build_default_pipeline, load_default_context, DailyFactor
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
SHORT_TIGHT_PCT = 0.10  # asymmetric: SHORT 用 P10
ATR_K = 1.5
CONTRACT_MULT = 200
SLIPPAGE_PCT = 0.0008


def build_pipeline_with_regime():
    base = build_default_pipeline()
    base.factors.extend([CloseSma60Factor(), Slope60dFactor(), VolRegimeFactor()])
    base.feature_names = [f.name for f in base.factors]
    return base


# ── Regime feature precompute (independent of pipeline) ──

def precompute_regime(closes: np.ndarray) -> dict:
    """Precompute per-day regime indicators for fast gate-checking."""
    n = len(closes)
    sma60 = np.full(n, np.nan)
    sma200 = np.full(n, np.nan)
    for i in range(n):
        if i >= 60:
            sma60[i] = closes[i-60+1:i+1].mean()
        if i >= 200:
            sma200[i] = closes[i-200+1:i+1].mean()
    return {
        "close_sma60": closes / np.maximum(sma60, 1),
        "close_sma200": closes / np.maximum(sma200, 1),
    }


# ── Gate functions: (idx, direction, regime) → bool (True = blocked) ──

def gate_none(idx, direction, regime):
    return False

def gate_sma60(idx, direction, regime):
    if direction == "SHORT" and not np.isnan(regime["close_sma60"][idx]):
        return regime["close_sma60"][idx] > 1.04
    if direction == "LONG" and not np.isnan(regime["close_sma60"][idx]):
        return regime["close_sma60"][idx] < 0.96
    return False

def gate_sma200(idx, direction, regime):
    if direction == "SHORT" and not np.isnan(regime["close_sma200"][idx]):
        return regime["close_sma200"][idx] > 1.10
    if direction == "LONG" and not np.isnan(regime["close_sma200"][idx]):
        return regime["close_sma200"][idx] < 0.90
    return False

def gate_dual_and(idx, direction, regime):
    s60 = regime["close_sma60"][idx]
    s200 = regime["close_sma200"][idx]
    if np.isnan(s60) or np.isnan(s200):
        return False
    if direction == "SHORT":
        return s60 > 1.04 and s200 > 1.05
    if direction == "LONG":
        return s60 < 0.96 and s200 < 0.95
    return False


def gate_dual_and_short_only(idx, direction, regime):
    """G3 但只禁 SHORT，LONG 在下跌段不限制。"""
    if direction != "SHORT":
        return False
    s60 = regime["close_sma60"][idx]
    s200 = regime["close_sma200"][idx]
    if np.isnan(s60) or np.isnan(s200):
        return False
    return s60 > 1.04 and s200 > 1.05

def gate_dual_or(idx, direction, regime):
    s60 = regime["close_sma60"][idx]
    s200 = regime["close_sma200"][idx]
    if np.isnan(s60) and np.isnan(s200):
        return False
    if direction == "SHORT":
        return (not np.isnan(s60) and s60 > 1.03) or (not np.isnan(s200) and s200 > 1.08)
    if direction == "LONG":
        return (not np.isnan(s60) and s60 < 0.97) or (not np.isnan(s200) and s200 < 0.92)
    return False


def run_gate(label: str, gate_fn: Callable, asymmetric: bool,
              ctx, dates, closes, highs, lows, atr20, regime):
    n = len(dates)
    fwd = np.array([
        closes[i + HOLD_DAYS] / closes[i] - 1 if i + HOLD_DAYS < n else np.nan
        for i in range(n)
    ])
    pipe_proto = build_pipeline_with_regime()
    X_full = pipe_proto.features_matrix(dates, ctx)

    cached_pipe = None
    cached_thr = None
    cached_thr_short = None
    last_train = -RETRAIN_EVERY
    trades = []
    daily_pnl = np.zeros(n)
    blocked_count = 0

    for td_idx in range(INITIAL_TRAIN_DAYS, n - HOLD_DAYS):
        if td_idx - last_train >= RETRAIN_EVERY:
            train_end = td_idx - HOLD_DAYS
            X_tr = X_full[:train_end]
            y_tr = fwd[:train_end]
            valid = ~(np.isnan(X_tr).any(axis=1) | np.isnan(y_tr))
            if valid.sum() < INITIAL_TRAIN_DAYS:
                continue
            X_tr_v = X_tr[valid]
            y_tr_v = y_tr[valid]
            cached_pipe = build_pipeline_with_regime()
            cached_pipe.train(X_tr_v, y_tr_v)
            pred_in = cached_pipe.predict(X_tr_v)
            cached_thr = (
                float(np.quantile(pred_in, 1 - TOP_PCT)),
                float(np.quantile(pred_in, BOT_PCT)),
            )
            cached_thr_short = float(np.quantile(pred_in, SHORT_TIGHT_PCT))
            last_train = td_idx

        if cached_pipe is None:
            continue

        x = X_full[td_idx:td_idx+1]
        if np.isnan(x).any():
            continue
        pred = float(cached_pipe.predict(x)[0])
        top_thr, bot_thr = cached_thr
        short_thr = cached_thr_short if asymmetric else bot_thr

        if pred >= top_thr:
            direction = "LONG"
        elif pred <= short_thr:
            direction = "SHORT"
        else:
            continue

        # Apply gate
        if gate_fn(td_idx, direction, regime):
            blocked_count += 1
            continue

        sl = ATR_K * atr20[td_idx] if not np.isnan(atr20[td_idx]) else None
        exit_idx, gross_ret, reason = simulate_trade(
            direction, td_idx, closes, highs, lows,
            hold_days=HOLD_DAYS, sl_pct=sl,
        )
        net_ret = gross_ret - SLIPPAGE_PCT
        pnl_pts = net_ret * closes[td_idx]
        daily_pnl[exit_idx] += pnl_pts
        trades.append({
            "entry_date": dates[td_idx], "exit_date": dates[exit_idx],
            "direction": direction, "pred": pred,
            "gross_ret": gross_ret, "net_ret": net_ret,
            "pnl_pts": pnl_pts, "pnl_yuan": pnl_pts * CONTRACT_MULT,
            "reason": reason,
        })

    if not trades:
        return {"label": label, "n": 0, "blocked": blocked_count}

    df = pd.DataFrame(trades)
    df["year"] = pd.to_datetime(df["entry_date"]).dt.year
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

    yearly = df.groupby(["year", "direction"])["pnl_yuan"].sum().unstack(fill_value=0)

    return {
        "label": label,
        "n": len(df),
        "blocked": blocked_count,
        "wr": (df["net_ret"] > 0).mean() * 100,
        "final_yuan": final,
        "ann_yuan": final / max(n_yrs, 0.1),
        "max_dd_yuan": max_dd,
        "sharpe": sharpe,
        "calmar": final / max(n_yrs, 0.1) / abs(max_dd) if max_dd < 0 else 0,
        "long_n": (df["direction"] == "LONG").sum(),
        "short_n": (df["direction"] == "SHORT").sum(),
        "long_pnl": df[df["direction"] == "LONG"]["pnl_yuan"].sum(),
        "short_pnl": df[df["direction"] == "SHORT"]["pnl_yuan"].sum(),
        "yearly": yearly,
    }


def main():
    print("Loading data...")
    ctx = load_default_context(DB_PATH)
    px = ctx.px_df.copy()
    iv_dates = set(ctx.iv_df["trade_date"].tolist())
    px_e = px[px["trade_date"].isin(iv_dates)].reset_index(drop=True)
    dates = px_e["trade_date"].tolist()
    closes = px_e["close"].astype(float).values
    highs = px_e["high"].astype(float).values
    lows = px_e["low"].astype(float).values
    atr20 = compute_atr20(px_e)
    regime = precompute_regime(closes)
    print(f"Range: {dates[0]} ~ {dates[-1]}")

    variants = [
        ("G0: A+D baseline",            gate_none,                  False),
        ("G3: AND (sym both sides)",    gate_dual_and,              False),
        ("G3s: AND (SHORT-only)",       gate_dual_and_short_only,   False),
        ("G3s+asym",                    gate_dual_and_short_only,   True),
        ("G4: asym SHORT P10 only",     gate_none,                  True),
    ]

    results = []
    for label, gate_fn, asym in variants:
        print(f"\n[Running] {label} ...")
        m = run_gate(label, gate_fn, asym, ctx, dates, closes, highs, lows, atr20, regime)
        results.append(m)

    # ── 总表 ──
    print(f"\n{'═' * 110}")
    print(" Walk-forward gate sweep (2.9 yr, retrain 20d, A+D backbone)")
    print(f"{'═' * 110}")
    print(f"  {'Variant':<32} {'N':>4} {'BLK':>4} {'WR':>5} "
          f"{'年化¥':>10} {'MaxDD¥':>10} {'Calmar':>6} {'Sharpe':>6} "
          f"{'L/S':>9} {'LONG¥':>10} {'SHORT¥':>10}")
    print("  " + "-" * 108)
    for m in results:
        if m["n"] == 0:
            print(f"  {m['label']:<32} (no trades, blocked {m['blocked']})")
            continue
        print(f"  {m['label']:<32} {m['n']:>4} {m['blocked']:>4} {m['wr']:>4.0f}% "
              f"{m['ann_yuan']:>+10,.0f} {m['max_dd_yuan']:>+10,.0f} "
              f"{m['calmar']:>5.2f} {m['sharpe']:>5.2f} "
              f"{m['long_n']:>3}/{m['short_n']:<3} "
              f"{m['long_pnl']:>+10,.0f} {m['short_pnl']:>+10,.0f}")

    # ── 年度分解 ──
    print(f"\n{'═' * 110}")
    print(" 年度 PnL 分解（关键关注 2025）")
    print(f"{'═' * 110}")
    years = [2023, 2024, 2025, 2026]
    print(f"  {'Variant':<32}  ", end="")
    for y in years:
        print(f"{y}_LONG    {y}_SHORT   {y}_TOT  ", end=" ")
    print()
    for m in results:
        if m["n"] == 0:
            continue
        print(f"  {m['label']:<32}  ", end="")
        for y in years:
            l = m["yearly"].loc[y, "LONG"] if y in m["yearly"].index and "LONG" in m["yearly"].columns else 0
            s = m["yearly"].loc[y, "SHORT"] if y in m["yearly"].index and "SHORT" in m["yearly"].columns else 0
            print(f"{l:>+9,.0f} {s:>+9,.0f} {l+s:>+9,.0f}  ", end=" ")
        print()


if __name__ == "__main__":
    main()
