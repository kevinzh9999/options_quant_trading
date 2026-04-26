#!/usr/bin/env python3
"""Walk-forward A+D+G3s + LONG-side bull-regime enhancements

Backbone: A+D+G3s (ATR×1.5 SL + Regime feats + SHORT-only AND gate)
Then针对 LONG in bull-regime (close/sma60>1.04 AND close/sma200>1.05)：
  L0: 不优化 (=A+D+G3s baseline)
  L1: 牛市 LONG hold 5d → 10d
  L2: 牛市 LONG hold 5d → 15d
  L3: 牛市 LONG SL ATR×1.5 → ×3 (吃急跌)
  L4: 牛市 LONG SL ATR×1.5 → ×4 (再放宽)
  L5: 牛市 LONG hold 10d + SL ×3
  L6: 牛市 LONG hold 15d + SL ×4
  L7: 牛市 LONG hold 10d + SL ×3 + 已盈利则再放宽到 ×5

输出: 完整 walk-forward + 年度分解，重点看 2025 LONG。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from typing import Optional

import numpy as np
import pandas as pd

from strategies.daily.factors import build_default_pipeline, load_default_context
from scripts.daily_robust_methods_compare import (
    CloseSma60Factor, Slope60dFactor, VolRegimeFactor, compute_atr20,
)
from scripts.daily_walkforward_gates import precompute_regime, gate_dual_and_short_only

DB_PATH = "/Users/kevinzhao/Documents/options_quant_trading/data/storage/trading.db"
HOLD_DAYS_DEFAULT = 5
RETRAIN_EVERY = 20
INITIAL_TRAIN_DAYS = 200
TOP_PCT = 0.20
BOT_PCT = 0.20
ATR_K_DEFAULT = 1.5
CONTRACT_MULT = 200
SLIPPAGE_PCT = 0.0008


def build_pipeline_with_regime():
    base = build_default_pipeline()
    base.factors.extend([CloseSma60Factor(), Slope60dFactor(), VolRegimeFactor()])
    base.feature_names = [f.name for f in base.factors]
    return base


def simulate_trade_dynamic(direction: str, entry_idx: int,
                            closes, highs, lows,
                            hold_days: int, sl_pct: Optional[float],
                            wider_sl_after_profit: Optional[float] = None,
                            profit_trigger_pct: float = 0.01):
    """Walk forward day-by-day with optional dynamic SL widening when profit reached."""
    entry_close = closes[entry_idx]
    sign = 1 if direction == "LONG" else -1
    profit_widened = False
    cur_sl = sl_pct
    n = len(closes)
    for offset in range(1, hold_days + 1):
        idx = entry_idx + offset
        if idx >= n:
            return n - 1, sign * (closes[-1] / entry_close - 1), "TIME"

        cur_close = closes[idx]
        cur_ret = sign * (cur_close - entry_close) / entry_close
        # Widen SL once profit reached
        if (wider_sl_after_profit is not None and not profit_widened
                and cur_ret >= profit_trigger_pct):
            cur_sl = wider_sl_after_profit
            profit_widened = True

        # Check SL using bar's worst point (path-aware)
        if cur_sl is not None:
            if direction == "LONG":
                worst_ret = (lows[idx] - entry_close) / entry_close
                if worst_ret <= -cur_sl:
                    return idx, -cur_sl, "SL"
            else:
                worst_ret = (entry_close - highs[idx]) / entry_close
                if worst_ret <= -cur_sl:
                    return idx, -cur_sl, "SL"

    exit_idx = entry_idx + hold_days
    return exit_idx, sign * (closes[exit_idx] / entry_close - 1), "TIME"


def is_bull_regime(idx, regime):
    s60 = regime["close_sma60"][idx]
    s200 = regime["close_sma200"][idx]
    if np.isnan(s60) or np.isnan(s200):
        return False
    return s60 > 1.04 and s200 > 1.05


def run_variant(label: str, ctx, dates, closes, highs, lows, atr20, regime,
                 long_bull_hold: int = HOLD_DAYS_DEFAULT,
                 long_bull_sl_k: float = ATR_K_DEFAULT,
                 long_bull_sl_k_after_profit: Optional[float] = None,
                 ):
    n = len(dates)
    fwd = np.array([
        closes[i + HOLD_DAYS_DEFAULT] / closes[i] - 1 if i + HOLD_DAYS_DEFAULT < n else np.nan
        for i in range(n)
    ])
    pipe_proto = build_pipeline_with_regime()
    X_full = pipe_proto.features_matrix(dates, ctx)

    cached_pipe = None
    cached_thr = None
    last_train = -RETRAIN_EVERY
    trades = []
    daily_pnl = np.zeros(n)

    for td_idx in range(INITIAL_TRAIN_DAYS, n - HOLD_DAYS_DEFAULT):
        if td_idx - last_train >= RETRAIN_EVERY:
            train_end = td_idx - HOLD_DAYS_DEFAULT
            X_tr = X_full[:train_end]
            y_tr = fwd[:train_end]
            valid = ~(np.isnan(X_tr).any(axis=1) | np.isnan(y_tr))
            if valid.sum() < INITIAL_TRAIN_DAYS:
                continue
            cached_pipe = build_pipeline_with_regime()
            cached_pipe.train(X_tr[valid], y_tr[valid])
            pred_in = cached_pipe.predict(X_tr[valid])
            cached_thr = (
                float(np.quantile(pred_in, 1 - TOP_PCT)),
                float(np.quantile(pred_in, BOT_PCT)),
            )
            last_train = td_idx

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

        # G3s SHORT gate
        if gate_dual_and_short_only(td_idx, direction, regime):
            continue

        bull = is_bull_regime(td_idx, regime)

        # Per-trade params
        if direction == "LONG" and bull:
            hold = long_bull_hold
            atr_k = long_bull_sl_k
            wider_sl = (long_bull_sl_k_after_profit * atr20[td_idx]
                        if long_bull_sl_k_after_profit is not None and not np.isnan(atr20[td_idx])
                        else None)
        else:
            hold = HOLD_DAYS_DEFAULT
            atr_k = ATR_K_DEFAULT
            wider_sl = None

        sl = atr_k * atr20[td_idx] if not np.isnan(atr20[td_idx]) else None
        # Don't go beyond array bounds
        if td_idx + hold >= n:
            continue

        exit_idx, gross_ret, reason = simulate_trade_dynamic(
            direction, td_idx, closes, highs, lows,
            hold_days=hold, sl_pct=sl,
            wider_sl_after_profit=wider_sl,
        )
        net_ret = gross_ret - SLIPPAGE_PCT
        pnl_pts = net_ret * closes[td_idx]
        daily_pnl[exit_idx] += pnl_pts
        trades.append({
            "entry_date": dates[td_idx], "exit_date": dates[exit_idx],
            "direction": direction, "pred": pred,
            "gross_ret": gross_ret, "net_ret": net_ret,
            "pnl_yuan": pnl_pts * CONTRACT_MULT,
            "reason": reason, "hold": exit_idx - td_idx,
            "bull": bull,
        })

    if not trades:
        return {"label": label, "n": 0}

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
        "label": label, "n": len(df),
        "wr": (df["net_ret"] > 0).mean() * 100,
        "final_yuan": final,
        "ann_yuan": final / max(n_yrs, 0.1),
        "max_dd_yuan": max_dd,
        "sharpe": sharpe,
        "calmar": (final / max(n_yrs, 0.1)) / abs(max_dd) if max_dd < 0 else 0,
        "long_n": (df["direction"] == "LONG").sum(),
        "short_n": (df["direction"] == "SHORT").sum(),
        "long_pnl": df[df["direction"] == "LONG"]["pnl_yuan"].sum(),
        "short_pnl": df[df["direction"] == "SHORT"]["pnl_yuan"].sum(),
        "yearly": yearly,
        "df": df,
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
        # (label, hold, sl_k, sl_k_after_profit)
        ("L0: A+D+G3s baseline",                    5,  1.5, None),
        ("L1: bull LONG hold 10d",                  10, 1.5, None),
        ("L2: bull LONG hold 15d",                  15, 1.5, None),
        ("L3: bull LONG SL ATR×3.0",                5,  3.0, None),
        ("L4: bull LONG SL ATR×4.0",                5,  4.0, None),
        ("L5: bull LONG hold 10d + SL ×3",          10, 3.0, None),
        ("L6: bull LONG hold 15d + SL ×4",          15, 4.0, None),
        ("L7: bull LONG hold 10d + ×3 → ×5 if profit", 10, 3.0, 5.0),
        ("L8: bull LONG hold 10d + SL ×4",          10, 4.0, None),
    ]

    results = []
    for label, hold, k, k_prof in variants:
        print(f"\n[Running] {label} ...")
        m = run_variant(label, ctx, dates, closes, highs, lows, atr20, regime,
                         long_bull_hold=hold, long_bull_sl_k=k,
                         long_bull_sl_k_after_profit=k_prof)
        results.append(m)

    print(f"\n{'═' * 110}")
    print(" Walk-forward LONG-bull optim sweep")
    print(f"{'═' * 110}")
    print(f"  {'Variant':<42} {'N':>4} {'WR':>5} "
          f"{'年化¥':>10} {'MaxDD¥':>10} {'Calmar':>6} {'Sharpe':>6} "
          f"{'LONG¥':>11} {'SHORT¥':>10}")
    print("  " + "-" * 108)
    for m in results:
        if m["n"] == 0:
            continue
        print(f"  {m['label']:<42} {m['n']:>4} {m['wr']:>4.0f}% "
              f"{m['ann_yuan']:>+10,.0f} {m['max_dd_yuan']:>+10,.0f} "
              f"{m['calmar']:>5.2f} {m['sharpe']:>5.2f} "
              f"{m['long_pnl']:>+11,.0f} {m['short_pnl']:>+10,.0f}")

    print(f"\n{'═' * 100}")
    print(" 年度 LONG PnL 分解 (2025 = bull market 主战场)")
    print(f"{'═' * 100}")
    years = [2023, 2024, 2025, 2026]
    print(f"  {'Variant':<42} {'2023L':>9} {'2024L':>11} {'2025L':>11} {'2026L':>11} "
          f"{'2025总':>11} {'年化总':>11}")
    for m in results:
        if m["n"] == 0:
            continue
        longs_by_year = []
        for y in years:
            v = m["yearly"].loc[y, "LONG"] if y in m["yearly"].index and "LONG" in m["yearly"].columns else 0
            longs_by_year.append(v)
        # 2025 total
        l25 = m["yearly"].loc[2025, "LONG"] if 2025 in m["yearly"].index and "LONG" in m["yearly"].columns else 0
        s25 = m["yearly"].loc[2025, "SHORT"] if 2025 in m["yearly"].index and "SHORT" in m["yearly"].columns else 0
        print(f"  {m['label']:<42} {longs_by_year[0]:>+9,.0f} {longs_by_year[1]:>+11,.0f} "
              f"{longs_by_year[2]:>+11,.0f} {longs_by_year[3]:>+11,.0f} "
              f"{l25+s25:>+11,.0f} {m['ann_yuan']:>+11,.0f}")


if __name__ == "__main__":
    main()
