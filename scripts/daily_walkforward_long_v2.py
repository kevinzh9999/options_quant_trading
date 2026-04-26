#!/usr/bin/env python3
"""Walk-forward A+D+G3s + 重新设计的 trend-LONG enhancement gate

诊断发现:
  - 2024 bull-LONG: 主要是"已在涨" (close/sma60 > 1.04 typical)
  - 2025 LONG: 主要是"牛市中买回调" (close/sma200 > 1.03 but close/sma60 oscillates)
  - 旧 L8 gate (close/sma60 > 1.04 AND close/sma200 > 1.05) 漏掉 86% 2025 LONG

新候选 gate:
  N0: L8 baseline (旧 strict gate)
  N1: gate = close/sma200 > 1.03   (放宽，覆盖 dip-buy)
  N2: gate = close/sma200 > 1.00   (更宽松)
  N3: 2-branch: strict bull (旧 gate) → hold 10d×4; dip-bull (200>1.05 AND 60<1.02) → hold 15d×4
  N4: gate = close/sma200 > 1.03，hold=15d (因为 2025 ret15 比 ret10 更高)
  N5: gate = close/sma200 > 1.03，hold=10d，SL ATR×4

每个变体里同时保持: A (default ATR×1.5 SL) + D (regime feats) + G3s SHORT gate.
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

from strategies.daily.factors import build_default_pipeline, load_default_context
from scripts.daily_robust_methods_compare import (
    CloseSma60Factor, Slope60dFactor, VolRegimeFactor, compute_atr20,
)
from scripts.daily_walkforward_gates import precompute_regime, gate_dual_and_short_only
from scripts.daily_walkforward_long_optim import simulate_trade_dynamic

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


# ── trend-LONG enhancement region functions ──
# Returns (hold, sl_k) or None if not in enhancement region.

def enh_strict(idx, regime):
    """L8 strict: close/sma60>1.04 AND close/sma200>1.05 → hold 10d, SL×4"""
    s60 = regime["close_sma60"][idx]
    s200 = regime["close_sma200"][idx]
    if np.isnan(s60) or np.isnan(s200):
        return None
    if s60 > 1.04 and s200 > 1.05:
        return (10, 4.0)
    return None


def enh_sma200_only(idx, regime):
    """N1/N5: close/sma200 > 1.03 → hold 10d, SL×4"""
    s200 = regime["close_sma200"][idx]
    if np.isnan(s200):
        return None
    if s200 > 1.03:
        return (10, 4.0)
    return None


def enh_sma200_only_15d(idx, regime):
    """N4: close/sma200 > 1.03 → hold 15d, SL×4"""
    s200 = regime["close_sma200"][idx]
    if np.isnan(s200):
        return None
    if s200 > 1.03:
        return (15, 4.0)
    return None


def enh_sma200_loose(idx, regime):
    """N2: close/sma200 > 1.00 → hold 10d, SL×4"""
    s200 = regime["close_sma200"][idx]
    if np.isnan(s200):
        return None
    if s200 > 1.00:
        return (10, 4.0)
    return None


def enh_two_branch(idx, regime):
    """N3: strict bull (旧) → 10d×4; dip-bull → 15d×4"""
    s60 = regime["close_sma60"][idx]
    s200 = regime["close_sma200"][idx]
    if np.isnan(s60) or np.isnan(s200):
        return None
    # strict bull
    if s60 > 1.04 and s200 > 1.05:
        return (10, 4.0)
    # dip-bull: above sma200 substantially but below sma60
    if s200 > 1.05 and s60 < 1.02:
        return (15, 4.0)
    return None


def enh_two_branch_v2(idx, regime):
    """N6: tighter dip-bull (close/sma200 > 1.03 AND close/sma60 < 1.02 → 15d×4),
    or strict (旧 gate → 10d×4)"""
    s60 = regime["close_sma60"][idx]
    s200 = regime["close_sma200"][idx]
    if np.isnan(s60) or np.isnan(s200):
        return None
    if s60 > 1.04 and s200 > 1.05:
        return (10, 4.0)
    if s200 > 1.03 and s60 < 1.02:
        return (15, 4.0)
    return None


def enh_sma200_only_10d_3x(idx, regime):
    """N7: close/sma200 > 1.03 → hold 10d, SL×3 (less aggressive SL)"""
    s200 = regime["close_sma200"][idx]
    if np.isnan(s200):
        return None
    if s200 > 1.03:
        return (10, 3.0)
    return None


def run_variant(label: str, enh_fn: Callable, ctx, dates, closes, highs, lows, atr20, regime):
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
    n_enh = 0

    for td_idx in range(INITIAL_TRAIN_DAYS, n - 16):
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

        # LONG enhancement?
        hold = HOLD_DAYS_DEFAULT
        atr_k = ATR_K_DEFAULT
        if direction == "LONG":
            enh = enh_fn(td_idx, regime)
            if enh is not None:
                hold, atr_k = enh
                n_enh += 1

        sl = atr_k * atr20[td_idx] if not np.isnan(atr20[td_idx]) else None
        if td_idx + hold >= n:
            continue

        exit_idx, gross_ret, reason = simulate_trade_dynamic(
            direction, td_idx, closes, highs, lows,
            hold_days=hold, sl_pct=sl,
        )
        net_ret = gross_ret - SLIPPAGE_PCT
        pnl_pts = net_ret * closes[td_idx]
        daily_pnl[exit_idx] += pnl_pts
        trades.append({
            "entry_date": dates[td_idx], "exit_date": dates[exit_idx],
            "direction": direction, "pnl_yuan": pnl_pts * CONTRACT_MULT,
            "net_ret": net_ret, "hold": exit_idx - td_idx, "reason": reason,
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
        "label": label, "n": len(df), "n_enh": n_enh,
        "wr": (df["net_ret"] > 0).mean() * 100,
        "ann_yuan": final / max(n_yrs, 0.1),
        "max_dd_yuan": max_dd,
        "sharpe": sharpe,
        "calmar": (final / max(n_yrs, 0.1)) / abs(max_dd) if max_dd < 0 else 0,
        "long_pnl": df[df["direction"] == "LONG"]["pnl_yuan"].sum(),
        "short_pnl": df[df["direction"] == "SHORT"]["pnl_yuan"].sum(),
        "yearly": yearly,
    }


def main():
    print("Loading...")
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

    variants = [
        ("N0: L8 strict (旧)",                    enh_strict),
        ("N1: sma200>1.03, hold10d ×4",          enh_sma200_only),
        ("N2: sma200>1.00, hold10d ×4",          enh_sma200_loose),
        ("N3: 2-branch (strict + dip>1.05)",      enh_two_branch),
        ("N4: sma200>1.03, hold15d ×4",          enh_sma200_only_15d),
        ("N5: 2-branch v2 (strict + dip>1.03)",   enh_two_branch_v2),
        ("N6: sma200>1.03, hold10d ×3",          enh_sma200_only_10d_3x),
    ]

    results = []
    for label, fn in variants:
        print(f"\n[Running] {label}...")
        m = run_variant(label, fn, ctx, dates, closes, highs, lows, atr20, regime)
        results.append(m)

    print(f"\n{'═' * 110}")
    print(" Walk-forward LONG-enhancement gate redesign")
    print(f"{'═' * 110}")
    print(f"  {'Variant':<40} {'N':>4} {'enh':>4} {'WR':>5} "
          f"{'年化¥':>10} {'MaxDD¥':>10} {'Calmar':>7} {'Sharpe':>7} "
          f"{'LONG¥':>11} {'SHORT¥':>10}")
    print("  " + "-" * 108)
    for m in results:
        if m["n"] == 0:
            continue
        print(f"  {m['label']:<40} {m['n']:>4} {m['n_enh']:>4} {m['wr']:>4.0f}% "
              f"{m['ann_yuan']:>+10,.0f} {m['max_dd_yuan']:>+10,.0f} "
              f"{m['calmar']:>6.2f} {m['sharpe']:>6.2f} "
              f"{m['long_pnl']:>+11,.0f} {m['short_pnl']:>+10,.0f}")

    print(f"\n{'═' * 100}")
    print(" 年度 LONG PnL 分解")
    print(f"{'═' * 100}")
    print(f"  {'Variant':<40} {'2023L':>9} {'2024L':>11} {'2025L':>11} {'2026L':>11} {'2025总':>11}")
    for m in results:
        if m["n"] == 0:
            continue
        years = [2023, 2024, 2025, 2026]
        longs = []
        for y in years:
            v = m["yearly"].loc[y, "LONG"] if y in m["yearly"].index and "LONG" in m["yearly"].columns else 0
            longs.append(v)
        l25 = m["yearly"].loc[2025, "LONG"] if 2025 in m["yearly"].index and "LONG" in m["yearly"].columns else 0
        s25 = m["yearly"].loc[2025, "SHORT"] if 2025 in m["yearly"].index and "SHORT" in m["yearly"].columns else 0
        print(f"  {m['label']:<40} {longs[0]:>+9,.0f} {longs[1]:>+11,.0f} "
              f"{longs[2]:>+11,.0f} {longs[3]:>+11,.0f} {l25+s25:>+11,.0f}")


if __name__ == "__main__":
    main()
