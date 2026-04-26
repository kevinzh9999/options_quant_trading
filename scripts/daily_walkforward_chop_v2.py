#!/usr/bin/env python3
"""Chop fix v2 — 用 trade-level 反馈 + post-rally detection 替代静态 chop indicator

D variants:
  D0: M7 baseline
  D1: enhanced trade feedback — 最近 5 笔 enhanced LONG WR<40% → 暂停 enhancement 10 天
  D2: dip-bull 只在 close/sma200 < 1.10 时增强 (避免高位 dip-buy)
  D3: enhancement 只在 slope_60d > 0 时启用 (require uptrend confirmation)
  D4: D2 + D3 combo
  D5: D1 + D2 combo
  D6: ATR-expansion 过滤 (recent ATR jumped > 30% → no enhancement)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from typing import Optional, Callable
from collections import deque

import numpy as np
import pandas as pd

from strategies.daily.factors import build_default_pipeline, load_default_context
from scripts.daily_robust_methods_compare import (
    CloseSma60Factor, Slope60dFactor, VolRegimeFactor, compute_atr20,
)
from scripts.daily_walkforward_gates import precompute_regime, gate_dual_and_short_only
from scripts.daily_walkforward_long_v2 import enh_two_branch_v2
from scripts.daily_walkforward_bear_mirror import bear_short_enh_loose
from scripts.daily_walkforward_long_optim import simulate_trade_dynamic
from scripts.daily_walkforward_2023_2026_fix import precompute_slope60

DB_PATH = "/Users/kevinzhao/Documents/options_quant_trading/data/storage/trading.db"
HOLD_DAYS_DEFAULT = 5
RETRAIN_EVERY = 20
INITIAL_TRAIN_DAYS = 200
TOP_PCT = 0.20
BOT_PCT = 0.20
ATR_K_DEFAULT = 1.5
CONTRACT_MULT = 200
SLIPPAGE_PCT = 0.0008


def build_pipe():
    base = build_default_pipeline()
    base.factors.extend([CloseSma60Factor(), Slope60dFactor(), VolRegimeFactor()])
    base.feature_names = [f.name for f in base.factors]
    return base


def precompute_atr_ratio(closes, highs, lows):
    """ATR(5) / ATR(20) — recent vs medium-term volatility."""
    atr20 = compute_atr20(pd.DataFrame({"close": closes, "high": highs, "low": lows}))
    n = len(closes)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
    atr5 = np.full(n, np.nan)
    for i in range(5, n):
        atr5[i] = tr[i-4:i+1].mean() / max(closes[i], 1)
    return atr5 / np.maximum(atr20, 1e-6)


def run_variant(label: str, ctx, dates, closes, highs, lows, atr20, regime, slope60,
                 atr_ratio,
                 use_trade_feedback=False, fb_lookback=5, fb_wr_thr=0.40, fb_cooldown=10,
                 dip_max_sma200=None,           # if not None, dip-bull only when sma200 < this
                 enh_require_slope_pos=False,   # require slope_60d > 0 for enhancement
                 atr_expand_thr=None,           # if not None, no enh when atr5/atr20 > this
                 ):
    n = len(dates)
    fwd = np.array([
        closes[i + HOLD_DAYS_DEFAULT] / closes[i] - 1 if i + HOLD_DAYS_DEFAULT < n else np.nan
        for i in range(n)
    ])
    pipe_proto = build_pipe()
    X_full = pipe_proto.features_matrix(dates, ctx)

    cached_pipe = None
    cached_thr = None
    last_train = -RETRAIN_EVERY
    trades = []
    daily_pnl = np.zeros(n)
    feedback_q = deque(maxlen=fb_lookback)  # 1 = winner, 0 = loser
    feedback_pause_until = -1

    for td_idx in range(INITIAL_TRAIN_DAYS, n - 16):
        if td_idx - last_train >= RETRAIN_EVERY:
            train_end = td_idx - HOLD_DAYS_DEFAULT
            X_tr = X_full[:train_end]
            y_tr = fwd[:train_end]
            valid = ~(np.isnan(X_tr).any(axis=1) | np.isnan(y_tr))
            if valid.sum() < INITIAL_TRAIN_DAYS:
                continue
            cached_pipe = build_pipe()
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

        if gate_dual_and_short_only(td_idx, direction, regime):
            continue

        # Feedback pause check
        feedback_paused = (use_trade_feedback and td_idx < feedback_pause_until)

        # ATR expansion filter
        atr_filtered = (atr_expand_thr is not None
                         and not np.isnan(atr_ratio[td_idx])
                         and atr_ratio[td_idx] > atr_expand_thr)

        # Slope filter
        slope_filtered = (enh_require_slope_pos
                           and (np.isnan(slope60[td_idx]) or slope60[td_idx] <= 0))

        # Determine enhancement
        hold = HOLD_DAYS_DEFAULT
        atr_k = ATR_K_DEFAULT
        is_enhanced = False

        if not feedback_paused and not atr_filtered:
            if direction == "LONG":
                enh = enh_two_branch_v2(td_idx, regime)
                if enh is not None:
                    # Check dip_max_sma200
                    if dip_max_sma200 is not None:
                        s200 = regime["close_sma200"][td_idx]
                        s60 = regime["close_sma60"][td_idx]
                        is_dip = s200 > 1.03 and s60 < 1.02
                        if is_dip and not np.isnan(s200) and s200 > dip_max_sma200:
                            enh = None
                    if enh is not None and not slope_filtered:
                        hold, atr_k = enh
                        is_enhanced = True
            elif direction == "SHORT":
                enh = bear_short_enh_loose(td_idx, regime)
                if enh is not None and not slope_filtered:
                    hold, atr_k = enh
                    is_enhanced = True

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

        # Update feedback queue (only enhanced trades count)
        if is_enhanced and use_trade_feedback:
            feedback_q.append(1 if net_ret > 0 else 0)
            if len(feedback_q) == fb_lookback:
                wr = sum(feedback_q) / len(feedback_q)
                if wr < fb_wr_thr:
                    feedback_pause_until = td_idx + fb_cooldown

        trades.append({
            "entry_date": dates[td_idx],
            "year": int(dates[td_idx][:4]),
            "month": dates[td_idx][:6],
            "direction": direction, "pred": pred,
            "net_ret": net_ret, "pnl_yuan": pnl_pts * CONTRACT_MULT,
            "hold_used": hold, "enhanced": is_enhanced,
            "feedback_paused": feedback_paused,
        })

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

    return {
        "label": label, "n": len(df),
        "wr": (df["net_ret"] > 0).mean() * 100,
        "ann_yuan": final / max(n_yrs, 0.1),
        "max_dd_yuan": max_dd,
        "sharpe": sharpe,
        "calmar": (final / max(n_yrs, 0.1)) / abs(max_dd) if max_dd < 0 else 0,
        "long_pnl": df[df["direction"] == "LONG"]["pnl_yuan"].sum(),
        "short_pnl": df[df["direction"] == "SHORT"]["pnl_yuan"].sum(),
        "n_paused": df["feedback_paused"].sum() if "feedback_paused" in df else 0,
        "yearly": df.groupby("year")["pnl_yuan"].sum(),
        "monthly_2026": df[df["year"] == 2026].groupby("month")["pnl_yuan"].sum(),
        "df": df,
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
    slope60 = precompute_slope60(closes)
    atr_ratio = precompute_atr_ratio(closes, highs, lows)

    variants = [
        # (label, fb, dip_max_sma200, slope_pos, atr_thr)
        ("D0: M7 baseline",                            False, None, False, None),
        ("D1: trade feedback (5笔WR<40% pause10d)",   True,  None, False, None),
        ("D2: dip-bull only when sma200<1.10",         False, 1.10, False, None),
        ("D3: enh require slope_60d>0",                False, None, True,  None),
        ("D4: D2 + D3",                                False, 1.10, True,  None),
        ("D5: D1 + D2",                                True,  1.10, False, None),
        ("D6: ATR_5/20 > 1.30 → no enh",               False, None, False, 1.30),
        ("D7: D2 + ATR filter",                        False, 1.10, False, 1.30),
        ("D8: dip-bull only sma200<1.08",              False, 1.08, False, None),
        ("D9: D1 + D2 + D6 (full)",                    True,  1.10, False, 1.30),
    ]

    results = []
    for label, fb, dms, sp, ar in variants:
        print(f"\n[Running] {label}...")
        m = run_variant(label, ctx, dates, closes, highs, lows, atr20, regime, slope60,
                         atr_ratio,
                         use_trade_feedback=fb, dip_max_sma200=dms,
                         enh_require_slope_pos=sp, atr_expand_thr=ar)
        results.append(m)

    print(f"\n{'═' * 110}")
    print(" 2026 Q1 fix sweep")
    print(f"{'═' * 110}")
    print(f"  {'Variant':<42} {'N':>4} {'WR':>5} "
          f"{'年化¥':>11} {'MaxDD¥':>11} {'Calmar':>7} {'Sharpe':>7} "
          f"{'LONG¥':>11} {'SHORT¥':>11}")
    print("  " + "-" * 108)
    for m in results:
        if m["n"] == 0:
            continue
        print(f"  {m['label']:<42} {m['n']:>4} {m['wr']:>4.0f}% "
              f"{m['ann_yuan']:>+11,.0f} {m['max_dd_yuan']:>+11,.0f} "
              f"{m['calmar']:>6.2f} {m['sharpe']:>6.2f} "
              f"{m['long_pnl']:>+11,.0f} {m['short_pnl']:>+11,.0f}")

    print(f"\n{'═' * 100}")
    print(" 年度 PnL")
    print(f"{'═' * 100}")
    print(f"  {'Variant':<42} {'2023':>9} {'2024':>11} {'2025':>11} {'2026':>11}")
    for m in results:
        if m["n"] == 0:
            continue
        ys = [m["yearly"].get(y, 0) for y in [2023, 2024, 2025, 2026]]
        print(f"  {m['label']:<42} {ys[0]:>+9,.0f} {ys[1]:>+11,.0f} "
              f"{ys[2]:>+11,.0f} {ys[3]:>+11,.0f}")

    print(f"\n{'═' * 100}")
    print(" 2026 月度 PnL")
    print(f"{'═' * 100}")
    for m in results:
        if m["n"] == 0:
            continue
        print(f"  {m['label']:<42}  ", end="")
        for mo in ["202601", "202602", "202603"]:
            v = m["monthly_2026"].get(mo, 0)
            print(f"{mo[-2:]}: {v:>+9,.0f}  ", end="")
        print()


if __name__ == "__main__":
    main()
