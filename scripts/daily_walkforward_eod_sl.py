#!/usr/bin/env python3
"""End-of-day close-based SL (避开盘中 wick)

不用 intraday low，只看 close。如果当日 close 跌破 SL 阈值，下个交易日 open 执行平仓。
针对 2026 Jan 黑天鹅期：盘中暴跌 + 隔日反弹的 wick whipsaw 模式。
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
from scripts.daily_walkforward_long_v2 import enh_two_branch_v2
from scripts.daily_walkforward_bear_mirror import bear_short_enh_loose

DB_PATH = "/Users/kevinzhao/Documents/options_quant_trading/data/storage/trading.db"
HOLD_DAYS_DEFAULT = 5
RETRAIN_EVERY = 20
INITIAL_TRAIN_DAYS = 200
TOP_PCT = 0.20
BOT_PCT = 0.20
ATR_K_DEFAULT = 1.5


def build_pipe():
    base = build_default_pipeline()
    base.factors.extend([CloseSma60Factor(), Slope60dFactor(), VolRegimeFactor()])
    base.feature_names = [f.name for f in base.factors]
    return base


def simulate_eod_sl(direction, entry_idx, closes, opens, highs, lows,
                     hold_days, sl_pct, eod_sl_pct: Optional[float] = None):
    """支持两种 SL:
       - sl_pct: intraday bar-low based (default ATR-based, 仍存在)
       - eod_sl_pct: close-based, 触发后次日 open 平仓
    """
    n = len(closes)
    entry_close = closes[entry_idx]
    sign = 1 if direction == "LONG" else -1
    eod_triggered = False

    for offset in range(1, hold_days + 1):
        idx = entry_idx + offset
        if idx >= n:
            return n - 1, sign * (closes[-1] / entry_close - 1), "TIME"

        # 如果上一日 EOD SL 触发，今天 open 平仓
        if eod_triggered:
            exit_open = opens[idx] if not np.isnan(opens[idx]) else closes[idx]
            return idx, sign * (exit_open - entry_close) / entry_close, "EOD_NEXT_OPEN"

        # Intraday ATR SL
        if sl_pct is not None:
            if direction == "LONG":
                worst_ret = (lows[idx] - entry_close) / entry_close
                if worst_ret <= -sl_pct:
                    return idx, -sl_pct, "SL"
            else:
                worst_ret = (entry_close - highs[idx]) / entry_close
                if worst_ret <= -sl_pct:
                    return idx, -sl_pct, "SL"

        # EOD close-based SL — flag for next-day open exit
        if eod_sl_pct is not None:
            if direction == "LONG":
                cur_close_ret = (closes[idx] - entry_close) / entry_close
            else:
                cur_close_ret = (entry_close - closes[idx]) / entry_close
            if cur_close_ret <= -eod_sl_pct:
                eod_triggered = True

    exit_idx = entry_idx + hold_days
    return exit_idx, sign * (closes[exit_idx] / entry_close - 1), "TIME"


def run_variant(label, ctx, dates, closes, opens, highs, lows, atr20, regime,
                 eod_sl_pct=None):
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
    reasons = {}

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

        hold = HOLD_DAYS_DEFAULT
        atr_k = ATR_K_DEFAULT
        is_enhanced = False
        if direction == "LONG":
            enh = enh_two_branch_v2(td_idx, regime)
            if enh is not None:
                hold, atr_k = enh
                is_enhanced = True
        else:
            enh = bear_short_enh_loose(td_idx, regime)
            if enh is not None:
                hold, atr_k = enh
                is_enhanced = True

        sl = atr_k * atr20[td_idx] if not np.isnan(atr20[td_idx]) else None
        if td_idx + hold >= n:
            continue
        eod_sl = eod_sl_pct if is_enhanced else None
        exit_idx, gross_ret, reason = simulate_eod_sl(
            direction, td_idx, closes, opens, highs, lows, hold, sl, eod_sl,
        )
        net_ret = gross_ret - 0.0008
        pnl_pts = net_ret * closes[td_idx]
        daily_pnl[exit_idx] += pnl_pts
        reasons[reason] = reasons.get(reason, 0) + 1
        trades.append({
            "entry_date": dates[td_idx], "year": int(dates[td_idx][:4]),
            "month": dates[td_idx][:6],
            "direction": direction, "net_ret": net_ret,
            "pnl_yuan": pnl_pts * 200, "reason": reason,
        })

    if not trades:
        return {"label": label, "n": 0}
    df = pd.DataFrame(trades)
    eval_start = INITIAL_TRAIN_DAYS
    cum = np.cumsum(daily_pnl[eval_start:]) * 200
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
        "max_dd_yuan": max_dd, "sharpe": sharpe,
        "calmar": (final / max(n_yrs, 0.1)) / abs(max_dd) if max_dd < 0 else 0,
        "long_pnl": df[df["direction"] == "LONG"]["pnl_yuan"].sum(),
        "short_pnl": df[df["direction"] == "SHORT"]["pnl_yuan"].sum(),
        "yearly": df.groupby("year")["pnl_yuan"].sum(),
        "monthly_2026": df[df["year"] == 2026].groupby("month")["pnl_yuan"].sum(),
        "reasons": reasons,
    }


def main():
    ctx = load_default_context(DB_PATH)
    px = ctx.px_df.copy()
    iv_dates = set(ctx.iv_df["trade_date"].tolist())
    px_e = px[px["trade_date"].isin(iv_dates)].reset_index(drop=True)
    dates = px_e["trade_date"].tolist()
    closes = px_e["close"].astype(float).values
    opens = px_e["open"].astype(float).values
    highs = px_e["high"].astype(float).values
    lows = px_e["low"].astype(float).values
    atr20 = compute_atr20(px_e)
    regime = precompute_regime(closes)

    variants = [
        ("E0: M7 baseline (no EOD SL)",         None),
        ("E1: EOD close ≤ -2% → next-day open",  0.02),
        ("E2: EOD close ≤ -3% → next-day open",  0.03),
        ("E3: EOD close ≤ -4% → next-day open",  0.04),
        ("E4: EOD close ≤ -2.5% → next-day open", 0.025),
        ("E5: EOD close ≤ -1.5% → next-day open", 0.015),
    ]

    results = []
    for label, eod in variants:
        print(f"\n[Running] {label}...")
        m = run_variant(label, ctx, dates, closes, opens, highs, lows, atr20, regime,
                         eod_sl_pct=eod)
        results.append(m)

    print(f"\n{'═' * 110}")
    print(" EOD-close SL sweep (next-day open exit)")
    print(f"{'═' * 110}")
    print(f"  {'Variant':<42} {'N':>4} {'WR':>5} "
          f"{'年化¥':>11} {'MaxDD¥':>11} {'Calmar':>7} {'Sharpe':>7} "
          f"{'LONG¥':>11} {'SHORT¥':>11}")
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
    print(" 2026 月度")
    print(f"{'═' * 100}")
    for m in results:
        if m["n"] == 0:
            continue
        print(f"  {m['label']:<42}  ", end="")
        for mo in ["202601", "202602", "202603"]:
            v = m["monthly_2026"].get(mo, 0)
            print(f"{mo[-2:]}: {v:>+9,.0f}  ", end="")
        print()

    print(f"\n{'═' * 100}")
    print(" Exit reasons")
    print(f"{'═' * 100}")
    for m in results:
        if m["n"] == 0:
            continue
        print(f"  {m['label']:<42} {m.get('reasons', {})}")


if __name__ == "__main__":
    main()
