#!/usr/bin/env python3
"""SL multiplier sweep on enhanced trades — 看更紧的 SL 能否救 2026 Jan

目前 enhancement 用 ATR×4 (~6% SL) 在 2026 Jan 12 笔全没触发，hold 10d 吃满下跌。
试更紧的 SL: ×3, ×2.5, ×2, ×1.75
对照 2024 牛市 — 紧 SL 是否过早把 winners 抽掉。
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
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
SLIPPAGE_PCT = 0.0008


def build_pipe():
    base = build_default_pipeline()
    base.factors.extend([CloseSma60Factor(), Slope60dFactor(), VolRegimeFactor()])
    base.feature_names = [f.name for f in base.factors]
    return base


def make_enh_strict(strict_atr_k, dip_atr_k):
    def fn(idx, regime):
        s60 = regime["close_sma60"][idx]
        s200 = regime["close_sma200"][idx]
        if np.isnan(s60) or np.isnan(s200):
            return None
        if s60 > 1.04 and s200 > 1.05:
            return (10, strict_atr_k)
        if s200 > 1.03 and s60 < 1.02:
            return (15, dip_atr_k)
        return None
    return fn


def make_short_strict(strict_atr_k, rip_atr_k):
    def fn(idx, regime):
        s60 = regime["close_sma60"][idx]
        s200 = regime["close_sma200"][idx]
        if np.isnan(s60) or np.isnan(s200):
            return None
        if s60 < 0.97 and s200 < 0.97:
            return (10, strict_atr_k)
        if s200 < 0.99 and s60 > 0.98:
            return (15, rip_atr_k)
        return None
    return fn


def run_variant(label, ctx, dates, closes, highs, lows, atr20, regime, lf, sf):
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
    n_sl_fired = 0

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
        if direction == "LONG":
            enh = lf(td_idx, regime)
            if enh is not None:
                hold, atr_k = enh
        else:
            enh = sf(td_idx, regime)
            if enh is not None:
                hold, atr_k = enh

        sl = atr_k * atr20[td_idx] if not np.isnan(atr20[td_idx]) else None
        if td_idx + hold >= n:
            continue
        exit_idx, gross_ret, reason = simulate_trade_dynamic(
            direction, td_idx, closes, highs, lows,
            hold_days=hold, sl_pct=sl,
        )
        if reason == "SL":
            n_sl_fired += 1
        net_ret = gross_ret - SLIPPAGE_PCT
        pnl_pts = net_ret * closes[td_idx]
        daily_pnl[exit_idx] += pnl_pts
        trades.append({
            "entry_date": dates[td_idx], "year": int(dates[td_idx][:4]),
            "month": dates[td_idx][:6],
            "direction": direction, "net_ret": net_ret,
            "pnl_yuan": pnl_pts * 200, "hold_used": hold, "reason": reason,
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
        "label": label, "n": len(df), "n_sl": n_sl_fired,
        "wr": (df["net_ret"] > 0).mean() * 100,
        "ann_yuan": final / max(n_yrs, 0.1),
        "max_dd_yuan": max_dd, "sharpe": sharpe,
        "calmar": (final / max(n_yrs, 0.1)) / abs(max_dd) if max_dd < 0 else 0,
        "long_pnl": df[df["direction"] == "LONG"]["pnl_yuan"].sum(),
        "short_pnl": df[df["direction"] == "SHORT"]["pnl_yuan"].sum(),
        "yearly": df.groupby("year")["pnl_yuan"].sum(),
        "monthly_2026": df[df["year"] == 2026].groupby("month")["pnl_yuan"].sum(),
    }


def main():
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
        ("W0: M7 baseline (strict×4 / dip×4)",  make_enh_strict(4.0, 4.0), make_short_strict(4.0, 4.0)),
        ("W1: strict×3 / dip×4",                 make_enh_strict(3.0, 4.0), make_short_strict(3.0, 4.0)),
        ("W2: strict×2.5 / dip×4",              make_enh_strict(2.5, 4.0), make_short_strict(2.5, 4.0)),
        ("W3: strict×2 / dip×4",                 make_enh_strict(2.0, 4.0), make_short_strict(2.0, 4.0)),
        ("W4: strict×1.5 (default) / dip×4",     make_enh_strict(1.5, 4.0), make_short_strict(1.5, 4.0)),
        ("W5: strict×3 / dip×3",                 make_enh_strict(3.0, 3.0), make_short_strict(3.0, 3.0)),
        ("W6: strict×2.5 / dip×3",              make_enh_strict(2.5, 3.0), make_short_strict(2.5, 3.0)),
        ("W7: strict×2 / dip×3",                 make_enh_strict(2.0, 3.0), make_short_strict(2.0, 3.0)),
    ]

    results = []
    for label, lf, sf in variants:
        print(f"\n[Running] {label}...")
        m = run_variant(label, ctx, dates, closes, highs, lows, atr20, regime, lf, sf)
        results.append(m)

    print(f"\n{'═' * 110}")
    print(" SL multiplier sweep")
    print(f"{'═' * 110}")
    print(f"  {'Variant':<42} {'N':>4} {'SL':>4} {'WR':>5} "
          f"{'年化¥':>11} {'MaxDD¥':>11} {'Calmar':>7} {'Sharpe':>7} "
          f"{'LONG¥':>11} {'SHORT¥':>11}")
    for m in results:
        if m["n"] == 0:
            continue
        print(f"  {m['label']:<42} {m['n']:>4} {m.get('n_sl',0):>4} "
              f"{m['wr']:>4.0f}% "
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
    print(" 2026 月度 (核心)")
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
