#!/usr/bin/env python3
"""测 chop-regime detector — 识别"涨跌涨跌"震荡期，避免 enhancement 被抽

当前 M7 strategy 在 2026 Q1 weak (年度仅 +219K，主要 Jan -152K)。
2026 Jan 是典型 choppy: dip-bull enhancement 锁仓 15d 反复被吃。

候选 chop 检测方法:
  efficiency_ratio (ER) = |close[t] - close[t-N]| / Σ|daily_returns|
    - ER 接近 1 = 单边趋势，ER 接近 0 = 来回震荡
    - chop if ER < 0.3
  vol_regime = rv5 / rv60
    - chop if vol_regime > 1.3 (vol expanding sharply)
  reversal_count = 最近 N 日 daily return 符号反转次数
    - chop if reversal_count > 5 in last 10 days
  range_no_direction = (high_N - low_N)/close > 0.05 AND |close - close_N_ago| < 0.02
    - chop if range big but net change small

变体策略:
  C0: M7 baseline (no chop filter)
  C1: ER<0.3 → 降级到 default (no enhancement)
  C2: vol_regime>1.3 → 降级
  C3: reversal_count>5 → 降级
  C4: range_no_direction → 降级
  C5: ANY of above → 降级
  C6: ER<0.3 → 完全跳过 trade (skip entirely)
  C7: ER<0.3 → 降级 + tighter SL (ATR×1.0)
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
from scripts.daily_walkforward_long_v2 import enh_two_branch_v2
from scripts.daily_walkforward_bear_mirror import bear_short_enh_loose
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


def build_pipe():
    base = build_default_pipeline()
    base.factors.extend([CloseSma60Factor(), Slope60dFactor(), VolRegimeFactor()])
    base.feature_names = [f.name for f in base.factors]
    return base


# ── Chop detectors (per-day arrays) ──

def precompute_chop_metrics(closes: np.ndarray) -> dict:
    n = len(closes)
    er10 = np.full(n, np.nan)        # efficiency ratio over last 10 days
    er20 = np.full(n, np.nan)        # over last 20 days
    rv5 = np.full(n, np.nan)
    rv60 = np.full(n, np.nan)
    reversals10 = np.full(n, np.nan)  # # of sign changes in last 10 daily returns
    range_pct_10 = np.full(n, np.nan) # (max-min)/close last 10
    net_pct_10 = np.full(n, np.nan)   # (close - close[-10])/close

    rets = np.zeros(n)
    for i in range(1, n):
        if closes[i-1] > 0:
            rets[i] = np.log(closes[i] / closes[i-1])

    for i in range(n):
        if i >= 10:
            window = closes[i-10:i+1]
            net = abs(window[-1] - window[0])
            traveled = np.sum(np.abs(np.diff(window)))
            if traveled > 0:
                er10[i] = net / traveled
            # reversals
            recent_rets = rets[i-9:i+1]
            signs = np.sign(recent_rets)
            signs = signs[signs != 0]
            if len(signs) > 1:
                reversals10[i] = (signs[1:] != signs[:-1]).sum()
            # range / net
            range_pct_10[i] = (window.max() - window.min()) / max(closes[i], 1)
            net_pct_10[i] = (window[-1] - window[0]) / max(window[0], 1)
        if i >= 20:
            window = closes[i-20:i+1]
            net = abs(window[-1] - window[0])
            traveled = np.sum(np.abs(np.diff(window)))
            if traveled > 0:
                er20[i] = net / traveled
        if i >= 60:
            rv5[i] = np.std(rets[i-5:i]) * np.sqrt(252)
            rv60[i] = np.std(rets[i-60:i]) * np.sqrt(252)

    return {
        "er10": er10, "er20": er20,
        "vol_regime": rv5 / np.maximum(rv60, 1e-6),
        "reversals10": reversals10,
        "range10": range_pct_10,
        "net10": net_pct_10,
    }


def is_chop(idx, chop, mode: str = "er") -> bool:
    """Return True if idx is in choppy regime (per detection mode)."""
    if mode == "off":
        return False
    if mode == "er":
        v = chop["er10"][idx]
        return not np.isnan(v) and v < 0.30
    if mode == "vol":
        v = chop["vol_regime"][idx]
        return not np.isnan(v) and v > 1.30
    if mode == "rev":
        v = chop["reversals10"][idx]
        return not np.isnan(v) and v > 5
    if mode == "rng":
        rng = chop["range10"][idx]
        net = chop["net10"][idx]
        return (not np.isnan(rng) and not np.isnan(net)
                and rng > 0.05 and abs(net) < 0.02)
    if mode == "any":
        return any(is_chop(idx, chop, m) for m in ["er", "vol", "rev", "rng"])
    if mode == "er_strict":
        v = chop["er10"][idx]
        return not np.isnan(v) and v < 0.25
    return False


def run_variant(label: str, ctx, dates, closes, highs, lows, atr20, regime, chop,
                 chop_mode: str = "off",
                 chop_action: str = "downgrade"):
    """
    chop_mode: detection rule (off/er/vol/rev/rng/any/er_strict)
    chop_action: 'downgrade' = revert to default 5d ×1.5, 'skip' = no trade,
                 'tight_sl' = downgrade + ATR×1.0
    """
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
    n_chop_hits = 0

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

        # Detect chop
        chop_hit = is_chop(td_idx, chop, chop_mode)
        if chop_hit:
            n_chop_hits += 1

        # Skip entirely if action='skip' and chop
        if chop_hit and chop_action == "skip":
            continue

        # Determine hold + sl_k
        hold = HOLD_DAYS_DEFAULT
        atr_k = ATR_K_DEFAULT
        is_enhanced = False

        if not chop_hit:
            if direction == "LONG":
                enh = enh_two_branch_v2(td_idx, regime)
                if enh is not None:
                    hold, atr_k = enh
                    is_enhanced = True
            elif direction == "SHORT":
                enh = bear_short_enh_loose(td_idx, regime)
                if enh is not None:
                    hold, atr_k = enh
                    is_enhanced = True
        else:
            # Chop downgrade: stay at default, optionally tight_sl
            if chop_action == "tight_sl":
                atr_k = 1.0

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
            "entry_date": dates[td_idx],
            "year": int(dates[td_idx][:4]),
            "month": dates[td_idx][:6],
            "direction": direction, "pred": pred,
            "net_ret": net_ret, "pnl_yuan": pnl_pts * CONTRACT_MULT,
            "hold_used": hold, "enhanced": is_enhanced,
            "chop_hit": chop_hit,
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
    yearly = df.groupby("year")["pnl_yuan"].sum()

    return {
        "label": label, "n": len(df),
        "n_chop_hits": n_chop_hits,
        "wr": (df["net_ret"] > 0).mean() * 100,
        "ann_yuan": final / max(n_yrs, 0.1),
        "max_dd_yuan": max_dd,
        "sharpe": sharpe,
        "calmar": (final / max(n_yrs, 0.1)) / abs(max_dd) if max_dd < 0 else 0,
        "long_pnl": df[df["direction"] == "LONG"]["pnl_yuan"].sum(),
        "short_pnl": df[df["direction"] == "SHORT"]["pnl_yuan"].sum(),
        "yearly": yearly,
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
    chop = precompute_chop_metrics(closes)

    # Show chop metric distribution per year (sanity)
    print("\n=== Chop metrics distribution by year ===")
    df_chop = pd.DataFrame({
        "date": dates,
        "year": [int(d[:4]) for d in dates],
        "er10": chop["er10"],
        "vol_regime": chop["vol_regime"],
        "reversals10": chop["reversals10"],
    })
    for y in [2023, 2024, 2025, 2026]:
        sub = df_chop[df_chop["year"] == y].dropna()
        if sub.empty:
            continue
        print(f"  {y}: er10 mean {sub['er10'].mean():.3f} | "
              f"vol_regime mean {sub['vol_regime'].mean():.3f} | "
              f"rev mean {sub['reversals10'].mean():.2f} | "
              f"er10<0.3 days {(sub['er10']<0.3).sum()} | "
              f"vol_regime>1.3 days {(sub['vol_regime']>1.3).sum()}")

    variants = [
        ("C0: M7 baseline (no chop)",          "off",        "downgrade"),
        ("C1: ER<0.30 → downgrade",            "er",         "downgrade"),
        ("C2: vol_regime>1.30 → downgrade",    "vol",        "downgrade"),
        ("C3: reversals>5 → downgrade",        "rev",        "downgrade"),
        ("C4: range>5% no-dir → downgrade",    "rng",        "downgrade"),
        ("C5: ANY chop → downgrade",           "any",        "downgrade"),
        ("C6: ER<0.30 → SKIP entirely",        "er",         "skip"),
        ("C7: ER<0.30 → downgrade + ATR×1.0",  "er",         "tight_sl"),
        ("C8: ER<0.25 → downgrade",            "er_strict",  "downgrade"),
        ("C9: ER<0.30 → SKIP + tight check",   "er",         "skip"),
    ]

    results = []
    for label, mode, action in variants:
        print(f"\n[Running] {label}...")
        m = run_variant(label, ctx, dates, closes, highs, lows, atr20, regime, chop,
                         chop_mode=mode, chop_action=action)
        results.append(m)

    print(f"\n{'═' * 110}")
    print(" Chop-filter sweep")
    print(f"{'═' * 110}")
    print(f"  {'Variant':<42} {'N':>4} {'chop':>5} {'WR':>5} "
          f"{'年化¥':>11} {'MaxDD¥':>11} {'Calmar':>7} {'Sharpe':>7} "
          f"{'LONG¥':>11} {'SHORT¥':>11}")
    print("  " + "-" * 108)
    for m in results:
        if m["n"] == 0:
            continue
        print(f"  {m['label']:<42} {m['n']:>4} {m.get('n_chop_hits',0):>5} "
              f"{m['wr']:>4.0f}% "
              f"{m['ann_yuan']:>+11,.0f} {m['max_dd_yuan']:>+11,.0f} "
              f"{m['calmar']:>6.2f} {m['sharpe']:>6.2f} "
              f"{m['long_pnl']:>+11,.0f} {m['short_pnl']:>+11,.0f}")

    print(f"\n{'═' * 100}")
    print(" 年度 PnL 分解")
    print(f"{'═' * 100}")
    print(f"  {'Variant':<42} {'2023':>9} {'2024':>11} {'2025':>11} {'2026':>11}")
    for m in results:
        if m["n"] == 0:
            continue
        years = [2023, 2024, 2025, 2026]
        ys = [m["yearly"].get(y, 0) for y in years]
        print(f"  {m['label']:<42} {ys[0]:>+9,.0f} {ys[1]:>+11,.0f} "
              f"{ys[2]:>+11,.0f} {ys[3]:>+11,.0f}")

    # 最佳 variant 看 2026 月度
    print(f"\n{'═' * 100}")
    print(" 各 variant 2026 月度 PnL")
    print(f"{'═' * 100}")
    for m in results:
        if m["n"] == 0:
            continue
        df = m["df"]
        d26 = df[df["year"] == 2026]
        if d26.empty:
            continue
        monthly = d26.groupby("month")["pnl_yuan"].sum()
        print(f"  {m['label']:<42}  ", end="")
        for mo in ["202601", "202602", "202603"]:
            v = monthly.get(mo, 0)
            print(f"{mo[-2:]}: {v:>+9,.0f}  ", end="")
        print()


if __name__ == "__main__":
    main()
