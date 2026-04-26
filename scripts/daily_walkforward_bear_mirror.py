#!/usr/bin/env python3
"""测试熊市镜像：LONG block in bear + SHORT enhancement in bear

当前策略:
  - G3s: bull → 阻 SHORT
  - N5: bull → 增强 LONG (extended/dip 两支)
  - bear 侧无任何 regime-aware 处理

镜像测试 (M-variants):
  M0: F0 baseline (no bear mirror)
  M1: + bear LONG block (close/sma60<0.96 AND close/sma200<0.95)
  M2: + bear SHORT extended (close/sma60<0.96 AND close/sma200<0.95 → hold 10d ×4)
  M3: + bear SHORT rip (close/sma200<0.97 AND close/sma60>0.98 → hold 15d ×4)
  M4: full mirror = M1 + M2 + M3
  M5: tighter mirror (close/sma200<0.93 / 0.95 边界)
  M6: looser mirror (close/sma200<0.97 主导)

每个变体看 2023 改善 + 不伤其他年份。
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


# ── Bear gates ──
def bear_long_block_strict(idx, regime):
    """Mirror of G3s SHORT block — block LONG when in strict bear."""
    s60 = regime["close_sma60"][idx]
    s200 = regime["close_sma200"][idx]
    if np.isnan(s60) or np.isnan(s200):
        return False
    return s60 < 0.96 and s200 < 0.95


def bear_long_block_loose(idx, regime):
    """Looser bear block — close/sma200 alone."""
    s200 = regime["close_sma200"][idx]
    if np.isnan(s200):
        return False
    return s200 < 0.95


def bear_short_enh_strict(idx, regime):
    """Mirror of N5 — bear SHORT enhancement.
    Returns (hold, atr_k) or None.
    Branch A (extended bear): close/sma60<0.96 AND close/sma200<0.95 → 10d ×4
    Branch B (rip-bear): close/sma200<0.97 AND close/sma60>0.98 → 15d ×4
    """
    s60 = regime["close_sma60"][idx]
    s200 = regime["close_sma200"][idx]
    if np.isnan(s60) or np.isnan(s200):
        return None
    if s60 < 0.96 and s200 < 0.95:
        return (10, 4.0)
    if s200 < 0.97 and s60 > 0.98:
        return (15, 4.0)
    return None


def bear_short_enh_loose(idx, regime):
    """Looser bear SHORT enhancement (less strict on close/sma200)."""
    s60 = regime["close_sma60"][idx]
    s200 = regime["close_sma200"][idx]
    if np.isnan(s60) or np.isnan(s200):
        return None
    if s60 < 0.97 and s200 < 0.97:
        return (10, 4.0)
    if s200 < 0.99 and s60 > 0.98:
        return (15, 4.0)
    return None


def run_variant(label: str, ctx, dates, closes, highs, lows, atr20, regime,
                 use_bear_long_block=False,
                 long_block_fn=bear_long_block_strict,
                 use_bear_short_enh=False,
                 short_enh_fn=bear_short_enh_strict):
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
    n_long_blocked = 0
    n_short_enh = 0

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

        # G3s SHORT gate (bull regime block)
        if gate_dual_and_short_only(td_idx, direction, regime):
            continue

        # bear LONG block (mirror of G3s)
        if direction == "LONG" and use_bear_long_block:
            if long_block_fn(td_idx, regime):
                n_long_blocked += 1
                continue

        # Determine hold + sl_k
        hold = HOLD_DAYS_DEFAULT
        atr_k = ATR_K_DEFAULT
        is_enhanced = False
        if direction == "LONG":
            enh = enh_two_branch_v2(td_idx, regime)
            if enh is not None:
                hold, atr_k = enh
                is_enhanced = True
        elif direction == "SHORT" and use_bear_short_enh:
            enh = short_enh_fn(td_idx, regime)
            if enh is not None:
                hold, atr_k = enh
                is_enhanced = True
                n_short_enh += 1

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
            "direction": direction, "pred": pred,
            "net_ret": net_ret, "pnl_yuan": pnl_pts * CONTRACT_MULT,
            "reason": reason, "hold_used": hold, "enhanced": is_enhanced,
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
    yearly = df.groupby(["year", "direction"])["pnl_yuan"].sum().unstack(fill_value=0)

    return {
        "label": label, "n": len(df),
        "n_long_blocked": n_long_blocked,
        "n_short_enh": n_short_enh,
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

    variants = [
        # (label, use_long_block, long_block_fn, use_short_enh, short_enh_fn)
        ("M0: F0 baseline (no bear mirror)",          False, bear_long_block_strict, False, bear_short_enh_strict),
        ("M1: +bear LONG block (strict)",             True,  bear_long_block_strict, False, bear_short_enh_strict),
        ("M2: +bear SHORT enh strict (no LONG block)", False, bear_long_block_strict, True,  bear_short_enh_strict),
        ("M3: M1+M2 full mirror strict",              True,  bear_long_block_strict, True,  bear_short_enh_strict),
        ("M4: bear LONG block loose (sma200<0.95)",   True,  bear_long_block_loose,  False, bear_short_enh_strict),
        ("M5: M4 + bear SHORT enh strict",            True,  bear_long_block_loose,  True,  bear_short_enh_strict),
        ("M6: M1 + bear SHORT enh loose",             True,  bear_long_block_strict, True,  bear_short_enh_loose),
        ("M7: bear SHORT enh loose only",             False, bear_long_block_strict, True,  bear_short_enh_loose),
    ]

    results = []
    for label, ulb, lb_fn, use_se, se_fn in variants:
        print(f"\n[Running] {label}...")
        m = run_variant(label, ctx, dates, closes, highs, lows, atr20, regime,
                         use_bear_long_block=ulb, long_block_fn=lb_fn,
                         use_bear_short_enh=use_se, short_enh_fn=se_fn)
        results.append(m)

    print(f"\n{'═' * 110}")
    print(" Bear-mirror sweep")
    print(f"{'═' * 110}")
    print(f"  {'Variant':<46} {'N':>4} {'Lblk':>5} {'Senh':>5} {'WR':>5} "
          f"{'年化¥':>11} {'MaxDD¥':>11} {'Calmar':>7} {'Sharpe':>7} "
          f"{'LONG¥':>11} {'SHORT¥':>11}")
    print("  " + "-" * 108)
    for m in results:
        if m["n"] == 0:
            continue
        print(f"  {m['label']:<46} {m['n']:>4} {m['n_long_blocked']:>5} "
              f"{m['n_short_enh']:>5} {m['wr']:>4.0f}% "
              f"{m['ann_yuan']:>+11,.0f} {m['max_dd_yuan']:>+11,.0f} "
              f"{m['calmar']:>6.2f} {m['sharpe']:>6.2f} "
              f"{m['long_pnl']:>+11,.0f} {m['short_pnl']:>+11,.0f}")

    print(f"\n{'═' * 100}")
    print(" 年度 PnL 分解")
    print(f"{'═' * 100}")
    print(f"  {'Variant':<46} {'2023':>9} {'2024':>11} {'2025':>11} {'2026':>11}")
    for m in results:
        if m["n"] == 0:
            continue
        years = [2023, 2024, 2025, 2026]
        totals = []
        for y in years:
            tot = 0
            if y in m["yearly"].index:
                tot = m["yearly"].loc[y].sum()
            totals.append(tot)
        print(f"  {m['label']:<46} {totals[0]:>+9,.0f} {totals[1]:>+11,.0f} "
              f"{totals[2]:>+11,.0f} {totals[3]:>+11,.0f}")

    # 各年 SHORT 分解 (看镜像是否生效)
    print(f"\n{'═' * 100}")
    print(" 各年 SHORT PnL (验证 bear SHORT enh 在 2023/2024 H1 是否有效)")
    print(f"{'═' * 100}")
    print(f"  {'Variant':<46} {'2023S':>9} {'2024S':>11} {'2025S':>11} {'2026S':>11}")
    for m in results:
        if m["n"] == 0:
            continue
        years = [2023, 2024, 2025, 2026]
        shorts = []
        for y in years:
            v = 0
            if y in m["yearly"].index and "SHORT" in m["yearly"].columns:
                v = m["yearly"].loc[y, "SHORT"]
            shorts.append(v)
        print(f"  {m['label']:<46} {shorts[0]:>+9,.0f} {shorts[1]:>+11,.0f} "
              f"{shorts[2]:>+11,.0f} {shorts[3]:>+11,.0f}")


if __name__ == "__main__":
    main()
