#!/usr/bin/env python3
"""测 2023 + 2026 修复方案，确保不伤 2024/2025

修复 1 (LONG entry filter, 解 2023 震荡市接飞刀问题):
  Require for any LONG entry: close/sma60 > 0.95 AND slope_60d > -0.0005
  (避开"加速下跌"的逆势抄底)

修复 2 (regime-break early exit, 解 2026 Jan 转换期 enhanced LONG 锁仓亏损):
  对 N5-enhanced LONG: if 持仓 day≥3 且 cumret < -0.02 → 立即退出
  (不等 ATR×4 SL 慢慢用完)

变体:
  F0: N5 baseline (当前最佳)
  F1: + LONG entry filter only
  F2: + regime-break exit only (loss_thr=2%, min_day=3)
  F3: + both
  F4: F1 + 更松 exit (loss_thr=1.5%, min_day=2)
  F5: F1 + 更紧 entry (close/sma60>0.97)
  F6: F1 + 更紧 exit (loss_thr=1.5%) + 更紧 entry
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


def precompute_slope60(closes: np.ndarray) -> np.ndarray:
    """Per-day slope_60d normalized."""
    n = len(closes)
    out = np.full(n, np.nan)
    for i in range(60, n):
        c = closes[i-60+1:i+1].astype(float)
        slope = float(np.polyfit(range(60), c, 1)[0])
        out[i] = slope / max(c.mean(), 1)
    return out


def long_entry_filter(idx, regime, slope60,
                       sma200_thr: float = 0.97,
                       slope_thr: float = -0.001) -> bool:
    """Return True = block LONG entry.

    Block only when in genuine bear regime:
      - close/sma200 < 0.97 (below 200-day MA = bear)
      - OR slope_60d very negative (in active downtrend)
    Note: 2025 dip-buy has close/sma200 > 1.03 so NOT blocked here.
    """
    s200 = regime["close_sma200"][idx]
    sl = slope60[idx]
    if np.isnan(s200) or np.isnan(sl):
        return False
    if s200 < sma200_thr and sl < slope_thr:
        return True
    return False


def simulate_with_break_exit(direction: str, entry_idx: int, closes, highs, lows,
                              hold_days: int, sl_pct: Optional[float],
                              break_loss_pct: Optional[float] = None,
                              break_min_day: int = 3):
    """Walk forward with optional regime-break early exit (only for LONG)."""
    entry_close = closes[entry_idx]
    sign = 1 if direction == "LONG" else -1
    n = len(closes)
    for offset in range(1, hold_days + 1):
        idx = entry_idx + offset
        if idx >= n:
            return n - 1, sign * (closes[-1] / entry_close - 1), "TIME"

        # ATR SL via worst path
        if sl_pct is not None:
            if direction == "LONG":
                worst_ret = (lows[idx] - entry_close) / entry_close
                if worst_ret <= -sl_pct:
                    return idx, -sl_pct, "SL"
            else:
                worst_ret = (entry_close - highs[idx]) / entry_close
                if worst_ret <= -sl_pct:
                    return idx, -sl_pct, "SL"

        # Regime-break early exit (only LONG, only after min_day)
        if (direction == "LONG" and break_loss_pct is not None
                and offset >= break_min_day):
            cur_ret = (closes[idx] - entry_close) / entry_close
            if cur_ret <= -break_loss_pct:
                return idx, cur_ret, "BREAK"

    exit_idx = entry_idx + hold_days
    return exit_idx, sign * (closes[exit_idx] / entry_close - 1), "TIME"


def run_variant(label: str, ctx, dates, closes, highs, lows, atr20, regime, slope60,
                 use_long_filter: bool = False,
                 long_filter_sma200_thr: float = 0.97,
                 long_filter_slope_thr: float = -0.001,
                 break_loss_pct: Optional[float] = None,
                 break_min_day: int = 3,
                 enh_require_slope_pos: bool = False):
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
    n_filtered = 0

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

        # G3s SHORT gate
        if gate_dual_and_short_only(td_idx, direction, regime):
            continue

        # 2023 fix: LONG entry filter (bear-only)
        if direction == "LONG" and use_long_filter:
            if long_entry_filter(td_idx, regime, slope60,
                                  long_filter_sma200_thr, long_filter_slope_thr):
                n_filtered += 1
                continue

        # Enhancement?
        hold = HOLD_DAYS_DEFAULT
        atr_k = ATR_K_DEFAULT
        is_enhanced = False
        if direction == "LONG":
            enh = enh_two_branch_v2(td_idx, regime)
            if enh is not None:
                # 2026 fix: require slope_60d > 0 to enable enhancement
                if enh_require_slope_pos and (np.isnan(slope60[td_idx])
                                                or slope60[td_idx] <= 0):
                    pass  # don't enhance
                else:
                    hold, atr_k = enh
                    is_enhanced = True

        sl = atr_k * atr20[td_idx] if not np.isnan(atr20[td_idx]) else None
        if td_idx + hold >= n:
            continue

        # 2026 fix: regime-break early exit only for enhanced LONG
        break_loss = break_loss_pct if (is_enhanced and direction == "LONG") else None

        exit_idx, gross_ret, reason = simulate_with_break_exit(
            direction, td_idx, closes, highs, lows,
            hold_days=hold, sl_pct=sl,
            break_loss_pct=break_loss, break_min_day=break_min_day,
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
        "label": label, "n": len(df), "n_filtered": n_filtered,
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
    slope60 = precompute_slope60(closes)

    variants = [
        # (label, use_filter, sma200_thr, slope_thr, break_loss, break_min_day, enh_require_slope_pos)
        ("F0: N5 baseline",                                       False, 0.97,  -0.001, None,  3, False),
        ("F8: bear filter (sma200<0.97 AND slope<-0.001)",       True,  0.97,  -0.001, None,  3, False),
        ("F9: F8 + enh require slope>0",                         True,  0.97,  -0.001, None,  3, True),
        ("F10: F8 + break exit 2.5%/d5",                         True,  0.97,  -0.001, 0.025, 5, False),
        ("F11: F9 + break exit 2.5%/d5",                         True,  0.97,  -0.001, 0.025, 5, True),
        ("F12: tighter bear filter (sma200<0.99 AND slope<0)",   True,  0.99,  0.000,  None,  3, False),
        ("F13: bear filter no slope (sma200<0.95 only)",         True,  0.95,  9.999,  None,  3, False),
        ("F14: enh require slope>0 only (no bear filter)",       False, 0.97,  -0.001, None,  3, True),
    ]

    results = []
    for label, uf, st, sl_thr, bl, bd, esp in variants:
        print(f"\n[Running] {label}...")
        m = run_variant(label, ctx, dates, closes, highs, lows, atr20, regime, slope60,
                         use_long_filter=uf, long_filter_sma200_thr=st,
                         long_filter_slope_thr=sl_thr,
                         break_loss_pct=bl, break_min_day=bd,
                         enh_require_slope_pos=esp)
        results.append(m)

    print(f"\n{'═' * 110}")
    print(" 2023+2026 Fix sweep")
    print(f"{'═' * 110}")
    print(f"  {'Variant':<48} {'N':>4} {'flt':>4} {'WR':>5} "
          f"{'年化¥':>11} {'MaxDD¥':>11} {'Calmar':>7} {'Sharpe':>7} "
          f"{'LONG¥':>11} {'SHORT¥':>10}")
    print("  " + "-" * 108)
    for m in results:
        if m["n"] == 0:
            continue
        print(f"  {m['label']:<48} {m['n']:>4} {m.get('n_filtered',0):>4} {m['wr']:>4.0f}% "
              f"{m['ann_yuan']:>+11,.0f} {m['max_dd_yuan']:>+11,.0f} "
              f"{m['calmar']:>6.2f} {m['sharpe']:>6.2f} "
              f"{m['long_pnl']:>+11,.0f} {m['short_pnl']:>+10,.0f}")

    print(f"\n{'═' * 100}")
    print(" 年度 PnL 分解 (LONG + SHORT)")
    print(f"{'═' * 100}")
    print(f"  {'Variant':<48} {'2023':>9} {'2024':>11} {'2025':>11} {'2026':>11}")
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
        print(f"  {m['label']:<48} {totals[0]:>+9,.0f} {totals[1]:>+11,.0f} "
              f"{totals[2]:>+11,.0f} {totals[3]:>+11,.0f}")

    # 最佳 variant 的 2026 月度 + 2023 月度 详细
    print(f"\n{'═' * 100}")
    print(" 最佳变体 的 2023 + 2026 月度详细 (按 ann_yuan 排第一)")
    print(f"{'═' * 100}")
    valid = [m for m in results if m["n"] > 0]
    best = max(valid, key=lambda m: m["ann_yuan"])
    print(f"  Best: {best['label']}\n")
    df3 = best["df"]
    df3["month"] = df3["entry_date"].astype(str).str[:6]
    for y in [2023, 2026]:
        print(f"\n  ── {y} 月度 ──")
        print(f"  {'month':<8} {'N':>3} {'WR':>5} {'PnL¥':>10} {'avg%':>7}")
        for m_str, sub in df3[df3["year"] == y].groupby("month"):
            wr = (sub["net_ret"] > 0).mean() * 100
            print(f"  {m_str:<8} {len(sub):>3} {wr:>4.0f}% "
                  f"{sub['pnl_yuan'].sum():>+10,.0f} {sub['net_ret'].mean()*100:>+6.3f}%")


if __name__ == "__main__":
    main()
