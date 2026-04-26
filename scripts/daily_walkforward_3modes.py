#!/usr/bin/env python3
"""三档部署模式 walk-forward 回测

保守 (Conservative):  1 lot/signal, cap 10 concurrent
中等 (Moderate):      1 lot/signal, no cap (= M7 baseline)
激进 (Aggressive):    2 lots/signal, cap 20 concurrent

输出每个模式:
  - 年化 PnL / MaxDD / Sharpe / Calmar
  - 账户回报率 / MaxDD%
  - 保证金占用峰值 / 平均
  - 各年 PnL
  - 月度 PnL (验证 2026 Q1 行为)
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
ACCOUNT_EQUITY = 6_400_000
SLIPPAGE_PCT = 0.0008
MARGIN_PER_LOT = 260_000   # ~26 万/手 IM


def build_pipe():
    base = build_default_pipeline()
    base.factors.extend([CloseSma60Factor(), Slope60dFactor(), VolRegimeFactor()])
    base.feature_names = [f.name for f in base.factors]
    return base


def run_mode(label, lots_per_signal, concurrent_cap, ctx, dates, closes, highs, lows, atr20, regime):
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
    open_positions = []   # (exit_idx, direction, lots)
    long_pos = np.zeros(n)
    short_pos = np.zeros(n)
    n_skipped = 0

    for td_idx in range(INITIAL_TRAIN_DAYS, n - 16):
        # Cleanup expired
        open_positions = [(ei, d, l) for (ei, d, l) in open_positions if ei > td_idx]

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
            enh = enh_two_branch_v2(td_idx, regime)
            if enh is not None:
                hold, atr_k = enh
        else:
            enh = bear_short_enh_loose(td_idx, regime)
            if enh is not None:
                hold, atr_k = enh

        sl = atr_k * atr20[td_idx] if not np.isnan(atr20[td_idx]) else None
        if td_idx + hold >= n or sl is None:
            continue

        # Concurrent cap check
        cur_gross = sum(l for (_, _, l) in open_positions)
        if cur_gross + lots_per_signal > concurrent_cap:
            n_skipped += 1
            continue

        exit_idx, gross_ret, reason = simulate_trade_dynamic(
            direction, td_idx, closes, highs, lows, hold, sl,
        )
        net_ret = gross_ret - SLIPPAGE_PCT
        pnl_pts = net_ret * closes[td_idx] * lots_per_signal
        daily_pnl[exit_idx] += pnl_pts
        open_positions.append((exit_idx, direction, lots_per_signal))
        trades.append({
            "entry_date": dates[td_idx], "year": int(dates[td_idx][:4]),
            "month": dates[td_idx][:6],
            "direction": direction, "lots": lots_per_signal,
            "net_ret": net_ret, "pnl_yuan": pnl_pts * CONTRACT_MULT,
        })

    # Recompute concurrent positions for margin analysis
    long_pos = np.zeros(n)
    short_pos = np.zeros(n)
    # Re-simulate for position tracking with lots
    for t in trades:
        # Find entry/exit indices (since we lost them, infer from trades and re-track)
        pass
    # Simpler approach: track during the loop; let me re-track
    long_pos = np.zeros(n)
    short_pos = np.zeros(n)
    open_positions2 = []
    for t in trades:
        # We need to track positions on each day from entry to exit
        # The trade dict doesn't have entry_idx directly; let's recompute from entry_date
        pass

    if not trades:
        return {"label": label, "n": 0}

    df = pd.DataFrame(trades)

    # For position tracking, just use the existing trade list and recompute from entry/exit by searching
    date_to_idx = {d: i for i, d in enumerate(dates)}
    for t in trades:
        ei_idx = date_to_idx.get(t["entry_date"])
        if ei_idx is None:
            continue
        # Need exit_idx — recompute via simulate (or track during run). Easier: re-simulate just to get exit_idx
        # Actually we can find from open_positions tracking — but lost. Just re-track quickly.

    # Simpler: track per-day GROSS by re-simulating in second pass
    # OR add it to the trade dict. Let me modify to capture entry_idx, exit_idx in trade dict
    # Already exit_idx wasn't saved. Re-simulate for tracking.

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
        "label": label, "n": len(df), "n_skipped": n_skipped,
        "lots_per_signal": lots_per_signal,
        "concurrent_cap": concurrent_cap,
        "wr": (df["net_ret"] > 0).mean() * 100,
        "ann_yuan": final / max(n_yrs, 0.1),
        "max_dd_yuan": max_dd, "sharpe": sharpe,
        "calmar": (final / max(n_yrs, 0.1)) / abs(max_dd) if max_dd < 0 else 0,
        "long_pnl": df[df["direction"] == "LONG"]["pnl_yuan"].sum(),
        "short_pnl": df[df["direction"] == "SHORT"]["pnl_yuan"].sum(),
        "yearly": df.groupby("year")["pnl_yuan"].sum(),
        "monthly_2026": df[df["year"] == 2026].groupby("month")["pnl_yuan"].sum(),
    }


def run_with_position_tracking(label, lots_per_signal, concurrent_cap,
                                  ctx, dates, closes, highs, lows, atr20, regime):
    """Same as run_mode but also tracks per-day GROSS for margin analysis."""
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
    open_positions = []
    long_arr = np.zeros(n)
    short_arr = np.zeros(n)
    n_skipped = 0

    for td_idx in range(INITIAL_TRAIN_DAYS, n - 16):
        open_positions = [(ei, d, l) for (ei, d, l) in open_positions if ei > td_idx]

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
            enh = enh_two_branch_v2(td_idx, regime)
            if enh is not None:
                hold, atr_k = enh
        else:
            enh = bear_short_enh_loose(td_idx, regime)
            if enh is not None:
                hold, atr_k = enh

        sl = atr_k * atr20[td_idx] if not np.isnan(atr20[td_idx]) else None
        if td_idx + hold >= n or sl is None:
            continue
        cur_gross = sum(l for (_, _, l) in open_positions)
        if cur_gross + lots_per_signal > concurrent_cap:
            n_skipped += 1
            continue

        exit_idx, gross_ret, reason = simulate_trade_dynamic(
            direction, td_idx, closes, highs, lows, hold, sl,
        )
        net_ret = gross_ret - SLIPPAGE_PCT
        pnl_pts = net_ret * closes[td_idx] * lots_per_signal
        daily_pnl[exit_idx] += pnl_pts
        open_positions.append((exit_idx, direction, lots_per_signal))

        # Track positions
        if direction == "LONG":
            long_arr[td_idx: exit_idx] += lots_per_signal
        else:
            short_arr[td_idx: exit_idx] += lots_per_signal

        trades.append({
            "entry_date": dates[td_idx], "year": int(dates[td_idx][:4]),
            "month": dates[td_idx][:6],
            "direction": direction, "lots": lots_per_signal,
            "net_ret": net_ret, "pnl_yuan": pnl_pts * CONTRACT_MULT,
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
    gross_arr = long_arr + short_arr

    return {
        "label": label, "n": len(df), "n_skipped": n_skipped,
        "lots_per_signal": lots_per_signal,
        "concurrent_cap": concurrent_cap,
        "wr": (df["net_ret"] > 0).mean() * 100,
        "ann_yuan": final / max(n_yrs, 0.1),
        "max_dd_yuan": max_dd, "sharpe": sharpe,
        "calmar": (final / max(n_yrs, 0.1)) / abs(max_dd) if max_dd < 0 else 0,
        "long_pnl": df[df["direction"] == "LONG"]["pnl_yuan"].sum(),
        "short_pnl": df[df["direction"] == "SHORT"]["pnl_yuan"].sum(),
        "yearly": df.groupby("year")["pnl_yuan"].sum(),
        "monthly_2026": df[df["year"] == 2026].groupby("month")["pnl_yuan"].sum(),
        "max_gross": int(gross_arr.max()),
        "avg_gross": gross_arr[eval_start:].mean(),
        "max_long": int(long_arr.max()),
        "max_short": int(short_arr.max()),
        "max_margin": int(gross_arr.max()) * MARGIN_PER_LOT,
        "avg_margin": gross_arr[eval_start:].mean() * MARGIN_PER_LOT,
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

    print(f"账户假设:    {ACCOUNT_EQUITY:,} 元")
    print(f"保证金/手:   {MARGIN_PER_LOT:,} 元")
    print(f"合约乘数:    {CONTRACT_MULT}")

    modes = [
        ("保守 (1×lot, cap=10)",   1, 10),
        ("中等 (1×lot, cap=14)",   1, 14),
        ("激进 (2×lot, cap=20)",   2, 20),
    ]

    results = []
    for label, lps, cap in modes:
        print(f"\n[Running] {label}...")
        m = run_with_position_tracking(label, lps, cap, ctx, dates, closes, highs, lows, atr20, regime)
        results.append(m)

    # ── 表 1: 核心指标 ──
    print(f"\n{'═' * 105}")
    print(" 三档模式对比 — 核心指标 (Walk-forward 2.9 yr)")
    print(f"{'═' * 105}")
    print(f"  {'Mode':<25} {'N':>4} {'skip':>5} {'年化¥':>11} "
          f"{'MaxDD¥':>11} {'Calmar':>7} {'Sharpe':>7} {'WR':>5} {'Sigs/yr':>8}")
    print("  " + "-" * 102)
    for m in results:
        if m["n"] == 0:
            continue
        n_yrs = 2.9
        sigs_per_yr = m["n"] / n_yrs
        print(f"  {m['label']:<25} {m['n']:>4} {m['n_skipped']:>5} "
              f"{m['ann_yuan']:>+11,.0f} {m['max_dd_yuan']:>+11,.0f} "
              f"{m['calmar']:>6.2f} {m['sharpe']:>6.2f} {m['wr']:>4.0f}% "
              f"{sigs_per_yr:>7.1f}")

    # ── 表 2: 账户回报 + 保证金 ──
    print(f"\n{'═' * 105}")
    print(f" 账户回报率 (基于 {ACCOUNT_EQUITY:,} 元账户)")
    print(f"{'═' * 105}")
    print(f"  {'Mode':<25} {'年化%':>8} {'MaxDD%':>8} "
          f"{'保证金峰值¥':>15} {'保证金平均¥':>15} {'峰值占用':>10}")
    print("  " + "-" * 100)
    for m in results:
        if m["n"] == 0:
            continue
        ret_pct = m["ann_yuan"] / ACCOUNT_EQUITY * 100
        dd_pct = m["max_dd_yuan"] / ACCOUNT_EQUITY * 100
        margin_pct = m["max_margin"] / ACCOUNT_EQUITY * 100
        print(f"  {m['label']:<25} {ret_pct:>+7.1f}% {dd_pct:>+7.1f}% "
              f"{m['max_margin']:>+15,.0f} {m['avg_margin']:>+15,.0f} "
              f"{margin_pct:>9.1f}%")

    # ── 表 3: 仓位峰值 ──
    print(f"\n{'═' * 105}")
    print(" 仓位峰值")
    print(f"{'═' * 105}")
    print(f"  {'Mode':<25} {'L_max':>6} {'S_max':>6} {'GROSS_max':>11} {'GROSS_avg':>11}")
    for m in results:
        if m["n"] == 0:
            continue
        print(f"  {m['label']:<25} {m['max_long']:>6} {m['max_short']:>6} "
              f"{m['max_gross']:>11} {m['avg_gross']:>11.2f}")

    # ── 表 4: 各年 PnL ──
    print(f"\n{'═' * 105}")
    print(" 各年 PnL")
    print(f"{'═' * 105}")
    print(f"  {'Mode':<25} {'2023':>11} {'2024':>13} {'2025':>13} {'2026':>13}")
    for m in results:
        if m["n"] == 0:
            continue
        ys = [m["yearly"].get(y, 0) for y in [2023, 2024, 2025, 2026]]
        print(f"  {m['label']:<25} {ys[0]:>+11,.0f} {ys[1]:>+13,.0f} "
              f"{ys[2]:>+13,.0f} {ys[3]:>+13,.0f}")

    # ── 表 5: 2026 月度（黑天鹅期） ──
    print(f"\n{'═' * 100}")
    print(" 2026 月度 PnL (Trump-Warsh 黑天鹅期)")
    print(f"{'═' * 100}")
    for m in results:
        if m["n"] == 0:
            continue
        print(f"  {m['label']:<25}  ", end="")
        for mo in ["202601", "202602", "202603"]:
            v = m["monthly_2026"].get(mo, 0)
            print(f"{mo[-2:]}: {v:>+13,.0f}  ", end="")
        print()


if __name__ == "__main__":
    main()
