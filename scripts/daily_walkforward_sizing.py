#!/usr/bin/env python3
"""测两种 position sizing 框架: B (fixed risk) 和 D (concurrent cap)

B: lots = round(account × risk_pct / (sl_pct × close × mult))
   每笔风险固定，enhanced trade 因 SL 宽自动 size 小

D: 单日 GROSS 持仓 cap = N 手
   超过 cap 的新信号被跳过

输出: PnL / MaxDD / Sharpe / Calmar，对比 M7 1-lot baseline
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import math
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
ACCOUNT_EQUITY = 6_400_000   # 640 万
SLIPPAGE_PCT = 0.0008


def build_pipe():
    base = build_default_pipeline()
    base.factors.extend([CloseSma60Factor(), Slope60dFactor(), VolRegimeFactor()])
    base.feature_names = [f.name for f in base.factors]
    return base


def run_strategy(label, ctx, dates, closes, highs, lows, atr20, regime,
                  sizing_mode="A_unit",
                  fixed_risk_pct=None,    # for sizing_mode="B"
                  concurrent_cap=None,    # for sizing_mode="D"
                  ):
    """sizing_mode:
        - 'A_unit': 1 lot per signal (baseline)
        - 'B':       fixed risk pct per trade → lots = round(equity × pct / (sl × close × mult))
        - 'D':       1 lot per signal but cap GROSS concurrent positions
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
    open_positions = []   # list of (exit_idx, direction, lots)
    n_skipped = 0

    def cleanup_expired(td_idx):
        return [(ei, d, l) for (ei, d, l) in open_positions if ei > td_idx]

    for td_idx in range(INITIAL_TRAIN_DAYS, n - 16):
        # Cleanup expired open positions
        open_positions = cleanup_expired(td_idx)

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

        # ── Sizing logic ──
        if sizing_mode == "A_unit":
            lots = 1
        elif sizing_mode == "B":
            # lots = round(equity × risk_pct / (sl_pct × close × mult))
            risk_per_lot = sl * closes[td_idx] * CONTRACT_MULT
            target_risk = ACCOUNT_EQUITY * fixed_risk_pct
            lots = max(1, round(target_risk / risk_per_lot))
            # Cap to reasonable max (e.g., don't exceed 10 lots even if risk is super low)
            lots = min(lots, 10)
        elif sizing_mode == "D":
            current_gross = sum(l for (_, _, l) in open_positions)
            if current_gross + 1 > concurrent_cap:
                n_skipped += 1
                continue
            lots = 1
        else:
            lots = 1

        if lots == 0:
            n_skipped += 1
            continue

        exit_idx, gross_ret, reason = simulate_trade_dynamic(
            direction, td_idx, closes, highs, lows, hold, sl,
        )
        net_ret = gross_ret - SLIPPAGE_PCT
        # PnL scales with lots
        pnl_pts = net_ret * closes[td_idx]
        pnl_yuan = pnl_pts * CONTRACT_MULT * lots
        daily_pnl[exit_idx] += pnl_pts * lots   # multiply lots into pts series
        open_positions.append((exit_idx, direction, lots))
        trades.append({
            "entry_date": dates[td_idx], "year": int(dates[td_idx][:4]),
            "direction": direction, "lots": lots, "sl_pct": sl,
            "net_ret": net_ret, "pnl_yuan": pnl_yuan,
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
        "label": label, "n": len(df), "n_skipped": n_skipped,
        "wr": (df["net_ret"] > 0).mean() * 100,
        "ann_yuan": final / max(n_yrs, 0.1),
        "max_dd_yuan": max_dd, "sharpe": sharpe,
        "calmar": (final / max(n_yrs, 0.1)) / abs(max_dd) if max_dd < 0 else 0,
        "avg_lots": df["lots"].mean(),
        "median_lots": df["lots"].median(),
        "max_lots": df["lots"].max(),
        "long_pnl": df[df["direction"] == "LONG"]["pnl_yuan"].sum(),
        "short_pnl": df[df["direction"] == "SHORT"]["pnl_yuan"].sum(),
        "yearly": df.groupby("year")["pnl_yuan"].sum(),
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

    print(f"Account equity assumption: {ACCOUNT_EQUITY:,} 元")
    print(f"Contract multiplier: {CONTRACT_MULT}")

    variants = [
        ("A0: 1 lot per signal (baseline)",        "A_unit", None,    None),
        ("B1: fixed 0.3% risk/trade",              "B",      0.003,   None),
        ("B2: fixed 0.5% risk/trade",              "B",      0.005,   None),
        ("B3: fixed 0.75% risk/trade",             "B",      0.0075,  None),
        ("B4: fixed 1.0% risk/trade",              "B",      0.01,    None),
        ("B5: fixed 1.5% risk/trade",              "B",      0.015,   None),
        ("B6: fixed 2.0% risk/trade",              "B",      0.02,    None),
        ("D1: cap 4 lots concurrent",              "D",      None,    4),
        ("D2: cap 6 lots concurrent",              "D",      None,    6),
        ("D3: cap 8 lots concurrent",              "D",      None,    8),
        ("D4: cap 10 lots concurrent",             "D",      None,    10),
        ("D5: cap 12 lots concurrent",             "D",      None,    12),
    ]

    results = []
    for label, mode, frp, cap in variants:
        print(f"\n[Running] {label}...")
        m = run_strategy(label, ctx, dates, closes, highs, lows, atr20, regime,
                          sizing_mode=mode, fixed_risk_pct=frp, concurrent_cap=cap)
        results.append(m)

    print(f"\n{'═' * 110}")
    print(f" Position sizing sweep (account = {ACCOUNT_EQUITY:,})")
    print(f"{'═' * 110}")
    print(f"  {'Variant':<40} {'N':>4} {'skip':>5} {'avgL':>6} {'maxL':>5} {'WR':>5} "
          f"{'年化¥':>11} {'MaxDD¥':>11} {'Calmar':>7} {'Sharpe':>7}")
    print("  " + "-" * 108)
    for m in results:
        if m["n"] == 0:
            continue
        print(f"  {m['label']:<40} {m['n']:>4} {m.get('n_skipped',0):>5} "
              f"{m['avg_lots']:>5.1f} {int(m['max_lots']):>5} {m['wr']:>4.0f}% "
              f"{m['ann_yuan']:>+11,.0f} {m['max_dd_yuan']:>+11,.0f} "
              f"{m['calmar']:>6.2f} {m['sharpe']:>6.2f}")

    # Annualized return on equity
    print(f"\n{'═' * 110}")
    print(" 账户回报率 (年化¥ / equity)")
    print(f"{'═' * 110}")
    for m in results:
        if m["n"] == 0:
            continue
        ret_pct = m["ann_yuan"] / ACCOUNT_EQUITY * 100
        dd_pct = m["max_dd_yuan"] / ACCOUNT_EQUITY * 100
        print(f"  {m['label']:<40} 年化 {ret_pct:>+6.1f}%  MaxDD {dd_pct:>+6.1f}%  "
              f"Calmar {m['calmar']:>5.2f}")

    # Yearly breakdown
    print(f"\n{'═' * 100}")
    print(" 年度 PnL")
    print(f"{'═' * 100}")
    print(f"  {'Variant':<40} {'2023':>9} {'2024':>11} {'2025':>11} {'2026':>11}")
    for m in results:
        if m["n"] == 0:
            continue
        ys = [m["yearly"].get(y, 0) for y in [2023, 2024, 2025, 2026]]
        print(f"  {m['label']:<40} {ys[0]:>+9,.0f} {ys[1]:>+11,.0f} "
              f"{ys[2]:>+11,.0f} {ys[3]:>+11,.0f}")


if __name__ == "__main__":
    main()
