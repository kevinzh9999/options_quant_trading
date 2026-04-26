#!/usr/bin/env python3
"""Self-backtest using the production daily_xgb modules end-to-end.

Verifies parity with the M7 baseline walk-forward research result:
  Expected: ~+1,572K annual / 1 lot, MaxDD -536K, Calmar 2.93, Sharpe 5.64

This re-runs the same logic but through the production code paths (factors.py,
regime.py, pipeline.py) instead of the standalone research scripts.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from strategies.daily_xgb.config import DailyXGBConfig
from strategies.daily_xgb.factors import build_full_pipeline, load_default_context
from strategies.daily_xgb.regime import compute_regime, decide_trade_params
from strategies.daily_xgb.pipeline import train_model

CONTRACT_MULT = 200
SLIPPAGE = 0.0008


def compute_atr20(closes, highs, lows):
    n = len(closes)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    atr = np.full(n, np.nan)
    for i in range(20, n):
        atr[i] = tr[i-19:i+1].mean()
    return atr / np.maximum(closes, 1)


def simulate_trade(direction, entry_idx, closes, highs, lows, hold_days, sl_pct):
    n = len(closes)
    entry_close = closes[entry_idx]
    sign = 1 if direction == "LONG" else -1
    for offset in range(1, hold_days + 1):
        idx = entry_idx + offset
        if idx >= n:
            return n - 1, sign * (closes[-1] / entry_close - 1), "TIME"
        if direction == "LONG":
            worst = (lows[idx] - entry_close) / entry_close
            if worst <= -sl_pct:
                return idx, -sl_pct, "SL"
        else:
            worst = (entry_close - highs[idx]) / entry_close
            if worst <= -sl_pct:
                return idx, -sl_pct, "SL"
    exit_idx = entry_idx + hold_days
    return exit_idx, sign * (closes[exit_idx] / entry_close - 1), "TIME"


def main():
    cfg = DailyXGBConfig()
    print(f"=== Self-backtest using daily_xgb production modules ===")
    print(f"Conservative mode: 1 lot/signal, cap={cfg.concurrent_cap}")

    ctx = load_default_context(cfg.db_path)
    iv_dates = set(ctx.iv_df["trade_date"].tolist())
    px_e = ctx.px_df[ctx.px_df["trade_date"].isin(iv_dates)].reset_index(drop=True)
    dates = px_e["trade_date"].tolist()
    closes = px_e["close"].astype(float).values
    highs = px_e["high"].astype(float).values
    lows = px_e["low"].astype(float).values
    atr20 = compute_atr20(closes, highs, lows)
    n = len(dates)

    # Walk-forward
    fwd5 = np.array([
        closes[i + cfg.hold_days_default] / closes[i] - 1
        if i + cfg.hold_days_default < n else np.nan
        for i in range(n)
    ])

    pipeline = build_full_pipeline()
    X_full = pipeline.features_matrix(dates, ctx)
    print(f"Feature matrix: {X_full.shape}")

    cached_model = None
    last_train = -cfg.retrain_every
    trades = []
    daily_pnl = np.zeros(n)
    open_positions = []   # (exit_idx, direction, lots)
    n_skipped_cap = 0
    n_skipped_gate = 0

    for td_idx in range(cfg.initial_train_days, n - 16):
        # Cleanup expired
        open_positions = [(ei, d, l) for (ei, d, l) in open_positions if ei > td_idx]

        # Retrain?
        if td_idx - last_train >= cfg.retrain_every:
            train_end_idx = td_idx - cfg.hold_days_default
            X_tr = X_full[:train_end_idx]
            y_tr = fwd5[:train_end_idx]
            valid = ~(np.isnan(X_tr).any(axis=1) | np.isnan(y_tr))
            if valid.sum() < cfg.initial_train_days:
                continue
            X_v = X_tr[valid]
            y_v = y_tr[valid]
            pipeline_new = build_full_pipeline()
            pipeline_new.train(
                X_v, y_v,
                n_estimators=cfg.xgb_n_estimators,
                max_depth=cfg.xgb_max_depth,
                learning_rate=cfg.xgb_learning_rate,
                min_child_weight=cfg.xgb_min_child_weight,
                subsample=cfg.xgb_subsample,
                colsample_bytree=cfg.xgb_colsample_bytree,
                reg_alpha=cfg.xgb_reg_alpha,
                reg_lambda=cfg.xgb_reg_lambda,
                random_state=cfg.xgb_random_state,
            )
            pred_in = pipeline_new.predict(X_v)
            cached_model = {
                "pipeline": pipeline_new,
                "top": float(np.quantile(pred_in, 1 - cfg.top_pct)),
                "bot": float(np.quantile(pred_in, cfg.bot_pct)),
            }
            last_train = td_idx

        if cached_model is None:
            continue
        x = X_full[td_idx:td_idx+1]
        if np.isnan(x).any():
            continue
        pred = float(cached_model["pipeline"].predict(x)[0])
        if pred >= cached_model["top"]:
            direction = "LONG"
        elif pred <= cached_model["bot"]:
            direction = "SHORT"
        else:
            continue

        # regime
        state = compute_regime(closes, td_idx)
        params = decide_trade_params(direction, state, cfg)
        if params is None:
            n_skipped_gate += 1
            continue

        if np.isnan(atr20[td_idx]):
            continue
        sl = params.atr_k * atr20[td_idx]
        if td_idx + params.hold_days >= n:
            continue

        # Cap check
        cur_gross = sum(l for (_, _, l) in open_positions)
        if cur_gross + cfg.lots_per_signal > cfg.concurrent_cap:
            n_skipped_cap += 1
            continue

        exit_idx, gross_ret, reason = simulate_trade(
            direction, td_idx, closes, highs, lows, params.hold_days, sl,
        )
        net_ret = gross_ret - SLIPPAGE
        pnl_pts = net_ret * closes[td_idx] * cfg.lots_per_signal
        daily_pnl[exit_idx] += pnl_pts
        open_positions.append((exit_idx, direction, cfg.lots_per_signal))

        trades.append({
            "entry_date": dates[td_idx],
            "year": int(dates[td_idx][:4]),
            "month": dates[td_idx][:6],
            "direction": direction,
            "enhancement": params.enhancement,
            "hold_used": params.hold_days,
            "atr_k": params.atr_k,
            "sl_pct": sl,
            "net_ret": net_ret,
            "pnl_yuan": pnl_pts * CONTRACT_MULT,
            "reason": reason,
        })

    df = pd.DataFrame(trades)
    eval_start = cfg.initial_train_days
    cum = np.cumsum(daily_pnl[eval_start:]) * CONTRACT_MULT
    final = cum[-1]
    peak = np.maximum.accumulate(cum)
    max_dd = (cum - peak).min()
    days_arr = daily_pnl[eval_start:]
    days_arr = days_arr[days_arr != 0]
    sharpe = (days_arr.mean() / days_arr.std() * np.sqrt(252)
              if len(days_arr) > 1 and days_arr.std() > 0 else 0)
    n_yrs = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[eval_start])).days / 365.0

    print(f"\n=== Production module backtest results ===")
    print(f"  Total trades:    {len(df)}")
    print(f"    LONG:          {(df['direction']=='LONG').sum()}")
    print(f"    SHORT:         {(df['direction']=='SHORT').sum()}")
    print(f"  Skipped (gate):  {n_skipped_gate}")
    print(f"  Skipped (cap):   {n_skipped_cap}")
    print(f"  WR:              {(df['net_ret']>0).mean()*100:.1f}%")
    print(f"  Avg net/trade:   {df['net_ret'].mean()*100:+.3f}%")
    print(f"  Total PnL:       {final:+,.0f} 元 ({n_yrs:.1f} 年)")
    print(f"  年化:            {final/max(n_yrs,0.1):+,.0f} 元/年")
    print(f"  MaxDD:           {max_dd:+,.0f}")
    print(f"  Sharpe:          {sharpe:.2f}")
    print(f"  Calmar:          {(final/max(n_yrs,0.1))/abs(max_dd):.2f}" if max_dd<0 else "  Calmar: N/A")

    print(f"\n=== Enhancement type breakdown ===")
    for et, sub in df.groupby("enhancement"):
        print(f"  {et:<12} N={len(sub):>3}  PnL={sub['pnl_yuan'].sum():>+12,.0f}  "
              f"avg={sub['net_ret'].mean()*100:+.2f}%  WR={(sub['net_ret']>0).mean()*100:.0f}%")

    print(f"\n=== Year × Direction ===")
    print(df.groupby(["year", "direction"])["pnl_yuan"].sum().unstack(fill_value=0).to_string())

    print(f"\n=== Expected (M7 baseline + cap=10) ===")
    print(f"  Annual:          ~+1,472K (cap=10)")
    print(f"  MaxDD:           ~-536K")
    print(f"  Calmar:          2.75")
    print(f"  Sharpe:          5.37")

    df.to_csv("/tmp/daily_xgb_self_bt.csv", index=False)
    print(f"\nSaved trades: /tmp/daily_xgb_self_bt.csv")


if __name__ == "__main__":
    main()
