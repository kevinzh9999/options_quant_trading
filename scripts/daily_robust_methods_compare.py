#!/usr/bin/env python3
"""Robust regime-aware enhancements — A/B/C/D individual + combinations

Baseline = strict OOS (train ≤ 2024-12-31, test 2025-01+, frozen model + thresholds)

Variants:
  A. ATR-scaled stop-loss (SL = k × ATR20 on entry day)
  B. Confidence filter (only trade pred in extreme tails P_hi/P_lo)
  C. Trailing stop (exit on 50% retracement of MFE)
  D. Regime features (add close/sma60, slope60, vol-regime to factor pipeline)

For each variant we measure:
  - N trades, WR, avg net %
  - Total OOS PnL (元 / 1手)
  - Annualized PnL
  - Max DD
  - Sharpe
  - Worst single trade

Trades are simulated day-by-day during the 5-day hold, so SL/trailing can fire
intraday (using close-to-close path, since we only have daily bars).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from typing import List, Optional

import numpy as np
import pandas as pd

from strategies.daily.factors import (
    DailyContext,
    DailyFactor,
    DailyFactorPipeline,
    build_default_pipeline,
    load_default_context,
)

DB_PATH = "/Users/kevinzhao/Documents/options_quant_trading/data/storage/trading.db"

HOLD_DAYS = 5
TOP_PCT = 0.20         # baseline LONG threshold = top 20%
BOT_PCT = 0.20         # baseline SHORT threshold = bot 20%
CONF_TOP_PCT = 0.10    # variant B: extreme tails (top 10% / bot 10%)
CONF_BOT_PCT = 0.10
CONTRACT_MULT = 200
SLIPPAGE_PCT = 0.0008  # 0.08% per trade
ATR_K = 1.5            # variant A: SL = k × ATR20
TRAIL_RETRACE = 0.50   # variant C: exit on 50% MFE retracement

TRAIN_END = "20241231"
TEST_START = "20250101"


# ═══════════════════════════════════════════════════════════════════════════
# Variant D — extra regime factors
# ═══════════════════════════════════════════════════════════════════════════

class CloseSma60Factor(DailyFactor):
    name = "close_sma60_ratio"
    category = "regime"
    def compute(self, td, ctx):
        h = ctx.px_history(td, 60)
        if h is None or len(h) < 60:
            return None
        return float(h.iloc[-1]["close"] / h["close"].mean())


class Slope60dFactor(DailyFactor):
    name = "slope_60d"
    category = "regime"
    def compute(self, td, ctx):
        h = ctx.px_history(td, 60)
        if h is None or len(h) < 60:
            return None
        c = h["close"].astype(float).values
        slope = float(np.polyfit(range(60), c, 1)[0])
        mean = float(c.mean())
        return slope / max(mean, 1)


class VolRegimeFactor(DailyFactor):
    """Recent 5d realized vol / 60d realized vol — high = vol expansion regime."""
    name = "vol_regime"
    category = "regime"
    def compute(self, td, ctx):
        h = ctx.px_history(td, 61)
        if h is None or len(h) < 61:
            return None
        c = h["close"].astype(float).values
        rets = np.diff(np.log(c))
        rv5 = float(np.std(rets[-5:]) * np.sqrt(252))
        rv60 = float(np.std(rets) * np.sqrt(252))
        return rv5 / max(rv60, 1e-6)


def build_pipeline_with_regime() -> DailyFactorPipeline:
    base = build_default_pipeline()
    base.factors.extend([CloseSma60Factor(), Slope60dFactor(), VolRegimeFactor()])
    base.feature_names = [f.name for f in base.factors]
    return base


# ═══════════════════════════════════════════════════════════════════════════
# Pre-compute ATR(20) per date — used by variant A
# ═══════════════════════════════════════════════════════════════════════════

def compute_atr20(px_df: pd.DataFrame) -> np.ndarray:
    """ATR20 in % of close, aligned to px_df row index."""
    h = px_df["high"].astype(float).values
    l = px_df["low"].astype(float).values
    c = px_df["close"].astype(float).values
    tr = np.zeros(len(c))
    tr[0] = h[0] - l[0]
    for i in range(1, len(c)):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    atr = np.full(len(c), np.nan)
    win = 20
    for i in range(win, len(c)):
        atr[i] = tr[i-win+1:i+1].mean()
    return atr / np.maximum(c, 1)   # as fraction of close


# ═══════════════════════════════════════════════════════════════════════════
# Trade simulator — supports A/C variants
# ═══════════════════════════════════════════════════════════════════════════

def simulate_trade(direction: str, entry_idx: int, closes: np.ndarray,
                    highs: np.ndarray, lows: np.ndarray,
                    hold_days: int = HOLD_DAYS,
                    sl_pct: Optional[float] = None,
                    use_trail: bool = False,
                    trail_retrace: float = TRAIL_RETRACE,
                    slippage: float = SLIPPAGE_PCT):
    """Walk forward day by day, applying SL / trailing stop. Return (exit_idx, gross_ret, exit_reason)."""
    entry_close = closes[entry_idx]
    sign = 1 if direction == "LONG" else -1
    mfe = 0.0  # max favorable excursion (in %)
    for offset in range(1, hold_days + 1):
        idx = entry_idx + offset
        if idx >= len(closes):
            return entry_idx + hold_days if entry_idx + hold_days < len(closes) else len(closes) - 1, \
                sign * (closes[-1] / entry_close - 1), "TIME"

        # Check SL using bar low/high (worst-case path)
        if sl_pct is not None:
            if direction == "LONG":
                worst = lows[idx]
                worst_ret = (worst - entry_close) / entry_close
                if worst_ret <= -sl_pct:
                    exit_px = entry_close * (1 - sl_pct)
                    return idx, (exit_px - entry_close) / entry_close, "SL"
            else:
                worst = highs[idx]
                worst_ret = (entry_close - worst) / entry_close
                if worst_ret <= -sl_pct:
                    exit_px = entry_close * (1 + sl_pct)
                    return idx, (entry_close - exit_px) / entry_close, "SL"

        # Track MFE on close basis
        cur_close = closes[idx]
        cur_ret = sign * (cur_close - entry_close) / entry_close
        if use_trail and cur_ret > mfe:
            mfe = cur_ret
        # Trailing stop fires when retracement > threshold of MFE
        if use_trail and mfe > 0.005:  # only after meaningful run-up (>0.5%)
            if cur_ret < mfe * (1 - trail_retrace):
                return idx, sign * (cur_close - entry_close) / entry_close, "TRAIL"

    exit_idx = entry_idx + hold_days
    return exit_idx, sign * (closes[exit_idx] / entry_close - 1), "TIME"


def run_variant(label: str, predictions: np.ndarray, closes: np.ndarray,
                 highs: np.ndarray, lows: np.ndarray, dates: list,
                 atr20: np.ndarray, train_end_idx: int, test_start_idx: int,
                 thresholds: tuple,
                 atr_sl: bool = False,
                 use_trail: bool = False,
                 fixed_sl: Optional[float] = None) -> dict:
    """Run backtest with variant-specific exit logic."""
    top_thr, bot_thr = thresholds
    trades = []
    daily_pnl = np.zeros(len(closes))

    for i in range(test_start_idx, len(closes) - HOLD_DAYS):
        pred = predictions[i]
        if np.isnan(pred):
            continue
        if pred >= top_thr:
            direction = "LONG"
        elif pred <= bot_thr:
            direction = "SHORT"
        else:
            continue

        # Per-trade SL
        sl = None
        if atr_sl and not np.isnan(atr20[i]):
            sl = ATR_K * atr20[i]
        elif fixed_sl is not None:
            sl = fixed_sl

        exit_idx, gross_ret, reason = simulate_trade(
            direction, i, closes, highs, lows,
            hold_days=HOLD_DAYS, sl_pct=sl,
            use_trail=use_trail,
        )
        net_ret = gross_ret - SLIPPAGE_PCT
        pnl_pts = net_ret * closes[i]
        daily_pnl[exit_idx] += pnl_pts
        trades.append({
            "entry_date": dates[i], "exit_date": dates[exit_idx],
            "direction": direction, "pred": pred,
            "gross_ret": gross_ret, "net_ret": net_ret,
            "pnl_pts": pnl_pts, "pnl_yuan": pnl_pts * CONTRACT_MULT,
            "reason": reason, "hold": exit_idx - i,
        })

    return _compile_metrics(label, trades, daily_pnl, dates, test_start_idx)


def _compile_metrics(label: str, trades: list, daily_pnl: np.ndarray,
                      dates: list, test_start_idx: int) -> dict:
    if not trades:
        return {"label": label, "n": 0}
    df = pd.DataFrame(trades)
    cum = np.cumsum(daily_pnl[test_start_idx:]) * CONTRACT_MULT
    final = cum[-1] if len(cum) else 0
    peak = np.maximum.accumulate(cum) if len(cum) else np.array([0])
    max_dd = (cum - peak).min() if len(cum) else 0
    days_arr = daily_pnl[test_start_idx:]
    days_arr = days_arr[days_arr != 0]
    sharpe = (days_arr.mean() / days_arr.std() * np.sqrt(252)
              if len(days_arr) > 1 and days_arr.std() > 0 else 0)
    n_yrs = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[test_start_idx])).days / 365.0
    return {
        "label": label,
        "n": len(df),
        "wr": (df["net_ret"] > 0).mean() * 100,
        "avg_pct": df["net_ret"].mean() * 100,
        "worst_pct": df["net_ret"].min() * 100,
        "best_pct": df["net_ret"].max() * 100,
        "final_yuan": final,
        "ann_yuan": final / max(n_yrs, 0.1),
        "max_dd_yuan": max_dd,
        "sharpe": sharpe,
        "long_n": (df["direction"] == "LONG").sum(),
        "short_n": (df["direction"] == "SHORT").sum(),
        "long_pnl": df[df["direction"] == "LONG"]["pnl_yuan"].sum(),
        "short_pnl": df[df["direction"] == "SHORT"]["pnl_yuan"].sum(),
        "trades": df,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def train_pipeline(pipe: DailyFactorPipeline, ctx: DailyContext,
                    dates: list, closes: np.ndarray, train_end_idx: int,
                    label: str):
    """Train pipeline frozen on data ≤ train_end_idx-HOLD_DAYS."""
    X_full = pipe.features_matrix(dates, ctx)
    fwd = np.array([
        closes[i + HOLD_DAYS] / closes[i] - 1 if i + HOLD_DAYS < len(closes) else np.nan
        for i in range(len(closes))
    ])
    cutoff = train_end_idx - HOLD_DAYS
    X_tr = X_full[:cutoff]
    y_tr = fwd[:cutoff]
    valid = ~(np.isnan(X_tr).any(axis=1) | np.isnan(y_tr))
    X_tr_v, y_tr_v = X_tr[valid], y_tr[valid]
    pipe.train(X_tr_v, y_tr_v)

    # In-sample preds for thresholds
    pred_in = pipe.predict(X_tr_v)
    top = float(np.quantile(pred_in, 1 - TOP_PCT))
    bot = float(np.quantile(pred_in, BOT_PCT))
    conf_top = float(np.quantile(pred_in, 1 - CONF_TOP_PCT))
    conf_bot = float(np.quantile(pred_in, CONF_BOT_PCT))
    train_ic = np.corrcoef(pred_in, y_tr_v)[0, 1]
    print(f"  [{label}] train IC: {train_ic:+.4f}  thr P{(1-TOP_PCT)*100:.0f}/P{BOT_PCT*100:.0f}: "
          f"{top:+.4f}/{bot:+.4f}  conf P{(1-CONF_TOP_PCT)*100:.0f}/P{CONF_BOT_PCT*100:.0f}: "
          f"{conf_top:+.4f}/{conf_bot:+.4f}")

    # Predict full series
    preds = np.full(len(dates), np.nan)
    valid_full = ~np.isnan(X_full).any(axis=1)
    preds[valid_full] = pipe.predict(X_full[valid_full])

    return preds, (top, bot), (conf_top, conf_bot), train_ic


def print_row(m: dict, header: bool = False):
    if header:
        print(f"  {'Variant':<22} {'N':>4} {'WR':>5} {'Avg%':>6} "
              f"{'Worst%':>7} {'Total¥':>10} {'年化¥':>10} {'MaxDD¥':>10} "
              f"{'Sharpe':>6} {'L/S':>9} {'L¥':>10} {'S¥':>10}")
        print("  " + "-" * 130)
        return
    if m["n"] == 0:
        print(f"  {m['label']:<22} (no trades)")
        return
    print(f"  {m['label']:<22} {m['n']:>4} {m['wr']:>4.0f}% "
          f"{m['avg_pct']:>+5.2f}% {m['worst_pct']:>+6.1f}% "
          f"{m['final_yuan']:>+10,.0f} {m['ann_yuan']:>+10,.0f} "
          f"{m['max_dd_yuan']:>+10,.0f} {m['sharpe']:>6.2f} "
          f"{m['long_n']:>3}/{m['short_n']:<3} "
          f"{m['long_pnl']:>+10,.0f} {m['short_pnl']:>+10,.0f}")


def main():
    print("Loading data + computing features...")
    ctx = load_default_context(DB_PATH)
    px = ctx.px_df.copy()
    iv_dates = set(ctx.iv_df["trade_date"].tolist())
    px_e = px[px["trade_date"].isin(iv_dates)].reset_index(drop=True)

    dates = px_e["trade_date"].tolist()
    closes = px_e["close"].astype(float).values
    highs = px_e["high"].astype(float).values
    lows = px_e["low"].astype(float).values
    atr20 = compute_atr20(px_e)

    train_end_idx = next((i for i, d in enumerate(dates) if d > TRAIN_END), len(dates))
    test_start_idx = next((i for i, d in enumerate(dates) if d >= TEST_START), len(dates))

    print(f"Range: {dates[0]} ~ {dates[-1]}")
    print(f"Train ≤ {dates[train_end_idx-1]} (idx {train_end_idx-1})")
    print(f"Test ≥ {dates[test_start_idx]} (idx {test_start_idx}, {len(dates)-test_start_idx} days)")

    # ── Train baseline pipeline (no regime feats) ──
    print("\n=== Training pipelines ===")
    pipe_base = build_default_pipeline()
    preds_base, thr_base, thr_conf, ic_base = train_pipeline(
        pipe_base, ctx, dates, closes, train_end_idx, "base")

    # ── Train regime-augmented pipeline (variant D) ──
    pipe_regime = build_pipeline_with_regime()
    preds_regime, thr_regime, _, ic_regime = train_pipeline(
        pipe_regime, ctx, dates, closes, train_end_idx, "regime")

    # ── Evaluate strict-OOS IC for both ──
    fwd = np.array([closes[i+HOLD_DAYS]/closes[i] - 1 if i+HOLD_DAYS < len(closes) else np.nan
                    for i in range(len(closes))])
    test_y = fwd[test_start_idx:]
    for label, p in [("base", preds_base), ("regime", preds_regime)]:
        valid = ~(np.isnan(p[test_start_idx:]) | np.isnan(test_y))
        ic = np.corrcoef(p[test_start_idx:][valid], test_y[valid])[0, 1]
        print(f"  [{label}] strict-OOS IC: {ic:+.4f}")

    # ── Run all variants ──
    print(f"\n{'=' * 130}")
    print(" Single-variant comparison")
    print(f"{'=' * 130}")
    print_row({}, header=True)

    results = {}

    # Baseline (no enhancements)
    results["baseline"] = run_variant(
        "0. Baseline", preds_base, closes, highs, lows, dates, atr20,
        train_end_idx, test_start_idx, thr_base)
    print_row(results["baseline"])

    # A. ATR-scaled SL
    results["A_atr"] = run_variant(
        f"A. ATR×{ATR_K} SL", preds_base, closes, highs, lows, dates, atr20,
        train_end_idx, test_start_idx, thr_base, atr_sl=True)
    print_row(results["A_atr"])

    # B. Confidence filter (extreme tails)
    results["B_conf"] = run_variant(
        f"B. Conf P{(1-CONF_TOP_PCT)*100:.0f}/P{CONF_BOT_PCT*100:.0f}",
        preds_base, closes, highs, lows, dates, atr20,
        train_end_idx, test_start_idx, thr_conf)
    print_row(results["B_conf"])

    # C. Trailing stop
    results["C_trail"] = run_variant(
        f"C. Trail {int(TRAIL_RETRACE*100)}%",
        preds_base, closes, highs, lows, dates, atr20,
        train_end_idx, test_start_idx, thr_base, use_trail=True)
    print_row(results["C_trail"])

    # D. Regime features
    results["D_regime"] = run_variant(
        "D. Regime feats",
        preds_regime, closes, highs, lows, dates, atr20,
        train_end_idx, test_start_idx, thr_regime)
    print_row(results["D_regime"])

    # ── Combinations ──
    print(f"\n{'=' * 130}")
    print(" Pairwise combinations")
    print(f"{'=' * 130}")
    print_row({}, header=True)

    # A + B
    results["A+B"] = run_variant(
        "A+B (ATR + Conf)", preds_base, closes, highs, lows, dates, atr20,
        train_end_idx, test_start_idx, thr_conf, atr_sl=True)
    print_row(results["A+B"])

    # A + C
    results["A+C"] = run_variant(
        "A+C (ATR + Trail)", preds_base, closes, highs, lows, dates, atr20,
        train_end_idx, test_start_idx, thr_base, atr_sl=True, use_trail=True)
    print_row(results["A+C"])

    # B + C
    results["B+C"] = run_variant(
        "B+C (Conf + Trail)", preds_base, closes, highs, lows, dates, atr20,
        train_end_idx, test_start_idx, thr_conf, use_trail=True)
    print_row(results["B+C"])

    # A + D
    results["A+D"] = run_variant(
        "A+D (ATR + Regime)", preds_regime, closes, highs, lows, dates, atr20,
        train_end_idx, test_start_idx, thr_regime, atr_sl=True)
    print_row(results["A+D"])

    # B + D
    results["B+D"] = run_variant(
        "B+D (Conf + Regime)", preds_regime, closes, highs, lows, dates, atr20,
        train_end_idx, test_start_idx,
        (float(np.quantile(pipe_regime.predict(pipe_regime.features_matrix(dates[:train_end_idx-HOLD_DAYS], ctx)[~np.isnan(pipe_regime.features_matrix(dates[:train_end_idx-HOLD_DAYS], ctx)).any(axis=1)]), 1-CONF_TOP_PCT)),
         float(np.quantile(pipe_regime.predict(pipe_regime.features_matrix(dates[:train_end_idx-HOLD_DAYS], ctx)[~np.isnan(pipe_regime.features_matrix(dates[:train_end_idx-HOLD_DAYS], ctx)).any(axis=1)]), CONF_BOT_PCT))))
    print_row(results["B+D"])

    # C + D
    results["C+D"] = run_variant(
        "C+D (Trail + Regime)", preds_regime, closes, highs, lows, dates, atr20,
        train_end_idx, test_start_idx, thr_regime, use_trail=True)
    print_row(results["C+D"])

    # ── Triple + quad ──
    print(f"\n{'=' * 130}")
    print(" Triple/Quad combinations")
    print(f"{'=' * 130}")
    print_row({}, header=True)

    results["A+B+C"] = run_variant(
        "A+B+C", preds_base, closes, highs, lows, dates, atr20,
        train_end_idx, test_start_idx, thr_conf, atr_sl=True, use_trail=True)
    print_row(results["A+B+C"])

    # A+B+D + ABCD: use regime conf thresholds
    X_reg_tr = pipe_regime.features_matrix(dates[:train_end_idx-HOLD_DAYS], ctx)
    valid_reg = ~np.isnan(X_reg_tr).any(axis=1)
    pred_reg_in = pipe_regime.predict(X_reg_tr[valid_reg])
    thr_reg_conf = (float(np.quantile(pred_reg_in, 1-CONF_TOP_PCT)),
                    float(np.quantile(pred_reg_in, CONF_BOT_PCT)))

    results["A+B+D"] = run_variant(
        "A+B+D", preds_regime, closes, highs, lows, dates, atr20,
        train_end_idx, test_start_idx, thr_reg_conf, atr_sl=True)
    print_row(results["A+B+D"])

    results["A+C+D"] = run_variant(
        "A+C+D", preds_regime, closes, highs, lows, dates, atr20,
        train_end_idx, test_start_idx, thr_regime, atr_sl=True, use_trail=True)
    print_row(results["A+C+D"])

    results["B+C+D"] = run_variant(
        "B+C+D", preds_regime, closes, highs, lows, dates, atr20,
        train_end_idx, test_start_idx, thr_reg_conf, use_trail=True)
    print_row(results["B+C+D"])

    results["A+B+C+D"] = run_variant(
        "A+B+C+D (all)", preds_regime, closes, highs, lows, dates, atr20,
        train_end_idx, test_start_idx, thr_reg_conf, atr_sl=True, use_trail=True)
    print_row(results["A+B+C+D"])

    # ── Ranking ──
    print(f"\n{'=' * 130}")
    print(" Ranking by annualized PnL (test period)")
    print(f"{'=' * 130}")
    rows = [(k, m) for k, m in results.items() if m.get("n", 0) > 0]
    rows.sort(key=lambda kv: -kv[1]["ann_yuan"])
    for k, m in rows:
        ratio = m["ann_yuan"] / abs(m["max_dd_yuan"]) if m["max_dd_yuan"] < 0 else 999
        print(f"  {m['label']:<22} 年化 {m['ann_yuan']:>+10,.0f}  MaxDD {m['max_dd_yuan']:>+10,.0f}  "
              f"Calmar {ratio:>5.2f}  Sharpe {m['sharpe']:>5.2f}  WR {m['wr']:>4.0f}%  N={m['n']}")

    # ── Variant-D feature importance ──
    print(f"\n=== Variant D feature importance (regime-augmented) ===")
    imp = pipe_regime.feature_importance()
    for _, r in imp.iterrows():
        bar = "█" * int(r["importance"] * 80)
        print(f"  [{r['category']:<6}] {r['feature']:<25} {r['importance']:.4f}  {bar}")


if __name__ == "__main__":
    main()
