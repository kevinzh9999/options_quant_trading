#!/usr/bin/env python3
"""诊断 2025 LONG vs 2024 bull-LONG 的差异

为什么同样的 "bull-regime + hold 10d + SL×4" 在 2024 大赚但 2025 小亏？

诊断维度:
  1. 各年 bull-LONG 入场后 day-by-day cumulative return 曲线 (mean ± std)
  2. 各年 bull-LONG 的 best-day / hold-end day 比较 — 看 MFE 衰减速度
  3. 2025 vs 2024 bull-regime 的 vol_regime / today_amp / RR 分布差异
  4. 2025 bull-LONG trade 的 hold=5,7,10,15,20 best/worst 假设回测
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

DB_PATH = "/Users/kevinzhao/Documents/options_quant_trading/data/storage/trading.db"
HOLD_DAYS = 5
RETRAIN_EVERY = 20
INITIAL_TRAIN_DAYS = 200
TOP_PCT = 0.20
BOT_PCT = 0.20


def build_pipeline_with_regime():
    base = build_default_pipeline()
    base.factors.extend([CloseSma60Factor(), Slope60dFactor(), VolRegimeFactor()])
    base.feature_names = [f.name for f in base.factors]
    return base


def is_bull(idx, regime):
    s60 = regime["close_sma60"][idx]
    s200 = regime["close_sma200"][idx]
    if np.isnan(s60) or np.isnan(s200):
        return False
    return s60 > 1.04 and s200 > 1.05


def collect_bull_long_signals(ctx, dates, closes, atr20, regime):
    """Return list of dicts: each LONG signal in bull regime + 30d forward path."""
    n = len(dates)
    fwd5 = np.array([closes[i+5]/closes[i]-1 if i+5 < n else np.nan for i in range(n)])
    pipe_proto = build_pipeline_with_regime()
    X_full = pipe_proto.features_matrix(dates, ctx)

    cached_pipe = None
    cached_thr = None
    last_train = -RETRAIN_EVERY
    signals = []

    for td_idx in range(INITIAL_TRAIN_DAYS, n - 30):
        if td_idx - last_train >= RETRAIN_EVERY:
            train_end = td_idx - 5
            X_tr = X_full[:train_end]
            y_tr = fwd5[:train_end]
            valid = ~(np.isnan(X_tr).any(axis=1) | np.isnan(y_tr))
            if valid.sum() < INITIAL_TRAIN_DAYS:
                continue
            cached_pipe = build_pipeline_with_regime()
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
        if pred < cached_thr[0]:
            continue
        if not is_bull(td_idx, regime):
            continue
        # Skip filter for SHORT side gate (we're LONG only here)

        entry_close = closes[td_idx]
        path = []
        for offset in range(0, 31):
            idx2 = td_idx + offset
            if idx2 >= n:
                break
            path.append((closes[idx2] - entry_close) / entry_close)

        signals.append({
            "entry_date": dates[td_idx],
            "year": int(dates[td_idx][:4]),
            "entry_close": entry_close,
            "atr20_pct": atr20[td_idx],
            "vol_regime": regime["close_sma60"][td_idx],   # placeholder
            "path": path,
            "pred": pred,
        })

    return signals


def main():
    print("Loading data...")
    ctx = load_default_context(DB_PATH)
    px = ctx.px_df.copy()
    iv_dates = set(ctx.iv_df["trade_date"].tolist())
    px_e = px[px["trade_date"].isin(iv_dates)].reset_index(drop=True)
    dates = px_e["trade_date"].tolist()
    closes = px_e["close"].astype(float).values
    atr20 = compute_atr20(px_e)
    regime = precompute_regime(closes)
    print(f"Range: {dates[0]} ~ {dates[-1]}")

    # 同时把 vol_regime / today_amp 准备好（per-row）
    rets = np.diff(np.log(closes))
    rv5 = np.full(len(closes), np.nan)
    rv60 = np.full(len(closes), np.nan)
    for i in range(60, len(closes)):
        rv5[i] = np.std(rets[i-5:i]) * np.sqrt(252)
        rv60[i] = np.std(rets[i-60:i]) * np.sqrt(252)
    vol_regime_arr = rv5 / np.maximum(rv60, 1e-6)

    print("\nCollecting bull-LONG signals (one pass)...")
    signals = collect_bull_long_signals(ctx, dates, closes, atr20, regime)
    print(f"Total bull-LONG signals: {len(signals)}")

    # 按年分组
    df_sig = pd.DataFrame(signals)
    print(f"\n{'═' * 80}")
    print("=== bull-LONG signal counts by year ===")
    print(f"{'═' * 80}")
    print(df_sig.groupby("year").size())

    # ── Day-by-day return path ──
    print(f"\n{'═' * 80}")
    print("=== Day-by-day mean cumulative return after bull-LONG entry ===")
    print(f"{'═' * 80}")
    print(f"  {'day':>3}", end="")
    for y in [2024, 2025]:
        print(f"  {y}_mean   {y}_med   {y}_pct25  {y}_pct75   N={int((df_sig['year']==y).sum())}", end="")
    print()
    for d in [0, 1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30]:
        print(f"  {d:>3}", end="")
        for y in [2024, 2025]:
            sub = df_sig[df_sig["year"] == y]
            paths = []
            for _, r in sub.iterrows():
                if d < len(r["path"]):
                    paths.append(r["path"][d])
            if not paths:
                print(f"  {' '*8}", end="")
                continue
            arr = np.array(paths)
            print(f"  {arr.mean()*100:>+6.2f}%  {np.median(arr)*100:>+6.2f}%  "
                  f"{np.quantile(arr, 0.25)*100:>+6.2f}%  {np.quantile(arr, 0.75)*100:>+6.2f}%", end="")
        print()

    # ── Hold-N comparison ──
    print(f"\n{'═' * 80}")
    print("=== If exit at day-N: avg return / WR ===")
    print(f"{'═' * 80}")
    print(f"  {'hold':>5}  ", end="")
    for y in [2024, 2025]:
        print(f"{y}_avg%  {y}_med%  {y}_WR    ", end="")
    print()
    for hold in [3, 5, 7, 10, 12, 15, 20, 25, 30]:
        print(f"  {hold:>4}d  ", end="")
        for y in [2024, 2025]:
            sub = df_sig[df_sig["year"] == y]
            rets_n = []
            for _, r in sub.iterrows():
                if hold < len(r["path"]):
                    rets_n.append(r["path"][hold])
            if not rets_n:
                print(f"{' '*22}", end="")
                continue
            arr = np.array(rets_n)
            wr = (arr > 0).mean() * 100
            print(f"{arr.mean()*100:>+5.2f}%  {np.median(arr)*100:>+5.2f}%  {wr:>4.0f}%   ", end="")
        print()

    # ── MFE 比较：trade 持仓中最高点出现在哪天 ──
    print(f"\n{'═' * 80}")
    print("=== Best day in 30-day window (when does MFE peak?) ===")
    print(f"{'═' * 80}")
    for y in [2024, 2025]:
        sub = df_sig[df_sig["year"] == y]
        peak_days = []
        peak_vals = []
        end5 = []
        end10 = []
        for _, r in sub.iterrows():
            path = r["path"]
            if len(path) < 31:
                continue
            best_idx = int(np.argmax(path[1:]) + 1)
            peak_days.append(best_idx)
            peak_vals.append(path[best_idx])
            end5.append(path[5] if len(path) > 5 else np.nan)
            end10.append(path[10] if len(path) > 10 else np.nan)
        if not peak_days:
            continue
        print(f"\n  {y}: N={len(peak_days)}")
        print(f"    Peak day distribution:")
        bins = [(1,3), (4,6), (7,10), (11,15), (16,20), (21,30)]
        for lo, hi in bins:
            n_in = sum(1 for d in peak_days if lo <= d <= hi)
            pct = n_in / len(peak_days) * 100
            bar = "█" * int(pct / 2)
            print(f"      day {lo:>2}-{hi:>2}: {n_in:>3} ({pct:>5.1f}%) {bar}")
        print(f"    Peak return:   mean {np.mean(peak_vals)*100:+.2f}%   "
              f"median {np.median(peak_vals)*100:+.2f}%")
        print(f"    Day-5 return:  mean {np.nanmean(end5)*100:+.2f}%   "
              f"median {np.nanmedian(end5)*100:+.2f}%")
        print(f"    Day-10 return: mean {np.nanmean(end10)*100:+.2f}%   "
              f"median {np.nanmedian(end10)*100:+.2f}%")

    # ── 2024 vs 2025 entry-day market state 差异 ──
    print(f"\n{'═' * 80}")
    print("=== Entry-day market state differences (2024 vs 2025 bull-LONG) ===")
    print(f"{'═' * 80}")
    print(f"  {'metric':<22}  {'2024_mean':>12}  {'2024_med':>12}  "
          f"{'2025_mean':>12}  {'2025_med':>12}")
    df_sig["close_sma60"] = df_sig["entry_date"].apply(
        lambda d: regime["close_sma60"][dates.index(d)])
    df_sig["close_sma200"] = df_sig["entry_date"].apply(
        lambda d: regime["close_sma200"][dates.index(d)])
    df_sig["vol_regime"] = df_sig["entry_date"].apply(
        lambda d: vol_regime_arr[dates.index(d)])
    df_sig["atr20_pct_x100"] = df_sig["atr20_pct"] * 100
    for col in ["close_sma60", "close_sma200", "vol_regime", "atr20_pct_x100", "pred"]:
        v24 = df_sig[df_sig["year"] == 2024][col].dropna()
        v25 = df_sig[df_sig["year"] == 2025][col].dropna()
        if v24.empty or v25.empty:
            continue
        print(f"  {col:<22}  {v24.mean():>+12.4f}  {v24.median():>+12.4f}  "
              f"{v25.mean():>+12.4f}  {v25.median():>+12.4f}")


if __name__ == "__main__":
    main()
