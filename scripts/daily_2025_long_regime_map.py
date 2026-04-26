#!/usr/bin/env python3
"""映射 2025 LONG 真实盈利 regime — 找到 enhancement 应该覆盖的边界

收集所有年份的 LONG signals (regardless of bull gate)，按年份 × regime feature
分桶看 hold=5/10/15 的 PnL 分布。

目标: 找到一个新的 "trend-LONG enhancement gate" 同时覆盖 2024 和 2025 盈利场景。
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
from scripts.daily_walkforward_gates import precompute_regime

DB_PATH = "/Users/kevinzhao/Documents/options_quant_trading/data/storage/trading.db"
RETRAIN_EVERY = 20
INITIAL_TRAIN_DAYS = 200
TOP_PCT = 0.20


def build_pipeline_with_regime():
    base = build_default_pipeline()
    base.factors.extend([CloseSma60Factor(), Slope60dFactor(), VolRegimeFactor()])
    base.feature_names = [f.name for f in base.factors]
    return base


def collect_all_long(ctx, dates, closes, regime, atr20):
    n = len(dates)
    fwd5 = np.array([closes[i+5]/closes[i]-1 if i+5 < n else np.nan for i in range(n)])
    pipe_proto = build_pipeline_with_regime()
    X_full = pipe_proto.features_matrix(dates, ctx)
    cached_pipe = None
    cached_thr = None
    last_train = -RETRAIN_EVERY
    rows = []
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
            cached_thr = float(np.quantile(pred_in, 1 - TOP_PCT))
            last_train = td_idx
        if cached_pipe is None:
            continue
        x = X_full[td_idx:td_idx+1]
        if np.isnan(x).any():
            continue
        pred = float(cached_pipe.predict(x)[0])
        if pred < cached_thr:
            continue
        ec = closes[td_idx]
        path = []
        for offset in range(0, 21):
            if td_idx + offset >= n:
                break
            path.append((closes[td_idx+offset] - ec) / ec)
        if len(path) < 16:
            continue
        rows.append({
            "entry_date": dates[td_idx],
            "year": int(dates[td_idx][:4]),
            "close_sma60": regime["close_sma60"][td_idx],
            "close_sma200": regime["close_sma200"][td_idx],
            "atr20": atr20[td_idx],
            "ret5": path[5],
            "ret10": path[10] if len(path) > 10 else np.nan,
            "ret15": path[15] if len(path) > 15 else np.nan,
            "ret20": path[20] if len(path) > 20 else np.nan,
            "mfe5": max(path[1:6]) if len(path) > 5 else np.nan,
            "mfe10": max(path[1:11]) if len(path) > 10 else np.nan,
            "mfe15": max(path[1:16]) if len(path) > 15 else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    ctx = load_default_context(DB_PATH)
    px = ctx.px_df.copy()
    iv_dates = set(ctx.iv_df["trade_date"].tolist())
    px_e = px[px["trade_date"].isin(iv_dates)].reset_index(drop=True)
    dates = px_e["trade_date"].tolist()
    closes = px_e["close"].astype(float).values
    atr20 = compute_atr20(px_e)
    regime = precompute_regime(closes)
    print(f"Range: {dates[0]} ~ {dates[-1]}")

    print("\nCollecting all LONG signals...")
    df = collect_all_long(ctx, dates, closes, regime, atr20)
    print(f"Total LONG signals: {len(df)}")
    print(f"\nBy year: {df.groupby('year').size().to_dict()}")

    # 按 close/sma60 × close/sma200 二维分桶看每年表现
    print(f"\n{'═' * 100}")
    print("=== 2024 vs 2025 LONG by (close/sma60, close/sma200) bucket ===")
    print(f"{'═' * 100}")

    df["s60_bucket"] = pd.cut(df["close_sma60"],
        bins=[0, 0.97, 1.00, 1.02, 1.04, 1.06, 1.10, 99],
        labels=["<0.97", "0.97-1.00", "1.00-1.02", "1.02-1.04", "1.04-1.06", "1.06-1.10", ">1.10"])
    df["s200_bucket"] = pd.cut(df["close_sma200"],
        bins=[0, 0.95, 1.00, 1.03, 1.06, 1.10, 99],
        labels=["<0.95", "0.95-1.00", "1.00-1.03", "1.03-1.06", "1.06-1.10", ">1.10"])

    for y in [2024, 2025]:
        sub = df[df["year"] == y]
        print(f"\n  ── {y} (N={len(sub)}) ── ret5 mean by bucket")
        pivot = sub.pivot_table(values="ret5", index="s60_bucket",
                                 columns="s200_bucket", aggfunc="mean", observed=True)
        print((pivot * 100).round(2).fillna(""))
        print(f"\n  ── {y} count by bucket")
        cnt = sub.pivot_table(values="ret5", index="s60_bucket",
                                columns="s200_bucket", aggfunc="count", observed=True)
        print(cnt.fillna(0).astype(int))

    # ── 2025 specifically — 各种 hold horizon 在 close/sma60 桶内 ──
    print(f"\n{'═' * 100}")
    print("=== 2025 LONG by close/sma60 bucket — hold horizon comparison ===")
    print(f"{'═' * 100}")
    print(f"  {'bucket':<14} {'N':>3}  {'ret5':>7}  {'ret10':>7}  {'ret15':>7}  "
          f"{'mfe5':>7}  {'mfe10':>7}  {'mfe15':>7}")
    sub_2025 = df[df["year"] == 2025]
    for bucket, grp in sub_2025.groupby("s60_bucket", observed=True):
        if len(grp) == 0:
            continue
        print(f"  {str(bucket):<14} {len(grp):>3}  "
              f"{grp['ret5'].mean()*100:>+6.2f}%  "
              f"{grp['ret10'].mean()*100:>+6.2f}%  "
              f"{grp['ret15'].mean()*100:>+6.2f}%  "
              f"{grp['mfe5'].mean()*100:>+6.2f}%  "
              f"{grp['mfe10'].mean()*100:>+6.2f}%  "
              f"{grp['mfe15'].mean()*100:>+6.2f}%")

    # ── 2024 同样看 ──
    print(f"\n  ── 2024 同 bucket 对照 ──")
    print(f"  {'bucket':<14} {'N':>3}  {'ret5':>7}  {'ret10':>7}  {'ret15':>7}  "
          f"{'mfe5':>7}  {'mfe10':>7}  {'mfe15':>7}")
    sub_2024 = df[df["year"] == 2024]
    for bucket, grp in sub_2024.groupby("s60_bucket", observed=True):
        if len(grp) == 0:
            continue
        print(f"  {str(bucket):<14} {len(grp):>3}  "
              f"{grp['ret5'].mean()*100:>+6.2f}%  "
              f"{grp['ret10'].mean()*100:>+6.2f}%  "
              f"{grp['ret15'].mean()*100:>+6.2f}%  "
              f"{grp['mfe5'].mean()*100:>+6.2f}%  "
              f"{grp['mfe10'].mean()*100:>+6.2f}%  "
              f"{grp['mfe15'].mean()*100:>+6.2f}%")

    # ── 寻找 2025 LONG 的真正盈利 regime ──
    print(f"\n{'═' * 100}")
    print("=== 2025 LONG: 哪个 (s60, s200) 组合是 day-10 mean 最强的? ===")
    print(f"{'═' * 100}")
    sub25 = df[df["year"] == 2025].copy()
    print(f"\nTop signals by ret10:")
    print(sub25.nlargest(10, "ret10")[["entry_date", "close_sma60", "close_sma200",
                                          "ret5", "ret10", "ret15", "mfe10"]].round(4).to_string())
    print(f"\nBottom signals by ret10:")
    print(sub25.nsmallest(10, "ret10")[["entry_date", "close_sma60", "close_sma200",
                                            "ret5", "ret10", "ret15", "mfe10"]].round(4).to_string())

    # ── 候选新 gate sweep ──
    print(f"\n{'═' * 100}")
    print("=== 候选放宽的 trend-LONG gate ===")
    print(f"{'═' * 100}")
    print(f"对于每个候选 gate，看它捕获多少 LONG 信号 + 这些信号的 ret5/10/15")
    print(f"  {'Gate':<46} {'2024_N':>7} {'2024_avg10':>12} "
          f"{'2025_N':>7} {'2025_avg10':>12} {'2025_avg15':>12}")
    candidates = [
        ("close/sma60 > 1.00", lambda r: r.close_sma60 > 1.00),
        ("close/sma60 > 1.02", lambda r: r.close_sma60 > 1.02),
        ("close/sma60 > 1.04", lambda r: r.close_sma60 > 1.04),
        ("close/sma200 > 1.00", lambda r: r.close_sma200 > 1.00),
        ("close/sma200 > 1.03", lambda r: r.close_sma200 > 1.03),
        ("close/sma200 > 1.05", lambda r: r.close_sma200 > 1.05),
        ("close/sma60>1.00 & close/sma200>1.00", lambda r: r.close_sma60 > 1.00 and r.close_sma200 > 1.00),
        ("close/sma60>1.02 & close/sma200>1.03", lambda r: r.close_sma60 > 1.02 and r.close_sma200 > 1.03),
        ("close/sma60>1.04 & close/sma200>1.05", lambda r: r.close_sma60 > 1.04 and r.close_sma200 > 1.05),
        ("close/sma60>1.00 & close/sma200>1.05", lambda r: r.close_sma60 > 1.00 and r.close_sma200 > 1.05),
        ("close/sma60>1.02 & close/sma200>1.05", lambda r: r.close_sma60 > 1.02 and r.close_sma200 > 1.05),
    ]
    for label, fn in candidates:
        sub24 = df[df["year"] == 2024]
        sub25 = df[df["year"] == 2025]
        m24 = sub24.apply(fn, axis=1)
        m25 = sub25.apply(fn, axis=1)
        n24 = m24.sum()
        n25 = m25.sum()
        a24 = sub24[m24]["ret10"].mean() * 100 if n24 > 0 else 0
        a25 = sub25[m25]["ret10"].mean() * 100 if n25 > 0 else 0
        a25_15 = sub25[m25]["ret15"].mean() * 100 if n25 > 0 else 0
        print(f"  {label:<46} {n24:>7} {a24:>+10.2f}%   "
              f"{n25:>7} {a25:>+10.2f}%   {a25_15:>+10.2f}%")


if __name__ == "__main__":
    main()
