#!/usr/bin/env python3
"""诊断 2025 牛市 SHORT 信号亏损的特征模式

从 walk-forward A+D 输出读 trades，看：
  1. 2025 全年 SHORT 笔数 / 亏损额 / WR
  2. SHORT 触发时 regime feature 分布 (slope_60d, close_sma60, vol_regime)
  3. 同期 LONG 笔数 / 盈利
  4. 如果加 hard gate "uptrend 时禁 SHORT"，过滤掉多少笔，省多少钱
  5. 对其他年份的影响 (2023/2024 同 gate 下是否误杀)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from strategies.daily.factors import load_default_context, build_default_pipeline
from scripts.daily_robust_methods_compare import (
    CloseSma60Factor, Slope60dFactor, VolRegimeFactor,
)

DB_PATH = "/Users/kevinzhao/Documents/options_quant_trading/data/storage/trading.db"
TRADES_CSV = "/tmp/daily_walkforward_AD_trades.csv"


def main():
    df = pd.read_csv(TRADES_CSV, dtype={"entry_date": str, "exit_date": str})
    df["entry_date"] = df["entry_date"].astype(str).str.zfill(8)
    df["year"] = pd.to_datetime(df["entry_date"]).dt.year

    print(f"=== 全样本 trades by year × direction ===")
    pivot = df.groupby(["year", "direction"]).agg(
        n=("pnl_yuan", "count"),
        pnl=("pnl_yuan", "sum"),
        wr=("net_ret", lambda x: (x > 0).mean() * 100),
        avg=("net_ret", lambda x: x.mean() * 100),
        worst=("net_ret", lambda x: x.min() * 100),
    ).round(2)
    print(pivot.to_string())

    # ── 加 regime 特征到每笔 trade ──
    print(f"\n=== 计算每笔 trade 入场日的 regime 特征 ===")
    ctx = load_default_context(DB_PATH)
    sma_f, slope_f, vol_f = CloseSma60Factor(), Slope60dFactor(), VolRegimeFactor()

    df["close_sma60"] = df["entry_date"].apply(lambda d: sma_f.compute(d, ctx))
    df["slope_60d"] = df["entry_date"].apply(lambda d: slope_f.compute(d, ctx))
    df["vol_regime"] = df["entry_date"].apply(lambda d: vol_f.compute(d, ctx))

    # ── 2025 SHORT vs LONG ──
    df_2025 = df[df["year"] == 2025]
    df_2025_short = df_2025[df_2025["direction"] == "SHORT"]
    df_2025_long = df_2025[df_2025["direction"] == "LONG"]
    print(f"\n=== 2025 SHORT 信号 (n={len(df_2025_short)}) ===")
    print(f"  Total PnL: {df_2025_short['pnl_yuan'].sum():+,.0f}")
    print(f"  WR: {(df_2025_short['net_ret'] > 0).mean() * 100:.1f}%")
    print(f"  slope_60d 分布:    mean {df_2025_short['slope_60d'].mean():+.5f}  "
          f"median {df_2025_short['slope_60d'].median():+.5f}")
    print(f"  close/sma60 分布:  mean {df_2025_short['close_sma60'].mean():.4f}  "
          f"median {df_2025_short['close_sma60'].median():.4f}")

    print(f"\n=== 2025 LONG 信号 (n={len(df_2025_long)}) ===")
    print(f"  Total PnL: {df_2025_long['pnl_yuan'].sum():+,.0f}")
    print(f"  WR: {(df_2025_long['net_ret'] > 0).mean() * 100:.1f}%")

    # ── 按 slope_60d 分桶看 SHORT 表现（全样本）──
    print(f"\n=== 全样本 SHORT 表现 by slope_60d 分桶 ===")
    df_short = df[df["direction"] == "SHORT"].dropna(subset=["slope_60d"]).copy()
    df_short["slope_bucket"] = pd.qcut(df_short["slope_60d"], 5, labels=False, duplicates="drop")
    bucket_summary = df_short.groupby("slope_bucket").agg(
        n=("pnl_yuan", "count"),
        slope_mean=("slope_60d", "mean"),
        pnl=("pnl_yuan", "sum"),
        wr=("net_ret", lambda x: (x > 0).mean() * 100),
        avg=("net_ret", lambda x: x.mean() * 100),
    ).round(2)
    print(f"  bucket  n   slope_mean    PnL      WR    avg%")
    for b, r in bucket_summary.iterrows():
        print(f"  Q{int(b)+1}     {int(r['n']):>3}  {r['slope_mean']:>+.5f}  "
              f"{r['pnl']:>+10,.0f}  {r['wr']:>4.0f}%  {r['avg']:>+5.2f}%")

    # ── 同样看 LONG ──
    print(f"\n=== 全样本 LONG 表现 by slope_60d 分桶 ===")
    df_long = df[df["direction"] == "LONG"].dropna(subset=["slope_60d"]).copy()
    df_long["slope_bucket"] = pd.qcut(df_long["slope_60d"], 5, labels=False, duplicates="drop")
    bucket_summary_l = df_long.groupby("slope_bucket").agg(
        n=("pnl_yuan", "count"),
        slope_mean=("slope_60d", "mean"),
        pnl=("pnl_yuan", "sum"),
        wr=("net_ret", lambda x: (x > 0).mean() * 100),
        avg=("net_ret", lambda x: x.mean() * 100),
    ).round(2)
    print(f"  bucket  n   slope_mean    PnL      WR    avg%")
    for b, r in bucket_summary_l.iterrows():
        print(f"  Q{int(b)+1}     {int(r['n']):>3}  {r['slope_mean']:>+.5f}  "
              f"{r['pnl']:>+10,.0f}  {r['wr']:>4.0f}%  {r['avg']:>+5.2f}%")

    # ── Hard gate sweep ──
    print(f"\n=== Hard gate sweep: 禁 SHORT when slope_60d > X ===")
    print(f"  全样本影响 (across years):")
    print(f"  {'gate':>10}  {'n_blocked':>9}  {'blocked_pnl':>12}  {'remaining_short_pnl':>20}  "
          f"{'2025_blocked':>13}  {'2025_saved':>11}  {'2024_blocked':>13}  {'2024_lost':>11}")
    for gate in [0.0, 0.0002, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003]:
        blocked = df_short[df_short["slope_60d"] > gate]
        kept = df_short[df_short["slope_60d"] <= gate]
        b25 = blocked[blocked["year"] == 2025]
        b24 = blocked[blocked["year"] == 2024]
        print(f"  {gate:>+.5f}  {len(blocked):>9}  {blocked['pnl_yuan'].sum():>+12,.0f}  "
              f"{kept['pnl_yuan'].sum():>+20,.0f}  "
              f"{len(b25):>13}  {-b25['pnl_yuan'].sum():>+11,.0f}  "
              f"{len(b24):>13}  {-b24['pnl_yuan'].sum():>+11,.0f}")
    print(f"  (saved/lost = pnl flipped sign — 正数=避免亏损, 负数=误杀盈利)")

    # ── 同样看 LONG 在下跌段的表现 (slope_60d < X) ──
    print(f"\n=== Hard gate sweep: 禁 LONG when slope_60d < X ===")
    print(f"  {'gate':>10}  {'n_blocked':>9}  {'blocked_pnl':>12}")
    for gate in [-0.003, -0.002, -0.0015, -0.001, -0.0005, -0.0002, 0.0]:
        blocked = df_long[df_long["slope_60d"] < gate]
        print(f"  {gate:>+.5f}  {len(blocked):>9}  {blocked['pnl_yuan'].sum():>+12,.0f}")

    # ── close/sma60 同样看 ──
    print(f"\n=== Hard gate sweep: 禁 SHORT when close/sma60 > X ===")
    print(f"  {'gate':>8}  {'n_blocked':>9}  {'blocked_pnl':>12}  "
          f"{'2025_blocked':>13}  {'2025_saved':>11}")
    for gate in [1.00, 1.02, 1.04, 1.06, 1.08, 1.10]:
        blocked = df_short[df_short["close_sma60"] > gate]
        b25 = blocked[blocked["year"] == 2025]
        print(f"  {gate:>.4f}  {len(blocked):>9}  {blocked['pnl_yuan'].sum():>+12,.0f}  "
              f"{len(b25):>13}  {-b25['pnl_yuan'].sum():>+11,.0f}")


if __name__ == "__main__":
    main()
