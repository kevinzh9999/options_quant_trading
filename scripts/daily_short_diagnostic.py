#!/usr/bin/env python3
"""诊断 test 期 SHORT 信号 — 模型为什么在牛市出 SHORT？

研究 trade-by-trade:
  - 这些 SHORT 信号触发时的 feature 分布
  - 当时的 trend regime (close vs 60d/200d MA)
  - 同期 LONG 信号 vs SHORT 信号 feature 对比
  - 重点看 SHORT-亏损 笔的 features，找模型"上当"的 pattern
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from strategies.daily.factors import (
    build_default_pipeline, load_default_context,
)

DB_PATH = "/Users/kevinzhao/Documents/options_quant_trading/data/storage/trading.db"
HOLD_DAYS = 5
TOP_PCT = 0.20
BOT_PCT = 0.20
TRAIN_END = "20241231"


def main():
    ctx = load_default_context(DB_PATH)
    px = ctx.px_df.copy()
    iv_dates = set(ctx.iv_df["trade_date"].tolist())
    px_e = px[px["trade_date"].isin(iv_dates)].reset_index(drop=True)
    dates = px_e["trade_date"].tolist()
    closes = px_e["close"].astype(float).values

    train_end_idx = next((i for i, d in enumerate(dates) if d > TRAIN_END), len(dates))

    # 计算 trend indicators (long-term context)
    sma60 = pd.Series(closes).rolling(60).mean().values
    sma200 = pd.Series(closes).rolling(200).mean().values
    # 60d slope
    def _slope_60(arr):
        if len(arr) < 60 or np.isnan(arr).any():
            return np.nan
        return np.polyfit(range(60), arr, 1)[0]
    slope60 = pd.Series(closes).rolling(60).apply(_slope_60, raw=True).values
    px_pct_above_sma60 = (closes - sma60) / sma60  # close 相对 60d MA

    # Train pipeline
    pipeline = build_default_pipeline()
    X_full = pipeline.features_matrix(dates, ctx)
    fwd_ret = np.array([
        closes[i + HOLD_DAYS] / closes[i] - 1 if i + HOLD_DAYS < len(closes) else np.nan
        for i in range(len(closes))
    ])
    train_cutoff = train_end_idx - HOLD_DAYS
    valid = ~(np.isnan(X_full[:train_cutoff]).any(axis=1) | np.isnan(fwd_ret[:train_cutoff]))
    pipeline.train(X_full[:train_cutoff][valid], fwd_ret[:train_cutoff][valid])
    pred_tr = pipeline.predict(X_full[:train_cutoff][valid])
    top_thr = float(np.quantile(pred_tr, 1 - TOP_PCT))
    bot_thr = float(np.quantile(pred_tr, BOT_PCT))

    predictions = np.full(len(dates), np.nan)
    valid_full = ~np.isnan(X_full).any(axis=1)
    predictions[valid_full] = pipeline.predict(X_full[valid_full])

    # ── 收集 test trades ──
    rows = []
    for i in range(train_end_idx, len(dates) - HOLD_DAYS):
        if np.isnan(predictions[i]):
            continue
        if predictions[i] >= top_thr:
            direction = "LONG"
        elif predictions[i] <= bot_thr:
            direction = "SHORT"
        else:
            continue
        gross_ret = (closes[i + HOLD_DAYS] - closes[i]) / closes[i]
        if direction == "SHORT":
            gross_ret = -gross_ret
        # Trend regime
        rows.append({
            "date": dates[i],
            "direction": direction,
            "pred": predictions[i],
            "gross_ret": gross_ret,
            "win": gross_ret > 0,
            "close": closes[i],
            "sma60": sma60[i],
            "sma200": sma200[i],
            "above_sma60_pct": px_pct_above_sma60[i],
            "slope60_per_day": slope60[i] / closes[i] if not np.isnan(slope60[i]) and closes[i] > 0 else np.nan,
            # 触发时的 features
            **{name: X_full[i, k] for k, name in enumerate(pipeline.feature_names)},
        })
    df = pd.DataFrame(rows)
    print(f"Test trades: {len(df)}")

    # ── 1. 牛市定义 ──
    # 用 close 相对 sma60 + slope60 判定 regime
    print(f"\n{'=' * 75}")
    print(" 1. 牛市/熊市/震荡 regime 划分（test 期）")
    print(f"{'=' * 75}")
    df["regime"] = "neutral"
    df.loc[(df["above_sma60_pct"] > 0.05) & (df["slope60_per_day"] > 0.001), "regime"] = "bull"
    df.loc[(df["above_sma60_pct"] < -0.05) & (df["slope60_per_day"] < -0.001), "regime"] = "bear"
    print("\n  规则:")
    print("    bull: close > SMA60 × 1.05 AND slope60 > +0.1%/day")
    print("    bear: close < SMA60 × 0.95 AND slope60 < -0.1%/day")
    print("    其它: neutral")
    print(f"\n  Test 期 regime 分布:")
    for r in ["bull", "neutral", "bear"]:
        sub = df[df["regime"] == r]
        if len(sub) > 0:
            print(f"    {r:<8}: {len(sub):>3} 天 ({len(sub)/len(df)*100:.0f}%)")

    # ── 2. SHORT 信号 vs trend regime cross-tab ──
    print(f"\n{'=' * 75}")
    print(" 2. SHORT/LONG 信号 vs regime 交叉")
    print(f"{'=' * 75}")
    print(f"\n  {'regime':<10}{'direction':<10}{'N':>4}{'WR':>6}{'avg gross':>10}")
    for reg in ["bull", "neutral", "bear"]:
        for d in ["LONG", "SHORT"]:
            sub = df[(df["regime"] == reg) & (df["direction"] == d)]
            if len(sub) == 0:
                continue
            wr = (sub["win"]).mean() * 100
            avg = sub["gross_ret"].mean() * 100
            tag = " ⚠️ 逆势" if (reg == "bull" and d == "SHORT") or (reg == "bear" and d == "LONG") else ""
            print(f"  {reg:<10}{d:<10}{len(sub):>4}{wr:>5.0f}%{avg:>+9.2f}%{tag}")

    # ── 3. 牛市里 SHORT 信号特征 vs 牛市里 LONG 特征对比 ──
    print(f"\n{'=' * 75}")
    print(" 3. 牛市里 SHORT 信号 vs LONG 信号特征对比（找模型'上当'的 pattern）")
    print(f"{'=' * 75}")
    bull_short = df[(df["regime"] == "bull") & (df["direction"] == "SHORT")]
    bull_long = df[(df["regime"] == "bull") & (df["direction"] == "LONG")]
    print(f"\n  Bull regime:  SHORT={len(bull_short)} (WR={bull_short['win'].mean()*100:.0f}%), "
          f"LONG={len(bull_long)} (WR={bull_long['win'].mean()*100:.0f}%)")
    if len(bull_short) > 5 and len(bull_long) > 5:
        feat_cols = pipeline.feature_names
        print(f"\n  特征均值差（SHORT - LONG，绝对值大 = 区分度强）:")
        diffs = []
        for f in feat_cols:
            if f in bull_short.columns and f in bull_long.columns:
                ms = bull_short[f].mean()
                ml = bull_long[f].mean()
                diff = ms - ml
                diffs.append((f, ms, ml, diff))
        diffs.sort(key=lambda x: abs(x[3]), reverse=True)
        print(f"  {'feature':<25}{'SHORT mean':>12}{'LONG mean':>12}{'diff':>10}")
        for f, ms, ml, diff in diffs[:12]:
            print(f"  {f:<25}{ms:>+12.4f}{ml:>+12.4f}{diff:>+10.4f}")

    # ── 4. 牛市里 SHORT 笔的 win/loss 对比 ──
    print(f"\n{'=' * 75}")
    print(" 4. 牛市 SHORT 信号 — 赢 vs 输 的特征差异")
    print(f"{'=' * 75}")
    if len(bull_short) > 10:
        win_sub = bull_short[bull_short["win"]]
        lose_sub = bull_short[~bull_short["win"]]
        print(f"  Win N={len(win_sub)}  Lose N={len(lose_sub)}")
        if len(win_sub) >= 3 and len(lose_sub) >= 3:
            diffs = []
            for f in pipeline.feature_names:
                if f in win_sub.columns and f in lose_sub.columns:
                    mw = win_sub[f].mean()
                    ml = lose_sub[f].mean()
                    diffs.append((f, mw, ml, mw - ml))
            diffs.sort(key=lambda x: abs(x[3]), reverse=True)
            print(f"  {'feature':<25}{'WIN':>12}{'LOSE':>12}{'diff':>10}")
            for f, mw, ml, d_ in diffs[:10]:
                print(f"  {f:<25}{mw:>+12.4f}{ml:>+12.4f}{d_:>+10.4f}")

    # ── 5. 简单 trend filter 的 effect 估算 ──
    print(f"\n{'=' * 75}")
    print(" 5. 假设 'bull regime 禁 SHORT、bear regime 禁 LONG' 的 effect")
    print(f"{'=' * 75}")
    df["filtered"] = False
    df.loc[(df["regime"] == "bull") & (df["direction"] == "SHORT"), "filtered"] = True
    df.loc[(df["regime"] == "bear") & (df["direction"] == "LONG"), "filtered"] = True
    n_filtered = df["filtered"].sum()
    kept = df[~df["filtered"]]
    print(f"\n  原 trades: {len(df)}")
    print(f"  逆势被过滤: {n_filtered}")
    print(f"  保留: {len(kept)}  WR={kept['win'].mean()*100:.0f}%")
    print(f"  原 total gross PnL: {df['gross_ret'].sum()*100:.1f}%")
    print(f"  过滤后 total: {kept['gross_ret'].sum()*100:.1f}%")
    print(f"  仅看被过滤的逆势 trades: {df[df['filtered']]['gross_ret'].sum()*100:.1f}%")

    # ── 6. 假设 stop-loss -2% 的 effect ──
    # 现在 backtest 是 5d 全持，无 intraperiod stop。
    # 实际逐日 PnL 不可得（need close-to-close 5d 序列）。
    # 这里粗略估算：如果 5d 内任一日 cumulative gross 跌破 -2%，假设那时止损
    print(f"\n{'=' * 75}")
    print(" 6. 假设 stop-loss -2% 的 effect（粗略估算）")
    print(f"{'=' * 75}")
    rows_sl = []
    for _, r in df.iterrows():
        d_idx = dates.index(r["date"])
        path_closes = closes[d_idx: d_idx + HOLD_DAYS + 1]
        if len(path_closes) < 2:
            continue
        # daily ret series
        rets = (path_closes[1:] - path_closes[0]) / path_closes[0]
        if r["direction"] == "SHORT":
            rets = -rets
        # 若 cumret 在某日 < -2%，止损在那一日 close
        stop_idx = None
        for j, ret in enumerate(rets):
            if ret < -0.02:
                stop_idx = j
                break
        if stop_idx is not None:
            new_ret = rets[stop_idx]  # 止损在那日，假设 close 价
            stopped = True
        else:
            new_ret = rets[-1]
            stopped = False
        rows_sl.append({
            **r.to_dict(),
            "stopped": stopped,
            "stop_day": stop_idx if stop_idx is not None else HOLD_DAYS,
            "ret_with_sl": new_ret,
        })
    df_sl = pd.DataFrame(rows_sl)
    n_stopped = df_sl["stopped"].sum()
    print(f"  止损触发: {n_stopped}/{len(df_sl)} ({n_stopped/len(df_sl)*100:.0f}%)")
    print(f"  原 avg gross: {df_sl['gross_ret'].mean()*100:+.3f}%")
    print(f"  with SL avg: {df_sl['ret_with_sl'].mean()*100:+.3f}%")
    print(f"  原 total: {df_sl['gross_ret'].sum()*100:+.1f}%")
    print(f"  with SL total: {df_sl['ret_with_sl'].sum()*100:+.1f}%")

    # 极端亏损改善
    big_loss_orig = (df_sl["gross_ret"] < -0.03).sum()
    big_loss_sl = (df_sl["ret_with_sl"] < -0.03).sum()
    print(f"\n  原 < -3% 大亏笔数: {big_loss_orig}")
    print(f"  with SL < -3% 大亏: {big_loss_sl} (砍掉 {big_loss_orig-big_loss_sl} 笔)")

    # ── 7. 联合 regime filter + SL ──
    print(f"\n  联合 (regime filter + SL): ")
    df_both = df_sl[~df_sl["filtered"]].copy()
    print(f"    保留 trades: {len(df_both)}")
    print(f"    Total gross: {df_both['ret_with_sl'].sum()*100:+.1f}%")
    print(f"    Avg/笔: {df_both['ret_with_sl'].mean()*100:+.3f}%")
    print(f"    WR: {(df_both['ret_with_sl'] > 0).mean() * 100:.0f}%")


if __name__ == "__main__":
    main()
