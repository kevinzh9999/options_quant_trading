#!/usr/bin/env python3
"""诊断 2023 + 2026 弱表现的根因

问题:
  N5 walk-forward 结果中 2023 +40K, 2026 +134K，远弱于 2024/2025。

诊断维度:
  1. 各年市场 regime: realized return / vol / close_sma60 / close_sma200 分布
  2. 各年信号特点: pred 强度分布、IC by year、trades count
  3. 各月 PnL breakdown
  4. 2023 是不是因为初始训练样本不够？看 train_ic by retrain
  5. 2023 市场是否真的"不可交易"(均值回归/横盘)，还是模型不行？
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
from scripts.daily_walkforward_long_optim import simulate_trade_dynamic

DB_PATH = "/Users/kevinzhao/Documents/options_quant_trading/data/storage/trading.db"
RETRAIN_EVERY = 20
INITIAL_TRAIN_DAYS = 200
TOP_PCT = 0.20
BOT_PCT = 0.20


def build_pipe():
    base = build_default_pipeline()
    base.factors.extend([CloseSma60Factor(), Slope60dFactor(), VolRegimeFactor()])
    base.feature_names = [f.name for f in base.factors]
    return base


def main():
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
    n = len(dates)

    # ── 1. 各年市场 regime 描述 ──
    df_px = pd.DataFrame({
        "date": dates, "close": closes,
        "close_sma60": regime["close_sma60"],
        "close_sma200": regime["close_sma200"],
        "atr20_pct": atr20 * 100,
    })
    df_px["year"] = df_px["date"].str[:4].astype(int)
    df_px["fwd5_ret"] = pd.Series(closes).pct_change(5).shift(-5)
    df_px["abs_fwd5"] = df_px["fwd5_ret"].abs()
    print(f"{'═' * 75}")
    print(" 各年市场 regime")
    print(f"{'═' * 75}")
    print(f"  {'year':<6} {'N_days':>7} {'YTD_return':>11} {'mean_atr':>9} "
          f"{'mean(c/sma60)':>14} {'mean(c/sma200)':>15} {'days_bull':>10} "
          f"{'days_bear':>10} {'std_fwd5':>10}")
    for y in [2023, 2024, 2025, 2026]:
        sub = df_px[df_px["year"] == y]
        if sub.empty:
            continue
        ytd = (sub["close"].iloc[-1] / sub["close"].iloc[0] - 1) * 100
        atr_mean = sub["atr20_pct"].mean()
        s60_mean = sub["close_sma60"].mean()
        s200_mean = sub["close_sma200"].mean()
        days_bull = ((sub["close_sma60"] > 1.04) & (sub["close_sma200"] > 1.05)).sum()
        days_bear = ((sub["close_sma60"] < 0.96) & (sub["close_sma200"] < 0.95)).sum()
        std_f5 = sub["fwd5_ret"].std() * 100
        print(f"  {y:<6} {len(sub):>7} {ytd:>+10.1f}% {atr_mean:>+8.2f}% "
              f"{s60_mean:>+13.4f} {s200_mean:>+14.4f} {days_bull:>10} "
              f"{days_bear:>10} {std_f5:>+9.2f}%")

    # ── 2. 完整 walk-forward 跑一遍记录 train_ic / trades 信息 ──
    print(f"\n{'═' * 75}")
    print(" Walk-forward train IC by retrain (是否随时间稳定?)")
    print(f"{'═' * 75}")
    fwd5 = np.array([closes[i+5]/closes[i]-1 if i+5 < n else np.nan for i in range(n)])
    pipe_proto = build_pipe()
    X_full = pipe_proto.features_matrix(dates, ctx)

    cached_pipe = None
    cached_thr = None
    last_train = -RETRAIN_EVERY
    trades = []
    train_log = []

    for td_idx in range(INITIAL_TRAIN_DAYS, n - 16):
        if td_idx - last_train >= RETRAIN_EVERY:
            train_end = td_idx - 5
            X_tr = X_full[:train_end]
            y_tr = fwd5[:train_end]
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
            ic_tr = np.corrcoef(pred_in, y_tr[valid])[0, 1]
            train_log.append({
                "td": dates[td_idx], "n_train": valid.sum(),
                "ic_tr": ic_tr,
                "top_thr": cached_thr[0], "bot_thr": cached_thr[1],
            })
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

        hold, atr_k = 5, 1.5
        if direction == "LONG":
            enh = enh_two_branch_v2(td_idx, regime)
            if enh is not None:
                hold, atr_k = enh

        sl = atr_k * atr20[td_idx] if not np.isnan(atr20[td_idx]) else None
        if td_idx + hold >= n:
            continue
        exit_idx, gross_ret, reason = simulate_trade_dynamic(
            direction, td_idx, closes, highs, lows,
            hold_days=hold, sl_pct=sl,
        )
        net_ret = gross_ret - 0.0008
        trades.append({
            "entry_date": dates[td_idx],
            "year": int(dates[td_idx][:4]),
            "month": dates[td_idx][:6],
            "direction": direction, "pred": pred,
            "hold_used": hold, "atr_k_used": atr_k,
            "net_ret": net_ret,
            "pnl_yuan": net_ret * closes[td_idx] * 200,
            "reason": reason,
            "in_strict_bull": (regime["close_sma60"][td_idx] > 1.04 and
                                regime["close_sma200"][td_idx] > 1.05),
            "in_dip_bull": (regime["close_sma200"][td_idx] > 1.03 and
                             regime["close_sma60"][td_idx] < 1.02),
        })

    df_t = pd.DataFrame(trades)
    df_log = pd.DataFrame(train_log)

    # Train IC by year
    df_log["year"] = df_log["td"].str[:4].astype(int)
    print(f"  {'year':<6} {'retrains':>9} {'mean_ic':>9} {'std_ic':>9} "
          f"{'min_ic':>9} {'max_ic':>9}")
    for y in [2023, 2024, 2025, 2026]:
        sub = df_log[df_log["year"] == y]
        if sub.empty:
            continue
        print(f"  {y:<6} {len(sub):>9} {sub['ic_tr'].mean():>+9.4f} "
              f"{sub['ic_tr'].std():>+9.4f} {sub['ic_tr'].min():>+9.4f} "
              f"{sub['ic_tr'].max():>+9.4f}")

    # ── 3. 月度 PnL 详细 (2023 + 2026) ──
    print(f"\n{'═' * 75}")
    print(" 2023 月度详细 (核心问题: 为什么 +40K 这么少?)")
    print(f"{'═' * 75}")
    print(f"  {'month':<8} {'N':>3} {'WR':>5} {'avg_ret':>9} {'PnL¥':>10} "
          f"{'long':>4} {'short':>5} {'enh_n':>6}")
    for m, sub in df_t[df_t["year"] == 2023].groupby("month"):
        wr = (sub["net_ret"] > 0).mean() * 100 if len(sub) else 0
        long_n = (sub["direction"] == "LONG").sum()
        short_n = (sub["direction"] == "SHORT").sum()
        enh_n = (sub["hold_used"] != 5).sum()
        print(f"  {m:<8} {len(sub):>3} {wr:>4.0f}% {sub['net_ret'].mean()*100:>+8.3f}% "
              f"{sub['pnl_yuan'].sum():>+10,.0f} {long_n:>4} {short_n:>5} {enh_n:>6}")

    print(f"\n=== 2026 月度详细 ===")
    print(f"  {'month':<8} {'N':>3} {'WR':>5} {'avg_ret':>9} {'PnL¥':>10} "
          f"{'long':>4} {'short':>5} {'enh_n':>6}")
    for m, sub in df_t[df_t["year"] == 2026].groupby("month"):
        wr = (sub["net_ret"] > 0).mean() * 100 if len(sub) else 0
        long_n = (sub["direction"] == "LONG").sum()
        short_n = (sub["direction"] == "SHORT").sum()
        enh_n = (sub["hold_used"] != 5).sum()
        print(f"  {m:<8} {len(sub):>3} {wr:>4.0f}% {sub['net_ret'].mean()*100:>+8.3f}% "
              f"{sub['pnl_yuan'].sum():>+10,.0f} {long_n:>4} {short_n:>5} {enh_n:>6}")

    # ── 4. 2023 vs 2024 vs 2025 同一信号强度下表现 ──
    print(f"\n{'═' * 75}")
    print(" 各年 LONG 信号强度 (pred) vs net return — 信号在不同 regime 多 informative")
    print(f"{'═' * 75}")
    print(f"  {'year':<6} {'long_N':>7} {'pred_mean':>10} {'pred_p90':>9} "
          f"{'avg_net':>9} {'WR':>5} {'IC_signal':>10}")
    for y in [2023, 2024, 2025, 2026]:
        sub = df_t[(df_t["year"] == y) & (df_t["direction"] == "LONG")]
        if sub.empty:
            continue
        ic = np.corrcoef(sub["pred"], sub["net_ret"])[0,1] if len(sub) > 5 else np.nan
        print(f"  {y:<6} {len(sub):>7} {sub['pred'].mean():>+9.4f} "
              f"{sub['pred'].quantile(0.9):>+8.4f} "
              f"{sub['net_ret'].mean()*100:>+8.3f}% {(sub['net_ret']>0).mean()*100:>4.0f}% "
              f"{ic:>+9.4f}")

    # ── 5. 2023 vs 2026 是不是因为没触发 enhancement? ──
    print(f"\n{'═' * 75}")
    print(" Enhancement 触发分布 by year")
    print(f"{'═' * 75}")
    print(f"  {'year':<6} {'total':>5} {'enh_total':>9} {'strict':>7} {'dip':>4}")
    for y in [2023, 2024, 2025, 2026]:
        sub = df_t[df_t["year"] == y]
        if sub.empty:
            continue
        strict = ((sub["direction"] == "LONG") & sub["in_strict_bull"]).sum()
        dip = ((sub["direction"] == "LONG") & sub["in_dip_bull"] & ~sub["in_strict_bull"]).sum()
        print(f"  {y:<6} {len(sub):>5} {strict + dip:>9} {strict:>7} {dip:>4}")

    # ── 6. 各年 trade 类型分布 ──
    print(f"\n{'═' * 75}")
    print(" Trade exit reason by year")
    print(f"{'═' * 75}")
    pivot = df_t.pivot_table(values="pnl_yuan", index="year", columns="reason",
                              aggfunc=["count", "sum"], fill_value=0)
    print(pivot)


if __name__ == "__main__":
    main()
