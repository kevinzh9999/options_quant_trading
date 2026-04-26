#!/usr/bin/env python3
"""IV/Skew/VRP 在不同 regime 期的分布差异 — 找 option-based confirmation filter

目标: 区分 "健康趋势 dip-buy" (2025 4月) vs "post-rally correction" (2026 Jan)
两者都满足 dip-bull (close/sma200>1.03 AND close/sma60<1.02)，但 2025 大赚 2026 大亏。

诊断:
  1. 2025 dip-buy 日子 vs 2026 Jan 日子的 IV/RR/VRP/term 分布
  2. 找出在 dip-bull regime 内的 separating signal
  3. 设计 confirmation filter
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import sqlite3
import numpy as np
import pandas as pd

from strategies.daily.factors import build_default_pipeline, load_default_context
from scripts.daily_robust_methods_compare import (
    CloseSma60Factor, Slope60dFactor, VolRegimeFactor, compute_atr20,
)
from scripts.daily_walkforward_gates import precompute_regime, gate_dual_and_short_only
from scripts.daily_walkforward_long_v2 import enh_two_branch_v2

DB_PATH = "/Users/kevinzhao/Documents/options_quant_trading/data/storage/trading.db"
RETRAIN_EVERY = 20
INITIAL_TRAIN_DAYS = 200
TOP_PCT = 0.20


def build_pipe():
    base = build_default_pipeline()
    base.factors.extend([CloseSma60Factor(), Slope60dFactor(), VolRegimeFactor()])
    base.feature_names = [f.name for f in base.factors]
    return base


def load_iv_full(db_path):
    """Load all IV/RR/VRP related fields from daily_model_output."""
    c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    df = pd.read_sql_query(
        "SELECT trade_date, atm_iv_market, rr_25d, vrp, "
        "realized_vol_5d, realized_vol_20d, "
        "iv_term_spread, iv_percentile_hist "
        "FROM daily_model_output WHERE underlying='IM'", c)
    df = df.sort_values("trade_date").reset_index(drop=True)
    # Compute changes locally
    df["iv_change"] = df["atm_iv_market"].diff()
    df["rr_change"] = df["rr_25d"].diff()
    df["vrp_change"] = df["vrp"].diff()
    c.close()
    df = df.set_index("trade_date")
    return df


def collect_dipbull_signals(ctx, dates, closes, regime, iv_df):
    """Collect all LONG signals where dip-bull enhancement triggered, with IV features."""
    n = len(dates)
    fwd5 = np.array([closes[i+5]/closes[i]-1 if i+5 < n else np.nan for i in range(n)])
    fwd10 = np.array([closes[i+10]/closes[i]-1 if i+10 < n else np.nan for i in range(n)])
    fwd15 = np.array([closes[i+15]/closes[i]-1 if i+15 < n else np.nan for i in range(n)])
    pipe_proto = build_pipe()
    X_full = pipe_proto.features_matrix(dates, ctx)
    cached_pipe = None
    cached_thr = None
    last_train = -RETRAIN_EVERY
    rows = []

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
        # dip-bull only
        s60 = regime["close_sma60"][td_idx]
        s200 = regime["close_sma200"][td_idx]
        if not (not np.isnan(s60) and not np.isnan(s200)
                and s200 > 1.03 and s60 < 1.02):
            continue
        td = dates[td_idx]
        if td not in iv_df.index:
            continue
        iv_row = iv_df.loc[td]
        rows.append({
            "date": td,
            "year": int(td[:4]),
            "month": td[:6],
            "close_sma60": s60,
            "close_sma200": s200,
            "ret5": fwd5[td_idx],
            "ret10": fwd10[td_idx],
            "ret15": fwd15[td_idx],
            "atm_iv": iv_row.get("atm_iv_market"),
            "rr_25d": iv_row.get("rr_25d"),
            "vrp": iv_row.get("vrp"),
            "rv5": iv_row.get("realized_vol_5d"),
            "rv20": iv_row.get("realized_vol_20d"),
            "iv_term": iv_row.get("iv_term_spread"),
            "iv_pct": iv_row.get("iv_percentile_hist"),
            "iv_chg": iv_row.get("iv_change"),
            "rr_chg": iv_row.get("rr_change"),
            "vrp_chg": iv_row.get("vrp_change"),
        })
    return pd.DataFrame(rows)


def main():
    ctx = load_default_context(DB_PATH)
    px = ctx.px_df.copy()
    iv_dates = set(ctx.iv_df["trade_date"].tolist())
    px_e = px[px["trade_date"].isin(iv_dates)].reset_index(drop=True)
    dates = px_e["trade_date"].tolist()
    closes = px_e["close"].astype(float).values
    regime = precompute_regime(closes)
    iv_df = load_iv_full(DB_PATH)

    df = collect_dipbull_signals(ctx, dates, closes, regime, iv_df)
    print(f"Total dip-bull LONG signals collected: {len(df)}")
    print(f"Year distribution: {df.groupby('year').size().to_dict()}")

    # Tag winning vs losing
    df["winner10"] = df["ret10"] > 0
    df["winner15"] = df["ret15"] > 0

    print(f"\n{'═' * 100}")
    print(" 各年 dip-bull 信号 IV/skew/VRP 均值对比")
    print(f"{'═' * 100}")
    print(f"  {'year':<6} {'N':>3}  ", end="")
    for col in ["atm_iv", "rr_25d", "vrp", "iv_term", "rv5", "rv20", "iv_chg", "rr_chg"]:
        print(f"{col:>9}  ", end="")
    print()
    for y in sorted(df["year"].unique()):
        sub = df[df["year"] == y]
        if sub.empty:
            continue
        print(f"  {y:<6} {len(sub):>3}  ", end="")
        for col in ["atm_iv", "rr_25d", "vrp", "iv_term", "rv5", "rv20", "iv_chg", "rr_chg"]:
            v = sub[col].mean()
            if pd.isna(v):
                print(f"{'NaN':>9}  ", end="")
            else:
                print(f"{v:>+8.4f}  ", end="")
        print()

    # 月度看 — 2025 Apr (大赢) vs 2026 Jan (大输)
    print(f"\n{'═' * 100}")
    print(" 2025 vs 2026 月度对比 (核心 separating signal hunt)")
    print(f"{'═' * 100}")
    print(f"  {'month':<8} {'N':>3} {'avg_ret10':>10} {'WR10':>5}  "
          f"{'atm_iv':>8} {'rr_25d':>9} {'vrp':>8} {'iv_term':>9} "
          f"{'iv_chg':>9} {'rr_chg':>9} {'rv5':>8} {'rv20':>8}")
    df["month"] = df["date"].astype(str).str[:6]
    for m, sub in df.groupby("month"):
        if int(m[:4]) < 2025:
            continue
        wr = (sub["ret10"] > 0).mean() * 100
        print(f"  {m:<8} {len(sub):>3} {sub['ret10'].mean()*100:>+9.2f}% "
              f"{wr:>4.0f}%  ", end="")
        for col in ["atm_iv", "rr_25d", "vrp", "iv_term", "iv_chg", "rr_chg", "rv5", "rv20"]:
            v = sub[col].mean()
            print(f"{v:>+8.4f}  " if not pd.isna(v) else f"{'NaN':>8}  ", end="")
        print()

    # Winner vs Loser dip-bull 总体对比
    print(f"\n{'═' * 100}")
    print(" 全样本 dip-bull: winner10 vs loser10 IV 特征对比 (找 separating signal)")
    print(f"{'═' * 100}")
    win = df[df["winner10"]]
    los = df[~df["winner10"]]
    print(f"  Winner N={len(win)}, Loser N={len(los)}")
    print(f"  {'feature':<14} {'win_mean':>11} {'win_med':>10}  "
          f"{'los_mean':>11} {'los_med':>10}  {'diff_mean':>11} {'t-stat':>9}")
    for col in ["atm_iv", "rr_25d", "vrp", "iv_term", "iv_chg", "rr_chg",
                "rv5", "rv20", "iv_pct", "vrp_chg"]:
        wv = win[col].dropna()
        lv = los[col].dropna()
        if len(wv) == 0 or len(lv) == 0:
            continue
        diff = wv.mean() - lv.mean()
        # naive t-stat
        from scipy import stats
        t, p = stats.ttest_ind(wv, lv, equal_var=False)
        marker = " ***" if abs(t) > 2 else (" *" if abs(t) > 1 else "")
        print(f"  {col:<14} {wv.mean():>+11.4f} {wv.median():>+10.4f}  "
              f"{lv.mean():>+11.4f} {lv.median():>+10.4f}  "
              f"{diff:>+11.4f} {t:>+8.2f}{marker}")

    # 在 winner/loser 上找候选 cutoff
    print(f"\n{'═' * 100}")
    print(" 候选 IV-based filter: dip-bull 增强需 [feature] 满足 condition")
    print(f"{'═' * 100}")
    print(f"  在全样本 dip-bull 上看，每个 filter 保留多少 winner / loser")

    candidates = [
        ("rr_25d > -0.04 (skew not too negative)", df["rr_25d"] > -0.04),
        ("rr_25d > -0.05",                          df["rr_25d"] > -0.05),
        ("rr_chg < 0.005 (skew not rapidly turning positive)", df["rr_chg"] < 0.005),
        ("iv_term > -0.005 (no severe backwardation)", df["iv_term"] > -0.005),
        ("iv_term > 0 (in contango)",               df["iv_term"] > 0),
        ("vrp > 0 (options expensive vs RV)",       df["vrp"] > 0),
        ("vrp > 0.02",                              df["vrp"] > 0.02),
        ("iv_chg < 0.01 (IV not spiking)",          df["iv_chg"] < 0.01),
        ("rv5 < 0.20 (recent vol not too high)",    df["rv5"] < 0.20),
        ("rv5 < rv20 * 1.3 (no vol expansion)",     df["rv5"] < df["rv20"] * 1.3),
    ]
    print(f"  {'Filter':<55} {'kept_N':>7} {'kept_WR':>8} {'kept_avg10':>11} "
          f"{'2026Jan_kept':>13}")
    for name, mask in candidates:
        kept = df[mask & ~mask.isna()]
        if len(kept) == 0:
            continue
        kept_wr = (kept["ret10"] > 0).mean() * 100
        kept_avg = kept["ret10"].mean() * 100
        kept_2026jan = kept[kept["month"] == "202601"]
        print(f"  {name:<55} {len(kept):>7} {kept_wr:>7.0f}% "
              f"{kept_avg:>+10.2f}% {len(kept_2026jan):>13}")

    # 日 detail for 2025-04 vs 2026-01
    print(f"\n{'═' * 100}")
    print(" 2025-04 (winners) vs 2026-01 (losers) 日详细对比")
    print(f"{'═' * 100}")
    for mo in ["202504", "202601"]:
        sub = df[df["month"] == mo].copy()
        if sub.empty:
            continue
        print(f"\n  ── {mo} ──")
        print(f"  {'date':<10} {'ret10':>7} {'rr_25d':>9} {'vrp':>8} {'iv_term':>9} "
              f"{'rv5':>7} {'iv_chg':>9}")
        for _, r in sub.iterrows():
            print(f"  {r['date']:<10} {r['ret10']*100:>+6.2f}% "
                  f"{r['rr_25d']:>+8.4f} {r['vrp']:>+7.4f} {r['iv_term']:>+8.4f} "
                  f"{r['rv5']:>+6.4f} {r['iv_chg']:>+8.4f}")


if __name__ == "__main__":
    main()
