#!/usr/bin/env python3
"""Strict bull (close/sma60>1.04 AND close/sma200>1.05) 的 IV 特征 — winner vs loser

特别看 2026 Jan 的 12 笔（全是 strict bull）的 IV 状态 vs 2024 winning strict bull。
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

DB_PATH = "/Users/kevinzhao/Documents/options_quant_trading/data/storage/trading.db"
RETRAIN_EVERY = 20
INITIAL_TRAIN_DAYS = 200
TOP_PCT = 0.20


def build_pipe():
    base = build_default_pipeline()
    base.factors.extend([CloseSma60Factor(), Slope60dFactor(), VolRegimeFactor()])
    base.feature_names = [f.name for f in base.factors]
    return base


def load_iv(db_path):
    c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    df = pd.read_sql_query(
        "SELECT trade_date, atm_iv_market, rr_25d, vrp, "
        "realized_vol_5d, realized_vol_20d, "
        "iv_term_spread, iv_percentile_hist "
        "FROM daily_model_output WHERE underlying='IM'", c)
    c.close()
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["iv_change"] = df["atm_iv_market"].diff()
    df["rr_change"] = df["rr_25d"].diff()
    df["vrp_change"] = df["vrp"].diff()
    df = df.set_index("trade_date")
    return df


def main():
    ctx = load_default_context(DB_PATH)
    px = ctx.px_df.copy()
    iv_dates = set(ctx.iv_df["trade_date"].tolist())
    px_e = px[px["trade_date"].isin(iv_dates)].reset_index(drop=True)
    dates = px_e["trade_date"].tolist()
    closes = px_e["close"].astype(float).values
    regime = precompute_regime(closes)
    iv_df = load_iv(DB_PATH)

    n = len(dates)
    fwd5 = np.array([closes[i+5]/closes[i]-1 if i+5 < n else np.nan for i in range(n)])
    fwd10 = np.array([closes[i+10]/closes[i]-1 if i+10 < n else np.nan for i in range(n)])
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
        s60 = regime["close_sma60"][td_idx]
        s200 = regime["close_sma200"][td_idx]
        if not (not np.isnan(s60) and not np.isnan(s200) and s60 > 1.04 and s200 > 1.05):
            continue
        td = dates[td_idx]
        if td not in iv_df.index:
            continue
        iv_row = iv_df.loc[td]
        rows.append({
            "date": td, "year": int(td[:4]), "month": td[:6],
            "ret5": fwd5[td_idx], "ret10": fwd10[td_idx],
            "atm_iv": iv_row["atm_iv_market"],
            "rr_25d": iv_row["rr_25d"],
            "vrp": iv_row["vrp"],
            "rv5": iv_row["realized_vol_5d"],
            "rv20": iv_row["realized_vol_20d"],
            "iv_term": iv_row["iv_term_spread"],
            "iv_pct": iv_row["iv_percentile_hist"],
            "iv_chg": iv_row["iv_change"],
            "rr_chg": iv_row["rr_change"],
            "vrp_chg": iv_row["vrp_change"],
        })

    df = pd.DataFrame(rows)
    print(f"Total strict-bull LONG signals: {len(df)}")
    print(f"By year: {df.groupby('year').size().to_dict()}")

    df["winner10"] = df["ret10"] > 0

    # Winner vs Loser
    print(f"\n=== 全样本 strict-bull: winner10 vs loser10 IV 对比 ===")
    win = df[df["winner10"]]
    los = df[~df["winner10"]]
    print(f"Winner N={len(win)}, Loser N={len(los)}")
    print(f"  {'feature':<14} {'win_mean':>11} {'los_mean':>11} {'diff':>11} {'t-stat':>9}")
    from scipy import stats
    for col in ["atm_iv", "rr_25d", "vrp", "iv_term", "iv_chg", "rr_chg",
                "rv5", "rv20", "iv_pct"]:
        wv = win[col].dropna()
        lv = los[col].dropna()
        if len(wv) == 0 or len(lv) == 0:
            continue
        diff = wv.mean() - lv.mean()
        t, p = stats.ttest_ind(wv, lv, equal_var=False)
        marker = " ***" if abs(t) > 2.5 else (" **" if abs(t) > 2 else (" *" if abs(t) > 1.5 else ""))
        print(f"  {col:<14} {wv.mean():>+11.4f} {lv.mean():>+11.4f} "
              f"{diff:>+11.4f} {t:>+8.2f}{marker}")

    # 2024 winners vs 2026 Jan losers
    print(f"\n=== 2024 strict-bull winners vs 2026 Jan losers 详细 ===")
    win24 = df[(df["year"] == 2024) & df["winner10"]]
    los26 = df[df["month"] == "202601"]
    print(f"\n  2024 winners (N={len(win24)}):")
    print(f"  {'date':<10} {'ret10':>7}  {'atm_iv':>8} {'rr_25d':>9} {'vrp':>8} "
          f"{'iv_term':>9} {'rv5':>7} {'iv_chg':>9}")
    for _, r in win24.iterrows():
        print(f"  {r['date']:<10} {r['ret10']*100:>+6.2f}%  "
              f"{r['atm_iv']:>+7.4f} {r['rr_25d']:>+8.4f} {r['vrp']:>+7.4f} "
              f"{r['iv_term']:>+8.4f} {r['rv5']:>+6.4f} {r['iv_chg']:>+8.4f}")

    print(f"\n  2026 Jan (N={len(los26)}):")
    print(f"  {'date':<10} {'ret10':>7}  {'atm_iv':>8} {'rr_25d':>9} {'vrp':>8} "
          f"{'iv_term':>9} {'rv5':>7} {'iv_chg':>9}")
    for _, r in los26.iterrows():
        print(f"  {r['date']:<10} {r['ret10']*100:>+6.2f}%  "
              f"{r['atm_iv']:>+7.4f} {r['rr_25d']:>+8.4f} {r['vrp']:>+7.4f} "
              f"{r['iv_term']:>+8.4f} {r['rv5']:>+6.4f} {r['iv_chg']:>+8.4f}")

    # Filter sweep
    print(f"\n=== Candidate strict-bull IV filters ===")
    candidates = [
        ("iv_chg < 0.005",                    df["iv_chg"] < 0.005),
        ("iv_chg < 0",                        df["iv_chg"] < 0),
        ("rr_25d < 0.02",                     df["rr_25d"] < 0.02),
        ("rr_25d > -0.05 AND rr_25d < 0.05",  (df["rr_25d"] > -0.05) & (df["rr_25d"] < 0.05)),
        ("vrp > -0.05 (vol not exploding)",   df["vrp"] > -0.05),
        ("rv5 < 0.30",                        df["rv5"] < 0.30),
        ("rv5 / rv20 < 1.5",                  df["rv5"] / df["rv20"] < 1.5),
        ("iv_pct < 70",                       df["iv_pct"] < 70),
        ("iv_term > 0",                       df["iv_term"] > 0),
    ]
    print(f"  {'Filter':<46} {'kept_N':>6} {'kept_WR':>8} {'kept_avg10':>11} "
          f"{'2026Jan_kept':>13}")
    for name, mask in candidates:
        kept = df[mask & ~mask.isna()]
        if len(kept) == 0:
            continue
        kept_wr = (kept["ret10"] > 0).mean() * 100
        kept_avg = kept["ret10"].mean() * 100
        kept_2026jan = kept[kept["month"] == "202601"]
        print(f"  {name:<46} {len(kept):>6} {kept_wr:>7.0f}% "
              f"{kept_avg:>+10.2f}% {len(kept_2026jan):>13}")


if __name__ == "__main__":
    main()
