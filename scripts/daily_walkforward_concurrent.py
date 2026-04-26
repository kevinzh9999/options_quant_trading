#!/usr/bin/env python3
"""计算 M7 strategy 实际同时持仓的峰值

每笔 trade 假设 1 手。LONG 和 SHORT 分别记，且同时存在时净仓位 = LONG - SHORT。
关键问题:
  - LONG 同时持仓最大几手?
  - SHORT 同时持仓最大几手?
  - 净仓位（多空相抵）最大几手?
  - 总名义仓位 (LONG + SHORT) 最大几手?
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
from scripts.daily_walkforward_bear_mirror import bear_short_enh_loose
from scripts.daily_walkforward_long_optim import simulate_trade_dynamic

DB_PATH = "/Users/kevinzhao/Documents/options_quant_trading/data/storage/trading.db"
HOLD_DAYS_DEFAULT = 5
RETRAIN_EVERY = 20
INITIAL_TRAIN_DAYS = 200
TOP_PCT = 0.20
BOT_PCT = 0.20
ATR_K_DEFAULT = 1.5


def build_pipe():
    base = build_default_pipeline()
    base.factors.extend([CloseSma60Factor(), Slope60dFactor(), VolRegimeFactor()])
    base.feature_names = [f.name for f in base.factors]
    return base


def collect_trades(label, ctx, dates, closes, highs, lows, atr20, regime,
                    iv_filter_fn=None, iv_lookup=None):
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

    for td_idx in range(INITIAL_TRAIN_DAYS, n - 16):
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

        # IV filter
        td = dates[td_idx]
        enh_allowed = True
        if iv_filter_fn is not None and iv_lookup is not None:
            iv_row = iv_lookup.get(td)
            if iv_row is not None:
                enh_allowed = iv_filter_fn(iv_row)

        hold = HOLD_DAYS_DEFAULT
        atr_k = ATR_K_DEFAULT
        if enh_allowed:
            if direction == "LONG":
                enh = enh_two_branch_v2(td_idx, regime)
                if enh is not None:
                    hold, atr_k = enh
            else:
                enh = bear_short_enh_loose(td_idx, regime)
                if enh is not None:
                    hold, atr_k = enh

        sl = atr_k * atr20[td_idx] if not np.isnan(atr20[td_idx]) else None
        if td_idx + hold >= n:
            continue
        exit_idx, gross_ret, reason = simulate_trade_dynamic(
            direction, td_idx, closes, highs, lows, hold, sl,
        )
        trades.append({
            "entry_idx": td_idx, "exit_idx": exit_idx,
            "entry_date": dates[td_idx], "exit_date": dates[exit_idx],
            "direction": direction, "hold_used": hold,
        })
    return pd.DataFrame(trades)


def analyze_concurrent(label, df, dates, n, eval_start):
    long_pos = np.zeros(n)
    short_pos = np.zeros(n)
    for _, r in df.iterrows():
        if r["direction"] == "LONG":
            long_pos[r["entry_idx"]: r["exit_idx"]] += 1
        else:
            short_pos[r["entry_idx"]: r["exit_idx"]] += 1
    net_pos = long_pos - short_pos
    gross_pos = long_pos + short_pos

    print(f"\n{'═' * 75}")
    print(f" {label}")
    print(f"{'═' * 75}")
    print(f"  Total trades: {len(df)}  (LONG {(df['direction']=='LONG').sum()}, "
          f"SHORT {(df['direction']=='SHORT').sum()})")
    print(f"  LONG 单边最大:   {int(long_pos.max())} 手")
    print(f"  SHORT 单边最大:  {int(short_pos.max())} 手")
    print(f"  净仓位 max +:    +{int(net_pos.max())} 手 (LONG bias)")
    print(f"  净仓位 max -:    {int(net_pos.min())} 手 (SHORT bias)")
    print(f"  GROSS (L+S) 最大: {int(gross_pos.max())} 手")
    print(f"  GROSS 平均:       {gross_pos[eval_start:].mean():.2f} 手")
    print(f"  GROSS 中位数:     {np.median(gross_pos[eval_start:]):.0f} 手")

    # Distribution
    print(f"\n  Gross 持仓分布:")
    gross_eval = gross_pos[eval_start:]
    for lots in range(int(gross_eval.max()) + 1):
        n_days = int((gross_eval == lots).sum())
        pct = n_days / len(gross_eval) * 100
        bar = "█" * int(pct / 2)
        print(f"    {lots:>2} 手: {n_days:>4} 天 ({pct:>5.1f}%) {bar}")

    # Yearly
    years = pd.Series([d[:4] for d in dates], dtype=str)
    print(f"\n  各年峰值 (LONG_max / SHORT_max / GROSS_max / GROSS_avg):")
    for y in ["2023", "2024", "2025", "2026"]:
        mask = (years == y).values
        if not mask.any():
            continue
        if not mask[eval_start:].any():
            continue
        sub_long = long_pos[mask]
        sub_short = short_pos[mask]
        sub_gross = gross_pos[mask]
        print(f"    {y}: L={int(sub_long.max())}  S={int(sub_short.max())}  "
              f"Gross={int(sub_gross.max())}  GrossAvg={sub_gross.mean():.2f}")
    return {"long": long_pos, "short": short_pos, "net": net_pos, "gross": gross_pos}


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
    n = len(dates)
    eval_start = INITIAL_TRAIN_DAYS

    # Load IV for IV3 filter
    import sqlite3
    c = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    iv_df = pd.read_sql_query(
        "SELECT trade_date, vrp FROM daily_model_output WHERE underlying='IM'", c)
    c.close()
    iv_df = iv_df.set_index("trade_date")
    iv_lookup = iv_df.to_dict("index")

    print("\n[Computing M7 trades...]")
    df_m7 = collect_trades("M7", ctx, dates, closes, highs, lows, atr20, regime)

    print("\n[Computing IV3 trades (vrp<0.02 enhancement filter)...]")
    df_iv3 = collect_trades("IV3", ctx, dates, closes, highs, lows, atr20, regime,
                              iv_filter_fn=lambda r: r["vrp"] < 0.02,
                              iv_lookup=iv_lookup)

    print(f"\n  Range: {dates[0]} ~ {dates[-1]}")
    print(f"  Eval starts at idx {eval_start}: {dates[eval_start]}")

    pos_m7 = analyze_concurrent("M7 baseline", df_m7, dates, n, eval_start)
    pos_iv3 = analyze_concurrent("IV3 (vrp<0.02 enhancement filter)", df_iv3, dates, n, eval_start)

    # Side-by-side comparison
    print(f"\n{'═' * 80}")
    print(" 仓位需求对比 (M7 vs IV3)")
    print(f"{'═' * 80}")
    print(f"  {'指标':<32} {'M7':>10} {'IV3':>10}")
    print(f"  {'LONG max':<32} {int(pos_m7['long'].max()):>10} {int(pos_iv3['long'].max()):>10}")
    print(f"  {'SHORT max':<32} {int(pos_m7['short'].max()):>10} {int(pos_iv3['short'].max()):>10}")
    print(f"  {'GROSS max (L+S)':<32} {int(pos_m7['gross'].max()):>10} {int(pos_iv3['gross'].max()):>10}")
    print(f"  {'GROSS avg':<32} {pos_m7['gross'][eval_start:].mean():>10.2f} "
          f"{pos_iv3['gross'][eval_start:].mean():>10.2f}")
    print(f"  {'NET max +':<32} {int(pos_m7['net'].max()):>+10} {int(pos_iv3['net'].max()):>+10}")
    print(f"  {'NET max -':<32} {int(pos_m7['net'].min()):>10} {int(pos_iv3['net'].min()):>10}")

    pass


if __name__ == "__main__":
    main()
