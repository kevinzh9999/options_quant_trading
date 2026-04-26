#!/usr/bin/env python3
"""精细追溯 2026 Jan 行情 + 12 笔 LONG 交易的进出场"""
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
from scripts.daily_walkforward_bear_mirror import bear_short_enh_loose
from scripts.daily_walkforward_long_optim import simulate_trade_dynamic

DB_PATH = "/Users/kevinzhao/Documents/options_quant_trading/data/storage/trading.db"
HOLD_DAYS_DEFAULT = 5
RETRAIN_EVERY = 20
INITIAL_TRAIN_DAYS = 200
TOP_PCT = 0.20
BOT_PCT = 0.20
ATR_K_DEFAULT = 1.5
SLIPPAGE_PCT = 0.0008


def build_pipe():
    base = build_default_pipeline()
    base.factors.extend([CloseSma60Factor(), Slope60dFactor(), VolRegimeFactor()])
    base.feature_names = [f.name for f in base.factors]
    return base


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

    # ── 2026 Jan + 周边日线行情 ──
    print(f"\n{'═' * 100}")
    print(" 2026 Jan 日线 (CSI 1000 — 现货)")
    print(f"{'═' * 100}")
    print(f"  {'date':<10} {'open':>8} {'high':>8} {'low':>8} {'close':>8} "
          f"{'chg%':>7} {'cum%_from_dec':>13} {'sma60_pct':>10} {'sma200_pct':>11} {'atr_pct':>9}")
    dec25_close = None
    for i, td in enumerate(dates):
        if td < "20251215" or td > "20260415":
            continue
        if td.startswith("20251231"):
            dec25_close = closes[i]
        if dec25_close is None:
            cum = 0
        else:
            cum = (closes[i] / dec25_close - 1) * 100
        chg = 0 if i == 0 else (closes[i]/closes[i-1]-1) * 100
        print(f"  {td:<10} {px_e.iloc[i]['open']:>8.0f} "
              f"{highs[i]:>8.0f} {lows[i]:>8.0f} {closes[i]:>8.0f} "
              f"{chg:>+6.2f}% {cum:>+12.2f}% "
              f"{regime['close_sma60'][i]:>+9.4f} {regime['close_sma200'][i]:>+10.4f} "
              f"{atr20[i]*100:>+8.2f}%")

    # ── 跑模型 + 记录 2026 Jan 全部 LONG signals + paths ──
    print(f"\n{'═' * 100}")
    print(" 2026 Jan 12 笔 LONG 交易详细 trace")
    print(f"{'═' * 100}")

    n = len(dates)
    fwd5 = np.array([closes[i+5]/closes[i]-1 if i+5 < n else np.nan for i in range(n)])
    pipe_proto = build_pipe()
    X_full = pipe_proto.features_matrix(dates, ctx)
    cached_pipe = None
    cached_thr = None
    last_train = -RETRAIN_EVERY
    trades_2026 = []

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

        td = dates[td_idx]
        if not td.startswith("202601"):
            continue
        if direction != "LONG":
            continue

        hold = HOLD_DAYS_DEFAULT
        atr_k = ATR_K_DEFAULT
        is_enhanced = False
        enh = enh_two_branch_v2(td_idx, regime)
        if enh is not None:
            hold, atr_k = enh
            is_enhanced = True

        sl = atr_k * atr20[td_idx] if not np.isnan(atr20[td_idx]) else None
        exit_idx, gross_ret, reason = simulate_trade_dynamic(
            direction, td_idx, closes, highs, lows,
            hold_days=hold, sl_pct=sl,
        )
        net_ret = gross_ret - SLIPPAGE_PCT
        pnl_pts = net_ret * closes[td_idx]
        # Build day-by-day path
        path = []
        for off in range(0, hold + 1):
            ii = td_idx + off
            if ii >= n:
                break
            path.append((dates[ii],
                         closes[ii],
                         (closes[ii] - closes[td_idx]) / closes[td_idx]))
        trades_2026.append({
            "entry_date": td,
            "entry_idx": td_idx,
            "exit_date": dates[exit_idx],
            "entry_close": closes[td_idx],
            "exit_close": closes[exit_idx],
            "direction": direction,
            "hold_used": hold,
            "atr_k": atr_k,
            "sl_pct": sl,
            "is_enhanced": is_enhanced,
            "enhancement_type": ("strict" if (regime['close_sma60'][td_idx] > 1.04 and
                                                regime['close_sma200'][td_idx] > 1.05)
                                 else ("dip" if (regime['close_sma200'][td_idx] > 1.03 and
                                                   regime['close_sma60'][td_idx] < 1.02)
                                       else "default")),
            "pred": pred,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "pnl_yuan": pnl_pts * 200,
            "reason": reason,
            "close_sma60": regime["close_sma60"][td_idx],
            "close_sma200": regime["close_sma200"][td_idx],
            "path": path,
        })

    print(f"  Trades found: {len(trades_2026)}")
    for t in trades_2026:
        print(f"\n  ── Entry {t['entry_date']} ({t['enhancement_type']}, hold {t['hold_used']}d, "
              f"SL {t['atr_k']}×ATR={t['sl_pct']*100:.2f}%) ──")
        print(f"     pred {t['pred']:+.4f}  c/sma60 {t['close_sma60']:.4f}  c/sma200 {t['close_sma200']:.4f}")
        print(f"     entry_close {t['entry_close']:.0f}  exit {t['exit_date']} @ {t['exit_close']:.0f}")
        print(f"     gross {t['gross_ret']*100:+.2f}%  net {t['net_ret']*100:+.2f}%  "
              f"PnL {t['pnl_yuan']:+,.0f}  reason {t['reason']}")
        for d_str, c, ret in t["path"]:
            marker = ""
            if d_str == t["entry_date"]:
                marker = " ← entry"
            elif d_str == t["exit_date"]:
                marker = " ← exit"
            print(f"     {d_str}  {c:.0f}  {ret*100:+.2f}%{marker}")

    # 总体 Jan 26 的 daily PnL 分布
    print(f"\n{'═' * 80}")
    print(" 2026-01 实际市场涨幅 vs 我们做 LONG 的总损失")
    print(f"{'═' * 80}")
    jan_idxs = [i for i, d in enumerate(dates) if d.startswith("202601")]
    if jan_idxs:
        i0, i1 = jan_idxs[0], jan_idxs[-1]
        m_chg = (closes[i1] / closes[i0-1] - 1) * 100  # vs Dec 31
        print(f"  Jan 起涨日(=Dec 31 close) → Jan {dates[i1][6:]} close: {m_chg:+.2f}%")
        net_pnl = sum(t["pnl_yuan"] for t in trades_2026)
        print(f"  我们 12 笔 LONG net PnL: {net_pnl:+,.0f}")
        print(f"  ⚠️  在单边上涨 {m_chg:+.2f}% 行情下，做 LONG 反而亏 {net_pnl:+,.0f} — 必有结构问题")


if __name__ == "__main__":
    main()
