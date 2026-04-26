#!/usr/bin/env python3
"""TqBacktest end-to-end simulation for Daily XGB strategy.

Runs in a chosen window (default: 2025-04-01 ~ 2025-04-30, includes a strong
LONG dip-buy episode). For each session:

  1. At T-1 close, signal_generator computes signal using DB factors as of T-1
     (mimics offline EOD signal generation)
  2. T 09:30 open: TqBacktest replays the day; we wait for first 5min bar,
     submit limit OPEN order at last_price ± offset
  3. SL check at each subsequent day's close (via DB lookup)
  4. T+N open exit (TqBacktest replays exit day)

Compares TqBacktest realized PnL to self-backtest expected PnL.

Note: TqBacktest replays minute bars during continuous-auction hours (~09:30-15:00).
We use 5-min bars to know "trading is open" and place orders at 09:30 first bar.

Usage:
    python scripts/daily_xgb_tq_backtest.py --start 20250401 --end 20250430
"""
from __future__ import annotations

import argparse
import os
import sys
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(ROOT) / ".env")
except ImportError:
    pass

import numpy as np
import pandas as pd

from strategies.daily_xgb.config import DailyXGBConfig
from strategies.daily_xgb.factors import build_full_pipeline, load_default_context
from strategies.daily_xgb.regime import compute_regime, decide_trade_params


def compute_atr20(closes, highs, lows):
    n = len(closes)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    atr = np.full(n, np.nan)
    for i in range(20, n):
        atr[i] = tr[i-19:i+1].mean()
    return atr / np.maximum(closes, 1)


def _trading_calendar(db_path: str) -> List[str]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = conn.execute(
            "SELECT DISTINCT trade_date FROM index_daily "
            "WHERE ts_code='000852.SH' ORDER BY trade_date"
        )
        return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


def _resolve_main_contract(td: str) -> str:
    """Pick IM main contract by OI from futures_daily as of td."""
    from utils.cffex_calendar import active_im_months, _third_friday, _yymm
    import pandas as pd
    cfg = DailyXGBConfig()
    conn = sqlite3.connect(f"file:{cfg.db_path}?mode=ro", uri=True)
    try:
        active = active_im_months(td)
        cur = conn.execute(
            "SELECT ts_code, oi FROM futures_daily "
            "WHERE ts_code LIKE 'IM%' AND trade_date <= ? "
            "ORDER BY trade_date DESC LIMIT 20", (td,)
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    if rows:
        df = pd.DataFrame(rows, columns=["ts_code", "oi"])
        df = df.groupby("ts_code").first().reset_index()
        df["oi"] = df["oi"].astype(float)
        df = df.sort_values("oi", ascending=False)
        for _, r in df.iterrows():
            tc = str(r["ts_code"])
            m = tc[2:6]
            if m in active:
                return f"CFFEX.IM{m}"
    # fallback
    bt = pd.Timestamp(f"{td[:4]}-{td[4:6]}-{td[6:]}")
    tf = _third_friday(bt.year, bt.month)
    if bt.date() < tf.date():
        m = _yymm(bt.year, bt.month)
    else:
        m = active[1] if len(active) > 1 else _yymm(bt.year, bt.month + 1)
    return f"CFFEX.IM{m}"


def _generate_signal_offline(date: str, cfg: DailyXGBConfig,
                                X_full, dates, closes, highs, lows, atr20,
                                pipeline_cache, fwd5):
    """Generate signal for date using cached pipeline from walk-forward retrain logic."""
    sig_idx = dates.index(date) if date in dates else -1
    if sig_idx < cfg.initial_train_days:
        return None

    # Find or train pipeline (walk-forward retrain every retrain_every days)
    train_key = sig_idx - (sig_idx % cfg.retrain_every)
    if train_key not in pipeline_cache:
        train_end = train_key - cfg.hold_days_default
        if train_end < cfg.initial_train_days:
            return None
        X_tr = X_full[:train_end]
        y_tr = fwd5[:train_end]
        valid = ~(np.isnan(X_tr).any(axis=1) | np.isnan(y_tr))
        if valid.sum() < cfg.initial_train_days:
            return None
        pipeline = build_full_pipeline()
        pipeline.train(
            X_tr[valid], y_tr[valid],
            n_estimators=cfg.xgb_n_estimators,
            max_depth=cfg.xgb_max_depth,
            learning_rate=cfg.xgb_learning_rate,
            min_child_weight=cfg.xgb_min_child_weight,
            subsample=cfg.xgb_subsample,
            colsample_bytree=cfg.xgb_colsample_bytree,
            reg_alpha=cfg.xgb_reg_alpha,
            reg_lambda=cfg.xgb_reg_lambda,
            random_state=cfg.xgb_random_state,
        )
        pred_in = pipeline.predict(X_tr[valid])
        pipeline_cache[train_key] = {
            "pipeline": pipeline,
            "top": float(np.quantile(pred_in, 1 - cfg.top_pct)),
            "bot": float(np.quantile(pred_in, cfg.bot_pct)),
        }
    pc = pipeline_cache[train_key]

    x = X_full[sig_idx:sig_idx+1]
    if np.isnan(x).any():
        return None
    pred = float(pc["pipeline"].predict(x)[0])
    if pred >= pc["top"]:
        direction = "LONG"
    elif pred <= pc["bot"]:
        direction = "SHORT"
    else:
        return None

    state = compute_regime(closes, sig_idx)
    params = decide_trade_params(direction, state, cfg)
    if params is None:
        return None
    if np.isnan(atr20[sig_idx]):
        return None
    return {
        "direction": direction,
        "hold_days": params.hold_days,
        "atr_k": params.atr_k,
        "sl_pct": params.atr_k * atr20[sig_idx],
        "enhancement": params.enhancement,
        "pred": pred,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20250410")
    ap.add_argument("--end", default="20250430")
    ap.add_argument("--quick", action="store_true",
                     help="Quick offline simulation (no TqBacktest replay)")
    args = ap.parse_args()

    cfg = DailyXGBConfig()
    print(f"=== TqBacktest Daily XGB simulation ===")
    print(f"Window: {args.start} ~ {args.end}")
    print(f"Mode: {'QUICK (offline)' if args.quick else 'TqBacktest replay'}")

    # Pre-compute signals offline
    ctx = load_default_context(cfg.db_path)
    iv_dates = set(ctx.iv_df["trade_date"].tolist())
    px_e = ctx.px_df[ctx.px_df["trade_date"].isin(iv_dates)].reset_index(drop=True)
    dates = px_e["trade_date"].tolist()
    closes = px_e["close"].astype(float).values
    highs = px_e["high"].astype(float).values
    lows = px_e["low"].astype(float).values
    opens = px_e["open"].astype(float).values
    atr20 = compute_atr20(closes, highs, lows)
    n = len(dates)

    pipeline = build_full_pipeline()
    X_full = pipeline.features_matrix(dates, ctx)
    fwd5 = np.array([
        closes[i + cfg.hold_days_default] / closes[i] - 1
        if i + cfg.hold_days_default < n else np.nan
        for i in range(n)
    ])
    pipeline_cache = {}

    # Find session indices in window
    window_idxs = [i for i, d in enumerate(dates)
                    if args.start <= d <= args.end]
    if not window_idxs:
        print(f"No trading days in window {args.start}~{args.end}")
        sys.exit(1)
    print(f"Sessions in window: {len(window_idxs)}")

    # Pre-generate signals for each day
    signals = {}
    for sig_idx in window_idxs:
        d = dates[sig_idx]
        s = _generate_signal_offline(d, cfg, X_full, dates, closes, highs, lows, atr20,
                                        pipeline_cache, fwd5)
        if s is not None:
            signals[d] = s

    print(f"Pre-generated signals: {len(signals)}")
    for d, s in sorted(signals.items()):
        print(f"  {d}  {s['direction']}  hold={s['hold_days']}  "
              f"sl={s['sl_pct']*100:.2f}%  enh={s['enhancement']}  pred={s['pred']:+.4f}")

    if args.quick:
        # Offline simulation: use DB OHLC for fills (T+1 open entry, T+N+1 open exit)
        print(f"\n=== Offline T+1 open simulation ===")
        trades = []
        for sig_date, s in signals.items():
            sig_idx = dates.index(sig_date)
            entry_idx = sig_idx + 1
            if entry_idx >= n:
                continue
            entry_open = opens[entry_idx]
            sl_pct = s["sl_pct"]
            sign = 1 if s["direction"] == "LONG" else -1
            sl_triggered = False
            exit_idx = None
            exit_price = None
            exit_reason = None
            for off in range(s["hold_days"]):
                cur_idx = entry_idx + off
                if cur_idx >= n:
                    cur_idx = n - 1
                    break
                if sl_triggered:
                    exit_idx = cur_idx
                    exit_price = opens[cur_idx]
                    exit_reason = "SL"
                    break
                # EOD SL check via close
                if s["direction"] == "LONG":
                    cur_ret = (closes[cur_idx] - entry_open) / entry_open
                else:
                    cur_ret = (entry_open - closes[cur_idx]) / entry_open
                if cur_ret <= -sl_pct:
                    sl_triggered = True
            if exit_idx is None:
                exit_idx = entry_idx + s["hold_days"]
                if exit_idx >= n:
                    exit_idx = n - 1
                exit_price = opens[exit_idx] if not np.isnan(opens[exit_idx]) else closes[exit_idx]
                exit_reason = "TIME"
            gross_ret = sign * (exit_price - entry_open) / entry_open
            net_ret = gross_ret - cfg.slippage_pct
            pnl = net_ret * entry_open * cfg.contract_mult * cfg.lots_per_signal
            trades.append({
                "signal_date": sig_date,
                "entry_date": dates[entry_idx],
                "exit_date": dates[exit_idx],
                "direction": s["direction"],
                "entry_price": entry_open,
                "exit_price": exit_price,
                "gross_ret_pct": gross_ret * 100,
                "net_ret_pct": net_ret * 100,
                "pnl_yuan": pnl,
                "reason": exit_reason,
            })

        if not trades:
            print("No trades")
            return
        df = pd.DataFrame(trades)
        print(df.to_string(index=False))
        print(f"\nTotal: {len(df)} trades")
        print(f"  Total PnL: {df['pnl_yuan'].sum():+,.0f}")
        print(f"  WR:        {(df['pnl_yuan']>0).mean()*100:.0f}%")
        print(f"  Reasons:   {df['reason'].value_counts().to_dict()}")
        df.to_csv("/tmp/daily_xgb_tqbt_offline.csv", index=False)
        print(f"\nSaved: /tmp/daily_xgb_tqbt_offline.csv")
        return

    # TqBacktest replay (per-session)
    print(f"\n=== Running TqBacktest replay ===")
    from tqsdk import TqApi, TqSim, TqBacktest, BacktestFinished, TqAuth

    auth = TqAuth(os.getenv("TQ_ACCOUNT", ""), os.getenv("TQ_PASSWORD", ""))

    cumulative_pnl = 0.0
    sim_trades = []

    # We need to handle each entry — TqBacktest one window per session
    # because positions persist across days only within same TqBacktest instance.
    # For simplicity, do whole-window replay with TqSim tracking everything.

    # Find earliest entry day and latest exit day
    sig_dates = sorted(signals.keys())
    if not sig_dates:
        print("No signals to test in window")
        return
    first_sig_idx = dates.index(sig_dates[0])
    last_sig_idx = dates.index(sig_dates[-1])
    last_signal = signals[sig_dates[-1]]
    last_exit_idx = min(last_sig_idx + 1 + last_signal["hold_days"], n - 1)
    bt_start = dates[first_sig_idx + 1]   # first entry day
    bt_end = dates[last_exit_idx]
    bt_start_dt = datetime.strptime(bt_start, "%Y%m%d").replace(hour=9, minute=0)
    bt_end_dt = datetime.strptime(bt_end, "%Y%m%d").replace(hour=15, minute=30)
    print(f"TqBacktest: {bt_start} 09:00 ~ {bt_end} 15:30")

    sim = TqSim(init_balance=10_000_000.0)
    import logging
    logging.getLogger("tqsdk").setLevel(logging.WARNING)
    api = TqApi(sim, auth=auth, backtest=TqBacktest(start_dt=bt_start_dt, end_dt=bt_end_dt))

    # Map of entry_date → signal
    entry_map = {}
    for sd, s in signals.items():
        si = dates.index(sd)
        if si + 1 < n:
            entry_map[dates[si + 1]] = (sd, s)

    # Open trade tracking
    open_trades = []  # (signal_date, signal, contract, entry_price, entry_idx, sl_price, planned_exit_date)

    # Subscribe to main contract on first day
    cur_contract = _resolve_main_contract(bt_start)
    print(f"Main contract: {cur_contract}")
    quote = api.get_quote(cur_contract)
    klines_5m = api.get_kline_serial(cur_contract, 300, 200)

    last_dt = None
    last_bar_ns = None
    placed_orders_today = set()
    closed_orders_today = set()

    try:
        while True:
            api.wait_update()
            if not api.is_changing(klines_5m):
                continue
            # Use iloc[-2] for the JUST-COMPLETED bar (last is being filled)
            if len(klines_5m) < 2:
                continue
            cur_ts = int(klines_5m.iloc[-2]["datetime"])
            if last_bar_ns is not None and cur_ts == last_bar_ns:
                continue
            last_bar_ns = cur_ts
            # Convert UTC ns → BJ datetime
            utc_dt = datetime.utcfromtimestamp(cur_ts / 1e9)
            cur_dt = utc_dt + timedelta(hours=8)
            cur_date = cur_dt.strftime("%Y%m%d")
            cur_hm = cur_dt.strftime("%H:%M")

            if last_dt is None or last_dt.date() != cur_dt.date():
                # New day
                placed_orders_today = set()
                closed_orders_today = set()
                print(f"  [DAY] {cur_date} (first bar @ {cur_hm} BJ)")
                # Re-resolve main contract
                new_contract = _resolve_main_contract(cur_date)
                if new_contract != cur_contract:
                    cur_contract = new_contract
                    quote = api.get_quote(cur_contract)
                    klines_5m = api.get_kline_serial(cur_contract, 300, 200)
                    print(f"  Switched to {cur_contract}")
            last_dt = cur_dt

            # First 5min bar after market open: submit OPEN orders for today's entries
            cur_minutes = cur_dt.hour * 60 + cur_dt.minute
            in_open_window = 9 * 60 + 30 <= cur_minutes <= 9 * 60 + 50
            if in_open_window and cur_date in entry_map:
                sd, s = entry_map[cur_date]
                key = (sd, "OPEN")
                if key not in placed_orders_today:
                    placed_orders_today.add(key)
                    bid = float(quote.bid_price1)
                    ask = float(quote.ask_price1)
                    last = float(quote.last_price)
                    if s["direction"] == "LONG":
                        tq_dir, tq_offset = "BUY", "OPEN"
                        limit_price = ask
                    else:
                        tq_dir, tq_offset = "SELL", "OPEN"
                        limit_price = bid
                    print(f"  [{cur_date} {cur_hm}] OPEN {s['direction']} {cur_contract} {cfg.lots_per_signal}@{limit_price:.1f}")
                    order = api.insert_order(cur_contract, direction=tq_dir, offset=tq_offset,
                                                volume=cfg.lots_per_signal, limit_price=limit_price)
                    deadline = cur_ts + 60 * 1e9
                    while not order.is_dead:
                        api.wait_update()
                        if int(klines_5m.iloc[-2]["datetime"]) > deadline:
                            break
                    filled = order.volume_orign - order.volume_left
                    if filled > 0:
                        ep = float(getattr(order, "trade_price", limit_price) or limit_price)
                        sl_price = (ep * (1 - s["sl_pct"]) if s["direction"] == "LONG"
                                     else ep * (1 + s["sl_pct"]))
                        # Compute planned exit date
                        cal = _trading_calendar(cfg.db_path)
                        si_cal = cal.index(cur_date) if cur_date in cal else len(cal) - 1
                        pe_idx = si_cal + s["hold_days"]
                        planned_exit_date = cal[min(pe_idx, len(cal) - 1)]
                        open_trades.append({
                            "signal_date": sd, "signal": s, "contract": cur_contract,
                            "entry_date": cur_date, "entry_price": ep,
                            "sl_price": sl_price,
                            "planned_exit_date": planned_exit_date,
                        })
                        print(f"    ✓ entry filled @ {ep:.1f}, sl={sl_price:.1f}, planned exit={planned_exit_date}")
                    else:
                        print(f"    ✗ entry not filled")

            # 14:55 ~ 15:00 — EOD SL/TIME check (catch any bar in this window)
            in_eod_window = 14 * 60 + 50 <= cur_minutes <= 14 * 60 + 59
            if in_eod_window:
                still_open = []
                for t in open_trades:
                    cur_close = float(quote.last_price)
                    s = t["signal"]
                    direction = s["direction"]
                    sl_hit = (direction == "LONG" and cur_close <= t["sl_price"]) or \
                             (direction == "SHORT" and cur_close >= t["sl_price"])
                    time_exit = cur_date >= t["planned_exit_date"]
                    if sl_hit or time_exit:
                        # Close at next-day open — for simplicity, close now via market order
                        if direction == "LONG":
                            tq_dir, tq_offset = "SELL", "CLOSE"
                            close_price = float(quote.bid_price1)
                        else:
                            tq_dir, tq_offset = "BUY", "CLOSE"
                            close_price = float(quote.ask_price1)
                        reason = "SL" if sl_hit else "TIME"
                        print(f"  [{cur_date} {cur_hm}] CLOSE {direction} {t['contract']} reason={reason} @ {close_price:.1f}")
                        order = api.insert_order(t["contract"], direction=tq_dir, offset=tq_offset,
                                                    volume=cfg.lots_per_signal, limit_price=close_price)
                        deadline = cur_ts + 30 * 1e9
                        while not order.is_dead:
                            api.wait_update()
                            if int(klines_5m.iloc[-2]["datetime"]) > deadline:
                                break
                        filled = order.volume_orign - order.volume_left
                        if filled > 0:
                            xp = float(getattr(order, "trade_price", close_price) or close_price)
                            sign = 1 if direction == "LONG" else -1
                            gross_ret = sign * (xp - t["entry_price"]) / t["entry_price"]
                            net_ret = gross_ret - cfg.slippage_pct
                            pnl = net_ret * t["entry_price"] * cfg.contract_mult * cfg.lots_per_signal
                            sim_trades.append({
                                "signal_date": t["signal_date"],
                                "entry_date": t["entry_date"],
                                "exit_date": cur_date,
                                "direction": direction,
                                "entry": t["entry_price"], "exit": xp,
                                "gross_ret%": gross_ret * 100,
                                "net_ret%": net_ret * 100,
                                "pnl_yuan": pnl, "reason": reason,
                            })
                            cumulative_pnl += pnl
                            print(f"    ✓ closed @ {xp:.1f}, pnl={pnl:+,.0f}")
                        else:
                            still_open.append(t)
                    else:
                        still_open.append(t)
                open_trades = still_open
    except BacktestFinished:
        print(f"\nTqBacktest finished.")
    finally:
        api.close()

    # Force-close any remaining
    for t in open_trades:
        print(f"  [WARN] trade still open at end: {t['signal_date']}")

    if sim_trades:
        print(f"\n=== TqBacktest results ===")
        df = pd.DataFrame(sim_trades)
        print(df.to_string(index=False))
        print(f"\nTotal: {len(df)}")
        print(f"  Cumulative PnL: {cumulative_pnl:+,.0f}")
        print(f"  WR:             {(df['pnl_yuan']>0).mean()*100:.0f}%")
        df.to_csv("/tmp/daily_xgb_tqbt_replay.csv", index=False)


if __name__ == "__main__":
    main()
