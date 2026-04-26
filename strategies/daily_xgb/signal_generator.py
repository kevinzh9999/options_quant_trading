"""Daily XGB signal generator.

Run after EOD (after daily_record.py eod completes). Outputs:
  1. JSON file tmp/daily_xgb_pending_{date}.json
  2. DB row in daily_xgb_signals

Usage:
    python -m strategies.daily_xgb.signal_generator                  # signal for today
    python -m strategies.daily_xgb.signal_generator --date 20260424
    python -m strategies.daily_xgb.signal_generator --dry-run        # don't write JSON/DB
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .config import DailyXGBConfig
from .factors import build_full_pipeline, load_default_context
from .pipeline import get_or_train_model, predict_for_date, TrainedModel
from .regime import compute_regime, decide_trade_params
from . import persist


def _bj_today() -> str:
    return (datetime.utcnow() + timedelta(hours=8)).strftime("%Y%m%d")


def _compute_atr20(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> np.ndarray:
    """ATR20 as fraction of close, returned per-day."""
    n = len(closes)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    atr = np.full(n, np.nan)
    for i in range(20, n):
        atr[i] = tr[i-19:i+1].mean()
    return atr / np.maximum(closes, 1)


def _next_trading_day(date_str: str, calendar: pd.DataFrame) -> str:
    """Get next trading day after date_str from index_daily."""
    nxt = calendar[calendar["trade_date"] > date_str].head(1)
    return str(nxt.iloc[0]["trade_date"]) if not nxt.empty else date_str


def generate_signal(date: str, cfg: Optional[DailyXGBConfig] = None,
                     dry_run: bool = False,
                     force_retrain: bool = False) -> Dict[str, Any]:
    """Generate signal for a given date (T = signal date, T+1 = entry day).

    Returns the signal record (JSON-serializable). If LONG/SHORT triggered,
    records to DB and writes JSON. If no signal, status = SKIPPED_NO_DIRECTION.
    """
    cfg = cfg or DailyXGBConfig()
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

    # Load all data
    ctx = load_default_context(cfg.db_path)
    iv_dates = set(ctx.iv_df["trade_date"].tolist())
    px_e = ctx.px_df[ctx.px_df["trade_date"].isin(iv_dates)].reset_index(drop=True)
    dates = px_e["trade_date"].tolist()
    closes = px_e["close"].astype(float).values
    highs = px_e["high"].astype(float).values
    lows = px_e["low"].astype(float).values

    if date not in dates:
        msg = f"Date {date} not found in eligible (IV-available) dates. Latest: {dates[-1]}"
        print(f"[!] {msg}")
        return {"signal_date": date, "status": "SKIPPED_NO_DATA", "reason": msg}

    sig_idx = dates.index(date)
    if sig_idx < cfg.initial_train_days:
        msg = f"Insufficient training data before {date} (idx {sig_idx} < {cfg.initial_train_days})"
        return {"signal_date": date, "status": "SKIPPED_INSUFFICIENT_DATA", "reason": msg}

    atr20 = _compute_atr20(closes, highs, lows)

    # Train (or load cached) model with cutoff = date
    if force_retrain:
        print(f"[*] FORCE retraining model with cutoff ≤ {date} ...")
    else:
        print(f"[*] Loading/training model with cutoff ≤ {date} ...")
    model = get_or_train_model(ctx, train_end_date=date, cfg=cfg,
                                  force_retrain=force_retrain)
    print(f"    Model: train_ic={model.train_ic:+.4f}  "
          f"top_thr={model.top_threshold:+.4f}  bot_thr={model.bot_threshold:+.4f}")

    # Predict for date
    pred, direction, feats = predict_for_date(model, ctx, date, cfg)
    if pred is None:
        msg = "Some features are NaN, cannot predict"
        return {"signal_date": date, "status": "SKIPPED_NAN_FEATURES", "reason": msg}

    print(f"    Pred: {pred:+.4f}  Direction: {direction}")

    # Determine regime
    state = compute_regime(closes, sig_idx)

    # Determine entry day (next trading day after date)
    next_idx = sig_idx + 1
    if next_idx >= len(dates):
        next_day = "TBD"
    else:
        next_day = dates[next_idx]

    record_base = {
        "signal_date": date,
        "underlying": cfg.underlying,
        "pred": pred,
        "top_thr": model.top_threshold,
        "bot_thr": model.bot_threshold,
        "train_end_date": model.train_end_date,
        "train_ic": model.train_ic,
        "n_train_samples": model.n_train_samples,
        "close_sma60": float(state.close_sma60) if not np.isnan(state.close_sma60) else None,
        "close_sma200": float(state.close_sma200) if not np.isnan(state.close_sma200) else None,
        "slope_60d": float(state.slope_60d) if not np.isnan(state.slope_60d) else None,
        "vol_regime": float(state.vol_regime) if not np.isnan(state.vol_regime) else None,
        "atr20_pct": float(atr20[sig_idx]) if not np.isnan(atr20[sig_idx]) else None,
        "entry_intended_open": float(closes[sig_idx]),  # 提示参考价 (T close)
        "lots_planned": cfg.lots_per_signal,
        "created_at": (datetime.utcnow() + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"),
    }

    if direction is None:
        record_base.update({
            "direction": "NONE",
            "enhancement_type": None,
            "hold_days": None,
            "atr_k": None,
            "sl_pct": None,
            "status": "SKIPPED_NO_DIRECTION",
            "reason": f"pred {pred:+.4f} between thresholds",
            "signal_json": None,
        })
        if not dry_run:
            persist.insert_signal(cfg, record_base)
        print(f"    Result: NO SIGNAL (pred between {model.bot_threshold:+.4f} and {model.top_threshold:+.4f})")
        return record_base

    # Decide trade params (gate + enhancement)
    params = decide_trade_params(direction, state, cfg)
    if params is None:
        # SHORT blocked by G3s
        record_base.update({
            "direction": direction,
            "enhancement_type": "blocked",
            "hold_days": None,
            "atr_k": None,
            "sl_pct": None,
            "status": "SKIPPED_GATE",
            "reason": "G3s SHORT block (bull regime)",
            "signal_json": None,
        })
        if not dry_run:
            persist.insert_signal(cfg, record_base)
        print(f"    Result: SHORT BLOCKED by G3s (close/sma60={state.close_sma60:.4f} > 1.04 AND close/sma200={state.close_sma200:.4f} > 1.05)")
        return record_base

    sl_pct = params.atr_k * atr20[sig_idx]

    # Build full signal record
    signal_json = {
        "strategy_id": cfg.strategy_id,
        "signal_date": date,
        "entry_date_planned": next_day,
        "underlying": cfg.underlying,
        "direction": direction,
        "lots": cfg.lots_per_signal,
        "hold_days": params.hold_days,
        "sl_pct": sl_pct,
        "atr_k": params.atr_k,
        "enhancement_type": params.enhancement,
        "pred": pred,
        "regime": {
            "close_sma60": record_base["close_sma60"],
            "close_sma200": record_base["close_sma200"],
            "slope_60d": record_base["slope_60d"],
            "vol_regime": record_base["vol_regime"],
            "atr20_pct": record_base["atr20_pct"],
        },
        "entry_intended_open": record_base["entry_intended_open"],
        "generated_at_bj": record_base["created_at"],
    }

    record_base.update({
        "direction": direction,
        "enhancement_type": params.enhancement,
        "hold_days": params.hold_days,
        "atr_k": params.atr_k,
        "sl_pct": sl_pct,
        "status": "PENDING",
        "reason": None,
        "signal_json": json.dumps(signal_json, ensure_ascii=False),
    })

    if not dry_run:
        persist.insert_signal(cfg, record_base)
        path = persist.write_pending_signal(cfg, signal_json)
        print(f"    Wrote pending signal → {path}")

    print(f"\n=== SIGNAL: {direction} {cfg.underlying} ===")
    print(f"    Entry: {next_day} open (planned)")
    print(f"    Lots:  {cfg.lots_per_signal}")
    print(f"    Hold:  {params.hold_days} 个交易日 ({params.enhancement})")
    print(f"    SL:    {sl_pct*100:.2f}% (ATR×{params.atr_k}, ATR20={atr20[sig_idx]*100:.2f}%)")
    print(f"    Pred:  {pred:+.4f}  (thr {model.bot_threshold:+.4f}/{model.top_threshold:+.4f})")
    print(f"    Regime: c/sma60={state.close_sma60:.4f}  c/sma200={state.close_sma200:.4f}  "
          f"slope60={state.slope_60d:+.5f}  vol_regime={state.vol_regime:.3f}")

    return record_base


def main():
    ap = argparse.ArgumentParser(description="Daily XGB signal generator")
    ap.add_argument("--date", default=None, help="Signal date (YYYYMMDD), default = today BJ")
    ap.add_argument("--dry-run", action="store_true", help="Don't write JSON or DB")
    ap.add_argument("--force-retrain", action="store_true",
                     help="Ignore cached model, retrain from scratch")
    args = ap.parse_args()

    cfg = DailyXGBConfig()
    date = args.date or _bj_today()
    print(f"=== Daily XGB Signal Generator ===")
    print(f"Signal date: {date}")
    print(f"Mode: 保守 (1 lot/signal, cap=10)")
    print(f"DB: {cfg.db_path}")
    print(f"Dry run: {args.dry_run}")
    print()

    try:
        result = generate_signal(date, cfg, dry_run=args.dry_run,
                                    force_retrain=args.force_retrain)
        if result.get("status") == "PENDING":
            sys.exit(0)
        else:
            print(f"\n[!] {result.get('status')}: {result.get('reason')}")
            sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
