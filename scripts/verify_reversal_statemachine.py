#!/usr/bin/env python3
"""Verify ReversalDetector class matches the inline state machine in v2_with_reversal_backtest.

For each trading day, feeds bars to both state machines and compares:
1. The inline backtest logic (from v2_with_reversal_backtest.py)
2. The ReversalDetector class (from reversal_signal.py)

They must produce identical reversal signals (direction, depth, timing).

Usage:
    python scripts/verify_reversal_statemachine.py
    python scripts/verify_reversal_statemachine.py --days 30
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from multiprocessing import Pool
from datetime import time as _time


def _verify_day(td):
    """Compare state machine outputs for one day."""
    from data.storage.db_manager import get_db
    from strategies.intraday.A_share_momentum_signal_v2 import _get_utc_time
    from strategies.intraday.reversal_signal import ReversalDetector

    db = get_db()
    date_dash = f"{td[:4]}-{td[4:6]}-{td[6:]}"

    all_bars = db.query_df(
        "SELECT datetime, open, high, low, close, volume FROM index_min "
        "WHERE symbol='000852' AND period=300 ORDER BY datetime")
    if all_bars is None or all_bars.empty:
        return td, True, []

    for c in ["open", "high", "low", "close", "volume"]:
        all_bars[c] = all_bars[c].astype(float)

    today_mask = all_bars["datetime"].str.startswith(date_dash)
    today_indices = all_bars.index[today_mask].tolist()
    if not today_indices:
        return td, True, []

    NO_TRADE_BEFORE = _time(1, 25)
    NO_TRADE_AFTER = _time(7, 5)

    # --- Inline state machine (from v2_with_reversal_backtest) ---
    state_inline = 'NONE'
    consec_bear_i = 0
    consec_bull_i = 0
    reversal_bull_count_i = 0
    reversal_bear_count_i = 0
    trend_confirm_price_i = 0.0
    trend_extreme_price_i = 0.0
    inline_signals = []

    # --- ReversalDetector class ---
    det = ReversalDetector("IM", {"enabled": True, "min_depth": 10, "max_depth": 30})
    class_signals = []

    for idx in today_indices:
        bar_open = float(all_bars.loc[idx, 'open'])
        bar_close = float(all_bars.loc[idx, 'close'])
        bar_high = float(all_bars.loc[idx, 'high'])
        bar_low = float(all_bars.loc[idx, 'low'])

        bar_5m = all_bars.loc[:idx].tail(199).copy()
        if len(bar_5m) < 16:
            continue

        utc_time = _get_utc_time(bar_5m)
        if not utc_time or utc_time < NO_TRADE_BEFORE or utc_time > NO_TRADE_AFTER:
            continue

        bar_dt = all_bars.loc[idx, 'datetime']

        # --- Inline logic (copy from v2_with_reversal_backtest) ---
        if bar_close > bar_open:
            candle = 'L'
        elif bar_close < bar_open:
            candle = 'S'
        else:
            candle = '-'

        if candle == 'L':
            consec_bull_i += 1
            consec_bear_i = 0
        elif candle == 'S':
            consec_bear_i += 1
            consec_bull_i = 0

        if consec_bear_i >= 4 and state_inline != 'BEAR_CONFIRMED':
            state_inline = 'BEAR_CONFIRMED'
            trend_confirm_price_i = bar_close
            trend_extreme_price_i = bar_close
            reversal_bull_count_i = 0

        if consec_bull_i >= 4 and state_inline != 'BULL_CONFIRMED':
            state_inline = 'BULL_CONFIRMED'
            trend_confirm_price_i = bar_close
            trend_extreme_price_i = bar_close
            reversal_bear_count_i = 0

        if state_inline == 'BEAR_CONFIRMED':
            trend_extreme_price_i = min(trend_extreme_price_i, bar_low)
        elif state_inline == 'BULL_CONFIRMED':
            trend_extreme_price_i = max(trend_extreme_price_i, bar_high)

        reversal_direction_i = None
        reversal_depth_i = 0
        if state_inline == 'BEAR_CONFIRMED':
            if candle == 'L':
                reversal_bull_count_i += 1
            elif candle == 'S':
                reversal_bull_count_i = 0
            if reversal_bull_count_i >= 3:
                depth = trend_confirm_price_i - trend_extreme_price_i
                if 10 <= depth <= 30:
                    reversal_direction_i = 'LONG'
                    reversal_depth_i = depth
                state_inline = 'BULL_CONFIRMED'
                trend_confirm_price_i = bar_close
                trend_extreme_price_i = bar_high
                reversal_bear_count_i = 0
                consec_bull_i = 3

        elif state_inline == 'BULL_CONFIRMED':
            if candle == 'S':
                reversal_bear_count_i += 1
            elif candle == 'L':
                reversal_bear_count_i = 0
            if reversal_bear_count_i >= 3:
                depth = trend_extreme_price_i - trend_confirm_price_i
                if 10 <= depth <= 30:
                    reversal_direction_i = 'SHORT'
                    reversal_depth_i = depth
                state_inline = 'BEAR_CONFIRMED'
                trend_confirm_price_i = bar_close
                trend_extreme_price_i = bar_low
                reversal_bull_count_i = 0
                consec_bear_i = 3

        if reversal_direction_i:
            inline_signals.append((bar_dt, reversal_direction_i, round(reversal_depth_i, 2)))

        # --- ReversalDetector class ---
        rev = det.update(bar_open, bar_high, bar_low, bar_close)
        if rev is not None:
            class_signals.append((bar_dt, rev.direction, round(rev.depth, 2)))

    # Compare
    match = inline_signals == class_signals
    diffs = []
    if not match:
        max_n = max(len(inline_signals), len(class_signals))
        for i in range(max_n):
            inl = inline_signals[i] if i < len(inline_signals) else None
            cls = class_signals[i] if i < len(class_signals) else None
            if inl != cls:
                diffs.append((i, inl, cls))

    return td, match, diffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    from data.storage.db_manager import get_db
    db = get_db()

    dates_df = db.query_df(
        "SELECT DISTINCT trade_date FROM index_daily "
        "WHERE ts_code='000852.SH' AND trade_date <= '20260414' "
        "ORDER BY trade_date DESC "
        f"LIMIT {args.days}"
    )
    dates = sorted(dates_df['trade_date'].tolist())
    print(f"Verifying ReversalDetector vs inline state machine for {len(dates)} days...")

    with Pool(7) as p:
        results = p.map(_verify_day, dates)

    all_ok = True
    for td, match, diffs in sorted(results, key=lambda x: x[0]):
        status = "OK" if match else "MISMATCH"
        if not match:
            all_ok = False
            print(f"  {td}: {status}")
            for i, inl, cls in diffs:
                print(f"    #{i}: inline={inl}  class={cls}")
        else:
            print(f"  {td}: {status}")

    if all_ok:
        print(f"\nAll {len(dates)} days match. ReversalDetector is verified.")
    else:
        print(f"\nMISMATCHES found! Fix ReversalDetector before proceeding.")
        sys.exit(1)


if __name__ == '__main__':
    main()
