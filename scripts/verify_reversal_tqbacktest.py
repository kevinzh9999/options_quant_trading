#!/usr/bin/env python3
"""TqBacktest verification for reversal implementation in monitor.

Runs TqBacktest for each date, driving the modified monitor with reversal logic.
Compares shadow trades output with the self-built v2+reversal backtest.

Usage:
    python scripts/verify_reversal_tqbacktest.py --date 20260410
    python scripts/verify_reversal_tqbacktest.py --days 5
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import argparse
from datetime import datetime
from typing import List


def run_comparison(td: str):
    """Compare self-built backtest with TqBacktest-driven monitor for one date."""
    # 1. Self-built backtest
    from scripts.v2_with_reversal_backtest import run_day_v2_reversal
    selfbuilt_trades = run_day_v2_reversal("IM", td, 10, 30, enable_reversal=True)

    # 2. TqBacktest-driven monitor
    from scripts.tq_backtest_monitor import run_tq_backtest
    from data.storage.db_manager import get_db
    db = get_db()

    # Backup existing shadow trades
    existing_shadow = db.query_df(
        f"SELECT * FROM shadow_trades WHERE trade_date='{td}'"
    )

    # Clear today's shadow records
    db._conn.execute(f"DELETE FROM shadow_trades WHERE trade_date='{td}'")
    db._conn.commit()

    try:
        tq_shadow = run_tq_backtest(td, ["IM"])
    finally:
        # Restore original shadow data
        db._conn.execute(f"DELETE FROM shadow_trades WHERE trade_date='{td}'")
        if existing_shadow is not None and len(existing_shadow) > 0:
            existing_shadow.to_sql("shadow_trades", db._conn, if_exists="append", index=False)
        db._conn.commit()

    # Extract TqBacktest trades
    tq_trades = []
    if tq_shadow is not None and len(tq_shadow) > 0:
        tq_im = tq_shadow[tq_shadow["symbol"] == "IM"].reset_index(drop=True)
        for _, row in tq_im.iterrows():
            tq_trades.append({
                "entry_time": row["entry_time"],
                "exit_time": row["exit_time"],
                "direction": row["direction"],
                "pnl": float(row["pnl_pts"]),
                "reason": row["exit_reason"],
            })

    return selfbuilt_trades, tq_trades


def compare_trades(selfbuilt, tq, td, tolerance=0.5):
    """Compare two trade lists, return match status and details."""
    match = True
    details = []

    max_n = max(len(selfbuilt), len(tq))
    for i in range(max_n):
        s = selfbuilt[i] if i < len(selfbuilt) else None
        t = tq[i] if i < len(tq) else None

        if s is None:
            details.append(f"  #{i+1}: EXTRA in TqBT: {t}")
            match = False
            continue
        if t is None:
            details.append(f"  #{i+1}: MISSING in TqBT: {s['direction']} {s['entry_time']}->{s['exit_time']} {s['pnl']:+.1f}pt {s.get('reason','')}")
            match = False
            continue

        # Compare key fields
        pnl_diff = abs(s["pnl"] - t["pnl"])
        time_match = s["entry_time"] == t["entry_time"] and s["exit_time"] == t["exit_time"]
        dir_match = s["direction"] == t["direction"]
        reason_match = s.get("reason", "") == t.get("reason", "")
        pnl_match = pnl_diff <= tolerance

        if time_match and dir_match and pnl_match and reason_match:
            details.append(f"  #{i+1}: OK  {s['direction']} {s['entry_time']}->{s['exit_time']} pnl={s['pnl']:+.1f}/{t['pnl']:+.1f}")
        else:
            match = False
            mismatches = []
            if not time_match:
                mismatches.append(f"time({s['entry_time']}->{s['exit_time']} vs {t['entry_time']}->{t['exit_time']})")
            if not dir_match:
                mismatches.append(f"dir({s['direction']} vs {t['direction']})")
            if not pnl_match:
                mismatches.append(f"pnl({s['pnl']:+.1f} vs {t['pnl']:+.1f}, diff={pnl_diff:.1f})")
            if not reason_match:
                mismatches.append(f"reason({s.get('reason','')} vs {t.get('reason','')})")
            details.append(f"  #{i+1}: MISMATCH {' '.join(mismatches)}")

    return match, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Single date YYYYMMDD")
    parser.add_argument("--days", type=int, default=None, help="Last N trading days")
    args = parser.parse_args()

    if args.date:
        dates = [args.date]
    elif args.days:
        from data.storage.db_manager import get_db
        db = get_db()
        dates_df = db.query_df(
            "SELECT DISTINCT trade_date FROM index_daily "
            "WHERE ts_code='000852.SH' AND trade_date <= '20260414' "
            f"ORDER BY trade_date DESC LIMIT {args.days}"
        )
        dates = sorted(dates_df['trade_date'].tolist())
    else:
        print("Specify --date or --days")
        return

    print(f"{'='*70}")
    print(f"  TqBacktest Verification for Reversal (IM)")
    print(f"  Dates: {len(dates)} days")
    print(f"{'='*70}\n")

    match_count = 0
    total = len(dates)
    for td in dates:
        print(f"\n--- {td} ---")
        try:
            selfbuilt, tq = run_comparison(td)
            s_pnl = sum(t["pnl"] for t in selfbuilt)
            t_pnl = sum(t["pnl"] for t in tq)
            ok, details = compare_trades(selfbuilt, tq, td)
            status = "MATCH" if ok else "MISMATCH"
            print(f"  Self-built: {len(selfbuilt)} trades, PnL={s_pnl:+.1f}pt")
            print(f"  TqBacktest: {len(tq)} trades, PnL={t_pnl:+.1f}pt")
            print(f"  Status: {status}")
            for d in details:
                print(d)
            if ok:
                match_count += 1
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n{'='*70}")
    print(f"  Result: {match_count}/{total} days matched")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
