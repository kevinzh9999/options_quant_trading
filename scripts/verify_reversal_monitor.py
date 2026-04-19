#!/usr/bin/env python3
"""Verify reversal implementation in monitor matches v2_with_reversal_backtest.

Runs the self-built v2+reversal backtest for 30 recent trading days,
then compares with TqBacktest-driven monitor output.

Usage:
    python scripts/verify_reversal_monitor.py
    python scripts/verify_reversal_monitor.py --days 10
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from multiprocessing import Pool
from collections import defaultdict


def _run_one(args):
    td, = args
    from scripts.v2_with_reversal_backtest import run_day_v2_reversal
    trades = run_day_v2_reversal("IM", td, 10, 30, enable_reversal=True)
    return td, trades


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    from data.storage.db_manager import get_db
    db = get_db()

    # Get last N trading days
    dates_df = db.query_df(
        "SELECT DISTINCT trade_date FROM index_daily "
        "WHERE ts_code='000852.SH' AND trade_date <= '20260414' "
        "ORDER BY trade_date DESC "
        f"LIMIT {args.days}"
    )
    dates = sorted(dates_df['trade_date'].tolist())
    print(f"Running self-built v2+reversal backtest for {len(dates)} days (IM, depth 10-30)")

    # Run self-built backtest
    tasks = [(td,) for td in dates]
    with Pool(7) as p:
        results = p.map(_run_one, tasks)

    COST = 147
    MULT = 200
    total_trades = 0
    total_pnl = 0.0
    rev_trades = 0
    rev_pnl = 0.0

    print(f"\n{'Date':>10s}  {'N':>3s}  {'PnL(pt)':>8s}  {'Net(Y)':>9s}  {'Rev':>3s}  Details")
    print("-" * 90)
    for td, trades in sorted(results, key=lambda x: x[0]):
        day_pnl = sum(t['pnl'] for t in trades)
        day_net = day_pnl * MULT - len(trades) * COST
        day_rev = [t for t in trades if t.get('source') == 'reversal' or t.get('reason') == 'REVERSAL_EXIT']
        total_trades += len(trades)
        total_pnl += day_pnl
        rev_trades += len(day_rev)
        rev_pnl += sum(t['pnl'] for t in day_rev)

        details = []
        for t in trades:
            src = 'R' if t.get('source') == 'reversal' else 'V'
            rsn = t.get('reason', '')[:15]
            details.append(f"{src}{t['direction'][0]} {t['entry_time']}->{t['exit_time']} {t['pnl']:+.1f}pt {rsn}")
        detail_str = " | ".join(details) if details else "(no trades)"
        print(f"{td:>10s}  {len(trades):3d}  {day_pnl:+8.1f}  {day_net:+9.0f}  {len(day_rev):3d}  {detail_str}")

    print("-" * 90)
    total_net = total_pnl * MULT - total_trades * COST
    print(f"{'TOTAL':>10s}  {total_trades:3d}  {total_pnl:+8.1f}  {total_net:+9.0f}  {rev_trades:3d}")
    print(f"  Reversal trades: {rev_trades}, PnL={rev_pnl:+.1f}pt ({rev_pnl*MULT:+.0f}Y)")


if __name__ == '__main__':
    main()
