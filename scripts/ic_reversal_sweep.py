#!/usr/bin/env python3
"""IC reversal depth parameter sweep with IS/OOS split.

IS: 20250101+, OOS: before 20250101
Runs v2+reversal backtest for IC with multiple depth configs.

Usage:
    python scripts/ic_reversal_sweep.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiprocessing import Pool
from collections import defaultdict


# Depth configs to test
DEPTH_CONFIGS = [
    (5, 25), (5, 30),
    (10, 25), (10, 30), (10, 35), (10, 40),
    (15, 30), (15, 40), (15, 50),
    (20, 40), (20, 50),
]

COST = 147
MULT = 200

_G_DEPTH = (10, 30)


def _init(depth):
    global _G_DEPTH
    _G_DEPTH = depth


def _worker(args):
    td, enable_rev = args
    from scripts.v2_with_reversal_backtest import run_day_v2_reversal
    trades = run_day_v2_reversal("IC", td, _G_DEPTH[0], _G_DEPTH[1],
                                  enable_reversal=enable_rev)
    return (td, enable_rev, trades)


def run_sweep(dates, depth):
    """Run full sweep for one depth config."""
    tasks = []
    for td in dates:
        tasks.append((td, False))  # v2 only (baseline)
        tasks.append((td, True))   # v2 + reversal

    with Pool(7, initializer=_init, initargs=(depth,)) as p:
        results = p.map(_worker, tasks)

    # Aggregate
    is_base_pnl = is_combo_pnl = 0.0
    is_base_n = is_combo_n = 0
    oos_base_pnl = oos_combo_pnl = 0.0
    oos_base_n = oos_combo_n = 0
    rev_trades_is = rev_trades_oos = 0

    for td, enable_rev, trades in results:
        is_period = td >= "20250101"
        n = len(trades)
        pnl = sum(t['pnl'] for t in trades)
        net = pnl * MULT - n * COST
        rev_n = sum(1 for t in trades if t.get('source') == 'reversal')

        if is_period:
            if enable_rev:
                is_combo_pnl += net
                is_combo_n += n
                rev_trades_is += rev_n
            else:
                is_base_pnl += net
                is_base_n += n
        else:
            if enable_rev:
                oos_combo_pnl += net
                oos_combo_n += n
                rev_trades_oos += rev_n
            else:
                oos_base_pnl += net
                oos_base_n += n

    return {
        'depth': depth,
        'is_base': is_base_pnl, 'is_combo': is_combo_pnl,
        'is_delta': is_combo_pnl - is_base_pnl,
        'is_base_n': is_base_n, 'is_combo_n': is_combo_n,
        'oos_base': oos_base_pnl, 'oos_combo': oos_combo_pnl,
        'oos_delta': oos_combo_pnl - oos_base_pnl,
        'oos_base_n': oos_base_n, 'oos_combo_n': oos_combo_n,
        'rev_is': rev_trades_is, 'rev_oos': rev_trades_oos,
    }


def run_monthly(dates, depth):
    """Run monthly breakdown for one depth config."""
    tasks = [(td, True) for td in dates]
    baseline_tasks = [(td, False) for td in dates]
    all_tasks = tasks + baseline_tasks

    with Pool(7, initializer=_init, initargs=(depth,)) as p:
        results = p.map(_worker, all_tasks)

    monthly = defaultdict(lambda: {
        'combo_n': 0, 'combo_pnl': 0.0,
        'base_n': 0, 'base_pnl': 0.0,
        'rev_n': 0, 'rev_pnl': 0.0,
    })

    for td, enable_rev, trades in results:
        month = td[:6]
        n = len(trades)
        pnl = sum(t['pnl'] for t in trades)
        net = pnl * MULT - n * COST

        if enable_rev:
            monthly[month]['combo_n'] += n
            monthly[month]['combo_pnl'] += net
            for t in trades:
                if t.get('source') == 'reversal':
                    monthly[month]['rev_n'] += 1
                    monthly[month]['rev_pnl'] += t['pnl'] * MULT - COST
        else:
            monthly[month]['base_n'] += n
            monthly[month]['base_pnl'] += net

    return monthly


def main():
    from data.storage.db_manager import get_db
    db = get_db()

    dates = db.query_df(
        "SELECT DISTINCT trade_date FROM index_daily "
        "WHERE ts_code='000905.SH' AND trade_date >= '20220722' AND trade_date <= '20260414' "
        "ORDER BY trade_date")
    all_dates = dates['trade_date'].tolist()
    is_dates = [d for d in all_dates if d >= '20250101']
    oos_dates = [d for d in all_dates if d < '20250101']
    print(f"IC reversal depth sweep")
    print(f"IS: {len(is_dates)} days (20250101+), OOS: {len(oos_dates)} days (<20250101)")
    print(f"Depth configs: {len(DEPTH_CONFIGS)}")
    print(f"{'='*100}\n")

    # Run all configs
    results = []
    for i, depth in enumerate(DEPTH_CONFIGS):
        print(f"  [{i+1}/{len(DEPTH_CONFIGS)}] depth {depth[0]}-{depth[1]}...")
        r = run_sweep(all_dates, depth)
        results.append(r)
        print(f"    IS: base={r['is_base']:+.0f} combo={r['is_combo']:+.0f} delta={r['is_delta']:+.0f}"
              f"  OOS: base={r['oos_base']:+.0f} combo={r['oos_combo']:+.0f} delta={r['oos_delta']:+.0f}"
              f"  rev_trades IS={r['rev_is']} OOS={r['rev_oos']}")

    # Print summary table
    print(f"\n{'='*100}")
    print(f" IC v2+Reversal Depth Sweep")
    print(f"{'='*100}")
    print(f" {'Depth':>8s} | {'IS Base':>10s} {'IS Combo':>10s} {'IS Delta':>10s} | "
          f"{'OOS Base':>10s} {'OOS Combo':>10s} {'OOS Delta':>10s} | {'RevIS':>5s} {'RevOOS':>6s} | {'Both>0':>6s}")
    print(f" ---------+------------------------------------+------------------------------------+-------------+-------")

    best = None
    for r in sorted(results, key=lambda x: x['oos_delta'], reverse=True):
        d = r['depth']
        both_pos = "YES" if r['is_delta'] > 0 and r['oos_delta'] > 0 else "no"
        marker = ""
        if both_pos == "YES" and (best is None or r['oos_delta'] > best['oos_delta']):
            best = r
        print(f" {d[0]:>3d}-{d[1]:<3d}  | {r['is_base']:+10.0f} {r['is_combo']:+10.0f} {r['is_delta']:+10.0f} | "
              f"{r['oos_base']:+10.0f} {r['oos_combo']:+10.0f} {r['oos_delta']:+10.0f} | {r['rev_is']:5d} {r['rev_oos']:6d} | {both_pos:>6s}")

    print(f"\n Best (IS>0 AND OOS>0, by OOS delta): ", end="")
    if best:
        print(f"depth {best['depth'][0]}-{best['depth'][1]}"
              f" IS={best['is_delta']:+.0f} OOS={best['oos_delta']:+.0f}")
    else:
        print("NONE (no config with both IS>0 and OOS>0)")

    # Monthly breakdown for best config
    if best:
        print(f"\n{'='*100}")
        print(f" Monthly Breakdown: IC depth {best['depth'][0]}-{best['depth'][1]}")
        print(f"{'='*100}")

        monthly = run_monthly(all_dates, best['depth'])
        months = sorted(monthly.keys())

        print(f" {'Month':>7s} | {'Base N':>7s} {'Base Net':>10s} | {'Combo N':>7s} {'Combo Net':>10s} | "
              f"{'Delta':>9s} | {'Rev N':>5s} {'Rev Net':>8s} | {'Period':>4s}")
        print(f" --------+-----------------------+-----------------------+----------+----------------+------")

        is_delta_total = oos_delta_total = 0
        pos_m = neg_m = 0
        for m in months:
            d = monthly[m]
            delta = d['combo_pnl'] - d['base_pnl']
            period = "IS" if m >= "202501" else "OOS"
            marker = "+" if delta > 0 else "-"
            if delta > 0:
                pos_m += 1
            else:
                neg_m += 1
            if period == "IS":
                is_delta_total += delta
            else:
                oos_delta_total += delta
            print(f" {m:>7s} | {d['base_n']:7d} {d['base_pnl']:+10.0f} | {d['combo_n']:7d} {d['combo_pnl']:+10.0f} | "
                  f"{delta:+9.0f}{marker} | {d['rev_n']:5d} {d['rev_pnl']:+8.0f} | {period:>4s}")

        print(f" --------+-----------------------+-----------------------+----------+----------------+------")
        print(f" IS total delta: {is_delta_total:+.0f}")
        print(f" OOS total delta: {oos_delta_total:+.0f}")
        print(f" Positive/Negative months: {pos_m}/{neg_m}")

    # Write results
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "tmp", "ic_reversal_results.md")
    with open(output_path, "w") as f:
        f.write("# IC Reversal Depth Sweep Results\n\n")
        f.write(f"IS: {len(is_dates)} days (20250101+), OOS: {len(oos_dates)} days (<20250101)\n\n")

        f.write("## Depth Sweep Table\n\n")
        f.write("| Depth | IS Base | IS Combo | IS Delta | OOS Base | OOS Combo | OOS Delta | RevIS | RevOOS | Both>0 |\n")
        f.write("|-------|---------|----------|----------|----------|-----------|-----------|-------|--------|--------|\n")
        for r in sorted(results, key=lambda x: x['oos_delta'], reverse=True):
            d = r['depth']
            both = "YES" if r['is_delta'] > 0 and r['oos_delta'] > 0 else "no"
            f.write(f"| {d[0]}-{d[1]} | {r['is_base']:+.0f} | {r['is_combo']:+.0f} | {r['is_delta']:+.0f} | "
                    f"{r['oos_base']:+.0f} | {r['oos_combo']:+.0f} | {r['oos_delta']:+.0f} | "
                    f"{r['rev_is']} | {r['rev_oos']} | {both} |\n")

        if best:
            f.write(f"\n## Best Config\n\n")
            f.write(f"**depth {best['depth'][0]}-{best['depth'][1]}** "
                    f"(IS delta={best['is_delta']:+.0f}, OOS delta={best['oos_delta']:+.0f})\n\n")

            f.write("## Monthly Breakdown\n\n")
            f.write("| Month | Base N | Base Net | Combo N | Combo Net | Delta | Rev N | Rev Net | Period |\n")
            f.write("|-------|--------|----------|---------|-----------|-------|-------|---------|--------|\n")
            for m in months:
                d = monthly[m]
                delta = d['combo_pnl'] - d['base_pnl']
                period = "IS" if m >= "202501" else "OOS"
                f.write(f"| {m} | {d['base_n']} | {d['base_pnl']:+.0f} | {d['combo_n']} | {d['combo_pnl']:+.0f} | "
                        f"{delta:+.0f} | {d['rev_n']} | {d['rev_pnl']:+.0f} | {period} |\n")

            f.write(f"\n## Recommendation\n\n")
            if best['is_delta'] > 0 and best['oos_delta'] > 0:
                f.write(f"Deploy IC reversal with depth {best['depth'][0]}-{best['depth'][1]}.\n")
                f.write(f"Both IS ({best['is_delta']:+.0f}Y) and OOS ({best['oos_delta']:+.0f}Y) show positive delta.\n")
            else:
                f.write("Do not deploy. No config shows consistent improvement in both IS and OOS.\n")
        else:
            f.write("\n## Recommendation\n\n")
            f.write("Do not deploy. No config shows positive delta in both IS and OOS.\n")

    print(f"\nResults written to {output_path}")


if __name__ == '__main__':
    main()
