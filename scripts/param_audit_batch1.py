#!/usr/bin/env python3
"""活跃参数审计批次1: trailing_stop_scale + me_ratio + TIME_STOP_MINUTES 双窗口扫描。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from data.storage.db_manager import get_db
from scripts.backtest_signals_day import run_day
from strategies.intraday.A_share_momentum_signal_v2 import SYMBOL_PROFILES

SPOTS = {'IC': '000905', 'IM': '000852'}

# 参数扫描空间
PARAMS = {
    'trailing_stop_scale': [1.0, 1.5, 2.0, 2.5, 3.0],
    'me_ratio': [0.06, 0.08, 0.10, 0.12, 0.15],
    'time_stop_minutes': [30, 45, 60, 75, 90],
}

# 当前值
CURRENT = {
    'IM': {'trailing_stop_scale': 1.5, 'me_ratio': 0.10, 'time_stop_minutes': 60},
    'IC': {'trailing_stop_scale': 2.0, 'me_ratio': 0.12, 'time_stop_minutes': 60},
}

# TIME_STOP_MINUTES需要修改的全局变量位置
# 在A_share_momentum_signal_v2.py中是 TIME_STOP_MINUTES = 60
# 需要通过修改模块级变量来改


def get_dates(db, spot):
    df = db.query_df(
        f"SELECT DISTINCT substr(datetime,1,10) as d FROM index_min "
        f"WHERE symbol='{spot}' AND period=300 ORDER BY d"
    )
    dates = [d.replace('-', '') for d in df['d'].tolist()]
    return dates[-900:] if len(dates) > 900 else dates


def _run_one_day(args):
    td, sym, overrides = args
    # 应用参数覆盖
    for key, val in overrides.items():
        if key == 'time_stop_minutes':
            # TIME_STOP_MINUTES是模块级常量，需要直接修改
            import strategies.intraday.A_share_momentum_signal_v2 as sig_mod
            sig_mod.TIME_STOP_MINUTES = val
            # 也需要修改factors.py里的对应值
            try:
                import strategies.intraday.factors as fmod
                # ExitEvaluator里的time_stop用profile或默认60
                # 实际上time_stop_minutes在check_exit里通过TIME_STOP_MINUTES读取
            except Exception:
                pass
        else:
            SYMBOL_PROFILES[sym][key] = val

    db = get_db()
    trades = run_day(sym, td, db, verbose=False)
    full = [t for t in trades if not t.get("partial")]
    pnl = sum(t['pnl_pts'] for t in full)
    n = len(full)
    return {'date': td, 'pnl': pnl, 'n': n}


def run_single_param_sweep(sym, dates, param_name, values):
    """对一个参数做单维扫描。"""
    n_workers = min(cpu_count(), 8)
    is_dates = set(dates[-219:])
    oos_dates = set(dates[:-219])

    results = []
    for val in values:
        # 构建覆盖dict：只改当前参数，其他保持新baseline
        overrides = {param_name: val}
        # 确保其他参数是新baseline值
        if param_name != 'trailing_stop_scale':
            overrides['trailing_stop_scale'] = CURRENT[sym]['trailing_stop_scale']
        if param_name != 'me_ratio':
            overrides['me_ratio'] = CURRENT[sym]['me_ratio']

        args_list = [(td, sym, overrides) for td in dates]
        with Pool(n_workers) as pool:
            day_results = pool.map(_run_one_day, args_list)

        is_pnl = sum(r['pnl'] for r in day_results if r['date'] in is_dates)
        oos_pnl = sum(r['pnl'] for r in day_results if r['date'] in oos_dates)
        is_n = sum(r['n'] for r in day_results if r['date'] in is_dates)
        oos_n = sum(r['n'] for r in day_results if r['date'] in oos_dates)
        is_daily = is_pnl / 219
        oos_daily = oos_pnl / 681
        ratio = is_daily / oos_daily if oos_daily > 0 else 99

        results.append({
            'param': param_name, 'value': val,
            'is_n': is_n, 'is_pnl': is_pnl, 'is_daily': is_daily,
            'oos_n': oos_n, 'oos_pnl': oos_pnl, 'oos_daily': oos_daily,
            'ratio': ratio, 'total_pnl': is_pnl + oos_pnl,
        })
        is_current = val == CURRENT[sym][param_name]
        mark = " ◀当前" if is_current else ""
        print(f"  {sym} {param_name}={val}: IS={is_pnl:+.0f} OOS={oos_pnl:+.0f} "
              f"ratio={ratio:.2f} total={is_pnl+oos_pnl:+.0f}{mark}")

    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("  活跃参数审计批次1: trailing_stop_scale + me_ratio + TIME_STOP")
    print("=" * 60)

    db = get_db()
    doc = ["# 活跃参数审计批次1\n"]
    doc.append("方法: 681/219双窗口单维扫描，每次只变一个参数，其他保持新baseline\n")
    doc.append(f"新baseline: IM(thr=60,sl=0.3%), IC(thr=55,sl=0.3%)\n")

    for param_name, values in PARAMS.items():
        doc.append(f"## {param_name}\n")

        for sym in ['IM', 'IC']:
            print(f"\n{'='*40}")
            print(f"  {sym} {param_name} sweep")
            print(f"{'='*40}")

            dates = get_dates(db, SPOTS[sym])
            rdf = run_single_param_sweep(sym, dates, param_name, values)

            # 找各类最优
            is_best = rdf.loc[rdf['is_pnl'].idxmax()]
            oos_best = rdf.loc[rdf['oos_pnl'].idxmax()]
            total_best = rdf.loc[rdf['total_pnl'].idxmax()]
            current_val = CURRENT[sym][param_name]
            current_row = rdf[rdf['value'] == current_val]

            doc.append(f"### {sym} (当前={current_val})\n")
            doc.append(f"| 值 | IS_PnL | IS日均 | OOS_PnL | OOS日均 | 合计 | 比值 |")
            doc.append(f"|-----|--------|--------|---------|---------|------|------|")
            for _, r in rdf.sort_values('total_pnl', ascending=False).iterrows():
                mark = ""
                if r['value'] == current_val: mark = " ◀当前"
                elif r['value'] == oos_best['value']: mark = " ★OOS最优"
                fmt_val = f"{r['value']:.2f}" if isinstance(r['value'], float) and r['value'] < 1 else str(r['value'])
                doc.append(f"| {fmt_val} | {r['is_pnl']:+.0f} | {r['is_daily']:+.1f} | "
                           f"{r['oos_pnl']:+.0f} | {r['oos_daily']:+.1f} | "
                           f"{r['total_pnl']:+.0f} | {r['ratio']:.2f} |{mark}")

            # 判定
            doc.append(f"\n**IS最优**: {is_best['value']} (IS={is_best['is_pnl']:+.0f})")
            doc.append(f"**OOS最优**: {oos_best['value']} (OOS={oos_best['oos_pnl']:+.0f})")
            doc.append(f"**综合最优**: {total_best['value']} (合计={total_best['total_pnl']:+.0f})")

            if len(current_row) > 0:
                cr = current_row.iloc[0]
                oos_gap = oos_best['oos_pnl'] - cr['oos_pnl']
                doc.append(f"**当前 vs OOS最优差距**: {oos_gap:+.0f}pt")
                if abs(oos_gap) < 200:
                    doc.append(f"→ 当前已接近最优，**不需要改动**")
                elif oos_gap > 200:
                    doc.append(f"→ OOS最优({oos_best['value']})显著优于当前({current_val})，**考虑切换**")
                else:
                    doc.append(f"→ 当前反而比OOS最优好，保持不变")
            doc.append("")

    # ═══════════════════════════════════════════════
    # 综合建议
    # ═══════════════════════════════════════════════
    doc.append("## 综合建议\n")
    doc.append("(根据以上数据给出每个参数的建议：保持/切换/需要更多研究)")

    report = "\n".join(doc)
    path = Path("tmp") / "param_audit_batch1.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
