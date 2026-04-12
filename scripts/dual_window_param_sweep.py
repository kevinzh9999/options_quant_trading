#!/usr/bin/env python3
"""IC/IM双窗口参数扫描：鲁棒性vs拟合性诊断。"""
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
THRESHOLDS = [45, 50, 55, 60, 65, 70]
STOP_LOSSES = [0.003, 0.004, 0.005, 0.006, 0.007]


def get_dates(db, spot):
    df = db.query_df(
        f"SELECT DISTINCT substr(datetime,1,10) as d FROM index_min "
        f"WHERE symbol='{spot}' AND period=300 ORDER BY d"
    )
    dates = [d.replace('-', '') for d in df['d'].tolist()]
    return dates[-900:] if len(dates) > 900 else dates


def _run_one_day(args):
    td, sym, thr, sl = args
    SYMBOL_PROFILES[sym]["signal_threshold"] = thr
    SYMBOL_PROFILES[sym]["stop_loss_pct"] = sl
    db = get_db()
    trades = run_day(sym, td, db, verbose=False)
    full = [t for t in trades if not t.get("partial")]
    result = {'date': td, 'pnl': sum(t['pnl_pts'] for t in full), 'n': len(full)}
    return result


def run_sweep_for_symbol(sym, dates):
    """对一个品种跑30个组合的全量回测。"""
    n_workers = min(cpu_count(), 8)
    is_dates = set(dates[-219:])
    oos_dates = set(dates[:-219])

    results = []
    for thr in THRESHOLDS:
        for sl in STOP_LOSSES:
            print(f"  {sym} thr={thr} sl={sl*100:.1f}%...", end=" ", flush=True)
            args_list = [(td, sym, thr, sl) for td in dates]
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
                'thr': thr, 'sl': sl, 'sl_pct': f"{sl*100:.1f}%",
                'is_n': is_n, 'is_pnl': is_pnl, 'is_daily': is_daily,
                'oos_n': oos_n, 'oos_pnl': oos_pnl, 'oos_daily': oos_daily,
                'ratio': ratio, 'total_pnl': is_pnl + oos_pnl,
            })
            print(f"IS={is_pnl:+.0f} OOS={oos_pnl:+.0f} ratio={ratio:.2f}")

    return pd.DataFrame(results)


def identify_key_combos(rdf, sym, current_thr, current_sl):
    """识别关键组合A/B/C/D/E。"""
    combos = {}

    # A: IS最优
    a = rdf.loc[rdf['is_pnl'].idxmax()]
    combos['A_IS最优'] = a

    # B: OOS最优
    b = rdf.loc[rdf['oos_pnl'].idxmax()]
    combos['B_OOS最优'] = b

    # C: 综合最优(IS+OOS)
    c = rdf.loc[rdf['total_pnl'].idxmax()]
    combos['C_综合最优'] = c

    # D: 鲁棒平衡(ratio最接近1.5)
    rdf_valid = rdf[rdf['oos_daily'] > 0]
    if len(rdf_valid) > 0:
        d = rdf_valid.loc[(rdf_valid['ratio'] - 1.5).abs().idxmin()]
        combos['D_鲁棒平衡'] = d

    # E: 当前
    e = rdf[(rdf['thr'] == current_thr) & (rdf['sl'] == current_sl)]
    if len(e) > 0:
        combos['E_当前'] = e.iloc[0]

    return combos


def main():
    print("=" * 60)
    print("  IC/IM 双窗口参数扫描")
    print("=" * 60)

    db = get_db()
    doc = ["# IC/IM 双窗口参数扫描 - 鲁棒性诊断\n"]

    all_results = {}

    for sym in ['IM', 'IC']:
        print(f"\n{'='*40}")
        print(f"  {sym} 30组合扫描")
        print(f"{'='*40}")
        dates = get_dates(db, SPOTS[sym])
        rdf = run_sweep_for_symbol(sym, dates)
        all_results[sym] = rdf

        # 完整结果表
        doc.append(f"## {sym} 完整结果 (30组合)\n")
        doc.append("| thr | sl | IS笔数 | IS_PnL | IS日均 | OOS笔数 | OOS_PnL | OOS日均 | 比值 |")
        doc.append("|-----|------|--------|--------|--------|---------|---------|---------|------|")
        for _, r in rdf.sort_values('total_pnl', ascending=False).iterrows():
            doc.append(f"| {int(r['thr'])} | {r['sl_pct']} | {int(r['is_n'])} | {r['is_pnl']:+.0f} | "
                       f"{r['is_daily']:+.1f} | {int(r['oos_n'])} | {r['oos_pnl']:+.0f} | "
                       f"{r['oos_daily']:+.1f} | {r['ratio']:.2f} |")
        doc.append("")

    # ═══════════════════════════════════════════════
    # 关键组合对比
    # ═══════════════════════════════════════════════
    doc.append("## 关键组合对比\n")

    current = {'IM': (55, 0.003), 'IC': (60, 0.005)}

    for sym in ['IM', 'IC']:
        rdf = all_results[sym]
        ct, cs = current[sym]
        combos = identify_key_combos(rdf, sym, ct, cs)

        doc.append(f"### {sym}\n")
        doc.append("| 类型 | thr | sl | IS_PnL | OOS_PnL | 合计 | 比值 |")
        doc.append("|------|-----|------|--------|---------|------|------|")
        for label, r in combos.items():
            doc.append(f"| {label} | {int(r['thr'])} | {r['sl_pct']} | {r['is_pnl']:+.0f} | "
                       f"{r['oos_pnl']:+.0f} | {r['total_pnl']:+.0f} | {r['ratio']:.2f} |")
        doc.append("")

    # ═══════════════════════════════════════════════
    # 交叉验证
    # ═══════════════════════════════════════════════
    doc.append("## 交叉验证\n")

    for sym in ['IM', 'IC']:
        rdf = all_results[sym]
        ct, cs = current[sym]
        combos = identify_key_combos(rdf, sym, ct, cs)

        doc.append(f"### {sym}\n")

        # OOS最优参数在IS上的表现
        b = combos['B_OOS最优']
        a = combos['A_IS最优']
        e = combos.get('E_当前')

        doc.append(f"- IS最优组合: thr={int(a['thr'])},sl={a['sl_pct']} → IS={a['is_pnl']:+.0f}")
        doc.append(f"- OOS最优组合: thr={int(b['thr'])},sl={b['sl_pct']} → IS={b['is_pnl']:+.0f}, OOS={b['oos_pnl']:+.0f}")
        if e is not None:
            doc.append(f"- 当前组合: thr={int(e['thr'])},sl={e['sl_pct']} → IS={e['is_pnl']:+.0f}, OOS={e['oos_pnl']:+.0f}")

        # IS最优在IS上 vs OOS最优在IS上的差距
        is_gap = a['is_pnl'] - b['is_pnl']
        doc.append(f"- IS红利(IS最优-OOS最优在IS上): {is_gap:+.0f}pt")
        doc.append(f"- 放弃IS红利后的合理IS预期: {b['is_pnl']:+.0f}pt ({b['is_pnl']/219:+.1f}pt/天)\n")

    # ═══════════════════════════════════════════════
    # 参数维度稳健性
    # ═══════════════════════════════════════════════
    doc.append("## 参数维度稳健性\n")

    for sym in ['IM', 'IC']:
        rdf = all_results[sym]
        combos = identify_key_combos(rdf, sym, *current[sym])
        a = combos['A_IS最优']
        b = combos['B_OOS最优']

        doc.append(f"### {sym}\n")
        doc.append(f"- IS最优: thr={int(a['thr'])}, sl={a['sl_pct']}")
        doc.append(f"- OOS最优: thr={int(b['thr'])}, sl={b['sl_pct']}")

        thr_diff = abs(int(a['thr']) - int(b['thr']))
        sl_diff = abs(a['sl'] - b['sl']) * 100

        if thr_diff <= 5 and sl_diff <= 0.1:
            doc.append(f"- **偏离小**: threshold差{thr_diff}, sl差{sl_diff:.1f}% → 参数稳定")
        elif thr_diff <= 10 and sl_diff <= 0.2:
            doc.append(f"- **偏离适中**: threshold差{thr_diff}, sl差{sl_diff:.1f}%")
        else:
            doc.append(f"- **偏离大**: threshold差{thr_diff}, sl差{sl_diff:.1f}% → 参数不稳定")

        # 偏离方向解读
        if int(a['thr']) < int(b['thr']):
            doc.append(f"  - IS偏好更低threshold(更激进入场) → IS段市场对信号更友好")
        elif int(a['thr']) > int(b['thr']):
            doc.append(f"  - IS偏好更高threshold(更挑剔) → IS段需要更强信号才赚钱")
        if a['sl'] < b['sl']:
            doc.append(f"  - IS偏好更紧止损 → IS段反向少，紧止损不误杀")
        elif a['sl'] > b['sl']:
            doc.append(f"  - IS偏好更宽止损 → IS段反向多或波动大")
        doc.append("")

    # ═══════════════════════════════════════════════
    # 综合判定
    # ═══════════════════════════════════════════════
    doc.append("## 综合判定\n")

    for sym in ['IM', 'IC']:
        rdf = all_results[sym]
        combos = identify_key_combos(rdf, sym, *current[sym])
        a = combos['A_IS最优']
        b = combos['B_OOS最优']
        e = combos.get('E_当前')

        thr_diff = abs(int(a['thr']) - int(b['thr']))
        sl_diff = abs(a['sl'] - b['sl']) * 100

        doc.append(f"### {sym}\n")

        # 当前组合vs OOS最优
        if e is not None:
            oos_gap = abs(e['oos_pnl'] - b['oos_pnl'])
            oos_gap_pct = oos_gap / abs(b['oos_pnl']) * 100 if b['oos_pnl'] != 0 else 0

            if oos_gap_pct < 5:
                doc.append(f"**判定P1**: 当前参数接近OOS最优(差{oos_gap_pct:.0f}%) — 不过拟合")
            elif thr_diff <= 5 and sl_diff <= 0.1:
                doc.append(f"**判定P2**: IS/OOS最优差异适中 — 适度过拟合")
            elif thr_diff > 10 or sl_diff > 0.2:
                doc.append(f"**判定P3**: IS/OOS最优差异大 — 严重过拟合或市场质变")
            else:
                doc.append(f"**判定P2**: IS/OOS最优差异适中")
        doc.append("")

    report = "\n".join(doc)
    path = Path("tmp") / "dual_window_param_sweep.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
