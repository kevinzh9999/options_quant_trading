#!/usr/bin/env python3
"""IC signal_threshold per-symbol 重验证。复用IM的方法论（2/3 IS + 1/3 OOS + MFE/MAE profiling）。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from data.storage.db_manager import get_db
from scripts.backtest_signals_day import run_day
from strategies.intraday.A_share_momentum_signal_v2 import SYMBOL_PROFILES


SYMBOL = 'IC'
SPOT = '000905'
THRESHOLDS = [45, 50, 55, 60, 65]


def get_dates(db):
    df = db.query_df(
        f"SELECT DISTINCT substr(datetime,1,10) as d FROM index_min "
        f"WHERE symbol='{SPOT}' AND period=300 ORDER BY d"
    )
    return [d.replace('-', '') for d in df['d'].tolist()]


def _run_one_day(args):
    """单日回测（供多进程调用）。args=(td, thr)"""
    td, thr = args
    # 子进程内设置threshold（fork后独立副本）
    SYMBOL_PROFILES[SYMBOL]["signal_threshold"] = thr
    db = get_db()
    trades = run_day(SYMBOL, td, db, verbose=False)
    full = [t for t in trades if not t.get("partial")]
    pnl = sum(t["pnl_pts"] for t in full)
    wins = sum(1 for t in full if t["pnl_pts"] > 0)
    return {'date': td, 'pnl': pnl, 'n': len(full), 'wins': wins, 'trades': full}


def run_threshold_scan(dates, thresholds, db):
    orig_thr = SYMBOL_PROFILES[SYMBOL].get("signal_threshold", 60)
    n_workers = min(cpu_count(), 8)
    results = []
    for thr in thresholds:
        SYMBOL_PROFILES[SYMBOL]["signal_threshold"] = thr
        args_list = [(td, thr) for td in dates]
        with Pool(n_workers) as pool:
            day_results = pool.map(_run_one_day, args_list)
        daily_pnl = [r['pnl'] for r in day_results]
        total_trades = sum(r['n'] for r in day_results)
        total_wins = sum(r['wins'] for r in day_results)
        total_pnl = sum(daily_pnl)
        wr = total_wins / total_trades * 100 if total_trades > 0 else 0
        avg = total_pnl / len(dates) if dates else 0
        std = np.std(daily_pnl) if daily_pnl else 1
        sharpe = avg / std * np.sqrt(252) if std > 0 else 0
        max_dd = _calc_max_dd(daily_pnl)
        results.append({
            'thr': thr, 'pnl': total_pnl, 'trades': total_trades,
            'wr': wr, 'avg_daily': avg, 'sharpe': sharpe,
            'days': len(dates), 'max_dd': max_dd,
        })
        print(f"  IC thr={thr} [{dates[0]}~{dates[-1]}]: {total_pnl:+.0f}pt "
              f"{total_trades}笔 WR={wr:.1f}% Sharpe={sharpe:.2f}")
    SYMBOL_PROFILES[SYMBOL]["signal_threshold"] = orig_thr
    return pd.DataFrame(results)


def _calc_max_dd(daily_pnls):
    if not daily_pnls:
        return 0
    cum = np.cumsum(daily_pnls)
    return float(np.max(np.maximum.accumulate(cum) - cum))


def _run_one_day_profiling(args):
    """单日回测+标注（供多进程调用，profiling用）。args=(td, thr)"""
    td, thr = args
    SYMBOL_PROFILES[SYMBOL]["signal_threshold"] = thr
    db = get_db()
    trades = run_day(SYMBOL, td, db, verbose=False)
    full = [t for t in trades if not t.get("partial")]
    for t in full:
        t['trade_date'] = td
        t['symbol'] = SYMBOL
    return full


def run_profiling(dates, db, bin_width=5, bin_min=45):
    """跑MFE/MAE profiling，返回per-bin统计。"""
    orig_thr = SYMBOL_PROFILES[SYMBOL].get("signal_threshold", 60)
    SYMBOL_PROFILES[SYMBOL]["signal_threshold"] = bin_min
    n_workers = min(cpu_count(), 8)
    args_list = [(td, bin_min) for td in dates]
    with Pool(n_workers) as pool:
        day_results = pool.map(_run_one_day_profiling, args_list)
    all_trades = [t for day in day_results for t in day]
    SYMBOL_PROFILES[SYMBOL]["signal_threshold"] = orig_thr

    if not all_trades:
        return pd.DataFrame()

    from scripts.entry_score_profiling import load_ohlcv, enrich_trades
    ohlcv = load_ohlcv(None, SYMBOL, dates[0], dates[-1])
    tdf = pd.DataFrame(all_trades)
    enriched = enrich_trades(tdf, ohlcv, n_bars_list=[24], amp_threshold=0.004)

    if len(enriched) == 0:
        return pd.DataFrame()

    bins = list(range(bin_min, 101, bin_width))
    if bins[-1] != 100:
        bins.append(100)
    labels = [f'[{bins[i]},{bins[i+1]})' for i in range(len(bins) - 2)] + [f'[{bins[-2]},{bins[-1]}]']
    enriched['score_bin'] = pd.cut(
        enriched['entry_score'], bins=bins, labels=labels,
        right=False, include_lowest=True
    )

    agg = enriched.groupby('score_bin', observed=True).agg(
        n=('entry_score', 'count'),
        med_mfe=('mfe_fixed_24', 'median'),
        med_mae=('mae_fixed_24', 'median'),
        avg_pnl=('pnl_pts', 'mean'),
        wr=('actual_win', 'mean'),
    ).reset_index()
    agg['mfe_mae'] = agg['med_mfe'] / agg['med_mae'].replace(0, np.nan)
    agg = agg[agg['n'] > 0]
    return agg


def main():
    print("=" * 60)
    print("  IC signal_threshold per-symbol 重验证")
    print("=" * 60)

    db = get_db()
    all_dates_full = get_dates(db)
    # 只取最近900天（与IM方法论一致）
    all_dates = all_dates_full[-900:] if len(all_dates_full) > 900 else all_dates_full
    n = len(all_dates)
    print(f"IC数据: {all_dates[0]}~{all_dates[-1]} ({n}天, 取最近900天)")

    # IS/OOS切分: 2/3 IS + 1/3 OOS (与IM方法论一致)
    split = int(n * 2 / 3)
    is_dates = all_dates[:split]
    oos_dates = all_dates[split:]
    # 最近30天sanity check
    last30 = all_dates[-30:]

    doc = ["# IC signal_threshold per-symbol 重验证\n"]
    doc.append(f"方法论: 复用IM验证流程 (2/3 IS + 1/3 OOS + MFE/MAE profiling)")
    doc.append(f"数据: IC {n}天 ({all_dates[0]}~{all_dates[-1]})")
    doc.append(f"IS: {len(is_dates)}天 ({is_dates[0]}~{is_dates[-1]})")
    doc.append(f"OOS: {len(oos_dates)}天 ({oos_dates[0]}~{oos_dates[-1]})")
    doc.append(f"最近30天: {last30[0]}~{last30[-1]}\n")

    # ═══════════════════════════════════════════════
    # 第一步: IS段 threshold sweep
    # ═══════════════════════════════════════════════
    print(f"\n[第一步] IS段 ({len(is_dates)}天) threshold sweep...")
    is_results = run_threshold_scan(is_dates, THRESHOLDS, db)
    is_best_thr = int(is_results.loc[is_results['pnl'].idxmax(), 'thr'])

    doc.append("## 第一步: IS段 Threshold Sweep\n")
    doc.append(f"| thr | PnL | 笔数 | WR | Sharpe | MaxDD |")
    doc.append(f"|-----|-----|------|-----|--------|-------|")
    for _, r in is_results.sort_values('pnl', ascending=False).iterrows():
        mark = " ◀最优" if int(r['thr']) == is_best_thr else ""
        doc.append(f"| {int(r['thr'])} | {r['pnl']:+.0f} | {int(r['trades'])} | "
                   f"{r['wr']:.1f}% | {r['sharpe']:.2f} | {r['max_dd']:.0f} |{mark}")
    doc.append(f"\n**IS最优: thr={is_best_thr}**\n")

    # ═══════════════════════════════════════════════
    # 第二步: MFE/MAE Profiling
    # ═══════════════════════════════════════════════
    print(f"\n[第二步] IS段 MFE/MAE Profiling...")
    prof = run_profiling(is_dates, db)

    doc.append("## 第二步: MFE/MAE Profiling (IS段)\n")
    if len(prof) > 0:
        doc.append(f"| Bin | N | Med_MFE | Med_MAE | MFE/MAE | Avg_PnL | WR |")
        doc.append(f"|-----|---|---------|---------|---------|---------|-----|")
        for _, r in prof.iterrows():
            expect = "正" if r['mfe_mae'] >= 1.0 else "**负**"
            doc.append(f"| {r['score_bin']} | {int(r['n'])} | {r['med_mfe']:.1f} | "
                       f"{r['med_mae']:.1f} | {r['mfe_mae']:.2f} | {r['avg_pnl']:+.1f} | "
                       f"{r['wr']*100:.0f}% | {expect}")
    else:
        doc.append("Profiling无数据")

    # 分析边界区间
    doc.append("\n### 边界区间分析\n")
    if len(prof) > 0:
        for _, r in prof.iterrows():
            bin_str = str(r['score_bin'])
            if r['mfe_mae'] < 1.0:
                doc.append(f"- {bin_str}: MFE/MAE={r['mfe_mae']:.2f} **负期望** — 应该过滤")
            elif r['mfe_mae'] < 1.2:
                doc.append(f"- {bin_str}: MFE/MAE={r['mfe_mae']:.2f} 边缘正期望")
            else:
                doc.append(f"- {bin_str}: MFE/MAE={r['mfe_mae']:.2f} 正期望 ✓")

    # ═══════════════════════════════════════════════
    # 第三步: OOS验证
    # ═══════════════════════════════════════════════
    print(f"\n[第三步] OOS段 ({len(oos_dates)}天) 验证...")
    # OOS只验证IS最优 + 当前值(60) + 邻近值
    oos_thrs = sorted(set([is_best_thr, 60] + [t for t in [is_best_thr - 5, is_best_thr + 5] if 45 <= t <= 65]))
    oos_results = run_threshold_scan(oos_dates, oos_thrs, db)

    doc.append(f"\n## 第三步: OOS验证\n")
    doc.append(f"IS最优thr={is_best_thr}, 验证: {oos_thrs}\n")
    doc.append(f"| thr | PnL | 笔数 | WR | Sharpe | MaxDD |")
    doc.append(f"|-----|-----|------|-----|--------|-------|")
    for _, r in oos_results.sort_values('pnl', ascending=False).iterrows():
        doc.append(f"| {int(r['thr'])} | {r['pnl']:+.0f} | {int(r['trades'])} | "
                   f"{r['wr']:.1f}% | {r['sharpe']:.2f} | {r['max_dd']:.0f} |")

    # IS vs OOS对比
    doc.append(f"\n### IS vs OOS对比 (thr={is_best_thr})\n")
    is_row = is_results[is_results['thr'] == is_best_thr].iloc[0]
    oos_row = oos_results[oos_results['thr'] == is_best_thr]
    if len(oos_row) > 0:
        oos_row = oos_row.iloc[0]
        doc.append(f"| 指标 | IS ({int(is_row['days'])}天) | OOS ({int(oos_row['days'])}天) |")
        doc.append(f"|------|-----|-----|")
        doc.append(f"| PnL | {is_row['pnl']:+.0f} | {oos_row['pnl']:+.0f} |")
        doc.append(f"| 笔数 | {int(is_row['trades'])} | {int(oos_row['trades'])} |")
        doc.append(f"| WR | {is_row['wr']:.1f}% | {oos_row['wr']:.1f}% |")
        doc.append(f"| Sharpe | {is_row['sharpe']:.2f} | {oos_row['sharpe']:.2f} |")
        doc.append(f"| MaxDD | {is_row['max_dd']:.0f} | {oos_row['max_dd']:.0f} |")

    # vs 当前值60对比
    oos_60 = oos_results[oos_results['thr'] == 60]
    oos_best = oos_results[oos_results['thr'] == is_best_thr]
    if len(oos_60) > 0 and len(oos_best) > 0 and is_best_thr != 60:
        diff_pnl = float(oos_best['pnl'].iloc[0] - oos_60['pnl'].iloc[0])
        diff_sharpe = float(oos_best['sharpe'].iloc[0] - oos_60['sharpe'].iloc[0])
        doc.append(f"\nthr={is_best_thr} vs thr=60 OOS差: PnL={diff_pnl:+.0f}pt Sharpe={diff_sharpe:+.2f}")

    # ═══════════════════════════════════════════════
    # 第四步: 最近30天sanity check
    # ═══════════════════════════════════════════════
    print(f"\n[第四步] 最近30天 sanity check...")
    sanity_thrs = sorted(set([is_best_thr, 60]))
    sanity_results = run_threshold_scan(last30, sanity_thrs, db)

    doc.append(f"\n## 第四步: 最近30天 Sanity Check\n")
    doc.append(f"| thr | PnL | 笔数 | WR |")
    doc.append(f"|-----|-----|------|-----|")
    for _, r in sanity_results.iterrows():
        doc.append(f"| {int(r['thr'])} | {r['pnl']:+.0f} | {int(r['trades'])} | {r['wr']:.1f}% |")

    # ═══════════════════════════════════════════════
    # 第五步: 跟IM的对比
    # ═══════════════════════════════════════════════
    doc.append(f"\n## 第五步: 跟IM(thr=55)的品种差异分析\n")
    doc.append(f"| 品种 | 最优thr | 当前thr | 品种特征 |")
    doc.append(f"|------|--------|--------|---------|")
    doc.append(f"| IM | 55 | 55(已上线) | 动量型,波动大 |")
    doc.append(f"| IC | {is_best_thr} | 60(待验证) | 动量型,节奏慢 |")

    if is_best_thr < 60:
        doc.append(f"\nIC最优{is_best_thr}<当前60: IC的边界区间[{is_best_thr},60)可能包含负期望交易")
    elif is_best_thr == 60:
        doc.append(f"\nIC最优=当前60: IC的60已经是合理值,不需要调整")
    else:
        doc.append(f"\nIC最优{is_best_thr}>当前60: IC需要更严格的信号过滤")

    # ═══════════════════════════════════════════════
    # 综合判定
    # ═══════════════════════════════════════════════
    doc.append(f"\n## 综合判定\n")

    # 判定逻辑
    if len(oos_best) > 0 and len(oos_60) > 0 and is_best_thr != 60:
        oos_best_pnl = float(oos_best['pnl'].iloc[0])
        oos_60_pnl = float(oos_60['pnl'].iloc[0])
        oos_best_sharpe = float(oos_best['sharpe'].iloc[0])

        if oos_best_pnl > oos_60_pnl and oos_best_sharpe > 0:
            doc.append(f"**判定IC1: IC应该改为thr={is_best_thr}** ✓")
            doc.append(f"- IS最优: thr={is_best_thr}")
            doc.append(f"- OOS验证: thr={is_best_thr} PnL={oos_best_pnl:+.0f} > thr=60 PnL={oos_60_pnl:+.0f}")
            doc.append(f"- 建议上线新threshold={is_best_thr}")
        elif oos_best_pnl <= oos_60_pnl:
            doc.append(f"**判定IC2: IC当前60已经是最优** ✓")
            doc.append(f"- IS最优thr={is_best_thr}在OOS上不如thr=60")
            doc.append(f"- 不改动IC threshold")
        else:
            doc.append(f"**判定IC3: 边缘情况**")
            doc.append(f"- IS最优thr={is_best_thr}，OOS表现混合")
    elif is_best_thr == 60:
        doc.append(f"**判定IC2: IC当前60已经是IS最优** ✓")
        doc.append(f"- IS段60就是最优,不需要调整")
    else:
        doc.append(f"**判定IC3: 数据不足以判定**")

    report = "\n".join(doc)
    path = Path("tmp") / "ic_signal_threshold_revalidation.md"
    with open(path, 'w') as f:
        f.write(report)
    print(f"\n报告已保存: {path}")
    print(report)


if __name__ == "__main__":
    main()
