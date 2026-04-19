#!/usr/bin/env python3
"""Score区间过滤研究: 只交易主力分数段的验证。"""
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

# 候选区间定义: (名称, IM区间列表, IC区间列表)
CANDIDATES = {
    'A': {'desc': '连续主力[60,80)', 'IM': [(60, 80)], 'IC': [(60, 80)]},
    'B': {'desc': '剔除W型异常', 'IM': [(60, 65), (70, 80)], 'IC': [(60, 65), (70, 80)]},
    'C': {'desc': '保守核心', 'IM': [(60, 75)], 'IC': [(70, 80)]},
    'D': {'desc': '极端聚焦', 'IM': [(60, 65)], 'IC': [(70, 75)]},
    'E': {'desc': '当前基线', 'IM': [(55, 100)], 'IC': [(60, 100)]},
    'F': {'desc': '排除高分段', 'IM': [(55, 85)], 'IC': [(60, 85)]},
    'G': {'desc': '仅排除负贡献', 'IM': [(55, 90), (95, 100)], 'IC': [(60, 65), (70, 100)]},
}


def get_dates(db, spot):
    df = db.query_df(
        f"SELECT DISTINCT substr(datetime,1,10) as d FROM index_min "
        f"WHERE symbol='{spot}' AND period=300 ORDER BY d"
    )
    dates = [d.replace('-', '') for d in df['d'].tolist()]
    return dates[-900:] if len(dates) > 900 else dates


def _run_one_day(args):
    td, sym, thr = args
    SYMBOL_PROFILES[sym]["signal_threshold"] = thr
    db = get_db()
    trades = run_day(sym, td, db, verbose=False)
    full = [t for t in trades if not t.get("partial")]
    for t in full:
        t['trade_date'] = td
        t['symbol'] = sym
    return full


def collect_trades(sym, dates, thr=45):
    n_workers = min(cpu_count(), 8)
    args_list = [(td, sym, thr) for td in dates]
    with Pool(n_workers) as pool:
        day_results = pool.map(_run_one_day, args_list)
    return [t for day in day_results for t in day]


def in_ranges(score, ranges):
    """检查score是否在任一区间内。"""
    return any(lo <= score < hi for lo, hi in ranges)


def filter_stats(trades_df, ranges, dates):
    """过滤后的统计。"""
    filtered = trades_df[trades_df['entry_score'].apply(lambda s: in_ranges(s, ranges))]
    if len(filtered) == 0:
        return {'pnl': 0, 'n': 0, 'avg': 0, 'sharpe': 0, 'wr': 0}

    # 按日聚合算sharpe
    filtered = filtered.copy()
    daily = filtered.groupby('trade_date')['pnl_pts'].sum()
    # 补上无交易的日子(pnl=0)
    all_days_pnl = pd.Series(0.0, index=dates)
    all_days_pnl.update(daily)
    daily_vals = all_days_pnl.values

    total_pnl = filtered['pnl_pts'].sum()
    n = len(filtered)
    avg = filtered['pnl_pts'].mean()
    wr = (filtered['pnl_pts'] > 0).sum() / n * 100
    daily_avg = np.mean(daily_vals)
    daily_std = np.std(daily_vals)
    sharpe = daily_avg / daily_std * np.sqrt(252) if daily_std > 0 else 0

    return {'pnl': total_pnl, 'n': n, 'avg': avg, 'sharpe': sharpe, 'wr': wr}


def main():
    print("=" * 60)
    print("  Score区间过滤研究")
    print("=" * 60)

    db = get_db()
    doc = ["# Score区间过滤研究\n"]

    # 收集全量trade数据
    all_trades = {}
    all_dates = {}
    for sym in ['IM', 'IC']:
        print(f"\n[{sym}] 收集交易数据...")
        dates = get_dates(db, SPOTS[sym])
        trades = collect_trades(sym, dates, thr=45)
        tdf = pd.DataFrame(trades)
        all_trades[sym] = tdf
        all_dates[sym] = dates
        print(f"  {len(tdf)}笔, {len(dates)}天")

    # ═══════════════════════════════════════════════
    # 第一步: 精细PnL拆解 + breakeven成本标注
    # ═══════════════════════════════════════════════
    doc.append("## 第一步: Score段PnL拆解 + Breakeven成本\n")

    for sym in ['IM', 'IC']:
        tdf = all_trades[sym]
        total_pnl = tdf['pnl_pts'].sum()

        doc.append(f"### {sym} ({len(tdf)}笔, 累计{total_pnl:+.0f}pt)\n")
        doc.append("| Score段 | 笔数 | AvgPnL | 累计PnL | 贡献% | 胜率 | BE成本 | 成本风险 |")
        doc.append("|---------|------|--------|---------|-------|------|--------|---------|")

        bins = list(range(55, 101, 5))
        bins.append(100)
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            if i == len(bins) - 2:
                sub = tdf[(tdf['entry_score'] >= lo) & (tdf['entry_score'] <= hi)]
                label = f"[{lo},{hi}]"
            else:
                sub = tdf[(tdf['entry_score'] >= lo) & (tdf['entry_score'] < hi)]
                label = f"[{lo},{hi})"
            if len(sub) == 0:
                continue
            avg = sub['pnl_pts'].mean()
            cum = sub['pnl_pts'].sum()
            contrib = cum / total_pnl * 100 if total_pnl != 0 else 0
            wr = (sub['pnl_pts'] > 0).sum() / len(sub) * 100
            risk = "安全" if avg >= 2.0 else ("边缘" if avg >= 0.5 else "**脆弱**")
            doc.append(f"| {label} | {len(sub)} | {avg:+.1f} | {cum:+.0f} | {contrib:+.0f}% | {wr:.0f}% | {avg:.1f} | {risk} |")
        doc.append("")

    # ═══════════════════════════════════════════════
    # 第二步: 候选区间设计
    # ═══════════════════════════════════════════════
    doc.append("## 第二步: 候选区间设计\n")
    doc.append("| 候选 | 描述 | IM区间 | IC区间 |")
    doc.append("|------|------|--------|--------|")
    for name, cfg in CANDIDATES.items():
        im_r = ', '.join(f'[{lo},{hi})' for lo, hi in cfg['IM'])
        ic_r = ', '.join(f'[{lo},{hi})' for lo, hi in cfg['IC'])
        doc.append(f"| {name} | {cfg['desc']} | {im_r} | {ic_r} |")
    doc.append("")

    # ═══════════════════════════════════════════════
    # 第三步: 全样本回测
    # ═══════════════════════════════════════════════
    doc.append("## 第三步: 全样本回测 (900天, 原始PnL不扣成本)\n")
    doc.append("| 候选 | IM_PnL | IM_笔数 | IM_单笔 | IM_Sharpe | IC_PnL | IC_笔数 | IC_单笔 | IC_Sharpe | 合计PnL |")
    doc.append("|------|--------|--------|--------|----------|--------|--------|--------|----------|---------|")

    full_results = {}
    for name, cfg in CANDIDATES.items():
        im_s = filter_stats(all_trades['IM'], cfg['IM'], all_dates['IM'])
        ic_s = filter_stats(all_trades['IC'], cfg['IC'], all_dates['IC'])
        total = im_s['pnl'] + ic_s['pnl']
        full_results[name] = {'IM': im_s, 'IC': ic_s, 'total': total}

        doc.append(f"| {name} | {im_s['pnl']:+.0f} | {im_s['n']} | {im_s['avg']:+.1f} | {im_s['sharpe']:.2f} | "
                   f"{ic_s['pnl']:+.0f} | {ic_s['n']} | {ic_s['avg']:+.1f} | {ic_s['sharpe']:.2f} | {total:+.0f} |")

    # 标注最优
    best_name = max(full_results, key=lambda k: full_results[k]['total'])
    baseline = full_results['E']['total']
    doc.append(f"\n**全样本最优: {best_name}** (合计{full_results[best_name]['total']:+.0f}pt, 基线E={baseline:+.0f}pt, "
               f"差异{full_results[best_name]['total'] - baseline:+.0f}pt)\n")

    # 排除基线后的top3
    non_baseline = {k: v for k, v in full_results.items() if k != 'E'}
    top3 = sorted(non_baseline, key=lambda k: non_baseline[k]['total'], reverse=True)[:3]
    doc.append(f"Top3候选(不含基线): {', '.join(top3)}\n")

    # ═══════════════════════════════════════════════
    # 第四步: IS/OOS验证
    # ═══════════════════════════════════════════════
    doc.append("## 第四步: IS/OOS验证\n")

    for sym in ['IM', 'IC']:
        dates = all_dates[sym]
        split = int(len(dates) * 2 / 3)
        is_dates = dates[:split]
        oos_dates = dates[split:]
        tdf = all_trades[sym]
        is_tdf = tdf[tdf['trade_date'].isin(set(is_dates))]
        oos_tdf = tdf[tdf['trade_date'].isin(set(oos_dates))]

        doc.append(f"### {sym} IS/OOS (IS={len(is_dates)}天, OOS={len(oos_dates)}天)\n")
        doc.append(f"| 候选 | IS_PnL | IS_笔数 | IS_单笔 | OOS_PnL | OOS_笔数 | OOS_单笔 | OOS改善vs基线 |")
        doc.append(f"|------|--------|--------|--------|---------|---------|---------|------------|")

        baseline_oos = None
        for name in ['E'] + top3:
            cfg = CANDIDATES[name]
            ranges = cfg[sym]
            is_s = filter_stats(is_tdf, ranges, is_dates)
            oos_s = filter_stats(oos_tdf, ranges, oos_dates)
            if name == 'E':
                baseline_oos = oos_s['pnl']
            oos_diff = oos_s['pnl'] - baseline_oos if baseline_oos is not None else 0
            doc.append(f"| {name} | {is_s['pnl']:+.0f} | {is_s['n']} | {is_s['avg']:+.1f} | "
                       f"{oos_s['pnl']:+.0f} | {oos_s['n']} | {oos_s['avg']:+.1f} | {oos_diff:+.0f} |")
        doc.append("")

    # ═══════════════════════════════════════════════
    # 第五步: 成本敏感性分析
    # ═══════════════════════════════════════════════
    doc.append("## 第五步: 成本敏感性分析\n")
    doc.append("最优候选 vs 基线E，不同成本假设下的IM+IC合计PnL:\n")

    costs = [0, 0.5, 0.8, 1.0, 1.5, 2.0]
    doc.append("| 候选 | " + " | ".join(f"cost={c}" for c in costs) + " |")
    doc.append("|------|" + "|".join("------" for _ in costs) + "|")

    for name in ['E', best_name] + [t for t in top3 if t != best_name][:1]:
        cfg = CANDIDATES[name]
        cells = []
        for cost in costs:
            im_s = filter_stats(all_trades['IM'], cfg['IM'], all_dates['IM'])
            ic_s = filter_stats(all_trades['IC'], cfg['IC'], all_dates['IC'])
            adj_pnl = (im_s['pnl'] - im_s['n'] * cost) + (ic_s['pnl'] - ic_s['n'] * cost)
            cells.append(f"{adj_pnl:+.0f}")
        doc.append(f"| {name} | " + " | ".join(cells) + " |")

    # 差异行
    cells_diff = []
    for cost in costs:
        best_cfg = CANDIDATES[best_name]
        base_cfg = CANDIDATES['E']
        im_best = filter_stats(all_trades['IM'], best_cfg['IM'], all_dates['IM'])
        ic_best = filter_stats(all_trades['IC'], best_cfg['IC'], all_dates['IC'])
        im_base = filter_stats(all_trades['IM'], base_cfg['IM'], all_dates['IM'])
        ic_base = filter_stats(all_trades['IC'], base_cfg['IC'], all_dates['IC'])
        best_adj = (im_best['pnl'] - im_best['n'] * cost) + (ic_best['pnl'] - ic_best['n'] * cost)
        base_adj = (im_base['pnl'] - im_base['n'] * cost) + (ic_base['pnl'] - ic_base['n'] * cost)
        cells_diff.append(f"{best_adj - base_adj:+.0f}")
    doc.append(f"| **差异** | " + " | ".join(cells_diff) + " |")

    doc.append(f"\n成本越高，过滤策略优势越大（因为笔数更少，成本节省更多）。\n")

    # ═══════════════════════════════════════════════
    # 第六步: 综合判定
    # ═══════════════════════════════════════════════
    doc.append("## 第六步: 综合判定\n")

    # 计算改进幅度
    best_total = full_results[best_name]['total']
    base_total = full_results['E']['total']
    improve_pct = (best_total - base_total) / base_total * 100 if base_total != 0 else 0

    doc.append(f"全样本最优候选: **{best_name}** ({CANDIDATES[best_name]['desc']})")
    doc.append(f"- 合计PnL: {best_total:+.0f} vs 基线{base_total:+.0f} (改善{improve_pct:+.1f}%)")
    doc.append(f"- IM: {full_results[best_name]['IM']['pnl']:+.0f} vs {full_results['E']['IM']['pnl']:+.0f}")
    doc.append(f"- IC: {full_results[best_name]['IC']['pnl']:+.0f} vs {full_results['E']['IC']['pnl']:+.0f}")

    if improve_pct >= 15:
        doc.append(f"\n**判定F1: 明显改进** ✓ (改善{improve_pct:+.1f}% >= 15%)")
        doc.append("需要IS/OOS验证确认稳定性后，推荐设计上线方案。")
    elif improve_pct >= 5:
        doc.append(f"\n**判定F2: 改进有限** ({improve_pct:+.1f}%)")
        doc.append("推荐做更精细的per-symbol优化。")
    else:
        doc.append(f"\n**判定F3: 没有可信改进** ({improve_pct:+.1f}%)")
        doc.append("Score过滤不是有效方向，回到原roadmap。")

    report = "\n".join(doc)
    path = Path("tmp") / "score_range_filter_research.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
