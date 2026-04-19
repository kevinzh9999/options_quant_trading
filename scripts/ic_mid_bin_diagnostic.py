#!/usr/bin/env python3
"""IC [55,60) 中分段塌陷诊断：IS vs OOS对比 + 描述性统计。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from data.storage.db_manager import get_db
from scripts.backtest_signals_day import run_day
from strategies.intraday.A_share_momentum_signal_v2 import SYMBOL_PROFILES

SYMBOL = 'IC'
SPOT = '000905'


def get_dates(db):
    df = db.query_df(
        f"SELECT DISTINCT substr(datetime,1,10) as d FROM index_min "
        f"WHERE symbol='{SPOT}' AND period=300 ORDER BY d"
    )
    return [d.replace('-', '') for d in df['d'].tolist()]


def _run_one_day(args):
    td, thr = args
    SYMBOL_PROFILES[SYMBOL]["signal_threshold"] = thr
    db = get_db()
    trades = run_day(SYMBOL, td, db, verbose=False)
    full = [t for t in trades if not t.get("partial")]
    for t in full:
        t['trade_date'] = td
        t['symbol'] = SYMBOL
    return full


def collect_trades(dates, thr=45):
    """收集所有交易（用低threshold抓全量信号）。"""
    n_workers = min(cpu_count(), 8)
    args_list = [(td, thr) for td in dates]
    with Pool(n_workers) as pool:
        day_results = pool.map(_run_one_day, args_list)
    all_trades = [t for day in day_results for t in day]
    return all_trades


def do_profiling(trades, dates, bin_width=5, bin_min=50):
    """对交易做MFE/MAE profiling。"""
    if not trades:
        return pd.DataFrame()

    from scripts.entry_score_profiling import load_ohlcv, enrich_trades
    ohlcv = load_ohlcv(None, SYMBOL, dates[0], dates[-1])
    tdf = pd.DataFrame(trades)
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
    return agg, enriched


def main():
    print("=" * 60)
    print("  IC [55,60) 中分段塌陷诊断")
    print("=" * 60)

    db = get_db()
    all_dates_full = get_dates(db)
    all_dates = all_dates_full[-900:] if len(all_dates_full) > 900 else all_dates_full
    n = len(all_dates)

    split = int(n * 2 / 3)
    is_dates = all_dates[:split]
    oos_dates = all_dates[split:]

    doc = ["# IC [55,60) 中分段塌陷诊断\n"]
    doc.append(f"数据: IC {n}天 ({all_dates[0]}~{all_dates[-1]})")
    doc.append(f"IS: {len(is_dates)}天, OOS: {len(oos_dates)}天\n")

    # ═══════════════════════════════════════════════
    # 第一步: IS vs OOS 对比 profiling
    # ═══════════════════════════════════════════════
    print("[第一步] 收集IS段交易...")
    is_trades = collect_trades(is_dates, thr=45)
    print(f"  IS: {len(is_trades)}笔")

    print("[第一步] 收集OOS段交易...")
    oos_trades = collect_trades(oos_dates, thr=45)
    print(f"  OOS: {len(oos_trades)}笔")

    print("[第一步] IS profiling...")
    is_prof, is_enriched = do_profiling(is_trades, is_dates)
    print("[第一步] OOS profiling...")
    oos_prof, oos_enriched = do_profiling(oos_trades, oos_dates)

    doc.append("## 第一步: IS vs OOS 对比 Profiling\n")
    doc.append("| Bin | IS_N | IS_MFE/MAE | IS_AvgPnL | OOS_N | OOS_MFE/MAE | OOS_AvgPnL | 一致? |")
    doc.append("|-----|------|-----------|-----------|-------|-------------|-----------|------|")

    is_dict = {str(r['score_bin']): r for _, r in is_prof.iterrows()}
    oos_dict = {str(r['score_bin']): r for _, r in oos_prof.iterrows()}
    all_bins = sorted(set(list(is_dict.keys()) + list(oos_dict.keys())))

    target_bin_is_neg = False
    target_bin_oos_neg = False

    for b in all_bins:
        is_r = is_dict.get(b)
        oos_r = oos_dict.get(b)
        is_n = int(is_r['n']) if is_r is not None else 0
        is_mm = f"{is_r['mfe_mae']:.2f}" if is_r is not None else "-"
        is_pnl = f"{is_r['avg_pnl']:+.1f}" if is_r is not None else "-"
        oos_n = int(oos_r['n']) if oos_r is not None else 0
        oos_mm = f"{oos_r['mfe_mae']:.2f}" if oos_r is not None else "-"
        oos_pnl = f"{oos_r['avg_pnl']:+.1f}" if oos_r is not None else "-"

        # 一致性判断
        if is_r is not None and oos_r is not None:
            both_pos = is_r['mfe_mae'] >= 1.0 and oos_r['mfe_mae'] >= 1.0
            both_neg = is_r['mfe_mae'] < 1.0 and oos_r['mfe_mae'] < 1.0
            consistent = "✓" if both_pos or both_neg else "✗"
        else:
            consistent = "-"

        mark = " ◀" if '[55,60)' in b else ""
        doc.append(f"| {b} | {is_n} | {is_mm} | {is_pnl} | {oos_n} | {oos_mm} | {oos_pnl} | {consistent} |{mark}")

        if '[55,60)' in b:
            if is_r is not None and is_r['mfe_mae'] < 1.0:
                target_bin_is_neg = True
            if oos_r is not None and oos_r['mfe_mae'] < 1.0:
                target_bin_oos_neg = True

    # 第一步判定
    doc.append(f"\n### 第一步判定\n")
    if target_bin_is_neg and target_bin_oos_neg:
        doc.append("**结构性确认** ✓: [55,60) 在IS和OOS上都是负期望（MFE/MAE < 1.0）")
        doc.append("进入第二步描述性统计。\n")
        do_step2 = True
    elif target_bin_is_neg and not target_bin_oos_neg:
        oos_55 = oos_dict.get('[55,60)')
        if oos_55 is not None and oos_55['mfe_mae'] < 1.10:
            doc.append("**边缘情况**: IS负期望, OOS中性（MFE/MAE在0.95-1.10之间）")
            doc.append("仍进入第二步做初步分析。\n")
            do_step2 = True
        else:
            doc.append("**统计噪音**: IS负期望但OOS正期望（MFE/MAE >= 1.1），现象未在OOS验证")
            doc.append("研究到此结束，不值得深挖。\n")
            do_step2 = False
    else:
        doc.append("**统计噪音**: 现象在IS段不存在或OOS未验证")
        do_step2 = False

    if do_step2:
        # ═══════════════════════════════════════════════
        # 第二步: 描述性统计
        # ═══════════════════════════════════════════════
        doc.append("## 第二步: [55,60) 区间描述性统计\n")

        # 合并IS+OOS的enriched数据
        all_enriched = pd.concat([is_enriched, oos_enriched], ignore_index=True)
        target = all_enriched[
            (all_enriched['entry_score'] >= 55) & (all_enriched['entry_score'] < 60)
        ].copy()
        full_sample = all_enriched.copy()

        doc.append(f"[55,60) 样本: {len(target)}笔 (全样本: {len(full_sample)}笔)\n")

        # 2.1 时段分布
        doc.append("### 2.1 时段分布\n")
        if 'entry_time' in target.columns:
            def get_session(t):
                if hasattr(t, 'hour'):
                    h, m = t.hour, t.minute
                else:
                    try:
                        ts = pd.to_datetime(str(t))
                        h, m = ts.hour, ts.minute
                    except Exception:
                        return 'unknown'
                # UTC → BJ sessions
                if h == 1 or (h == 2 and m < 30):
                    return '09:35-10:30'
                elif (h == 2 and m >= 30) or h == 3:
                    return '10:30-11:30'
                elif h == 5 and m < 30:
                    return '13:00-13:30'
                elif (h == 5 and m >= 30) or (h == 6 and m < 30):
                    return '13:30-14:30'
                elif h == 6 and m >= 30:
                    return '14:30-14:50'
                else:
                    return 'other'

            target['session'] = target['entry_time'].apply(get_session)
            full_sample['session_full'] = full_sample['entry_time'].apply(get_session)

            doc.append("| 时段 | [55,60)笔数 | [55,60)占比 | [55,60)AvgPnL | 全样本占比 |")
            doc.append("|------|-----------|-----------|-------------|----------|")
            sessions = ['09:35-10:30', '10:30-11:30', '13:00-13:30', '13:30-14:30', '14:30-14:50']
            for s in sessions:
                t_sub = target[target['session'] == s]
                f_sub = full_sample[full_sample['session_full'] == s]
                if len(t_sub) >= 5:
                    doc.append(f"| {s} | {len(t_sub)} | {len(t_sub)/len(target)*100:.0f}% | "
                               f"{t_sub['pnl_pts'].mean():+.1f} | {len(f_sub)/len(full_sample)*100:.0f}% |")
            doc.append("")

        # 2.2 多空比例
        doc.append("### 2.2 多空比例\n")
        if 'direction' in target.columns:
            dir_col = 'direction'
        elif 'entry_direction' in target.columns:
            dir_col = 'entry_direction'
        else:
            dir_col = None

        if dir_col:
            doc.append(f"| 方向 | [55,60)笔数 | [55,60)占比 | [55,60)AvgPnL | [55,60)WR | 全样本占比 |")
            doc.append(f"|------|-----------|-----------|-------------|----------|----------|")
            for d in target[dir_col].unique():
                t_sub = target[target[dir_col] == d]
                f_sub = full_sample[full_sample[dir_col] == d]
                wr = (t_sub['pnl_pts'] > 0).sum() / len(t_sub) * 100 if len(t_sub) > 0 else 0
                doc.append(f"| {d} | {len(t_sub)} | {len(t_sub)/len(target)*100:.0f}% | "
                           f"{t_sub['pnl_pts'].mean():+.1f} | {wr:.0f}% | {len(f_sub)/len(full_sample)*100:.0f}% |")
            doc.append("")
        else:
            doc.append("direction列不存在，跳过多空分析\n")

        # 2.3 子分量分布
        doc.append("### 2.3 子分量分布\n")
        score_cols = ['entry_m_score', 'entry_v_score', 'entry_q_score', 'entry_b_score', 'entry_s_score']
        available_cols = [c for c in score_cols if c in target.columns]
        if available_cols:
            doc.append(f"| 子分量 | [55,60)均值 | [55,60)中位数 | 全样本均值 | 差异 |")
            doc.append(f"|--------|-----------|------------|----------|------|")
            for col in available_cols:
                t_mean = target[col].mean()
                t_med = target[col].median()
                f_mean = full_sample[col].mean()
                doc.append(f"| {col.replace('entry_','')} | {t_mean:.1f} | {t_med:.1f} | {f_mean:.1f} | {t_mean-f_mean:+.1f} |")
            doc.append("")

            # 2.4 V分特殊关注
            if 'entry_v_score' in target.columns:
                doc.append("### 2.4 V分(regime选择器)分组\n")
                doc.append("| V分组 | 笔数 | AvgPnL | WR |")
                doc.append("|-------|------|--------|-----|")
                for lo, hi, label in [(0, 10, 'V<=10'), (10, 20, 'V 10-20'), (20, 31, 'V 20-30')]:
                    sub = target[(target['entry_v_score'] >= lo) & (target['entry_v_score'] < hi)]
                    if len(sub) >= 10:
                        wr = (sub['pnl_pts'] > 0).sum() / len(sub) * 100
                        doc.append(f"| {label} | {len(sub)} | {sub['pnl_pts'].mean():+.1f} | {wr:.0f}% |")
                    elif len(sub) > 0:
                        doc.append(f"| {label} | {len(sub)} | (样本<10,不解读) | |")
                doc.append("")
        else:
            doc.append("子分量列不存在，跳过分量分析\n")

        # 2.5 当日gap分布（如果有数据）
        # gap需要从bar数据算，这里简化跳过
        doc.append("### 2.5 Gap分布\n")
        doc.append("(需要额外加载日线数据计算gap，本次诊断跳过。如需要可在后续研究中补充。)\n")

    # ═══════════════════════════════════════════════
    # 第三步: 综合解读
    # ═══════════════════════════════════════════════
    doc.append("## 第三步: 综合解读\n")

    if not do_step2:
        doc.append("第一步判定为统计噪音，不做进一步分析。")
        doc.append("\n**结论**: [55,60)的负期望在OOS未验证，属于IS段的统计噪音。")
        doc.append("IC的信号质量分布在OOS上可能是单调的，中分段塌陷不是结构性问题。")
        doc.append("\n**建议**: 不需要针对[55,60)做任何策略修改，直接进入下一个roadmap项目。")
    else:
        doc.append("(根据第二步的具体数据，在此给出最佳假设解释。)")
        # 动态判断
        if available_cols and 'entry_v_score' in target.columns:
            v_mean_target = target['entry_v_score'].mean()
            v_mean_full = full_sample['entry_v_score'].mean()
            if v_mean_target < v_mean_full - 3:
                doc.append(f"\n**最可能的解释: B (regime不友好)**")
                doc.append(f"[55,60)区间的V分均值={v_mean_target:.1f}，低于全样本均值{v_mean_full:.1f}。")
                doc.append(f"这些信号多数发生在V分较低（不利regime）的环境下。")
                doc.append(f"不是[55,60)这个分数段本身有问题，而是低V分环境下的信号整体质量差。")
            elif available_cols and 'entry_m_score' in target.columns:
                m_mean_target = target['entry_m_score'].mean()
                m_mean_full = full_sample['entry_m_score'].mean()
                if abs(m_mean_target - 25) < 5:
                    doc.append(f"\n**最可能的解释: A (M分踩线效应)**")
                    doc.append(f"[55,60)区间的M分均值={m_mean_target:.1f}，接近M分中档阈值25。")
                    doc.append(f"这些是'M分刚好达标'的弱信号。")
                else:
                    doc.append(f"\n**最可能的解释: D (多重微弱因素)**")
                    doc.append(f"没有单一子分量显示明显异常，可能是多个微弱因素的组合。")
            else:
                doc.append(f"\n**最可能的解释: D (多重微弱因素)**")
        else:
            doc.append(f"\n**解释: E (数据不足以判断)**")

        doc.append(f"\n**建议**: 记录发现但不立即行动。进入roadmap下一个项目（IC stop_loss_pct）。")
        doc.append(f"把此发现作为未来V分/M分优化时的参考。")

    report = "\n".join(doc)
    path = Path("tmp") / "ic_mid_bin_diagnostic.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
