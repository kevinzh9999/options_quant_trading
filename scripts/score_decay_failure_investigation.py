#!/usr/bin/env python3
"""Score衰减出场规则失败调查 - R3判定前疑点澄清。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from data.storage.db_manager import get_db
from scripts.backtest_signals_day import run_day as _original_run_day
from strategies.intraday.A_share_momentum_signal_v2 import (
    SYMBOL_PROFILES, SignalGeneratorV2
)

SPOTS = {'IC': '000905', 'IM': '000852'}


def get_dates(db, spot):
    df = db.query_df(
        f"SELECT DISTINCT substr(datetime,1,10) as d FROM index_min "
        f"WHERE symbol='{spot}' AND period=300 ORDER BY d"
    )
    dates = [d.replace('-', '') for d in df['d'].tolist()]
    return dates[-900:] if len(dates) > 900 else dates


def _bj_to_utc(bj):
    try:
        h, m = int(bj[:2]), int(bj[3:5])
        h -= 8
        if h < 0: h += 24
        return f"{h:02d}:{m:02d}"
    except Exception:
        return ""


def _run_baseline(args):
    """正常backtest，不加任何score_path。"""
    td, sym, thr = args
    SYMBOL_PROFILES[sym]["signal_threshold"] = thr
    db = get_db()
    trades = _original_run_day(sym, td, db, verbose=False)
    full = [t for t in trades if not t.get("partial")]
    for t in full:
        t['trade_date'] = td
        t['symbol'] = sym
    return full


def _run_with_score_path(args):
    """Backtest + score_path + price_path enrichment。"""
    td, sym, thr = args
    SYMBOL_PROFILES[sym]["signal_threshold"] = thr
    db = get_db()

    trades = _original_run_day(sym, td, db, verbose=False)
    full = [t for t in trades if not t.get("partial")]
    if not full:
        return []

    spot = SPOTS[sym]
    df = db.query_df(
        f"SELECT datetime, open, high, low, close, volume FROM index_min "
        f"WHERE symbol='{spot}' AND period=300 ORDER BY datetime"
    )
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = df[c].astype(float)
    df.index = pd.to_datetime(df['datetime'])

    td_fmt = f"{td[:4]}-{td[4:6]}-{td[6:8]}"
    day_mask = df.index.strftime('%Y-%m-%d') == td_fmt
    day_end_idx = df.index.get_indexer(df[day_mask].index)
    if len(day_end_idx) == 0:
        for t in full:
            t['trade_date'] = td
            t['symbol'] = sym
            t['score_path'] = []
            t['price_path'] = []
        return full

    last_day_idx = day_end_idx[-1]
    start_idx = max(0, day_end_idx[0] - 199)
    all_bars = df.iloc[start_idx:last_day_idx + 1]
    today_indices = df[day_mask].index.tolist()
    gen = SignalGeneratorV2({"min_signal_score": 50})

    for t in full:
        t['trade_date'] = td
        t['symbol'] = sym
        entry_bj = t.get('entry_time', '')
        exit_bj = t.get('exit_time', '')
        direction = t.get('direction', '')
        if not entry_bj or not exit_bj:
            t['score_path'] = []
            t['price_path'] = []
            continue

        entry_utc = _bj_to_utc(entry_bj)
        exit_utc = _bj_to_utc(exit_bj)
        score_path = []
        price_path = []
        in_holding = False

        for idx in today_indices:
            dt_str = str(df.loc[idx, 'datetime'])
            bar_utc = dt_str[11:16]
            _h, _m = int(bar_utc[:2]), int(bar_utc[3:5])
            _m += 5
            if _m >= 60: _h += 1; _m -= 60
            exec_utc = f"{_h:02d}:{_m:02d}"

            if exec_utc >= entry_utc and not in_holding:
                in_holding = True
            if in_holding and exec_utc > exit_utc:
                break
            if in_holding:
                price_path.append((float(df.loc[idx, 'open']), float(df.loc[idx, 'close'])))
                pos_in_all = all_bars.index.get_loc(idx)
                window_start = max(0, pos_in_all - 198)
                bar_5m = all_bars.iloc[window_start:pos_in_all + 1]
                try:
                    b15 = bar_5m.resample('15min', label='left', closed='left').agg({
                        'open': 'first', 'high': 'max', 'low': 'min',
                        'close': 'last', 'volume': 'sum'}).dropna()
                    bar_15m = b15.iloc[:-1] if len(b15) > 1 else b15
                except Exception:
                    bar_15m = pd.DataFrame()
                result = gen.score_all(
                    sym, bar_5m, bar_15m if not bar_15m.empty else None,
                    None, None, None, zscore=None, is_high_vol=True,
                    d_override=None, vol_profile=None)
                if result:
                    score_path.append(result['total'] if result['direction'] == direction else -result['total'])
                else:
                    score_path.append(0)

        t['score_path'] = score_path
        t['price_path'] = price_path
        t['hold_bars'] = len(score_path)
    return full


def main():
    print("=" * 60)
    print("  Score衰减出场规则失败调查")
    print("=" * 60)

    db = get_db()
    doc = ["# Score衰减出场规则失败调查\n"]
    n_workers = min(cpu_count(), 8)

    # ═══════════════════════════════════════════════
    # 调查1: 基线为什么变了
    # ═══════════════════════════════════════════════
    doc.append("## 调查1: 基线不一致原因\n")

    for sym in ['IM', 'IC']:
        dates = get_dates(db, SPOTS[sym])
        thr = SYMBOL_PROFILES[sym].get("signal_threshold", 60)

        # 正常baseline (无score_path过滤)
        args = [(td, sym, thr) for td in dates]
        with Pool(n_workers) as pool:
            day_results = pool.map(_run_baseline, args)
        all_trades = [t for day in day_results for t in day]
        total_pnl = sum(t['pnl_pts'] for t in all_trades)
        n_trades = len(all_trades)

        # 短持仓trade统计
        short_trades = [t for t in all_trades if t.get('hold_bars_actual',
                        # 估算hold_bars: 用entry/exit时间差
                        1) <= 2]

        doc.append(f"### {sym} (thr={thr})\n")
        doc.append(f"- 正常baseline: **{n_trades}笔, {total_pnl:+.0f}pt**")

        # 对比之前的数字
        if sym == 'IM':
            doc.append(f"- Part1基线(thr=45后过滤): 2946笔, +6939pt")
            doc.append(f"- Part2衰减模拟基线: +7062pt")
            doc.append(f"- 本次正常baseline: {n_trades}笔, {total_pnl:+.0f}pt")
        else:
            doc.append(f"- Part1基线(thr=45后过滤): 2320笔, +3874pt")
            doc.append(f"- Part2衰减模拟基线: +1725pt")
            doc.append(f"- 本次正常baseline: {n_trades}笔, {total_pnl:+.0f}pt")

        # 按hold time分布
        doc.append(f"\n按入场到出场时间差分布:")
        for t in all_trades:
            eb = t.get('entry_time', '00:00')
            xb = t.get('exit_time', '00:00')
            try:
                eh, em = int(eb[:2]), int(eb[3:5])
                xh, xm = int(xb[:2]), int(xb[3:5])
                mins = (xh - eh) * 60 + (xm - em)
                t['hold_minutes'] = mins
            except Exception:
                t['hold_minutes'] = 0

        hm = pd.Series([t['hold_minutes'] for t in all_trades])
        short_n = (hm <= 10).sum()
        short_pnl = sum(t['pnl_pts'] for t in all_trades if t['hold_minutes'] <= 10)
        doc.append(f"- 持仓<=10min: {short_n}笔, PnL={short_pnl:+.0f}pt")
        doc.append(f"- 持仓>10min: {n_trades - short_n}笔, PnL={total_pnl - short_pnl:+.0f}pt\n")

    doc.append("### 基线差异原因分析\n")
    doc.append("Part1(score区间过滤)用thr=45收集全量trade后post-hoc过滤到[60,100)。")
    doc.append("Part2(衰减模拟)用thr=55/60直接跑backtest，然后过滤掉score_path<3bar的trade。")
    doc.append("差异来源:")
    doc.append("1. thr=45 vs thr=60的position blocking效应(入场不同)")
    doc.append("2. score_path>=3bar过滤丢弃了短持仓trade\n")

    # ═══════════════════════════════════════════════
    # 调查2+3: IM A3触发详细分析
    # ═══════════════════════════════════════════════
    doc.append("## 调查2+3: 触发详细分析\n")

    sym = 'IM'
    dates = get_dates(db, SPOTS[sym])
    thr = SYMBOL_PROFILES[sym].get("signal_threshold", 60)
    print(f"\n[{sym}] 收集带score_path的交易数据...")
    args = [(td, sym, thr) for td in dates]
    with Pool(n_workers) as pool:
        day_results = pool.map(_run_with_score_path, args)
    all_trades = [t for day in day_results for t in day]
    valid = [t for t in all_trades if len(t.get('score_path', [])) >= 3 and len(t.get('price_path', [])) >= 3]
    print(f"  {len(all_trades)}笔总计, {len(valid)}笔有效")

    # 基线PnL (全部trade，不过滤)
    baseline_all_pnl = sum(t['pnl_pts'] for t in all_trades)
    baseline_valid_pnl = sum(t['pnl_pts'] for t in valid)
    dropped_pnl = baseline_all_pnl - baseline_valid_pnl
    doc.append(f"### {sym} 基线对账\n")
    doc.append(f"- 全部trade: {len(all_trades)}笔, {baseline_all_pnl:+.0f}pt")
    doc.append(f"- 有效trade(>=3bar): {len(valid)}笔, {baseline_valid_pnl:+.0f}pt")
    doc.append(f"- 被丢弃的短持仓trade: {len(all_trades)-len(valid)}笔, {dropped_pnl:+.0f}pt\n")

    # A3规则模拟 (入场后2bar, score跌>=20)
    X = 20
    triggered = []
    not_triggered = []
    for t in valid:
        sp = t['score_path']
        pp = t['price_path']
        entry_p = t['entry_price']
        direction = t['direction']
        orig_pnl = t['pnl_pts']

        drop_2bar = sp[0] - sp[2]
        if drop_2bar >= X and len(pp) > 3:
            # 触发: 在bar index 3的open出场
            exit_p = pp[3][0]
            if direction == 'LONG':
                new_pnl = exit_p - entry_p
            else:
                new_pnl = entry_p - exit_p

            t_copy = dict(t)
            t_copy['pnl_at_trigger'] = new_pnl
            t_copy['pnl_if_held'] = orig_pnl
            t_copy['pnl_diff'] = new_pnl - orig_pnl
            t_copy['score_drop_2bar'] = drop_2bar
            t_copy['correct_trigger'] = new_pnl > orig_pnl  # 触发后PnL更好
            triggered.append(t_copy)
        else:
            not_triggered.append(t)

    doc.append(f"### {sym} A3规则(2bar跌>=20) 触发详情\n")
    doc.append(f"- 触发: {len(triggered)}笔 ({len(triggered)/len(valid)*100:.0f}%)")
    doc.append(f"- 未触发: {len(not_triggered)}笔\n")

    if triggered:
        trig_df = pd.DataFrame(triggered)
        # 调查2: 触发PnL统计
        doc.append("#### 触发立即出场的实际PnL\n")
        doc.append(f"- avg pnl_at_trigger: {trig_df['pnl_at_trigger'].mean():+.1f}pt")
        doc.append(f"- avg pnl_if_held: {trig_df['pnl_if_held'].mean():+.1f}pt")
        doc.append(f"- avg pnl_diff: {trig_df['pnl_diff'].mean():+.1f}pt")
        doc.append(f"- 总净改善: {trig_df['pnl_diff'].sum():+.0f}pt")
        doc.append(f"- 对账: 触发总PnL({trig_df['pnl_at_trigger'].sum():+.0f}) + 未触发总PnL({sum(t['pnl_pts'] for t in not_triggered):+.0f}) "
                   f"= {trig_df['pnl_at_trigger'].sum() + sum(t['pnl_pts'] for t in not_triggered):+.0f}")
        doc.append(f"- 基线(有效): {baseline_valid_pnl:+.0f}\n")

        # 触发瞬间PnL分布
        doc.append("#### 触发瞬间PnL分布\n")
        doc.append("| pnl_at_trigger区间 | 笔数 | 占比 |")
        doc.append("|-------------------|------|------|")
        for lo, hi, label in [(-999, -10, '<-10'), (-10, -5, '-10~-5'), (-5, 0, '-5~0'),
                              (0, 5, '0~5'), (5, 10, '5~10'), (10, 999, '>10')]:
            sub = trig_df[(trig_df['pnl_at_trigger'] >= lo) & (trig_df['pnl_at_trigger'] < hi)]
            doc.append(f"| {label} | {len(sub)} | {len(sub)/len(trig_df)*100:.0f}% |")

        # 调查3: 应该触发 vs 误触发
        doc.append("\n#### 应该触发 vs 误触发\n")
        correct = trig_df[trig_df['correct_trigger']]
        wrong = trig_df[~trig_df['correct_trigger']]
        doc.append(f"| 组 | 笔数 | 占比 | avg触发PnL | avg持有PnL | avg差异 |")
        doc.append(f"|---|------|------|----------|----------|--------|")
        doc.append(f"| 应该触发 | {len(correct)} | {len(correct)/len(trig_df)*100:.0f}% | "
                   f"{correct['pnl_at_trigger'].mean():+.1f} | {correct['pnl_if_held'].mean():+.1f} | "
                   f"{correct['pnl_diff'].mean():+.1f} |")
        doc.append(f"| 误触发 | {len(wrong)} | {len(wrong)/len(trig_df)*100:.0f}% | "
                   f"{wrong['pnl_at_trigger'].mean():+.1f} | {wrong['pnl_if_held'].mean():+.1f} | "
                   f"{wrong['pnl_diff'].mean():+.1f} |")

        # 特征对比
        doc.append("\n#### 两组特征对比\n")
        doc.append("| 特征 | 应该触发 | 误触发 |")
        doc.append("|------|---------|-------|")
        doc.append(f"| entry_score均值 | {correct['entry_score'].mean():.1f} | {wrong['entry_score'].mean():.1f} |")
        doc.append(f"| score_drop_2bar均值 | {correct['score_drop_2bar'].mean():.1f} | {wrong['score_drop_2bar'].mean():.1f} |")
        doc.append(f"| pnl_at_trigger均值 | {correct['pnl_at_trigger'].mean():+.1f} | {wrong['pnl_at_trigger'].mean():+.1f} |")

        if 'direction' in trig_df.columns:
            for d in ['LONG', 'SHORT']:
                c_d = correct[correct['direction'] == d]
                w_d = wrong[wrong['direction'] == d]
                doc.append(f"| {d}占比 | {len(c_d)/len(correct)*100:.0f}% | {len(w_d)/len(wrong)*100:.0f}% |")

        # 子分量对比
        for col in ['entry_m_score', 'entry_v_score', 'entry_q_score']:
            if col in trig_df.columns:
                doc.append(f"| {col} | {correct[col].mean():.1f} | {wrong[col].mean():.1f} |")

        # 按score_drop大小分段看正确率
        doc.append("\n#### Score跌幅 vs 触发正确率\n")
        doc.append("| 跌幅区间 | 总触发 | 正确触发 | 正确率 |")
        doc.append("|---------|-------|---------|-------|")
        for lo, hi, label in [(20, 30, '20-30'), (30, 40, '30-40'), (40, 50, '40-50'), (50, 999, '50+')]:
            sub = trig_df[(trig_df['score_drop_2bar'] >= lo) & (trig_df['score_drop_2bar'] < hi)]
            if len(sub) >= 10:
                corr = sub[sub['correct_trigger']].shape[0]
                doc.append(f"| {label} | {len(sub)} | {corr} | {corr/len(sub)*100:.0f}% |")
            elif len(sub) > 0:
                doc.append(f"| {label} | {len(sub)} | (N<10) | |")

    # ═══════════════════════════════════════════════
    # IC也做同样分析
    # ═══════════════════════════════════════════════
    sym = 'IC'
    dates_ic = get_dates(db, SPOTS[sym])
    thr_ic = SYMBOL_PROFILES[sym].get("signal_threshold", 60)
    print(f"\n[{sym}] 收集带score_path的交易数据...")
    args = [(td, sym, thr_ic) for td in dates_ic]
    with Pool(n_workers) as pool:
        day_results = pool.map(_run_with_score_path, args)
    ic_all = [t for day in day_results for t in day]
    ic_valid = [t for t in ic_all if len(t.get('score_path', [])) >= 3 and len(t.get('price_path', [])) >= 3]

    doc.append(f"\n### IC 基线对账\n")
    doc.append(f"- 全部trade: {len(ic_all)}笔, {sum(t['pnl_pts'] for t in ic_all):+.0f}pt")
    doc.append(f"- 有效trade(>=3bar): {len(ic_valid)}笔, {sum(t['pnl_pts'] for t in ic_valid):+.0f}pt")
    doc.append(f"- 被丢弃: {len(ic_all)-len(ic_valid)}笔, {sum(t['pnl_pts'] for t in ic_all)-sum(t['pnl_pts'] for t in ic_valid):+.0f}pt\n")

    # IC A3触发
    ic_triggered = []
    ic_not = []
    for t in ic_valid:
        sp = t['score_path']
        pp = t['price_path']
        drop = sp[0] - sp[2]
        if drop >= X and len(pp) > 3:
            exit_p = pp[3][0]
            new_pnl = (exit_p - t['entry_price']) if t['direction'] == 'LONG' else (t['entry_price'] - exit_p)
            t_c = dict(t)
            t_c['pnl_at_trigger'] = new_pnl
            t_c['pnl_if_held'] = t['pnl_pts']
            t_c['pnl_diff'] = new_pnl - t['pnl_pts']
            t_c['correct_trigger'] = new_pnl > t['pnl_pts']
            t_c['score_drop_2bar'] = drop
            ic_triggered.append(t_c)
        else:
            ic_not.append(t)

    doc.append(f"### IC A3触发: {len(ic_triggered)}笔 ({len(ic_triggered)/len(ic_valid)*100:.0f}%)\n")
    if ic_triggered:
        ic_tdf = pd.DataFrame(ic_triggered)
        doc.append(f"- avg pnl_at_trigger: {ic_tdf['pnl_at_trigger'].mean():+.1f}")
        doc.append(f"- avg pnl_if_held: {ic_tdf['pnl_if_held'].mean():+.1f}")
        doc.append(f"- avg pnl_diff: {ic_tdf['pnl_diff'].mean():+.1f}")
        doc.append(f"- 总净改善: {ic_tdf['pnl_diff'].sum():+.0f}pt")
        ic_corr = ic_tdf[ic_tdf['correct_trigger']]
        ic_wrong = ic_tdf[~ic_tdf['correct_trigger']]
        doc.append(f"- 应该触发: {len(ic_corr)} ({len(ic_corr)/len(ic_tdf)*100:.0f}%), 误触发: {len(ic_wrong)} ({len(ic_wrong)/len(ic_tdf)*100:.0f}%)")

        doc.append("\n#### IC Score跌幅 vs 触发正确率\n")
        doc.append("| 跌幅区间 | 总触发 | 正确触发 | 正确率 |")
        doc.append("|---------|-------|---------|-------|")
        for lo, hi, label in [(20, 30, '20-30'), (30, 40, '30-40'), (40, 50, '40-50'), (50, 999, '50+')]:
            sub = ic_tdf[(ic_tdf['score_drop_2bar'] >= lo) & (ic_tdf['score_drop_2bar'] < hi)]
            if len(sub) >= 10:
                corr = sub[sub['correct_trigger']].shape[0]
                doc.append(f"| {label} | {len(sub)} | {corr} | {corr/len(sub)*100:.0f}% |")

    # ═══════════════════════════════════════════════
    # 综合判定
    # ═══════════════════════════════════════════════
    doc.append("\n## 最终判定\n")
    doc.append("(根据以上数据确认R3/rejected/pending)")

    report = "\n".join(doc)
    path = Path("tmp") / "score_decay_rule_failure_investigation.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
