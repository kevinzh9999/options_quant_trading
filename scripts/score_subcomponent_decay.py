#!/usr/bin/env python3
"""Score子分量衰减拆解研究：M/V/Q/B/S各自的衰减与PnL关系。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from scipy.stats import pearsonr
from data.storage.db_manager import get_db
from scripts.backtest_signals_day import run_day as _original_run_day
from strategies.intraday.A_share_momentum_signal_v2 import (
    SYMBOL_PROFILES, SignalGeneratorV2
)

SPOTS = {'IC': '000905', 'IM': '000852'}
COMPONENTS = ['m', 'v', 'q', 'b', 's']
COMP_MAX = {'m': 50, 'v': 30, 'q': 20, 'b': 20, 's': 15}


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


def _run_one_day(args):
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
            for c in COMPONENTS:
                t[f'{c}_path'] = []
            t['score_path'] = []
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
            for c in COMPONENTS:
                t[f'{c}_path'] = []
            t['score_path'] = []
            continue

        entry_utc = _bj_to_utc(entry_bj)
        exit_utc = _bj_to_utc(exit_bj)
        paths = {c: [] for c in COMPONENTS}
        score_path = []
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
                pos = all_bars.index.get_loc(idx)
                ws = max(0, pos - 198)
                bar_5m = all_bars.iloc[ws:pos + 1]
                try:
                    b15f = bar_5m.resample('15min', label='left', closed='left').agg({
                        'open': 'first', 'high': 'max', 'low': 'min',
                        'close': 'last', 'volume': 'sum'}).dropna()
                    bar_15m = b15f.iloc[:-1] if len(b15f) > 1 else b15f
                except Exception:
                    bar_15m = pd.DataFrame()

                result = gen.score_all(
                    sym, bar_5m, bar_15m if not bar_15m.empty else None,
                    None, None, None, zscore=None, is_high_vol=True,
                    d_override=None, vol_profile=None)

                if result and result['direction'] == direction:
                    score_path.append(result['total'])
                    paths['m'].append(result.get('s_momentum', 0))
                    paths['v'].append(result.get('s_volatility', 0))
                    paths['q'].append(result.get('s_volume', 0))
                    paths['b'].append(result.get('s_breakout', 0))
                    raw = result.get('raw_total', 0)
                    paths['s'].append(raw - result.get('s_momentum', 0) - result.get('s_volatility', 0)
                                      - result.get('s_volume', 0) - result.get('s_breakout', 0))
                elif result:
                    score_path.append(-result['total'])
                    for c in COMPONENTS:
                        paths[c].append(0)
                else:
                    score_path.append(0)
                    for c in COMPONENTS:
                        paths[c].append(0)

        t['score_path'] = score_path
        for c in COMPONENTS:
            t[f'{c}_path'] = paths[c]
        t['hold_bars'] = len(score_path)

    return full


def calc_decay(path):
    """计算衰减指标。"""
    if len(path) < 3:
        return {'decay_total': 0, 'decay_from_peak': 0, 'drop_2bar': 0}
    entry = path[0]
    exit_v = path[-1]
    peak = max(path)
    return {
        'decay_total': entry - exit_v,
        'decay_from_peak': peak - exit_v,
        'drop_2bar': entry - path[2],
    }


def main():
    print("=" * 60)
    print("  Score 子分量衰减拆解研究")
    print("=" * 60)

    db = get_db()
    n_workers = min(cpu_count(), 8)
    doc = ["# Score 子分量衰减拆解研究\n"]

    for sym in ['IM', 'IC']:
        print(f"\n[{sym}] 收集带子分量path的交易数据...")
        dates = get_dates(db, SPOTS[sym])
        thr = SYMBOL_PROFILES[sym].get("signal_threshold", 60)
        args = [(td, sym, thr) for td in dates]
        with Pool(n_workers) as pool:
            day_results = pool.map(_run_one_day, args)
        all_trades = [t for day in day_results for t in day]
        valid = [t for t in all_trades if len(t.get('score_path', [])) >= 3]
        print(f"  {len(all_trades)}笔总计, {len(valid)}笔有效")

        tdf = pd.DataFrame(valid)
        doc.append(f"## {sym} ({len(valid)}笔有效)\n")

        # 计算各子分量衰减
        for comp in ['score'] + COMPONENTS:
            col = f'{comp}_path'
            for metric in ['decay_total', 'decay_from_peak', 'drop_2bar']:
                tdf[f'{comp}_{metric}'] = tdf[col].apply(
                    lambda p: calc_decay(p)[metric] if isinstance(p, list) and len(p) >= 3 else 0)

        # ═══════ Step 4: 单子分量相关性 ═══════
        doc.append("### Step 4: 单子分量Pearson相关 (decay vs PnL)\n")
        doc.append("| 子分量 | r(decay_total) | p | r(decay_from_peak) | p | r(drop_2bar) | p |")
        doc.append("|--------|---------------|---|-------------------|---|-------------|---|")

        for comp in ['score'] + COMPONENTS:
            cells = [comp]
            for metric in ['decay_total', 'decay_from_peak', 'drop_2bar']:
                col = f'{comp}_{metric}'
                valid_mask = tdf[col].notna() & tdf['pnl_pts'].notna()
                if valid_mask.sum() >= 50:
                    r, p = pearsonr(tdf.loc[valid_mask, col], tdf.loc[valid_mask, 'pnl_pts'])
                    cells.extend([f"{r:.3f}", f"{p:.4f}"])
                else:
                    cells.extend(["-", "-"])
            doc.append(f"| {' | '.join(cells)} |")
        doc.append("")

        # ═══════ Step 5: 按衰减分组看PnL ═══════
        doc.append("### Step 5: 按衰减分组的PnL表现\n")
        for comp in COMPONENTS:
            max_val = COMP_MAX[comp]
            bins_def = [
                (-999, -max_val*0.1, '上升'),
                (-max_val*0.1, max_val*0.1, '稳定'),
                (max_val*0.1, max_val*0.3, '小衰'),
                (max_val*0.3, max_val*0.6, '中衰'),
                (max_val*0.6, 999, '严重'),
            ]
            col = f'{comp}_decay_total'
            doc.append(f"**{comp.upper()}分 (最大{max_val}):**")
            doc.append(f"| 衰减组 | 笔数 | AvgPnL | 胜率 |")
            doc.append(f"|--------|------|--------|------|")
            for lo, hi, label in bins_def:
                sub = tdf[(tdf[col] >= lo) & (tdf[col] < hi)]
                if len(sub) >= 50:
                    wr = (sub['pnl_pts'] > 0).sum() / len(sub) * 100
                    doc.append(f"| {label} | {len(sub)} | {sub['pnl_pts'].mean():+.1f} | {wr:.0f}% |")
                elif len(sub) > 0:
                    doc.append(f"| {label} | {len(sub)} | (N<50) | |")
            doc.append("")

        # ═══════ Step 6: 领先指标分析 ═══════
        doc.append("### Step 6: 领先指标（亏损trade中哪个子分量最先衰减）\n")
        losers = tdf[tdf['pnl_pts'] < -10]
        if len(losers) >= 50:
            first_counts = {c: 0 for c in COMPONENTS}
            for _, row in losers.iterrows():
                first_bar = 999
                first_comp = None
                for comp in COMPONENTS:
                    path = row[f'{comp}_path']
                    if not isinstance(path, list) or len(path) < 3:
                        continue
                    thr_val = COMP_MAX[comp] * 0.2
                    for i in range(1, len(path)):
                        if path[0] - path[i] >= thr_val:
                            if i < first_bar:
                                first_bar = i
                                first_comp = comp
                            break
                if first_comp:
                    first_counts[first_comp] += 1

            total_first = sum(first_counts.values())
            doc.append(f"亏损trade(PnL<-10): {len(losers)}笔\n")
            doc.append("| 子分量 | 最先衰减次数 | 占比 |")
            doc.append("|--------|-----------|------|")
            for comp in COMPONENTS:
                pct = first_counts[comp] / total_first * 100 if total_first > 0 else 0
                doc.append(f"| {comp.upper()} | {first_counts[comp]} | {pct:.0f}% |")
        else:
            doc.append(f"亏损trade(PnL<-10)不足50笔，跳过")
        doc.append("")

        # ═══════ Step 7: 组合衰减分析 ═══════
        doc.append("### Step 7: 组合衰减分析\n")
        # 定义"严重衰减" = decay >= max * 0.3
        for comp in COMPONENTS:
            tdf[f'{comp}_severe'] = tdf[f'{comp}_decay_total'] >= COMP_MAX[comp] * 0.3

        combos = [
            ('仅M严重', lambda r: r['m_severe'] & ~r['v_severe'] & ~r['q_severe']),
            ('仅V严重', lambda r: r['v_severe'] & ~r['m_severe'] & ~r['q_severe']),
            ('M+V同时', lambda r: r['m_severe'] & r['v_severe']),
            ('M+V+Q同时', lambda r: r['m_severe'] & r['v_severe'] & r['q_severe']),
            ('无严重衰减', lambda r: ~r['m_severe'] & ~r['v_severe'] & ~r['q_severe']),
        ]

        doc.append("| 组合 | 笔数 | AvgPnL | 胜率 |")
        doc.append("|------|------|--------|------|")
        for label, cond in combos:
            mask = cond(tdf)
            sub = tdf[mask]
            if len(sub) >= 50:
                wr = (sub['pnl_pts'] > 0).sum() / len(sub) * 100
                doc.append(f"| {label} | {len(sub)} | {sub['pnl_pts'].mean():+.1f} | {wr:.0f}% |")
            elif len(sub) > 0:
                doc.append(f"| {label} | {len(sub)} | (N<50) | |")
        doc.append("")

        # ═══════ Step 8: 亏损vs盈利的子分量早期衰减对比 ═══════
        doc.append("### Step 8: 亏损vs盈利trade的前2bar子分量衰减\n")
        big_loss = tdf[tdf['pnl_pts'] < -5]
        big_win = tdf[tdf['pnl_pts'] > 5]

        if len(big_loss) >= 50 and len(big_win) >= 50:
            doc.append("| 子分量 | 亏损组前2bar衰减 | 盈利组前2bar衰减 | 差异 | 区分力 |")
            doc.append("|--------|---------------|---------------|------|--------|")
            for comp in COMPONENTS:
                col = f'{comp}_drop_2bar'
                loss_val = big_loss[col].mean()
                win_val = big_win[col].mean()
                diff = loss_val - win_val
                strength = "**强**" if abs(diff) > COMP_MAX[comp] * 0.1 else ("中等" if abs(diff) > COMP_MAX[comp] * 0.05 else "弱")
                doc.append(f"| {comp.upper()} | {loss_val:+.1f} | {win_val:+.1f} | {diff:+.1f} | {strength} |")
        else:
            doc.append(f"亏损/盈利组不足50笔，跳过")
        doc.append("")

    # ═══════ 综合判定 ═══════
    doc.append("## 综合判定\n")
    doc.append("(根据以上数据选择D1/D2/D3/D4)")

    report = "\n".join(doc)
    path = Path("tmp") / "score_subcomponent_decay_decomposition.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
