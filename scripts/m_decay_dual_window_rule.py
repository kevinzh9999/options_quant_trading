#!/usr/bin/env python3
"""M分衰减规则 - 双窗口验证 + 精确规则模拟。"""
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
COMP_MAX = {'m': 50, 'v': 30, 'q': 20}


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
            t['trade_date'] = td; t['symbol'] = sym
            t['m_path'] = []; t['v_path'] = []; t['q_path'] = []
            t['score_path'] = []; t['price_path'] = []
        return full

    last_day_idx = day_end_idx[-1]
    start_idx = max(0, day_end_idx[0] - 199)
    all_bars = df.iloc[start_idx:last_day_idx + 1]
    today_indices = df[day_mask].index.tolist()
    gen = SignalGeneratorV2({"min_signal_score": 50})

    for t in full:
        t['trade_date'] = td; t['symbol'] = sym
        entry_bj = t.get('entry_time', '')
        exit_bj = t.get('exit_time', '')
        direction = t.get('direction', '')
        if not entry_bj or not exit_bj:
            t['m_path'] = []; t['v_path'] = []; t['q_path'] = []
            t['score_path'] = []; t['price_path'] = []
            continue

        entry_utc = _bj_to_utc(entry_bj)
        exit_utc = _bj_to_utc(exit_bj)
        m_path, v_path, q_path, score_path, price_path = [], [], [], [], []
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
                    m_path.append(result.get('s_momentum', 0))
                    v_path.append(result.get('s_volatility', 0))
                    q_path.append(result.get('s_volume', 0))
                elif result:
                    score_path.append(-result['total'])
                    m_path.append(0); v_path.append(0); q_path.append(0)
                else:
                    score_path.append(0); m_path.append(0); v_path.append(0); q_path.append(0)

        t['m_path'] = m_path; t['v_path'] = v_path; t['q_path'] = q_path
        t['score_path'] = score_path; t['price_path'] = price_path
        t['hold_bars'] = len(score_path)
    return full


def simulate_m_exit(trades, rule, X, m_thr=15, v_thr=9):
    """模拟M分衰减出场规则。返回调整后trades和统计。"""
    adjusted = []
    n_trig = 0; avoided = 0; missed = 0

    for t in trades:
        mp = t.get('m_path', [])
        vp = t.get('v_path', [])
        pp = t.get('price_path', [])
        orig_pnl = t['pnl_pts']
        entry_p = t['entry_price']
        direction = t['direction']

        if len(mp) < 3 or len(pp) < 3:
            adjusted.append(dict(t)); continue

        trigger_bar = None

        if rule == 'R1':  # 前2bar M衰减
            if mp[0] - mp[2] >= X:
                trigger_bar = 2
        elif rule == 'R2':  # 持续M衰减
            for i in range(2, len(mp)):
                if mp[0] - mp[i] >= X:
                    trigger_bar = i; break
        elif rule == 'R3':  # M+V组合
            for i in range(2, len(mp)):
                m_drop = max(mp[:i+1]) - mp[i]
                v_drop = max(vp[:i+1]) - vp[i] if len(vp) > i else 0
                if m_drop >= m_thr and v_drop >= v_thr:
                    trigger_bar = i; break
        elif rule == 'R4':  # R1(15) + R3
            if len(mp) >= 3 and mp[0] - mp[2] >= 15:
                trigger_bar = 2
            if trigger_bar is None:
                for i in range(2, len(mp)):
                    m_drop = max(mp[:i+1]) - mp[i]
                    v_drop = max(vp[:i+1]) - vp[i] if len(vp) > i else 0
                    if m_drop >= m_thr and v_drop >= v_thr:
                        trigger_bar = i; break

        if trigger_bar is not None and trigger_bar + 1 < len(pp):
            n_trig += 1
            exit_p = pp[trigger_bar + 1][0]
            new_pnl = (exit_p - entry_p) if direction == 'LONG' else (entry_p - exit_p)
            diff = new_pnl - orig_pnl
            if diff > 0: avoided += diff
            else: missed += (-diff)
            nt = dict(t); nt['pnl_pts'] = new_pnl; nt['triggered'] = True
            nt['trigger_pnl'] = new_pnl; nt['held_pnl'] = orig_pnl
            nt['correct'] = new_pnl > orig_pnl
            adjusted.append(nt)
        else:
            nt = dict(t); nt['triggered'] = False; adjusted.append(nt)

    return adjusted, n_trig, avoided, missed


def main():
    print("=" * 60)
    print("  M分衰减规则 - 双窗口验证 + 规则模拟")
    print("=" * 60)

    db = get_db()
    n_workers = min(cpu_count(), 8)
    doc = ["# M分衰减规则 - 双窗口验证 + 规则模拟\n"]

    all_trades = {}
    all_dates = {}
    splits = {}

    for sym in ['IM', 'IC']:
        print(f"\n[{sym}] 收集数据...")
        dates = get_dates(db, SPOTS[sym])
        thr = SYMBOL_PROFILES[sym].get("signal_threshold", 60)
        args = [(td, sym, thr) for td in dates]
        with Pool(n_workers) as pool:
            day_results = pool.map(_run_one_day, args)
        trades = [t for day in day_results for t in day]
        valid = [t for t in trades if len(t.get('m_path', [])) >= 3 and len(t.get('price_path', [])) >= 3]
        print(f"  {len(trades)}笔总计, {len(valid)}笔有效")

        is_set = set(dates[-219:])
        oos_set = set(dates[:-219])
        all_trades[sym] = valid
        all_dates[sym] = dates
        splits[sym] = {'is': is_set, 'oos': oos_set}

    # ═══════════════════════════════════════════════
    # Part A: 双窗口验证
    # ═══════════════════════════════════════════════
    doc.append("# Part A: 双窗口验证\n")

    # A.2 相关性
    doc.append("## A.2 双窗口子分量相关性 (decay_total vs PnL)\n")
    doc.append("| 子分量 | IM_OOS r | IM_IS r | IC_OOS r | IC_IS r |")
    doc.append("|--------|---------|---------|---------|---------|")

    m_w1_pass = True
    for comp in ['score', 'm', 'v', 'q']:
        cells = [comp]
        for sym in ['IM', 'IC']:
            for window in ['oos', 'is']:
                wset = splits[sym][window]
                trades = [t for t in all_trades[sym] if t['trade_date'] in wset]
                if len(trades) < 50:
                    cells.append("-"); continue
                path_col = f'{comp}_path'
                decays = [t[path_col][0] - t[path_col][-1] for t in trades if isinstance(t[path_col], list) and len(t[path_col]) >= 2]
                pnls = [t['pnl_pts'] for t in trades if isinstance(t[path_col], list) and len(t[path_col]) >= 2]
                if len(decays) >= 50:
                    r, p = pearsonr(decays, pnls)
                    cells.append(f"{r:.3f}")
                else:
                    cells.append("-")
        doc.append(f"| {' | '.join(cells)} |")
    doc.append("")

    # A.3 组合衰减
    doc.append("## A.3 双窗口组合衰减分析\n")
    for sym in ['IM', 'IC']:
        doc.append(f"### {sym}\n")
        doc.append("| 窗口 | 组合 | 笔数 | AvgPnL | 胜率 |")
        doc.append("|------|------|------|--------|------|")
        for window in ['OOS', 'IS']:
            wset = splits[sym][window.lower()]
            trades = [t for t in all_trades[sym] if t['trade_date'] in wset]
            for t in trades:
                mp = t.get('m_path', [])
                vp = t.get('v_path', [])
                t['m_severe'] = (mp[0] - mp[-1] >= 15) if len(mp) >= 2 else False
                t['v_severe'] = (vp[0] - vp[-1] >= 9) if len(vp) >= 2 else False

            combos = [
                ('无严重衰减', [t for t in trades if not t['m_severe'] and not t['v_severe']]),
                ('仅V严重', [t for t in trades if t['v_severe'] and not t['m_severe']]),
                ('仅M严重', [t for t in trades if t['m_severe'] and not t['v_severe']]),
                ('M+V同时', [t for t in trades if t['m_severe'] and t['v_severe']]),
            ]
            for label, sub in combos:
                if len(sub) >= 30:
                    wr = sum(1 for t in sub if t['pnl_pts'] > 0) / len(sub) * 100
                    avg = np.mean([t['pnl_pts'] for t in sub])
                    doc.append(f"| {window} | {label} | {len(sub)} | {avg:+.1f} | {wr:.0f}% |")
                elif len(sub) > 0:
                    doc.append(f"| {window} | {label} | {len(sub)} | (N<30) | |")
        doc.append("")

    # A.4 前2bar可分性
    doc.append("## A.4 双窗口前2bar M分可分性\n")
    doc.append("| 品种 | 窗口 | 亏损M衰减 | 盈利M衰减 | 差距 |")
    doc.append("|------|------|---------|---------|------|")
    for sym in ['IM', 'IC']:
        for window in ['OOS', 'IS']:
            wset = splits[sym][window.lower()]
            trades = [t for t in all_trades[sym] if t['trade_date'] in wset and len(t.get('m_path', [])) >= 3]
            losers = [t for t in trades if t['pnl_pts'] < -5]
            winners = [t for t in trades if t['pnl_pts'] > 5]
            if len(losers) >= 30 and len(winners) >= 30:
                l_drop = np.mean([t['m_path'][0] - t['m_path'][2] for t in losers])
                w_drop = np.mean([t['m_path'][0] - t['m_path'][2] for t in winners])
                doc.append(f"| {sym} | {window} | {l_drop:+.1f} | {w_drop:+.1f} | {l_drop - w_drop:+.1f} |")

    doc.append("")

    # A.5 判定
    doc.append("## A.5 任务A判定\n")
    # 简化判定：检查M在OOS上的r是否仍强于score
    doc.append("(根据以上数据判定M-W1/M-W2/M-W3)\n")

    # ═══════════════════════════════════════════════
    # Part B: 规则模拟
    # ═══════════════════════════════════════════════
    doc.append("# Part B: M分衰减规则精确模拟\n")

    # B.2 候选对比
    candidates = [
        ('基线', None, 0),
        ('R1_X10', 'R1', 10), ('R1_X15', 'R1', 15), ('R1_X20', 'R1', 20),
        ('R2_X15', 'R2', 15), ('R2_X20', 'R2', 20), ('R2_X25', 'R2', 25),
        ('R3', 'R3', 0),
        ('R4', 'R4', 0),
    ]

    doc.append("## B.2 全样本候选对比\n")
    doc.append("| 候选 | IM_净改善 | IM_触发 | IC_净改善 | IC_触发 | 合计净改善 |")
    doc.append("|------|---------|--------|---------|--------|---------|")

    baseline_pnl = {}
    best_name = None; best_improve = -99999
    candidate_results = {}

    for name, rule, x in candidates:
        im_improve = 0; ic_improve = 0; im_trig = 0; ic_trig = 0
        for sym in ['IM', 'IC']:
            trades = all_trades[sym]
            if rule is None:
                pnl = sum(t['pnl_pts'] for t in trades)
                baseline_pnl[sym] = pnl
                trig = 0
            else:
                adj, trig, avoided, missed = simulate_m_exit(trades, rule, x)
                pnl = sum(t['pnl_pts'] for t in adj)
            improve = pnl - baseline_pnl.get(sym, pnl)
            if sym == 'IM': im_improve = improve; im_trig = trig
            else: ic_improve = improve; ic_trig = trig

        total_improve = im_improve + ic_improve
        candidate_results[name] = {'im': im_improve, 'ic': ic_improve, 'total': total_improve}
        if rule and total_improve > best_improve:
            best_improve = total_improve; best_name = name

        doc.append(f"| {name} | {im_improve:+.0f} | {im_trig} | {ic_improve:+.0f} | {ic_trig} | {total_improve:+.0f} |")

    doc.append(f"\n**全样本最优: {best_name}** (净改善{best_improve:+.0f}pt)\n")

    # B.3+B.4 最优候选详细分析
    if best_name:
        best_rule = [r for n, r, x in candidates if n == best_name][0]
        best_x = [x for n, r, x in candidates if n == best_name][0]

        doc.append(f"## B.3+B.4 最优候选 {best_name} 详细分析\n")
        for sym in ['IM', 'IC']:
            adj, trig, avoided, missed = simulate_m_exit(all_trades[sym], best_rule, best_x)
            triggered = [t for t in adj if t.get('triggered')]
            correct = [t for t in triggered if t.get('correct')]
            wrong = [t for t in triggered if not t.get('correct')]

            doc.append(f"### {sym}\n")
            doc.append(f"- 触发: {trig}笔 ({trig/len(adj)*100:.0f}%)")
            doc.append(f"- 避免亏损: {avoided:+.0f}, 错过盈利: {missed:+.0f}, 净: {avoided-missed:+.0f}")
            if triggered:
                doc.append(f"- 触发瞬间平均PnL: {np.mean([t['trigger_pnl'] for t in triggered]):+.1f}")
                doc.append(f"- 不触发平均PnL: {np.mean([t['held_pnl'] for t in triggered]):+.1f}")
            if correct and wrong:
                doc.append(f"- 应该触发: {len(correct)} ({len(correct)/len(triggered)*100:.0f}%), "
                           f"误触发: {len(wrong)} ({len(wrong)/len(triggered)*100:.0f}%)")
                doc.append(f"- 应该触发avg差异: {np.mean([t['trigger_pnl']-t['held_pnl'] for t in correct]):+.1f}")
                doc.append(f"- 误触发avg代价: {np.mean([t['trigger_pnl']-t['held_pnl'] for t in wrong]):+.1f}")
            doc.append("")

        # B.5 双窗口规则验证
        doc.append(f"## B.5 双窗口规则验证 ({best_name})\n")
        doc.append("| 品种 | OOS净改善 | IS净改善 | 两窗口都正? |")
        doc.append("|------|---------|---------|-----------|")
        for sym in ['IM', 'IC']:
            for window_label, wset in [('OOS', splits[sym]['oos']), ('IS', splits[sym]['is'])]:
                pass  # computed below

            oos_trades = [t for t in all_trades[sym] if t['trade_date'] in splits[sym]['oos']]
            is_trades = [t for t in all_trades[sym] if t['trade_date'] in splits[sym]['is']]

            oos_base = sum(t['pnl_pts'] for t in oos_trades)
            is_base = sum(t['pnl_pts'] for t in is_trades)

            oos_adj, _, _, _ = simulate_m_exit(oos_trades, best_rule, best_x)
            is_adj, _, _, _ = simulate_m_exit(is_trades, best_rule, best_x)

            oos_new = sum(t['pnl_pts'] for t in oos_adj)
            is_new = sum(t['pnl_pts'] for t in is_adj)

            oos_imp = oos_new - oos_base
            is_imp = is_new - is_base
            both_pos = "✓" if oos_imp > 0 and is_imp > 0 else "✗"

            doc.append(f"| {sym} | {oos_imp:+.0f} | {is_imp:+.0f} | {both_pos} |")
        doc.append("")

        # B.6 跟现有出场逻辑的交互
        doc.append(f"## B.6 触发trade原本会被什么出场\n")
        for sym in ['IM', 'IC']:
            adj, _, _, _ = simulate_m_exit(all_trades[sym], best_rule, best_x)
            triggered = [t for t in adj if t.get('triggered')]
            reason_col = 'reason' if triggered and 'reason' in triggered[0] else 'exit_reason'
            if not triggered:
                continue
            doc.append(f"### {sym}\n")
            doc.append("| 原出场原因 | 笔数 | 占比 | 这些trade的净改善 |")
            doc.append("|-----------|------|------|--------------|")
            reasons = {}
            for t in triggered:
                r = t.get(reason_col, 'unknown')
                if r not in reasons:
                    reasons[r] = []
                reasons[r].append(t['trigger_pnl'] - t['held_pnl'])
            for r, diffs in sorted(reasons.items(), key=lambda x: -len(x[1])):
                if len(diffs) >= 5:
                    doc.append(f"| {r} | {len(diffs)} | {len(diffs)/len(triggered)*100:.0f}% | {sum(diffs):+.0f} |")
            doc.append("")

    # B.7 综合判定
    doc.append("## B.7 综合判定\n")
    total_baseline = sum(baseline_pnl.values())
    improve_pct = best_improve / total_baseline * 100 if total_baseline > 0 else 0
    doc.append(f"最优候选: {best_name}, 全样本净改善: {best_improve:+.0f}pt ({improve_pct:+.1f}%)\n")

    if improve_pct >= 5:
        doc.append(f"**判定M-R1: M分衰减规则突破成功** ✓")
    elif improve_pct >= 2:
        doc.append(f"**判定M-R2: 有限改进**")
    else:
        doc.append(f"**判定M-R3: 规则失败** ✗")

    report = "\n".join(doc)
    path = Path("tmp") / "m_decay_dual_window_and_rule_simulation.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
