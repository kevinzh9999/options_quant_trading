#!/usr/bin/env python3
"""Score衰减出场规则 - 精确逐bar回放模拟。"""
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


def _run_one_day_with_decay(args):
    """收集trade + score_path + price_path，然后模拟各种decay exit规则。"""
    td, sym, thr = args
    SYMBOL_PROFILES[sym]["signal_threshold"] = thr
    db = get_db()

    # 正常backtest
    trades = _original_run_day(sym, td, db, verbose=False)
    full = [t for t in trades if not t.get("partial")]
    if not full:
        return []

    # 加载bar数据
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

        entry_time_bj = t.get('entry_time', '')
        exit_time_bj = t.get('exit_time', '')
        direction = t.get('direction', '')

        if not entry_time_bj or not exit_time_bj:
            t['score_path'] = []
            t['price_path'] = []
            continue

        entry_utc = _bj_to_utc(entry_time_bj)
        exit_utc = _bj_to_utc(exit_time_bj)

        score_path = []
        price_path = []  # (open, close) of each holding bar
        in_holding = False

        for idx in today_indices:
            dt_str = str(df.loc[idx, 'datetime'])
            bar_utc = dt_str[11:16]
            _h, _m = int(bar_utc[:2]), int(bar_utc[3:5])
            _m += 5
            if _m >= 60:
                _h += 1; _m -= 60
            exec_utc = f"{_h:02d}:{_m:02d}"

            if exec_utc >= entry_utc and not in_holding:
                in_holding = True
            if in_holding and exec_utc > exit_utc:
                break

            if in_holding:
                bar_open = float(df.loc[idx, 'open'])
                bar_close = float(df.loc[idx, 'close'])
                price_path.append((bar_open, bar_close))

                pos_in_all = all_bars.index.get_loc(idx)
                window_start = max(0, pos_in_all - 198)
                bar_5m = all_bars.iloc[window_start:pos_in_all + 1]

                try:
                    bar_15m_full = bar_5m.resample('15min', label='left', closed='left').agg({
                        'open': 'first', 'high': 'max', 'low': 'min',
                        'close': 'last', 'volume': 'sum'
                    }).dropna()
                    bar_15m = bar_15m_full.iloc[:-1] if len(bar_15m_full) > 1 else bar_15m_full
                except Exception:
                    bar_15m = pd.DataFrame()

                result = gen.score_all(
                    sym, bar_5m, bar_15m if not bar_15m.empty else None,
                    None, None, None,
                    zscore=None, is_high_vol=True, d_override=None,
                    vol_profile=None,
                )

                if result:
                    if result['direction'] == direction:
                        score_path.append(result['total'])
                    else:
                        score_path.append(-result['total'])
                else:
                    score_path.append(0)

        t['score_path'] = score_path
        t['price_path'] = price_path
        t['hold_bars'] = len(score_path)

    return full


def simulate_decay_exit(trades_list, rule_type, X, sym):
    """模拟score衰减出场规则，返回调整后的trade列表。

    rule_type:
      'A': 入场后2bar一次性检查(bar index 2)
      'B': 持续检查(每根bar)
    """
    adjusted = []
    n_triggered = 0
    avoided_loss = 0  # 触发后避免的亏损
    missed_gain = 0   # 触发后错过的盈利

    for t in trades_list:
        sp = t.get('score_path', [])
        pp = t.get('price_path', [])
        orig_pnl = t.get('pnl_pts', 0)
        direction = t.get('direction', '')
        entry_p = t.get('entry_price', 0)

        if len(sp) < 3 or len(pp) < 3:
            adjusted.append(dict(t))
            continue

        trigger_bar = None

        if rule_type == 'A':
            # 入场后第2根bar检查
            drop = sp[0] - sp[2]
            if drop >= X:
                trigger_bar = 2
        elif rule_type == 'B':
            # 持续检查: score从峰值跌>=X
            peak = sp[0]
            for i in range(2, len(sp)):
                peak = max(peak, sp[i])
                if peak - sp[i] >= X:
                    trigger_bar = i
                    break

        if trigger_bar is not None and trigger_bar + 1 < len(pp):
            n_triggered += 1
            # 在trigger_bar+1的open出场
            exit_p = pp[trigger_bar + 1][0]  # next bar open
            if direction == 'LONG':
                new_pnl = exit_p - entry_p
            else:
                new_pnl = entry_p - exit_p

            diff = new_pnl - orig_pnl
            if diff > 0:
                avoided_loss += diff  # 提前出避免了亏损
            else:
                missed_gain += (-diff)  # 提前出错过了盈利

            new_t = dict(t)
            new_t['pnl_pts'] = new_pnl
            new_t['exit_reason'] = f'score_decay_{rule_type}{X}'
            new_t['decay_triggered'] = True
            adjusted.append(new_t)
        else:
            new_t = dict(t)
            new_t['decay_triggered'] = False
            adjusted.append(new_t)

    return adjusted, n_triggered, avoided_loss, missed_gain


def calc_stats(trades, dates):
    if not trades:
        return {'pnl': 0, 'n': 0, 'avg': 0, 'sharpe': 0}
    pnls = [t['pnl_pts'] for t in trades]
    n = len(pnls)
    daily = {}
    for t in trades:
        d = t.get('trade_date', '')
        daily[d] = daily.get(d, 0) + t['pnl_pts']
    all_days = pd.Series(0.0, index=dates)
    for d, p in daily.items():
        if d in all_days.index:
            all_days[d] = p
    vals = all_days.values
    avg = np.mean(vals)
    std = np.std(vals)
    sharpe = avg / std * np.sqrt(252) if std > 0 else 0
    return {'pnl': sum(pnls), 'n': n, 'avg': np.mean(pnls), 'sharpe': sharpe}


def main():
    print("=" * 60)
    print("  Score衰减出场规则 - 精确逐bar回放模拟")
    print("=" * 60)

    db = get_db()
    doc = ["# Score衰减出场规则 - 精确模拟\n"]

    all_trades_raw = {}
    all_dates = {}

    for sym in ['IM', 'IC']:
        print(f"\n[{sym}] 收集带score_path+price_path的交易数据...")
        dates = get_dates(db, SPOTS[sym])
        orig_thr = SYMBOL_PROFILES[sym].get("signal_threshold", 60)
        n_workers = min(cpu_count(), 8)
        args_list = [(td, sym, orig_thr) for td in dates]
        with Pool(n_workers) as pool:
            day_results = pool.map(_run_one_day_with_decay, args_list)
        trades = [t for day in day_results for t in day]
        # 过滤有效trade
        valid = [t for t in trades if len(t.get('score_path', [])) >= 3 and len(t.get('price_path', [])) >= 3]
        print(f"  {len(trades)}笔总计, {len(valid)}笔有效(>=3 bar)")
        all_trades_raw[sym] = valid
        all_dates[sym] = dates

    # ═══════════════════════════════════════════════
    # 第二步: 参数sweep
    # ═══════════════════════════════════════════════
    doc.append("## 第二步: 参数Sweep (900天全样本)\n")

    candidates = [
        ('基线', None, 0),
        ('A1', 'A', 10), ('A2', 'A', 15), ('A3', 'A', 20),
        ('B1', 'B', 10), ('B2', 'B', 15), ('B3', 'B', 20),
        ('B4', 'B', 25), ('B5', 'B', 30),
    ]

    doc.append("| 候选 | 类型 | X | IM_PnL | IM_触发 | IC_PnL | IC_触发 | 合计 | vs基线 |")
    doc.append("|------|------|---|--------|--------|--------|--------|------|--------|")

    baseline_total = 0
    sweep_results = {}

    for name, rule, x in candidates:
        im_pnl = 0; ic_pnl = 0; im_trig = 0; ic_trig = 0

        for sym in ['IM', 'IC']:
            if rule is None:
                # 基线
                pnl = sum(t['pnl_pts'] for t in all_trades_raw[sym])
                s = calc_stats(all_trades_raw[sym], all_dates[sym])
                trig = 0
            else:
                adj, trig, avoided, missed = simulate_decay_exit(all_trades_raw[sym], rule, x, sym)
                pnl = sum(t['pnl_pts'] for t in adj)
                s = calc_stats(adj, all_dates[sym])

            if sym == 'IM':
                im_pnl = pnl; im_trig = trig
            else:
                ic_pnl = pnl; ic_trig = trig

        total = im_pnl + ic_pnl
        if name == '基线':
            baseline_total = total
        diff = total - baseline_total

        sweep_results[name] = {'im': im_pnl, 'ic': ic_pnl, 'total': total, 'diff': diff}
        doc.append(f"| {name} | {rule or '-'} | {x or '-'} | {im_pnl:+.0f} | {im_trig} | "
                   f"{ic_pnl:+.0f} | {ic_trig} | {total:+.0f} | {diff:+.0f} |")

    # 最优候选
    best_name = max([n for n, _, _ in candidates if n != '基线'],
                    key=lambda n: sweep_results[n]['total'])
    best_diff = sweep_results[best_name]['diff']
    improve_pct = best_diff / baseline_total * 100 if baseline_total != 0 else 0
    doc.append(f"\n**全样本最优: {best_name}** (改善{best_diff:+.0f}pt, {improve_pct:+.1f}%)\n")

    # ═══════════════════════════════════════════════
    # 第三步: 触发统计（最优候选）
    # ═══════════════════════════════════════════════
    doc.append("## 第三步: 触发统计\n")

    best_rule = [r for n, r, x in candidates if n == best_name][0]
    best_x = [x for n, r, x in candidates if n == best_name][0]

    for sym in ['IM', 'IC']:
        adj, trig, avoided, missed = simulate_decay_exit(all_trades_raw[sym], best_rule, best_x, sym)
        triggered_trades = [t for t in adj if t.get('decay_triggered')]
        not_triggered = [t for t in adj if not t.get('decay_triggered')]

        orig_triggered_pnl = sum(t['pnl_pts'] for t in all_trades_raw[sym]
                                  if t.get('trade_date') in {tt['trade_date'] for tt in triggered_trades})

        doc.append(f"### {sym} ({best_name})\n")
        doc.append(f"- 触发: {trig}笔 ({trig/len(adj)*100:.0f}%)")
        doc.append(f"- 触发trade的新PnL: {sum(t['pnl_pts'] for t in triggered_trades):+.0f}pt "
                   f"(avg={np.mean([t['pnl_pts'] for t in triggered_trades]):+.1f})")
        doc.append(f"- 不触发trade的PnL: {sum(t['pnl_pts'] for t in not_triggered):+.0f}pt "
                   f"(avg={np.mean([t['pnl_pts'] for t in not_triggered]):+.1f})")
        doc.append(f"- 避免的亏损: {avoided:+.0f}pt")
        doc.append(f"- 错过的盈利: {missed:+.0f}pt")
        doc.append(f"- 净改善: {avoided - missed:+.0f}pt\n")

    # ═══════════════════════════════════════════════
    # 第四步: IS/OOS验证
    # ═══════════════════════════════════════════════
    doc.append("## 第四步: IS/OOS验证\n")

    # 选top3
    top_names = sorted([n for n, _, _ in candidates if n != '基线'],
                       key=lambda n: sweep_results[n]['total'], reverse=True)[:3]

    doc.append(f"| 候选 | IM_IS | IM_OOS | IM_OOS改善 | IC_IS | IC_OOS | IC_OOS改善 |")
    doc.append(f"|------|-------|--------|----------|-------|--------|----------|")

    for cand_name in ['基线'] + top_names:
        rule = [r for n, r, x in candidates if n == cand_name][0]
        x_val = [x for n, r, x in candidates if n == cand_name][0]

        row = [cand_name]
        baseline_oos_vals = {}
        for sym in ['IM', 'IC']:
            dates = all_dates[sym]
            split = int(len(dates) * 2 / 3)
            is_dates_set = set(dates[:split])
            oos_dates_set = set(dates[split:])

            is_trades = [t for t in all_trades_raw[sym] if t['trade_date'] in is_dates_set]
            oos_trades = [t for t in all_trades_raw[sym] if t['trade_date'] in oos_dates_set]

            if rule is None:
                is_pnl = sum(t['pnl_pts'] for t in is_trades)
                oos_pnl = sum(t['pnl_pts'] for t in oos_trades)
            else:
                is_adj, _, _, _ = simulate_decay_exit(is_trades, rule, x_val, sym)
                oos_adj, _, _, _ = simulate_decay_exit(oos_trades, rule, x_val, sym)
                is_pnl = sum(t['pnl_pts'] for t in is_adj)
                oos_pnl = sum(t['pnl_pts'] for t in oos_adj)

            if cand_name == '基线':
                baseline_oos_vals[sym] = oos_pnl

            oos_diff = oos_pnl - baseline_oos_vals.get(sym, oos_pnl)
            row.extend([f"{is_pnl:+.0f}", f"{oos_pnl:+.0f}", f"{oos_diff:+.0f}"])

        doc.append(f"| {' | '.join(row)} |")
    doc.append("")

    # ═══════════════════════════════════════════════
    # 第五步: 跟G候选叠加
    # ═══════════════════════════════════════════════
    doc.append("## 第五步: 跟G候选叠加\n")

    G_RANGES = {'IM': [(55, 90), (95, 100)], 'IC': [(60, 65), (70, 100)]}

    def in_ranges(score, ranges):
        return any(lo <= score < hi for lo, hi in ranges)

    configs = {
        '基线E': (None, None, None),
        'G候选': ('G_only', None, None),
        f'{best_name}': (None, best_rule, best_x),
        f'G+{best_name}': ('G_only', best_rule, best_x),
    }

    doc.append("| 配置 | IM_PnL | IC_PnL | 合计 | vs基线 |")
    doc.append("|------|--------|--------|------|--------|")

    for cfg_name, (g_filter, rule, x_val) in configs.items():
        total = 0
        im_p = 0; ic_p = 0
        for sym in ['IM', 'IC']:
            trades = all_trades_raw[sym]
            if g_filter:
                trades = [t for t in trades if in_ranges(t.get('entry_score', 0), G_RANGES[sym])]
            if rule:
                trades, _, _, _ = simulate_decay_exit(trades, rule, x_val, sym)
            pnl = sum(t['pnl_pts'] for t in trades)
            if sym == 'IM': im_p = pnl
            else: ic_p = pnl
        total = im_p + ic_p
        diff = total - baseline_total
        doc.append(f"| {cfg_name} | {im_p:+.0f} | {ic_p:+.0f} | {total:+.0f} | {diff:+.0f} |")
    doc.append("")

    # ═══════════════════════════════════════════════
    # 第六步: 成本敏感性
    # ═══════════════════════════════════════════════
    doc.append("## 第六步: 成本敏感性\n")

    costs = [0, 0.5, 0.8, 1.0, 1.5, 2.0]
    doc.append("| 配置 | " + " | ".join(f"c={c}" for c in costs) + " |")
    doc.append("|------|" + "|".join("------" for _ in costs) + "|")

    for cfg_name in ['基线E', f'G+{best_name}']:
        g_filter, rule, x_val = configs[cfg_name]
        cells = []
        for cost in costs:
            total_adj = 0
            for sym in ['IM', 'IC']:
                trades = all_trades_raw[sym]
                if g_filter:
                    trades = [t for t in trades if in_ranges(t.get('entry_score', 0), G_RANGES[sym])]
                if rule:
                    trades, _, _, _ = simulate_decay_exit(trades, rule, x_val, sym)
                pnl = sum(t['pnl_pts'] for t in trades) - len(trades) * cost
                total_adj += pnl
            cells.append(f"{total_adj:+.0f}")
        doc.append(f"| {cfg_name} | " + " | ".join(cells) + " |")
    doc.append("")

    # ═══════════════════════════════════════════════
    # 第七步: 综合判定
    # ═══════════════════════════════════════════════
    doc.append("## 第七步: 综合判定\n")
    doc.append(f"全样本最优: {best_name} (改善{improve_pct:+.1f}%)")

    if improve_pct >= 15:
        doc.append(f"\n**判定R1: Score衰减出场是革命性发现** ✓")
    elif improve_pct >= 5:
        doc.append(f"\n**判定R2: 显著但有限改进**")
    else:
        doc.append(f"\n**判定R3: 没有显著改善** ✗")

    report = "\n".join(doc)
    path = Path("tmp") / "score_decay_exit_simulation.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
