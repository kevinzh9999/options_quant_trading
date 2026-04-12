#!/usr/bin/env python3
"""快速检查：M衰减触发的trade里，"该出场"vs"不该出场"的V/Q模式差异。"""
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
            t['price_path'] = []
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
            t['m_path'] = []; t['v_path'] = []; t['q_path'] = []; t['price_path'] = []
            continue

        entry_utc = _bj_to_utc(entry_bj)
        exit_utc = _bj_to_utc(exit_bj)
        m_path, v_path, q_path, price_path = [], [], [], []
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
                    m_path.append(result.get('s_momentum', 0))
                    v_path.append(result.get('s_volatility', 0))
                    q_path.append(result.get('s_volume', 0))
                else:
                    m_path.append(0); v_path.append(0); q_path.append(0)

        t['m_path'] = m_path; t['v_path'] = v_path; t['q_path'] = q_path
        t['price_path'] = price_path
    return full


def main():
    print("=" * 60)
    print("  M衰减触发trade的V/Q模式差异检查")
    print("=" * 60)

    db = get_db()
    n_workers = min(cpu_count(), 8)
    doc = ["# M衰减触发trade的V/Q模式差异\n"]
    doc.append("核心问题：M衰减触发的trade里，'会被STOP_LOSS切的'(该出场) vs '会被TREND_COMPLETE走完的'(不该出场)，")
    doc.append("V分和Q分在触发时刻有没有区别？\n")

    for sym in ['IM', 'IC']:
        print(f"\n[{sym}] 收集数据...")
        dates = get_dates(db, SPOTS[sym])
        thr = SYMBOL_PROFILES[sym].get("signal_threshold", 60)
        args = [(td, sym, thr) for td in dates]
        with Pool(n_workers) as pool:
            day_results = pool.map(_run_one_day, args)
        trades = [t for day in day_results for t in day]
        valid = [t for t in trades if len(t.get('m_path', [])) >= 5]
        print(f"  {len(valid)}笔有效")

        # 找M衰减触发的trade (R2_X25: M从入场衰减>=25)
        triggered = []
        for t in valid:
            mp = t['m_path']
            for i in range(2, len(mp)):
                if mp[0] - mp[i] >= 25:
                    # 记录触发时刻的V和Q状态
                    vp = t['v_path']
                    qp = t['q_path']
                    t['trigger_bar'] = i
                    t['v_at_trigger'] = vp[i] if i < len(vp) else 0
                    t['v_at_entry'] = vp[0] if len(vp) > 0 else 0
                    t['v_decay_at_trigger'] = vp[0] - vp[i] if i < len(vp) else 0
                    t['v_max_to_trigger'] = max(vp[:i+1]) - vp[i] if i < len(vp) else 0
                    t['q_at_trigger'] = qp[i] if i < len(qp) else 0
                    t['q_at_entry'] = qp[0] if len(qp) > 0 else 0
                    t['q_decay_at_trigger'] = qp[0] - qp[i] if i < len(qp) else 0
                    triggered.append(t)
                    break

        if not triggered:
            doc.append(f"## {sym}: 无触发trade\n")
            continue

        tdf = pd.DataFrame(triggered)
        reason_col = 'reason' if 'reason' in tdf.columns else 'exit_reason'

        # 分成两组：会被STOP_LOSS vs 会被TREND_COMPLETE
        stop_group = tdf[tdf[reason_col].str.contains('STOP', na=False)]
        trend_group = tdf[tdf[reason_col].str.contains('TREND', na=False)]

        doc.append(f"## {sym} ({len(triggered)}笔M衰减触发)\n")
        doc.append(f"- 会被STOP_LOSS: {len(stop_group)}笔 (该出场)")
        doc.append(f"- 会被TREND_COMPLETE: {len(trend_group)}笔 (不该出场)\n")

        if len(stop_group) >= 30 and len(trend_group) >= 30:
            doc.append("### 触发时刻的V/Q状态对比\n")
            doc.append("| 指标 | STOP_LOSS组(该出场) | TREND_COMPLETE组(不该出场) | 差异 | 可区分? |")
            doc.append("|------|-------------------|-------------------------|------|--------|")

            metrics = [
                ('V分(入场)', 'v_at_entry'),
                ('V分(触发时)', 'v_at_trigger'),
                ('V分衰减(入场→触发)', 'v_decay_at_trigger'),
                ('V分衰减(峰值→触发)', 'v_max_to_trigger'),
                ('Q分(入场)', 'q_at_entry'),
                ('Q分(触发时)', 'q_at_trigger'),
                ('Q分衰减(入场→触发)', 'q_decay_at_trigger'),
                ('触发bar位置', 'trigger_bar'),
                ('M分(入场)', lambda df: df['m_path'].apply(lambda x: x[0])),
            ]

            for label, col in metrics:
                if callable(col):
                    s_val = col(stop_group).mean()
                    t_val = col(trend_group).mean()
                else:
                    s_val = stop_group[col].mean()
                    t_val = trend_group[col].mean()
                diff = s_val - t_val
                # 判断可区分性
                if label.startswith('V'):
                    thresh = 3
                elif label.startswith('Q'):
                    thresh = 2
                else:
                    thresh = 2
                separable = "**是**" if abs(diff) >= thresh else "否"
                doc.append(f"| {label} | {s_val:.1f} | {t_val:.1f} | {diff:+.1f} | {separable} |")

            # 额外：V分是否衰减的二分检查
            doc.append("\n### V分衰减二分检查\n")
            doc.append("| V分状态(触发时) | STOP_LOSS笔数 | STOP_LOSS占比 | TREND_COMPLETE笔数 | TREND_COMPLETE占比 |")
            doc.append("|---------------|-------------|-------------|-------------------|-------------------|")

            for v_label, v_cond in [
                ('V未衰减(衰减<3)', lambda df: df['v_decay_at_trigger'] < 3),
                ('V小衰减(3-9)', lambda df: (df['v_decay_at_trigger'] >= 3) & (df['v_decay_at_trigger'] < 9)),
                ('V严重衰减(>=9)', lambda df: df['v_decay_at_trigger'] >= 9),
            ]:
                s_n = v_cond(stop_group).sum()
                t_n = v_cond(trend_group).sum()
                s_pct = s_n / len(stop_group) * 100
                t_pct = t_n / len(trend_group) * 100
                doc.append(f"| {v_label} | {s_n} ({s_pct:.0f}%) | | {t_n} ({t_pct:.0f}%) | |")

            # Q分二分检查
            doc.append("\n### Q分衰减二分检查\n")
            doc.append("| Q分状态(触发时) | STOP_LOSS笔数(占比) | TREND_COMPLETE笔数(占比) |")
            doc.append("|---------------|-------------------|----------------------|")
            for q_label, q_cond in [
                ('Q未衰减(衰减<2)', lambda df: df['q_decay_at_trigger'] < 2),
                ('Q衰减(>=2)', lambda df: df['q_decay_at_trigger'] >= 2),
            ]:
                s_n = q_cond(stop_group).sum()
                t_n = q_cond(trend_group).sum()
                doc.append(f"| {q_label} | {s_n} ({s_n/len(stop_group)*100:.0f}%) | {t_n} ({t_n/len(trend_group)*100:.0f}%) |")

            # 组合检查：M衰减 + V也衰减 vs M衰减 + V不衰减
            doc.append("\n### 关键组合：M衰减时V是否同时衰减\n")
            doc.append("| 组合 | 笔数 | 该出场(STOP) | 不该出场(TREND) | 该出场占比 |")
            doc.append("|------|------|-----------|-------------|---------|")

            m_trig_all = tdf  # 所有M衰减触发的trade
            for combo_label, combo_cond in [
                ('M衰减+V不衰减', m_trig_all['v_decay_at_trigger'] < 5),
                ('M衰减+V也衰减(>=5)', m_trig_all['v_decay_at_trigger'] >= 5),
                ('M衰减+V严重衰减(>=9)', m_trig_all['v_decay_at_trigger'] >= 9),
                ('M衰减+Q不衰减', m_trig_all['q_decay_at_trigger'] < 5),
                ('M衰减+Q也衰减(>=5)', m_trig_all['q_decay_at_trigger'] >= 5),
            ]:
                sub = m_trig_all[combo_cond]
                if len(sub) < 20:
                    continue
                n_stop = sub[sub[reason_col].str.contains('STOP', na=False)].shape[0]
                n_trend = sub[sub[reason_col].str.contains('TREND', na=False)].shape[0]
                stop_pct = n_stop / (n_stop + n_trend) * 100 if (n_stop + n_trend) > 0 else 0
                # 也看这个组合的实际PnL
                avg_pnl = sub['pnl_pts'].mean()
                doc.append(f"| {combo_label} | {len(sub)} | {n_stop} | {n_trend} | {stop_pct:.0f}% (avg={avg_pnl:+.1f}) |")

        doc.append("")

    doc.append("## 结论\n")
    doc.append("(V/Q能否区分M衰减触发的'该出场'和'不该出场'？)")

    report = "\n".join(doc)
    path = Path("tmp") / "m_decay_vq_filter_check.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
