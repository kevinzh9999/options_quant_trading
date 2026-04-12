#!/usr/bin/env python3
"""Score衰减信号研究: 持仓期间score演变与PnL的关系。

方法: 修改run_day在持仓期间记录每根bar的score到score_path。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from copy import deepcopy
from data.storage.db_manager import get_db
from scripts.backtest_signals_day import run_day as _original_run_day
from strategies.intraday.A_share_momentum_signal_v2 import SYMBOL_PROFILES

SPOTS = {'IC': '000905', 'IM': '000852'}


# Monkey-patch方案: 在run_day内部的循环里，持仓时记录score到position['score_path']
# 由于run_day是一个复杂的函数，最干净的方式是直接调用它并在trade结果中
# 添加score_path字段。
#
# 但run_day的position是局部变量，无法从外部注入。
#
# 最实际的方案: 不修改run_day，而是对每笔已知trade，
# 用独立的score_all调用在持仓bar上重算score。

def get_dates(db, spot):
    df = db.query_df(
        f"SELECT DISTINCT substr(datetime,1,10) as d FROM index_min "
        f"WHERE symbol='{spot}' AND period=300 ORDER BY d"
    )
    dates = [d.replace('-', '') for d in df['d'].tolist()]
    return dates[-900:] if len(dates) > 900 else dates


def _run_one_day(args):
    """收集trade + 在持仓bar上重算score。"""
    td, sym, thr = args
    SYMBOL_PROFILES[sym]["signal_threshold"] = thr
    db = get_db()

    # 正常backtest获取trade
    trades = _original_run_day(sym, td, db, verbose=False)
    full = [t for t in trades if not t.get("partial")]

    if not full:
        return []

    # 加载当天bar数据用于score重算
    from strategies.intraday.A_share_momentum_signal_v2 import SignalGeneratorV2
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

    # 需要历史bar做lookback
    day_end_idx = df.index.get_indexer(df[day_mask].index)
    if len(day_end_idx) == 0:
        for t in full:
            t['trade_date'] = td
            t['symbol'] = sym
            t['score_path'] = []
        return full

    last_day_idx = day_end_idx[-1]
    start_idx = max(0, day_end_idx[0] - 199)
    all_bars = df.iloc[start_idx:last_day_idx + 1]

    today_indices = df[day_mask].index.tolist()

    gen = SignalGeneratorV2({"min_signal_score": 50})

    # 加载vol_profile
    vol_profile = None
    try:
        lookback_start = pd.Timestamp(td_fmt) - pd.Timedelta(days=30)
        vol_df = db.query_df(
            f"SELECT datetime, volume FROM index_min "
            f"WHERE symbol='{spot}' AND period=300 "
            f"AND datetime >= '{lookback_start.strftime('%Y-%m-%d')}' "
            f"AND datetime < '{td_fmt}' ORDER BY datetime"
        )
        if len(vol_df) > 20:
            vol_profile = vol_df['volume'].astype(float).values
    except Exception:
        pass

    # 对每笔trade，找到持仓期间的bar并逐根算score
    for t in full:
        t['trade_date'] = td
        t['symbol'] = sym

        entry_time_bj = t.get('entry_time', '')
        exit_time_bj = t.get('exit_time', '')
        direction = t.get('direction', '')

        if not entry_time_bj or not exit_time_bj:
            t['score_path'] = []
            continue

        # BJ time → UTC time for matching
        def bj_to_utc_str(bj):
            try:
                h, m = int(bj[:2]), int(bj[3:5])
                h -= 8
                if h < 0: h += 24
                return f"{h:02d}:{m:02d}"
            except Exception:
                return ""

        entry_utc = bj_to_utc_str(entry_time_bj)
        exit_utc = bj_to_utc_str(exit_time_bj)

        score_path = []
        in_holding = False

        for bar_i, idx in enumerate(today_indices):
            dt_str = str(df.loc[idx, 'datetime'])
            bar_utc = dt_str[11:16]
            # +5min for execution time
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
                # Build bar_5m window
                pos_in_all = all_bars.index.get_loc(idx)
                window_start = max(0, pos_in_all - 198)
                bar_5m = all_bars.iloc[window_start:pos_in_all + 1]

                # Build 15m
                try:
                    bar_15m_full = bar_5m.resample('15min', label='left', closed='left').agg({
                        'open': 'first', 'high': 'max', 'low': 'min',
                        'close': 'last', 'volume': 'sum'
                    }).dropna()
                    bar_15m = bar_15m_full.iloc[:-1] if len(bar_15m_full) > 1 else bar_15m_full
                except Exception:
                    bar_15m = pd.DataFrame()

                # vol_profile传None避免numpy array truthiness bug
                result = gen.score_all(
                    sym, bar_5m, bar_15m if not bar_15m.empty else None,
                    None, None, None,
                    zscore=None, is_high_vol=True, d_override=None,
                    vol_profile=None,
                )

                if result:
                    # 取持仓方向对应的score
                    if result['direction'] == direction:
                        score_path.append(result['total'])
                    else:
                        # 反向信号，score为负（表示原方向支持减弱）
                        score_path.append(-result['total'])
                else:
                    score_path.append(0)

        t['score_path'] = score_path
        t['score_max'] = max(score_path) if score_path else 0
        t['score_min'] = min(score_path) if score_path else 0
        t['hold_bars'] = len(score_path)

    return full


def collect_with_score_path(sym, dates, thr):
    """多进程收集带score_path的trade。"""
    n_workers = min(cpu_count(), 8)
    args_list = [(td, sym, thr) for td in dates]
    with Pool(n_workers) as pool:
        day_results = pool.map(_run_one_day, args_list)
    return [t for day in day_results for t in day]


def analyze_decay(tdf, sym):
    """分析score衰减与PnL的关系。"""
    lines = []
    lines.append(f"### {sym} Score衰减分析 ({len(tdf)}笔)\n")

    # 计算衰减指标
    tdf = tdf.copy()
    tdf['entry_score'] = tdf.get('entry_score', tdf['score_path'].apply(lambda p: p[0] if p else 0))

    def calc_decay(row):
        sp = row['score_path']
        if not sp or len(sp) < 2:
            return pd.Series({'decay_total': 0, 'decay_from_peak': 0, 'peak_position': 0, 'early_drop_2bar': 0})
        entry = sp[0]
        exit_s = sp[-1]
        peak = max(sp)
        peak_idx = sp.index(peak)
        decay_total = entry - exit_s
        decay_from_peak = peak - exit_s
        peak_pos = peak_idx / (len(sp) - 1) if len(sp) > 1 else 0
        early_drop = entry - sp[min(2, len(sp)-1)]
        return pd.Series({'decay_total': decay_total, 'decay_from_peak': decay_from_peak,
                          'peak_position': peak_pos, 'early_drop_2bar': early_drop})

    decay_df = tdf.apply(calc_decay, axis=1)
    tdf = pd.concat([tdf, decay_df], axis=1)

    # 2.2 衰减跟PnL的关系
    lines.append("#### 衰减幅度 vs PnL\n")
    lines.append("| decay_total组 | 笔数 | AvgPnL | 胜率 |")
    lines.append("|-------------|------|--------|------|")
    bins_decay = [(-999, -10, '上升(d<-10)'), (-10, 10, '稳定(-10~10)'),
                  (10, 25, '小衰(10-25)'), (25, 40, '中衰(25-40)'), (40, 999, '严重(>40)')]
    for lo, hi, label in bins_decay:
        sub = tdf[(tdf['decay_total'] >= lo) & (tdf['decay_total'] < hi)]
        if len(sub) >= 30:
            wr = (sub['pnl_pts'] > 0).sum() / len(sub) * 100
            lines.append(f"| {label} | {len(sub)} | {sub['pnl_pts'].mean():+.1f} | {wr:.0f}% |")
        elif len(sub) > 0:
            lines.append(f"| {label} | {len(sub)} | (N<30) | |")

    # Pearson相关
    valid = tdf[tdf['hold_bars'] >= 2]
    if len(valid) >= 30:
        from scipy.stats import pearsonr
        r, p = pearsonr(valid['decay_total'], valid['pnl_pts'])
        lines.append(f"\nPearson相关(decay_total vs pnl): r={r:.3f}, p={p:.4f}")
        r2, p2 = pearsonr(valid['decay_from_peak'], valid['pnl_pts'])
        lines.append(f"Pearson相关(decay_from_peak vs pnl): r={r2:.3f}, p={p2:.4f}")

    # 2.3 peak位置
    lines.append("\n#### Score最高点位置 vs PnL\n")
    lines.append("| peak位置 | 笔数 | AvgPnL | 胜率 |")
    lines.append("|---------|------|--------|------|")
    for lo, hi, label in [(0, 0.2, '入场即最大'), (0.2, 0.5, '早期最大'),
                          (0.5, 0.8, '后期最大'), (0.8, 1.01, '出场前最大')]:
        sub = tdf[(tdf['peak_position'] >= lo) & (tdf['peak_position'] < hi)]
        if len(sub) >= 30:
            wr = (sub['pnl_pts'] > 0).sum() / len(sub) * 100
            lines.append(f"| {label} | {len(sub)} | {sub['pnl_pts'].mean():+.1f} | {wr:.0f}% |")
        elif len(sub) > 0:
            lines.append(f"| {label} | {len(sub)} | (N<30) | |")

    # 3.3 入场后2bar跌幅作为早期信号
    lines.append("\n#### 入场后2bar内score跌幅 vs 后续PnL\n")
    lines.append("| 2bar内跌幅 | 笔数 | AvgPnL | 胜率 |")
    lines.append("|-----------|------|--------|------|")
    for lo, hi, label in [(-999, 0, '上升或持平'), (0, 10, '跌0-10'),
                          (10, 20, '跌10-20'), (20, 999, '跌>20')]:
        sub = tdf[(tdf['early_drop_2bar'] >= lo) & (tdf['early_drop_2bar'] < hi)]
        if len(sub) >= 30:
            wr = (sub['pnl_pts'] > 0).sum() / len(sub) * 100
            lines.append(f"| {label} | {len(sub)} | {sub['pnl_pts'].mean():+.1f} | {wr:.0f}% |")
        elif len(sub) > 0:
            lines.append(f"| {label} | {len(sub)} | (N<30) | |")

    lines.append("")
    return lines, tdf


def test_decay_exit_rules(tdf, sym):
    """假设性score衰减出场规则测试。"""
    lines = []
    lines.append(f"### {sym} 假设性出场规则测试\n")

    # 基线PnL
    baseline_pnl = tdf['pnl_pts'].sum()
    baseline_n = len(tdf)
    lines.append(f"基线: {baseline_n}笔, 累计{baseline_pnl:+.0f}pt\n")

    # 规则1: score从入场跌幅>=X就出场
    lines.append("#### 规则1: score从入场跌>=X\n")
    lines.append("| X | 触发笔数 | 触发后AvgPnL节省 | 估算总PnL变化 |")
    lines.append("|---|---------|---------------|------------|")
    for x in [15, 20, 25, 30]:
        # 找到decay_total >= x的trade
        triggered = tdf[tdf['decay_total'] >= x]
        if len(triggered) >= 10:
            # 假设在score跌到entry-x时就出场，大约在score_path中间
            # 简化估算：这些trade如果提前出场，PnL大约减半（因为在持仓中间出场）
            # 更精确的估算需要逐bar回放，这里用描述性统计
            avg_pnl_triggered = triggered['pnl_pts'].mean()
            lines.append(f"| {x} | {len(triggered)} ({len(triggered)/len(tdf)*100:.0f}%) | "
                         f"avg={avg_pnl_triggered:+.1f} | (需逐bar回放) |")
        else:
            lines.append(f"| {x} | {len(triggered)} | (N<10) | |")

    # 规则3: 入场后2bar内score跌>=Y就出场
    lines.append("\n#### 规则3: 入场后2bar内score跌>=Y\n")
    lines.append("| Y | 触发笔数 | 触发trade的AvgPnL | 不触发trade的AvgPnL |")
    lines.append("|---|---------|-----------------|-------------------|")
    for y in [10, 15, 20]:
        triggered = tdf[tdf['early_drop_2bar'] >= y]
        not_triggered = tdf[tdf['early_drop_2bar'] < y]
        if len(triggered) >= 10 and len(not_triggered) >= 10:
            lines.append(f"| {y} | {len(triggered)} ({len(triggered)/len(tdf)*100:.0f}%) | "
                         f"{triggered['pnl_pts'].mean():+.1f} | {not_triggered['pnl_pts'].mean():+.1f} |")
        else:
            lines.append(f"| {y} | {len(triggered)} | (样本不足) | |")

    lines.append("")
    return lines


def main():
    print("=" * 60)
    print("  Score衰减信号研究")
    print("=" * 60)

    db = get_db()
    doc = ["# Score衰减信号研究\n"]

    for sym in ['IM', 'IC']:
        print(f"\n[{sym}] 收集带score_path的交易数据...")
        dates = get_dates(db, SPOTS[sym])
        orig_thr = SYMBOL_PROFILES[sym].get("signal_threshold", 60)
        trades = collect_with_score_path(sym, dates, orig_thr)
        print(f"  {len(trades)}笔, 其中{sum(1 for t in trades if t.get('score_path'))}笔有score_path")

        tdf = pd.DataFrame(trades)
        # 过滤有score_path的
        tdf = tdf[tdf['score_path'].apply(lambda x: len(x) >= 2 if isinstance(x, list) else False)]
        print(f"  有效(>=2 bar score_path): {len(tdf)}笔")

        doc.append(f"## {sym}\n")
        doc.append(f"数据: {dates[0]}~{dates[-1]} ({len(dates)}天)")
        doc.append(f"有效trade: {len(tdf)}笔 (score_path >= 2 bar)\n")

        # 验证: 打印第一笔的score_path
        if len(tdf) > 0:
            first = tdf.iloc[0]
            doc.append(f"验证(第1笔): entry_score={first.get('entry_score',0)}, "
                       f"score_path={first['score_path'][:5]}..., hold_bars={first['hold_bars']}\n")

        # 衰减分析
        decay_lines, tdf_enriched = analyze_decay(tdf, sym)
        doc.extend(decay_lines)

        # 假设性出场规则
        rule_lines = test_decay_exit_rules(tdf_enriched, sym)
        doc.extend(rule_lines)

    # 综合判定
    doc.append("## 综合判定\n")
    doc.append("(根据以上数据选择D1/D2/D3)")

    report = "\n".join(doc)
    path = Path("tmp") / "score_decay_signal_research.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
