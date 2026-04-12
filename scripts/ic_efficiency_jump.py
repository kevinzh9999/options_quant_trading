#!/usr/bin/env python3
"""IC效率跳变诊断：IS 4倍于OOS的原因分析。复用已有trade数据+bar数据。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from data.storage.db_manager import get_db
from scripts.backtest_signals_day import run_day
from strategies.intraday.A_share_momentum_signal_v2 import SYMBOL_PROFILES

SYM = 'IC'
SPOT = '000905'


def get_dates(db):
    df = db.query_df(
        f"SELECT DISTINCT substr(datetime,1,10) as d FROM index_min "
        f"WHERE symbol='{SPOT}' AND period=300 ORDER BY d"
    )
    dates = [d.replace('-', '') for d in df['d'].tolist()]
    return dates[-900:] if len(dates) > 900 else dates


def _run_one_day(args):
    td, thr = args
    SYMBOL_PROFILES[SYM]["signal_threshold"] = thr
    db = get_db()
    trades = run_day(SYM, td, db, verbose=False)
    full = [t for t in trades if not t.get("partial")]
    for t in full:
        t['trade_date'] = td
        t['symbol'] = SYM
        eb = t.get('entry_time', '00:00')
        xb = t.get('exit_time', '00:00')
        try:
            eh, em = int(eb[:2]), int(eb[3:5])
            xh, xm = int(xb[:2]), int(xb[3:5])
            t['hold_minutes'] = (xh - eh) * 60 + (xm - em)
            t['hold_bars'] = max(1, t['hold_minutes'] // 5)
        except Exception:
            t['hold_minutes'] = 0
            t['hold_bars'] = 0
    return full


def main():
    print("=" * 60)
    print("  IC 效率跳变诊断")
    print("=" * 60)

    db = get_db()
    dates = get_dates(db)
    n = len(dates)
    is_dates_list = dates[-219:]
    oos_dates_list = dates[:-219]
    is_set = set(is_dates_list)
    oos_set = set(oos_dates_list)

    doc = ["# IC 效率跳变诊断\n"]
    doc.append(f"数据: IC {n}天 ({dates[0]}~{dates[-1]})")
    doc.append(f"IS(训练窗口): 最近219天 ({is_dates_list[0]}~{is_dates_list[-1]})")
    doc.append(f"OOS: 早期681天 ({oos_dates_list[0]}~{oos_dates_list[-1]})\n")

    # ═══════════════════════════════════════════════
    # Step 1: 市场环境对比
    # ═══════════════════════════════════════════════
    print("[Step 1] 加载bar数据计算市场环境...")
    bar_df = db.query_df(
        f"SELECT datetime, open, high, low, close, volume FROM index_min "
        f"WHERE symbol='{SPOT}' AND period=300 ORDER BY datetime"
    )
    for c in ['open', 'high', 'low', 'close', 'volume']:
        bar_df[c] = bar_df[c].astype(float)
    bar_df.index = pd.to_datetime(bar_df['datetime'])
    bar_df['date'] = bar_df.index.strftime('%Y%m%d')

    # 日级统计
    daily_stats = []
    for date_str, day_bars in bar_df.groupby('date'):
        if len(day_bars) < 10:
            continue
        dh = day_bars['high'].max()
        dl = day_bars['low'].min()
        do = float(day_bars.iloc[0]['open'])
        dc = float(day_bars.iloc[-1]['close'])
        vol = day_bars['volume'].sum()
        amp_pct = (dh - dl) / do * 100 if do > 0 else 0

        # ATR(5) proxy: 用当天的bar级true range均值
        tr = np.maximum(day_bars['high'] - day_bars['low'],
                        np.maximum(abs(day_bars['high'] - day_bars['close'].shift(1)),
                                   abs(day_bars['low'] - day_bars['close'].shift(1))))
        atr5 = tr.tail(5).mean() if len(tr) >= 5 else tr.mean()

        daily_stats.append({
            'date': date_str, 'amp_pct': amp_pct, 'atr5': atr5,
            'volume': vol, 'close': dc, 'open': do,
            'up': dc > do,
        })

    ddf = pd.DataFrame(daily_stats)
    ddf_is = ddf[ddf['date'].isin(is_set)]
    ddf_oos = ddf[ddf['date'].isin(oos_set)]

    doc.append("## Step 1: 市场环境对比\n")
    doc.append("| 指标 | OOS 681天 | IS 219天 | 差异 |")
    doc.append("|------|----------|---------|------|")

    metrics = [
        ('日均振幅(%)', 'amp_pct', '{:.2f}'),
        ('日均ATR(5bar)', 'atr5', '{:.1f}'),
        ('日均成交量', 'volume', '{:.0f}'),
        ('上涨天占比', 'up', '{:.0f}%'),
    ]
    for label, col, fmt in metrics:
        oos_v = ddf_oos[col].mean()
        is_v = ddf_is[col].mean()
        if col == 'up':
            oos_s = f"{oos_v*100:.0f}%"
            is_s = f"{is_v*100:.0f}%"
            diff = f"{(is_v-oos_v)*100:+.0f}%"
        else:
            oos_s = fmt.format(oos_v)
            is_s = fmt.format(is_v)
            diff_v = (is_v - oos_v) / oos_v * 100 if oos_v != 0 else 0
            diff = f"{diff_v:+.0f}%"
        doc.append(f"| {label} | {oos_s} | {is_s} | {diff} |")

    # 价格水平
    oos_price = ddf_oos['close'].mean()
    is_price = ddf_is['close'].mean()
    doc.append(f"| 平均收盘价 | {oos_price:.0f} | {is_price:.0f} | {(is_price-oos_price)/oos_price*100:+.0f}% |")
    doc.append("")

    # ═══════════════════════════════════════════════
    # Step 2: 信号质量对比（需要trade数据）
    # ═══════════════════════════════════════════════
    print("[Step 2] 收集IC交易数据...")
    thr = SYMBOL_PROFILES[SYM].get("signal_threshold", 60)
    n_workers = min(cpu_count(), 8)
    args = [(td, thr) for td in dates]
    with Pool(n_workers) as pool:
        day_results = pool.map(_run_one_day, args)
    all_trades = [t for day in day_results for t in day]
    tdf = pd.DataFrame(all_trades)
    print(f"  {len(tdf)}笔")

    tdf_is = tdf[tdf['trade_date'].isin(is_set)]
    tdf_oos = tdf[tdf['trade_date'].isin(oos_set)]

    doc.append("## Step 2: 信号质量对比\n")
    doc.append("| 指标 | OOS 681天 | IS 219天 | 差异 |")
    doc.append("|------|----------|---------|------|")
    doc.append(f"| 总笔数 | {len(tdf_oos)} | {len(tdf_is)} | |")
    doc.append(f"| 日均信号数 | {len(tdf_oos)/681:.1f} | {len(tdf_is)/219:.1f} | {len(tdf_is)/219 - len(tdf_oos)/681:+.1f} |")
    doc.append(f"| 总PnL | {tdf_oos['pnl_pts'].sum():+.0f} | {tdf_is['pnl_pts'].sum():+.0f} | |")
    doc.append(f"| 日均PnL | {tdf_oos['pnl_pts'].sum()/681:+.1f} | {tdf_is['pnl_pts'].sum()/219:+.1f} | |")

    # Score子分量
    for col, label in [('entry_score', 'AvgScore'), ('entry_m_score', 'M分'),
                        ('entry_v_score', 'V分'), ('entry_q_score', 'Q分')]:
        if col in tdf.columns:
            oos_v = tdf_oos[col].mean()
            is_v = tdf_is[col].mean()
            doc.append(f"| {label} | {oos_v:.1f} | {is_v:.1f} | {is_v-oos_v:+.1f} |")

    # 胜率
    oos_wr = (tdf_oos['pnl_pts'] > 0).sum() / len(tdf_oos) * 100
    is_wr = (tdf_is['pnl_pts'] > 0).sum() / len(tdf_is) * 100
    doc.append(f"| 胜率 | {oos_wr:.0f}% | {is_wr:.0f}% | {is_wr-oos_wr:+.0f}% |")

    # 平均PnL
    doc.append(f"| AvgPnL | {tdf_oos['pnl_pts'].mean():+.1f} | {tdf_is['pnl_pts'].mean():+.1f} | {tdf_is['pnl_pts'].mean()-tdf_oos['pnl_pts'].mean():+.1f} |")
    doc.append("")

    # ═══════════════════════════════════════════════
    # Step 3: 时间分布
    # ═══════════════════════════════════════════════
    doc.append("## Step 3: 时间分布\n")

    # IS按月
    doc.append("### IS 219天月度PnL\n")
    doc.append("| 月份 | 笔数 | PnL | 日均PnL | 占IS总PnL% |")
    doc.append("|------|------|-----|---------|----------|")
    tdf_is_copy = tdf_is.copy()
    tdf_is_copy['month'] = tdf_is_copy['trade_date'].apply(lambda x: x[:6])
    is_total = tdf_is['pnl_pts'].sum()
    for month in sorted(tdf_is_copy['month'].unique()):
        sub = tdf_is_copy[tdf_is_copy['month'] == month]
        days_in_month = len(set(sub['trade_date']))
        pnl = sub['pnl_pts'].sum()
        daily = pnl / days_in_month if days_in_month > 0 else 0
        contrib = pnl / is_total * 100 if is_total != 0 else 0
        doc.append(f"| {month} | {len(sub)} | {pnl:+.0f} | {daily:+.1f} | {contrib:+.0f}% |")

    # OOS按半年
    doc.append(f"\n### OOS 681天半年PnL\n")
    doc.append("| 时段 | 笔数 | PnL | 日均PnL |")
    doc.append("|------|------|-----|---------|")
    tdf_oos_copy = tdf_oos.copy()
    periods = [
        ('2022H2', '20220701', '20221231'),
        ('2023H1', '20230101', '20230630'),
        ('2023H2', '20230701', '20231231'),
        ('2024H1', '20240101', '20240630'),
        ('2024H2', '20240701', '20241231'),
        ('2025H1', '20250101', '20250519'),
    ]
    for label, start, end in periods:
        sub = tdf_oos_copy[(tdf_oos_copy['trade_date'] >= start) & (tdf_oos_copy['trade_date'] <= end)]
        if len(sub) == 0:
            continue
        days = len(set(sub['trade_date']))
        pnl = sub['pnl_pts'].sum()
        daily = pnl / days if days > 0 else 0
        doc.append(f"| {label} | {len(sub)} | {pnl:+.0f} | {daily:+.1f} |")
    doc.append("")

    # ═══════════════════════════════════════════════
    # Step 4: 出场原因对比
    # ═══════════════════════════════════════════════
    doc.append("## Step 4: 出场原因对比\n")
    reason_col = 'reason' if 'reason' in tdf.columns else 'exit_reason'

    doc.append("| exit_reason | OOS笔数 | OOS_AvgPnL | IS笔数 | IS_AvgPnL | 差异 |")
    doc.append("|------------|--------|-----------|--------|----------|------|")

    all_reasons = sorted(tdf[reason_col].unique())
    for r in all_reasons:
        oos_sub = tdf_oos[tdf_oos[reason_col] == r]
        is_sub = tdf_is[tdf_is[reason_col] == r]
        if len(oos_sub) < 10 and len(is_sub) < 10:
            continue
        oos_avg = oos_sub['pnl_pts'].mean() if len(oos_sub) > 0 else 0
        is_avg = is_sub['pnl_pts'].mean() if len(is_sub) > 0 else 0
        doc.append(f"| {r} | {len(oos_sub)} | {oos_avg:+.1f} | {len(is_sub)} | {is_avg:+.1f} | {is_avg-oos_avg:+.1f} |")
    doc.append("")

    # ═══════════════════════════════════════════════
    # Step 5: 综合判定
    # ═══════════════════════════════════════════════
    doc.append("## Step 5: 综合判定\n")

    # 自动化判定逻辑
    amp_oos = ddf_oos['amp_pct'].mean()
    amp_is = ddf_is['amp_pct'].mean()
    amp_change = (amp_is - amp_oos) / amp_oos * 100

    wr_change = is_wr - oos_wr
    avg_pnl_change = tdf_is['pnl_pts'].mean() - tdf_oos['pnl_pts'].mean()

    doc.append(f"**关键指标变化:**")
    doc.append(f"- 市场振幅: OOS {amp_oos:.2f}% → IS {amp_is:.2f}% (变化{amp_change:+.0f}%)")
    doc.append(f"- 胜率: OOS {oos_wr:.0f}% → IS {is_wr:.0f}% (变化{wr_change:+.0f}%)")
    doc.append(f"- 单笔PnL: OOS {tdf_oos['pnl_pts'].mean():+.1f} → IS {tdf_is['pnl_pts'].mean():+.1f} (变化{avg_pnl_change:+.1f})")
    doc.append(f"- IS/OOS效率比: {(tdf_is['pnl_pts'].sum()/219) / (tdf_oos['pnl_pts'].sum()/681):.2f}\n")

    if abs(amp_change) > 20:
        doc.append(f"**振幅变化显著({amp_change:+.0f}%)**，市场环境是主要因素")
    if wr_change > 5:
        doc.append(f"**胜率提升显著({wr_change:+.0f}%)**，信号质量有改善")
    doc.append("")

    doc.append("(根据以上数据选择E1/E2/E3/E4)")

    report = "\n".join(doc)
    path = Path("tmp") / "ic_efficiency_jump_diagnostic.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
