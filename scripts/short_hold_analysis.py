#!/usr/bin/env python3
"""IM/IC短持仓trade per-symbol差异描述性分析。"""
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
        # 计算hold_bars从entry/exit时间
        eb = t.get('entry_time', '00:00')
        xb = t.get('exit_time', '00:00')
        try:
            eh, em = int(eb[:2]), int(eb[3:5])
            xh, xm = int(xb[:2]), int(xb[3:5])
            mins = (xh - eh) * 60 + (xm - em)
            t['hold_minutes'] = mins
            t['hold_bars'] = max(1, mins // 5)
        except Exception:
            t['hold_minutes'] = 0
            t['hold_bars'] = 0
    return full


def hold_group(bars):
    if bars <= 2: return '极短(<=2bar)'
    if bars <= 4: return '短(3-4bar)'
    if bars <= 12: return '中(5-12bar)'
    return '长(>12bar)'


def main():
    print("=" * 60)
    print("  短持仓trade per-symbol差异分析")
    print("=" * 60)

    db = get_db()
    doc = ["# IM/IC 短持仓 Trade Per-Symbol 差异分析\n"]

    all_data = {}
    for sym in ['IM', 'IC']:
        dates = get_dates(db, SPOTS[sym])
        thr = SYMBOL_PROFILES[sym].get("signal_threshold", 60)
        print(f"\n[{sym}] 收集交易数据 (thr={thr})...")
        n_workers = min(cpu_count(), 8)
        args = [(td, sym, thr) for td in dates]
        with Pool(n_workers) as pool:
            day_results = pool.map(_run_one_day, args)
        trades = [t for day in day_results for t in day]
        tdf = pd.DataFrame(trades)
        tdf['hold_group'] = tdf['hold_bars'].apply(hold_group)
        all_data[sym] = tdf
        print(f"  {len(tdf)}笔")

    # ═══════════════════════════════════════════════
    # B.1 持仓时长精细分组
    # ═══════════════════════════════════════════════
    doc.append("## B.1 持仓时长精细分组\n")
    groups = ['极短(<=2bar)', '短(3-4bar)', '中(5-12bar)', '长(>12bar)']

    doc.append("| 品种 | 持仓时长 | 笔数 | 占比 | AvgPnL | 累计PnL | 胜率 |")
    doc.append("|------|---------|------|------|--------|---------|------|")
    for sym in ['IM', 'IC']:
        tdf = all_data[sym]
        total = len(tdf)
        for g in groups:
            sub = tdf[tdf['hold_group'] == g]
            if len(sub) == 0:
                continue
            wr = (sub['pnl_pts'] > 0).sum() / len(sub) * 100
            doc.append(f"| {sym} | {g} | {len(sub)} | {len(sub)/total*100:.0f}% | "
                       f"{sub['pnl_pts'].mean():+.1f} | {sub['pnl_pts'].sum():+.0f} | {wr:.0f}% |")
    doc.append("")

    # ═══════════════════════════════════════════════
    # B.2 短持仓的exit_reason分布
    # ═══════════════════════════════════════════════
    doc.append("## B.2 短持仓(<=4bar)的Exit Reason分布\n")
    doc.append("| 品种 | exit_reason | 笔数 | 占比 | AvgPnL |")
    doc.append("|------|-----------|------|------|--------|")

    for sym in ['IM', 'IC']:
        tdf = all_data[sym]
        short = tdf[tdf['hold_bars'] <= 4]
        if len(short) == 0:
            continue
        reason_col = 'reason' if 'reason' in short.columns else 'exit_reason'
        if reason_col not in short.columns:
            doc.append(f"| {sym} | (无exit_reason列) | | | |")
            continue
        for reason in short[reason_col].value_counts().index:
            sub = short[short[reason_col] == reason]
            if len(sub) >= 5:
                doc.append(f"| {sym} | {reason} | {len(sub)} | {len(sub)/len(short)*100:.0f}% | "
                           f"{sub['pnl_pts'].mean():+.1f} |")
    doc.append("")

    # ═══════════════════════════════════════════════
    # B.3 入场特征对比
    # ═══════════════════════════════════════════════
    doc.append("## B.3 入场特征对比\n")

    score_cols = ['entry_score', 'entry_m_score', 'entry_v_score', 'entry_q_score',
                  'entry_b_score', 'entry_s_score']

    doc.append("| 维度 | IM短(<=4bar) | IM长(>4bar) | IC短(<=4bar) | IC长(>4bar) |")
    doc.append("|------|-----------|-----------|-----------|-----------|")

    for col in score_cols:
        vals = []
        for sym in ['IM', 'IC']:
            tdf = all_data[sym]
            if col not in tdf.columns:
                vals.extend(['-', '-'])
                continue
            short = tdf[tdf['hold_bars'] <= 4]
            long_ = tdf[tdf['hold_bars'] > 4]
            vals.append(f"{short[col].mean():.1f}" if len(short) > 0 else '-')
            vals.append(f"{long_[col].mean():.1f}" if len(long_) > 0 else '-')
        doc.append(f"| {col.replace('entry_','')} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |")

    # 方向占比
    if 'direction' in all_data['IM'].columns:
        vals = []
        for sym in ['IM', 'IC']:
            tdf = all_data[sym]
            short = tdf[tdf['hold_bars'] <= 4]
            long_ = tdf[tdf['hold_bars'] > 4]
            s_long = (short['direction'] == 'LONG').sum() / len(short) * 100 if len(short) > 0 else 0
            l_long = (long_['direction'] == 'LONG').sum() / len(long_) * 100 if len(long_) > 0 else 0
            vals.extend([f"{s_long:.0f}%", f"{l_long:.0f}%"])
        doc.append(f"| LONG占比 | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |")
    doc.append("")

    # ═══════════════════════════════════════════════
    # B.4 方向性分析
    # ═══════════════════════════════════════════════
    doc.append("## B.4 方向性分析\n")
    doc.append("| 品种 | 持仓 | LONG笔 | LONG_PnL | LONG_WR | SHORT笔 | SHORT_PnL | SHORT_WR |")
    doc.append("|------|------|--------|---------|---------|---------|----------|---------|")

    for sym in ['IM', 'IC']:
        tdf = all_data[sym]
        if 'direction' not in tdf.columns:
            continue
        for g in ['极短(<=2bar)', '短(3-4bar)']:
            sub = tdf[tdf['hold_group'] == g]
            if len(sub) < 30:
                continue
            for d in ['LONG', 'SHORT']:
                ds = sub[sub['direction'] == d]
                if len(ds) < 10:
                    continue
            long_s = sub[sub['direction'] == 'LONG']
            short_s = sub[sub['direction'] == 'SHORT']
            l_wr = (long_s['pnl_pts'] > 0).sum() / len(long_s) * 100 if len(long_s) > 0 else 0
            s_wr = (short_s['pnl_pts'] > 0).sum() / len(short_s) * 100 if len(short_s) > 0 else 0
            doc.append(f"| {sym} | {g} | {len(long_s)} | {long_s['pnl_pts'].mean():+.1f} | {l_wr:.0f}% | "
                       f"{len(short_s)} | {short_s['pnl_pts'].mean():+.1f} | {s_wr:.0f}% |")
    doc.append("")

    # ═══════════════════════════════════════════════
    # B.5 综合解读
    # ═══════════════════════════════════════════════
    doc.append("## B.5 综合解读\n")

    # 检查止损差异假设
    im_short = all_data['IM'][all_data['IM']['hold_bars'] <= 4]
    ic_short = all_data['IC'][all_data['IC']['hold_bars'] <= 4]

    reason_col_im = 'reason' if 'reason' in im_short.columns else 'exit_reason'
    reason_col_ic = 'reason' if 'reason' in ic_short.columns else 'exit_reason'

    if reason_col_im in im_short.columns and reason_col_ic in ic_short.columns:
        im_sl = im_short[im_short[reason_col_im].str.contains('STOP|stop', na=False)]
        ic_sl = ic_short[ic_short[reason_col_ic].str.contains('STOP|stop', na=False)]
        im_sl_pct = len(im_sl) / len(im_short) * 100 if len(im_short) > 0 else 0
        ic_sl_pct = len(ic_sl) / len(ic_short) * 100 if len(ic_short) > 0 else 0
        im_sl_pnl = im_sl['pnl_pts'].mean() if len(im_sl) > 0 else 0
        ic_sl_pnl = ic_sl['pnl_pts'].mean() if len(ic_sl) > 0 else 0

        doc.append(f"### 止损假设检验(P1)\n")
        doc.append(f"- IM短持仓止损占比: {im_sl_pct:.0f}% ({len(im_sl)}笔, avg={im_sl_pnl:+.1f})")
        doc.append(f"- IC短持仓止损占比: {ic_sl_pct:.0f}% ({len(ic_sl)}笔, avg={ic_sl_pnl:+.1f})")
        doc.append(f"- IM stop_loss=0.3%, IC stop_loss=0.5%")
        if im_sl_pct > ic_sl_pct + 10:
            doc.append(f"- **P1支持**: IM止损更紧(0.3%)导致更多短持仓被止损打掉")
        else:
            doc.append(f"- **P1不支持**: 止损占比差异不显著")
        doc.append("")

    doc.append("(根据以上数据给出最佳解释P1-P5)")

    report = "\n".join(doc)
    path = Path("tmp") / "short_hold_per_symbol_diff.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
