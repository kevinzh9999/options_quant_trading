#!/usr/bin/env python3
"""v1_im和v1_ic双窗口验证 + 设计披露 + 成本效益。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from data.storage.db_manager import get_db
from scripts.backtest_signals_day import run_day
from strategies.intraday.A_share_momentum_signal_v2 import SYMBOL_PROFILES
from strategies.intraday.experimental.score_components_new import (
    compute_m_score, compute_v_score, compute_q_score,
    compute_session_bonus, compute_gap_bonus,
)

SPOTS = {'IC': '000905', 'IM': '000852'}
V2_THR = {'IM': 60, 'IC': 55}

# v1_im数据驱动映射（从Phase C结果硬编码）
V1_IM_M_THRESHOLDS = [
    (0, 0.00122, 0), (0.00122, 0.00191, 17), (0.00191, 0.00247, 11),
    (0.00247, 0.00319, 33), (0.00319, 0.00393, 28), (0.00393, 0.00497, 6),
    (0.00497, 0.00632, 22), (0.00632, 0.00872, 50), (0.00872, 0.0128, 39),
    (0.0128, 1.0, 44),
]
V1_IM_V_THRESHOLDS = [
    (0, 0.666, 30), (0.666, 0.776, 20), (0.776, 0.864, 3),
    (0.864, 0.958, 13), (0.958, 1.065, 10), (1.065, 1.215, 17),
    (1.215, 1.424, 0), (1.424, 1.655, 27), (1.655, 1.987, 7),
    (1.987, 100.0, 23),
]
V1_IM_SESSION = {9: 10, 10: -10, 11: 4, 13: 0, 14: 8}


def v1_im_m(raw_mom):
    for lo, hi, s in V1_IM_M_THRESHOLDS:
        if lo <= raw_mom < hi:
            return s
    return V1_IM_M_THRESHOLDS[-1][2]

def v1_im_v(raw_atr):
    for lo, hi, s in V1_IM_V_THRESHOLDS:
        if lo <= raw_atr < hi:
            return s
    return V1_IM_V_THRESHOLDS[-1][2]

def v1_im_session(h):
    return V1_IM_SESSION.get(h, 0)


def get_dates(db, spot):
    df = db.query_df(
        f"SELECT DISTINCT substr(datetime,1,10) as d FROM index_min "
        f"WHERE symbol='{spot}' AND period=300 ORDER BY d"
    )
    dates = [d.replace('-', '') for d in df['d'].tolist()]
    return dates[-900:] if len(dates) > 900 else dates


def _run_one_day(args):
    td, sym = args
    SYMBOL_PROFILES[sym]["signal_threshold"] = 20
    db = get_db()
    trades = run_day(sym, td, db, verbose=False)
    full = [t for t in trades if not t.get("partial")]
    for t in full:
        t['trade_date'] = td; t['symbol'] = sym
        t['_raw_mom'] = abs(t.get('raw_mom_5m', 0.0))
        t['_raw_atr'] = t.get('raw_atr_ratio', 0.0)
        t['_raw_vpct'] = t.get('raw_vol_pct', -1.0)
        t['_raw_vratio'] = t.get('raw_vol_ratio', -1.0)
        try: t['_hour_bj'] = int(t.get('entry_time', '13:00')[:2])
        except: t['_hour_bj'] = 13
        t['_gap_aligned'] = t.get('entry_gap_aligned', False)
    return full


def apply_v1_im(tdf):
    tdf = tdf.copy()
    tdf['v1_m'] = tdf['_raw_mom'].apply(v1_im_m)
    tdf['v1_v'] = tdf['_raw_atr'].apply(v1_im_v)
    tdf['v1_q'] = tdf.apply(lambda r: compute_q_score(r['_raw_vpct'], r['_raw_vratio']), axis=1)
    tdf['v1_session'] = tdf['_hour_bj'].apply(v1_im_session)
    tdf['v1_gap'] = tdf['_gap_aligned'].apply(compute_gap_bonus)
    tdf['v1_total'] = tdf['v1_m'] + tdf['v1_v'] + tdf['v1_q'] + tdf['v1_session'] + tdf['v1_gap']
    return tdf


def apply_v1_ic(tdf):
    tdf = tdf.copy()
    tdf['v1_m'] = tdf['_raw_mom'].apply(compute_m_score)
    tdf['v1_v'] = tdf['_raw_atr'].apply(compute_v_score)
    tdf['v1_q'] = tdf.apply(lambda r: compute_q_score(r['_raw_vpct'], r['_raw_vratio']), axis=1)
    tdf['v1_session'] = tdf['_hour_bj'].apply(compute_session_bonus)
    tdf['v1_gap'] = tdf['_gap_aligned'].apply(compute_gap_bonus)
    tdf['v1_total'] = tdf['v1_m'] + tdf['v1_v'] + tdf['v1_q'] + tdf['v1_session'] + tdf['v1_gap']
    return tdf


def dual_window_table(tdf, v1_thr, v2_thr, dates, doc):
    """双窗口验证表。"""
    is_set = set(dates[-219:])
    oos_set = set(dates[:-219])

    v2_all = tdf[tdf['entry_score'] >= v2_thr]
    v1_all = tdf[tdf['v1_total'] >= v1_thr]

    segments = [
        ('IS 219', is_set), ('OOS 681', oos_set), ('全样本 900', set(dates)),
    ]

    doc.append("| 指标 | v2 IS | v2 OOS | v1 IS | v1 OOS | v1 全样本 |")
    doc.append("|------|-------|--------|-------|--------|---------|")

    for metric_name in ['信号数', '总PnL', '单笔PnL', '胜率']:
        cells = []
        for sys_name, sys_trades in [('v2', v2_all), ('v1', v1_all)]:
            for seg_name, seg_set in segments:
                if sys_name == 'v2' and seg_name == '全样本 900':
                    continue
                sub = sys_trades[sys_trades['trade_date'].isin(seg_set)]
                n = len(sub)
                if n == 0:
                    cells.append('-')
                    continue
                if metric_name == '信号数':
                    cells.append(str(n))
                elif metric_name == '总PnL':
                    cells.append(f"{sub['pnl_pts'].sum():+.0f}")
                elif metric_name == '单笔PnL':
                    cells.append(f"{sub['pnl_pts'].mean():+.2f}")
                elif metric_name == '胜率':
                    cells.append(f"{(sub['pnl_pts']>0).sum()/n*100:.1f}%")
        doc.append(f"| {metric_name} | {' | '.join(cells)} |")

    # Sweet spot三条件在OOS上检查
    v2_oos = v2_all[v2_all['trade_date'].isin(oos_set)]
    v1_oos = v1_all[v1_all['trade_date'].isin(oos_set)]
    v2_oos_n = len(v2_oos)
    v1_oos_n = len(v1_oos)
    v2_oos_total = v2_oos['pnl_pts'].sum()
    v1_oos_total = v1_oos['pnl_pts'].sum()
    v2_oos_avg = v2_oos['pnl_pts'].mean() if v2_oos_n > 0 else 0
    v1_oos_avg = v1_oos['pnl_pts'].mean() if v1_oos_n > 0 else 0

    sig_reduce = 1 - (v1_oos_n / v2_oos_n) if v2_oos_n > 0 else 0
    pnl_loss = 1 - (v1_oos_total / v2_oos_total) if v2_oos_total > 0 else 0
    eff_improve = (v1_oos_avg - v2_oos_avg) / abs(v2_oos_avg) if v2_oos_avg != 0 else 0

    doc.append(f"\n### OOS 681天 Sweet Spot三条件\n")
    doc.append(f"| 条件 | 值 | 阈值 | 通过? |")
    doc.append(f"|------|---|------|-------|")
    doc.append(f"| 信号减少 | {sig_reduce*100:.1f}% | >=20% | {'✓' if sig_reduce >= 0.20 else '✗'} |")
    doc.append(f"| 总PnL损失 | {pnl_loss*100:.1f}% | <8% | {'✓' if pnl_loss < 0.08 else '✗'} |")
    doc.append(f"| 单笔效率提升 | {eff_improve*100:.1f}% | >=25% | {'✓' if eff_improve >= 0.25 else '✗'} |")

    all_pass = sig_reduce >= 0.20 and pnl_loss < 0.08 and eff_improve >= 0.25
    return all_pass, sig_reduce, pnl_loss, eff_improve


def main():
    print("=" * 60)
    print("  v1_im + v1_ic 双窗口验证")
    print("=" * 60)

    db = get_db()
    n_workers = min(cpu_count(), 8)

    all_data = {}
    for sym in ['IM', 'IC']:
        print(f"\n[{sym}] 收集数据...")
        dates = get_dates(db, SPOTS[sym])
        args = [(td, sym) for td in dates]
        with Pool(n_workers) as pool:
            day_results = pool.map(_run_one_day, args)
        trades = [t for day in day_results for t in day]
        tdf = pd.DataFrame(trades)
        all_data[sym] = {'tdf': tdf, 'dates': dates}
        print(f"  {len(tdf)}笔")

    doc = ["# v1_im + v1_ic 双窗口验证\n"]

    # ═══════════════════════════════════════════════
    # V1: v1_im双窗口验证
    # ═══════════════════════════════════════════════
    doc.append("## V1: v1_im双窗口验证 (thr=60)\n")
    im_tdf = apply_v1_im(all_data['IM']['tdf'])
    v1_im_pass, *v1_im_metrics = dual_window_table(
        im_tdf, 60, V2_THR['IM'], all_data['IM']['dates'], doc)

    if v1_im_pass:
        doc.append(f"\n**判定V1-A: v1_im完全成立** ✓")
    elif sum(1 for m in v1_im_metrics if m) >= 2:  # 简化判定
        doc.append(f"\n**判定V1-B: v1_im部分成立**")
    else:
        doc.append(f"\n**判定V1-C: v1_im失败** ✗")
    doc.append("")

    # ═══════════════════════════════════════════════
    # V2: v1_ic双窗口验证
    # ═══════════════════════════════════════════════
    doc.append("## V2: v1_ic双窗口验证 (thr=55)\n")
    ic_tdf = apply_v1_ic(all_data['IC']['tdf'])
    v1_ic_pass, *v1_ic_metrics = dual_window_table(
        ic_tdf, 55, V2_THR['IC'], all_data['IC']['dates'], doc)

    if v1_ic_pass:
        doc.append(f"\n**判定V2-A: v1_ic完全成立** ✓")
    elif sum(1 for m in v1_ic_metrics if m) >= 2:
        doc.append(f"\n**判定V2-B: v1_ic部分成立**")
    else:
        doc.append(f"\n**判定V2-C: v1_ic失败** ✗")
    doc.append("")

    # ═══════════════════════════════════════════════
    # V3: v1_im设计披露
    # ═══════════════════════════════════════════════
    doc.append("## V3: v1_im完整设计披露\n")

    doc.append("### M分映射(数据驱动, 10桶按PnL排序)\n")
    doc.append("| raw_mom_5m范围 | M分 | 设计依据PnL |")
    doc.append("|-------------|-----|---------|")
    im_pnl_lookup = {0: 0.5, 17: 1.0, 11: 0.9, 33: 2.8, 28: 2.2, 6: 0.7, 22: 1.2, 50: 4.1, 39: 2.8, 44: 3.5}
    for lo, hi, s in V1_IM_M_THRESHOLDS:
        doc.append(f"| [{lo:.5f}, {hi:.5f}) | {s} | +{im_pnl_lookup.get(s, 0):.1f} |")

    doc.append("\n### V分映射(数据驱动, 10桶按PnL排序)\n")
    doc.append("| raw_atr_ratio范围 | V分 | 设计依据PnL |")
    doc.append("|----------------|-----|---------|")
    im_v_lookup = {30: 4.7, 20: 2.2, 3: -0.2, 13: 1.5, 10: 1.2, 17: 2.0, 0: -0.9, 27: 4.3, 7: 0.8, 23: 4.1}
    for lo, hi, s in V1_IM_V_THRESHOLDS:
        doc.append(f"| [{lo:.3f}, {hi:.3f}) | {s} | +{im_v_lookup.get(s, 0):.1f} |")

    doc.append(f"\n### 时段加分: {V1_IM_SESSION}")
    doc.append(f"### Q分: 保持v2逻辑, max=15 (percentile 15/8/0)")
    doc.append(f"### Gap: 对齐+5, 不对齐0")
    doc.append(f"\n### 总分 = M(0-50) + V(0-30) + Q(0-15) + Session(-10~+10) + Gap(0-5)")
    doc.append(f"### 最优threshold = 60")

    doc.append(f"\n### 过拟合风险评估\n")
    doc.append("| 组件 | 方法 | 自由度 | 风险 |")
    doc.append("|------|------|-------|------|")
    doc.append("| M分 | 数据驱动10桶×score | 10个分数值 | **高** |")
    doc.append("| V分 | 数据驱动10桶×score | 10个分数值 | **高** |")
    doc.append("| Q分 | 保持v2(3档) | 2个阈值 | 低 |")
    doc.append("| Session | 数据驱动5个值 | 5个分数值 | 中 |")
    doc.append("| Gap | 固定(0/5) | 1个阈值 | 低 |")
    doc.append("\n**整体过拟合风险: 高** (M分和V分都是数据驱动高自由度)")
    doc.append("")

    # ═══════════════════════════════════════════════
    # V4: 成本效益精确计算
    # ═══════════════════════════════════════════════
    doc.append("## V4: 成本效益精确计算\n")

    # 成本参数
    fee_rate = 0.00023  # 万分之2.3
    im_value = 8000 * 200  # IM合约价值 ≈ 160万
    ic_value = 7000 * 200  # IC合约价值 ≈ 140万
    im_cost_per_trade = 4 * fee_rate * im_value  # 4笔(开+锁+双平)
    ic_cost_per_trade = 4 * fee_rate * ic_value

    doc.append(f"### 成本参数\n")
    doc.append(f"- IM每笔成本: 4 × {fee_rate} × {im_value:,.0f} = **{im_cost_per_trade:.0f}元**")
    doc.append(f"- IC每笔成本: 4 × {fee_rate} × {ic_value:,.0f} = **{ic_cost_per_trade:.0f}元**")

    # IM
    im_v2_n = len(im_tdf[im_tdf['entry_score'] >= V2_THR['IM']])
    im_v1_n = len(im_tdf[im_tdf['v1_total'] >= 60])
    im_saved = im_v2_n - im_v1_n
    im_cost_saved = im_saved * im_cost_per_trade
    im_pnl_loss = (im_tdf[im_tdf['entry_score'] >= V2_THR['IM']]['pnl_pts'].sum() -
                   im_tdf[im_tdf['v1_total'] >= 60]['pnl_pts'].sum()) * 200
    im_net = im_cost_saved - im_pnl_loss

    # IC
    ic_v2_n = len(ic_tdf[ic_tdf['entry_score'] >= V2_THR['IC']])
    ic_v1_n = len(ic_tdf[ic_tdf['v1_total'] >= 55])
    ic_saved = ic_v2_n - ic_v1_n
    ic_cost_saved = ic_saved * ic_cost_per_trade
    ic_pnl_loss = (ic_tdf[ic_tdf['entry_score'] >= V2_THR['IC']]['pnl_pts'].sum() -
                   ic_tdf[ic_tdf['v1_total'] >= 55]['pnl_pts'].sum()) * 200
    ic_net = ic_cost_saved - ic_pnl_loss

    doc.append(f"\n### 净收益计算\n")
    doc.append(f"| 项 | IM | IC | 合计 |")
    doc.append(f"|---|---|---|---|")
    doc.append(f"| 信号减少 | {im_saved}笔 | {ic_saved}笔 | {im_saved+ic_saved}笔 |")
    doc.append(f"| 成本节省 | {im_cost_saved:,.0f}元 | {ic_cost_saved:,.0f}元 | {im_cost_saved+ic_cost_saved:,.0f}元 |")
    doc.append(f"| PnL损失 | {im_pnl_loss:,.0f}元 | {ic_pnl_loss:,.0f}元 | {im_pnl_loss+ic_pnl_loss:,.0f}元 |")
    doc.append(f"| **净节省** | **{im_net:,.0f}元** | **{ic_net:,.0f}元** | **{im_net+ic_net:,.0f}元** |")
    doc.append(f"| **年化** | **{im_net/3.67:,.0f}元/年** | **{ic_net/3.67:,.0f}元/年** | **{(im_net+ic_net)/3.67:,.0f}元/年** |")
    doc.append("")

    # ═══════════════════════════════════════════════
    # 综合判定
    # ═══════════════════════════════════════════════
    doc.append("## 综合判定\n")
    doc.append(f"- v1_im: {'V1-A完全成立' if v1_im_pass else 'V1-B/C'}")
    doc.append(f"- v1_ic: {'V2-A完全成立' if v1_ic_pass else 'V2-B/C'}")
    doc.append(f"- 年化净节省: {(im_net+ic_net)/3.67:,.0f}元/年")

    report = "\n".join(doc)
    path = Path("tmp") / "v1_validation_summary.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
