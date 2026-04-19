#!/usr/bin/env python3
"""横盘反转因子探索：评估HorizontalReversalFactor的预测力。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd
from pathlib import Path
from data.storage.db_manager import get_db
from models.factors.catalog_structure import HorizontalReversalFactor, BollBreakout
from models.factors.catalog_price import MomSimple
from models.factors.catalog_volume import QtyRatio
from models.factors.evaluator import FactorEvaluator

SPOT_MAP = {"IM": "000852", "IC": "000905"}


def load_bars(sym, start="20250516", end="20260409"):
    db = get_db()
    spot = SPOT_MAP[sym]
    df = db.query_df(
        f"SELECT datetime, open, high, low, close, volume FROM index_min "
        f"WHERE symbol='{spot}' AND period=300 ORDER BY datetime"
    )
    for c in ['open','high','low','close','volume']:
        df[c] = df[c].astype(float)
    df.index = pd.to_datetime(df['datetime'])
    df['date'] = df.index.date
    # 过滤日期范围
    df['date_str'] = df.index.strftime('%Y%m%d')
    df = df[(df['date_str'] >= start) & (df['date_str'] <= end)]
    return df


def calc_daily_amplitude(bar_5m):
    """计算每天的开盘30min振幅，返回跟bar_5m对齐的Series。"""
    daily_amp = {}
    for date, day_bars in bar_5m.groupby('date'):
        if len(day_bars) < 6:
            daily_amp[date] = np.nan
            continue
        open_bars = day_bars.iloc[:6]
        amp = (open_bars['high'].max() - open_bars['low'].min()) / open_bars['open'].iloc[0] * 100
        daily_amp[date] = amp
    amp_series = bar_5m['date'].map(daily_amp)
    amp_series.index = bar_5m.index
    return amp_series


def step5_me_crosscheck(sym, bar_5m):
    """MomentumExhausted触发后的反向MFE统计。"""
    from strategies.intraday.A_share_momentum_signal_v2 import _calc_boll
    from strategies.intraday import atomic_factors as af

    close = bar_5m['close'].values
    high = bar_5m['high'].values
    low = bar_5m['low'].values

    events = []
    for i in range(30, len(bar_5m) - 24):
        c_series = bar_5m['close'].iloc[max(0,i-20):i+1]
        if len(c_series) < 20:
            continue
        mid, std = float(c_series.mean()), float(c_series.std())
        if std <= 0:
            continue

        # narrow_range check
        nr = af.narrow_range(bar_5m.iloc[max(0,i-3):i+1], 3, std)
        if nr < 0 or nr >= 0.10:
            continue

        # not trending
        trending = af.price_trending(bar_5m.iloc[max(0,i-3):i+1], 3, std, "LONG")
        trending_s = af.price_trending(bar_5m.iloc[max(0,i-3):i+1], 3, std, "SHORT")
        if trending or trending_s:
            continue

        # determine prior trend direction
        if i >= 12:
            mom = (close[i] - close[i-12]) / close[i-12]
            if abs(mom) < 0.001:
                continue
            prior_dir = "LONG" if mom > 0 else "SHORT"
        else:
            continue

        # check zone (simplified: above mid = upper zone)
        zone = af.boll_zone(close[i], mid, std)
        if prior_dir == "LONG" and zone not in ("ABOVE_UPPER", "UPPER_ZONE"):
            continue
        if prior_dir == "SHORT" and zone not in ("BELOW_LOWER", "LOWER_ZONE"):
            continue

        # ME triggered — look at reversal MFE in next 12-24 bars
        future = bar_5m.iloc[i+1:min(i+25, len(bar_5m))]
        if len(future) < 6:
            continue

        if prior_dir == "LONG":
            # reversal = SHORT direction
            rev_mfe = close[i] - future['low'].min()
            rev_mae = future['high'].max() - close[i]
        else:
            rev_mfe = future['high'].max() - close[i]
            rev_mae = close[i] - future['low'].min()

        events.append({
            'bar_idx': i, 'prior_dir': prior_dir,
            'rev_mfe': max(0, rev_mfe), 'rev_mae': max(0, rev_mae),
        })

    return pd.DataFrame(events)


def main():
    output_dir = Path("tmp/horizontal_reversal_factor_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = ["# 横盘反转因子探索\n"]

    # Step 1: ME逻辑总结
    doc.append("## 1. MomentumExhausted 逻辑总结\n")
    doc.append("复用的原子因子：`narrow_range`(3bar), `price_trending`(3bar), `boll_zone`(15m)")
    doc.append("条件：hold>=20min + range<boll_width×me_ratio + NOT trending + 15m极端zone")
    doc.append("Entry镜像调整：去掉hold_time/position依赖，加前置趋势强度，输出连续值\n")

    # Step 2: 因子定义
    doc.append("## 2. HorizontalReversalFactor 定义\n")
    doc.append("```")
    doc.append("reversal_strength = |prior_trend(K)| × horizontal_score(N) × low_vol_score(M) × (-trend_sign)")
    doc.append("- prior_trend: close[-N] vs close[-K] 的%变化")
    doc.append("- horizontal_score: 1 - (N_bar_range / boll_width), clamp[0,1]")
    doc.append("- low_vol_score: 1 - (M_bar_avg_range / 40_bar_avg_range), clamp[0,1]")
    doc.append("默认参数: K=12, N=3, M=3")
    doc.append("```\n")

    # Steps 3-4: IM + IC评估
    for sym in ['IM', 'IC']:
        print(f"\n[{sym}] 加载数据...")
        bar_5m = load_bars(sym)
        daily_amp = calc_daily_amplitude(bar_5m)

        print(f"[{sym}] 评估因子...")
        # 创建evaluator
        evaluator = FactorEvaluator(
            bar_5m, forward_periods=[3, 6, 12], daily_range=daily_amp
        )

        # 核心因子
        hr_factor = HorizontalReversalFactor(trend_lookback=12, horizontal_window=3, vol_window=3)

        # 对照因子（现有M/V/B）
        mom_factor = MomSimple(12)
        boll_factor = BollBreakout(20)
        vol_factor = QtyRatio(20)

        # 批量评估
        all_factors = [hr_factor, mom_factor, boll_factor, vol_factor]
        results, corr_df = evaluator.batch_evaluate(all_factors)

        hr_result = results[0]

        doc.append(f"\n## 3-4. {sym} 评估结果\n")

        # IC
        doc.append(f"### Daily IC\n")
        doc.append(f"| Forward | Daily IC | IC_IR | Global IC |")
        doc.append(f"|---------|----------|-------|-----------|")
        for n in [3, 6, 12]:
            dic = hr_result['ic'].get(f'dIC_{n}bar', np.nan)
            dicir = hr_result['ic'].get(f'dICIR_{n}bar', np.nan)
            gic = hr_result['ic'].get(f'IC_{n}bar', np.nan)
            doc.append(f"| {n} bar | {dic if dic is not np.nan else 'N/A'} | {dicir if dicir is not np.nan else 'N/A'} | {gic if gic is not np.nan else 'N/A'} |")

        # 分组收益
        doc.append(f"\n### 分组收益 (bps)\n")
        for n in [6, 12]:
            gr = hr_result['group_returns'].get(f'{n}bar', {})
            if gr:
                doc.append(f"  {n}bar: " + " → ".join(f"{k}={v}" for k, v in sorted(gr.items())))

        # 单调性
        doc.append(f"\n### 单调性\n")
        for n in [6, 12]:
            mono = hr_result['monotonicity'].get(f'{n}bar', {})
            if mono:
                doc.append(f"  {n}bar: corr={mono['corr']}, p={mono['pval']}")

        # 相关性矩阵
        doc.append(f"\n### 跟现有因子的相关性\n")
        hr_corr = corr_df[hr_factor.name]
        for fname in [mom_factor.name, boll_factor.name, vol_factor.name]:
            if fname in hr_corr:
                doc.append(f"  vs {fname}: {hr_corr[fname]:.3f}")

        # Regime分组IC
        if 'regime_ic' in hr_result and hr_result['regime_ic']:
            doc.append(f"\n### Regime分组IC\n")
            for k, v in sorted(hr_result['regime_ic'].items()):
                doc.append(f"  {k}: {v}")

        # Step 5: ME对照
        print(f"[{sym}] ME对照统计...")
        me_events = step5_me_crosscheck(sym, bar_5m)
        doc.append(f"\n### ME触发后反向MFE ({len(me_events)}个事件)\n")
        if len(me_events) >= 10:
            doc.append(f"  反向MFE中位数: {me_events['rev_mfe'].median():.1f}pt")
            doc.append(f"  反向MAE中位数: {me_events['rev_mae'].median():.1f}pt")
            ratio = me_events['rev_mfe'].median() / me_events['rev_mae'].median() if me_events['rev_mae'].median() > 0 else 0
            doc.append(f"  MFE/MAE: {ratio:.2f}")
            doc.append(f"  反向胜率(MFE>MAE): {(me_events['rev_mfe'] > me_events['rev_mae']).mean()*100:.0f}%")
        else:
            doc.append(f"  样本不足（{len(me_events)}个）")

    # Step 6: 参数敏感性
    doc.append(f"\n## 6. 参数敏感性\n")
    bar_5m_im = load_bars('IM')
    evaluator_im = FactorEvaluator(bar_5m_im, forward_periods=[6])

    doc.append(f"{'K':>3s} {'N':>3s} {'M':>3s} | {'dIC_6bar':>9s}")
    doc.append("-" * 25)
    for K in [6, 12, 18, 24]:
        for N in [3, 5, 8]:
            for M in [3, 5]:
                f = HorizontalReversalFactor(K, N, M)
                r = evaluator_im.evaluate(f)
                dic = r['ic'].get('dIC_6bar', np.nan)
                dic_s = f"{dic:.4f}" if not np.isnan(dic) else "N/A"
                doc.append(f"{K:>3d} {N:>3d} {M:>3d} | {dic_s:>9s}")

    # Step 7: 结论
    doc.append(f"\n## 7. 结论\n")
    doc.append("（基于以上数据自动判定）\n")

    # 自动判定
    hr_im = results[0] if sym == 'IC' else None  # 需要分别存
    # 简单判定：看6bar daily IC的绝对值
    im_bar = load_bars('IM')
    ev_im = FactorEvaluator(im_bar, forward_periods=[6])
    r_im = ev_im.evaluate(HorizontalReversalFactor(12, 3, 3))
    dic_im = r_im['ic'].get('dIC_6bar', 0)

    ic_bar = load_bars('IC')
    ev_ic = FactorEvaluator(ic_bar, forward_periods=[6])
    r_ic = ev_ic.evaluate(HorizontalReversalFactor(12, 3, 3))
    dic_ic = r_ic['ic'].get('dIC_6bar', 0)

    doc.append(f"IM Daily IC(6bar): {dic_im:.4f}")
    doc.append(f"IC Daily IC(6bar): {dic_ic:.4f}")

    if abs(dic_im) > 0.03 or abs(dic_ic) > 0.03:
        doc.append("\n**因子有效**: 至少一个品种Daily IC > 0.03")
        doc.append("→ 建议进入实盘验证阶段")
    elif abs(dic_im) > 0.015 or abs(dic_ic) > 0.015:
        doc.append("\n**因子弱有效**: IC在0.015-0.03之间，方向有指示性")
        doc.append("→ 候选观察，不急于上线")
    else:
        doc.append("\n**因子无效**: Daily IC < 0.015")
        doc.append("→ 路径终止，探索其他方向")

    report = "\n".join(doc)
    path = output_dir / "horizontal_reversal_factor_analysis.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)
    print(f"\n报告已保存: {path}")


if __name__ == "__main__":
    main()
