#!/usr/bin/env python3
"""First+Short子组合补充描述性分析。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd
from pathlib import Path
from data.storage.db_manager import get_db
from models.factors.catalog_structure import HorizontalReversalSimple

AMP_THR = 0.4


def load_first_short():
    db = get_db()
    df = db.query_df(
        "SELECT datetime, open, high, low, close, volume FROM index_min "
        "WHERE symbol='000852' AND period=300 ORDER BY datetime"
    )
    for c in ['open','high','low','close','volume']:
        df[c] = df[c].astype(float)
    df.index = pd.to_datetime(df['datetime'])
    df['date'] = df.index.date

    daily_amp = {}
    for date, day_bars in df.groupby('date'):
        if len(day_bars) < 6: continue
        ob = day_bars.iloc[:6]
        daily_amp[date] = (ob['high'].max() - ob['low'].min()) / ob['open'].iloc[0] * 100
    df['amp'] = df['date'].map(daily_amp)

    hr = HorizontalReversalSimple(12, 3)
    df['factor'] = hr.compute_series(df)
    df['fwd15'] = df['close'].pct_change(15).shift(-15) * 10000

    low_sig = df[(df['amp'] < AMP_THR) & (df['factor'] != 0)].copy()
    low_sig['trigger_order'] = low_sig.groupby('date').cumcount() + 1
    fs = low_sig[(low_sig['trigger_order'] == 1) & (low_sig['factor'] == -1)].copy()
    fs['adj15'] = -fs['fwd15']  # 翻转：正=预测正确
    return fs


def main():
    fs = load_first_short()
    target = fs['adj15'].dropna()
    n = len(target)

    doc = ["# First+Short 子组合补充描述性分析\n"]
    doc.append(f"样本: {n}笔, forward=15 bar, 调整后收益(正=预测正确)\n")

    # ══════════════════════════════════════════════
    # 第一步：4段时间分布
    # ══════════════════════════════════════════════
    doc.append("## 1. 四段时间分布\n")
    all_dates = sorted(fs['date'].unique())
    q_size = max(1, len(all_dates) // 4)

    doc.append(f"| 段 | 时间范围 | N | 均值(bps) | 中位数 | 标准差 |")
    doc.append(f"|---|---------|---|-----------|--------|--------|")

    q_data_list = []
    for i in range(4):
        start = i * q_size
        end = min((i + 1) * q_size, len(all_dates))
        if start >= len(all_dates): break
        q_dates = set(all_dates[start:end])
        q_data = fs[fs['date'].isin(q_dates)]['adj15'].dropna()
        q_data_list.append(q_data)
        if len(q_data) >= 2:
            doc.append(f"| Q{i+1} | {min(q_dates)}~{max(q_dates)} | {len(q_data)} | "
                       f"{q_data.mean():+.1f} | {q_data.median():+.1f} | {q_data.std():.1f} |")
        else:
            doc.append(f"| Q{i+1} | {min(q_dates)}~{max(q_dates)} | {len(q_data)} | N/A | | |")

    if len(q_data_list) >= 4:
        q4_mean = q_data_list[3].mean() if len(q_data_list[3]) >= 2 else np.nan
        doc.append(f"\n**Q4（最近时段）均值: {q4_mean:+.1f} bps**")

    # ══════════════════════════════════════════════
    # 第二步：胜率和盈亏分布
    # ══════════════════════════════════════════════
    doc.append("\n## 2. 胜率和盈亏分布\n")

    wins = target[target > 0]
    losses = target[target < 0]
    draws = target[target == 0]
    win_rate = len(wins) / n * 100

    doc.append(f"| 组 | 笔数 | 占比 | 均值(bps) | 中位数 | 最大值 | 最小值 |")
    doc.append(f"|---|---|---|---|---|---|---|")
    doc.append(f"| 盈利 | {len(wins)} | {len(wins)/n*100:.0f}% | {wins.mean():+.1f} | {wins.median():+.1f} | {wins.max():+.1f} | {wins.min():+.1f} |")
    doc.append(f"| 亏损 | {len(losses)} | {len(losses)/n*100:.0f}% | {losses.mean():+.1f} | {losses.median():+.1f} | {losses.max():+.1f} | {losses.min():+.1f} |")
    if len(draws) > 0:
        doc.append(f"| 平局 | {len(draws)} | {len(draws)/n*100:.0f}% | | | | |")

    pnl_ratio = wins.mean() / abs(losses.mean()) if len(losses) > 0 and losses.mean() != 0 else np.inf
    expected = win_rate/100 * wins.mean() - (1-win_rate/100) * abs(losses.mean())

    doc.append(f"\n| 指标 | 值 |")
    doc.append(f"|------|-----|")
    doc.append(f"| **胜率** | **{win_rate:.0f}%** |")
    doc.append(f"| **盈亏比** | **{pnl_ratio:.2f}** |")
    doc.append(f"| 期望值 | {expected:+.1f} bps (应≈均值{target.mean():+.1f}) |")

    # ══════════════════════════════════════════════
    # 第三步：极端值检查
    # ══════════════════════════════════════════════
    doc.append("\n## 3. 极端值检查\n")

    sorted_vals = target.sort_values(ascending=False)
    doc.append("**最大5笔盈利:**")
    for i, v in enumerate(sorted_vals.head(5)):
        doc.append(f"  {i+1}. {v:+.1f} bps")

    doc.append("\n**最大5笔亏损:**")
    for i, v in enumerate(sorted_vals.tail(5).iloc[::-1]):
        doc.append(f"  {i+1}. {v:+.1f} bps")

    # 剔除检验
    trimmed_top3 = target.sort_values().iloc[:-3]  # 去掉最大3笔
    trimmed_bot3 = target.sort_values().iloc[3:]   # 去掉最小3笔

    doc.append(f"\n**剔除检验:**")
    doc.append(f"| 场景 | N | 均值(bps) | 变化 |")
    doc.append(f"|------|---|-----------|------|")
    doc.append(f"| 全部 | {n} | {target.mean():+.1f} | — |")
    doc.append(f"| 剔除最大3笔盈利 | {len(trimmed_top3)} | {trimmed_top3.mean():+.1f} | {trimmed_top3.mean()-target.mean():+.1f} |")
    doc.append(f"| 剔除最大3笔亏损 | {len(trimmed_bot3)} | {trimmed_bot3.mean():+.1f} | {trimmed_bot3.mean()-target.mean():+.1f} |")

    robust = trimmed_top3.mean() >= 8

    # ══════════════════════════════════════════════
    # 第四步：综合判定
    # ══════════════════════════════════════════════
    doc.append("\n## 4. 综合判定\n")

    q4_mean = q_data_list[3].mean() if len(q_data_list) >= 4 and len(q_data_list[3]) >= 2 else 0
    trimmed_mean = trimmed_top3.mean()

    # 情况A条件检查
    cond_a1 = q4_mean >= 12 and win_rate >= 55
    cond_a2 = q4_mean >= 15 and trimmed_mean >= 8
    cond_a3 = all(q_data_list[i].mean() < q_data_list[i+1].mean()
                  for i in range(min(3, len(q_data_list)-1))
                  if len(q_data_list[i]) >= 2 and len(q_data_list[i+1]) >= 2)

    doc.append(f"| 条件 | 结果 |")
    doc.append(f"|------|------|")
    doc.append(f"| A1: Q4>=12 + 胜率>=55% | Q4={q4_mean:+.1f}, WR={win_rate:.0f}% → {'✓' if cond_a1 else '✗'} |")
    doc.append(f"| A2: Q4>=15 + 剔除后>=8 | Q4={q4_mean:+.1f}, trim={trimmed_mean:+.1f} → {'✓' if cond_a2 else '✗'} |")
    doc.append(f"| A3: 4段单调递增 | {'✓' if cond_a3 else '✗'} |")

    if cond_a1 or cond_a2 or cond_a3:
        doc.append(f"\n**判定A：升级到小仓位试运行候选** ✓\n")
        doc.append("### 试运行方案\n")
        doc.append("- 起始仓位: 标准仓位的30%（约9,600元风险/笔）")
        doc.append("- 观察周期: 12个月（预期约10笔信号）")
        doc.append("- 升级触发: 实盘10笔以上且净收益为正 → 提升到标准仓位50%")
        doc.append("- 放弃触发: 连续5笔亏损 或 10笔后净收益为负 → 归档")
        doc.append("- 执行条件: 仅在IM低振幅日（开盘30min振幅<0.4%）的首次触发+看空信号")
    elif q4_mean >= 5 or (win_rate >= 50 and pnl_ratio > 2):
        doc.append(f"\n**判定B：维持观察池** — Q4偏弱或信号依赖盈亏比")
        doc.append(f"- 重新评估: 50笔样本或6个月后")
    else:
        doc.append(f"\n**判定C：归档放弃** — Q4失效或信号不��靠")

    report = "\n".join(doc)
    path = Path("tmp") / "horizontal_reversal_first_short_supplementary.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
