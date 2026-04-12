#!/usr/bin/env python3
"""横盘反转因子 First+Short 子组合统计检验。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from data.storage.db_manager import get_db
from models.factors.catalog_structure import HorizontalReversalSimple

AMP_THR = 0.4


def load_and_prepare():
    db = get_db()
    df = db.query_df(
        "SELECT datetime, open, high, low, close, volume FROM index_min "
        "WHERE symbol='000852' AND period=300 ORDER BY datetime"
    )
    for c in ['open','high','low','close','volume']:
        df[c] = df[c].astype(float)
    df.index = pd.to_datetime(df['datetime'])
    df['date'] = df.index.date

    # Amplitude
    daily_amp = {}
    for date, day_bars in df.groupby('date'):
        if len(day_bars) < 6: continue
        ob = day_bars.iloc[:6]
        daily_amp[date] = (ob['high'].max() - ob['low'].min()) / ob['open'].iloc[0] * 100
    df['amp'] = df['date'].map(daily_amp)

    # Factor
    hr = HorizontalReversalSimple(12, 3)
    df['factor'] = hr.compute_series(df)

    # Forward returns
    for n in [12, 15, 18]:
        df[f'fwd{n}'] = df['close'].pct_change(n).shift(-n) * 10000

    # Low amp + signal
    low_sig = df[(df['amp'] < AMP_THR) & (df['factor'] != 0)].copy()
    low_sig['trigger_order'] = low_sig.groupby('date').cumcount() + 1

    # First + Short (-1)
    first_short = low_sig[(low_sig['trigger_order'] == 1) & (low_sig['factor'] == -1)].copy()
    # 调整后收益 = fwd × (-1)，让正值=预测正确
    for n in [12, 15, 18]:
        first_short[f'adj{n}'] = -first_short[f'fwd{n}']

    return df, first_short


def main():
    df, fs = load_and_prepare()
    doc = ["# 横盘反转因子 First+Short 子组合统计检验\n"]
    doc.append(f"子组合: IM低振幅日 × 当日首次触发 × 看空信号(-1)")
    doc.append(f"样本数: **{len(fs)}笔**")
    doc.append(f"数据: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')} (900天)\n")

    # ══════════════════════════════════════════════
    # 第一步：统计检验
    # ══════════════════════════════════════════════
    doc.append("## 1. 统计显著性检验\n")

    target = fs['adj15'].dropna()
    n = len(target)
    doc.append(f"Forward period: 15 bar (75分钟)")
    doc.append(f"有效样本: {n}笔\n")

    # 描述性统计
    doc.append("### 描述性统计\n")
    doc.append(f"| 指标 | 值 |")
    doc.append(f"|------|-----|")
    doc.append(f"| 均值 | {target.mean():+.2f} bps |")
    doc.append(f"| 中位数 | {target.median():+.2f} bps |")
    doc.append(f"| 标准差 | {target.std():.2f} bps |")
    doc.append(f"| 最小值 | {target.min():+.1f} bps |")
    doc.append(f"| 最大值 | {target.max():+.1f} bps |")
    doc.append(f"| 25分位 | {target.quantile(0.25):+.1f} bps |")
    doc.append(f"| 75分位 | {target.quantile(0.75):+.1f} bps |")

    # t检验
    doc.append("\n### t检验 (H0: 均值=0, H1: 均值>0, 单边)\n")
    t_stat, p_two = stats.ttest_1samp(target, 0)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    doc.append(f"| 指标 | 值 |")
    doc.append(f"|------|-----|")
    doc.append(f"| t统计量 | {t_stat:.3f} |")
    doc.append(f"| p值(单边) | **{p_one:.4f}** |")
    doc.append(f"| 自由度 | {n-1} |")
    if p_one < 0.01:
        doc.append(f"| 结论 | **高度显著** (p<0.01) ✓✓ |")
    elif p_one < 0.05:
        doc.append(f"| 结论 | **显著** (p<0.05) ✓ |")
    elif p_one < 0.15:
        doc.append(f"| 结论 | 边缘显著 (p<0.15) |")
    else:
        doc.append(f"| 结论 | 不显著 (p>0.15) ✗ |")

    # Bootstrap
    doc.append("\n### Bootstrap 95% 置信区间 (1000次)\n")
    np.random.seed(42)
    boot_means = []
    for _ in range(1000):
        sample = target.sample(n=n, replace=True)
        boot_means.append(sample.mean())
    boot_means = np.array(boot_means)
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)
    doc.append(f"| 指标 | 值 |")
    doc.append(f"|------|-----|")
    doc.append(f"| 95% CI 下界 | **{ci_lower:+.2f}** bps |")
    doc.append(f"| 95% CI 上界 | {ci_upper:+.2f} bps |")
    if ci_lower > 5:
        doc.append(f"| 结论 | **明显显著** (下界>5bps) ✓✓ |")
    elif ci_lower > 0:
        doc.append(f"| 结论 | **边缘显著** (下界>0但<5bps) ✓ |")
    else:
        doc.append(f"| 结论 | 不显著 (下界<0) ✗ |")

    # Effect size
    doc.append("\n### 效应量\n")
    cohens_d = target.mean() / target.std() if target.std() > 0 else 0
    doc.append(f"| 指标 | 值 |")
    doc.append(f"|------|-----|")
    doc.append(f"| Cohen's d | **{cohens_d:.3f}** |")
    if cohens_d > 0.8:
        doc.append(f"| 结论 | 大效应 (d>0.8) ✓✓ |")
    elif cohens_d > 0.5:
        doc.append(f"| 结论 | 中等效应 (d>0.5) ✓ |")
    elif cohens_d > 0.2:
        doc.append(f"| 结论 | 小效应 (d>0.2) |")
    else:
        doc.append(f"| 结论 | 无效应 (d<0.2) ✗ |")

    # 多forward检验
    doc.append("\n### 多forward period交叉检验\n")
    doc.append(f"| Forward | N | 均值(bps) | t值 | p(单边) | 95%CI下界 |")
    doc.append(f"|---------|---|-----------|-----|---------|-----------|")
    for n_fwd in [12, 15, 18]:
        col = f'adj{n_fwd}'
        data = fs[col].dropna()
        if len(data) < 10:
            continue
        t, p2 = stats.ttest_1samp(data, 0)
        p1 = p2 / 2 if t > 0 else 1 - p2 / 2
        boot = [data.sample(n=len(data), replace=True).mean() for _ in range(500)]
        ci_l = np.percentile(boot, 2.5)
        sig = "✓" if p1 < 0.05 else ""
        doc.append(f"| {n_fwd} | {len(data)} | {data.mean():+.1f} | {t:.2f} | {p1:.3f} | {ci_l:+.1f} | {sig}")

    # ══════════════════════════════════════════════
    # 第二步：4段时间分布
    # ══════════════════════════════════════════════
    doc.append("\n## 2. 四段时间分布\n")
    all_dates = sorted(fs['date'].unique())
    q_size = max(1, len(all_dates) // 4)
    quarters = []
    for i in range(4):
        start = i * q_size
        end = min((i + 1) * q_size, len(all_dates))
        if start >= len(all_dates):
            break
        q_dates = set(all_dates[start:end])
        quarters.append((f"Q{i+1}", q_dates,
                         str(min(q_dates)) if q_dates else "",
                         str(max(q_dates)) if q_dates else ""))

    doc.append(f"| 段 | 时间范围 | N | Mean(bps) | 趋势 |")
    doc.append(f"|---|---------|---|-----------|------|")
    q_means = []
    for qname, qdates, qstart, qend in quarters:
        q_data = fs[fs['date'].isin(qdates)]['adj15'].dropna()
        m = q_data.mean() if len(q_data) >= 3 else np.nan
        q_means.append(m)
        m_s = f"{m:+.1f}" if pd.notna(m) else "N/A"
        doc.append(f"| {qname} | {qstart}~{qend} | {len(q_data)} | {m_s} | |")

    valid_means = [m for m in q_means if pd.notna(m)]
    if len(valid_means) >= 3:
        if all(valid_means[i] <= valid_means[i+1] for i in range(len(valid_means)-1)):
            doc.append("\n趋势: **单调递增** — 因子在变强 ✓")
        elif all(valid_means[i] >= valid_means[i+1] for i in range(len(valid_means)-1)):
            doc.append("\n趋势: **单调递减** — 因子在衰退 ⚠")
        elif valid_means[-1] >= valid_means[0]:
            doc.append("\n趋势: 非单调但最近>最早 — 整体偏正 ✓")
        else:
            doc.append("\n趋势: 非单调且最近<最早 — 需要警惕")

    # ══════════════════════════════════════════════
    # 第三步：实盘可行性
    # ══════════════════════════════════════════════
    doc.append("\n## 3. 实盘可行性估算\n")
    im_price = 7500
    multiplier = 200
    account = 6400000
    risk_pct = 0.005
    cost_bps = 4
    signal_per_year = len(fs) / (900 / 252)  # 年化

    gross_bps = target.mean()
    net_bps = gross_bps - cost_bps
    pts_per_trade = im_price * net_bps / 10000
    yuan_per_trade = pts_per_trade * multiplier
    yuan_per_year = yuan_per_trade * signal_per_year
    pct_of_account = yuan_per_year / account * 100

    doc.append(f"| 指标 | 值 |")
    doc.append(f"|------|-----|")
    doc.append(f"| 低振幅日/年 | ~17天 |")
    doc.append(f"| First+Short信号/年 | ~{signal_per_year:.0f}笔 |")
    doc.append(f"| 单笔毛收益 | {gross_bps:+.1f} bps |")
    doc.append(f"| 单笔成本 | {cost_bps} bps |")
    doc.append(f"| 单笔净收益 | {net_bps:+.1f} bps |")
    doc.append(f"| 单笔点数 | {pts_per_trade:+.1f} pt |")
    doc.append(f"| 单笔元(1手) | {yuan_per_trade:+,.0f} 元 |")
    doc.append(f"| 年化元(1手) | **{yuan_per_year:+,.0f} 元** |")
    doc.append(f"| 占账户权益 | {pct_of_account:.2f}% |")

    # ══════════════════════════════════════════════
    # 第四步：综合判定
    # ══════════════════════════════════════════════
    doc.append("\n## 4. 综合判定\n")

    pass_t = p_one < 0.05
    pass_ci = ci_lower > 5
    pass_time = len(valid_means) >= 2 and valid_means[-1] >= valid_means[0] * 0.5
    pass_pnl = yuan_per_year > 10000

    doc.append(f"| 条件 | 结果 |")
    doc.append(f"|------|------|")
    doc.append(f"| t检验 p<0.05 | {'✓' if pass_t else '✗'} (p={p_one:.4f}) |")
    doc.append(f"| Bootstrap CI下界>5bps | {'✓' if pass_ci else '✗'} (下界={ci_lower:+.1f}) |")
    doc.append(f"| 时间分布不恶化 | {'✓' if pass_time else '✗'} |")
    doc.append(f"| 年化净收益>1万 | {'✓' if pass_pnl else '✗'} ({yuan_per_year:+,.0f}元) |")

    if pass_t and pass_ci and pass_time and pass_pnl:
        doc.append(f"\n**判定A：升级到小仓位试运行候选** ✓\n")
        doc.append("### 试运行方案\n")
        doc.append(f"- 起始仓位: 标准仓位的30%（约{account*risk_pct*0.3:,.0f}元风险/笔）")
        doc.append(f"- 观察周期: 12个月（预期约{signal_per_year:.0f}笔信号）")
        doc.append(f"- 升级触发: 实盘12笔以上且净收益为正")
        doc.append(f"- 放弃触发: 连续5笔亏损 或 12笔后净收益为负")
    elif pass_t or (ci_lower > 0 and pass_time):
        doc.append(f"\n**判定B：维持观察池** — 统计边缘显著或条件部分满足")
        doc.append(f"- 重新评估: 样本累积到50笔时")
    else:
        doc.append(f"\n**判定C：归档放弃** — 统计不显著")

    report = "\n".join(doc)
    path = Path("tmp") / "horizontal_reversal_first_short_validation.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
