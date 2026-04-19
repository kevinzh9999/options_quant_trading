#!/usr/bin/env python3
"""新Target=0.3深度统计验证（基于entry_price定义）。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats as scipy_stats
from data.storage.db_manager import get_db


def load_im():
    db = get_db()
    df = db.query_df(
        "SELECT datetime, open, high, low, close, volume FROM index_min "
        "WHERE symbol='000852' AND period=300 ORDER BY datetime"
    )
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = df[c].astype(float)
    df.index = pd.to_datetime(df['datetime'])
    df['date'] = df.index.date
    return df


def is_rhythm_swing(day, open_p):
    if len(day) < 10:
        return False
    ob = day.iloc[:6]
    o30 = (ob['high'].max() - ob['low'].min()) / ob['open'].iloc[0] * 100
    dh, dl = day['high'].max(), day['low'].min()
    full = (dh - dl) / open_p * 100
    ratio = full / o30 if o30 > 0 else 99
    am = day[(day.index.hour >= 1) & (day.index.hour < 4)]
    pm = day[(day.index.hour >= 5) & (day.index.hour < 7)]
    am_a = (am['high'].max() - am['low'].min()) / open_p * 100 if len(am) > 0 else 0
    pm_a = (pm['high'].max() - pm['low'].min()) / open_p * 100 if len(pm) > 0 else 0
    return 0.4 <= o30 <= 1.2 and ratio < 1.8 and full < 1.5 and am_a >= pm_a * 0.8


def find_candidates(df):
    daily_close = df.groupby('date')['close'].last()
    all_dates = sorted(df['date'].unique())
    candidates = []
    for i, date in enumerate(all_dates):
        day = df[df['date'] == date]
        if len(day) < 10:
            continue
        open_p = float(day.iloc[0]['open'])
        prev_c = float(daily_close.get(all_dates[i - 1], open_p)) if i > 0 else open_p
        if not is_rhythm_swing(day, open_p):
            continue
        gap = (open_p - prev_c) / prev_c * 100 if prev_c > 0 else 0
        if gap < -0.2:
            continue
        early = day[(day.index.hour == 1) | (day.index.hour == 2)]
        if len(early) < 4:
            continue
        eh, el = early['high'].max(), early['low'].min()
        if (eh - el) / open_p * 100 < 0.4:
            continue
        peak_time = early['high'].idxmax()

        after = day[day.index > peak_time]
        count = 0
        entry_price = None
        entry_idx = None
        for j, (idx, bar) in enumerate(after.iterrows()):
            if float(bar['high']) <= eh:
                count += 1
            else:
                count = 0
            if count >= 3:
                remaining = after.iloc[j + 1:]
                if len(remaining) > 0:
                    entry_price = float(remaining.iloc[0]['open'])
                    entry_idx = remaining.index[0]
                break

        candidates.append({
            'date': date, 'day_bars': day,
            'early_peak': eh, 'early_low': el,
            'early_amp': eh - el, 'peak_time': peak_time,
            'entry_price': entry_price, 'entry_idx': entry_idx,
            'open_price': open_p, 'prev_close': prev_c,
            'gap_pct': gap,
        })
    return candidates


def backtest(cand, target_pct, use_new_target):
    if cand['entry_price'] is None:
        return None
    day = cand['day_bars']
    peak = cand['early_peak']
    amp = cand['early_amp']
    entry_p = cand['entry_price']
    entry_idx = cand['entry_idx']
    stop = peak + 3
    if use_new_target:
        target = entry_p - amp * target_pct
    else:
        target = peak - amp * target_pct

    for idx, bar in day[day.index >= entry_idx].iterrows():
        bh, bl = float(bar['high']), float(bar['low'])
        if bh >= stop:
            return {'pnl': entry_p - stop, 'reason': 'stop', 'date': cand['date'],
                    'entry_p': entry_p, 'exit_p': stop}
        if bl <= target:
            return {'pnl': entry_p - target, 'reason': 'target', 'date': cand['date'],
                    'entry_p': entry_p, 'exit_p': target}
        if idx.hour >= 6:
            exit_p = float(bar['open'])
            return {'pnl': entry_p - exit_p, 'reason': 'time', 'date': cand['date'],
                    'entry_p': entry_p, 'exit_p': exit_p}
    last = day.iloc[-1]
    return {'pnl': entry_p - float(last['close']), 'reason': 'eod', 'date': cand['date'],
            'entry_p': entry_p, 'exit_p': float(last['close'])}


def main():
    print("加载数据...")
    df = load_im()
    print("筛选候选日...")
    candidates = find_candidates(df)
    n = len(candidates)
    print(f"候选日: {n}天")

    # 跑新target=0.3
    trades_new = []
    for c in candidates:
        t = backtest(c, 0.3, use_new_target=True)
        if t:
            t['gap_pct'] = c['gap_pct']
            trades_new.append(t)

    # 跑旧target=0.8 (对比用)
    trades_old = []
    for c in candidates:
        t = backtest(c, 0.8, use_new_target=False)
        if t:
            trades_old.append(t)

    tdf = pd.DataFrame(trades_new)
    pnl = tdf['pnl']
    n_trades = len(tdf)

    doc = ["# 新 Target=0.3 深度统计验证\n"]
    doc.append(f"参数: 新定义 target = entry_price - early_amp x 0.3")
    doc.append(f"其他: 入场3bar无新高, 止损peak+3, 时间平仓14:00")
    doc.append(f"样本: {n_trades}笔 (196天候选日)\n")

    # ═══════════════════════════════════════════════
    # 第一步: 4段时间分布
    # ═══════════════════════════════════════════════
    doc.append("## 第一步: 4段时间分布\n")
    q_size = n_trades // 4
    doc.append("| 段 | 时间范围 | N | 胜率 | 平均pnl | 中位数pnl | 累计pnl |")
    doc.append("|---|---------|---|------|---------|----------|---------|")

    q_avgs = []
    for i in range(4):
        start = i * q_size
        end = (i + 1) * q_size if i < 3 else n_trades
        q = tdf.iloc[start:end]
        qp = q['pnl']
        wr = (qp > 0).sum() / len(qp) * 100
        q_avgs.append(qp.mean())
        doc.append(f"| Q{i + 1} | {q['date'].iloc[0]}~{q['date'].iloc[-1]} | {len(q)} | "
                   f"{wr:.0f}% | {qp.mean():+.1f}pt | {qp.median():+.1f}pt | {qp.sum():+.0f}pt |")

    all_above_4 = all(a >= 4.0 for a in q_avgs)
    doc.append(f"\n4段都>=+4pt: {'✓' if all_above_4 else '✗'} ({', '.join(f'{a:+.1f}' for a in q_avgs)})")

    # ═══════════════════════════════════════════════
    # 第二步: 统计显著性
    # ═══════════════════════════════════════════════
    doc.append("\n## 第二步: 统计显著性\n")

    # 2.1 描述性统计
    doc.append("### 2.1 描述性统计（全样本）\n")
    doc.append(f"| 指标 | 值 |")
    doc.append(f"|------|-----|")
    doc.append(f"| N | {n_trades} |")
    doc.append(f"| 均值 | {pnl.mean():+.2f}pt |")
    doc.append(f"| 中位数 | {pnl.median():+.2f}pt |")
    doc.append(f"| 标准差 | {pnl.std():.2f}pt |")
    doc.append(f"| 25% | {pnl.quantile(0.25):+.1f}pt |")
    doc.append(f"| 75% | {pnl.quantile(0.75):+.1f}pt |")
    doc.append(f"| 最小值 | {pnl.min():+.1f}pt |")
    doc.append(f"| 最大值 | {pnl.max():+.1f}pt |")
    cohens_d = pnl.mean() / pnl.std() if pnl.std() > 0 else 0
    doc.append(f"| Cohen's d | {cohens_d:.3f} |")

    # 2.2 t检验（全样本）
    doc.append("\n### 2.2 t检验（全样本, H0: mean=0）\n")
    t_stat, p_val_two = scipy_stats.ttest_1samp(pnl.values, 0)
    p_val_one = p_val_two / 2 if t_stat > 0 else 1 - p_val_two / 2
    doc.append(f"- t统计量: {t_stat:.3f}")
    doc.append(f"- p值(单侧): {p_val_one:.6f}")
    doc.append(f"- p < 0.01: {'✓' if p_val_one < 0.01 else '✗'}")
    doc.append(f"- p < 0.001: {'✓' if p_val_one < 0.001 else '✗'}")

    # 2.3 Bootstrap 95% CI
    doc.append("\n### 2.3 Bootstrap 95% CI（全样本）\n")
    np.random.seed(42)
    boot_means = []
    vals = pnl.values
    for _ in range(1000):
        sample = np.random.choice(vals, size=len(vals), replace=True)
        boot_means.append(np.mean(sample))
    ci_lo = np.percentile(boot_means, 2.5)
    ci_hi = np.percentile(boot_means, 97.5)
    doc.append(f"- 95% CI: [{ci_lo:+.2f}, {ci_hi:+.2f}]")
    doc.append(f"- CI下界 >= +4: {'✓' if ci_lo >= 4.0 else '✗'}")

    # 2.4 OOS后半33笔单独检验
    doc.append("\n### 2.4 OOS后半33笔单独检验\n")
    oos2_start = 130 + 66 // 2  # IS=130, OOS前半=33
    oos2_trades = tdf.iloc[oos2_start:]
    oos2_pnl = oos2_trades['pnl']
    doc.append(f"样本: {len(oos2_trades)}笔, avg={oos2_pnl.mean():+.1f}pt, WR={((oos2_pnl > 0).sum() / len(oos2_pnl) * 100):.0f}%")
    if len(oos2_pnl) >= 3:
        t2, p2_two = scipy_stats.ttest_1samp(oos2_pnl.values, 0)
        p2_one = p2_two / 2 if t2 > 0 else 1 - p2_two / 2
        doc.append(f"- t统计量: {t2:.3f}")
        doc.append(f"- p值(单侧): {p2_one:.4f}")
        doc.append(f"- p < 0.05: {'✓' if p2_one < 0.05 else '✗'}")

        # Bootstrap CI for OOS2
        boot2 = []
        v2 = oos2_pnl.values
        for _ in range(1000):
            s = np.random.choice(v2, size=len(v2), replace=True)
            boot2.append(np.mean(s))
        ci2_lo = np.percentile(boot2, 2.5)
        ci2_hi = np.percentile(boot2, 97.5)
        doc.append(f"- Bootstrap 95% CI: [{ci2_lo:+.2f}, {ci2_hi:+.2f}]")

    # ═══════════════════════════════════════════════
    # 第三步: 胜率盈亏分布
    # ═══════════════════════════════════════════════
    doc.append("\n## 第三步: 胜率和盈亏分布\n")
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    doc.append("| 组 | 笔数 | 占比 | 均值 | 中位数 | 最大 | 最小 |")
    doc.append("|---|------|------|------|--------|------|------|")
    doc.append(f"| 盈利 | {len(wins)} | {len(wins)/n_trades*100:.0f}% | {wins.mean():+.1f} | "
               f"{wins.median():+.1f} | {wins.max():+.1f} | {wins.min():+.1f} |")
    doc.append(f"| 亏损 | {len(losses)} | {len(losses)/n_trades*100:.0f}% | {losses.mean():+.1f} | "
               f"{losses.median():+.1f} | {losses.max():+.1f} | {losses.min():+.1f} |")

    wr = len(wins) / n_trades * 100
    pnl_ratio = wins.mean() / abs(losses.mean()) if len(losses) > 0 and losses.mean() != 0 else np.inf
    expected = wr / 100 * wins.mean() - (1 - wr / 100) * abs(losses.mean())
    doc.append(f"\n| 指标 | 值 |")
    doc.append(f"|------|-----|")
    doc.append(f"| **胜率** | **{wr:.0f}%** |")
    doc.append(f"| **盈亏比** | **{pnl_ratio:.2f}** |")
    doc.append(f"| 期望值 | {expected:+.1f}pt (应约={pnl.mean():+.1f}) |")

    # ═══════════════════════════════════════════════
    # 第四步: 极端值检查
    # ═══════════════════════════════════════════════
    doc.append("\n## 第四步: 极端值检查\n")
    doc.append("**最赚5笔:**")
    for _, t in tdf.nlargest(5, 'pnl').iterrows():
        doc.append(f"  {t['date']} {t['reason']} {t['pnl']:+.1f}pt")
    doc.append("\n**最亏5笔:**")
    for _, t in tdf.nsmallest(5, 'pnl').iterrows():
        doc.append(f"  {t['date']} {t['reason']} {t['pnl']:+.1f}pt")

    sorted_pnl = pnl.sort_values()
    trim_top3 = sorted_pnl.iloc[:-3]
    trim_bot3 = sorted_pnl.iloc[3:]
    trim_both3 = sorted_pnl.iloc[3:-3]

    doc.append(f"\n**剔除检验:**")
    doc.append(f"| 场景 | N | 均值 | 变化 |")
    doc.append(f"|------|---|------|------|")
    doc.append(f"| 全部 | {n_trades} | {pnl.mean():+.1f} | - |")
    doc.append(f"| 剔除最赚3笔 | {len(trim_top3)} | {trim_top3.mean():+.1f} | {trim_top3.mean()-pnl.mean():+.1f} |")
    doc.append(f"| 剔除最亏3笔 | {len(trim_bot3)} | {trim_bot3.mean():+.1f} | {trim_bot3.mean()-pnl.mean():+.1f} |")
    doc.append(f"| 剔除各3笔 | {len(trim_both3)} | {trim_both3.mean():+.1f} | {trim_both3.mean()-pnl.mean():+.1f} |")
    doc.append(f"\n剔除最赚3笔后均值>=+5: {'✓' if trim_top3.mean() >= 5.0 else '✗'}")

    # ═══════════════════════════════════════════════
    # 第五步: 退出原因分布
    # ═══════════════════════════════════════════════
    doc.append("\n## 第五步: 退出原因分布\n")
    doc.append("| 退出原因 | 笔数 | 占比 | 平均pnl | 胜率 |")
    doc.append("|---------|------|------|---------|------|")
    for reason in ['target', 'stop', 'time', 'eod']:
        sub = tdf[tdf['reason'] == reason]
        if len(sub) == 0:
            continue
        sub_wr = (sub['pnl'] > 0).sum() / len(sub) * 100
        doc.append(f"| {reason} | {len(sub)} | {len(sub)/n_trades*100:.0f}% | "
                   f"{sub['pnl'].mean():+.1f} | {sub_wr:.0f}% |")

    tgt_rate = (tdf['reason'] == 'target').sum() / n_trades * 100
    doc.append(f"\ntarget触发率>=50%: {'✓' if tgt_rate >= 50 else '✗'} ({tgt_rate:.0f}%)")

    # ═══════════════════════════════════════════════
    # 第六步: 跟旧基线逐笔对比（OOS 66笔）
    # ═══════════════════════════════════════════════
    doc.append("\n## 第六步: 跟旧基线逐笔对比（OOS 66笔）\n")

    # OOS范围: index 130~195
    oos_new = tdf.iloc[130:]
    oos_old_df = pd.DataFrame(trades_old).iloc[130:]

    # 按日期对齐
    new_by_date = {t['date']: t['pnl'] for _, t in oos_new.iterrows()}
    old_by_date = {t['date']: t['pnl'] for _, t in oos_old_df.iterrows()}

    common_dates = sorted(set(new_by_date.keys()) & set(old_by_date.keys()))
    new_better = 0
    old_better = 0
    same = 0
    diffs = []

    doc.append("| 日期 | 旧pnl(tgt=0.8) | 新pnl(tgt=0.3) | 差异 | 谁好 |")
    doc.append("|------|---------------|---------------|------|------|")
    for d in common_dates:
        np_val = new_by_date[d]
        op_val = old_by_date[d]
        diff = np_val - op_val
        diffs.append(diff)
        if abs(diff) < 0.5:
            winner = "≈"
            same += 1
        elif diff > 0:
            winner = "新"
            new_better += 1
        else:
            winner = "旧"
            old_better += 1
        doc.append(f"| {d} | {op_val:+.1f} | {np_val:+.1f} | {diff:+.1f} | {winner} |")

    total_compared = len(common_dates)
    doc.append(f"\n**逐笔对比统计:**")
    doc.append(f"- 比较天数: {total_compared}")
    doc.append(f"- 新更好: {new_better} ({new_better/total_compared*100:.0f}%)")
    doc.append(f"- 旧更好: {old_better} ({old_better/total_compared*100:.0f}%)")
    doc.append(f"- 差不多(±0.5pt): {same} ({same/total_compared*100:.0f}%)")
    if diffs:
        doc.append(f"- 差异均值: {np.mean(diffs):+.1f}pt")
        doc.append(f"- 差异中位数: {np.median(diffs):+.1f}pt")

    new_win_rate = new_better / total_compared * 100 if total_compared > 0 else 0
    doc.append(f"\n新定义逐笔胜率>=60%: {'✓' if new_win_rate >= 60 else '✗'} ({new_win_rate:.0f}%)")

    # ═══════════════════════════════════════════════
    # 第七步: 按gap组拆解
    # ═══════════════════════════════════════════════
    doc.append("\n## 第七步: 按Gap组拆解\n")

    def gap_group(g):
        if abs(g) <= 0.1:
            return '无gap(<=0.1%)'
        elif g > 0.1 and g <= 0.5:
            return '小gap(0.1-0.5%)'
        elif g > 0.5 and g <= 1.0:
            return '中gap(0.5-1.0%)'
        elif g > 1.0:
            return '大gap(>1.0%)'
        else:
            return '跳低(<-0.1%)'

    tdf['gap_group'] = tdf['gap_pct'].apply(gap_group)

    doc.append("### 新target=0.3\n")
    doc.append("| gap组 | N | WR | Avg | Total |")
    doc.append("|-------|---|----|-----|-------|")
    group_order = ['跳低(<-0.1%)', '无gap(<=0.1%)', '小gap(0.1-0.5%)', '中gap(0.5-1.0%)', '大gap(>1.0%)']
    for grp in group_order:
        sub = tdf[tdf['gap_group'] == grp]
        if len(sub) < 2:
            continue
        wr = (sub['pnl'] > 0).sum() / len(sub) * 100
        doc.append(f"| {grp} | {len(sub)} | {wr:.0f}% | {sub['pnl'].mean():+.1f} | {sub['pnl'].sum():+.0f} |")

    doc.append("\n### 旧基线target=0.8对照\n")
    doc.append("| gap组 | N | WR | Avg | Total |")
    doc.append("|-------|---|----|-----|-------|")
    old_df = pd.DataFrame(trades_old)
    old_df['gap_pct'] = [c['gap_pct'] for c in candidates[:len(old_df)]]
    old_df['gap_group'] = old_df['gap_pct'].apply(gap_group)
    for grp in group_order:
        sub = old_df[old_df['gap_group'] == grp]
        if len(sub) < 2:
            continue
        wr = (sub['pnl'] > 0).sum() / len(sub) * 100
        doc.append(f"| {grp} | {len(sub)} | {wr:.0f}% | {sub['pnl'].mean():+.1f} | {sub['pnl'].sum():+.0f} |")

    # ═══════════════════════════════════════════════
    # 综合判定
    # ═══════════════════════════════════════════════
    doc.append("\n## 综合判定\n")

    checks = {
        '4段都>=+4pt': all_above_4,
        't检验 p<0.01': p_val_one < 0.01,
        'Bootstrap CI下界>=+4': ci_lo >= 4.0,
        'OOS后半 t检验 p<0.05': p2_one < 0.05 if len(oos2_pnl) >= 3 else False,
        '剔除最赚3笔后>=+5': trim_top3.mean() >= 5.0,
        'target触发率>=50%': tgt_rate >= 50,
        '逐笔对比新定义胜率>=60%': new_win_rate >= 60,
    }

    doc.append("| 条件 | 结果 | 通过? |")
    doc.append("|------|------|-------|")
    pass_count = 0
    for name, passed in checks.items():
        doc.append(f"| {name} | | {'✓' if passed else '✗'} |")
        if passed:
            pass_count += 1

    doc.append(f"\n通过: {pass_count}/{len(checks)}\n")

    if pass_count == len(checks):
        doc.append(f"**判定P1: 可以进入下一步(上线前准备)** ✓")
        doc.append(f"全部{len(checks)}项条件通过。")
    elif pass_count >= len(checks) - 2:
        doc.append(f"**判定P2: 信号真实但仍有不确定性**")
        doc.append(f"{pass_count}/{len(checks)}项通过。")
        fails = [k for k, v in checks.items() if not v]
        doc.append(f"未通过: {', '.join(fails)}")
    else:
        doc.append(f"**判定P3: 信号不够可靠** ✗")
        doc.append(f"仅{pass_count}/{len(checks)}项通过。")
        fails = [k for k, v in checks.items() if not v]
        doc.append(f"未通过: {', '.join(fails)}")

    report = "\n".join(doc)
    path = Path("tmp") / "new_target_03_deep_validation.md"
    with open(path, 'w') as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
