# 节奏摆动日研究归档 (2026-04)

## 研究目标

在节奏摆动日（中等振幅、无明显趋势延续的日子）上建立"冲高反转"做空策略。
解决现有 signal_v2 在这类日子上反复被假突破打损的问题。

## 关键定义

### 节奏摆动日 (Rhythm Swing Day)

```python
def is_rhythm_swing_day(open30_amp_pct, full_amp_pct, am_amp_pct, pm_amp_pct):
    return (
        0.4 <= open30_amp_pct <= 1.2        # 开盘30分钟振幅适中
        and (full_amp_pct / open30_amp_pct) < 1.8  # 全天振幅不远超开盘
        and full_amp_pct < 1.5              # 全天振幅上限
        and am_amp_pct >= pm_amp_pct * 0.8  # 上午振幅不弱于下午
    )
```

占比约24%（年化~60天）。

### 冲高型子集

在节奏摆动日基础上：
- 早盘09:30-10:30振幅 >= 0.4%
- gap >= -0.2%（非大幅低开）
- 占节奏摆动日约75%

## 研究链路

| 步骤 | 内容 | 结果 |
|------|------|------|
| 1 | 节奏摆动日定义与验证 | 4/7和4/10 sanity check通过，定义可用 |
| 2 | 理想化测试 | 196笔 +7.5pt/笔, 70%WR, +1469pt累计 |
| 3 | IS/OOS参数优化 | IS最优stop=8,tgt=0.8: OOS衰退到+5.1pt |
| 4 | OOS衰减诊断 | 基线/优化版同步衰退→市场环境变化,非过拟合 |
| 5 | 基本面分析 | peak→低点理论36pt,入场后可用13pt,入场滞后损失65% |
| 6 | 出场优化(紧target) | tgt=0.3/0.4/0.5全部亏钱(旧定义bug) |
| 7 | Target bug发现 | tgt=0.3时66%交易入场价<target位→伪触发 |
| 8 | Bug修复+IS/OOS | 新定义tgt=0.3 OOS后半+6.9pt(亮点),但整体A3判定 |
| 9 | 新tgt=0.3深度验证 | P2判定:5/7通过,Q1只+0.6pt,逐笔对比33%胜率 |
| 10 | Gap基本面分析 | B1判定:gap是重要维度,大gap组+12.6pt |
| 11 | 对称入场出场 | S3判定:路径B(急反转)97%入场,整体+2.2pt远逊基线 |
| 12 | 上下文感知出场 | CA3判定:4规则全部不如旧基线,CA1 OOS后半-1.0pt |

## 核心数据（最终确认）

| 指标 | 值 |
|------|-----|
| 节奏摆动日占比 | ~24%（年化60天） |
| 冲高型占比 | ~75% |
| 最优OOS avg_pnl | +5.6pt/笔（旧target=0.8基线） |
| OOS后半段avg | +3.3pt |
| 全样本胜率 | 70% |
| 盈亏比 | 0.90（胜率依赖型） |
| 理论空间(peak→低点) | 36pt中位数 |
| 实际可抓取 | 6-8pt（抓取率~20%） |

## 决定不上线的理由

1. 年化预期3-5万/手（+5.6pt × 55笔 × 200元），相对上线维护成本偏低
2. 胜率依赖型策略脆弱（盈亏比<1，单笔大亏-54pt到-67pt）
3. OOS后半段持续衰退（+3.3pt），趋势不明
4. 所有"更聪明"的出场规则都不如简单target=0.8
5. 有更高优先级的研究方向

## 可能的后续方向（如果未来重评）

1. **Gap过滤器**：大gap日(>0.5%) WR=80%+, avg=+10-13pt。样本少(20天)但方向明确
2. **跳低反弹型**（做多方向）：未研究，可能是独立alpha
3. **IC品种测试**：本次只做了IM
4. **Regime实时判定**：盘中能否识别当天是节奏摆动日（研究前置条件）

## 相关代码

| 文件 | 功能 |
|------|------|
| scripts/rhythm_swing_filter.py | 节奏摆动日定义与筛选 |
| scripts/rhythm_swing_short_test.py | 理想化回测（基线参数） |
| scripts/rhythm_swing_param_opt.py | IS/OOS参数优化 |
| scripts/rhythm_swing_drawdown_analysis.py | 基本面分析（回落空间） |
| scripts/rhythm_swing_exit_and_entry.py | 出场优化+早入场分析 |
| scripts/rhythm_swing_target_bug.py | Target定义bug诊断 |
| scripts/rhythm_swing_new_target_and_gap.py | 新Target IS/OOS+Gap分析 |
| scripts/rhythm_swing_tgt03_validation.py | 新tgt=0.3深度统计验证 |
| scripts/rhythm_swing_symmetric.py | 对称入场出场逻辑 |
| scripts/rhythm_swing_context_exit.py | 上下文感知出场逻辑 |

## 相关产出文档

所有在 `tmp/archive/rhythm_swing/` 目录下：

- rhythm_swing_day_filter.md
- rhythm_swing_short_idealized_test.md
- rhythm_swing_short_param_optimization.md
- rhythm_swing_oos_decay_diagnostic.md
- rhythm_swing_drawdown_fundamental_analysis.md
- rhythm_swing_exit_optimization_and_early_entry.md
- target_mechanism_bug_diagnostic.md
- new_target_isoos_and_gap_analysis.md
- new_target_03_deep_validation.md
- symmetric_entry_exit_logic.md
- context_aware_exit_logic.md
