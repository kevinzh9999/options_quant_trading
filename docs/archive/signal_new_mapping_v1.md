# Signal New Mapping V1 系统文档

> 创建: 2026-04-12
> 状态: Shadow trade 观察期（v2实盘 + v1并行shadow）

## 一、系统概述

v1是独立于v2的**入场评分系统**，用数据驱动的映射替代v2的手工分档。
出场逻辑完全复用v2的check_exit（trailing_stop、ME、MID_BREAK等不变）。

### 设计动机

MVQB诊断（2026-04-12）发现v2评分系统的3个结构性问题：
1. V分(ATR ratio)的PnL关系是U型，v2假设单调递减是错的
2. M分的0.3%边界处两侧PnL几乎相同，分档不是有效断点
3. 10:00-11:00时段PnL最差(+0~+1pt)，14:00-15:00最好(+3~+4pt)，v2没有利用

### 解决方案

用900天全量trade数据驱动设计每个子分量的映射：
- 按原始连续值分10桶，每桶算实际PnL
- 按PnL排序分配0~max_score的分数
- 新增Session加分和Gap加分维度

## 二、评分公式

```
total_score = M(0-50) + V(0-30) + Q(0-15) + Session(-10~+10) + Gap(0~+5)
范围: -10 到 110
```

剔除了v2的B分（布林带突破，几乎从未触发）和S分（启动因子，方向反）。

## 三、Per-Symbol 映射

### v1_im（IM专用）

基于IM 900天trade数据驱动设计。

**M分映射（10桶）：**

| raw_mom_5m范围 | M分 | 设计依据PnL |
|-------------|-----|---------|
| [0, 0.00122) | 0 | +0.5 |
| [0.00122, 0.00191) | 17 | +1.0 |
| [0.00191, 0.00247) | 11 | +0.9 |
| [0.00247, 0.00319) | 33 | +2.8 |
| [0.00319, 0.00393) | 28 | +2.2 |
| [0.00393, 0.00497) | 6 | +0.7 |
| [0.00497, 0.00632) | 22 | +1.2 |
| [0.00632, 0.00872) | 50 | +4.1 |
| [0.00872, 0.0128) | 39 | +2.8 |
| [0.0128, 1.0) | 44 | +3.5 |

**V分映射（10桶，不规则多峰）：**

| raw_atr_ratio范围 | V分 | 设计依据PnL |
|----------------|-----|---------|
| [0, 0.666) | 30 | +4.7 |
| [0.666, 0.776) | 20 | +2.2 |
| [0.776, 0.864) | 3 | -0.2 |
| [0.864, 0.958) | 13 | +1.5 |
| [0.958, 1.065) | 10 | +1.2 |
| [1.065, 1.215) | 17 | +2.0 |
| [1.215, 1.424) | 0 | -0.9 |
| [1.424, 1.655) | 27 | +4.3 |
| [1.655, 1.987) | 7 | +0.8 |
| [1.987, 100) | 23 | +4.1 |

**Session加分：** {09:+10, 10:-10, 11:+4, 13:0, 14:+8}

**Q分：** percentile法 15/8/0（阈值75%/25%）

**Gap：** aligned +5, not 0

**Threshold：** 60

### v1_ic（IC专用）

当前参数值与v1_im相同（两品种市场结构相似），独立存储便于未来独立优化。

**Threshold：** 60

**映射阈值：** 同v1_im（_IC_M_THRESHOLDS, _IC_V_THRESHOLDS, _IC_SESSION独立定义）

## 四、Backtest 验证结果

### v1_im Sweet Spot (thr=60)

| 指标 | v2基线 | v1_im | 改善 |
|------|--------|-------|------|
| 信号数 | 2598 | 1175 | -54.8% |
| 总PnL | +6296 | +6116 | -2.9% |
| 单笔PnL | +2.42 | +5.21 | +114.8% |
| 胜率 | 45% | 51% | +6% |

扩展窗口验证（v1_im在所有60-681天窗口都优于v2）：

| 窗口 | v1-v2总PnL |
|------|----------|
| 60天 | +340 |
| 120天 | +384 |
| 450天 | +490 |
| 681天 | +220 |

### v1_ic Baseline (thr=60, IM参数集)

| 指标 | v2基线 | v1_ic | 改善 |
|------|--------|-------|------|
| 信号数 | 2944 | 1010 | -65.7% |
| 总PnL | +4785 | +3396 | -29.0% |
| 单笔PnL | +1.63 | +3.36 | +106.9% |

IC的v1总PnL低于v2，但单笔效率2x稳定。作为baseline待未来独立优化。

## 五、工程架构

### 代码位置

```
strategies/intraday/experimental/
├── __init__.py
├── score_components_new.py      # 统一映射函数（v1_ic旧版，保留参考）
├── signal_new_mapping_v1_im.py  # IM专用评分模块
├── signal_new_mapping_v1_ic.py  # IC专用评分模块（独立参数）
└── monitor_v1_overlay.py        # Monitor overlay（shadow trade管理）
```

### Monitor 集成

v1不是独立进程，而是**overlay**嵌入在现有monitor的`_on_new_bar()`末尾：

```python
# monitor.py _on_new_bar() 末尾
if self._v1_overlay:
    self._v1_overlay.on_bar(bar_data, bar_15m_data, current_time_utc)
```

- 共享v2的TQ数据源、bar构建、vol_profile
- 独立的shadow position管理
- 写入`shadow_trades_new_mapping`表
- v2主流程完全不受影响（try/except包裹）

### 数据库

```sql
-- 新表
shadow_trades_new_mapping (
    ... v2 shadow_trades 的所有字段 ...
    raw_mom_5m, raw_atr_ratio, raw_vol_pct,
    entry_session, session_bonus, gap_aligned, gap_bonus,
    signal_version TEXT  -- 'v1_im' 或 'v1_ic'
)

-- v2表加了raw_字段（兼容）
shadow_trades ADD COLUMN raw_mom_5m, raw_atr_ratio, raw_vol_pct, entry_session, gap_aligned
```

### Executor

`order_executor.py` 加了 `--signal-source v2|v1` 参数，默认v2。
4周观察期后根据shadow trade数据决定是否切换。

## 六、观察期计划

### 4周观察标准

| 指标 | 成功标准 |
|------|---------|
| 累计PnL | > v2 +5% |
| 单笔效率 | > v2 +10% |
| 胜率 | >= v2 -3% |
| 最大单日亏损 | < v2 × 1.5 |
| 信号频率 | v2的0.5x~2.0x |

### 时间线

- 2026-04-14（周一）：启动monitor，v1 overlay开始记录shadow trade
- 第1周：确认系统稳定运行
- 第2周末：第一次中期review
- 第4周末：决策点（是否上线v1）

## 七、已排除的方向

| 方向 | 结果 | 原因 |
|------|------|------|
| v1_ic统一U型映射 | 失败 | 近期120天-549pt(-42%)，session_bonus=-10误杀大赚trade |
| IC per-symbol(900天) | 失败 | 近期120天-501pt，跟统一映射同样问题 |
| IC per-symbol(681天/219天交叉) | 部分 | 方向B近120天有sweet spot但全样本/681天失败 |
| Score衰减出场规则 | 失败 | 信号r=-0.5但不可操作（原则26：物理双重性） |
| Score区间过滤 | 失败 | 最优G候选仅+3.6%改善，不值得 |
