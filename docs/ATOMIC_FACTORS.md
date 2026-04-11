# 原子因子与算子参考

> 代码: `strategies/intraday/atomic_factors.py`
> 所有因子为纯计算函数，无交易逻辑。Entry和Exit策略共享。
>
> **时间标准: 全部UTC**（所有数据库表已统一为UTC，BJ时间仅用于面板显示）
> **Baseline: IM +2112  IC +2137 = +4249pt/219天**（中性外部数据）

## 原子因子

### 价格类

| 因子 | 函数 | 输入 | 输出 | 说明 |
|------|------|------|------|------|
| 价格动量 | `momentum(close, lb)` | close数组, lookback根数 | float (%) | `(close[-1] - close[-lb-1]) / close[-lb-1]` |
| 动量方向 | `momentum_direction(mom)` | 动量值 | "LONG"/"SHORT"/"" | 正→LONG，负→SHORT |
| 振幅 | `amplitude(close, n=48)` | close数组, 窗口 | float (%) | `(max-min)/first`，衡量区间波动 |

### 布林带类

| 因子 | 函数 | 输入 | 输出 | 说明 |
|------|------|------|------|------|
| 布林参数 | `boll_params(series, period=20)` | close序列, 周期 | (mid, std) | 20根均值和标准差 |
| 布林zone | `boll_zone(price, mid, std)` | 价格, 中轨, 标准差 | zone字符串 | 6个区域判定 |
| 布林宽度 | `boll_width(std)` | 标准差 | float | `4 × std` |

**布林zone定义:**
```
ABOVE_UPPER  : price >= mid + 2σ
UPPER_ZONE   : mid + σ <= price < mid + 2σ
MID_UPPER    : mid <= price < mid + σ
MID_LOWER    : mid - σ <= price < mid
LOWER_ZONE   : mid - 2σ <= price < mid - σ
BELOW_LOWER  : price < mid - 2σ
```

### 波动率类

| 因子 | 函数 | 输入 | 输出 | 说明 |
|------|------|------|------|------|
| ATR | `atr(H, L, C, period)` | OHLC数组, 周期 | float | Average True Range |
| ATR比率 | `atr_ratio(H, L, C, short=5, long=40)` | OHLC, 短/长周期 | float | `ATR(5)/ATR(40)`。<1=扩张, >1=收缩 |

### 成交量类

| 因子 | 函数 | 输入 | 输出 | 说明 |
|------|------|------|------|------|
| 量能分位 | `volume_percentile(vol, hist)` | 当前量, 历史同时段列表 | float (0-1) | 在历史分布中的百分位 |
| 量比 | `volume_ratio(vol, avg)` | 当前量, 均值 | float | `cur/avg`，>1.5=放量 |

### K线形态类

| 因子 | 函数 | 输入 | 输出 | 说明 |
|------|------|------|------|------|
| K线收窄 | `narrow_range(bar, n=3, std)` | bar数据, n根, 布林std | float | `range/boll_width`，越小=越收窄 |
| 趋势持续 | `price_trending(bar, n=3, std, dir)` | bar数据, 方向 | bool | close变化 > 布林宽度×5% |
| 突破前高低 | `breakout_prev_range(C, H, L, n=5)` | OHLC, 窗口 | (bool, dir) | 突破前n根bar极值 |

### 持仓状态类

| 因子 | 函数 | 输入 | 输出 | 说明 |
|------|------|------|------|------|
| 持仓时间 | `hold_time(entry_utc, cur_utc)` | 时间字符串HH:MM | int (分钟) | 入场到当前的分钟数 |
| 盈亏比 | `pnl_pct(price, entry, dir)` | 当前价, 入场价, 方向 | float (%) | 正=盈利, 负=亏损 |
| 回撤幅度 | `trailing_drawdown(price, extreme, dir)` | 当前价, 极值 | float (%) | 从极值的回撤比例 |

### 技术指标类

| 因子 | 函数 | 输入 | 输出 | 说明 |
|------|------|------|------|------|
| RSI | `rsi(close, period=14)` | close数组 | float (0-100) | 相对强弱指标 |

---

## Entry组合因子

> 代码: `strategies/intraday/factors.py` → `ScoringFactor`子类
> 由原子因子组合而成，输出整数分值。

| 因子 | 类名 | 分值 | 组合的原子因子 |
|------|------|------|--------------|
| M 动量 | `MomentumFactor` | 0-50 | `amplitude` → 动态lb选择; `momentum` × 2周期; `momentum_direction` |
| V 波动率 | `VolatilityFactor` | 0-30 | `atr`, `atr_ratio` |
| Q 成交量 | `VolumeFactor` | 0-20 | `volume_percentile`（优先）; `volume_ratio`（fallback）|
| B 突破 | `BreakoutFactor` | 0-20 | `boll_zone`, `momentum_direction`, `volume_ratio` |
| S 启动 | `StartupFactor` | 0-15 | `breakout_prev_range`, `amplitude`, `volume_percentile` |

## Exit条件因子

> 代码: `strategies/intraday/factors.py` → `ExitCondition`子类
> 由原子因子组合而成，输出exit决策。

| 条件 | 类名 | 优先级 | 组合的原子因子 |
|------|------|--------|--------------|
| 止损 | `StopLossCondition` | P10 | `pnl_pct` |
| 午休 | `LunchCloseCondition` | P15 | `pnl_pct`, `trailing_drawdown` |
| 跟踪止盈 | `TrailingStopCondition` | P30 | `trailing_drawdown`, `hold_time`, `boll_zone` |
| 趋势完成 | `TrendCompleteCondition` | P40 | `boll_zone`(5m), `boll_zone`(15m) |
| 动量耗尽 | `MomentumExhaustedCondition` | P50 | `narrow_range`, `price_trending`, `boll_zone`, `hold_time` |
| 中轨突破 | `MidBreakCondition` | P60 | `boll_zone`(5m), `boll_zone`(15m), 连续计数 |
| 超时止损 | `TimeStopCondition` | P70 | `hold_time`, `pnl_pct` |

---

## 乘数算子（Entry后处理管道）

> 作用于因子总分之上的乘法调节。当前全部中性化=1.0。

| 算子 | 范围 | 作用 | 当前状态 |
|------|------|------|---------|
| daily_mult | 0.7-1.2 | 日线顺/逆势 | 中性化 (daily_bar=None) |
| intraday_filter | 0.3-1.0 | 日内涨跌幅方向过滤 | 中性化 (daily_bar=None) |
| time_weight | 0.3-1.2 | 时段权重 (per-symbol session_multiplier) | **活跃** |
| sentiment_mult | 0.5-1.5 | 期权情绪 (IV/RR/PCR) | 中性化 (sentiment=None) |
| zscore_filter | 0-1.5× | 极端Z值硬过滤/增幅 | 中性化 (zscore=None) |
| rsi_bonus | +0-15 | RSI极端值加分 | 中性化 (zscore=None) |

---

## Per-Symbol配置项

> 单一来源: `SYMBOL_PROFILES` in `A_share_momentum_signal_v2.py`

| 配置 | 类型 | IM | IC | 说明 |
|------|------|-----|-----|------|
| `factor_weights` | dict | 全1.0 | 全1.0 | Entry因子权重 {momentum:1.0, ...} |
| `exit_weights` | dict | — | — | Exit条件开关 {STOP_LOSS:1.0, MID_BREAK:0, ...} |
| `signal_threshold` | int | 50 (验证中→55) | 60 | 开仓最低分 |
| `stop_loss_pct` | float | 0.003 | 0.005 | 止损百分比 |
| `trailing_stop_scale` | float | 1.5 | 2.0 | trailing宽度倍数 |
| `me_ratio` | float | 0.10 | 0.12 | ME收窄判定阈值 |
| `dm_trend` | float | 1.1 | 1.1 | 日线顺势乘数 |
| `dm_contrarian` | float | 0.9 | 0.9 | 日线逆势乘数 |
| `session_multiplier` | dict | per时段 | per时段 | 5个交易时段的权重 |

---

## 因子评估避坑清单

> 从横盘反转因子等研究中提炼的经验教训。

1. **Daily IC要配合分组幅度验证**：IC数值本身不能决定因子价值。全局Daily IC=0.11但分组后Pos-Neg均值差<5bps→无实用价值。必须看分组后的绝对收益差（≥10bps才算有效）。

2. **高振幅日存在drift污染**：高振幅日往往有方向性趋势，因子IC可能反映的是drift而非因子预测力。评估时要考虑demean或跟benchmark对比。

3. **前置经济解释成立 ≠ 统计验证成立**：两者都是必要条件。"横盘后反转"逻辑上合理但统计上不支持（两品种都是趋势延续）。

4. **能排除一个方向也是有价值的研究结果**：记录到TODO.md负面知识区，避免未来重复投入。

5. **exit条件不能直接对称化为entry因子**：MomentumExhausted作为平仓信号有效（停止亏损），但不等于平仓点是反转入场点。exit和entry的信号逻辑不对称。

6. **Regime分组IC方向矛盾时，先做分组幅度验证**：矛盾可能是假象（如本次IM/IC高振幅日看似方向相反，实际分组后都是趋势延续）。
