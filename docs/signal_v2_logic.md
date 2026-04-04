# SignalGeneratorV2 信号系统完整文档

> 最后更新：2026-04-04 | 代码位置：`strategies/intraday/A_share_momentum_signal_v2.py`

---

## 1. 系统概览

SignalGeneratorV2 是一个多维度评分系统，对A股股指期货（IM/IC/IF/IH）生成日内交易信号。每根5分钟K线产生一个0-100分的评分，超过阈值（默认60分，IC用65分）触发交易信号。

### 信号评分总公式

```
total = clamp(raw_total × daily_mult × intraday_filter × time_weight × sentiment_mult, 0, 100)
      → Z-Score硬过滤（高波动时）
      → RSI反转加分（|Z|>2时）
```

其中 `raw_total = M分 + V分 + Q分 + B分`（最高120分，乘以各打折系数后压缩到0-100）。

### 信号评分流水线

```
    bar_5m (已完成的5分钟K线)
         │
    ┌────┴────┐
    │ 评分维度 │
    ├─────────┤
    │ M 动量   │ 0-50分  ← 5分钟+15分钟双周期动量
    │ V 波动率 │ 0-30分  ← ATR短期/长期比值
    │ Q 成交量 │ 0-20分  ← 当前量/20根均量
    │ B 布林突破│ 0-20分  ← 中轨突破+放量+多周期确认
    └────┬────┘
         │ raw_total = M + V + Q + B
         │
    ┌────┴────────────────┐
    │ 乘数链（逐步打折）   │
    ├────────────────────┤
    │ × daily_mult       │ 0.8-1.2  ← 日线方向一致性
    │ × intraday_filter  │ 0.3-1.0  ← 日内涨跌幅惩罚
    │ × time_weight      │ 0.6-1.1  ← 时段权重
    │ × sentiment_mult   │ 0.5-1.5  ← 期权情绪
    └────┬────────────────┘
         │ total = clamp(0, 100)
         │
    ┌────┴────────────────┐
    │ 过滤层               │
    ├────────────────────┤
    │ Z-Score硬过滤       │ 高波动时：超卖做空→阻断，超买做多→阻断
    │ RSI反转加分         │ |Z|>2时：RSI反转确认→+10分
    └────┬────────────────┘
         │
    score >= threshold? → 触发信号
```

---

## 2. 数据输入

### 实时数据（每根5分钟K线更新）

| 数据 | 来源 | 说明 |
|------|------|------|
| bar_5m | index_min (现货指数) | 最近200根5分钟K线，**不含当前forming bar** |
| bar_15m | 从bar_5m重采样 | 15分钟K线（同样排除forming bar） |
| 期货盘口 | TQ实时 | bid1/ask1/last_price（仅monitor用，backtest无） |

### 盘前数据（每天加载一次）

| 数据 | 来源 | 日期范围 | 说明 |
|------|------|---------|------|
| daily_df | index_daily | trade_date < 当天 | 最近30天日线收盘价 |
| Z-Score (ema20, std20) | index_daily | trade_date < 当天 | 现货指数EMA20和STD20 |
| is_high_vol | daily_model_output | trade_date < 当天 | GARCH forecast / 24.9 > 1.2 |
| sentiment | daily_model_output | trade_date < 当天 | ATM IV变化、RR、VRP |
| d_override | morning_briefing | trade_date = 当天 | 盘前方向覆盖（LONG/SHORT各一个系数） |

### Lookahead防护

| 检查项 | 处理方式 |
|--------|---------|
| 5分钟K线 | `bar_5m_signal = bar_5m.iloc[:-1]`（排除当前forming bar） |
| 15分钟K线 | 从bar_5m_signal重采样（不含当前bar） |
| 日线数据 | `trade_date < 当天`（严格小于） |
| GARCH/sentiment | `trade_date < 当天` |
| Z-Score | `trade_date < 当天` |
| Morning Briefing | `trade_date = 当天`（盘前产生，合法） |

---

## 3. 评分维度详解

### M分：动量（0-50分）

**计算**：
```
mom_5m = (close[-1] - close[-13]) / close[-13]    # 12根bar = 60分钟动量
mom_15m = (c15[-1] - c15[-7]) / c15[-7]           # 6根15m bar = 90分钟动量
```

**方向一致性检查**：如果5分钟和15分钟方向不一致 → M=0，无信号

**评分**：
```
|mom_5m| > 0.3%  → 35分（强动量）
|mom_5m| > 0.2%  → 25分（中动量）
|mom_5m| > 0.1%  → 15分（弱动量）
|mom_5m| <= 0.1% →  0分（无动量）

15分钟方向一致 → +15分（一致性加成）
最高50分
```

**跨日行为**：开盘前几根bar的lookback会跨到昨天（如09:45时lookback到昨天14:20），隔夜gap被计入动量。这不是bug——**跳空gap的方向信号对动量品种（IM/IC）有价值**。

### V分：波动率（0-30分）

**计算**：
```
ATR_short = 最近5根bar的平均真实波幅
ATR_long  = 最近40根bar的平均真实波幅
ratio = ATR_short / ATR_long
```

**评分**：
```
ratio < 0.7  → 30分（短期波动率远低于长期 = 蓄势待发）
ratio < 0.9  → 25分
ratio < 1.1  → 15分
ratio < 1.5  →  5分
ratio >= 1.5 →  0分（波动已经很大 = 可能快结束）
```

**注意**：V分是**逆向指标**——短期波动率越低（相对长期），信号越强。这意味着在"平静后的突破"时信号最强。

### Q分：成交量（0-20分）

**计算**：
```
ratio = volume[-1] / mean(volume[-20:])    # 当前bar量 / 20根均量
```

**评分**：
```
ratio > 1.5 → 20分（放量确认）
ratio > 0.5 → 10分（正常量）
ratio <= 0.5 →  0分（缩量）
```

### B分：布林带突破加分（0-20分）

仅在M分>0且有方向时触发。检测中轨突破：

**做空信号**：前5根bar在中轨上方 + 当根跌破中轨 → 10分基础分
**做多信号**：前5根bar在中轨下方 + 当根涨破中轨 → 10分基础分

**额外加分**：
```
放量突破  → +2分（当根量 > 前5根均量）
窄带突破  → +3分（当前带宽 < 历史均宽×0.8）
15分钟同向确认 → +3~5分
最高20分
```

---

## 4. 乘数链详解

### daily_mult：日线方向乘数（0.8-1.2）

基于最近5个交易日的收盘价动量判断日线方向：
```
daily_mom = (close[-1] - close[-6]) / close[-6]   # 5日收益率

如果 |daily_mom| < 0.2% → dm = 1.0（中性）
如果信号方向和日线一致 → dm = dm_trend（默认1.2）
如果信号方向和日线相反 → dm = dm_contrarian（默认0.8）
```

**Per-symbol配置**（SYMBOL_PROFILES）：
| 品种 | dm_trend | dm_contrarian | 原因 |
|------|----------|---------------|------|
| IM | 1.2 | 0.8 | 动量型，顺势好 |
| IC | 1.2 | 0.8 | 动量型 |
| **IF** | **1.0** | **1.0** | **均值回归型，逆势59%WR是利润主力** |
| IH | 1.2 | 0.8 | 改善marginal，暂不改 |

**d_override覆盖**：Morning Briefing可以按方向覆盖daily_mult（如LONG=0.7, SHORT=1.1）。

### intraday_filter：日内涨跌幅过滤（0.3-1.0）

用当前价格 vs 昨日收盘价计算日内涨跌幅，对极端涨跌做方向感知惩罚。

**高波动模式**（is_high_vol=True）：
```
涨>3%:   做多×0.8  做空×0.3（强烈惩罚逆势做空）
涨2-3%:  做多×0.9  做空×0.5
涨1-2%:  做多×1.0  做空×0.7（Z<-2时做多不罚）
<1%:     1.0

跌>3%:   做空×0.8  做多×0.3（Z<-2时做多×0.7）
跌2-3%:  做空×0.9  做多×0.5
跌1-2%:  做空×1.0  做多×0.7
```

**低波动模式**：阈值更低（1.5%），惩罚更温和。

### time_weight：时段权重（0.6-1.1）

| 时段(BJ) | 权重 | 说明 |
|----------|------|------|
| 09:35-10:30 | 1.0 | 开盘常规 |
| 10:30-11:30 | 1.1 | 上午最强时段 |
| 13:00-13:30 | **1.0** | 午后开盘（2026-04-02从0.8上调，+76pt） |
| 13:30-14:30 | 1.0 | 下午常规 |
| 14:30-14:50 | 0.6 | 尾盘衰减 |

每品种可在SYMBOL_PROFILES中自定义session_multiplier。

### sentiment_mult：期权情绪乘数（0.5-1.5）

五个调节因子叠加：
1. **IV变动**：IV急升>2pp → 做多-0.15 / IV急降>2pp → 做空-0.15
2. **Skew(RR)**：看跌偏向加大 → 做多-0.10
3. **PCR逆向**：极度悲观(>1.5) → 做多+0.15 / 极度乐观(<0.7) → 做空+0.15
4. **期限结构倒挂** → -0.05
5. **VRP<0** → 做空减分

---

## 5. 过滤层

### Z-Score硬过滤（仅高波动区间）

```
Z < -3.0:  做空→阻断(0分)    做多→+50%
Z < -2.5:  做空→阻断(0分)    做多→+30%
Z < -2.0:  做空×0.3          做多→+20%
Z < -1.5:  做空×0.6          做多→+10%

Z > +3.0:  做多→阻断(0分)    做空→+50%
... 对称 ...
```

**设计思路**：均值回归——极端超卖时阻断做空、鼓励做多。低波动区间不启用。

### RSI反转加分（高波动 + |Z|>2）

```
Z < -2 + LONG + RSI从低位回升 → +10分
Z > +2 + SHORT + RSI从高位回落 → +10分
```

---

## 6. 平仓系统

### 价格双层约定（2026-04-02修复，重要！）

check_exit 在实盘（monitor）中使用双价格层：
- **`current_price`（期货last_price）**：用于止损/跟踪止盈/PnL/盈亏判断（P1b/P2/P3/P7）
- **`spot_price`（现货close）→ `boll_price`**：用于Bollinger zone判断（P4/P5/P6）
- **`highest_since`/`lowest_since`**：用期货价格追踪（与entry_price同源）

回测中 `spot_price=0`（默认），fallback到 `current_price`，全用现货，不受影响。

**历史教训**：混用现货和期货价格导致IM贴水3.5%被误判为亏损→假STOP_LOSS。

### 平仓优先级（P1-P7）

| 优先级 | 条件 | 触发 | 价格层 |
|--------|------|------|---------|
| **P0** | bar的high/low穿过止损位 | `STOP_LOSS` | 期货（backtest用现货） |
| **P1** | 14:45 BJ | `EOD_CLOSE` | — |
| **P1b** | 亏损 > 0.5% | `STOP_LOSS` | 期货 |
| **P2** | 11:25前 + 亏损 | `LUNCH_CLOSE` | 期货 |
| **P2b** | 11:25前 + 盈利 + trailing 0.3% | `LUNCH_TRAIL` | 期货 |
| **P3** | 动态trailing stop | `TRAILING_STOP` | 期货 |
| **P4** | 5m+15m都在布林带极端 | `TREND_COMPLETE` | 现货Bollinger |
| **P5** | 持仓>=20min + 3根窄幅 + 15m极端 | `MOMENTUM_EXHAUSTED` | 现货Bollinger |
| **P6** | 2根破中轨 + 15m确认 | `MID_BREAK` | 现货Bollinger |
| **P7** | 60min无盈利 | `TIME_STOP` | 期货 |

### P0 止损（backtest特有的bar内止损）

在check_exit之前，用bar的high/low判断是否穿过止损位：
```python
if direction == "LONG" and bar_low <= stop_price:
    exit at stop_price (not bar close)
if direction == "SHORT" and bar_high >= stop_price:
    exit at stop_price
```
止损优先级最高，触发后跳过所有其他exit检查。

### P3 动态trailing stop

trailing stop宽度随持仓时间扩大：
```
<15min:  0.5%（刚开仓，保护本金）
<30min:  0.6%
<60min:  0.8%
>=60min: 1.0%（长趋势，给空间）

15分钟趋势确认 + 盈利>0.5% → 额外+0.2%
```

Per-symbol可通过`trailing_stop_enabled`和`trailing_stop_scale`控制。

**IC trailing_stop_scale=2.0（2026-04-04）**：IC的trailing stop宽度是默认值的2倍（IC震荡性强，小波动容易触发后快速反弹），+186pt。

### P5 MOMENTUM_EXHAUSTED（2026-04-04更新参数）

ME触发条件：`最近3根bar的total_range / boll_width < narrow_ratio`（加上15m极端确认）

**narrow_ratio 0.20→0.10（2026-04-04）**：只有更极端的窄幅K线才算动量耗尽，IM+IC +393pt。旧的0.20触发过于频繁（尤其在震荡市日内的正常整理期也触发）。

### P6 MID_BREAK（2026-04-04更新参数）

**bars 2→3（2026-04-04）**：需要连续3根K线破中轨（而非2根），减少假突破，IM+IC +76pt。

---

## 7. 品种配置

### SIGNAL_ROUTING

所有品种统一使用v2（2026-04-01验证）。v3在干净数据下对所有品种都不优于v2。

### 品种参数差异

| 参数 | IM | IC | IF | IH |
|------|-----|-----|-----|-----|
| 阈值 | 60 | **65** | 60 | 60 |
| dm_trend/contra | 1.2/0.8 | 1.2/0.8 | **1.0/1.0** | 1.2/0.8 |
| 动量lookback | 12 (60min) | 12 | **12**（修复自18） | **12**（修复自18） |
| 日线lookback | 5天 | 5天 | 20天 | 20天 |
| 反转过滤 | 否 | 否 | 是(2%) | 是(1%) |
| trailing_stop_scale | 1.0x | **2.0x** | 1.0x | 1.0x |
| 实盘状态 | 实盘 | 实盘 | 观察 | 放弃 |

注：IF/IH的动量lookback原配置为18，但V2代码存在硬编码bug（`MOM_5M_LOOKBACK=12`覆盖所有品种），配置值从未生效。2026-04-04修复读取逻辑后，grid search确认12对四品种均最优，IF/IH配置也改为12。

### 品种特征

- **IM/IC**（中小盘动量型）：波动大、利润厚、BE>2.5pt、适合趋势跟随
- **IF**（大盘蓝筹均值回归型）：逆势59%WR是利润主力、dm改中性后+75%
- **IH**（上证50）：波动最小、avg|PnL|=5.7pt、BE<1pt无法覆盖成本

---

## 8. 开仓时间窗口

```
09:45 - 11:20  上午开仓窗口     t=1.0 / 10:30后t=1.1
11:20 - 13:05  午休禁止开仓
13:05 - 14:30  下午开仓窗口     t=1.0（2026-04-02从0.8上调，+76pt/+7%）
14:30 -        禁止开仓         t=0.7→0.6（尾盘）

冷却期：平仓后15分钟内不允许同方向再开仓（统一，不分级）

实盘品种限制：IntradayConfig.tradeable = {"IM", "IC"}
  - 只有tradeable品种通过strategy.on_bar开仓/占position_mgr槽位
  - IF/IH面板显示评分但不触发信号/shadow/signal_file
```

### 时段权重说明

`session_multiplier` 定义在 `SYMBOL_PROFILES[sym]` 中（不是模块级 `_TIME_WEIGHTS`）。
score_all 通过 `_get_session_weight()` 读取 `_SESSION_UTC` + `session_multiplier`。

**注意**：`_TIME_WEIGHTS` 不被 score_all 使用（历史遗留），修改它无效。

---

## 9. 已知的"有益偏差"

以下看似"bug"的行为经验证后确认是有益的，**不应修复**：

1. **跨日GAP在M/V/Q中的影响**：隔夜gap让ATR偏高→V分偏低→抑制开盘噪音信号。验证修复后IM -53%。

2. **Monitor的forming bar Q=0**：TQ最后一根bar是未完成bar（volume≈0），在monitor中充当天然开盘过滤器。与backtest的`iloc[:-1]`行为一致。

3. **volume跨日均值**：集合竞价放量拉高20根均值→后续bar的Q偏低→延迟入场。验证session-aware均量后反而更差。

---

## 10. Baseline（2026-04-04更新）

回测日期：20260204-20260402（34天），所有lookahead bias已修复，含全部04-02~04-04参数优化和bug修复。

| 品种 | 总PnL | WR | 状态 |
|------|-------|-----|------|
| IM | +983pt | ~60% | 实盘 |
| IC | +1048pt | ~57% | 实盘 |
| IF | ~+330pt | ~59% | 观察 |
| IH | ~+130pt | ~53% | 放弃 |

IM+IC合计 +2031pt（vs 旧baseline +1207pt，+824pt, +68%总改善）。

**各项改善分解（IM+IC合计）**：
| 优化项 | Delta |
|-------|-------|
| 午后session_weight 0.8→1.0 | +76pt |
| 动量lookback硬编码修复（IF/IH 18→12） | 含在总数内 |
| ME narrow_range ratio 0.20→0.10 | +393pt |
| MID_BREAK bars 2→3 | +76pt |
| IC trailing_stop_scale=2.0x | +186pt |
| 15分钟重采样 label='right'→'left' | ~+93pt（关键修复） |
| 其他修复（现货期货价格层等） | 余量 |

---

## 11. 已验证不实施的参数变更

所有研究均用30+天回测数据验证。以下方向经测试后确认不实施。

注：以下各小节的PnL数字是研究时的baseline（早于04-04修复），与当前§10的baseline不直接可比。Delta（相对变化量）仍然有效。

### 分级冷却（2026-04-02验证）

当前：统一15分钟同方向冷却。测试按exit_reason分级。

| 方案 | IM+IC PnL | vs baseline | 验证方式 |
|------|-----------|-------------|---------|
| 统一15min（当前） | +1131 | — | — |
| ME=20,SL=25,TC=15 | +995 | **-136** | 独立回测循环，cooldown_map按reason查表 ✅ |
| ME=25,SL=30,TC=10 | +966 | **-165** | 同上 ✅ |
| 全部20min | +796 | **-335** | 同上 ✅ |

结论：延长冷却全部亏钱。STOP_LOSS后延长冷却代价最高——被止损的方向有时是最好的再入场机会。

### ADX趋势强度乘数（2026-04-02验证）

| 方案 | IM+IC PnL | vs baseline | 验证方式 |
|------|-----------|-------------|---------|
| 无ADX（当前） | +1079 | — | — |
| ADX<15×0.85, >25×1.1 | +971 | **-108** | 事后模拟，乘在最终score上 ⚠ |
| MACD顺势×1.05, 逆势×0.9 | +993 | **-86** | 同上 ⚠ |
| ADX+MACD组合 | +918 | **-161** | 同上 ⚠ |

⚠ 注意：ADX/MACD研究的patch方式是事后模拟（乘在已加权的最终score上），与集成进score_all不完全等价。但结论方向（整体亏钱）应可靠——核心原因是低ADX区间反而有高WR好交易。

ADX分组发现（供未来参考）：
- ADX>35 是IM的"死亡区间"（25笔44%WR，净亏-42pt）
- ADX<15 反而最赚钱（8笔75%WR，+114pt）
- 待100+笔ADX>35样本后重新评估硬过滤

### 技术指标冗余性（2026-04-02验证）

| 指标 | vs M相关 | vs V相关 | vs Q相关 | 结论 |
|------|---------|---------|---------|------|
| ADX | r=0.31 | 低 | 低 | 有限独立性，乘数效果为负 |
| MACD | <0.1 | <0.1 | <0.1 | 高独立性但无预测力 |
| CCI | <0.1 | 低 | 低 | 和WR高度重复(r=0.90) |
| WR | <0.1 | 低 | 低 | 和CCI高度重复 |

### 午后时间权重（2026-04-02验证并实施）

| t_afternoon | IM+IC PnL | vs baseline | 状态 |
|---|---|---|---|
| 0.8（旧） | +1131 | — | 已废弃 |
| 0.9 | +1163 | +32 | — |
| **1.0** | **+1207** | **+76 (+7%)** | **已实施** |

⚠ 重要教训：首次研究monkey-patch了`_TIME_WEIGHTS`，但score_all实际用`SYMBOL_PROFILES[sym]["session_multiplier"]`。Patch了错误的变量导致假阳性结论（"三个值完全一样"）。正确patch后发现t=1.0显著优于0.8。

**修改研究脚本时必须验证修改是否对目标函数生效。**

### 平仓参数敏感性分析（2026-04-04新增）

以34天回测为基础，IM+IC合计。baseline = +1207pt（含t=1.0修正）。

**ME narrow_range ratio**：

| narrow_ratio | IM+IC Delta | 说明 |
|---|---|---|
| 0.30 | -xxx | 触发太频繁 |
| 0.20（旧） | 0 | baseline |
| **0.10** | **+393pt** | **已实施** |
| 0.05 | 略低于0.10 | 极少触发，接近不触发 |

**MID_BREAK bars**：

| bars | IM+IC Delta | 说明 |
|---|---|---|
| 2（旧） | 0 | baseline |
| **3** | **+76pt** | **已实施** |
| 4 | +52pt | 改善减少 |

**IC trailing_stop_scale**：

| scale | IC Delta | 说明 |
|---|---|---|
| 1.0x（旧） | 0 | baseline |
| 1.5x | +90pt | — |
| **2.0x** | **+186pt** | **已实施** |
| 2.5x | +110pt | 回落 |

### ME/TC Score确认退出（2026-04-03研究，不实施）

当ME/TC触发时，检查prev_score。若score仍高于阈值，阻断退出（趋势宏观结构未变）。

| 方案 | IM+IC PnL | vs baseline | 状态 |
|------|-----------|-------------|------|
| ME+TC score确认 | +1512 | +130(+9.4%) | ❌ 不实施 |
| 只ME确认 | +1467 | +85 | 未单独验证 |

**稳健性不通过原因**：前半段-58pt（后半段+187pt）、阈值系数0.8变负、最大单日贡献53%、仅14/34天改善。数据不足，待100天数据后重新验证。

脚本：`score_confirmed_exit_research.py`, `score_exit_robustness.py`

### 8组核心参数Sensitivity分析（2026-04-02，parameter_sensitivity.py）

一次只变一个参数组，其他保持当前值。IM+IC 34天回测。

**稳定（不动）的参数**：

| 参数 | 当前值 | 测试范围 | 结论 |
|------|-------|---------|------|
| 止损幅度 | 0.5% | 0.3-0.8% | 0.4%略好+11pt但在平台区 |
| Trailing stop | 阶梯0.5-1.0% | ×0.8~×1.2缩放 | 当前就是IM最优 |
| 日线乘数dm | 1.2/0.8 | 1.0/1.0~1.5/0.5 | 当前最优，过激恶化 |
| IM intraday_filter | 1/2/3% | 0.8/1.5/2.5~1.5/3/4% | 当前是IM最优 |
| 动量断点 | 0.30% | 0.20-0.40% | 当前最优 |
| 成交量阈值 | surge=1.5 | 1.0-2.5 | IC略好surge=2.0(+48pt)但幅度小 |

**有潜在优化空间但需验证的参数**：

| 参数 | 品种 | 当前→候选 | Delta | 稳健性 | 状态 |
|------|------|----------|-------|--------|------|
| IC intraday_filter | IC | 1/2/3%→1.2/2.5/3.5% | +118pt(+21%) | 20.6%波动（谨慎区间） | 待shadow验证 |
| IC trailing stop | IC | 当前→加宽 | +50pt(+9%) | 未做详细验证 | 低优先级 |

**已验证不改的参数**：

| 参数 | 品种 | 候选 | Delta | 不改原因 |
|------|------|------|-------|---------|
| IM lookback | IM | 12→10 | +65pt | **邻域不平滑**（lb=9跳到+810），只赢2/5周，前半段不如12 |
| ME最小持仓 | IM | 20→45min | +155pt→实测+610~707 | 波动11.2%稳健但45min本身不优于20min |
| ME最小持仓 | IC | 20→10min | +89pt→实测0~-38pt | IC对ME不敏感，改了没区别 |

**动量lookback硬编码修复**：

V2的`_score_momentum`原来用硬编码`MOM_5M_LOOKBACK=12`，SYMBOL_PROFILES的值从未生效。已修复为从prof读取。Grid search确认lb=12对四品种都是最优或接近最优，IF/IH配置从18改回12。

### Z-Score极值反转bonus（2026-04-01验证）

| 方案 | IM | IC | IF/IH | 验证方式 |
|------|----|----|-------|---------|
| 无bonus（当前） | — | — | — | — |
| +10~20分bonus | -12pt | +24pt | ≈0 | 直接加到total上（z-filter后） ✅ |

结论：现有Z-Score+RSI层已覆盖，额外加分边际效果<1%。

---

## 12. 研究方法论备忘

### Patch正确性检查清单

修改参数做回测时，必须验证：

1. **找到实际被score_all调用的变量**——不要假设变量名匹配就是同一个
   - 反例：`_TIME_WEIGHTS` vs `session_multiplier`
2. **单日验证**——先用一个已知受影响的日期跑，确认score变了
3. **事后模拟 vs 真实集成**——乘在最终score上 ≠ 加进raw_total，因为乘数链会放大/缩小
4. **Reload模块**——Python缓存import，改了模块级变量后需要`importlib.reload()`

### 回测研究脚本索引

| 脚本 | 研究内容 | 结论 | Patch方式验证 |
|------|---------|------|-------------|
| parameter_sensitivity.py | 8组核心参数grid search | IC filter有空间 | monkey-patch ✅ |
| robustness_check.py | ME持仓+IC filter稳健性 | ME不改，IC待shadow | monkey-patch ✅ |
| afternoon_cooldown_research.py | 午后t权重 + 分级冷却 | 冷却有效，**t权重patch错误** | 冷却✅ t❌ |
| adx_research.py | ADX独立信息量 | 不加 | N/A（纯统计） |
| adx_macd_backtest.py | ADX/MACD乘数回测 | 不加 | 事后模拟⚠ |
| hurst_research.py | Hurst regime | 象限监控（不加评分） | N/A（纯统计） |
| overnight_strategy_research.py | 隔夜策略 | 不做 | N/A（纯统计） |
| vwap_research.py | VWAP偏离度 | 和M冗余r=0.67 | N/A（纯统计） |
| body_ratio_research.py | K线实体占比 | IC有害不加 | 独立回测 ✅ |
| cross_rank_research.py | 跨品种rank | -50%亏钱 | 独立回测 ✅ |
| style_spread_research.py | IM-IH风格差过滤 | 硬过滤亏钱 | 独立回测 ✅ |
| rr_backfill_research.py | 历史RR回算 | 无线性预测力 | N/A（纯统计） |
| score_confirmed_exit_research.py | ME/TC score确认退出 | 不稳健不实施 | 独立回测 ✅ |
| score_exit_robustness.py | 上项的稳健性验证 | 前半段-58pt | 4项测试 ✅ |
| exit_param_sensitivity.py | 退出参数灵敏度 | ME ratio/MID bars/IC trailing | 独立回测 ✅ |

---

## 13. Shadow验证跟踪（待实盘数据确认后决定）

以下参数变更已通过回测初步验证，但稳健性处于谨慎区间或样本不足，需要在实盘shadow中观察2-4周后决定是否实施。

### IC intraday_filter阈值放宽（优先级：高）

| 项目 | 详情 |
|------|------|
| 当前值 | 1.0%/2.0%/3.0%（三档，与IM相同） |
| 候选值 | **1.2%/2.5%/3.5%** |
| 回测改善 | +118pt (+21%)，IC从+571→+688 |
| 稳健性 | 20.6%波动（谨慎区间15-30%），1.2~1.3/2.5~2.7/3.5~3.7%形成平台 |
| 风险 | 1.5/3.0/4.0%回落到持平，边界敏感 |
| 验证方法 | Monitor shadow记录的entry_f值分布，看1.0-1.2%区间信号的实盘WR |
| 决策时间 | 2026-04-16（2周后） |

### IM-IH Style Spread中性区间过滤（优先级：低）

| 项目 | 详情 |
|------|------|
| 发现 | 中性区间(-0.09%~+0.04%)：73%WR，均+13.8pt |
| 问题 | 硬过滤全亏钱（-28~-248pt），但中性区间确实好 |
| 候选方向 | 只过滤IM大负spread（IM跑输IH>0.17%时21笔WR=43%，-114pt） |
| 样本限制 | 21笔太少 |
| 验证方法 | signal_log已自动记录style_spread，积累100+笔后统计 |
| 决策时间 | 2026-07（3个月后） |

### ADX>35硬过滤（优先级：低）

| 项目 | 详情 |
|------|------|
| 发现 | IM ADX>35：25笔44%WR，净亏-42pt |
| 候选方向 | ADX>35时阻断开仓 |
| 样本限制 | 25笔不够确认pattern |
| 验证方法 | signal_log已自动记录adx_14，积累100+笔ADX>35后统计 |
| 决策时间 | 2026-07（3个月后） |

### Hurst动态权重（优先级：低）

| 项目 | 详情 |
|------|------|
| 当前状态 | 已加入象限监控，不参与评分 |
| 候选方向 | 低Hurst(<0.45)时降低阈值或提高逆势dm |
| 前置条件 | 至少10天H<0.45数据（当前0天） |
| 验证方法 | daily_model_output已记录hurst_60d |
| 决策时间 | 视市场环境 |
