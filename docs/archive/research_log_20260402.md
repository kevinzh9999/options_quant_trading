# 研究日志 2026-04-02：现货期货价格混用修复 + 信号流审计 + 参数优化

## 背景

实盘运行中发现多个critical bug：IM/IC做空一开仓就被假止损（hold_minutes=0），monitor重启后信号不发出，非实盘品种占用开仓槽位等。全天紧急修复+审计+参数优化。

---

## 一、Critical Bug 修复

### 1. 现货/期货价格混用（最严重）

**发现**：shadow entry_price=期货bid1（如7443），check_exit的current_price=现货close（如7731）。IM贴水3.5%，SHORT持仓一开仓就"亏损3.5%"，瞬间假触发STOP_LOSS。

**影响**：所有做空信号开仓即止损（hold_minutes=0），同时Bollinger zone判断完全错误（期货价 vs 现货布林带）。

**修复**：check_exit新增`spot_price`参数，实现双价格层：
- 止损/跟踪止盈/PnL → `current_price`（期货last_price，与entry_price同源）
- Bollinger zone/MID_BREAK → `boll_price`（现货close，与bar_5m同源）
- `highest_since`/`lowest_since` → 期货价格
- 回测不传spot_price（默认0→fallback到current_price），不受影响

### 2. CLOSE信号timestamp UTC vs 北京时间

**发现**：OPEN信号用`datetime.now()`（北京时间），CLOSE信号用`datetime.utcnow()`（UTC）。executor的`_restore_positions_from_log`按datetime排序，LOCK(01:50 UTC)排在OPEN(09:50 BJ)前面→先减后加→算出错误净持仓。

**修复**：CLOSE信号改用`datetime.now()`。executor的restore改为两遍扫描（先OPEN再CLOSE/LOCK），不依赖排序。

### 3. Executor锁仓检查无TQ对账

**发现**：`_check_locked_positions`查order_log中的LOCK记录，但不验证TQ实际是否有锁仓。每次启动都重复提示已处理的LOCK。

**修复**：锁仓检查移到TQ连接之后，查TQ实际持仓验证。新增`lock_resolved`列，已处理的不再提示。

### 4. 非实盘品种占用position_mgr槽位

**发现**：strategy.on_bar对所有4个品种生成信号。IF score=78先占了position_mgr的1个槽位（max_total_lots=2），IM占第2个，IC score=82（最高分实盘品种）进不来。

**修复**：IntradayConfig新增`tradeable={"IM","IC"}`，strategy.on_bar跳过非tradeable品种。

### 5. prompted_bars跨session恢复阻止新信号

**发现**：重启时从shadow_state.json恢复全部prompted_bars。已平仓品种的旧bar时间戳仍在集合中→新session的同品种高分信号被静默丢弃。

**修复**：只恢复活跃shadow持仓品种的去重记录。

### 6. position_mgr孤儿占位

**发现**：_on_new_bar被品种A触发时处理所有品种。strategy.on_bar给品种B创建position_mgr占位，但prompted_bars用B的旧bar时间戳判定重复→跳过信号。占位永不清理，阻止B后续所有信号。

**修复**：prompted_bars阻止时调用`remove_by_symbol`撤销占位。

---

## 二、全面审计（10场景验证）

修复后对10个关键场景做了完整代码路径审计，全部通过：
1. 正常开平仓流程 ✅
2. 重启恢复活跃shadow ✅
3. 重启后shadow已平仓 ✅
4. 多品种同bar信号 ✅
5. Executor拒绝信号 ✅
6. check_exit价格一致性（P1-P7） ✅
7. Executor重启+LOCK ✅
8. _on_new_bar多次触发 ✅
9. Tradeable过滤 ✅
10. risk_mgr跨重启状态 ✅

---

## 三、技术指标研究

### ADX趋势强度（30天回测，不实施）

| ADX区间 | IM笔数 | IM WR | IM总PnL |
|---------|--------|-------|---------|
| <15 | 8 | 75% | +114 |
| 15-25 | 41 | 59% | +251 |
| 25-35 | 21 | 71% | +291 |
| >35 | 25 | **44%** | **-42** |

ADX>35是"死亡区间"（追入太晚），但ADX乘数方案（<15×0.85, >25×1.1）反而亏-108pt——低ADX的8笔75%WR是好交易，被打折了。

### MACD histogram确认（30天回测，不实施）

IM中MACD>0做空（逆势）只有6笔17%WR，样本太小。IC中MACD方向无预测力（逆势57%WR）。MACD乘数方案-86pt。

### 冗余性矩阵

| 指标 | vs M | vs V | vs Q | 结论 |
|------|------|------|------|------|
| ADX | r=0.31 | 低 | 低 | 有限独立性，乘数效果为负 |
| MACD | <0.1 | <0.1 | <0.1 | 高独立性但无预测力 |
| CCI | <0.1 | 低 | 低 | 和WR高度重复(r=0.90) |
| WR | <0.1 | 低 | 低 | 和CCI高度重复 |

**结论**：四个技术指标均不加入信号系统。ADX>35硬过滤和MACD顺逆势待3个月数据积累后重新验证。

---

## 四、Hurst指数Regime Detection

### 计算方法
R/S分析法，60天滚动窗口，日线close。H>0.55趋势期，H<0.45震荡期。

### 核心发现
- 四品种长期Hurst均值≈0.67，A股日线天然偏趋势
- 自相关lag1>0.85，状态切换很慢
- 当前IM Hurst=0.78（P97，极强趋势期）
- Hurst对波动率有预测力（p<0.001），对方向无预测力

### 实施
加入象限判断（不参与信号评分）：
- A1: 高IV+VRP>0+震荡 → 卖方最佳
- A2: 高IV+VRP>0+趋势 → 顺势卖方可以
- B1: 高IV+VRP<0+震荡 → 等VRP转正
- B2: 高IV+VRP<0+趋势 → 最危险，禁止卖方

---

## 五、参数优化

### 午后时间权重（13:00-13:30 BJ）

**关键发现**：之前研究脚本monkey-patch了`_TIME_WEIGHTS`，但score_all实际用的是`SYMBOL_PROFILES[sym]["session_multiplier"]`。patch了错误的变量导致假阳性结论（"三个值完全一样"）。

正确patch后的结果：

| t_afternoon | IM | IC | 合计 | vs baseline |
|---|---|---|---|---|
| 0.8（当前） | +626 | +505 | +1131 | — |
| 0.9 | +598 | +565 | +1163 | +32 |
| **1.0** | **+636** | **+571** | **+1207** | **+76（+7%）** |

**结论**：t=1.0最优，+76pt（+7%）。IC受益最大（+66pt）。午后开盘30分钟的降权不合理——该时段信号质量和其他时段一样好。

### 分级冷却（同方向再入场冷却时间）

| 方案 | 合计PnL | vs baseline |
|---|---|---|
| 统一15min（当前） | +1131 | — |
| ME=20,SL=25 | +995 | -136 |
| ME=25,SL=30 | +966 | -165 |
| 全部20min | +796 | -335 |

**结论**：延长冷却全部亏钱。STOP_LOSS后延长冷却代价最高。保持统一15min。

---

## 六、隔夜持仓策略研究

10年日线数据分析结论：
- A股隔夜收益结构性为负（IM -0.09%/天，胜率40.5%）
- 尾盘方向对隔夜收益无统计显著预测力（p>0.3）
- 大跌后隔夜不是均值回归而是动量延续
- 唯一正期望窗口：尾盘跌→14:55做多→次日09:45平（WR=71%，仅52笔样本）
- 结论：维持"14:30后不开仓"规则

---

## 七、待决定

- [ ] 午后t从0.8改到1.0（+76pt/+7%）——待确认
- [ ] ADX>35硬过滤——待3个月数据
- [ ] Hurst加入信号评分——待低Hurst期数据

---

## 相关文件

| 文件 | 内容 |
|------|------|
| scripts/overnight_strategy_research.py | 隔夜策略分析 |
| scripts/adx_research.py | ADX独立信息量分析 |
| scripts/adx_macd_backtest.py | ADX/MACD回测验证 |
| scripts/hurst_research.py | Hurst指数研究 |
| scripts/afternoon_cooldown_research.py | 午后权重+冷却分析（⚠ patch了错误变量） |
| docs/lookahead_audit.md | F19修复记录 |
