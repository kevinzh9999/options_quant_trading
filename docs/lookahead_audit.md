# 未来数据泄漏审计报告

日期：2026-04-01
范围：backtest_signals_day.py + monitor.py + A_share_momentum_signal_v2.py

---

## A. K线数据（backtest_signals_day.py）

### [A1] bar_5m 构建 — ✅ 无泄漏
- 代码位置：backtest_signals_day.py 第220-226行
- 当前行为：`bar_5m_signal = bar_5m.iloc[:-1]`，排除当前forming bar
- 信号评分用 `bar_5m_signal`，执行价格用 `bar_5m.iloc[-1]["close"]`
- 与monitor一致：monitor也用 `k5.iloc[:-1]` 排除TQ的forming bar

### [A2] bar_15m 构建 — ✅ 无泄漏
- 代码位置：backtest_signals_day.py 第241行
- 当前行为：`bar_15m = _build_15m_from_5m(bar_5m_signal)`，从信号bars重采样
- 15m bars来源于已排除当前bar的5m数据，无泄漏

### [A3] score_all 内部指标计算 — ✅ 无泄漏（基于传入的bar_5m_signal）
score_all 接收 `bar_5m_signal`（已排除当前bar），内部所有计算基于此数据：

- **动量（M）**：`close_5m[-1] - close_5m[-N-1]`（A_share_momentum_signal_v2.py 第987行）
  - close_5m 来自 bar_5m_signal，[-1] = 上一根完成bar ✅
- **布林带**：`closes.rolling(20).mean().iloc[-1]`（第190行）
  - 基于传入的 bar_5m_signal 的close序列 ✅
- **ATR**：`_atr(highs, lows, closes, period)`（第761行）
  - 用 `range(-period, 0)` 的负索引，全部在传入数据范围内 ✅
- **RSI**：`_calc_rsi(close_5m)`（第511行）
  - 用 `np.diff(closes)` 的最后 period 个差值 ✅
- **成交量（Q）**：`volume_5m[-1] / mean(volume_5m[-20:])`（第1030行）
  - volume_5m 来自 bar_5m_signal，[-1] = 上一根完成bar的volume ✅
- **布林带突破**：`_score_boll_breakout`（第195行）
  - 所有 `close_5m[i]` 索引在传入数据范围内 ✅

### [A3-NOTE] score_all 的时间提取 — ⚠ 存在差异（影响极小）
- 代码位置：A_share_momentum_signal_v2.py 第785-797行 `_get_utc_time(bar_5m)`
- 当前行为：提取 `bar_5m.index[-1]` 的时间，用于 NO_TRADE_BEFORE/AFTER 检查和 session_weight
- 因为传入的是 `bar_5m_signal`（排除当前bar），提取的时间是上一根bar的时间
- 例如：处理09:45 bar时，提取的utc_time是09:40 bar的时间 = 01:40 UTC
- 影响：
  - NO_TRADE_BEFORE = 01:35 UTC (09:35 BJ)：09:40仍通过 ✅
  - session_weight：时段权重可能因为5分钟偏移取到不同时段的权重，**但这与monitor行为一致**（monitor看到的也是上一根bar的时间）
- 结论：行为与monitor一致，不算泄漏

---

## B. 日线数据

### [B4] daily_df 构建 — ✅ 无泄漏
- 代码位置：backtest_signals_day.py 第192-195行
- 当前行为：`daily_all[daily_all["trade_date"] < td].tail(30)`
- 严格排除当天，只用T-1及之前的日线

### [B5] daily_mult（5日动量） — ✅ 无泄漏
- 代码位置：A_share_momentum_signal_v2.py 第1041-1048行
- 当前行为：`closes[-1] - closes[-MOM_DAILY_LOOKBACK-1]`
- daily_bar 已在backtest中截断为 `trade_date < td`，所以 closes[-1] = T-1日收盘价 ✅

### [B6] Z-Score（EMA20/STD20） — ✅ 无泄漏
- 代码位置：backtest_signals_day.py 第111-123行
- 当前行为：`spot_all[spot_all["trade_date"] < target_date].tail(30)`
- Z-Score用 `signal_price`（上一根完成bar的close）计算：`(signal_price - ema20) / std20`（第238行）
- 严格排除当天 ✅

### [B7] GARCH regime — ✅ 无泄漏
- 代码位置：backtest_signals_day.py 第127-134行
- 当前行为：`trade_date < '{td}'` 过滤
- 读取的是T-1日的 garch_forecast_vol ✅

### [B7-NOTE] GARCH 硬编码 underlying='IM' — ⚠ 功能问题（非泄漏）
- 所有品种（IF/IH/IC）都用IM的GARCH判断高低波动
- 这不是数据泄漏，但是逻辑不精确。当前可接受因为品种间波动率高度相关

---

## C. 情绪/波动率数据

### [C8] sentiment（ATM IV / VRP） — ✅ 无泄漏
- 代码位置：backtest_signals_day.py 第142-157行
- 当前行为：`trade_date < '{td}'` 过滤，取最近2条记录
- 用的是T-1和T-2日的数据 ✅

### [C8-NOTE] sentiment 硬编码 underlying='IM' — ⚠ 功能问题（非泄漏）
- 同B7-NOTE，所有品种都用IM的期权情绪数据

### [C9] morning_briefing d_override — ✅ 无泄漏
- 代码位置：backtest_signals_day.py 第166-178行
- 当前行为：`trade_date = '{td}'` 读取当天的briefing
- Morning briefing在盘前产生，用当天数据是正确的 ✅

### [C10] RR（Risk Reversal） — ✅ 无泄漏
- 代码位置：backtest_signals_day.py 第153-154行
- 当前行为：和sentiment一起从 `daily_model_output WHERE trade_date < '{td}'` 读取
- 用的是T-1的RR ✅

---

## D. 开仓/平仓价格

### [D11] 开仓价格 — ✅ 无泄漏
- 代码位置：backtest_signals_day.py 第356行
- 当前行为：`entry_p = price + slippage`，其中 `price = bar_5m.iloc[-1]["close"]`
- 信号在bar_T-1上产生，开仓用bar_T的close（= 下一根完成bar的收盘价）
- 这模拟了"信号产生→下一根bar执行"的时序 ✅

### [D12] 平仓价格 — ✅ 无泄漏
- 代码位置：backtest_signals_day.py 第280行
- 当前行为：`exit_p = price - slippage`，同上用bar_T的close
- exit信号在bar_T-1上产生，退出在bar_T执行 ✅

### [D13] 止损判断 — ❌ 存在泄漏
- 代码位置：A_share_momentum_signal_v2.py 第332-339行（check_exit内）
- 当前行为：止损用 `current_price`（= bar_T的close）判断
- 问题：止损应该在bar_T的high/low触及时触发，而不是等到close。但backtest的 `current_price` 是bar_T的close，这意味着：
  - 如果bar_T的low触及止损位但close回到止损位上方，backtest会**漏掉这个止损**（乐观偏差）
  - 如果bar_T的close刚好触及止损但盘中一度远低于止损位，backtest的exit_price = close（比实际止损出场价更好）
- 严重程度：**中等**。止损用close而非high/low是乐观偏差，可能高估了P&L
- 修复建议：在check_exit前先检查 `low < stop_price`（LONG）或 `high > stop_price`（SHORT），触发时exit_price用stop_price而非close

### [D13-NOTE] 极值追踪与执行时序 — ⚠ 偏乐观
- 代码位置：backtest_signals_day.py 第259-262行
- 当前行为：`position["highest_since"] = max(highest_since, high)`，其中 high = bar_T的high
- 问题：highest_since用bar_T的high更新，但trailing_stop用bar_T的close判断
- 这意味着：如果bar_T先到新高再回落到trailing stop以下（close），backtest会在同一根bar既更新了有利极值又触发了止盈，时序上不可能同时发生
- 严重程度：**低**。出现频率低，且影响有限

---

## E. Exit逻辑

### [E14] Trailing Stop — ✅ 无泄漏（但见D13-NOTE）
- 代码位置：A_share_momentum_signal_v2.py 第411-420行
- 当前行为：`dd = (highest - current_price) / highest`
- highest_since来自已处理的bar，current_price来自bar_T的close
- check_exit接收的bar_5m是bar_5m_signal（排除当前bar），Bollinger计算正确
- 无数据泄漏，但有D13-NOTE中描述的同bar极值+止盈矛盾

### [E15] MOMENTUM_EXHAUSTED — ✅ 无泄漏
- 代码位置：A_share_momentum_signal_v2.py 第437-458行
- 当前行为：`bar_5m["close"].astype(float).iloc[-3:]`，3根bar来自bar_5m_signal
- 这3根是已完成的bar，不含当前forming bar ✅
- hold_minutes用entry_time和current_time_utc（backtest中 = bar_T的时间）✅

### [E16] TREND_COMPLETE — ✅ 无泄漏
- 代码位置：A_share_momentum_signal_v2.py 第422-433行
- 当前行为：用 zone_5m 和 zone_15m，它们来自bar_5m_signal的Bollinger带
- current_price用于判断价格在Bollinger带中的位置 ✅

### [E17] MID_BREAK — ⚠ 存在差异
- 代码位置：A_share_momentum_signal_v2.py 第463-479行
- 当前行为：`five_below = current_price < b5_mid`
- current_price = bar_T的close（执行价），b5_mid来自bar_5m_signal（信号bars）
- 这里用execution price对比signal bars的Bollinger中轨，时间上有1根bar偏移
- 严重程度：**极低**。这个偏移是合理的（你在bar_T执行时看到price已经破了上一根bar计算的中轨）

---

## F. Monitor特有

### [F18] monitor bar数据排除 — ✅ 与backtest一致
- 代码位置：monitor.py 第887-902行
- 当前行为：`k5.iloc[:-1]` 和 `k15.iloc[:-1]` 排除TQ的forming bar
- 与backtest的 `bar_5m.iloc[:-1]` 一致 ✅

### [F19] monitor shadow exit的cur_price — ✅ 2026-04-02修复
- **修复前问题**：`cur_price = float(b5.iloc[-1]["close"])`（现货close），但entry_price是期货bid1/ask1
  - IM贴水3-4%，SHORT持仓一开仓就"亏损3.5%"，瞬间假触发STOP_LOSS
  - 同理 Bollinger zone 判断用期货价对比现货布林带，zone分类完全错误
- **修复后**：`check_exit` 新增 `spot_price` 参数，实现双价格层：
  - 止损/跟踪止盈/PnL → 用 `current_price`（期货last_price，与entry_price同源）
  - Bollinger zone/MID_BREAK → 用 `boll_price`（现货close，与bar_5m同源）
  - `highest_since`/`lowest_since` 极值追踪 → 用期货价格
  - 回测不传 `spot_price`（默认0 → fallback到current_price），回测全用现货不受影响
- **exit_price**用期货 `fq.last_price`（与entry同源）
- 严重程度：**已修复**。修复前是critical bug（实盘假止损），不是差异

### [F20] monitor entry价格 — ⚠ 存在差异（设计选择）
- 代码位置：monitor.py shadow注册处
- 当前行为：shadow entry_price = 期货盘口 ask1（LONG）或 bid1（SHORT）
- 与backtest差异：backtest entry_price = 现货bar_T的close + slippage
- 差异分析：
  - Monitor用实时期货盘口价（更贴近实盘执行）
  - Backtest用现货close（更稳定但与实际交易标的不同）
  - 两者相差约 spot-futures basis（IM约270点，3-4%）
- 严重程度：**低**。这是设计选择而非数据泄漏。P&L点数差异主要来自basis

---

## 发现的泄漏/问题汇总

### ❌ 确认泄漏/系统性偏移（已修复）

| # | 问题 | 严重程度 | 状态 |
|---|------|---------|------|
| D13 | 止损用bar close判断而非high/low | 中等 | ✅ 已修复（commit 4b64bce） |
| F19 | check_exit现货/期货价格混用导致假止损 | **严重** | ✅ 已修复（2026-04-02，双价格层） |
| G1 | 15分钟重采样 label='right'→'left'，时间标签偏移15分钟 | **高** | ✅ 已修复（2026-04-04） |

### ⚠ 差异（非泄漏但影响可比性）

| # | 问题 | 严重程度 | 影响 |
|---|------|---------|------|
| D13-NOTE | 同bar极值更新+trailing stop矛盾 | 低 | 偶尔出现不可能的同bar新高+止盈 |
| F20 | shadow entry用期货盘口价，backtest用现货close | 低 | P&L不可直接比较（设计选择） |
| A3-NOTE | session_weight时间偏移5分钟 | 极低 | 与monitor一致 |
| B7-NOTE | GARCH/sentiment硬编码IM | 功能问题 | 非IM品种波动率判断不精确 |
| E17 | MID_BREAK用execution price对比signal bars Bollinger | 极低 | 合理的时序行为 |

### ✅ 已确认无泄漏

A1, A2, A3, B4, B5, B6, B7, C8, C9, C10, D11, D12, E14, E15, E16, F18, F19（修复后）, G1（修复后）

---

## 修复状态（2026-04-04更新）

1. **D13（止损）**：✅ 已修复（commit 4b64bce）。check_exit前用bar high/low检查止损位，exit_price=stop_price。修复后PnL+72pt。
2. **F19（现货/期货混用）**：✅ 已修复（2026-04-02）。check_exit双价格层：止损用期货，Bollinger用现货。修复前IM/IC做空一开仓就假止损。
3. **F20（entry价格差异）**：设计选择，不修。Monitor用期货盘口价更贴近实盘。
4. **D13-NOTE（同bar极值矛盾）**：已通过止损优先级修复缓解（止损在极值更新前检查）。
5. **G1（15分钟重采样对齐）**：✅ 已修复（2026-04-04）。详见下方G1条目。

---

## G. 时间对齐问题

### [G1] 15分钟K线重采样对齐 — ❌ 存在系统性偏移（已修复）

- 代码位置：`backtest_signals_day.py` + `A_share_momentum_signal_v2.py` 中的 `_build_15m_from_5m()`
- **问题**：`resample('15min', label='right', closed='right')`（旧版默认）使每根15m bar的标签是该时间窗口的**结束时间**（右端）。
  - 例如：09:30-09:45这根15m bar的标签是09:45，而非09:30
  - 导致时段权重判断偏移15分钟（09:30-09:45的bar被认为是09:45时段）
  - 持仓时间计算用15m bar时间戳，偏移15分钟影响所有时间相关的exit判断
  - TREND_COMPLETE/MID_BREAK等使用15m bar时间的判断全部受影响
- **修复**：改为 `resample('15min', label='left', closed='left')`，标签为窗口左端（开始时间）
- **影响**：这是影响所有包含15m逻辑的回测结果的系统性bug。修复后IM+IC改善显著（具体数值含在整体+824pt中）
- **严重程度**：**高**。影响回测准确性，但修复前后的策略逻辑是相同的，修复只是让回测更准确反映实盘行为
- **Monitor中的对应代码**：monitor.py中的15m重采样也应统一使用`label='left', closed='left'`
