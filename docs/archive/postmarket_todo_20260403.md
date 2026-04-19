# 盘后待办 2026-04-03

## Bug修复（已commit待重启生效）

- [x] `remove_by_symbol` 清理 lock_pairs（cc9c807）— IM 10:20 ME退出后无法再开仓的root cause
- [x] `prompted_bars` 用 bar_data 时间戳去重（ba714d5）— 信号重复发送2次
- [x] vol_monitor 持仓行权价不跨月污染（f00d483）— MO2606-P-6000的6000导致全部订阅失败

## 盘中观察记录（需复盘分析）

### 1. IM 10:20 ME退出时 score=72→82
- **又一个高分ME退出案例**
- ME退出恰好在反转点（10:20触底反弹）
- 这次ME退对了——如果继续持有会被反转吃掉利润
- 但系统在反弹后仍然给做空信号（M分滞后60分钟窗口）
- **记入ME研究日志：ME有时能间接"预测"反转（动量耗尽→价格反转）**

### 2. IC持仓86+分钟未平仓（退出系统结构性弱点）
- entry=09:55 SHORT @7384，从+21pt跌到-29pt又回到+1pt
- **所有退出阈值都差一点没触发**：
  - TIME_STOP：微利~1pt → profitable=True → 不触发
  - TRAILING_STOP：回撤0.81% < 1.0%阈值 → 不触发
  - STOP_LOSS：7413 < 7421(0.5%) → 不触发
  - ME/TC：K线形态不满足
- **根因**：退出参数设计为趋势市，震荡市里价格在阈值边缘反复穿越但从不满足
- **待研究**：
  - TIME_STOP 改为"60min后浮盈<N pt就退"（而非严格亏损才退）
  - 或"60min后无条件退"
  - 回测验证不同方案的效果

### 3. IM/IC ME触发不一致
- 走势几乎同步（截图确认），但IM 10:20 ME退出，IC一直没ME
- 原因：`total_range / boll_width` 数值边界——IM刚好<0.20触发，IC刚好>0.20没触发
- **待研究**：IC/IM互为参考机制——如果IM已ME且IC走势相同，IC的ME条件放宽
- 已记入signal_log的style_spread和cross_rank数据可用于分析

### 4. M分反转敏感度不足
- 10:20开始V型反转60pt，但10:30-10:35 M分仍然50分做空
- 12根bar（60分钟）窗口对10分钟反转完全迟钝
- 这是TODO里已记录的"M分短期反转敏感度不足"问题
- **待研究**：短期动量确认（最近3-6根bar的方向和长期动量是否一致）

### 5. 信号重复发送（已修复待重启）
- 09:50 IM SHORT 信号发了2次给executor
- executor正确处理（第二次跳过），但不应该发2次
- 根因：prompted_bars用_last_bar_time（per-symbol不同步）
- 已修复：改用bar_data时间戳

### 6. 09:35面板全显示 `--`
- score_all因时间过滤(UTC 01:30 < 01:35)返回None
- 09:40恢复正常，09:45开始可发信号
- 这不是新bug，是一直存在的行为
- **可优化**：考虑在面板中显示"(盘前)"而非"--"

## 盘后回测

### IC 4/3 逐K线回放
```bash
python scripts/backtest_signals_day.py --symbol IC --date 20260403
python scripts/backtest_signals_day.py --symbol IM --date 20260403
```
对比monitor实盘signal_log，确认回测和实盘信号一致。

## 代码改进（盘后实施）

### 优先级高
1. **TIME_STOP 逻辑改进**：60min后微利不退是结构性弱点，研究替代方案
2. **面板去重**：4品种触发4次_on_new_bar各打印一次面板，改为只打印一次

### 优先级中
3. **IC/IM互参考ME**：利用cross_rank/style_spread数据
4. **M分短期反转检测**：3-6根bar方向确认

### 优先级低
5. **面板09:35前显示"(盘前)"**
6. **vol_monitor 09:30行情延迟处理**
