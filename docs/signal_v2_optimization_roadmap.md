# Signal V2 参数审计 + 优化 Roadmap

> 审计时间: 2026-04-11
> 修订 v2: 2026-04-11（per-symbol原则强化 + 底层因子优先级调整）
> 覆盖范围: A_share_momentum_signal_v2.py, factors.py, atomic_factors.py, strategy.py, monitor.py, order_executor.py, position.py, risk.py, backtest_signals_day.py

---

## 一、完整参数清单

### 1.1 SYMBOL_PROFILES 品种配置

| 参数 | IM | IC | IF | IH | 默认 |
|------|----|----|----|----|------|
| style | MOMENTUM | MOMENTUM | HYBRID | MEAN_REVERSION | HYBRID |
| signal_threshold | **55** | **60** | 60 | 60 | 60 |
| stop_loss_pct | **0.003** | 0.005 | 0.005 | 0.005 | 0.005 |
| trailing_stop_scale | **1.5** | **2.0** | 1.0 | 1.0 | 1.0 |
| dm_trend | 1.1 | 1.1 | **1.0** | 1.1 | 1.1 |
| dm_contrarian | 0.9 | 0.9 | **1.0** | 0.9 | 0.9 |
| me_ratio | 0.10 | **0.12** | 0.10 | 0.10 | 0.10 |
| momentum_lookback_daily | 5 | 5 | **20** | **20** | 5 |
| reversal_filter | False | False | **True** | **True** | True |
| reversal_threshold | 0 | 0 | **0.02** | **0.01** | 0.015 |
| trend_weight | 60 | 60 | 50 | **35** | 50 |
| vol_weight | 25 | 25 | 30 | 30 | 30 |
| volume_weight | 15 | 15 | 20 | 15 | 20 |
| reversal_weight | 0 | 0 | 0 | **20** | 0 |
| tradeable | **YES** | **YES** | NO | NO | NO |

**品种差异验证状态：**

| 参数 | IM值 | IC值 | 差异验证? | 备注 |
|------|------|------|----------|------|
| signal_threshold | 55 | 60 | IM=900天验证,IC=旧值 | **IC的60需要per-symbol重验证** |
| stop_loss_pct | 0.003 | 0.005 | IM=稳健性三检,IC=默认 | **IC的0.5%需要per-symbol验证** |
| trailing_stop_scale | 1.5 | 2.0 | 两个都做过215天验证 | 已验证 |
| me_ratio | 0.10 | 0.12 | IC=独立验证 | 已验证 |
| dm_trend/contrarian | 1.1/0.9 | 1.1/0.9 | **合并样本验证,未per-symbol** | **需要per-symbol独立验证** |

### 1.2 评分因子参数

| 参数 | 值 | 物理含义 | 上次优化 | per-symbol? |
|------|------|---------|---------|------------|
| _DYN_LB_LOW | 4 | 低振幅日M分lookback(20min) | 方案E 216天 | 全局 |
| _DYN_LB_HIGH | 12 | 高振幅日M分lookback(60min) | 方案E 216天 | 全局 |
| _DYN_AMP_THR | 0.015 | 振幅切换阈值(1.5%) | 方案E 邻域验证 | 全局 |
| _MOM_15M_LB | 6 | 15分钟动量lookback | **从未优化** | **全局,应per-symbol** |
| M分阈值 0.3%/0.2%/0.1% | 35/25/15分 | 动量幅度→分数映射 | **从未优化** | **全局,应per-symbol** |
| M分最大 | 50 | 动量最高分 | 从未优化 | 全局 |
| 15m确认加分 | +15 | 15分钟同向确认 | 从未优化 | 全局 |
| ATR_SHORT/LONG | 5/40 | ATR短/长周期 | 从未优化 | 全局 |
| V分ATR ratio阈值 | 0.7/0.9/1.1/1.5 | 波动率→分数映射 | **从未优化** | **全局,应per-symbol** |
| V分最大 | 30 | 波动率最高分 | 从未优化 | 全局 |
| Q分量比阈值 | 1.5/0.5 | 成交量surge/low | 从未优化 | 全局 |
| Q分最大 | 20 | 成交量最高分 | 从未优化 | 全局 |
| VOLUME_PCT_HIGH/LOW | 0.75/0.25 | Q分百分位阈值 | 从未优化 | 全局 |
| B分基础 | 10 | 布林带突破基分 | 从未优化 | 全局 |
| B分放量/窄带/15m | +2/+3/+5/+3 | 突破加分细节 | 从未优化 | 全局 |
| B分最大 | 20 | 突破最高分 | 从未优化 | 全局 |
| S分条件 | 5bar/1.5x/60%pct | 启动因子条件 | 从未优化 | 全局 |
| S分最大 | 15 | 启动最高分 | 从未优化 | 全局 |
| BOLL参数 | 20/2 | 布林带SMA/STD | 标准参数 | 全局 |
| OPEN_RANGE_FILTER_PCT | 0.004 | 开盘振幅过滤(0.4%) | 215天验证 | 全局 |

### 1.3 乘数管道参数

| 参数 | 值 | 物理含义 | 上次优化 | per-symbol? |
|------|------|---------|---------|------------|
| daily_mult(顺势) | 1.2 | 日线方向一致加成 | 固定 | 全局 |
| daily_mult(逆势) | 0.8 | 日线方向相反惩罚 | 2026-03-30 | 全局 |
| daily_mult(中性) | 1.0 | 日线方向不明 | 固定 | 全局 |
| dm_trend/contrarian | per-symbol | 趋势/逆势额外乘数 | **合并样本验证** | per-symbol(值相同) |
| intraday_filter | 方向感知表 | 日内涨跌幅乘数 | 手动设定 | 全局 |
| Z-score乘数 | 0~1.5 | 估值极端乘数 | 手动设定 | 全局 |
| session_multiplier | per-symbol | 时段乘数 | 部分验证 | per-symbol(PROFILES) |

**Session weight 机制澄清（v2修订）：**
- `_get_session_weight()` 读取 PROFILES 里的 `session_multiplier`（per-symbol）
- `_get_time_weight()` 读取全局 `_TIME_WEIGHTS`，但**在score_all里未被调用**
- **实际生效的是per-symbol版本**，没有两套并行冲突
- 全局 `_TIME_WEIGHTS` 是死代码，可以清理但不影响行为

### 1.4 出场条件参数

| 参数 | 值 | IM特殊值 | IC特殊值 | 上次优化 |
|------|------|---------|---------|---------|
| stop_loss_pct | 0.005 | **0.003** | 0.005 | IM=稳健性三检 |
| trailing_stop基础(0-15min) | 0.5% | ×1.5 | ×2.0 | 215天验证 |
| trailing_stop(15-30min) | 0.6% | ×1.5 | ×2.0 | - |
| trailing_stop(30-60min) | 0.8% | ×1.5 | ×2.0 | - |
| trailing_stop(>60min) | 1.0% | ×1.5 | ×2.0 | - |
| trailing_15m确认加宽 | +0.2% | - | - | 从未优化 |
| trailing_profit加宽 | +0.2%(>0.5%) | - | - | 从未优化 |
| TRAILING_STOP_LUNCH | 0.3% | - | - | 从未优化 |
| me_min_hold | 20min | - | - | 2026-03-30 |
| me_ratio(narrow_range) | 0.10 | 0.10 | **0.12** | IC独立验证 |
| mid_break_bars | 3 | - | - | 2026-04-04 |
| **TIME_STOP_MINUTES** | **60** | **-** | **-** | **从未优化,全局** |
| time_stop仅亏损时 | PnL<=0 | - | - | 从未优化 |
| EOD_CLOSE_UTC | 06:45 | - | - | 固定 |
| LUNCH_CLOSE_UTC | 03:25 | - | - | 固定 |
| NO_OPEN_EOD | 06:30 | - | - | 从14:15推迟 |

### 1.5 时间窗口参数

| 参数 | 值(UTC) | BJ时间 | 上次优化 |
|------|---------|--------|---------|
| NO_TRADE_BEFORE | 01:35 | 09:35 | 固定 |
| NO_TRADE_AFTER | 06:50 | 14:50 | 固定 |
| NO_OPEN_LUNCH_START | 03:20 | 11:20 | 固定 |
| NO_OPEN_LUNCH_END | 05:05 | 13:05 | 固定 |
| NO_OPEN_EOD | 06:30 | 14:30 | 从14:15推迟 |
| session_multiplier | per-symbol | - | PROFILES中定义 |

### 1.6 风控和执行参数

| 参数 | 值 | 物理含义 |
|------|------|---------|
| EXEC_MAX_LOTS | 1 | 实盘验证期每笔最大1手 |
| max_lots_per_symbol | 1 | 单品种最大持仓 |
| max_total_lots | 2 | 全品种最大持仓 |
| MAX_DAILY_ORDERS | 10 | 单日最大下单数 |
| max_daily_trades_per_symbol | 5 | 单品种单日最大交易 |
| max_consecutive_losses | 3 | 连亏3笔暂停 |
| max_daily_loss | 50,000 | 日亏损上限(元) |
| SIGNAL_EXPIRY_SECS | 300 | 信号有效期(5min) |
| OPEN_FILL_TIMEOUT | 60 | 开仓超时(秒) |
| CLOSE_FILL_TIMEOUT | 30 | 平仓超时(秒) |
| COOLDOWN_MINUTES(回测) | 15 | 回测冷却期 |

---

## 二、Per-Symbol 优化机会

### 2.1 IM 优化机会（按ROI排序）

| # | 参数 | 当前值 | 优化可能性 | 预期改进 | 成本 | ROI |
|---|------|--------|-----------|---------|------|-----|
| 1 | V分ATR ratio阈值 | 0.7/0.9/1.1/1.5全局 | per-symbol阈值 | 中-大 | 中 | **A** |
| 2 | M分阈值(0.3/0.2/0.1%) | 全局固定 | per-symbol mapping | 中-大 | 低-中 | **A** |
| 3 | dm_trend/contrarian | 1.1/0.9(合并验证) | IM独立sweep | 中 | 低 | **A** |
| 4 | TIME_STOP_MINUTES | 60(全局) | IM独立(45-90) | 中 | 低 | **B** |
| 5 | session_multiplier | per-symbol已生效 | 验证各时段贡献 | 小-中 | 低 | **B** |
| 6 | trailing_stop基础宽度 | 0.5-1.0%×1.5 | 微调宽度 | 小 | 低 | **B** |
| 7 | _MOM_15M_LB | 6(全局) | per-symbol(4-8) | 小 | 低 | **B** |
| 8 | reversal_filter | False | 评估启用的可能性 | 小 | 中 | **C** |
| 9 | B分加分细节 | 固定+2/+3/+5 | per-symbol | 小 | 中 | **C** |
| 10 | intraday_filter乘数表 | 手动设定 | 回测验证 | 小 | 高 | **C** |

### 2.2 IC 优化机会（按ROI排序）

| # | 参数 | 当前值 | 优化可能性 | 预期改进 | 成本 | ROI |
|---|------|--------|-----------|---------|------|-----|
| 1 | signal_threshold | 60(旧值) | 50-65 sweep | 中-大 | 低 | **A** |
| 2 | stop_loss_pct | 0.005(默认) | 0.003-0.005 | 中 | 低 | **A** |
| 3 | V分ATR ratio阈值 | 0.7/0.9/1.1/1.5全局 | per-symbol阈值 | 中-大 | 中 | **A** |
| 4 | M分阈值 | 全局固定 | per-symbol mapping | 中-大 | 低-中 | **A** |
| 5 | dm_trend/contrarian | 1.1/0.9(合并验证) | IC独立sweep | 中 | 低 | **A** |
| 6 | TIME_STOP_MINUTES | 60(全局) | IC独立(45-90) | 中 | 低 | **B** |
| 7 | session_multiplier | per-symbol已生效 | 验证各时段贡献 | 小-中 | 低 | **B** |
| 8 | trailing_stop基础宽度 | 0.5-1.0%×2.0 | 微调scale或基础 | 小 | 低 | **B** |
| 9 | _MOM_15M_LB | 6(全局) | per-symbol(4-8) | 小 | 低 | **B** |
| 10 | reversal_filter | False | 评估启用的可能性 | 小 | 中 | **C** |

### 2.3 IF 初始配置问题

IF 当前不在 tradeable 集合中。优先级低于IM/IC边际优化。
- IF是HYBRID风格(dm=1.0/1.0)，逆势交易是利润来源，参数逻辑与IM/IC完全不同
- 建议：等IM/IC的第一批+第二批优化完成后再评估

---

## 三、规则层面优化点

| # | 代码位置 | 规则描述 | 可能优化方向 | 优先级 |
|---|---------|---------|------------|--------|
| 1 | signal_v2 L1365-1441 | intraday_filter乘数表 | 回测验证每个阈值的边际贡献 | 中 |
| 2 | signal_v2 L732-796 | Z-score乘数表 | 验证各区间乘数是否最优 | 中 |
| 3 | factors.py L428-473 | trailing_stop时间分段宽度 | per-symbol验证4段宽度 | 低 |
| 4 | factors.py L497-523 | ME触发条件 | 三个子条件的阈值per-symbol | 低 |
| 5 | signal_v2 L257-297 | S分启动因子 | 三个条件是否过严/过松 | 低 |
| 6 | SYMBOL_PROFILES | **reversal_filter开关** | **IM/IC当前关闭,未per-symbol验证** | **中** |

**reversal_filter说明（v2新增）：**
- IM/IC=False, IF/IH=True，但IM/IC的False从未经过per-symbol验证
- 这是一个规则级开关，不是连续参数
- 研究方法：在IM/IC上分别跑reversal_filter=True的回测，对比False基线
- 如果某个品种开启reversal_filter有显著改善，说明当前关闭是次优的

---

## 四、Roadmap

### 底层参数验证（优先于所有表层优化）

> **战略考量**：V分和M分是核心因子，它们的阈值从未系统验证。所有其他参数的优化都建立在"当前V分/M分阈值是合理的"这个未验证假设上。如果底层阈值是偏的，表层优化可能在错误方向上努力。
>
> **务实权衡**：底层验证需要更多时间（基本面分析+sweep+验证），不能阻塞快速可做的表层优化。策略是：**第一批做独立的快速项目，同时启动底层验证作为并行项目**。

### 第一批（1-2周）—— 快速独立项目 ✅ 全部完成

**项目1: IC signal_threshold + stop_loss_pct** ✅ 2026-04-11
- 681/219双窗口30组合扫描
- IC: thr 60→55, sl 0.5%→0.3% (OOS +912pt改善)
- IM: thr 55→60 (OOS +526pt改善)

**项目2: trailing_stop_scale** ✅ 2026-04-12
- IM 1.5 / IC 2.0 均已是OOS最优，不动

**项目3: me_ratio** ✅ 2026-04-12
- IM 0.10已最优，IC 0.12→0.10 (OOS +354pt)

**项目4: TIME_STOP_MINUTES** ✅ 2026-04-12
- 30-90全部PnL相同，参数无效果，不动

**dm_trend/contrarian** — 当前不活跃(daily_bar=None中性化)，不需要审计

### 第二批 —— 底层因子验证 → 已转化为 v1 新映射系统

**项目5-6 (V分/M分阈值)** → 转化为signal_new_mapping_v1
- MVQB诊断完成（2026-04-12）：V分U型、M分分档无效
- 不再在v2框架内优化，而是通过v1数据驱动映射解决
- 详见 `docs/signal_new_mapping_v1.md`

**项目7: _MOM_15M_LB** ⏳ 待做
- 15m lookback仍需per-symbol验证

### 第三批（1-2月）—— 表层优化

**项目8: session_multiplier per-symbol优化** ⏱️2-3天
- 机制已澄清（PROFILES的session_multiplier生效，全局_TIME_WEIGHTS是死代码）
- per-symbol做时段贡献分析，找出各品种最赚/最亏的时段
- 可能调整某些时段的乘数

**项目9: trailing_stop基础宽度 per-symbol** ⏱️1周
- 4段时间宽度(0.5/0.6/0.8/1.0%)从未做per-symbol验证
- scale已验证(IM=1.5,IC=2.0)，但基础宽度本身是否最优未知

**项目10: reversal_filter开关验证** ⏱️2-3天
- IM/IC当前关闭，未经per-symbol验证
- 在IM/IC上分别跑reversal=True回测，对比基线

### 长期 —— C级ROI

**项目11:** intraday_filter乘数表验证 (2-3周)
**项目12:** Z-score乘数表验证 (2-3周)
**项目13:** B分/S分细节参数 (1周)
**项目14:** IF上线评估 (2-4周)

---

## 五、不建议优化的参数

| 参数 | 当前值 | 理由 |
|------|--------|------|
| IM signal_threshold | 55 | 2026-04-09 900天IS/OOS验证,至少3个月不动 |
| IM trailing_stop_scale | 1.5 | 215天per-symbol验证 |
| IC trailing_stop_scale | 2.0 | 215天per-symbol验证 |
| me_ratio(IC) | 0.12 | IC独立验证 |
| mid_break_bars | 3 | 2026-04-04验证 |
| me_min_hold | 20min | 2026-03-30验证(+16%) |
| 动态lb阈值 | 1.5% | 方案E邻域验证,12个邻域全正 |
| 动态lb值 | 4/12 | 方案E 216天验证(+20.2%) |
| OPEN_RANGE_FILTER_PCT | 0.4% | 215天验证 |
| BOLL参数 | 20/2 | 标准参数 |
| 风控参数(max_lots等) | 见上表 | 风控不做优化 |

**注：dm_trend/contrarian已从此列表移除**——原验证是合并样本(+691pt合计)，不符合per-symbol原则，需要per-symbol独立验证（见第一批项目3）。

---

## 六、备注

- 审计覆盖了signal_v2核心打分+出场+执行+风控的全部硬编码参数
- 未覆盖: models/factors/(研究框架), vol_monitor(波动率监控), morning_briefing(已禁用d_override)
- 局限性: 预期改进和ROI评级是基于经验判断，实际需要回测验证
- 本文档是活文档，每完成一个优化项目后更新状态
- 全局`_TIME_WEIGHTS`是死代码（score_all中未调用），可以后续清理

---

## 修订日志

### 2026-04-11 修订 v2

基于review发现的遗漏和per-symbol原则强化：

- **修订1**: V分ATR ratio阈值从B级 → **A级**。V分是核心regime选择器，阈值从未验证，是所有后续优化的基础假设。归入第二批"底层因子验证"。
- **修订2**: M分阈值从B级 → **A级**。核心打分因子，整齐数字从未sweep，per-symbol差异明确。归入第二批"底层因子验证"。
- **修订3**: TIME_STOP_MINUTES加入per-symbol优化机会，评为**B级**。全局60分钟从未优化，不同品种动量延续时间不同。归入第一批快速项目。
- **修订4**: dm_trend/contrarian从"不建议优化"移到第一批优化列表，评为**A级**。原验证是合并样本(+691pt)，不符合per-symbol原则。
- **修订5**: _MOM_15M_LB明确为全局硬编码，标注"应per-symbol"，归入第二批。需要代码改造(从常量改为PROFILES配置)。
- **修订6**: session_weight机制已澄清——PROFILES的session_multiplier是实际生效的per-symbol版本，全局_TIME_WEIGHTS在score_all中未被调用（死代码）。从"代码调查"改回"per-symbol优化"，移到第三批。
- **修订7**: reversal_filter加入规则层面讨论。IM/IC当前关闭未经per-symbol验证。归入第三批。
