# 项目上下文

## 这是什么项目
A股股指期货/期权多策略量化交易系统（实盘运行中）。
- 数据层：Tushare 历史数据 + 天勤 TqSdk 实时行情和交易
- 标的：IM（中证1000期货）、MO（中证1000期权）为主
- 核心策略：① VRP 做空波动率（卖出 MO OTM Strangle）② 贴水捕获（多 IM 期货 + 保护性 Put）

---

## 关键技术决策

### 期权定价
- **使用 PCP 隐含 Forward Price**，不用期货收盘价
- 从看涨/看沽平价公式反推每个到期月份的隐含远期价格
- 入口：`scripts/portfolio_analysis.py` → `calc_implied_forwards_by_expiry()`

### IML 连续合约映射（重要！）
- Tushare 命名：`IM.CFX`/`IML.CFX` = 当月，`IML1.CFX` = 次月，`IML2.CFX` = 当季，`IML3.CFX` = 隔季
- 映射数组：`_IML_CODES = ["IM.CFX", "IML1.CFX", "IML2.CFX", "IML3.CFX"]`（index 0 = 当月）
- 单一真相来源：`utils/cffex_calendar.py`，包含 `active_im_months()`、`get_im_futures_prices()`、`get_main_contract()`
- **主力合约选择**：`get_main_contract(symbol, api=None)` 有TQ时按持仓量(open_interest)选主力，离线fallback到按到期日选近月
- Monitor和Executor统一调用此函数，不再各自实现合约选择逻辑
- 历史教训：off-by-one 会导致贴水计算错乱，不要在 cffex_calendar 以外重复实现合约映射

### 贴水计算
- **现货基准**：优先用 `000852.SH`（中证1000指数）from `index_daily` 表，fallback 到 `IM.CFX`
- 年化公式：`abs(futures_price - spot_price) / spot_price * (365 / DTE)`
- 信号阈值：> 15% STRONG / 10-15% MEDIUM / 5-10% WEAK

### 期权链数据
- `options_daily` 表的 `underlying_code` 字段为 NULL，不能依赖
- 必须从 `ts_code` 字段 parse（正则 `MO(\d{4})-(C|P)-(\d+)`）
- 入口：`scripts/portfolio_analysis.py` → `_parse_mo()`

### 合约乘数
- **MO 期权**：乘数 = 100（每手 100 份）
- **IM 期货**：乘数 = 200

### Greeks 计算
- **用 implied forward 而非期货价**作为标的价，使 BS 模型和 IV 保持一致
- 组合 Greeks 汇总：`models/pricing/greeks.py` → `calc_portfolio_greeks()`

### 日内信号数据源（2026-03-25决定）
- **所有日内信号计算统一用现货指数，不用期货**
- 现货TQ代码：IM→SSE.000852, IF→SSE.000300, IH→SSE.000016, IC→SSE.000905
- 跨日比较（prev_close, daily_mult）→ 现货（无换月跳变）
- 日内技术指标（BOLL/RSI/MA）→ 现货K线
- 期货只用于：面板显示价格、计算贴水、归档、实际下单
- 数据表：`index_min`（现货5分钟K线）、`index_daily`（现货日线）

### 现货/期货价格分层（2026-04-02修复，重要！）
- **IM贴水3-4%，IC贴水2-3%，同一计算中混用现货和期货价格会导致严重错误**
- **信号评分**（score_all）：全部用现货（布林带/RSI/动量/成交量）
- **持仓管理**（check_exit止损/跟踪止盈/PnL）：全部用期货（与entry_price同源）
- **Bollinger平仓zone**：用现货价格（与bar_5m同源），通过 `spot_price` 参数传入check_exit
- check_exit 的 `spot_price` 参数：实盘传现货close，回测传0（fallback到current_price，回测全用现货）
- **历史教训**：shadow entry_price=期货bid1，check_exit用现货close做止损判断，贴水3.5%直接假触发STOP_LOSS
- `highest_since`/`lowest_since` 极值追踪也必须用期货价格（与entry_price同源）
- strategy.py 的 position_mgr 用现货价格（与信号entry_price/stop_loss同源，只做占位不做实际止损）

### 信号架构统一（2026-03-25修复）
- `strategy.py` 必须使用 `A_share_momentum_signal_v2.py` 的 `SignalGeneratorV2`
- 不能用旧版 `signal.py` 的 `IntradaySignalGenerator`
- strategy层和monitor面板层必须传入相同参数：`zscore`, `is_high_vol`, `sentiment`

### intraday_filter（日内涨跌幅）
- 用 当前价 vs 昨日现货收盘价（含跳空gap）
- **prevClose取值**：`daily_bar` 中 `trade_date < 当天` 的最后一行（不是 `iloc[-2]`，避免当天日线未入库时取到前天）
- 高波动区间阈值放宽：2-3%逆势 d=0.5（不是0.3）
- 方向感知：顺势不打折或轻打折，逆势重打折
- **低波动模式修复（2026-03-30）**：旧版 `_intraday_filter_mild` 把顺势/逆势搞反（顺势追跌f=0.3），已修正为与高波动模式一致的方向逻辑
- **日线逆势惩罚（对称，2026-03-30调整）**：
  - 逆势统一：`daily_mult = 0.8`（做多做空对称，干净数据验证）
  - 顺势：`daily_mult = 1.2`，中性：`daily_mult = 1.0`
  - 注：之前逆势做空曾设为0.5，基于含未来数据泄漏的回测（WR=42%,-2.5pt），修复数据泄漏后干净数据显示WR=60%,+26pt，改为0.8后+72pt增量（+13%），breakeven滑点4.2pt

### 开盘振幅过滤器（2026-04-04新增）
- 开盘30分钟（前6根5分钟bar）的振幅 < 0.4% → 后续不开新仓
- 215天验证：低振幅日(<1%)均PnL=-27pt/天，过滤13天+274pt改善，误杀率15%
- 开盘30min振幅与全天振幅的相关系数r=0.66（实用价值）
- `check_low_amplitude(bar_5m)` 在 `A_share_momentum_signal_v2.py` 中定义
- Monitor 在 10:00 BJ (UTC 02:00) 后检查，backtest 同样逻辑
- 10:00前的交易正常进行（因为需要6根bar数据才能判断）

### dm_trend/dm_contrarian（2026-04-04调整）
- **IM/IC/IH：1.1/0.9**（从1.2/0.8调整，215天最优+691pt差距最大的参数）
- **IF：保持1.0/1.0**（均值回归型，逆势是利润来源）
- 默认fallback：1.1/0.9
- 215天验证：1.0/1.0(中性) > 1.1/0.9(轻度) > 1.2/0.8(当前) > 1.3/0.7(激进)
- 历史教训：1.2/0.8对逆势惩罚过重，逆势交易在长期也是盈利的

### 平仓信号系统（2026-03-24新增，03-26优先级调整，04-02~04-04参数优化）
- 7个优先级：EOD_CLOSE > STOP_LOSS > LUNCH_CLOSE > TRAILING_STOP > TREND_COMPLETE > MOMENTUM_EXHAUSTED > MID_BREAK > TIME_STOP
- STOP_LOSS 提升到 LUNCH_CLOSE 之前（防止午休前大亏不止损）
- **Per-symbol止损（2026-04-08）**：IM=0.3%（稳健性三检通过+366pt），IC/IF/IH=0.5%（默认）
- TREND_COMPLETE：5分钟上轨 + 15分钟上轨 = 两个周期都到极端才平仓
- 用15分钟布林带判断趋势阶段，5分钟只用于跟踪止盈
- 动态跟踪止盈：持仓时间越长宽度越大（0.5%→1.0%）
- **IC trailing_stop_scale=2.0x（2026-04-04）**：IC的trailing stop宽度是IM的2倍（IC震荡性更强），+186pt
- **IM trailing_stop_scale=1.5x（2026-04-04）**：215天验证IM 1.5x > 1.0x（+259pt），止盈过紧频繁被震出
- **MOMENTUM_EXHAUSTED最小持仓20分钟（2026-03-30）**：5-15min止出75%是过早的（54笔中36笔），加最小持仓4根K线。32天PnL +634→+735pt（+16%），breakeven滑点2.5→4.0pt
- **ME narrow_range ratio 0.20→0.10（2026-04-04）**：ME触发条件收严（只有更窄的K线才算耗尽），IM+IC +393pt合计
- **MID_BREAK bars 2→3（2026-04-04）**：需要3根K线破中轨（而非2根），IM+IC +76pt合计
- 平仓后15分钟同方向冷却期
- 开仓窗口：09:45~11:20, 13:05~14:30（`NO_OPEN_EOD = "06:30"` = 14:30 BJ，从14:15推迟，尾盘信号WR=67-80%）

### VRP 计算（2026-03-26重构）
- **VRP = IV - Blended RV**（不再直接用 GARCH）
- Blended RV 正常 = `0.4×RV5d + 0.4×RV20d + 0.2×GARCH`
- GARCH Sanity Check：当 `GARCH > max(RV5d, RV20d) × 1.4` 时标记不可靠，降权为 `0.5×RV5d + 0.5×RV20d`
- GARCH 模型参数本身不改，只改下游使用方式
- `garch_reliable` 字段写入 `daily_model_output`

### Hurst指数 Regime Detection（2026-04-02新增）
- **R/S分析法计算Hurst(60d)**，用日线close滚动60天窗口
- H>0.55 趋势期，H<0.45 震荡期，0.45-0.55 中性
- 当前仅用于象限判断和面板显示，**不参与信号评分**（待数据积累后验证）
- 象限体系升级：高IV象限按Hurst细分为A1/A2/B1/B2
  - A1: 高IV+VRP>0+震荡 → 卖方最佳甜点区
  - A2: 高IV+VRP>0+趋势 → 顺势卖方可以，逆势危险
  - B1: 高IV+VRP<0+震荡 → 等VRP转正，可小仓卖方
  - B2: 高IV+VRP<0+趋势 → 最危险，禁止卖方
- Morning Briefing 和 quadrant_monitor 均输出 Hurst 值和历史分位

### 研究指标自动记录（2026-04-02新增）
- signal_log 每笔交易自动记录以下研究指标（不参与评分，仅用于积累样本后统计分析）：
  - `adx_14`：ADX趋势强度（待100+笔ADX>35样本后验证硬过滤）
  - `body_ratio`：K线实体占比（IM/IC品种特征不同，待更多样本）
  - `vwap_offset`：VWAP偏离度（与M冗余r=0.67，不计分）
  - `style_spread`：IM-IH收益率差（中性区间73%WR，待100+笔验证）
  - `cross_rank`：跨品种相对强弱rank（整体亏钱-50%，不计分）

### 情绪乘数 sentiment_mult（2026-03-26修正）
- VRP 不参与日内信号的情绪打折（VRP是波动率交易指标，与日内方向无关）
- 保留的调节因子：IV变动、Skew变化、PCR逆向、期限结构倒挂
- 乘数范围 0.5~1.5，做多时IV急升减分、做空时IV急降减分

### 布林带突破信号（2026-03-26新增）
- `score_all` 中新增 `_score_boll_breakout()` 维度（0~20分额外加分）
- 5分钟K线从中轨上方跌破中轨（做空）或从下方突破（做多）→ +10分
- 前置条件：突破前5根K线在中轨同侧 + 动量维度已有方向（`s_mom > 0`）
- 放量突破 +2分、窄带突破 +3分、15分钟同方向确认 +5分
- 面板/回测中显示 `B{n}` 标注

### 动量lookback动态切换（2026-04-07，方案E）
- **动态 lb=4/12**：开盘30min振幅<1.5%用lb=4（20分钟），否则lb=12（60分钟）
- 原理：赚钱日趋势持续35-40min用lb=12捕捉大波动，亏钱日震荡25-30min反转用lb=4及时止损
- 216天回测：IM+2120 IC+2836 = **+4956pt**（+20.2% vs 旧baseline +4124）
- 稳健性：时间分半前+21%后+30%，12个邻域全正，逐月无亏损
- `_score_momentum()` 根据 `check_low_amplitude()` 返回的振幅动态选择lookback
- 振幅阈值1.5%在1.0-2.0%范围内稳健（12个邻域全正）

### 日内策略品种配置（2026-04-08更新，方案E 216天回测验证）

| 品种 | 类型 | 状态 | 阈值 | dm(顺/逆) | trail_scale | M分lb | PnL(216d) | 均PnL |
|------|------|------|------|-----------|-------------|-------|-----------|-------|
| IM | 动量 | **实盘** | 50 | 1.1/0.9 | 1.5x | 动态4/12 | +2518 | +11.5 |
| IC | 动量 | **实盘** | 60 | 1.1/0.9 | 2.0x | 动态4/12 | +2142 | +9.8 |
| IF | 均值回归 | 观察 | 60 | 1.0/1.0 | 1.0x | 动态4/12 | 未测 | — |
| IH | — | 放弃 | 60 | 1.1/0.9 | 1.0x | 动态4/12 | 未测 | — |

IM+IC合计 **+4660pt/219天**（2025-05-16~2026-04-09）。含Q分分位数+15m修复+IM thr=50+IC ME=0.12。

**04-07~08 方案E改动（+20.2% vs 旧baseline +4124pt）：**
- M分lookback：静态lb=12 → 动态lb=4/12（振幅<1.5%用4，否则12）
- signal_threshold：IM 60→55，IC 65→60（动态lb提升了低分段信号质量）
- d_override禁用（briefing dm累计-19.2%负贡献，改为纯算法dm 1.1/0.9）

**04-04三项改动（215天敏感分析后实施，+21%/+717pt）：**
- dm_trend/dm_contrarian 1.2/0.8→1.1/0.9（IM/IC/IH，IF保持1.0/1.0）
- IM trailing_stop_scale 1.0→1.5x
- 开盘30min振幅<0.4%过滤器（`check_low_amplitude()`）

**此前改善来源（34天验证期）：**
- 15分钟重采样对齐修复（label='left'）：最大单项改善
- ME narrow_range ratio 0.20→0.10（+393pt合计）
- MID_BREAK bars 2→3（+76pt合计）
- IC trailing_stop_scale=2.0x（+186pt）
- 午后session_weight 0.8→1.0（+76pt）

- **实盘品种**：IM+IC，`IntradayConfig.tradeable = {"IM", "IC"}`，只有这两个品种通过strategy开仓占position_mgr槽位、写信号给executor、注册shadow持仓
- **IF观察**：monitor全品种监控面板显示评分，但不占position_mgr槽位、不触发开仓。IF的BE=1.2pt余量小，等shadow验证2-4周后决定
- **IH放弃日内**：BE=0.6pt无法覆盖滑点+手续费
- IC的thr=60（从65下调）：动态lb改善了低分段信号质量，60-64分不再是"死亡区间"
- IF的dm=1.0/1.0：IF逆势59%WR是利润主力，中性dm比惩罚逆势(0.8)+75%
- 全品种统一v2：v3消灭了IF/IH的逆势交易（利润来源），干净数据验证v2更优

### 信号阈值（2026-04-09更新）
- **IM阈值50**（分半+185/+186完美对称，+370pt），IC/IF/IH=60
- 阈值来源：`SYMBOL_PROFILES.signal_threshold`（IM=50, IC=60, IF=60, IH=60）
- 可通过 `--threshold` 参数覆盖回测阈值

### Monitor/Executor 职责分离（2026-03-31重构，04-02补充）
- **Monitor只负责信号生成**：评分→写`tmp/signal_pending.json`→注册shadow持仓→面板显示。不prompt，不阻塞
- **Monitor额外职责**：每5分钟bar更新时写出期货持仓到`tmp/futures_positions.json`供executor对账
- **Executor负责交易执行**：轮询JSON(1秒)→展示→确认→TQ限价单→撤单→记录
- **Executor持久TQ连接**：启动时建立TQ连接并保持，下单时零延迟（不用每次临时连接）
- **平仓信号opt-out**：60s无响应自动执行（开仓仍为opt-in：超时自动放弃）
- **激进价自动追单**：平仓限价未成交→自动以±2点激进价重新下单，不再手工确认
- **平仓否决记录**：操作者按n否决平仓→记录CLOSE_DENIED→写`tmp/denied_positions.json`→次日启动提醒
- Monitor exit触发时写CLOSE JSON供executor平仓
- shadow trades用期货价格记录entry/exit/PnL（不是现货）
- **SHADOW面板浮盈基准**：用期货last_price（与entry_price一致），fallback现货close
- **CLOSE信号timestamp**：用`datetime.now()`（北京时间），与OPEN信号一致（不用utcnow）
- **两进程完全独立**：Monitor不知道executor做了什么，executor不读position_mgr。通过JSON文件单向通信。executor自己跟TQ对账，自己判断是否执行信号
- **position_mgr角色**：纯占位计数器（控制can_open/max_total_lots），不做实际止损/PnL。实际持仓管理由shadow系统（monitor端）和positions字典（executor端）分别负责

### Monitor 盘中重启与信号系统Bug修复（2026-04-02~04-03）
- **prompted_bars去重（2026-04-03）**：改用bar_data时间戳而非`_last_bar_time`（per-symbol不同步）去重，修复信号重复发送2次给executor的bug
- **remove_by_symbol清理lock_pairs（2026-04-03）**：退出后清理lock_pairs，修复IM ME退出后无法再开仓的bug（position_mgr孤立entry）
- **非可交易品种不占position_mgr槽位（2026-04-02）**：IF/IH面板显示但不占tradeable槽位，修复position_mgr被非实盘品种占满导致IM/IC无法开仓
- **孤立持仓清理**：启动恢复时如发现position_mgr中有孤立entry（无对应shadow），自动清理

### Monitor 盘中重启状态恢复（2026-04-01修复）
- **问题**：重启后position_mgr.positions清空，`_total_net_lots()=0`，突破max_total_lots=2限制
- **解决**：shadow持仓持久化到`tmp/shadow_state.json`，启动时恢复三层状态：
  1. `_shadow_positions`（活跃shadow持仓）← shadow_state.json（当天）
  2. `position_mgr`占位（inject_position）← 使can_open/total_net_lots正确拦截
  3. `daily_trades` + `risk_mgr`计数 ← shadow_trades表（当天已平仓记录）
- shadow持仓变更时自动写JSON，exit时同步调用`remove_by_symbol()`释放占位

### 仓位管理（2026-03-28决定）
- **Fixed Risk 0.5%**：每笔最大亏损 = 账户权益 × 0.5%
- 手数三级递减：monitor建议N手 → executor减半N/2手 → **实际执行1手**（`EXEC_MAX_LOTS=1`实盘验证期）
- 品种乘数：`CONTRACT_MULT = {IF:300, IH:300, IM:200, IC:200}`
- 安全规则：单日最多10单、亏损超1%停止开仓、信号>5分钟过期不执行

### Executor 信号记录（2026-03-31新增）
- `executor_log`表：每个收到的信号完整记录生命周期
- 字段：signal_time/receive_time/operator_response(Y/N/TIMEOUT/EXPIRED)/order_status/filled_lots/cancel_time
- signal_json保留原始JSON留底
- 撤单机制：开仓60秒/平仓30秒未成交自动撤单，平仓未成交自动以激进价追单
- 持仓追踪：executor内部`_positions`字典，无持仓忽略CLOSE、重复开仓跳过
- 持仓恢复：启动时从`order_log`推断当天净持仓（两遍扫描：先OPEN再CLOSE/LOCK，不依赖timestamp排序），再用TQ实盘对账校正
- 持仓对账：每60秒读取`tmp/futures_positions.json`（monitor写出），与内部positions对账；TQ无持仓但executor有记录→自动清除
- **锁仓检查TQ对账**（2026-04-02修复）：启动时查`order_log`中未resolved的LOCK记录→TQ查实际持仓验证→TQ无锁仓自动标记`lock_resolved`→有锁仓才提示。`order_log`新增`lock_resolved`列
- **历史教训**：LOCK记录的timestamp是UTC，OPEN是北京时间，导致restore两遍扫描前先减后加算出错误净持仓。已改为两遍扫描（先OPEN再LOCK）彻底避免排序依赖

### 股指期货平今手续费（2026-03-28决定）
- 中金所股指期货（IM/IF/IH/IC）平今手续费万分之2.3，是开仓/平昨的10倍
- **日内平仓用锁仓代替**：反向开仓（SELL+OPEN 或 BUY+OPEN），不用 CLOSE/CLOSETODAY
- 次日开盘后双向平仓（平昨仓，手续费万分之0.23）
- 期权（MO/IO）不受影响，正常用 offset="CLOSE"
- order_executor 自动判断品种，锁仓记录 action="LOCK"
- 启动时检查前日锁仓持仓并提醒

### vol_monitor 订阅修复（2026-04-03）
- **问题**：持仓的行权价（如MO2605-P-6000中的6000）被加入当月订阅的行权价列表，导致跨月污染，对应日期的全部行权价订阅失败
- **修复**：订阅时按到期月份过滤，每个月份只订阅ATM附近的行权价，持仓行权价仅在匹配到期月份时才加入
- **ATM优先测试**：订阅失败时先测试ATM合约是否可连接，快速定位是网络还是行权价问题

### Morning Briefing（2026-03-29新增）
- 盘前运行 `python scripts/morning_briefing.py`，综合5维度评分输出方向Guidance
- 数据源：Tushare（A50/美股/恒生/涨跌家数/成交额）+ 本地DB（IV/VRP/价格位置）
- 输出 JSON 到 `tmp/morning_briefing.json`
- **d_override已禁用（2026-04-07）**：briefing dm累计-19.2%负贡献，现改为纯算法dm 1.1/0.9。Monitor/Backtest不再加载d_override
- 结果写入 `morning_briefing` 表 + `logs/briefing/YYYYMMDD.md`
- **本地DB优先**：`download_briefing_history.py` 预下载历史数据到7个表，briefing优先读本地（快速），Tushare作fallback
- 增量更新：`python scripts/download_briefing_history.py --update`

### 回测注意事项（重要！2026-04-01全面审计，04-04新增15m修复）
- **Lookahead fix**：`bar_5m_signal = bar_5m.iloc[:-1]`，信号评分用上一根完成bar，执行价用当前bar close（详见 `docs/lookahead_audit.md`）
- **止损精确化**：check_exit前先用bar high/low检查止损位，触发时exit_price=stop_price（不是close）
- **15分钟K线重采样对齐（2026-04-04，关键！）**：`resample('15min', label='left', closed='left')`。旧版用`label='right'`导致每根15m bar的标签时间偏移15分钟，影响全部回测结果（时段判断、持仓时间、MID_BREAK等）。修复后IM+IC改善显著
- `daily_df` 必须按回测日期截断（`trade_date < td`）
- Z-Score 也必须按日期截断计算（`trade_date < td`）
- sentiment / GARCH regime 也必须按回测日期加载（`trade_date < '{td}'`）
- **prevClose** 用 `trade_date < replay_date` 过滤，不是 `iloc[-2]`
- **bar_15m** 从 `bar_5m_signal` 构建（也排除当前bar）
- `is_open_allowed` 检查在 `update()` 中拦截，不在 `score_all` 中
- **跨日GAP不修复**：M/V/Q中的隔夜gap"失真"是有益的开盘过滤器（验证修复后IM -53%）
- **per-symbol阈值**：`effective_threshold` 从 SYMBOL_PROFILES.signal_threshold 读取（IC=65）
- **--version v2/v3/auto**：支持切换信号版本回测
- **动量lookback动态切换（2026-04-07，方案E）**：V2的`_score_momentum`根据开盘30min振幅动态选择lb=4或12（振幅<1.5%用4，否则12）。原硬编码lb=12已替换。详见"动量lookback动态切换"章节

---

## 数据库

### 文件路径（四库分离，2026-04-06）
- **主库**：`data/storage/trading.db`（~680MB）— 期货/现货K线 + 模型 + 信号/交易 + Briefing
- **期权库**：`data/storage/options_data.db`（~4.7GB）— options_daily / options_contracts / options_min
- **Tick库**：`data/storage/tick_data.db`（~28.5GB）— IM/IC/IF 主连逐笔tick
- **ETF库**：`data/storage/etf_data.db`（~56MB）— 512100/510500/510300/510050 5分钟K线
- **打开方式**：`from data.storage.db_manager import get_db; db = get_db()`
  - `get_db()` 自动创建双库 DBManager，期权表查询自动路由到 options_data.db
  - 上层代码不需要关心哪个表在哪个库

### 主要表和数据量（截至 2026-04-06）

**trading.db:**
| 表 | 行数 | 时间范围 | 说明 |
|---|---|---|---|
| futures_daily | 48,349 | 2015~04-03 | IM/IF/IH/IC 日线 |
| futures_min | 2,447,680 | 2016~04-03 | 主连 1m/5m K线 |
| index_daily | 10,932 | 2015~04-03 | 000852/000300/000016/000905 |
| index_min | 1,986,912 | 2018~04-03 | 现货 1m/5m K线（日内策略核心） |
| daily_model_output | 190 | 2025-06~04-03 | GARCH/IV/VRP/贴水/Greeks |

**options_data.db:**
| 表 | 行数 | 时间范围 | 说明 |
|---|---|---|---|
| options_daily | 764,114 | 2019~04-03 | MO/IO/HO 日线 |
| options_min | 23,880,339 | 2020~04-03 | MO/IO/HO 5分钟K线 |

**tick_data.db:**
| 表 | 行数 | 时间范围 | 说明 |
|---|---|---|---|
| futures_tick | 141,405,179 | 2016~04-03 | IM/IC/IF 逐笔tick |

**etf_data.db:**
| 表 | 行数 | 时间范围 | 说明 |
|---|---|---|---|
| etf_min | 331,872 | 2018~04-03 | 4 ETF 5分钟K线 |

### daily_model_output 字段（含最新增加的）
```
trade_date, underlying, garch_current_vol, garch_forecast_vol,
realized_vol_20d, realized_vol_60d, atm_iv, vrp, vrp_percentile, signal,
net_delta, net_gamma, net_theta, net_vega,
discount_rate_iml1, discount_rate_iml2, discount_rate_iml3,
discount_signal, recommended_contract,
garch_5d_forecast_date, rv_5d_actual, forecast_error,   ← 预测回溯
atm_iv_market,                                          ← 市场IV（期货价格based）
pnl_total, pnl_realized, pnl_unrealized,                ← P&L
pnl_delta, pnl_gamma, pnl_theta, pnl_vega, pnl_residual, ← P&L归因
iv_percentile_hist, signal_primary,                     ← IV分位主信号
garch_reliable,                                         ← GARCH可靠性（1=可靠，0=偏高已降权）
hurst_60d,                                              ← Hurst指数60日滚动（2026-04-02新增）
rr_25d                                                  ← 25D Risk Reversal（2026-04-02新增，插值法）
```

### 时间标准（重要！两套时间共存）

系统中存在两套时间标准，**混淆会导致严重bug**（如2026-04-08的回测全天M=0事件）。

| 数据源 | 时间标准 | 格式示例 | 说明 |
|--------|---------|---------|------|
| **index_min** | **UTC** | `2026-04-08 01:30:00` | TQ K线的datetime字段（纳秒时间戳转换），09:30 BJ = 01:30 UTC |
| **futures_min** | **UTC** | `2026-04-08 01:30:00` | 同上 |
| **options_min** | **UTC** | `2026-04-08 01:30:00` | 同上 |
| **futures_tick** | **UTC** | `2026-04-08 01:30:00.500000` | 同上（含微秒） |
| **etf_min** | **UTC** | `2026-04-08 01:30:00` | 同上 |
| **signal_log** | **BJ** | `2026-04-08 09:25:15` | monitor用`datetime.now()`写入 |
| **orderbook_snapshots** | **BJ** | `2026-04-08 09:25:15` | 同上 |
| **order_log** | **BJ** | `2026-04-08 10:10:00` | executor用`datetime.now()`写入 |
| **shadow_trades** | **BJ** | entry_time/exit_time均为BJ | monitor写入 |
| **trade_decisions** | **BJ** | `2026-04-08 10:10:00` | executor写入 |
| **vol_monitor_snapshots** | **BJ** | `2026-04-08 09:25:00` | vol_monitor写入 |
| **daily表**（futures_daily等） | **日期** | `20260408` | 无时分秒，不涉及时区 |
| **morning_briefing** | **日期** | `20260408` | 同上 |

**规则**：
- TQ K线/Tick数据 → 入库时统一转UTC（EOD归档中`pd.to_datetime(ns, unit="ns")`自动是UTC）
- Monitor/Executor实时写入 → `datetime.now()`即BJ时间，直接存
- 回测代码中`_get_utc_time(bar_5m)`从bar的datetime提取时间，**必须是UTC**才能和`NO_TRADE_BEFORE/AFTER`常量正确比较
- `_utc_to_bj(utc_str)`加8小时用于显示

**历史教训**：
- 2026-04-08 EOD重构引入DataDownloader，其CSV输出BJ时间但入库时未减8小时→index_min存了BJ时间→回测`_get_utc_time`判断09:30>07:00超出交易窗口→全天score_all返回None
- 批量修复index_min时误伤signal_log等BJ时间表→信号记录时间错乱

---

## 实盘账户
- 期货公司：宏源期货（broker="H宏源期货"）
- 配置在 `.env` 文件中（TQ_ACCOUNT, TQ_PASSWORD, TQ_BROKER_ID, TQ_ACCOUNT_ID, TQ_BROKER_PASSWORD）

## 当前持仓（2026-03-27 收盘后更新）
- MO2605-C-8600 空 3 手（卖Call端）
- MO2605-P-6800 空 3 手（卖Put端）
- MO2605-P-6600 空 5 手（卖Put端，3/27新开）
- 账户权益约640万，浮盈+18,640元

---

## 每日操作流程

```bash
# 推荐：一键启动全天（tmux 4窗口 + 定时EOD）
make trading-day

# 或分步执行：
make briefing                        # 盘前 Morning Briefing
make monitor                        # 盘中日内信号
make vol                            # 盘中波动率监控
make quadrant                       # 盘中象限监控
make executor                       # 半自动下单
make eod                            # 收盘后 EOD
make backtest                       # 回测今天

# 其他
python scripts/portfolio_analysis.py # 持仓分析
streamlit run dashboard/app.py       # Dashboard
python scripts/daily_record.py model --date YYYYMMDD  # 补跑模型
```

---

## 项目当前状态（2026-04）

### 已完成 ✅
| 模块 | 说明 |
|---|---|
| 数据层 | Tushare/TqSdk/SQLite/统一接口/质量检查/所有下载脚本/现货分钟线归档 |
| 模型层 | GJR-GARCH / RealizedVol / ImpliedVol / VolSurface / Greeks / 统计模型 / 技术指标，638+ 测试全绿 |
| 贴水策略 | DiscountSignal / DiscountPosition（含 Put Spread 对比）/ DiscountBacktest / DiscountCaptureStrategy |
| 日内策略 | v2/v3信号系统 / 4层Z-Score过滤 / 布林带突破+平仓 / 现货数据源 / 离线回测 / 信号质量分析 / 216天全量敏感分析 / 振幅过滤器 / 动态lb |
| 半自动下单 | order_executor.py / 限价单+手工确认 / 锁仓（股指期货平今优化）/ TqBacktest验证 |
| Morning Briefing | P1(跨市场/宽度/成交额/波动率/价格) + P2(北向/融资/PCR/ETF/期货持仓) + 极端值逆向修正 |
| 象限监控 | quadrant_monitor.py / A/B/C/D判定 / 持仓匹配 / 象限切换alert |
| EOD 记录系统 | TQ 快照 / 模型输出 / GARCH 预测回溯回填 / 市场状态评估 / BlendedRV VRP |
| Portfolio 分析 | IV/Greeks/VRP/贴水/Put候选对比表（裸 Put + Spread 方案） |
| Vol Monitor | 实时波动率面板 / ATM IV / BlendedRV VRP / GARCH可靠性 / IV期限结构 / Skew / Greeks / 贴水 / 自动扫描持仓月份 |
| Dashboard | 14 页面：含 Briefing历史 / 象限监控 / 信号分析 等 |
| 运维自动化 | Makefile / run_trading_day.sh(tmux) / launchd定时任务 |
| utils | cffex_calendar.py（合约映射单一来源） |

### 骨架已建、逻辑未实现 ⏳
- `strategies/vol_arb/`、`trend_following/`、`spread_trading/`、`mean_reversion/`（文件存在，核心逻辑为空）
- `risk/risk_checker.py`、`risk/position_sizer.py`（接口已定义，逻辑未实现）
- `execution/order_manager.py`、`execution/tq_executor.py`（骨架）
- `analysis/pnl_attribution.py`、`analysis/performance.py`（骨架）
- `backtest/`（BacktestEngine 等骨架存在，未连通策略）

---

## 代码结构关键路径

```
utils/cffex_calendar.py          ← IML 映射单一真相，修改需谨慎
data/storage/schemas.py          ← 所有建表 SQL + ALTER TABLE 语句
data/storage/db_manager.py       ← DBManager，用 query_df() 查询
models/pricing/greeks.py         ← calc_all_greeks / calc_portfolio_greeks
models/volatility/garch_model.py ← GJRGARCHModel
strategies/discount_capture/     ← 完整实现的贴水策略包
strategies/intraday/             ← 日内策略（v2/v3信号 + 平仓系统 + monitor）
  A_share_momentum_signal_v2.py  ← 信号生成器V2/V3 + 平仓系统 + Z-Score过滤
  monitor.py                     ← 盘中实时信号监控 + 数据记录
  strategy.py                    ← IntradayStrategy（必须用SignalGeneratorV2）
scripts/daily_record.py          ← EOD 主流程（~1500 行）
scripts/portfolio_analysis.py    ← 持仓分析主流程（~950 行）
scripts/vol_monitor.py           ← 期权波动率实时监控面板（~1650 行）
scripts/backtest_signals_day.py  ← 日内信号离线回测（含振幅过滤）
scripts/exit_sensitivity.py      ← Exit参数敏感分析（参数化check_exit）
scripts/sensitivity_215d.py      ← 全量215天敏感分析（7参数×IM+IC）
dashboard/pages/model_diagnostics.py ← 含 GARCH 预测回测 section
```

---

## 新增字段/功能时的注意事项

1. **新增 DB 列**：在 `schemas.py` 的 `CREATE TABLE` 和 `DAILY_MODEL_OUTPUT_ALTER_SQLS`（或对应表的 ALTER 列表）都要加，`DBManager.initialize_tables()` 会自动执行 ALTER（列已存在时静默忽略）

2. **写入新列**：`upsert_dataframe()` 用 `INSERT OR REPLACE`，会覆盖整行。如果只更新个别列（如 backfill），用 `db._conn.execute("UPDATE ...")` 加 `db._conn.commit()`

3. **编辑文件前先 Read**：Edit 工具要求 old_string 精确匹配，必须先读取当前内容

4. **合约映射改动**：任何涉及 IML/当月/次月/当季/隔季的代码，统一走 `utils/cffex_calendar.py`，不要在其他地方重复实现

5. **Python 解释器**：`/opt/homebrew/Caskroom/miniforge/base/envs/quant/bin/python`

---

## 常用命令速查

### 日内信号回测
```bash
# 单日
python scripts/backtest_signals_day.py --symbol IM --date 20260324

# 多日（逗号分隔，内置汇总输出）
python scripts/backtest_signals_day.py --symbol IM --date 20260324,20260323,20260320

# 215天完整回测（2025-05-16 ~ 2026-04-03，index_min全量）
python scripts/backtest_signals_day.py --symbol IM --date 20250516-20260403
python scripts/backtest_signals_day.py --symbol IC --date 20250516-20260403

# 敏感分析（全量215天，7参数×IM+IC并排）
python scripts/sensitivity_215d.py --param P1          # 单参数
python scripts/sensitivity_215d.py --param ALL          # 全部7参数
```

### 日内监控
```bash
python -m strategies.intraday.monitor
```

### 波动率监控
```bash
python scripts/vol_monitor.py              # 实时模式
python scripts/vol_monitor.py --snapshot   # 快照模式（从数据库读）
```

### 组合分析
```bash
python scripts/portfolio_analysis.py
```

### EOD收盘记录（17:30前必须跑完）
```bash
python scripts/daily_record.py eod
```

### 现货5分钟K线补拉
```bash
python scripts/backfill_minute_bars.py --symbol 000852 --exchange SSE --bars 10000
```

### 数据库
```
data/storage/trading.db     — 主库（期货/现货/模型/信号/交易）
data/storage/options_data.db — 期权库（options_daily/contracts/min）
data/storage/tick_data.db   — Tick库（IM/IC/IF逐笔）
data/storage/etf_data.db    — ETF库（4品种5分钟）
```
