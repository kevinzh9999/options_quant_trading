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
- 单一真相来源：`utils/cffex_calendar.py`，包含 `active_im_months()` 和 `get_im_futures_prices()`
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

### 平仓信号系统（2026-03-24新增，03-26优先级调整）
- 7个优先级：EOD_CLOSE > STOP_LOSS > LUNCH_CLOSE > TRAILING_STOP > TREND_COMPLETE > MOMENTUM_EXHAUSTED > MID_BREAK > TIME_STOP
- STOP_LOSS 提升到 LUNCH_CLOSE 之前（防止午休前大亏不止损）
- TREND_COMPLETE：5分钟上轨 + 15分钟上轨 = 两个周期都到极端才平仓
- 用15分钟布林带判断趋势阶段，5分钟只用于跟踪止盈
- 动态跟踪止盈：持仓时间越长宽度越大（0.5%→1.0%）
- **MOMENTUM_EXHAUSTED最小持仓20分钟（2026-03-30）**：5-15min止出75%是过早的（54笔中36笔），加最小持仓4根K线。32天PnL +634→+735pt（+16%），breakeven滑点2.5→4.0pt
- 平仓后15分钟同方向冷却期
- 开仓窗口：09:45~11:20, 13:05~14:30（`NO_OPEN_EOD = "06:30"` = 14:30 BJ，从14:15推迟，尾盘信号WR=67-80%）

### VRP 计算（2026-03-26重构）
- **VRP = IV - Blended RV**（不再直接用 GARCH）
- Blended RV 正常 = `0.4×RV5d + 0.4×RV20d + 0.2×GARCH`
- GARCH Sanity Check：当 `GARCH > max(RV5d, RV20d) × 1.4` 时标记不可靠，降权为 `0.5×RV5d + 0.5×RV20d`
- GARCH 模型参数本身不改，只改下游使用方式
- `garch_reliable` 字段写入 `daily_model_output`

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

### 信号阈值（2026-03-26决定）
- **默认阈值 60**（从55上调），敏感性分析显示 thr=60 在所有滑点水平下优于 thr=55
- thr=60 交易更少但每笔更精准，盈亏平衡滑点更高
- 可通过 `--threshold` 参数覆盖回测阈值

### Monitor/Executor 职责分离（2026-03-31重构）
- **Monitor只负责信号生成**：评分→写`tmp/signal_pending.json`→注册shadow持仓→面板显示。不prompt，不阻塞
- **Monitor额外职责**：每5分钟bar更新时写出期货持仓到`tmp/futures_positions.json`供executor对账
- **Executor负责交易执行**：轮询JSON(1秒)→展示→确认→TQ限价单→撤单→记录
- **平仓信号opt-out**：60s无响应自动执行（开仓仍为opt-in：超时自动放弃）
- **激进价自动追单**：平仓限价未成交→自动以±2点激进价重新下单，不再手工确认
- **平仓否决记录**：操作者按n否决平仓→记录CLOSE_DENIED→写`tmp/denied_positions.json`→次日启动提醒
- Monitor exit触发时写CLOSE JSON供executor平仓
- shadow trades用期货价格记录entry/exit/PnL（不是现货）

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
- 持仓恢复：启动时从`order_log`推断当天净持仓（OPEN-CLOSE/LOCK），再用TQ实盘对账校正。解决盘中重启后positions丢失的问题
- 持仓对账：每60秒读取`tmp/futures_positions.json`（monitor写出），与内部positions对账；TQ无持仓但executor有记录→自动清除

### 股指期货平今手续费（2026-03-28决定）
- 中金所股指期货（IM/IF/IH/IC）平今手续费万分之2.3，是开仓/平昨的10倍
- **日内平仓用锁仓代替**：反向开仓（SELL+OPEN 或 BUY+OPEN），不用 CLOSE/CLOSETODAY
- 次日开盘后双向平仓（平昨仓，手续费万分之0.23）
- 期权（MO/IO）不受影响，正常用 offset="CLOSE"
- order_executor 自动判断品种，锁仓记录 action="LOCK"
- 启动时检查前日锁仓持仓并提醒

### Morning Briefing（2026-03-29新增）
- 盘前运行 `python scripts/morning_briefing.py`，综合5维度评分输出方向Guidance
- 数据源：Tushare（A50/美股/恒生/涨跌家数/成交额）+ 本地DB（IV/VRP/价格位置）
- 输出 JSON 到 `tmp/morning_briefing.json`，Monitor 启动时读取覆盖 daily_mult
- 结果写入 `morning_briefing` 表 + `logs/briefing/YYYYMMDD.md`
- **本地DB优先**：`download_briefing_history.py` 预下载历史数据到7个表，briefing优先读本地（快速），Tushare作fallback
- 增量更新：`python scripts/download_briefing_history.py --update`

### 回测注意事项（重要！）
- `backtest_signals_day.py` 的 `daily_df` 必须按回测日期截断（防止未来数据泄漏）
- Z-Score 也必须按日期截断计算
- sentiment / GARCH regime 也必须按回测日期加载（`AND trade_date <= '{td}'`）
- **prevClose** 用 `trade_date < replay_date` 过滤，不是 `iloc[-2]`
- **bar_15m** 必须传入 `score_all`（传 None 会导致 M 维度缺 15 分一致性 bonus）
- `is_open_allowed` 检查在 `update()` 中拦截，不在 `score_all` 中（否则面板重启后无法显示评分）

---

## 数据库

### 文件路径
- 主数据库：`data/storage/trading.db`（代码用 `ConfigLoader().get_db_path()`）
- 旧路径别名：`data/market_data.db`（config.example.yaml 默认值，实际不使用）
- 打开方式：`DBManager(ConfigLoader().get_db_path())`

### 主要表和数据量（截至 2026-03）
| 表 | 行数 | 时间范围 | 说明 |
|---|---|---|---|
| futures_daily | ~44,500 | 2015~2026 | IM/IF/IH/IC 日线 |
| options_daily | ~755,000 | 2019~2026 | MO/IO 日线 |
| index_daily | ~10,880 | 2015~2026 | 000852/000300/000905/000016 |
| index_min | 累积中 | 2026-03~ | 现货5分钟K线（日内信号数据源） |
| daily_model_output | 累积中 | 2026-03~ | GARCH/IV/VRP/贴水/Greeks |
| account_snapshots | 累积中 | 2026-03~ | 每日账户快照 |
| position_snapshots | 累积中 | 2026-03~ | 每日持仓快照 |

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
garch_reliable                                          ← GARCH可靠性（1=可靠，0=偏高已降权）
```

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

## 项目当前状态（2026-03）

### 已完成 ✅
| 模块 | 说明 |
|---|---|
| 数据层 | Tushare/TqSdk/SQLite/统一接口/质量检查/所有下载脚本/现货分钟线归档 |
| 模型层 | GJR-GARCH / RealizedVol / ImpliedVol / VolSurface / Greeks / 统计模型 / 技术指标，638+ 测试全绿 |
| 贴水策略 | DiscountSignal / DiscountPosition（含 Put Spread 对比）/ DiscountBacktest / DiscountCaptureStrategy |
| 日内策略 | v2/v3信号系统 / 4层Z-Score过滤 / 布林带突破+平仓 / 现货数据源 / 离线回测 / 信号质量分析 |
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
scripts/backtest_signals_day.py  ← 日内信号离线回测
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

# 30天完整回测（截至2026-03-26）
python scripts/backtest_signals_day.py --symbol IM --date 20260204,20260205,20260206,20260209,20260210,20260211,20260212,20260213,20260225,20260226,20260227,20260302,20260303,20260304,20260305,20260306,20260309,20260310,20260311,20260312,20260313,20260316,20260317,20260318,20260319,20260320,20260323,20260324,20260325,20260326
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
data/storage/trading.db (SQLite)
关键表：index_min, index_daily, futures_min, futures_daily,
       signal_log, trade_records, position_snapshots,
       volatility_history, daily_model_output, vol_monitor_snapshots
```
