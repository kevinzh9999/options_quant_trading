# 开发进度追踪

> 最后更新：2026-04-01（Executor持久连接 + Monitor重启状态恢复 + Shadow浮盈修复）

## 总体状态

| 阶段 | 状态 | 说明 |
|------|------|------|
| Step 0: 项目初始化 | ✅ 完成 | 项目骨架已创建 |
| Step 0.5: 账户持仓模块 | ✅ 完成 | AccountManager（宏源期货）|
| Step 1: 多策略框架重构 | ✅ 完成 | strategies/ + models/ 子包 + backtest/ |
| Step 2: 数据层实现 | ✅ 完成 | 存储/Tushare/TqSdk/统一接口/质量检查/下载脚本 |
| Step 3: 模型层实现 | ✅ 完成 | 638 测试全绿（含 smoke test） |
| Step 4: 策略层实现 | 🔄 部分完成 | discount_capture + vol_arb + trend_following + intraday（v2/v3信号+平仓+回测） |
| Step 5: 风控层实现 | ⏳ 待开发 | |
| Step 6: 执行层实现 | 🔄 部分完成 | order_executor半自动下单+持仓恢复+对账+锁仓 |
| Step 7: 分析层实现 | ⏳ 待开发 | |
| Step 8: Dashboard 实现 | ✅ 完成 | 14页面（含Briefing/象限/信号分析）|
| Step 9: 回测框架实现 | ✅ 完成 | DataFeed + SimBroker + BacktestEngine + BacktestReport，52测试全绿 |
| Step 10: 实盘部署 | ⏳ 待开发 | |

---

## Step 0: 项目初始化 ✅

- [x] 目录结构创建
- [x] config/ - 配置管理骨架
- [x] data/ - 数据层骨架（Tushare/天勤/SQLite/统一接口）
- [x] models/ - 模型层骨架（RV/GARCH/IV/波动率曲面/Greeks）
- [x] signals/ - 信号层骨架（VRP信号/信号数据结构）
- [x] risk/ - 风控层骨架（风控检查/仓位计算）
- [x] execution/ - 执行层骨架（订单管理/天勤执行）
- [x] analysis/ - 分析层骨架（P&L归因/绩效统计/模型诊断）
- [x] dashboard/ - Dashboard骨架（Streamlit + 4个页面）
- [x] tests/ - 测试骨架（4个测试文件）
- [x] requirements.txt / .gitignore / README.md
- [x] run_daily.py 主脚本框架

---

## Step 0.5: 账户持仓模块 ✅

### data/sources/account_manager.py
- [x] `get_account_summary()` - 账户资金概况（权益/可用/保证金/风险度）
- [x] `get_all_positions()` - 全量持仓列表（期货 + 期权，按方向展开）
- [x] `get_option_positions()` - 期权持仓（含行权价/到期日/认购认沽/标的）
- [x] `get_futures_positions()` - 期货持仓
- [ ] `get_position_greeks()` - 组合 Greeks 汇总（依赖 models/greeks，待实现）
- [x] `get_margin_detail()` - 各持仓保证金明细
- [x] `is_account_ready()` - 账户连接状态检查（NaN/异常处理）
- [x] `parse_option_symbol()` - 天勤期权代码解析（正则，支持 CFFEX.IO2406-C-3800）

---

## Step 2: 数据层实现 ✅

### 子模块完成情况

| 子模块 | 文件 | 状态 | 测试数 |
|--------|------|------|--------|
| SQLite 存储层 | data/storage/db_manager.py | ✅ | 37 |
| 表结构定义 | data/storage/schemas.py | ✅ | - |
| Tushare 客户端 | data/sources/tushare_client.py | ✅ | 43 |
| 天勤 TqSdk 客户端 | data/sources/tq_client.py | ✅ | 38 |
| 账户持仓管理 | data/sources/account_manager.py | ✅ | 37 |
| 统一数据接口 | data/unified_api.py | ✅ | 76 |
| 数据质量检查 | data/quality_check.py | ✅ | 59 |
| 股指期货日线下载 | data/download_scripts/download_futures_daily.py | ✅ | - |
| 期货分钟线下载 | data/download_scripts/download_futures_min.py | ✅ | - |
| 期权日线下载 | data/download_scripts/download_options_daily.py | ✅ | - |
| 商品期货日线下载 | data/download_scripts/download_commodity_daily.py | ✅ | - |

### 核心设计决策

- **本地优先策略**：DB → Tushare fallback（auto_download=True）→ TqSdk（分钟线）
- **幂等 upsert**：所有写入使用 `INSERT OR REPLACE`，支持重复运行
- **增量更新**：`get_latest_date(table, ts_code)` 检测尾部缺口，仅补充缺失数据
- **symbol 归一化**：`_normalize_symbol()` 支持 TqSdk/Tushare/裸合约/品种代码四种格式
- **质量检查 dict 接口**：`check_futures_daily(df, ts_code) -> Dict`，保留 `QualityReport` 向后兼容

### 已知限制

- **分钟线需要付费权限**：Tushare `ft_mins` 接口需额外积分，免费账户无法使用
- **TqSdk 实盘依赖**：`tq_client.py` 需安装 tqsdk 且有有效天勤账号，测试全部 mock
- **期权历史数据深度**：Tushare `opt_daily` 接口仅支持 CFFEX 中金所股指期权（IO/MO），
  上交所 ETF 期权（510050）需单独处理
- **`get_position_greeks()`**：AccountManager 中该方法依赖 `models/greeks.py`，
  尚未实现，调用会抛 `NotImplementedError`

---

## Step 1: 多策略框架重构 ✅

### 新增目录结构

```
strategies/
  base.py              ✅ Signal、StrategyConfig、BaseStrategy ABC
  registry.py          ✅ StrategyRegistry（动态实例化）
  vol_arb/             ✅ 波动率套利策略（4个文件）
  trend_following/     ✅ 趋势跟踪策略（4个文件）
  spread_trading/      ✅ 价差套利策略（4个文件）
  mean_reversion/      ✅ 均值回归策略（3个文件）

models/
  volatility/          ✅ realized_vol + garch_model + vol_forecast
  pricing/             ✅ implied_vol + vol_surface + greeks
  statistics/          ✅ cointegration + ou_process + regime_detection
  indicators/          ✅ trend + momentum + volatility_ind + volume
  （原 models/*.py 保留，作为向后兼容的重新导出）

backtest/
  engine.py            ✅ BacktestEngine 主控
  data_feed.py         ✅ DataFeed 数据源
  broker.py            ✅ SimulatedBroker 模拟撮合
  report.py            ✅ BacktestReport 绩效报告

risk/
  portfolio_risk.py    ✅ PortfolioRiskManager 组合级风控

analysis/
  strategy_comparison.py ✅ 多策略绩效对比

data/download_scripts/
  download_commodity_daily.py ✅ 商品期货日线下载

data/storage/schemas.py
  新增 4 个表：          ✅ commodity_daily / strategy_signals / strategy_trades / strategy_pnl

dashboard/pages/
  futures_monitor.py   ✅ 期货行情监控（骨架）
  portfolio.py         ✅ 多策略组合总览（骨架）

tests/
  test_models/         ✅ 模型层测试目录
  test_strategies/     ✅ 策略层测试（含 test_base_strategy.py）
  test_data/           ✅ 数据层测试目录
  test_risk/           ✅ 风控层测试目录
```

### 配置更新
- [x] config.example.yaml 新增 strategies + portfolio_risk 配置段
- [x] requirements.txt 新增 statsmodels + hmmlearn
- [x] run_daily.py 更新为多策略调度框架
- [x] run_backtest.py 新增回测入口脚本
- [x] execution/order_manager.py 更新 VRPSignal 导入路径
- [x] signals/ 保留为向后兼容的重新导出

---

## Step 2: 数据层实现 ⏳

### data/storage/db_manager.py
- [ ] `upsert_df()` - DataFrame 批量写入
- [ ] `upsert_rows()` - 字典列表批量写入
- [ ] `query_df()` - SELECT 查询返回 DataFrame
- [ ] `query_scalar()` - 标量查询
- [ ] `get_max_date()` - 获取最大日期（用于增量更新）

### data/sources/tushare_client.py
- [ ] `_get_api()` - 初始化 tushare pro_api
- [ ] `get_futures_daily()` - 期货日线
- [ ] `get_futures_min()` - 期货分钟线
- [ ] `get_futures_contracts()` - 期货合约信息
- [ ] `get_options_daily()` - 期权日线
- [ ] `get_options_contracts()` - 期权合约信息
- [ ] `get_trade_calendar()` - 交易日历
- [ ] `_call_with_retry()` - 带重试的 API 调用

### data/sources/tq_client.py
- [ ] `connect()` / `disconnect()` - 连接管理
- [ ] `get_kline_serial()` - 历史 K 线
- [ ] `get_quote()` - 实时行情
- [ ] `get_account_info()` - 账户信息
- [ ] `get_positions()` - 持仓查询

### data/unified_api.py
- [ ] `get_futures_daily()` - 统一期货日线查询
- [ ] `get_futures_min()` - 统一期货分钟线查询
- [ ] `get_continuous_futures()` - 连续合约拼接
- [ ] `get_options_daily()` - 统一期权日线查询
- [ ] `get_trade_dates()` - 交易日历查询
- [ ] `tushare_to_tq_symbol()` - 代码转换

### data/quality_check.py
- [ ] `check_futures_daily()` - 期货日线质量检查
- [ ] `check_options_daily()` - 期权日线质量检查
- [ ] `check_missing_dates()` - 缺失日期检查
- [ ] `check_null_values()` - 空值统计
- [ ] `check_price_outliers()` - 异常值检测

### data/download_scripts/
- [ ] `download_futures_daily.py` - 期货日线下载
- [ ] `download_options_daily.py` - 期权日线下载
- [ ] `download_futures_min.py` - 期货分钟线下载
- [ ] `download_commodity_daily.py` - 商品期货日线下载

---

## Step 3: 模型层实现 ⏳

### models/volatility/ ✅
- [x] `realized_vol.py` - RealizedVolCalculator + compute_realized_vol / compute_rolling_rv（25 测试）
- [x] `garch_model.py` - GARCHModel + GJRGARCHModel（49 测试）
- [x] `vol_forecast.py` - VolForecaster（HAR-RV + GARCH + Ensemble，37 测试）

### models/pricing/ ✅
- [x] `implied_vol.py` - bs_price / calc_implied_vol（Newton/Bisect/Brent）
- [x] `vol_surface.py` - VolSmile / VolSurface.build_from_options_df
- [x] `greeks.py` - calc_all_greeks / calc_portfolio_greeks
- 共 84 测试

### models/statistics/ ✅
- [x] `cointegration.py` - cointegration_test（EG/Johansen）/ estimate_hedge_ratio / rolling_cointegration
- [x] `ou_process.py` - fit_ou_process / ou_half_life / simulate_ou
- [x] `regime_detection.py` - HMMRegimeDetector.fit/predict/get_state_statistics
- 共 58 测试

### models/indicators/ ✅
- [x] `trend.py` - calc_sma/ema/macd/adx + `TrendIndicators`（含 donchian_channel）
- [x] `momentum.py` - calc_rsi/roc/stochastic + `MomentumIndicators`（含 momentum_factor）
- [x] `volatility_ind.py` - calc_atr/bollinger_bands/historical_vol + `VolatilityIndicators`（含 keltner_channel）
- [x] `volume.py` - calc_obv/vwap
- 共 52+47+71=170 测试

### models/volatility/vol_forecast.py ✅ 新增
- [x] `VolForecast` - 统一波动率预测接口（ewma/garch/har 三种方法）
- [x] `ewma_forecast` - RiskMetrics EWMA（λ=0.94）

### tests/test_models/test_smoke.py ✅ 新增
- 端到端 smoke test：IF日线→RV→GARCH预测 / IO期权→IV→波动率曲面→ATM IV→VRP
- DB 有数据时执行完整链路验证，无数据时自动跳过

---

## Step 4: 策略层实现 🔄

### strategies/discount_capture/ ✅ 新增（2026-03-17）

| 文件 | 状态 | 说明 |
|------|------|------|
| `signal.py` | ✅ | DiscountSignal: calculate_discount / get_discount_history / get_discount_percentile / generate_signal |
| `position.py` | ✅ | DiscountPosition: calculate_futures_lots / select_protective_put / calculate_strategy_pnl_scenarios |
| `strategy.py` | ✅ | DiscountCaptureStrategy(BaseStrategy): generate_signals / on_fill / calculate_positions |
| `backtest.py` | ✅ | DiscountBacktest: 月度入场 / 持有至近期 / P&L 统计 / print_report |
| `gamma_scalper.py` | ✅ | GammaScalper: delta/price/time 三种对冲模式，RehedgeRecord 记录 |
| `gamma_strategy.py` | ✅ | DiscountGammaStrategy: select_best_futures_contract / select_optimal_put / calc_initial_put_lots |
| `gamma_backtest.py` | ✅ | DiscountGammaBacktester: run_daily_only / run_with_scalping / 敏感性分析 / CLI |
| `__init__.py` | ✅ | 包导出 |

#### Gamma Scalping 回测结果（2026-03-19）
- **日线级回测** (2022-07 ~ 2026-03): 贴水+522K, Put成本-295K, 总收益-4.0%
- **含Gamma Scalping** (2023-08 ~ 2026-03): 总收益+8.3%, 年化+3.1%, Sharpe 1.17
- **Gamma/Theta覆盖率: 106%**（Gamma利润完全覆盖Theta成本，Put保护免费）
- **最优对冲阈值: 0.2%** 价格变动，日均对冲~4次

#### 核心逻辑
- **贴水率计算**：`(futures_price - spot_price) / spot_price * (365 / DTE)`，正值=贴水
- **信号阈值**：年化贴水率 > 15% STRONG / 10-15% MEDIUM / 5-10% WEAK
- **Put 选择**：按 delta 或行权价距离匹配目标 OTM Put，计算最大亏损和保护比率
- **历史百分位**：用 IML1/2/3 历史 raw_discount_rate 分布计算当前百分位

#### 集成点
- `scripts/portfolio_analysis.py`：在 print_report() 末尾调用 print_discount_section()
- `scripts/daily_record.py`：_eod_model_output() 计算贴水率并写入 daily_model_output 新列
- `dashboard/pages/discount_monitor.py`：7th dashboard 页面（KPI/历史走势/情景P&L/Put对比/回测）
- `data/storage/schemas.py`：daily_model_output 增加 5 列 + ALTER TABLE 语句
- `config/config.example.yaml`：新增 discount_capture 策略配置段

### strategies/vol_arb/ ✅ 新增（2026-03-18）
- [x] `signal.py` - VRPSignalGenerator: fit_garch / compute_vrp / select_strangle_legs
- [x] `strategy.py` - VolArbStrategy: generate_signals（SELL_VOL strangle）
- [x] `backtest.py` - run_vol_arb_backtest CLI 入口

### strategies/trend_following/ ✅ 新增（2026-03-18）
- [x] `signal.py` - TrendSignalGenerator: compute_indicators / get_signal（SMA/Donchian/ADX/ATR止损）
- [x] `strategy.py` - TrendFollowingStrategy: generate_signals（波动率目标仓位）
- [x] `position.py` - TrendPositionSizer: calculate_lots / calculate_stop_loss
- [x] `backtest.py` - run_trend_backtest CLI 入口

### strategies/intraday/ ✅ 增强（2026-03-24~26）

| 文件 | 状态 | 说明 |
|------|------|------|
| `A_share_momentum_signal_v2.py` | ✅ | v2统一评分(M50/V30/Q20) + v3品种差异化 + 4层Z-Score过滤 + 方向感知intraday_filter + 多周期布林带平仓系统 + RSI回归确认 + 期权情绪乘数 |
| `signal.py` | ⚠ 废弃 | 旧版IntradaySignalGenerator，不再使用（strategy.py必须用V2） |
| `strategy.py` | ✅ | IntradayStrategy，使用SignalGeneratorV2，传入zscore/is_high_vol/sentiment |
| `monitor.py` | ✅ | 盘中实时监控 + 信号评估 + DB记录(signal_log/orderbook_snapshots) |
| `risk.py` | ✅ | 日内风控（最大持仓/日亏损限制） |
| `position.py` | ✅ | 仓位管理（固定手数/ATR动态） |
| `backtest.py` | ✅ | 日内策略回测框架 |

#### 评分公式
- 三维度：动量(50) + 波动率(30) + 成交量(20) = raw_total
- 乘数链：raw × daily_mult(d) × intraday_filter × time_weight(t) × sentiment_mult(s)
- 后处理：Z-Score硬过滤 → RSI回归bonus → clamp(0,100)
- 信号阈值：score ≥ 60 触发

### strategies/spread_trading/
- [ ] `pairs.py` - PairsManager.update / _test_cointegration
- [ ] `signal.py` - SpreadSignalGenerator.generate
- [ ] `strategy.py` - SpreadTradingStrategy.generate_signals
- [ ] `backtest.py` - SpreadTradingBacktester.run

### strategies/mean_reversion/
- [ ] `signal.py` - MeanReversionSignalGenerator.generate
- [ ] `strategy.py` - MeanReversionStrategy.generate_signals
- [ ] `backtest.py` - MeanReversionBacktester.run

---

## Step 5: 风控层实现 ⏳

### risk/risk_checker.py
- [ ] `run_all_checks()` - 全量风控检查
- [ ] `check_margin()` / `check_daily_loss()` - 基础检查
- [ ] `check_delta_exposure()` / `check_vega_exposure()` - Greeks 敞口
- [ ] `check_liquidity()` - 流动性检查

### risk/position_sizer.py
- [ ] `calc_position_size()` - 仓位计算主入口
- [ ] `_fixed_vega_size()` - 固定 Vega 法
- [ ] `_fixed_lots_size()` - 固定手数法

### risk/portfolio_risk.py
- [ ] `check_portfolio_risk()` - 组合级风险检查
- [ ] `calc_strategy_correlations()` - 策略相关性
- [ ] `calc_portfolio_var()` - 组合 VaR

---

## Step 6: 执行层实现 ⏳

### execution/order_manager.py
- [ ] `signal_to_orders()` - 信号转订单
- [ ] `close_group()` / `roll_group()` - 平仓/滚仓
- [ ] `update_order_status()` - 状态更新

### execution/tq_executor.py
- [ ] `submit_order()` / `wait_for_fill()` / `cancel_order()`
- [ ] `reconcile_positions()` - 持仓对账

---

## Step 7: 分析层实现 ⏳

### analysis/
- [ ] `pnl_attribution.py` - attribute_daily_pnl
- [ ] `performance.py` - PerformanceAnalyzer.analyze
- [ ] `model_diagnostics.py` - ModelDiagnostics.diagnose_garch
- [ ] `strategy_comparison.py` - compare_strategies / analyze_regime_performance

---

## Step 8: Dashboard 实现 ✅

- [x] `dashboard/app.py` - 主入口，侧边栏导航，6页面动态加载
- [x] `dashboard/pages/portfolio.py` - 组合总览（KPI卡片/净值曲线/持仓表/Greeks摘要/近期成交）
- [x] `dashboard/pages/vol_monitor.py` - 波动率监控（RV vs IV/GARCH条件波动率/期限结构/微笑）
- [x] `dashboard/pages/greeks_dashboard.py` - Greeks 分析（情景分析/历史走势/持仓明细）
- [x] `dashboard/pages/futures_monitor.py` - 期货监控（归一化价格/基差/跨品种Z-score/RV对比）
- [x] `dashboard/pages/performance_dashboard.py` - 策略绩效（累计收益/日盈亏/回撤/月度统计）
- [x] `dashboard/pages/model_diagnostics.py` - 模型诊断（GARCH参数/残差分布/QQ图/诊断检验）

---

## Step 9: 回测框架实现 ✅（2026-03-18）

- [x] `backtest/data_feed.py` - DataFeed: preload / get_history（严格无前视偏差）/ get_options_chain
- [x] `backtest/broker.py` - SimBroker: open/close position / margin check / mark-to-market / PnL
- [x] `backtest/engine.py` - BacktestEngine: event-driven day loop / signal execution / progress reporting
- [x] `backtest/report.py` - BacktestReport: metrics / equity curve / monthly returns / print_report / save_to_csv
- [x] `run_backtest.py` - 统一 CLI（--strategy / --start / --end / --capital / --list / --param）
- [x] Tests: 12 broker + 6 engine + 11 report + 24 trend = 52 tests，全部通过

---

## 期权波动率监控面板（2026-03-19）

- [x] `scripts/vol_monitor.py` - 独立期权波动率监控面板
  - [x] 盘中实时模式：TQ 订阅 MO 期权链 + IM 期货，每5分钟刷新
  - [x] snapshot 模式：从数据库读取最新数据，输出一次
  - [x] 波动率概览：ATM IV / RV / VRP / GARCH / IV分位 / VRP分位
  - [x] IV期限结构：多到期月 ATM IV + 隐含Forward + 形态判断
  - [x] IV Skew：行权价级别 IV + Delta + 25D RR/BF
  - [x] 持仓Greeks实时更新：Delta/Gamma/Theta/Vega
  - [x] 贴水监控：现货 vs 期货 + 年化贴水率
  - [x] 综合信号：VRP/期限结构/Skew/持仓安全距离
  - [x] 数据持久化：vol_monitor_snapshots 表 + Markdown 日志
- [x] `data/storage/schemas.py` - 新增 vol_monitor_snapshots 表 + daily_model_output 扩展字段

## 日内策略增强（2026-03-19）

- [x] 布林带多周期评分维度（12分）：5m/15m/日线三级布林带
- [x] 突破"记忆"：曾突破开盘区间但回归时仍给部分分
- [x] Monitor 增强面板：VWAP/偏离%/BOLL状态/趋势箭头/维度明细
- [x] 日线数据加载：monitor 启动时从 DB 加载，修复日线维度永远0分
- [x] VWAP 趋势跟随阈值优化

---

## 2026-03-24：日内策略重构 + VRP双轨制 + 平仓系统

### 平仓信号系统
- [x] 7优先级退出系统：EOD > StopLoss > TrailingStop > TrendComplete > MomentumExhausted > MidBreak > TimeStop
- [x] 多周期布林带退出：5m+15m 双确认（TrendComplete 要求两个周期都到极端）
- [x] 动态跟踪止盈：持仓时间越长宽度越大（0.5%→1.0%），15m趋势确认+0.2%
- [x] 动量耗尽检测：3根K线窄幅震荡 + 15m极端 + 非趋势中
- [x] 平仓后15分钟同方向冷却期

### 信号系统改进
- [x] Z-Score 4层过滤：第一层高波动判断 → 第二层Z-Score硬过滤 → 第三层日内涨跌幅 → 第四层RSI回归确认
- [x] 方向感知 intraday_filter：顺势不打折，逆势重打折（高波动下2%是正常波动）
- [x] VRP 双轨制：IV分位为主信号（不依赖GARCH），VRP降为辅助参考
- [x] 离线信号回测工具 `backtest_signals_day.py`

### Vol Monitor / Dashboard 修复
- [x] vol_monitor 数据源修复（从 volatility_history 读取完整886天历史）
- [x] Dashboard 波动率页面单位转换修复（百分比 vs 小数）
- [x] 持仓月份合约自动订阅（确保 Greeks 准确）
- [x] IV surge 预警（基于日内现货涨跌幅）

---

## 2026-03-25：信号架构统一 + 全面切换现货数据源

### 信号架构统一
- [x] `strategy.py` 强制使用 `SignalGeneratorV2`（废弃旧版 `IntradaySignalGenerator`）
- [x] strategy层传入 zscore / is_high_vol / sentiment 参数（与monitor一致）
- [x] intraday_filter 改用 昨日现货收盘价（含跳空gap），不用今日开盘价
- [x] 时段权重改为品种专属 `session_multiplier`（从 SYMBOL_PROFILES 读取）

### 全面切换现货指数
- [x] 所有日内信号计算统一用现货指数（IM→000852, IF→000300, IH→000016, IC→000905）
- [x] 新增 `index_min` 表，归档现货5分钟K线
- [x] 跨日比较（prev_close, daily_mult）→ 现货日线（无换月跳变）
- [x] 日内技术指标（BOLL/RSI/MA）→ 现货分钟K线
- [x] monitor 5分钟K线时间戳对齐到标准时点

### 回测修复
- [x] `backtest_signals_day.py` 的 daily_df 按回测日期截断（防止未来数据泄漏）
- [x] Z-Score 按日期截断计算
- [x] 高波动区间 intraday_filter 阈值放宽（2-3%逆势 0.5 而非 0.3）

---

## 2026-03-26：GARCH Blended RV + Vol Monitor 时间对齐

### VRP 计算重构
- [x] VRP = IV - Blended RV（替换原 IV - GARCH）
- [x] Blended RV = 0.4×RV5d + 0.4×RV20d + 0.2×GARCH（GARCH可靠时）
- [x] GARCH Sanity Check：GARCH > max(RV5d,RV20d)×1.4 时降权为 0.5×RV5d + 0.5×RV20d
- [x] `garch_reliable` 字段写入 daily_model_output
- [x] daily_record.py 新增 rv_5d 计算，VRP 改用 blended RV
- [x] vol_monitor 面板：显示 Blended RV 行 + GARCH 可靠性标签 + VRP 信号降级
- [x] 下游 sentiment_mult 自动获得修正后 VRP（从 vol_monitor_snapshots 读取）

### 情绪乘数修正
- [x] 移除 VRP 对日内信号的 -0.10 惩罚（VRP是波动率指标，与日内方向无关）

### 回测关键bug修复（3个导致回测与live不一致的bug）
- [x] **bar_15m 传 None**：回测的 `score_all` 未传入15分钟K线，导致M维度缺少15m一致性bonus（+15分）
- [x] **prevClose = iloc[-2]**：当天日线未入库时取到前天收盘价，导致 intraday_return 和 d 乘数严重偏差
- [x] prevClose 改为 `trade_date < 当天` 的最后一行（信号层和回测层同步修复）
- [x] 回测 sentiment / GARCH regime 按回测日期加载（修复未来数据泄漏）
- [x] `is_open_allowed` 从 `score_all` 移到 `update`（面板重启后仍能显示评分）

### 布林带突破信号维度
- [x] `_score_boll_breakout()`: 5m中轨突破+10, 放量+2, 窄带+3, 15m确认+5（最高20分）
- [x] 前置条件：`s_mom > 0`（动量已确认方向才加分，防止假突破）
- [x] v2 和 v3 同步支持，面板和回测显示 `B{n}` 标注

### 30天回测最终结果
```
总盈亏: +373.7pt = +74,744元/手
胜率(天): 70%  交易: 113笔  笔胜率: 49%
盈亏比: 1.48  均盈+23.5pt  均亏-15.8pt
LONG: 61笔 +252.8pt  SHORT: 52笔 +120.9pt
```

### Monitor 盘口 + 阈值 + 滑点分析
- [x] Monitor 信号触发显示盘口价格和建议限价（bid/ask/last + 排队价/吃盘口价）
- [x] 默认信号阈值 55→60（敏感性分析：thr=60 在 slip=2 时仍 +112pt，thr=55 仅 +18pt）
- [x] 回测滑点模拟：`--slippage` 参数 + `--sensitivity` 九宫格 + breakeven 分析
- [x] 开仓窗口 09:35→09:45（避开开盘噪音）
- [x] 止损优先级提升到午休强平之前（防止 3/04 类 -105pt 大亏）
- [x] NO_OPEN_EOD 14:40→14:15（至少留45分钟给交易，14:20开仓到EOD强平只有25分钟）
- [x] 回测 sentiment 参数对齐检查：term_structure_shape 列为空已记入 TODO

### Vol Monitor 改进
- [x] 启动时对齐到整5分钟时点（等待到下一个 :00/:05/:10/...）
- [x] 刷新逻辑改为检查 minute%5==0（不再用启动偏移的300秒间隔）
- [x] RV趋势行：5d vs 20d（波动率在回落/上升）
- [x] GARCH 可靠性标签 + VRP 不可靠时信号降级

---

## 2026-03-27：Monitor 修复 + Vol Monitor 持仓月份扫描

- [x] Monitor 同一根K线信号去重（`_prompted_bars` set，防止5分钟内重复触发4次）
- [x] 确认超时 10s→30s
- [x] vol_monitor 自动扫描 TQ 实时持仓月份，展开完整行权价链 + 对应 IM 期货订阅
- [x] V=0 波动率过滤验证需求记入 TODO（待收盘后回测验证）

---

## 2026-03-28：象限监控 + 信号质量分析 + 逆势惩罚加强

### 策略象限监控面板
- [x] `scripts/quadrant_monitor.py`：四象限判定（A/B/C/D + B1/B2子状态）
- [x] 持仓匹配评估 + 象限切换 alert + 贴水/账户显示
- [x] `--once` 单次模式 + 持续监控模式

### 信号质量分析
- [x] `scripts/signal_quality_analysis.py`：V-Score / Score Band / Daily Mult 三维度分析
- [x] backtest_signals_day.py 记录 entry_score/entry_v_score/entry_daily_mult/entry_raw_total
- [x] 关键发现：逆势做空(S+d=0.7) 0%胜率 avg=-11.8pt

### 逆势惩罚加强
- [x] `_daily_direction_multiplier` 逆势惩罚从 0.7 → 0.5
- [x] 效果：d=0.7交易从13笔→0笔，总PnL +412→+460pt，breakeven滑点 2.1→3.0pt
- [x] SHORT胜率从51%→56%

### Position Sizing 研究 + 实施
- [x] `scripts/position_sizing_research.py`：5种方法对比（Fixed/FixedRisk/Kelly/VolAdj/ScoreWeight）
- [x] 结论：Fixed Risk 0.5% 最实用（PnL 3倍，MaxDD 0.82%，Sharpe 6.32）
- [x] Half Kelly 加 max_lots=10 限制（原始94手太激进）
- [x] Monitor 信号触发显示建议手数（Fixed Risk 0.5%，从 account_snapshots 读权益）

### 半自动下单系统
- [x] `scripts/order_executor.py`：信号驱动的半自动下单（限价单+手工确认+60秒超时）
- [x] Monitor 写入 `/tmp/signal_pending.json`，order_executor 监控并展示订单
- [x] 开仓超时自动放弃，止损超时持续警告
- [x] 实际手数 = 建议手数 ÷ 2，安全规则（10单/日、1%亏损上限）
- [x] `order_log` 表记录所有下单（含 SKIPPED/TIMEOUT_SKIP）

### 文档更新
- [x] DAILY_ROUTINE.md 全面更新（象限监控+阈值60+时间窗口）
- [x] TODO.md 标记已完成项 + 新增策略切换提示需求

---

## 2026-03-29：Morning Briefing P1

- [x] `scripts/morning_briefing.py`：五维度综合评分（跨市场/市场宽度/成交额/波动率/价格位置）
- [x] Tushare API：A50(XIN9) / SPX / IXIC / HSI + 全A股涨跌家数 + 上证成交额
- [x] 本地DB：ATM IV / VRP / IV分位 / 期限结构 / 20日价格位置
- [x] 输出：方向(偏多/偏空/中性) + 置信度(1-5) + d_override建议
- [x] 持久化：morning_briefing表 + /tmp/morning_briefing.json + logs/briefing/
- [x] 极端值逆向修正（score<-40+涨跌比<0.3 → 置信度封顶3星）
- [x] **P2指标**：北向资金(moneyflow_hsgt) + 融资余额(margin) + PCR(options_daily) + ETF份额(fund_share)
- [x] fut_holding 接口无数据（暂不可用），其余P2全部正常
- [x] 评分范围扩展至±130，置信度阈值调整（50/35/20/10）
- [x] fut_holding 修复（全量获取CFFEX+本地过滤IM），3/26: 净空42992手
- [x] 成交额修复（沪+深合计 + 千元→亿元 + 20日区间位置）
- [x] Dashboard新增3页面：Briefing历史 + 象限监控 + 信号分析
- [x] 一键启动脚本 `run_trading_day.sh`（tmux 4窗口 + 盘前/盘后自动化）
- [x] Makefile 快捷命令（make briefing/monitor/eod/backtest/trading-day）
- [x] launchd 定时任务（8:55 briefing + 18:30 delayed EOD）
- [x] `scripts/download_briefing_history.py`：7个历史数据表批量下载（全量/增量/--skip）
- [x] morning_briefing 优先从本地DB读取，Tushare作为fallback（运行更快）
- [x] schemas.py 新增7表：global_index_daily / market_breadth / market_turnover / northbound_flow / margin_data / etf_share / fut_holding_summary

### 日内信号系统优化（2026-03-30）

- [x] 逆势惩罚不对称化：做多d=0.8/做空d=0.5（被压制117做多信号WR=56%→释放，64做空WR=42%→保持拦截）
- [x] `_intraday_filter_mild` 顺势/逆势方向修正（旧版搞反：顺势追跌f=0.3，逆势抄底f=1.3）
- [x] Z-Score顺势不阻断：已验证，revert（slip=0时+776pt优于+735pt，但breakeven从4.0pt恶化到3.0pt，实盘滑点下更差）
- [x] MOMENTUM_EXHAUSTED最小持仓20分钟（54笔中36笔在5-15min过早止出，过早率70-75%）
- [x] 回测面板dm/f列拆分（d列原来显示intraday_filter误导，改为dm=daily_mult，f=intraday_filter）
- [x] signal_quality_analysis扩展：跳空追势/反弹距离/趋势反转/Direction×DailyMult/IntradayFilter分支验证
- [x] intraday_filter对称化/跳空惩罚/IV恐慌环境动态调整：均已验证无效，保持原值

最终选择：dm不对称 + ME最小持仓20min（不改Z-Score、不改intraday_filter）
32天回测PnL：+436pt → +634pt(dm不对称) → +735pt(+ME20min)
Breakeven滑点：~2.5pt → ~4.0pt

### 未来数据泄漏修复 + 逆势对称化（2026-03-30）

- [x] **回测数据泄漏修复**：`backtest_signals_day.py` 所有日线查询从 `trade_date <= td` 改为 `trade_date < td`，修复盘后跑回测时当天数据已入库导致dm计算错误
- [x] **逆势做空dm 0.5→0.8**：原d=0.5决策基于含未来数据的回测（64信号WR=42%,avg=-2.5pt）；干净数据显示94信号WR=60%,avg=+26pt，改为0.8后+72pt增量（+13%），breakeven 4.2pt
- [x] **逆势惩罚对称化**：做多/做空逆势统一dm=0.8，简化逻辑

最终参数（干净数据，25天验证）：
- 逆势：dm=0.8（做多做空对称）｜顺势：dm=1.2｜中性：dm=1.0
- sensitivity thr=60: slip=0→+543pt, slip=2→+287pt, slip=3→+159pt, breakeven≈4.2pt

### Monitor/Executor 重构（2026-03-31）

- [x] **Monitor不再prompt**：信号触发直接写JSON+注册shadow，不阻塞等输入
- [x] **Executor负责所有交易确认**：读JSON→展示→Y/N→TQ下单
- [x] **所有CLOSE信号超时持续提醒**（不只止损，TRAILING_STOP等也持续提醒）
- [x] **executor_log表**：每个信号完整记录（response/order_status/filled/cancel/signal_json）
- [x] **撤单机制**：开仓60秒/平仓30秒未成交自动撤单，平仓提示改激进价
- [x] **信号过期**：>5分钟的信号标记EXPIRED不执行
- [x] **持仓追踪**：executor内部_positions，无持仓忽略CLOSE、重复开仓跳过
- [x] **实际执行限定1手**（EXEC_MAX_LOTS=1，实盘验证期）
- [x] **品种乘数修正**：IF/IH=300, IM/IC=200（monitor+executor统一）
- [x] **Shadow M/V/Q key修正**：momentum→s_momentum（之前全部记为0）
- [x] **SQLite WAL统一**：所有脚本sqlite3.connect加timeout=30+WAL+busy_timeout
- [x] **开仓截止推迟到14:30**（NO_OPEN_EOD从14:15改为14:30，尾盘信号WR=67-80%）
- [x] **DB损坏修复**：dump+reimport修复Google Drive同步导致的corruption

---

## 2026-04-01：Executor持久连接 + Monitor重启恢复 + Bug修复

### Executor 持久TQ连接
- [x] 启动时建立TQ连接并保持，下单时直接使用（零延迟，之前每次下单临时连接需2-3秒）
- [x] 主力合约OI查询改为批量订阅+5秒deadline防阻塞（之前逐个wait_update可能卡住）
- [x] 持仓扫描前先订阅所有候选合约quote，保证OI数据到达
- [x] 对账不误删刚成交的持仓（快照比成交旧时保留）

### Monitor 盘中重启状态恢复
- [x] **问题**：重启后position_mgr.positions清空，`_total_net_lots()=0`，4个品种同时突破max_total_lots=2发信号
- [x] **shadow持仓持久化**：`_save_shadow_state()` → `tmp/shadow_state.json`，每次shadow变更自动写
- [x] **启动恢复三层**：① shadow_state.json→_shadow_positions ② inject_position→position_mgr占位 ③ shadow_trades→daily_trades+risk_mgr
- [x] **退出清理**：shadow exit时`remove_by_symbol()`释放position_mgr占位
- [x] position.py新增`inject_position()`（极端止损不触发）和`remove_by_symbol()`

### Bug修复
- [x] **SHADOW浮盈基准错误**：面板用现货close计算浮盈，但entry_price是期货bid/ask，导致浮盈虚高~200pt（等于贴水）。改用期货last_price，fallback现货close
- [x] **跨市场数据**：Morning Briefing增加DJI+实时更新+过期标注+权重调整
- [x] **主力合约选择统一**：所有合约选择统一到`cffex_calendar.py`，按持仓量(OI)而非到期日
- [x] **signal_pending.json竞态**：改为列表模式（追加写入），平仓确认分级（紧急opt-out/非紧急opt-in）

---

## 已知问题 / TODO

- [ ] **Morning Briefing外盘数据滞后（2026-03-31）**：`make briefing` 未自动更新本地DB，若忘记先跑 `download_briefing_history.py --update`，`global_index_daily` 会用旧数据（如3/31 briefing用了3/27的IXIC -2.15%而非实际-0.73%）。修复方案：在 `Makefile` 的 `briefing` target 中自动先跑 `--update`，或在 `morning_briefing.py` 启动时检查DB数据是否超过1个交易日并警告。
- [ ] TqSdk 异步架构需要特殊处理（事件循环）
- [ ] 期货分钟线数据量大，需要分批下载和压缩存储
- [ ] 多标的同时运行时的相关性处理
- [ ] 节假日前的 Theta 加速效应处理
- [ ] models/ 中的旧路径文件（如 models/greeks.py）将在下一步清理

---

*本文件由 Claude Code 辅助维护，请在每次 Step 完成后更新状态。*
