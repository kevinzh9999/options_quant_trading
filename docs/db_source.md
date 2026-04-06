# 数据库架构与数据总览

> 最后更新：2026-04-06（DB拆分 + 历史数据批量下载完成）

## 数据库架构（四库分离）

| 库 | 大小 | 职责 |
|---|---|---|
| **trading.db** | ~680 MB | 主库：期货/现货K线 + 模型输出 + 信号/交易 + Briefing + 账户 |
| **options_data.db** | ~4.7 GB | 期权库：MO/IO/HO 日线 + 合约信息 + 5分钟K线 |
| **tick_data.db** | ~28.5 GB | Tick库：IM/IC/IF 主连逐笔tick |
| **etf_data.db** | ~56 MB | ETF库：512100/510500/510300/510050 5分钟K线 |

代码通过 `DBManager(db_path, options_db_path)` 双库模式自动路由：
- 期权表（options_daily/options_contracts/options_min）→ options_data.db
- 其他表 → trading.db
- 工厂函数 `get_db()` 自动配置，上层代码无需关心

---

## trading.db (680 MB)

### 期货/现货 K线

| 表 | 行数 | 范围 | 来源 | 用途 |
|---|---:|---|---|---|
| futures_daily | 48,349 | 2015~04-03 | Tushare | IM/IF/IH/IC 日线 OHLCV+OI |
| futures_min | 2,447,680 | 2016~04-03 | TQ Pro | 主连 1m/5m K线 |
| index_daily | 10,932 | 2015~04-03 | Tushare | 000852/000300/000016/000905 日线 |
| index_min | 1,986,912 | 2018~04-03 | TQ Pro | 现货 1m/5m K线（日内策略核心） |

### 模型输出

| 表 | 行数 | 范围 | 来源 | 用途 |
|---|---:|---|---|---|
| daily_model_output | 190 | 2025-06~04-03 | 本地EOD | ATM IV/RV/VRP/RR/Hurst/贴水/Greeks |
| volatility_history | 886 | 2022~03-20 | 本地计算 | RV多周期+GARCH |
| vol_monitor_snapshots | 706 | 03-19~04-03 | 盘中TQ | 盘中IV/VRP/Greeks快照 |

### 信号/交易记录

| 表 | 行数 | 范围 | 来源 | 用途 |
|---|---:|---|---|---|
| signal_log | 2,085 | 03-19~04-03 | Monitor | 每根bar信号评分(M/V/Q/B) |
| shadow_trades | 14 | 04-01~04-03 | Monitor | Shadow模拟交易 |
| executor_log | 45 | 04-01~04-03 | Executor | 信号→执行全生命周期 |
| order_log | 28 | 04-01~04-03 | Executor | TQ下单/撤单/成交 |
| trade_decisions | 41 | 03-19~04-03 | Executor | 操作者Y/N决策 |
| trade_records | 238 | 03-16~04-03 | TQ EOD | 全部成交记录 |
| orderbook_snapshots | 2,089 | 03-19~04-03 | Monitor | 盘口bid/ask快照 |

### Briefing 数据源

| 表 | 行数 | 范围 | 来源 | 用途 |
|---|---:|---|---|---|
| global_index_daily | 16,123 | 2010~03-31 | Tushare | A50/SPX/DJI/HSI等全球指数 |
| market_breadth | 3,942 | 2010~03-31 | Tushare | 涨跌家数/涨停跌停 |
| market_turnover | 3,942 | 2010~03-31 | Tushare | 沪深成交额 |
| northbound_flow | 2,677 | 2014~03-31 | Tushare | 北向资金 |
| margin_data | 3,456 | 2012~03-31 | Tushare | 融资余额 |
| etf_share | 3,284 | 2015~03-31 | Tushare | ETF份额变化 |
| fut_holding_summary | 8,920 | 2015~03-31 | Tushare | 期货多空持仓 |
| option_pcr_daily | 3,204 | 2019~03-31 | Tushare | PCR(看跌看涨比) |
| morning_briefing | 7 | 03-27~04-03 | 本地盘前 | Briefing评分/d_override |

### 账户/持仓

| 表 | 行数 | 范围 | 来源 |
|---|---:|---|---|
| account_snapshots | 14 | 03-16~04-03 | TQ EOD |
| position_snapshots | 69 | 03-16~04-03 | TQ EOD |
| tq_snapshots | 20 | 03-18~03-23 | TQ |
| daily_reports | 20 | 03-18~04-03 | EOD |

---

## options_data.db (4.7 GB)

| 表 | 行数 | 范围 | 来源 | 用途 |
|---|---:|---|---|---|
| options_daily | 764,114 | 2019-12~04-03 | Tushare | MO/IO/HO 期权日线 |
| options_contracts | 9,444 | — | Tushare | 合约信息(行权价/到期日/乘数) |
| options_min | 23,880,339 | 2020-02~04-03 | TQ Pro | MO/IO/HO 5分钟K线 |

options_min 按品种明细：

| 品种 | 合约数 | 行数 | 范围 |
|---|---:|---:|---|
| MO (中证1000期权) | 2,850 | 7,298,459 | 2022-07~04-03 |
| IO (沪深300期权) | 4,063 | 10,367,769 | 2020-02~04-03 |
| HO (上证50期权) | 2,072 | 5,142,876 | 2022-12~04-03 |

---

## tick_data.db (28.5 GB)

| 品种 | Tick数 | 天数 | 范围 | 日均tick |
|---|---:|---:|---|---:|
| IM (中证1000) | 23,859,398 | 896 | 2022-07-22~04-03 | ~26,600 |
| IC (中证500) | 58,359,259 | 2,488 | 2016-01-05~04-03 | ~23,500 |
| IF (沪深300) | 59,186,522 | 2,488 | 2016-01-05~04-03 | ~23,800 |
| **合计** | **141,405,179** | | | |

---

## etf_data.db (56 MB)

| ETF | 代码 | 行数 | 范围 |
|---|---|---:|---|
| 中证1000ETF | 512100 | 43,728 | 2022-07~04-03 |
| 中证500ETF | 510500 | 96,048 | 2018-01~04-03 |
| 沪深300ETF | 510300 | 96,048 | 2018-01~04-03 |
| 上证50ETF | 510050 | 96,048 | 2018-01~04-03 |

注：TQ 5分钟历史数据最早到 2018 年初，2013-2018 的 ETF 分钟线无法获取。

---

## 数据来源

| 来源 | 接口 | 数据类型 | 频率 |
|---|---|---|---|
| **Tushare** | pro_api | 日线(期货/期权/指数)、合约信息、Briefing数据源 | 日频 |
| **TQ Pro** | DataDownloader / get_kline_data_series | 分钟K线(1m/5m)、Tick | 历史批量 |
| **TQ 实时** | get_kline_serial / get_quote | 盘中5分钟bar、盘口行情 | 实时 |
| **本地计算** | daily_record.py / vol_monitor.py | GARCH/IV/VRP/Greeks/Hurst | 日频/盘中 |

---

## 下载工具

| 脚本 | 用途 |
|---|---|
| `scripts/download_pro_v2.py` | TQ DataDownloader 批量下载（K线+Tick，并行，最快） |
| `scripts/download_kline_pro.py` | TQ get_kline_data_series 逐段下载（旧版，兼容） |
| `scripts/download_tick_pro.py` | TQ get_tick_data_series 逐天下载（旧版） |
| `data/download_scripts/download_*.py` | Tushare 日线增量下载 |
| `scripts/download_briefing_history.py` | Briefing 7表批量/增量下载 |
| `scripts/backfill_minute_bars.py` | TQ 分钟线补拉（旧版） |
