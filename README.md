# A股股指期货/期权波动率交易系统

基于 GARCH 模型和波动率风险溢价（VRP）的 A 股股指期权做空波动率策略系统。

## 策略概述

1. 用 **GJR-GARCH(1,1)** 预测未来持仓期的条件波动率
2. 与期权市场的**隐含波动率（ATM IV）**对比，计算 VRP
3. 当 VRP = (IV - GARCH_Vol) / GARCH_Vol > 阈值时，**做空波动率**（卖出 OTM Strangle）
4. 通过 Delta/Vega 风控约束控制敞口

## 交易标的

| 品种 | 代码 | 交易所 |
|------|------|--------|
| 沪深300股指期货 | IF | 中金所 |
| 上证50股指期货 | IH | 中金所 |
| 中证500股指期货 | IC | 中金所 |
| 中证1000股指期货 | IM | 中金所 |
| 沪深300股指期权 | IO | 中金所 |
| 中证1000股指期权 | MO | 中金所 |

## 项目结构

```
options_quant_trading/
├── config/                  # 配置管理
│   ├── config.example.yaml  # 配置模板
│   └── config_loader.py     # 配置加载和验证
├── data/                    # 数据层
│   ├── sources/             # 数据源（Tushare + 天勤）
│   ├── storage/             # SQLite 存储层
│   ├── unified_api.py       # 统一数据接口
│   ├── quality_check.py     # 数据质量检查
│   └── download_scripts/    # 历史数据下载脚本
├── models/                  # 模型层
│   ├── realized_vol.py      # 已实现波动率（5分钟RV）
│   ├── garch_model.py       # GJR-GARCH 模型
│   ├── implied_vol.py       # BS定价和IV反推
│   ├── vol_surface.py       # 波动率曲面
│   └── greeks.py            # 期权Greeks
├── signals/                 # 信号层
│   ├── vrp_signal.py        # VRP信号生成
│   └── signal_types.py      # 信号数据结构
├── risk/                    # 风控层
│   ├── risk_checker.py      # 事前风控检查
│   └── position_sizer.py    # 仓位计算
├── execution/               # 执行层
│   ├── order_manager.py     # 订单管理
│   └── tq_executor.py       # 天勤交易执行
├── analysis/                # 分析层
│   ├── pnl_attribution.py   # P&L归因
│   ├── performance.py       # 策略绩效统计
│   └── model_diagnostics.py # 模型诊断
├── dashboard/               # Streamlit Dashboard
│   ├── app.py               # 主入口
│   └── pages/               # 各功能页面
├── tests/                   # 测试
├── notebooks/               # Jupyter探索笔记（.gitignore）
├── logs/                    # 运行日志（.gitignore）
├── run_daily.py             # 每日主运行脚本
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

```bash
cp config/config.example.yaml config/config.yaml
# 编辑 config/config.yaml，填入 Tushare Token 和天勤账户
```

也可通过环境变量配置（推荐）：

```bash
export TUSHARE_TOKEN=your_token_here
export TQ_ACCOUNT=your_account
export TQ_PASSWORD=your_password
```

### 3. 下载历史数据

```bash
# 下载期货日线（约2015年至今）
python data/download_scripts/download_futures_daily.py --start 20150101

# 下载期权日线（约2019年至今）
python data/download_scripts/download_options_daily.py --start 20190101

# 下载期货5分钟线（用于计算RV，数据量较大）
python data/download_scripts/download_futures_min.py --start 20200101
```

### 4. 每日运行（干跑模式）

```bash
python run_daily.py --dry-run
```

### 5. 启动 Dashboard

```bash
streamlit run dashboard/app.py
```

## 风控参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vrp_threshold` | 0.05 | VRP 入场阈值（5%） |
| `holding_period` | 5 | 持仓周期（交易日） |
| `max_margin_ratio` | 0.50 | 最大保证金占用比例 |
| `max_daily_loss_ratio` | 0.02 | 单日最大亏损比例 |
| `max_delta_exposure` | 500,000 | 最大Delta敞口（元） |
| `max_vega_exposure` | 100,000 | 最大Vega敞口（元/1%波动率） |

## 开发进度

详见 [PROGRESS.md](PROGRESS.md)

## 注意事项

- `config/config.yaml` 已在 `.gitignore` 中，请勿提交真实密钥
- 实盘模式（`--live`）前请充分回测和验证
- 股指期权每手合约乘数为 100，注意保证金计算

## 依赖版本

- Python 3.10+
- arch >= 6.3 (GARCH 模型)
- tushare >= 1.4
- tqsdk >= 3.0
- streamlit >= 1.30
