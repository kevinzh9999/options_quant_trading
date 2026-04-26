# Daily XGB 跨日策略

> 跨日 trend-following swing strategy，与 intraday 完全隔离。
> Walk-forward 2.9 年验证：年化 +1,572K (1 手), Calmar 2.93, Sharpe 5.64。
> 实盘部署：保守模式（1 lot/signal, cap=10）→ 真实执行后年化 +21% 账户。

---

## 策略 lineage

研究历程的核心发现按顺序：

1. **5min reversal XGBoost 失败** (2026-04 研究): Test AUC 0.50，5min 时间尺度的反转模式被 noise 主导，模型无法学习。

2. **Daily IV/RR/VRP 信号显著** (突破): IV/期限结构/Skew 是日级信号，把 horizon 从 5min 升到 5d 后 Test IC 0.10-0.19。Walk-forward 5-fold 全正。

3. **因子化框架 + Strict OOS 验证**:
   - 18 默认因子（IV 8 + RV 2 + Price 8）
   - Strict OOS（train ≤ 2024-12, predict 2025+ 不重训）OOS IC +0.115
   - Walk-forward retrain every 20 days，OOS IC +0.097（一致）
   - **结论**：信号 robust，不是 partial overfit

4. **2024 H2 牛市发现 +132 万元年化** (1×lot): 但 MaxDD -83 万。

5. **MaxDD 归因分析**: SHORT 端在 2024-2025 牛市亏 -58 万 是 DD 主因。

6. **A+D 组合发现**: 加 ATR×1.5 SL + 3 个 regime 因子 (close/sma60, slope_60d, vol_regime) → 年化 +918K, Calmar 1.16, Sharpe 3.44。

7. **G3s SHORT block**: close/sma60>1.04 AND close/sma200>1.05 → 阻 SHORT。2025 牛市 SHORT 从 -323K → -55K。Calmar 升到 2.22。

8. **N5 LONG enhancement (2-branch)**:
   - 严格牛 (close/sma60>1.04 AND close/sma200>1.05) → hold 10d, SL ATR×4
   - 牛市 dip-buy (close/sma200>1.03 AND close/sma60<1.02) → hold 15d, SL ATR×4
   - 2025 LONG 从 +405K → +1,255K (+267%)。Calmar 2.13。

9. **M7 SHORT 镜像 enhancement**: 用户 insight "策略是不是只在牛市好用"。镜像 N5：
   - 严格熊 (close/sma60<0.97 AND close/sma200<0.97) → hold 10d, SL ATR×4
   - 熊市反弹 (close/sma200<0.99 AND close/sma60>0.98) → hold 15d, SL ATR×4
   - 2024 H1 SHORT 从 +766K → +1,797K，2023 全年 +40K → +272K。
   - **完整 M7 baseline: 年化 +1,572K, Calmar 2.93, Sharpe 5.64**。

10. **2026 Q1 黑天鹅深度分析**:
    - 2026 Jan 12 笔 LONG 净 -152K，源于 1/30-2/2 Trump-Warsh Fed Chair 提名导致全球风险资产同步抛售
    - 测试 50+ 修复方案：chop detector / vol filter / IV term filter / SL 收紧 / hold 缩短 / EOD SL / break-even / trailing stop / day-1 fast exit / regime feedback
    - **唯一 Pareto 改进**：IV3 (VRP < 0.02 enhancement filter)，年化 -10% 换 Calmar +17%
    - **核心结论**：2/2 单日 -3.39% 暴跌后立即反弹是 wick whipsaw，任何 intraday/EOD SL 都在低点击中后错过反弹
    - **接受 2026 Jan -152K 是 trend strategy 的 black swan tax**（占年化 9.7%）

11. **执行模型现实化**: backtest 假设 T close 进出场，实际 T+1 open 进出场。Realistic walk-forward = +1,343K 年化（vs idealized +1,472K，-9%）。

12. **仓位 sizing 三档对比**:
    - 保守 1×lot, cap=10: 年化 +1,472K (+23% 账户), MaxDD -8.4%, Calmar 2.75
    - 中等 1×lot, no cap (=baseline): 年化 +1,572K, Calmar 2.93
    - 激进 2×lot, cap=20: 年化 +2,944K (+46% 账户), MaxDD -16.7%, Calmar 2.75
    - **结论**："1 lot per signal" 是合理的隐含 sizing；用 fixed-risk per trade 反而破坏 enhancement alpha
    - **决策**：保守模式部署

---

## 模型架构

### 21 因子 (18 默认 + 3 regime)

| 类别 | 因子 | 含义 |
|---|---|---|
| **iv** (8) | atm_iv_market | 期权 ATM IV |
| | iv_pct_60d | IV 60日历史分位 |
| | rr_25d | 25-delta risk reversal (skew) |
| | vrp | IV - blended RV |
| | iv_term_spread | 当月 IV - 季月 IV |
| | iv_change | ATM IV 日变化 |
| | rr_change | RR 日变化 |
| | vrp_change | VRP 日变化 |
| **rv** (2) | realized_vol_5d | 5日 RV (年化) |
| | realized_vol_20d | 20日 RV |
| **price** (8) | today_return | 今日 (close-open)/open |
| | today_amp | 今日 (high-low)/open |
| | gap | 今日 open vs 昨日 close |
| | ret_1d, ret_5d, ret_20d | 1/5/20 日 close-to-close |
| | pos_in_20d | close 在 20 日 high-low 区间位置 |
| | slope_5d | 5 日 close 线性回归斜率 |
| **regime** (3) | close_sma60_ratio | close / SMA(60) |
| | slope_60d | 60 日 close slope (normalized) |
| | vol_regime | RV5 / RV60 (vol expansion ratio) |

### XGBoost 超参（walk-forward 验证）

```python
n_estimators=200, max_depth=4, learning_rate=0.03,
min_child_weight=10, subsample=0.8, colsample_bytree=0.7,
reg_alpha=0.5, reg_lambda=2.0, random_state=42
```

### Regime gates (G3s + N5 + M7)

```
G3s (SHORT 屏蔽，bull regime):
   close/sma60 > 1.04 AND close/sma200 > 1.05  → 拦截 SHORT 信号

N5 (LONG enhancement):
   strict bull: close/sma60 > 1.04 AND close/sma200 > 1.05 → hold 10d, SL ATR×4
   dip bull:    close/sma200 > 1.03 AND close/sma60 < 1.02 → hold 15d, SL ATR×4

M7 (SHORT enhancement, mirror of N5):
   extended bear: close/sma60 < 0.97 AND close/sma200 < 0.97 → hold 10d, SL ATR×4
   rip bear:      close/sma200 < 0.99 AND close/sma60 > 0.98 → hold 15d, SL ATR×4

Default (其他情况):
   hold 5d, SL ATR×1.5
```

### 信号阈值

```
Top threshold:  in-sample preds P80 → LONG signal
Bot threshold:  in-sample preds P20 → SHORT signal
中间区:        SKIPPED_NO_DIRECTION
```

---

## 代码结构

```
strategies/daily_xgb/
├── config.py              # DailyXGBConfig — 保守模式参数硬编码
├── factors.py             # 21 因子定义 + build_full_pipeline()
├── regime.py              # RegimeState, G3s/N5/M7 gate 函数
├── pipeline.py            # XGBoost wrapper, train/predict, 模型缓存
├── persist.py             # DB IO (4 张表) + JSON pipe IO
├── signal_generator.py    # 主入口: generate_signal(date) → DB+JSON
├── position_manager.py    # OPEN trades 追踪 + EOD check_exits
└── risk_guard.py          # 4 层熔断 (kill/daily/weekly/margin)

scripts/
├── daily_xgb_executor.py        # 独立 TQ executor
├── daily_xgb_self_backtest.py   # 自研 backtest 验证生产代码
└── daily_xgb_tq_backtest.py     # TqBacktest 实盘模拟
```

## DB 表（独立 namespace）

```sql
daily_xgb_signals       -- 每个 signal 完整记录
daily_xgb_trades        -- 完整交易生命周期 (entry → exit)
daily_xgb_orders        -- executor 下单 audit
daily_xgb_executor_log  -- executor 事件流
```

## JSON pipes

```
tmp/daily_xgb_pending_{date}.json   # signal_generator 写, executor 读
tmp/daily_xgb_positions.json        # executor 写, 监控用
tmp/daily_xgb_kill.flag             # 操作员紧急熔断 (touch 此文件即停)
```

## TQ 资源隔离

- 独立 TqApi 进程
- 独立 order_id 前缀: `DXGB_OPEN_xxx` / `DXGB_CLOSE_xxx`
- 锁仓模式 (CFFEX 股指期货支持同时 LONG/SHORT 持仓)
- 与 intraday executor 共用账户但**逻辑持仓互不干扰**

---

## 测试验证

### 1. 自研 backtest (production 代码 vs research baseline)

```
$ python scripts/daily_xgb_self_backtest.py
Total trades:    268
WR:              60.4%
Total PnL:       +4,307,749 元 (2.9 年)
年化:             +1,472,218 元/年   ← 期望 +1,472K ✓
MaxDD:           -536,066            ← 期望 -536K   ✓
Sharpe:          5.37                ← 期望 5.37    ✓
Calmar:          2.75                ← 期望 2.75    ✓

Enhancement type breakdown:
  default      N=121  PnL=+757,792  WR=50%
  m7_extended  N= 22  PnL=+582,233  WR=86%   ← 熊市主力
  m7_rip       N= 29  PnL=+1,125,623  WR=86%
  n5_dip       N= 29  PnL=+1,086,820  WR=72%   ← 牛市 dip-buy
  n5_strict    N= 67  PnL=+755,282  WR=55%
```

### 2. TqBacktest 实盘模拟 (2025-04-07 ~ 2025-04-25)

```
$ make dxgb-tq-bt
Trades:           13
WR:               92%
Cumulative PnL:   +605,201 元
账户回报:         10M → 10.62M = +6.16% 一个月
TQ 算年化:        +64.59%
TQ Sharpe:        3.72
TQ MaxDD:         1.58%
手续费/笔:        27.81 元
```

验证了：
- 限价单 09:35 自动成交（first 5min bar 后）
- EOD SL 触发 (4/8 trade -4.62% 止损)
- TIME 退出 (hold_days 满) 在 14:55 触发
- TqSim 真实手续费 + 保证金占用计算
- 最高峰持仓 9 手（cap=10 内安全）

---

## 部署运维

### 每日操作

**盘后 17:30** — 信号生成（每日重训，~2 min）：

```bash
make dxgb-signal
# = python -m strategies.daily_xgb.signal_generator
# 输出 tmp/daily_xgb_pending_{date}.json
# 可能 status: PENDING / SKIPPED_NO_DIRECTION / SKIPPED_GATE / SKIPPED_INSUFFICIENT_DATA
```

**次日 09:25** — Executor 启动：

```bash
make dxgb-executor
# 显示 pending 信号 + 风险状态
# 操作员 [y/n] (60s timeout=skip)
# 09:35 自动开仓
# 14:55 自动 EOD check（TIME / SL）
```

### 紧急熔断

```bash
# 暂停新开仓（已有持仓继续 EOD 平仓）
touch tmp/daily_xgb_kill.flag

# 解除
rm tmp/daily_xgb_kill.flag
```

### 风险熔断条件（任一触发即拦截新开仓）

| 条件 | 阈值 | 说明 |
|---|---|---|
| Kill switch | tmp/daily_xgb_kill.flag 存在 | 永久暂停 |
| 单日亏损 | < -3% × account_equity | 单日触发后当天不再开 |
| 单周累亏 | < -5% × account_equity | 7 日窗口 |
| 保证金占用 | > 85% × account_equity | 防爆仓 |
| 并发上限 | > 10 手 | 保守模式硬上限 |

---

## 未来扩展点

1. **每月重训改为按需** — 当前每日重训 ~2min 成本，可改为每月第一个交易日重训一次
2. **品种扩展** — 当前仅 IM。IC/IF 未单独优化，可平移测试
3. **滑点压力测试** — 当前默认 8bp，需 12-25bp stress test
4. **集成 Dashboard** — 加 Daily XGB tab 展示当前持仓、近期信号、累计 PnL
5. **黑天鹅事件标签** — 历史复盘标签（2018 贸易战/2020 COVID/2024 carry unwind/2026 Warsh）用于回测加 stress 场景
