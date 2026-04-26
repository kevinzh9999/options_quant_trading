# 每日交易运营手册

> 干净版（2026-04-26 重写）。两条独立策略管线：
>   - **Intraday 日内**：5min K线，盘中 monitor + executor
>   - **Daily XGB 跨日**：21 因子 XGBoost + regime gate，盘后生成信号 + 次日开盘执行
>
> 完全隔离：各自的信号 JSON、DB 表、TQ executor 进程、order_id 前缀。

---

## 时间表（全天）

```
09:00     盘前 — Briefing + 复习昨日 Daily XGB pending 信号
09:25     启动 intraday monitor + intraday executor
09:25     启动 daily_xgb_executor（review 昨晚生成的信号 → 集合竞价委托）
09:30     开盘
09:30-09:35  Daily XGB 限价开仓（first 5min bar 后）
09:45-11:20  Intraday 主交易时段
11:20-13:05  午休（系统禁开仓）
13:05-14:30  Intraday 下午时段
14:30-14:55  Intraday 不开新仓，只管理持仓
14:55     Daily XGB EOD 检查（TIME/SL 触发的次日开盘平仓 flag）
15:00     收盘
17:00-17:30  EOD: daily_record.py eod
17:30     Daily XGB 信号生成（写次日 pending JSON）
18:00-18:30  交易笔记 / 复盘
```

---

## 一、盘前（08:50 - 09:25）

```bash
# 1. Morning Briefing（综合方向 / Hurst / 象限）
make briefing

# 2. 看昨日 EOD 报告
cat logs/eod/$(date -v-1d +%Y%m%d).md

# 3. 看 Daily XGB 上一交易日生成的 pending 信号
ls -la tmp/daily_xgb_pending_*.json
cat tmp/daily_xgb_pending_*.json | jq '.'

# 4. 检查 Daily XGB 风险状态
python -m strategies.daily_xgb.risk_guard
```

### 决策

- 当前 regime 是什么（A/B/C/D 象限 + Hurst）
- Daily XGB 信号方向 / 手数 / SL 是否合理
- 持仓的止损线距离当前价多远
- 一句话写下今日方向 Guidance

---

## 二、盘中启动（09:25 之前）

### tmux 一键启动（推荐）

```bash
make trading-day
# 启动 5 窗口 tmux session：
#   0: intraday monitor (IM)
#   1: intraday monitor (IC)
#   2: vol_monitor
#   3: quadrant_monitor
#   4: intraday order_executor
#   5: daily_xgb_executor    ← 新增
```

### 或分步启动

```bash
# Intraday
python -m strategies.intraday.monitor --symbol IM
python -m strategies.intraday.monitor --symbol IC
python scripts/vol_monitor.py
python scripts/quadrant_monitor.py
python scripts/order_executor.py

# Daily XGB（独立 executor 进程）
python scripts/daily_xgb_executor.py
```

### Daily XGB executor 启动后做什么

- 读 `tmp/daily_xgb_pending_*.json`（上一交易日盘后生成）
- 显示信号 + regime + 风险状态 → 操作员 [y/n] (60s timeout=skip)
- 09:30 后第一根 5min bar：限价单委托（LONG 取 ask, SHORT 取 bid）
- 60s 未成交 → 撤单 → 激进价 +2pt 重试
- 成交后写 `daily_xgb_trades` (status=OPEN) + `daily_xgb_positions.json`
- 14:55 自动跑 EOD check，TIME / SL 触发的下个开盘平仓

---

## 三、盘中（09:30 - 14:30）

### Intraday — 看 monitor 面板

- 信号得分变化趋势（升高=趋势加强）
- 哪个品种最强，方向是什么
- 情绪乘数 / Z-Score 是否极端
- vol_monitor: ATM IV / RR Skew / Theta 累计 / 持仓安全距离

### Daily XGB — 一般无操作

- Daily XGB 是 swing 策略，盘中不交易
- 持仓自动跟踪 SL（基于 EOD close）
- 14:55 EOD check 触发的平仓在次日 09:30 执行

### 信号触发处理

**Intraday 信号 (>=60 分)**：
1. Monitor 自动写 `tmp/signal_pending_{sym}.json` + 注册 shadow
2. Executor 弹窗：建议 N 手 → 减半 N/2 → 实际 1 手（EXEC_MAX_LOTS=1）
3. [y/n] 60s 超时自动放弃；STOP_LOSS / EOD / LUNCH 是 opt-out
4. 60s 限价未成交自动撤单 + 激进价重试

**Daily XGB 信号**：
1. 09:30-09:35 自动开仓（一日一笔，cap=10）
2. 14:55 EOD 检查所有 OPEN 持仓
3. SL（close ≤ entry × (1-sl_pct)）或 TIME（hold_days 到期）→ 次日 09:30 平仓

### 异常

- 标的日内跌/涨 >2%：检查 Daily XGB 持仓 SL 距离
- IV 突然飙 5pp+：可能事件冲击，先评估
- Monitor 重启：自动从 `shadow_state.json` + `daily_xgb_positions.json` 恢复
- Daily XGB executor 重启：从 DB `daily_xgb_trades WHERE status='OPEN'` 恢复

---

## 四、收盘（15:00 - 17:30）

```bash
# 1. EOD 数据归档（17:00 后跑，等 Tushare 出数据）
make eod
# 等同: python scripts/daily_record.py eod
# 入库: futures_daily, options_daily, daily_model_output, account_snapshots, ...

# 2. 持仓分析
python scripts/portfolio_analysis.py

# 3. 波动率快照
python scripts/vol_monitor.py --snapshot

# 4. 回测今天的日内信号（验证 backtest 与实盘 shadow 对齐）
make backtest
```

---

## 五、Daily XGB 信号生成（17:30 - 18:00）

```bash
# 当日 EOD 完成后，生成次日交易信号
python -m strategies.daily_xgb.signal_generator

# 输出:
#   - tmp/daily_xgb_pending_{date}.json  ← executor 次日早盘消费
#   - daily_xgb_signals 表新行 (status=PENDING)
# 训练: 当前实现每日 retrain (用全部历史 ≤ 今日 close)
#   - 训练时间 ~2 min
#   - 模型缓存 logs/daily_xgb/models/model_{date}.pkl
```

每日操作员需要：
- `cat tmp/daily_xgb_pending_*.json` 查看次日信号
- 如有 `direction=NONE` 或 `status=SKIPPED_GATE` 则次日无 daily 交易

---

## 六、盘后复盘（18:30 - 19:00）

### Intraday 复盘

```sql
-- shadow trades vs 实际执行差异
SELECT entry_time, symbol, direction, entry_price,
       exit_time, exit_price, exit_reason, pnl_pts
FROM shadow_trades
WHERE trade_date = strftime('%Y%m%d', 'now')
ORDER BY entry_time;

-- executor 实际执行
SELECT signal_time, symbol, direction, action,
       operator_response, filled_lots, filled_price, order_status
FROM executor_log
WHERE receive_time >= date('now')
ORDER BY receive_time;
```

### Daily XGB 复盘

```sql
-- 今日 Daily XGB 信号
SELECT signal_date, direction, status, hold_days, sl_pct, pred,
       enhancement_type
FROM daily_xgb_signals
WHERE signal_date = strftime('%Y%m%d', 'now')
ORDER BY created_at DESC;

-- 当前 OPEN 持仓
SELECT id, signal_date, entry_date, direction, entry_price, entry_lots,
       sl_price, planned_exit_date, enhancement_type
FROM daily_xgb_trades
WHERE status='OPEN'
ORDER BY entry_date;

-- 今日已平仓
SELECT signal_date, entry_date, exit_date, direction, entry_price, exit_price,
       exit_reason, pnl_yuan
FROM daily_xgb_trades
WHERE status='CLOSED' AND exit_date = strftime('%Y%m%d', 'now');

-- Daily XGB executor 事件
SELECT event_time, event_type, order_id, details
FROM daily_xgb_executor_log
WHERE event_time >= date('now')
ORDER BY event_time;
```

### 写交易笔记

```bash
mkdir -p logs/notes
nano logs/notes/$(date +%Y%m%d).md
```

记录：
1. 盘前 Guidance vs 实际走势 — 对了还是错了
2. Intraday 信号 vs 你的盘感 — 几次一致 / 矛盾
3. Daily XGB 信号 — 信号方向是否符合 regime 预期
4. 改进想法

---

## 七、周末（周五收盘后 30min）

```bash
# 本周账户权益曲线
python -c "
import sqlite3, pandas as pd
conn = sqlite3.connect('data/storage/trading.db')
df = pd.read_sql('''
    SELECT trade_date, balance, float_profit
    FROM account_snapshots
    WHERE trade_date >= strftime(\"%Y%m%d\", \"now\", \"-7 days\")
    ORDER BY trade_date
''', conn)
print(df.to_string())
"

# 本周 Daily XGB PnL
python -c "
import sqlite3
conn = sqlite3.connect('data/storage/trading.db')
rows = conn.execute('''
    SELECT entry_date, direction, enhancement_type, pnl_yuan, exit_reason
    FROM daily_xgb_trades
    WHERE status=\"CLOSED\"
        AND exit_date >= strftime(\"%Y%m%d\", \"now\", \"-7 days\")
    ORDER BY exit_date
''').fetchall()
total = 0
for r in rows:
    print(r)
    total += r[3] or 0
print(f'Total: {total:+,.0f}')
"
```

回答的问题：
- 本周哪条策略赚钱？哪条亏？
- Intraday vs Daily XGB 哪个 Sharpe 高
- Daily XGB enhancement (n5_dip / m7_rip) 触发频率与命中率
- Risk circuit breaker 有没有触发过

---

## 附：常用命令速查

### 信号 / 执行（双策略）

```bash
# === Intraday ===
make briefing                              # Morning Briefing
make monitor-im                             # IM 信号 monitor
make monitor-ic                             # IC 信号 monitor
make vol                                    # 波动率 monitor
make quadrant                               # 象限 monitor
make executor                               # Intraday 半自动下单
make backtest                               # 回测今天

# === Daily XGB ===
python -m strategies.daily_xgb.signal_generator             # 生成次日信号 (盘后跑)
python -m strategies.daily_xgb.signal_generator --date YYYYMMDD --dry-run
python scripts/daily_xgb_executor.py                        # 独立 executor (live)
python scripts/daily_xgb_executor.py --dry-run              # 模拟
python scripts/daily_xgb_executor.py --eod-only             # 只跑 EOD 检查
python scripts/daily_xgb_executor.py --signal-only          # 只处理 pending 一次
python scripts/daily_xgb_self_backtest.py                   # 自研 backtest 验证生产代码
python scripts/daily_xgb_tq_backtest.py --start YYYYMMDD --end YYYYMMDD --quick  # 快速 sim
python scripts/daily_xgb_tq_backtest.py --start YYYYMMDD --end YYYYMMDD          # TqBacktest 完整回放
```

### 风险熔断

```bash
# 紧急停止 Daily XGB 开新仓（不影响已有持仓 EOD 平仓）
touch tmp/daily_xgb_kill.flag

# 解除
rm tmp/daily_xgb_kill.flag

# 查看 Daily XGB 风险状态
python -c "from strategies.daily_xgb.risk_guard import status_report; from strategies.daily_xgb.config import DailyXGBConfig; print(status_report(DailyXGBConfig()))"
```

### 数据维护

```bash
make eod                                    # EOD 数据归档
python scripts/daily_record.py model        # 补跑模型输出
python scripts/portfolio_analysis.py        # 持仓分析
python scripts/backfill_minute_bars.py --symbol 000852 --exchange SSE --bars 10000
```

### tmux session

```bash
tmux attach -t trading_YYYYMMDD             # 进入交易 session
# Ctrl-b 0/1/2/3/4/5 切换窗口
# Ctrl-b d 退出 session（不关闭）
```

---

## Daily XGB 部署模式说明

**保守模式锁死**（无 conservative/aggressive 切换）：
- 每信号 1 手
- 同时持仓上限 10 手（concurrent_cap）
- ATR×1.5 默认 SL，N5/M7 enhancement 用 ATR×4
- 单周亏损 > 5% 暂停新开仓
- 单日亏损 > 3% 暂停新开仓
- 保证金占用 > 85% 暂停新开仓

**预期收益（walk-forward 2.9 yr）**：
- 年化 +1,472K / 1 手 (= +23% 账户)
- MaxDD -536K (= -8.4% 账户)
- Calmar 2.75，Sharpe 5.37
- 真实执行（T+1 open 进出场）打 9% 折扣 ≈ +21% 账户

如需切换到激进 2× 模式（年化 +46% 账户但 MaxDD -16.8%），改 `strategies/daily_xgb/config.py` 中 `lots_per_signal` + `concurrent_cap`。
