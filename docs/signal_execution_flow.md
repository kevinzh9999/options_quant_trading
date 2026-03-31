# 信号生成 → Executor 执行 完整流程

本文档详细描述日内策略信号从生成到最终执行的完整链路。

---

## 架构概览

```
┌──────────────────┐     JSON文件      ┌──────────────────┐
│  Intraday Monitor │ ──────────────▶  │  Order Executor   │
│  (信号生成)       │  signal_pending  │  (交易执行)        │
└──────────────────┘                   └──────────────────┘
       │                                      │
       │ 每5分钟bar                            │ 每1秒轮询
       ▼                                      ▼
  tmp/signal_pending.json              TQ API 限价单
  tmp/futures_positions.json           executor_log / order_log
```

**职责分离**：Monitor 只生成信号，不阻塞不等待；Executor 负责确认和下单。两者通过 `tmp/signal_pending.json` 文件解耦。

---

## 一、开仓流程

### 1.1 Monitor 端：信号生成

**触发时机**：每根 5 分钟现货 K 线收盘时（`_on_new_bar`）

**计算流程**：

```
现货5分钟K线 ─▶ SignalGeneratorV2/V3.score_all() 内部：
                  │
                  ├── 维度计算: raw = s_mom + s_vol + s_qty + s_breakout
                  ├── × daily_mult（顺势1.2 / 中性1.0 / 逆势0.8）
                  ├── × intraday_filter（日内涨跌幅折扣，方向感知）
                  ├── × time_weight（时段权重）
                  ├── × sentiment_mult（期权情绪：IV变动/Skew/PCR/期限结构）
                  ├── clamp(0, 100)
                  ├── Z-Score 硬过滤（极端Z值可能归零）
                  └── RSI 回归 bonus
                  │
                  ▼
              score_all() 返回最终 total + direction
                  │
                  ▼
              Monitor: total >= 60?  ──否──▶ 记录score=0到signal_log
                  │
                 是
                  │
                  ▼
              is_open_allowed() 时间窗口检查
              （09:45~11:20, 13:05~14:30）
                  │
                  ▼
              写JSON + 注册shadow持仓
```

注意：daily_mult、intraday_filter、sentiment_mult 等乘数在 `score_all()` **内部**作用于分数，
不是在阈值判断之后。Monitor 拿到的 `total` 已经是经过所有乘数调整后的最终值。

**生成 JSON 信号**（`_append_signal` 追加到列表）：

`signal_pending.json` 存储 **JSON 数组**，每个元素是一个信号对象。
Executor 读取后按顺序处理所有信号（CLOSE 在 OPEN 之前），然后删除文件。

```json
[
  {
    "timestamp": "2026-03-31 02:15:00",
    "symbol": "IM",
    "direction": "LONG",
    "action": "OPEN",
    "score": 72,
    "bid1": 7648.2,
    "ask1": 7648.8,
    "last": 7648.6,
    "suggested_lots": 3,
    "limit_price": 7648.8,
    "reason": ""
  }
]
```

**限价计算规则**：
- 做多（LONG）：`limit_price = ask1`（买一排队）
- 做空（SHORT）：`limit_price = bid1`（卖一排队）

**建议手数计算**（`_calc_suggested_lots`）：
```
risk_per_trade = 账户权益 × 0.5%
stop_loss_amount = entry_price × 合约乘数 × 0.5%
suggested_lots = risk_per_trade / stop_loss_amount
```

**写入文件后**：
- 写入 `tmp/signal_pending.json`
- 注册 shadow 持仓（`_shadow_positions[sym]`），记录入场价、时间、维度明细
- 记录到 `signal_log` 表

### 1.2 Executor 端：接收与处理

**轮询**：主循环每 1 秒检查 `tmp/signal_pending.json` 是否存在

**处理流程**：

```
读取JSON ─▶ 删除文件 ─▶ 写入 executor_log（无论后续是否下单）
                │
                ▼
           信号过期检查（>5分钟 → EXPIRED，跳过）
                │
                ▼
           安全检查（开仓专属）
           ├── 单日下单数 >= 10 → REJECTED (MAX_ORDERS)
           ├── 单日亏损 > 1% → REJECTED (MAX_LOSS)
           └── 同品种已有同方向持仓 → REJECTED (DUPLICATE)
                │
                ▼
           手数三级递减
           ├── Monitor建议: N手
           ├── Executor减半: max(1, N÷2)
           └── 实盘上限: min(上一步, EXEC_MAX_LOTS=1)
                │
                ▼
           展示订单详情
           ┌─────────────────────────────────────────┐
           │ 新信号 | 2026-03-31 10:15:00             │
           │ 品种: IM          方向: 买入开仓          │
           │ 限价: 7648.8                              │
           │ 手数: 建议3手 → 减半2手 → 实际执行1手     │
           │ 信号分数: 72                              │
           │ 止损价: 7610.6      最大亏损: ¥7,656      │
           └─────────────────────────────────────────┘
                │
                ▼
           操作者确认 [y/n] (60s timeout)
           ├── y → 下单
           ├── n → SKIPPED，记录后返回
           └── 超时(无输入) → TIMEOUT_SKIP，自动放弃
```

**开仓确认是 opt-in**：必须主动按 `y` 才会执行，超时自动放弃。

### 1.3 TQ 下单

```
确认后 ─▶ TqClient 连接
          │
          ▼
     api.insert_order()
     合约: 主力合约（_resolve_near_month 自动检测当月/季月）
     方向: BUY(做多) / SELL(做空)
     开平: OPEN
     价格: 限价单（排队价）
     手数: 1手（实盘验证期）
          │
          ▼
     等待成交（60秒超时，每10秒打印状态）
          │
          ├── 全部成交 → FILLED
          ├── 部分成交 → PARTIAL（撤剩余）
          └── 未成交 → 撤单 → TIMEOUT_CANCEL
          │
          ▼
     更新 executor 内部 positions 字典
     记录到 executor_log + order_log
```

### 1.4 状态记录

每个信号无论最终是否下单，都完整记录到 `executor_log` 表：

| 字段 | 说明 |
|------|------|
| signal_time | Monitor生成信号的时间 |
| receive_time | Executor收到的时间 |
| operator_response | Y / N / TIMEOUT / EXPIRED / REJECTED / DRY_RUN |
| order_submitted | 0=未下单, 1=已下单 |
| order_status | FILLED / PARTIAL / TIMEOUT_CANCEL / NOT_SUBMITTED / ERROR |
| filled_lots | 实际成交手数 |
| filled_price | 成交均价 |
| cancel_time | 撤单时间（如有） |
| signal_json | 原始信号JSON留底 |

---

## 二、平仓流程

### 2.1 Monitor 端：退出信号生成

**触发时机**：每根 5 分钟 K 线时，对所有 shadow 持仓调用 `check_exit()`

**7 个平仓优先级**（高 → 低）：

| 优先级 | 类型 | 条件 |
|--------|------|------|
| 1 | EOD_CLOSE | 收盘前强制平仓（UTC 06:55 = BJ 14:55） |
| 2 | STOP_LOSS | 亏损超过止损阈值 |
| 3 | LUNCH_CLOSE | 午休前平仓（UTC 03:25 = BJ 11:25） |
| 4 | TRAILING_STOP | 动态跟踪止盈回撤（宽度随持仓时间 0.5%→1.0%） |
| 5 | TREND_COMPLETE | 5分钟+15分钟布林带都到极端 |
| 6 | MOMENTUM_EXHAUSTED | 动量衰竭（最小持仓20分钟/4根K线） |
| 7 | MID_BREAK / TIME_STOP | 中轨突破 / 时间止损 |

**检查逻辑**：
```
shadow_positions[sym] ─▶ check_exit()
                          │
                          ├── 更新极值（highest_since / lowest_since）
                          ├── 按优先级逐个检查退出条件
                          └── should_exit=True → 生成平仓信号
```

**数据源区分**：
- **退出条件判断**（布林带、ATR、中轨突破等）：使用**现货 K 线**（与回测一致）
- **退出价格和 PnL**：使用**期货价格**（entry 和 exit 都用期货 bid/ask/last）

**生成 CLOSE JSON**：

```json
{
  "timestamp": "2026-03-31 02:45:00",
  "symbol": "IM",
  "direction": "LONG",                // 原持仓方向
  "action": "CLOSE",
  "reason": "STOP_LOSS",
  "bid1": 7632.4,
  "ask1": 7633.0,
  "last": 7632.8,
  "suggested_lots": 3,
  "limit_price": 7632.4,              // LONG平仓用bid1, SHORT平仓用ask1
  "pnl_pts": -16.4
}
```

**限价计算规则**（与开仓方向相反）：
- 平多头（原 LONG）：`limit_price = bid1`（卖一排队出场）
- 平空头（原 SHORT）：`limit_price = ask1`（买一排队出场）

**写入文件后**：
- 写入 `tmp/signal_pending.json`
- **立即删除 shadow 持仓**（`del _shadow_positions[sym]`）
- 记录到 `shadow_trades` 表

### 2.2 Executor 端：平仓处理

**与开仓的关键区别（三级确认模式）**：

| 维度 | 开仓 | 紧急平仓 | 非紧急平仓 |
|------|------|---------|-----------|
| 触发原因 | — | STOP_LOSS / EOD_CLOSE / LUNCH_CLOSE | TRAILING_STOP / TREND_COMPLETE / 其他 |
| 确认模式 | opt-in | **opt-out** | opt-in |
| 超时行为 | 自动放弃 | **自动下单** | 自动放弃 |
| 未成交追单 | 无 | **自动激进价追单** | 无 |
| 按n否决 | SKIPPED | CLOSE_DENIED | CLOSE_DENIED |
| 手数 | 三级递减 | 持仓手数（全部平仓） | 持仓手数（全部平仓） |
| 成交超时 | 60秒 | **30秒** | **30秒** |

**处理流程**：

```
读取CLOSE JSON ─▶ 前置检查
                   │
                   ├── positions[sym] 不存在 → REJECTED (NO_POSITION)
                   └── 信号过期(>5min) → EXPIRED
                   │
                   ▼
              手数 = positions[sym]["lots"]（全部平仓）
                   │
                   ▼
              锁仓判断
              ├── 股指期货(IM/IF/IH/IC) → 锁仓（反向OPEN）
              │   原LONG → SELL+OPEN, 原SHORT → BUY+OPEN
              │   (避免平今手续费10倍)
              └── 其他（期权等） → 正常CLOSE
                   │
                   ▼
              展示订单详情
              ┌──────────────────────────────────────────────┐
              │ ⚠ 平仓信号 | 2026-03-31 10:45:00             │
              │ 品种: IM2604        持仓: 多头1手              │
              │ 操作: 卖出开仓(锁仓)  原因: STOP_LOSS          │
              │ 限价: 7632.4 (bid1)                           │
              │ 浮亏: -16.4pt = ¥-3,280                       │
              │ 说明: 股指期货平今手续费10倍，改用锁仓。         │
              └──────────────────────────────────────────────┘
                   │
                   ▼
              操作者确认 [y/n]
              │
              ├─ 紧急(STOP_LOSS/EOD/LUNCH):
              │   提示 "60s无响应将自动执行"
              │   ├── y → 下单
              │   ├── n → CLOSE_DENIED
              │   └── 超时 → AUTO_TIMEOUT，自动下单
              │
              └─ 非紧急(TRAILING_STOP等):
                  提示 "60s timeout"
                  ├── y → 下单
                  ├── n → CLOSE_DENIED
                  └── 超时 → TIMEOUT_SKIP，放弃
```

### 2.3 TQ 下单 + 激进价追单

```
确认/自动执行 ─▶ api.insert_order()
                 方向: 锁仓=反向OPEN / 普通=CLOSE
                 价格: 限价单
                 │
                 ▼
            等待成交（30秒超时）
                 │
                 ├── 成交 → FILLED，清除 positions[sym]
                 │
                 └── 未成交 → 撤单
                      │
                      ▼
                 自动激进价追单（不弹确认）
                 ├── 获取最新盘口
                 ├── 多头: aggr_price = bid1 - 2.0 (10跳)
                 ├── 空头: aggr_price = ask1 + 2.0 (10跳)
                 ├── 打印: "限价未成交，自动以激进价 XXXX.X 重新下单"
                 └── 再等15秒
                      │
                      ├── 成交 → FILLED
                      └── 仍未成交 → 打印 "⚠ 激进价也未成交！请手动处理持仓"
```

### 2.4 平仓否决（CLOSE_DENIED）

操作者按 `n` 明确否决平仓时：

1. **记录到 order_log**：`action="CLOSE_DENIED"`, `status="CLOSE_DENIED"`
2. **写入 `tmp/denied_positions.json`**：
   ```json
   [
     {
       "date": "20260331",
       "symbol": "IM",
       "contract": "CFFEX.IM2604",
       "direction": "LONG",
       "lots": 1,
       "deny_time": "10:45:30",
       "deny_reason": "STOP_LOSS",
       "entry_price": 7648.8
     }
   ]
   ```
3. **次日 Executor 启动时**：读取文件，打印醒目警告提醒操作者检查是否已手动处理
4. **注意**：此时 Monitor 端 shadow 持仓已被删除，不会再生成该品种的平仓信号

---

## 三、Executor 启动与持仓恢复

### 3.1 启动流程

Executor 启动时按顺序执行以下检查：

```
main() 启动
  │
  ▼
① _check_locked_positions()     ← 前日锁仓提醒
  │
  ▼
② _check_denied_positions()     ← 前日否决平仓提醒
  │
  ▼
③ _restore_positions_from_log() ← 从 order_log 恢复当天持仓
  │
  ▼
④ _reconcile_positions()        ← 用 TQ 实盘数据校正
  │
  ▼
  进入主循环（每1秒轮询信号）
```

### 3.2 持仓恢复（重启场景）

**问题**：Executor 的 `positions` 字典存在内存中。盘中崩溃重启后字典清空，后续平仓信号会被 REJECTED（NO_POSITION）。

**解决**：启动时从 `order_log` 表推断当天应有持仓。

```
查询当天 order_log WHERE status='FILLED'
  │
  ▼
按品种汇总：
  action=OPEN  → 对应方向加仓
  action=CLOSE/LOCK → 对应方向减仓
  │
  ▼
net_long - net_short ≠ 0 → 写入 positions 字典
  │
  ▼
打印: "从order_log恢复持仓: IM 多1手@7648"
```

**恢复后**由 `_reconcile_positions()` 用 TQ 实盘数据二次校正——如果 order_log 和 TQ 实际不一致（如手动操作），以 TQ 为准。

### 3.3 持仓对账（盘中持续）

**数据流**：
```
Monitor (每5分钟) ─▶ tmp/futures_positions.json ─▶ Executor (每60秒读取)
     │                                                    │
     │ api.get_position()                                 │ 对账逻辑
     ▼                                                    ▼
  TQ 实盘持仓                                    executor.positions 字典
```

**对账规则**：

| 场景 | 处理 |
|------|------|
| TQ 有持仓，Executor 有记录，数量一致 | 正常，跳过 |
| TQ 有持仓，Executor 有记录，**数量不一致** | 打印警告，**更新为TQ实际值** |
| TQ 有持仓，Executor **无记录** | 打印提醒（可能是手动开仓），**不自动补录** |
| TQ **无持仓**，Executor 有记录 | 打印警告（已被手动平仓），**自动清除executor记录** |

**数据新鲜度**：
- `futures_positions.json` 中包含 `timestamp` 字段
- Executor 只使用 10 分钟内的数据（Monitor 可能已停止）
- 超过 10 分钟的数据静默忽略

### 3.4 恢复优先级

```
order_log 推断  →  TQ 实盘校正  →  最终 positions
  (第一手段)        (第二手段，权威)
```

- order_log 恢复的是"executor 自己开过什么"，覆盖了重启丢失的内存
- TQ 对账修正"实际还剩什么"，处理手动操作和部分成交等边界情况
- 两步顺序不能反：先恢复再校正，确保 TQ 有持仓但 executor 无记录时能正确判断是"恢复后匹配"还是"手动开仓"

---

## 四、锁仓次日处理

股指期货日内平仓用锁仓代替（避免平今手续费 10 倍）。

**次日流程**：
1. Executor 启动时 `_check_locked_positions()` 查询 `order_log` 表中 `action='LOCK'` 的记录
2. 打印提醒：哪些品种需要双向平仓
3. 操作者在盘中**手动**平仓（双向都用 CLOSE，平昨仓手续费低）

---

## 五、关键配置常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `SIGNAL_EXPIRY_SECS` | 300 | 信号过期时间（5分钟） |
| `OPEN_FILL_TIMEOUT` | 60 | 开仓等成交超时（秒） |
| `CLOSE_FILL_TIMEOUT` | 30 | 平仓等成交超时（秒） |
| `EXEC_MAX_LOTS` | 1 | 实盘验证期最大手数 |
| `MAX_DAILY_ORDERS` | 10 | 单日最大下单次数 |
| `MAX_DAILY_LOSS_PCT` | 0.01 | 单日最大亏损比例 |
| `STOP_LOSS_PCT` | 0.005 | 止损比例（0.5%） |

---

## 六、文件路径汇总

| 文件 | 写入方 | 读取方 | 说明 |
|------|--------|--------|------|
| `tmp/signal_pending.json` | Monitor | Executor | 信号列表（JSON数组，追加写入，executor读后即删） |
| `tmp/futures_positions.json` | Monitor | Executor | 期货实盘持仓（每5分钟更新） |
| `tmp/denied_positions.json` | Executor | Executor | 被否决的平仓记录（次日提醒后清除） |
| `tmp/morning_briefing.json` | morning_briefing.py | Monitor | 盘前方向 Guidance |

---

## 七、数据库表

| 表 | 写入方 | 说明 |
|----|--------|------|
| `signal_log` | Monitor | 每根K线的信号评分（含score=0） |
| `shadow_trades` | Monitor | shadow 持仓完整生命周期 |
| `executor_log` | Executor | 每个收到的信号的处理记录 |
| `order_log` | Executor | 实际下单记录（向后兼容） |
| `trade_decisions` | Monitor | 信号决策记录 |
