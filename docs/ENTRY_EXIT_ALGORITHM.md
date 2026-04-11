# Entry / Exit 算法

> 代码入口: `strategies/intraday/A_share_momentum_signal_v2.py`
> 因子组合: `strategies/intraday/factors.py`
> 原子因子: `strategies/intraday/atomic_factors.py`

---

## Entry算法

### 信号评分流程 (`FactorCombiner.combine()`)

```
输入: ScoringContext(symbol, bar_5m[199根], bar_15m, vol_profile, profile)

Step 1: 因子计算（加法）
  ┌─ M 动量 (0-50) ──────────────────────────────────────────┐
  │  amp = amplitude(close, 48)                               │
  │  lb = 4 if amp < 1.5% else 12          动态lookback       │
  │  mom_5m = momentum(close, lb)                             │
  │  dir_5m = momentum_direction(mom_5m)                      │
  │  mom_15m = momentum(close_15m, 6)       15m确认           │
  │  if dir_5m ≠ dir_15m → M=0, 无方向                       │
  │  base = {>0.3%→35, >0.2%→25, >0.1%→15, else→0}          │
  │  bonus = +15 if 15m confirms                              │
  │  M = min(50, base + bonus)                                │
  │  输出: (M, direction)                                     │
  └──────────────────────────────────────────────────────────┘
  
  ┌─ V 波动率 (0-30) ────────────────────────────────────────┐
  │  ratio = atr_ratio(H, L, C, 5, 40)                       │
  │  V = {<0.7→30, <0.9→25, <1.1→15, <1.5→5, else→0}        │
  └──────────────────────────────────────────────────────────┘
  
  ┌─ Q 成交量 (0-20) ────────────────────────────────────────┐
  │  pct = volume_percentile(cur_vol, hist_vols)   优先       │
  │  Q = {>75%→20, >25%→10, else→0}                          │
  │  fallback: ratio = volume_ratio(cur, avg20)               │
  │  Q = {>1.5x→20, >0.5x→10, else→0}                       │
  └──────────────────────────────────────────────────────────┘
  
  ┌─ B 突破 (0-20, 需M>0) ──────────────────────────────────┐
  │  5m从中轨一侧突破另一侧                     base=10      │
  │  + 放量确认                                  +2           │
  │  + 窄带突破(std<均值×0.8)                    +3           │
  │  + 15m同方向确认                             +5           │
  └──────────────────────────────────────────────────────────┘
  
  ┌─ S 启动 (0-15, 需direction) ────────────────────────────┐
  │  突破前5bar高/低 + 振幅扩张>1.5x + 量能>60%分位          │
  │  全部满足 → S=15                                         │
  └──────────────────────────────────────────────────────────┘

Step 2: 加权求和
  raw = M×w_m + V×w_v + Q×w_q + B×w_b + S×w_s
  (w_x 从 SYMBOL_PROFILES["factor_weights"]，默认全1.0)

Step 3: 乘数管道（当前全中性化=1.0）
  adjusted = raw × daily_mult × intraday_filter × time_weight × sentiment_mult

Step 4: 硬过滤（当前中性化）
  total = clamp(adjusted, 0, 100)
  total = zscore_filter(total)
  total += rsi_bonus
```

### 信号过滤流程 (`update()` + `strategy.on_bar()`)

```
score_all() 返回 {total, direction, ...}
         │
         ▼
  total < min_signal_score(50)? ──Yes──→ None
         │ No
  direction 为空? ──Yes──→ None
         │ No
  is_open_allowed(time)? ──No──→ None     ← 时间约束
         │ Yes
  生成 IntradaySignal(score, dir, entry_price=close, stop_loss)
         │
         ▼
  strategy.on_bar():
  score < per_symbol_threshold? ──Yes──→ Skip    ← IM=50, IC=60
         │ No
  risk_mgr.check_pre_trade? ──Fail──→ Skip       ← 日亏/次数/连亏
         │ Pass
  position_mgr.can_open? ──No──→ Skip             ← 已有持仓/超限
         │ Yes
         ▼
       OPEN
```

### Entry时间约束

| 约束 | 时间 (BJ) | UTC | 检查位置 |
|------|----------|-----|---------|
| 开盘静默 | < 09:45 | < 01:45 | `is_open_allowed()` in `update()` |
| 午休禁开 | 11:20 ~ 13:05 | 03:20 ~ 05:05 | `is_open_allowed()` in `update()` |
| 尾盘禁开 | > 14:30 | > 06:30 | `is_open_allowed()` in `update()` |
| 低振幅日 | 10:00后判断 | 02:00 | `monitor._on_new_bar()` |
| 单品种上限 | 全天 | — | `risk_mgr`: ≤5笔/品种/天 |
| 连续亏损 | 全天 | — | `risk_mgr`: 连亏≥3笔暂停 |
| 日亏损限额 | 全天 | — | `risk_mgr`: 超限暂停 |

**开仓窗口（is_open_allowed = True）:**
```
09:45 ─── 11:20    13:05 ─── 14:30
  ████████████        ████████████
```

---

## Exit算法

### 评估流程 (`ExitEvaluator.evaluate()`)

```
输入: position, current_price, bar_5m, bar_15m, current_time, symbol

Step 1: 硬约束（不可配置，不可禁用）
  ┌─────────────────────────────────┐
  │  if time >= 14:45 → EOD_CLOSE  │
  └─────────────────────────────────┘

Step 2: 预计算原子因子
  hold_min  = hold_time(entry_time, cur_time)
  loss_pct  = -pnl_pct(price, entry, dir)        正值=亏损
  pnl_val   = pnl_pct(price, entry, dir)          正值=盈利
  zone_5m   = boll_zone(price, mid_5m, std_5m)
  zone_15m  = boll_zone(price, mid_15m, std_15m)
  b5_mid, b5_std = boll_params(close_5m, 20)

Step 3: 按优先级评估（每个条件可通过exit_weights禁用）

  P10 ── StopLoss ──────────────────────────────────
  │  loss_pct > stop_loss_pct                       │
  │  IM=0.3%, IC=0.5%                               │
  │  → STOP_LOSS (URGENT)                           │
  ─────────────────────────────────────────────────

  P15 ── LunchClose ────────────────────────────────
  │  time in 11:25~13:00:                           │
  │    亏损 → LUNCH_CLOSE (URGENT)                  │
  │    盈利 + trailing_drawdown > 0.3%              │
  │         → LUNCH_TRAIL                           │
  ─────────────────────────────────────────────────

  P30 ── TrailingStop ──────────────────────────────
  │  trail% = f(hold_time):                         │
  │    <15min: 0.5%                                 │
  │    15-30min: 0.6%                               │
  │    30-60min: 0.8%                               │
  │    >60min: 1.0%                                 │
  │  if 15m趋势确认 + 盈利>0.5%: +0.2%             │
  │  trail% × trailing_stop_scale (IM=1.5x,IC=2.0x)│
  │  if trailing_drawdown > trail% AND profitable   │
  │  → TRAILING_STOP                                │
  ─────────────────────────────────────────────────

  P40 ── TrendComplete ────────────────────────────
  │  LONG:  zone_5m=ABOVE_UPPER AND                 │
  │         zone_15m=ABOVE_UPPER                    │
  │  SHORT: zone_5m=BELOW_LOWER AND                 │
  │         zone_15m=BELOW_LOWER                    │
  │  → TREND_COMPLETE                               │
  ─────────────────────────────────────────────────

  P50 ── MomentumExhausted ────────────────────────
  │  hold_time >= 20min                             │
  │  AND narrow_range(3bar) < me_ratio              │
  │      (IM=0.10, IC=0.12)                         │
  │  AND NOT price_trending(3bar)                   │
  │  AND zone_15m in 极端zone                       │
  │  → MOMENTUM_EXHAUSTED                           │
  ─────────────────────────────────────────────────

  P60 ── MidBreak ─────────────────────────────────
  │  price破布林中轨 连续 >= 3根bar                 │
  │  AND zone_15m在对侧                             │
  │  → MID_BREAK                                    │
  ─────────────────────────────────────────────────

  P70 ── TimeStop ─────────────────────────────────
  │  hold_time > 60min AND pnl_pct <= 0             │
  │  → TIME_STOP                                    │
  ─────────────────────────────────────────────────

  无条件触发 → NO_EXIT
```

### Exit时间约束

| 约束 | 时间 (BJ) | UTC | 行为 | 可配置 |
|------|----------|-----|------|--------|
| 午休平仓（亏损） | 11:25 | 03:25 | LUNCH_CLOSE 强平 | exit_weights可禁用 |
| 午休紧trailing（盈利） | 11:25~13:00 | 03:25~05:00 | 回撤>0.3% → LUNCH_TRAIL | exit_weights可禁用 |
| 尾盘强制平仓 | 14:45 | 06:45 | EOD_CLOSE 强平 | **不可禁用** |

---

## 数据流总览

```
TQ 5min K线 → bar_data[symbol]
                │
    ┌───────────┴───────────────┐
    │                           │
    ▼                           ▼
 Entry侧                    Exit侧
 score_all()                check_exit()
    │                           │
    ▼                           ▼
 原子因子计算               原子因子计算
 momentum                   pnl_pct
 atr_ratio                  trailing_drawdown
 volume_percentile          boll_zone (5m+15m)
 boll_zone                  narrow_range
    │                       hold_time
    ▼                           │
 因子加权求和                    ▼
 M+V+Q+B+S                 按优先级评估
    │                       P10→P15→P30→P40→P50→P60→P70
    ▼                           │
 乘数管道                       ▼
 ×dm ×f ×tw ×sent          EXIT / NO_EXIT
    │
    ▼
 过滤 + 阈值检查
    │
    ▼
 OPEN / Skip
```
