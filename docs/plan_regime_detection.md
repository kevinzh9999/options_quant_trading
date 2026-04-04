# 项目计划：期权指标 Regime Detection

> 目标：用期权市场数据（IV/RV/Skew）构建 intraday_regime_score，辅助判断当天适合日内交易还是多日持仓。
>
> 创建：2026-04-04

---

## 预判与核心假设

**大部分期权指标很可能只是振幅的滞后代理。** IV/RV用过去N天数据计算，而振幅过滤器用当天开盘30分钟实时数据——实时数据天然比历史衍生指标更快。

**但有一个指标可能有独立价值：IV期限结构倒挂。** 近月IV>远月IV意味着市场预期短期内有事件冲击——这个信息是振幅过滤器看不到的（事件还没发生，开盘30分钟可能很平静，但午后突然爆发）。Phase 2需重点验证此指标。

**无论regime score最终是否有用，Phase 1的数据回算都必须做。** 补全215天的IV/RV/RR数据对其他研究（象限监控历史验证、sentiment_mult校准、VRP分位数准确性）都有长期价值。

---

## 数据基础

| 数据源 | 条数 | 范围 | 用途 |
|--------|------|------|------|
| `options_daily` | 763K | 2019~2026 | 回算ATM IV、RR、IV term structure |
| `daily_model_output` | 近期 | 2026-03~ | atm_iv/vrp/rr_25d（需补全） |
| `index_daily` | 10.8K | 2015~2026 | RV计算（close-to-close） |
| `futures_daily` | 44.5K | 2015~2026 | 期货基差/贴水 |

### 关键已有代码
- `scripts/portfolio_analysis.py:get_atm_iv()` — 从options_daily回算ATM IV（PCP Forward）
- `scripts/vol_monitor.py:calc_market_atm_iv()` — 市场ATM IV（期货价格based）
- `models/volatility/realized_vol.py` — RV计算器（close-to-close/Parkinson/GK）
- `models/volatility/garch_model.py` — GJR-GARCH
- `scripts/daily_record.py:_eod_model_output()` — 当前IV/RV/VRP计算流水线

---

## Phase 1：历史数据回算（1-2天）⬅ 优先执行

### 1.1 回算ATM IV（215天）

```python
# scripts/backfill_iv_history.py
# 逐日从 options_daily 回算 ATM IV → 写入 daily_model_output.atm_iv
# 复用 portfolio_analysis.py:get_atm_iv() 和 vol_monitor.py:calc_market_atm_iv()

for trade_date in dates_to_fill:
    chain = db.query_df("SELECT * FROM options_daily WHERE trade_date=? AND ts_code LIKE 'MO%'")
    spot = get_spot_close(trade_date)  # from index_daily
    atm_iv, source = get_atm_iv(spot, chain, trade_date)
    market_iv = calc_market_atm_iv(chain, ..., futures_price)
    # UPDATE daily_model_output SET atm_iv=?, atm_iv_market=? WHERE trade_date=?
```

注意事项：
- DTE过滤：≥14天（排除近到期Gamma distortion）
- 优先14-45天到期的合约
- options_daily 的 underlying_code 为NULL，需从 ts_code parse

### 1.2 回算 RV_5d

```python
# 从 index_daily 的 close 计算
# RV_5d = std(log_returns[-5:]) * sqrt(252)
# 写入 daily_model_output 新字段 realized_vol_5d
```

### 1.3 回算 25D Risk Reversal

```python
# 复用 daily_record.py 中的 RR 计算逻辑
# 25D RR = IV(25D Call) - IV(25D Put)
# 需要插值法找25D strike
```

### 1.4 回算 IV Term Structure（重点）

```python
# 新字段 iv_term_spread = 近月ATM_IV - 远月ATM_IV
# 正值 = 近月贵（短期恐慌）= 倒挂
# 负值 = 远月贵（正常期限结构）
```

**为什么IV期限结构是重点**：倒挂意味着市场预期短期冲击——开盘可能平静但午后爆发。这是振幅过滤器的盲区（振幅过滤只能看到已发生的波动，看不到"即将发生"的波动）。

**Phase 1 产出**：`daily_model_output` 补全215天的 atm_iv, atm_iv_market, realized_vol_5d, rr_25d, iv_term_spread

**附加价值（即使regime score无用）**：
- 象限监控可验证历史象限判断的准确性
- sentiment_mult可用更长历史校准参数
- VRP分位数计算更准确（样本从30天扩大到215天）

---

## Phase 2：相关性验证（1天）⬅ 优先执行

### 2.1 候选指标

| 指标 | 计算 | 假设 | 预判 |
|------|------|------|------|
| RV_5d/RV_20d | 短期/长期已实现波动率比 | >1.2 = 日内波动放大 | 可能是振幅滞后代理 |
| IV percentile | ATM IV在1年历史中的分位 | P80+ = 高波动regime | 可能是振幅滞后代理 |
| **IV term spread** | **近月IV - 远月IV** | **>0 = 短期恐慌/事件预期** | **最可能有独立价值** |
| RR change | \|rr_25d - rr_25d_prev\| | 急升 = 恐慌加剧 | 待验证 |
| VRP | IV - blended RV | >0 = 期权贵 | 方向性弱 |
| IV/RV ratio | ATM_IV / RV_20d | >1.3 = 期权溢价高 | 可能和IV percentile冗余 |

### 2.2 验证矩阵

对215天每天计算：
1. 候选指标 vs 当天日内策略PnL → Pearson r
2. 候选指标 vs 当天日内振幅 → Pearson r（检查是否只是振幅代理）
3. 控制日内振幅后的偏相关 → 增量预测力

**核心问题**：期权指标能否提供比"开盘30min振幅"更早（盘前可知）或更独立的信息？

**重点验证 IV term spread**：
- 找出IV倒挂日（iv_term_spread > 0），看这些日子的振幅和PnL分布
- 特别关注"开盘振幅低但全天振幅高"的日子——如果IV倒挂能预测这类日子，就是真正的增量

### 2.3 判断标准

- 如果候选指标与PnL的相关性 < 0.1 → 无用
- 如果候选指标与振幅的相关性 > 0.8 → 只是振幅的代理，无增量
- 如果偏相关（控制振幅后）仍 > 0.15 → 有独立增量，值得纳入

### 2.4 决策点

Phase 2结束后做Go/No-Go决策：
- **Go**：至少1个指标有独立增量 → 继续Phase 3
- **No-Go**：所有指标都只是振幅代理 → 停止，Phase 1数据仍有长期价值

---

## Phase 3：Regime Score 构建（1天）— 仅Phase 2通过后执行

### 3.1 构建综合指标

```python
intraday_regime_score = (
    w1 * normalize(RV_5d/RV_20d) +
    w2 * normalize(IV_percentile) +
    w3 * normalize(IV_term_spread) +
    w4 * normalize(RR_change_speed)
)
```

权重由Phase 2的相关性分析决定，或等权起步。

### 3.2 回测验证

三种方案对比：
- A: 单独振幅过滤（当前已实施）
- B: 单独regime_score过滤
- C: 振幅过滤 + regime_score（两者AND/OR）

### 3.3 应用方式（由弱到强）

1. **面板显示**：morning_briefing输出regime_score，操作者参考
2. **阈值调节**：regime_score低时提高信号阈值（60→70）
3. **直接过滤**：regime_score极低时不开仓（替代或补充振幅过滤）

---

## Phase 4：集成（0.5天）— 仅Phase 3验证有效后执行

- `morning_briefing.py` 输出当天 regime_score
- `monitor.py` 面板显示
- 作为 d_override 的辅助参考

---

## 风险

1. ATM IV回算质量取决于options_daily的流动性（深度虚值合约可能有噪音）
2. **大部分期权指标很可能只是振幅的滞后代理**（IV/RV用历史数据，振幅过滤用实时数据）
3. 215天样本做多指标组合容易过拟合
4. VRP/IV percentile需要更长历史数据才稳定

## 成功标准

- Phase 2 发现至少1个指标的偏相关（控制振幅后）> 0.15
- Phase 3 组合后比单独振幅过滤多改善 > 100pt/215天
