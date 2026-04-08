# 日内策略研究路线图（2026-04-08 V3合并版）

> 基于方案E baseline（IM+IC **+4956pt/216天**，均+23pt/天）。
> 核心原则：**先榨干现有参数空间 → 再挖掘新数据源alpha → 最后做策略品类扩展**
> 过拟合防线：每项改动必须通过稳健性三检（时间分半、参数邻域、单日贡献<30%）

---

## 一、当前系统评估

### 已有维度

| 维度 | 衡量内容 | 信息源 | 当前状态 |
|------|---------|--------|---------|
| M分（0-50） | 动量方向和速度 | 5m/15m close差值，动态lb=4/12 | 方案E核心改进 |
| V分（0-30） | 波动率扩张/收缩 | ATR短期/长期比 | 唯一正Daily IC因子，已最优 |
| Q分（0-20） | 成交量确认 | 当前volume/20根均量 | 弱正IC，稳定 |
| B分（0-20） | 布林带突破 | 中轨穿越+放量+多周期 | Daily IC为负，但经乘数链过滤后正贡献 |
| daily_mult | 顺逆势方向 | 日线5日动量 | dm=1.1/0.9（04-04调整）|
| sentiment_mult | 期权情绪 | IV/VRP/Skew/PCR（日频） | 04-04验证偏相���<0.15 |
| Z-Score过滤 | 极端偏离保护 | EMA20/STD20 | 安全阀，不动 |
| 振幅过滤 | regime切换 | 开盘30min振幅 | r=0.43最强PnL预测因子 |

### 已验证不实施的方向（20+方向，完整测试）

| 方向 | 结果 | 原因 | 脚本 |
|------|------|------|------|
| VWAP偏离度 | r(M)=0.67 | 与M高度冗余 | vwap_research.py |
| K线实体占比 | IC -252pt | 品种差异过大 | body_ratio_research.py |
| 跨品种rank | -350~-598pt | 统计不显著 | cross_rank_research.py |
| ADX/MACD/CCI/WR | 全部负或冗余 | 与M/V/Q高度共线 | adx_macd_backtest.py |
| Z-Score极值bonus | IM-12/IC+24pt | 现有层已覆盖 | mean_reversion研究 |
| 分级冷却 | -136~-335pt | 延长冷却全部亏钱 | afternoon_cooldown |
| ME/TC score确认退出 | +130pt但不稳健 | 前半段-58pt | score_confirmed_exit |
| 101 Alphas | 全部无效 | 截面选股因子不适用单品种时序 | factor_research.py |
| BAND_REVERSAL退出 | -32% | 因子有效但独立exit劣化 | band_reversal研究 |

**核心结论**：5分钟OHLCV因子空间已到信息提取上限。进一步alpha须来自新数据源。

### 数据资产盘点

| 资产 | 规模 | 时间范围 | 状态 |
|------|------|---------|------|
| index_min (000852) | 258K bars | 2022-07~2026-04 | 回测核心 |
| index_min (000300/905/016) | 各577K bars | 2018-01~2026-04 | IC/IF/IH回测 |
| futures_tick (IM) | 24M rows | 2022-07~2026-04 | **未开发** |
| futures_tick (IC/IF) | 各58-59M rows | 2016-01~2026-04 | **���开发** |
| options_min (MO/IO/HO) | 24M rows | 2020-02~2026-04 | **未开发** |
| shadow_trades | 23笔 | 2026-04-01~08 | 实盘刚起步 |

---

## 二、历史版本演进

### V2.1（2026-04初）✅ 已完成

全部在5分钟OHLCV维度内优化，因子研究+参数调优+bug修复。

```
✅ 20+因子/指标全面测试（全部不实施，确认OHLCV上限）
✅ ME narrow_range ratio 0.20→0.10（+393pt）
✅ MID_BREAK bars 2→3（+76pt）
✅ IC trailing_stop_scale=2.0x（+186pt）
✅ 午后session_weight 0.8→1.0（+76pt）
✅ dm_trend/dm_contrarian 1.2/0.8→1.1/0.9（+691pt差距）
✅ IM trailing_stop_scale 1.0→1.5x（+259pt）
✅ 开盘30min振幅<0.4%过滤器
✅ 15分钟重采样对齐修复 label='left'（关键bug）
✅ 动量lookback硬编码修复（IF/IH 18→12）
✅ 研究指标自动记录（ADX/body_ratio/VWAP/style_spread/cross_rank）
✅ Hurst(60d) + rr_25d 写入daily_model_output
Baseline: IM+IC +2031pt
```

### 方案E（2026-04-07~08）✅ 已完成

动态M分lookback + d_override禁用 + 阈值下调。

```
✅ 动态lb=4/12（振幅<1.5%用4，否则12）← Phase 1"低振幅日IC为负"的直接应用
✅ signal_threshold：IM 60→55，IC 65→60
✅ d_override禁用（briefing dm累计-19.2%负贡献）
✅ EOD重构：四库分离 + DataDownloader并行归档
Baseline: IM+IC +4956pt（+20.2% vs V2.1）
```

---

## 三、Phase 1：Exit系统精细化 ✅ 已完成（2026-04-08）

**目标**：不改入场逻辑，通过优化平仓参数提升5-10%。
**结论**：**当前exit参数配置已是最优，无进一步优化空间。**

### 1.1 动态Trailing Stop ❌ 不实施

ATR-based trailing（开盘30min ATR × K替代固定阶梯）全面劣于当前固定阶梯。
- 897天全量sweep，K=[0.8, 1.0, 1.2, 1.5, 2.0, 2.5] 全部为负
- 最优K=1.5时 -158pt，最差K=0.8时 -900pt
- 原因：开盘ATR只反映早盘波动，日内波动可能剧变，固定阶梯反而更稳健
- 脚本：`sensitivity_215d.py --param P8`

### 1.2 动态Stop Loss ❌ 不实施

ATR-based止损（day_atr × N替代固定0.5%）全面劣于固定百分比。
- 897天全量sweep，N=[1.5, 2.0, 2.5, 3.0, 4.0] 全部为负
- 最优N=2.0时 -161pt，最差N=4.0时 -959pt
- 脚本：`sensitivity_215d.py --param P9`

### 1.3 ME × MID_BREAK 联合优化 ✅ 当前即最优

2D grid（6×4=24组合，217天）验证无交互效应。
- 当前(ME=0.10, MID=3) = +5410pt 就是联合最优
- MID=3列全面最优，ME=0.10行最优
- 独立最优 = 联合最优，无交互效应
- 脚本：`sensitivity_2d_me_mid.py`

### 1.4 LUNCH_CLOSE ❌ 暂不实施

2D grid（4×5=20组合，217天）发现边缘改善但不稳健。
- 最优：11:25 + 0.2% trailing = +5557pt（vs当前+5410，+147pt / +2.7%）
- 但0.2%是边缘值（0.3/0.4/0.5%都≈+5410，只有0.2%跳高），过拟合风险
- 时间维度：11:25确认是最优时间点
- 脚本：`sensitivity_2d_lunch.py`


---

## 四、Phase 2：Tick数据Alpha挖掘（2-4周）

**目标**：利用28.5GB tick数据（1.4亿行），构建5分钟OHLCV之外的新信息维度。这是**突破当前信息源瓶颈**的核心一步。

### 2.1 基础设施：Tick→5min聚合Pipeline

tick数据在`tick_data.db`中，字段：`last_price, bid_price1, bid_volume1, ask_price1, ask_volume1, volume, amount, open_interest`。

**建设内容**：
```python
# models/factors/tick_aggregator.py
def aggregate_tick_to_5min(symbol, date) -> pd.DataFrame:
    """聚合为5分钟级微观结构指标，与index_min时间对齐"""
    return DataFrame with columns:
        ofi,              # 订单流不平衡（Cont 2014）
        obi,              # 盘口不平衡 (bid1_vol - ask1_vol) / total
        trade_count,      # 成交笔数
        spread_mean,      # 平均bid-ask spread (bp)
        large_trade_pct,  # 大单占比（>P95阈值）
        price_impact,     # 单位成交量的价格变化
        oi_change,        # 5分钟持仓量变化
```

**缓存**：结果写入`tick_features_5m`新表，避免回测重复计算。
**数据覆盖**：IM从2022-07开始，覆盖216天全部回测日。

### 2.2 OFI（订单流不平衡）因子

经典OFI定义（Cont, Kukanov & Stoikov, 2014）：
```python
# 每个tick：
#   bid_price上升 → bid贡献 = +Δbid_volume
#   bid_price下降 → bid贡献 = -bid_volume_prev
#   ask_price下降 → ask贡献 = +Δask_volume
#   ask_price上升 → ask贡献 = -ask_volume_prev
# OFI_5min = Σ(bid贡献 - ask贡献)
```

**接入方式**：
- 先跑因子评估（IC/Daily IC/分组收益/regime分组）
- 和M/V/Q的相关性必须<0.5
- 通过后作为新的O分（0-15分），或替代/增强Q分

### 2.3 盘口不平衡（OBI）因子

```python
OBI = mean over 5min of [(bid1_volume - ask1_volume) / (bid1_volume + ask1_volume)]
```

更简单的即时买卖压力指标。A股期货盘口深度较浅，噪音可能大，需要验证信噪比。

### 2.4 成交强度（Trade Intensity）

```python
trade_intensity = count(ticks in 5min) / rolling_mean(count, 48)  # vs 4小时均值
```

和Q分（成交量）的区别：同样1万手成交，100笔大单 vs 10000笔小单含义不同。

### 2.5 大单占比

```python
# 先统计P95阈值：SELECT PERCENTILE(volume_delta, 0.95) FROM tick WHERE ...
large_trade_pct = sum(vol WHERE single_trade_vol > P95) / total_vol
```

### Phase 2 预期与风险
- 保守：确认1-2个IC>0.03且独立的新因子
- 乐观：接入后+5-10%增量
- 风险：tick噪音大，OFI近似误差可能导致因子无效
- **关键里程碑**：2.1 Pipeline完成后，2.2-2.5可并行评估

---

## 五、Phase 3：期权微观结构Alpha（2-4周，可与Phase 2并行）

**目标**：利用24M行期权5分钟K线（options_min），构建option-implied日内信号。

**前置风险**：04-04已验证日频期权指标偏相关<0.15，但5分钟频率可能有不同结论。

### 3.1 实时PCR 5分钟版

```python
# 每5分钟计算当月MO的 Put/Call 成交量比
pcr_5m = sum(put_volume_5m for all MO puts) / sum(call_volume_5m for all MO calls)
pcr_change = pcr_5m / ema(pcr_5m, 12)  # vs 1小时均值
```

接入：作为sentiment_mult的实时补充（当前盘中不更新）。

### 3.2 ATM IV 5分钟变化

```python
iv_change_5m = (atm_iv_now - atm_iv_30min_ago) / atm_iv_30min_ago
# IV急升(>5%) → 恐慌 → 惩罚做多
# IV急降(>5%) → 乐观 → 惩罚做空
```

数据来源：从options_min用ATM合约close反推IV，或从vol_monitor获取。

### 3.3 期权成交量异动

```python
put_surge = put_vol_5m / ema(put_vol_5m, 12) > 3.0  # Put异常放量
# → 机构买保险 → 做多信号惩罚
```

### Phase 3 预期与风险
- 保守：仍然无增量（MO流动性有限，5min PCR噪音极大）
- 乐观：+3-5%增量（5min先导性 > 日频）
- 决策点：Phase 2的tick因子如果已提供足够增量，Phase 3可降优先级

---

## 六、Phase 4：跨品���信号增强（1-2周）

### 4.1 ���种共识度软乘数

之前cross_rank硬过滤亏钱（-350~-598pt），改为软乘数。

```python
# IM和IC同向动量 → 中小盘共识 → 信号更可靠
consensus = sign(mom_IM_5m) == sign(mom_IC_5m)
consensus_mult = 1.05 if consensus else 0.95  # 轻微调节
```

### 4.2 style_spread继续积累

IM-IH中性区间73%WR（37笔），但样本不足。等100+笔后重新评估。

### Phase 4 预期
- +0-2%。关键价值是信号质量辅助判断。

---

## 七、Phase 5：策略品类扩展（长期，4-8周+）

### 5.1 隔夜持仓策略

**方向**：
- 信号：尾盘14:30-14:55动量方向 + 成交量确认 + 振幅>0.5%
- 持有至次日09:45（避开开盘噪音），用锁仓实现
- 止损：前一日ATR × N
- 信号阈值高于日内（隔夜风险溢价）

**数据验证**：先统计"尾盘趋势延续到次日开盘"的历史频率和幅度。
**关键风险**：隔夜gap不可控，需单独的风控框架。
**前提**：index_min从2022-07有3.7年数据，但5分钟归档可能仅近1年，需确认。

### 5.2 波动率日内择时

结合VRP象限和日内信号：
- 高IV + VRP>0 时：日内动量策略（现有）+ 卖MO虚值期权（日内开平）
- 前提：MO日内bid-ask spread需可接受

### 5.3 IF实盘接入

shadow验证中（目前仅1笔），继续积累2-4周。
接入标准：30+笔，avg PnL > +2pt/笔（覆盖slippage+commission）。

---

## 八、实施时间线

```
2026-04 中旬  Phase 1: Exit系统精细化 ✅ 全部验证完毕，当前参数已最优
              ├── 1.1 动态trailing（ATR-based）        ❌ 不实施
              ├── 1.2 动态stop loss（ATR-based）       ❌ 不实施
              ├─��� 1.3 ME × MID_BREAK 联合优化
              └── 1.4 LUNCH_CLOSE参数sweep
              预��：+5-10%（+250~500pt）

2026-04底~05中  Phase 2: Tick数据Alpha
              ├── 2.1 Tick→5min聚合Pipeline          ← 基础设施优先
              ├── 2.2 OFI因子评估
              ├── 2.3 OBI���子评估
              ├── 2.4 ���交强度因子
              └── 2.5 大���占比因子
              预期：确认1-2个新因子，+5-10%

2026-05       Phase 3: 期权微观结构（可��Phase 2并行）
              ├── 3.1 5min PCR
              ├── 3.2 ATM IV 5min变化
              └── 3.3 期权成交量异动
              预期：+0-5%（可能无增量）

2026-05底     Phase 4: 跨品种增强
              ├── 4.1 品��共识度软乘数
              └── 4.2 style_spread继续积累
              预期��+0-2%

2026-06+      Phase 5: 策略品类扩展
              ├── 5.1 隔夜持仓策略研究
              ├── 5.2 波动率日内择时
              └── 5.3 IF实盘接入评估
              预期：新收益来源

2026-10+      V4.0: HMM/LightGBM非线性融合（需500+笔交易积累）
              ├── HMM环境概率底座（辅助角色，不替代Z-Score安全阀）
              └── LightGBM树模型打分（废除人工线性权重）
```

---

## 九、过拟合防护协议

### 验证标准

| 检验 | 方法 | 通过标准 |
|------|------|---------|
| 时间分半 | 前108天 vs 后108天 | 两半段均为正 |
| 参数邻域 | 最优参数±20% | 邻域内全正或高原 |
| 单日贡献 | 最大单日PnL / 总改善 | <30% |
| 逐月检查 | 12个月各自PnL | 亏损月<2个 |
| 样本外 | 保留最近30天不参与优化 | 样本���正 |

### 参数预算

- 当前独立参数：~10个
- 216天/~1100笔交易支撑上限：~15个
- **剩余预算：~5个**。新增参数必须从此预算扣除

### 新因子/参数引入决策树

```
因子IC > 0.03 且 vs现有因子相关性 < 0.5？
  ├── 否 → 放弃
  └── 是 → 216天回测PnL > baseline？
        ├── 否 → 放弃
        └── 是 → 稳健性三检全过？
              ├── 否 → 记录为"待数据积累"
              └── 是 → 参数预算够？
                    ├── 否 → 排队（或替换现有弱参数）
                    └── 是 → 实施
```

### 实盘退化判断标准

- 连续4周均PnL < +5pt/天 → 检查市场regime变化
- 连续2月合计亏损 → 参数重新校准
- 底线：+3.5pt/天（216天保守基线）

---

## 十、待数据积累后验证（Follow-up）

| 项目 | 需要数据 | 预计可测时间 |
|------|---------|------------|
| ADX>35硬过滤 | 100+笔ADX>35样本 | 2026-07 |
| IM-IH style spread | 100+笔中性区间 | 2026-07 |
| Hurst动态dm调整 | 10+天H<0.45 | 视市场波动率 |
| ts_rank lookback稳健性 | 400+天数据 | 2026-08 |
| V分增强研究 | 400+天��据 | 2026-08 |
| 隔夜策略完整验证 | index_min 2+年 | 2027+ |

---

## 十一、工程方法论

### 新因子接入检查清单

1. **独立性**：vs M/V/Q Pearson相关性 < 0.5
2. **预测力**：按因子值分4组，各组WR和avg PnL有显著差异
3. **单日验证**：用已知受影响日期确认因子值变化符合预期
4. **Patch正确性**：确认修改的变量是score_all实际调用的
5. **216天回测**：加入后PnL > baseline
6. **Sensitivity**：参数±20%结果稳健

### 关键教训

1. monkey-patch了错误变量（`_TIME_WEIGHTS` vs `session_multiplier`）→ 假阳性
2. 事后模拟（乘在最终score上）≠ 真实集成（加进raw_total）→ 结论方向有偏差
3. "有益偏差"不应修复（跨日GAP、forming bar Q=0）→ 修复后反而亏钱
4. 现货/期货价格混用是致命bug → 持仓管理必须统一价格源
5. 时间标签对齐影响全部时间相关逻辑 → 永远用`label='left', closed='left'`
6. SYMBOL_PROFILES参数只在代码真正读取时才生效 → 修复前确认读取路径
7. 稳健性三检（时间分段+参数邻域+单日贡献）三项全过才实施
8. 日频期权指标偏相关<0.15 → 不等于5分钟级别也无效，但要有预期管理
