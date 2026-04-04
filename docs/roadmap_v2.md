# SignalGenerator 升级蓝图 V2（2026-04-02更新）

> 基于 101 Formulaic Alphas 论文、现代量化前沿扫描、及 V2 系统34天实盘/回测验证的综合修订版。
> 核心原则：**先榨干现有数据的线性逻辑 → 再引入外部数据 → 最后才上非线性模型**。
> 最后更新：2026-04-04

---

## 当前系统评估

### 已有维度

| 维度 | 衡量内容 | 信息源 | 101 Alphas对应 |
|------|---------|--------|---------------|
| M分（0-50） | 动量方向和速度 | 5m/15m close差值 | Alpha#19, #101 |
| V分（0-30） | 波动率扩张/收缩 | ATR短期/长期比 | 部分覆盖 |
| Q分（0-20） | 成交量确认 | 当前volume/20根均量 | Alpha#12 |
| B分（0-20） | 布林带突破 | 中轨穿越+放量+多周期 | Alpha#32 |
| daily_mult | 多周期方向一致 | 日线5日动量 | Alpha#19 |
| sentiment_mult | 期权情绪 | IV/VRP/Skew/PCR（日频） | — |
| Z-Score过滤 | 极端偏离保护 | EMA20/STD20 | Alpha#32 |
| 趋势启动检测器 | 突破+放量+振幅三重确认 | 5m OHLCV | — |
| Hurst(60d) | 趋势/震荡regime | 日线R/S分析 | — |

### 已验证不加的维度（2026-04-04完整测试）

| 维度 | 测试方法 | 结果 | 原因 | 脚本 |
|------|---------|------|------|------|
| ADX趋势强度乘数 | 事后模拟⚠ | IM+IC -108pt | 低ADX反而高WR，乘数打折好交易 | adx_macd_backtest.py |
| MACD顺逆势确认 | 事后模拟⚠ | IM+IC -86pt | IC中逆势WR=57%，无预测力 | adx_macd_backtest.py |
| ADX+MACD组合 | 事后模拟⚠ | IM+IC -161pt | 两个负效果叠加更差 | adx_macd_backtest.py |
| CCI/Williams%R | 相关性分析 | r>0.9 | 和M/V/Q高度冗余 | adx_research.py |
| Z-Score极值bonus | 直接加total ✅ | IM-12/IC+24pt | 现有Z-Score+RSI层已覆盖 | mean_reversion_research |
| 分级冷却 | 独立回测循环 ✅ | -136~-335pt | 延长冷却全部亏钱 | afternoon_cooldown_research.py |
| OU动态TIME_STOP | grid search | 60min最优 | 半衰期日间波动太大(std>mean) | mean_reversion_research |
| ~~VWAP偏离度~~ | 相关性+回测 ✅ | r(M)=0.67 | **与M高度冗余**，非独立维度 | vwap_research.py |
| ~~K线实体占比~~ | 独立回测循环 ✅ | IC -252pt | IM有效但**IC灾难性**，品种差异过大 | body_ratio_research.py |
| ~~跨品种rank~~ | 独立回测循环 ✅ | -350~-598pt | rank过滤误杀好信号 | cross_rank_research.py |
| 午后t=0.8 | session_multiplier patch ✅ | t=1.0 +76pt | **已改为1.0（实施）** | 手动验证 |
| ME/TC score确认退出 | 独立回测循环 ✅ | +130pt但不稳健 | 前半段-58pt，阈值敏感 | score_confirmed_exit_research.py |
| 隔夜策略 | 纯统计（样本不足） | 数据不足 | 需要2-3年5分钟数据 | overnight_strategy_research.py |

⚠ = 事后模拟（乘在最终score上），与真实集成可能有偏差，但结论方向可靠。
✅ = 验证方式正确。

### 核心发现

**101 Alphas 的三个高频因子全部验证无效**：
- VWAP偏离：与M冗余r=0.67（本质是动量的时间积分）
- K线实体占比：品种差异过大（IM有效IC有害）
- 跨品种rank：统计不显著（p>0.6），乘数方案-50%PnL

**14个OHLCV因子全面测试完毕（2026-04-02，101 Alphas高频子集）**：
- 所有常见技术指标（ADX/MACD/CCI/Williams%R/VWAP/K线形态/跨品种rank）全部验证无增量价值
- 共测试20+方向，全部不实施

**结论**：当前M/V/Q+乘数链的架构已接近5分钟OHLCV数据的信息提取上限。进一步改善需要引入新数据源（期权实时数据、tick数据）或更长的时间积累。

**平仓系统已优化至较优参数（2026-04-04）**：
- ME narrow_ratio=0.10, MID_BREAK bars=3, IC trailing_scale=2.0x
- 当前baseline IM+IC +2031pt，breakeven滑点约 5-6pt
- 进一步改善需等ME/TC score确认方案（100天数据后验证）

---

## ~~V2.1：数学重构层~~ → 已完成测试，全部不实施

> 2026-04-02完成全部3项验证。101 Alphas的高频因子在A股5分钟OHLCV上无增量价值。

### ~~1. VWAP偏离度~~ ❌ 已验证不实施

**测试结果**（vwap_research.py）：
- r(VWAP_offset, M_proxy) = **0.67~0.68**，远超0.5独立性阈值
- VWAP偏离本质是日内价格动量的时间积分，和M高度冗余
- "便宜处买"WR=48%反而不如"追涨"WR=54%——VWAP是动量延续信号，非逆向
- lag1_ACF=0.85，半衰期25-30分钟

**潜在用途**（不作为评分维度，但可用于执行层）：
- 执行时机优化：信号触发后等价格回到VWAP附近再入场
- 持仓止盈参考：偏离VWAP±1%时回归风险增加

### ~~2. K线实体占比~~ ❌ 已验证不实施

**测试结果**（body_ratio_research.py）：
- IM有效：强阳线做多WR=72% vs 阴线矛盾WR=44%，方案B(3根一致性) +35pt
- IC有害：阴线矛盾组反而+4.9pt/笔，方案A **-252pt(-33.6%)**
- 品种差异太大，无法作为通用调节器
- 方案B合计仅+2pt，不值得增加复杂度

**有价值发现**（供后续研究）：
- IM做多+下影线大(>0.5)：WR=38.5%（陷阱信号）
- SHORT+中上影线(0.3~0.5)：均PnL=+13.8pt（最佳信号）

### ~~3. 跨品种相对强弱rank~~ ❌ 已验证不实施

**测试结果**（cross_rank_research.py）：
- rank与交易结果Pearson相关全部不显著（p>0.6）
- 方案A(rank乘数)：IM+IC **-598pt(-50%)**
- 方案B(方向感知)：IM+IC **-350pt(-29%)**
- rank过滤误杀好信号

**有价值发现**（待数据积累后深入研究）：
- IM-IH style spread中性区间（-0.09%~+0.04%）：73%WR，均+13.8pt，共+510pt
- 原因：中性spread=全市场共识移动（非风格轮动），IM信号质量最高
- 待100+笔样本后验证是否可作为"信号质量过滤器"

---

## V2.5：重心与情绪层（需引入衍生数据）

> 难度：⭐⭐ | 周期：2周 | 数据需求：期权实时数据、VWAP累计

### 1. 量价相关性（补充Q分，不替代）

**灵感来源**：Alpha#2, #6, #13, #44频繁使用`correlation(price, volume, N)`。

**公式**：
```
pv_corr = correlation(close_5m[-10:], volume_5m[-10:])  # 10根bar滚动相关性
```

**系统接入**：作为Q分的惩罚/确认系数：
```python
if pv_corr < -0.3 and direction == "LONG":
    # 缩量上涨（量价背离），M分打折0.8
elif pv_corr > 0.5 and direction matches price_direction:
    # 量价齐升/齐跌，Q分额外+5
```

**实施策略**：先作为独立信号跑2周shadow观察，不计入总分。如果发现pv_corr极端负值时原本高分交易的WR显著下降，再接入。

### 2. 期权Skew实时化（升级sentiment_mult）

**当前问题**：sentiment_mult使用日频数据（ATM IV变化、RR），盘中MO的Skew可能在5分钟内剧变但sentiment_mult不会更新。

**升级方案**：
```python
# vol_monitor已经每5分钟计算25D RR
# 接入monitor，作为实时sentiment_mult的补充
rr_5m = vol_monitor.get_current_25d_rr()  # 当前5分钟的25D RR

if rr_5m > 8.0:  # Put skew极端（如今天的+10.3pp）
    sentiment_mult *= 0.85  # 惩罚做多
elif rr_5m < 2.0:  # Call skew极端
    sentiment_mult *= 0.85  # 惩罚做空
```

**数据可用性**：vol_monitor已经在计算，只需要建立monitor→vol_monitor的数据通道。

### 3. Hurst动态权重（等低波动数据积累后）✅ 部分完成

**已完成**（2026-04-02）：
- Hurst(60d)加入象限监控：A1/A2/B1/B2细分（趋势vs震荡）
- morning_briefing显示Hurst值和历史分位
- 当前IM Hurst=0.78（P97，极强趋势期）→ B2象限（禁止卖方）

**30天回测验证**（hurst_research.py）：
- Hurst对波动率有预测力（p<0.001），对方向无预测力
- 自相关lag1>0.85，状态切换慢
- 但30天回测全在H>0.6，**无低Hurst样本**，无法验证震荡期效果

**待做**（需要低Hurst数据积累）：
- [ ] 至少10天H<0.45数据后验证：低Hurst时逆势信号是否更好
- [ ] 高Hurst时顺势信号是否可以降低阈值（-5分）
- [ ] Hurst动态调整dm_trend/dm_contrarian的比例

---

## V3.0：微观结构层（需引入Tick/L2数据）

> 难度：⭐⭐⭐⭐ | 周期：1-2月 | 数据需求：Tick级别数据

### 1. 订单流不平衡近似（OFI Proxy）

**数据限制**：TQ标准API不提供逐笔成交的买卖方向标记。

**近似方案**：
```python
# 用tick级别的价格变化方向 × 成交量变化近似BSI
BSI_proxy = sum(ΔVolume * sign(ΔPrice))  # 每5分钟内累计
```

**风险**：噪音很大，信噪比可能不够。必须先确认TQ的tick数据频率和字段，再评估可行性。

**替代方案**：如果TQ有bid1/ask1的快照数据，可以计算盘口不平衡：
```python
OBI = (bid1_volume - ask1_volume) / (bid1_volume + ask1_volume)
```

### 2. Volume Profile / POC

**数据需求**：需要tick级别或1分钟级别的成交分布。用5分钟K线只能做粗略近似。

**近似方案**：
```python
# 用5分钟K线的typical_price=(H+L+C)/3和volume构建价格-成交量分布
# POC = 成交量最大的价格区间
price_bins = np.linspace(day_low, day_high, 20)
volume_profile = np.histogram(typical_prices, bins=price_bins, weights=volumes)
poc = price_bins[np.argmax(volume_profile)]
```

**系统接入**：用`(Close - POC) / ATR`作为偏离度指标，补充或替代B分的布林带逻辑。

---

## V4.0：状态机与非线性融合（需大量数据积累）

> 难度：⭐⭐⭐⭐⭐ | 周期：6月+ | 前置条件：500+笔交易数据

### 1. HMM环境概率底座（辅助角色，不替代硬过滤）

**定位**：HMM输出市场状态概率（震荡/趋势/恐慌），作为因子权重的动态调节器。

**关键约束**：
- ⚠ HMM**不替代**Z-Score硬过滤（Z-Score是确定性安全阀，可解释性100%）
- HMM只能微调M/V/Q的相对权重（如震荡时降低M权重、提高V权重）
- 异常状态必须有fallback到确定性规则的机制

**实施**：
```python
# HMM输出
state_probs = hmm.predict_proba([features])  # [P_震荡, P_趋势, P_恐慌]

# 动态权重调节（不触碰Z-Score安全阀）
if state_probs[0] > 0.7:  # 高概率震荡
    m_weight = 0.7  # 降低动量权重
    v_weight = 1.3  # 提高波动率权重
elif state_probs[1] > 0.7:  # 高概率趋势
    m_weight = 1.3
    v_weight = 0.7
else:
    m_weight = 1.0
    v_weight = 1.0
```

### 2. LightGBM树模型打分（终极进化）

**前置条件**：
- 至少500笔交易数据（约6个月×4品种）
- 完整的特征工程pipeline（V2.1-V3.0的所有因子）
- 独立的训练/验证/测试集划分（时间序列分割，不能随机）

**架构**：
```python
# Features: M, V, Q, B, VWAP_offset, body_ratio, cross_rank, pv_corr, skew_5m, hurst, ...
# Label: 未来15分钟的return > 0 ? 1 : 0
# 或者回归: 未来15分钟的return

model = LightGBMClassifier(
    max_depth=4,       # 严格限制防过拟合
    n_estimators=100,
    min_child_samples=20,  # 每个叶子至少20个样本
    subsample=0.8,
    colsample_bytree=0.7,
)
```

**关键防护**：
- 树深度限制max_depth=4（防过拟合）
- Walk-forward验证（不是简单的train/test split）
- 模型输出的probability score仍需经过Z-Score安全阀

---

## 实施时间线（2026-04-04更新）

```
2026-04 V2.1  ✅ 已完成测试（VWAP/Body Ratio/Cross Rank/ME-Score全部不实施）
              ✅ Hurst加入象限监控（A1/A2/B1/B2细分）
              ✅ 午后session_weight改1.0（+76pt）
              ✅ ME narrow_range ratio 0.20→0.10（+393pt）
              ✅ MID_BREAK bars 2→3（+76pt）
              ✅ IC trailing_stop_scale=2.0x（+186pt）
              ✅ 15分钟重采样修复 label='left'（关键bug修复）
              ✅ 动量lookback硬编码修复（IF/IH 18→12）
              ✅ 研究指标自动记录（ADX/body_ratio/VWAP/style_spread/cross_rank）
              ✅ hurst_60d + rr_25d写入daily_model_output
              IM+IC baseline: +2031pt（+68% vs 03-26旧baseline）
              
2026-05 V2.5  量价correlation + 期权Skew实时化
              ↓ 积累低Hurst数据 + style spread样本（100+笔）
2026-06 V3.0  OFI近似（确认TQ数据后）+ Volume Profile
              ↓ 积累500+笔交易
2026-10 V4.0  HMM辅助（需pip install hmmlearn）+ LightGBM打分
```

### Follow-up（待数据积累后验证）

| 项目 | 需要数据 | 预计可测时间 |
|------|---------|------------|
| ADX>35硬过滤 | 100+笔ADX>35样本 | 2026-07（约3个月后） |
| MACD顺逆势 | IM中MACD>0做空50+笔 | 2026-07 |
| IM-IH style spread过滤器 | 100+笔中性区间交易 | 2026-07 |
| Hurst动态dm调整 | 10+天H<0.45数据 | 视市场波动率 |
| IM body_ratio单品种 | 100+笔IM交易 | 2026-06 |
| 隔夜策略验证 | index_min 2-3年数据 | 2027+ |

---

## 工程方法论

### 新因子接入检查清单

每个新因子必须通过以下验证才能进入信号系统：

1. **独立性检验**：和现有M/V/Q的Pearson相关性 < 0.5
2. **预测力检验**：按因子值分组（4组），各组WR和avg PnL有显著差异
3. **单日验证**：用已知受影响的日期确认因子值变化符合预期
4. **Patch正确性**：确认修改的变量是score_all实际调用的（教训：`_TIME_WEIGHTS` vs `session_multiplier`）
5. **30天回测**：加入因子后总PnL不低于baseline
6. **Sensitivity分析**：因子参数在合理范围内（±20%）结果稳健

### 过拟合防护

- 35天/100笔交易能支撑的独立参数：5-8个（当前已接近上限）
- 每新增1个参数，需要额外~15笔交易数据支撑
- 新因子优先作为乘数/过滤器（0个新参数），其次作为评分维度（1-2个新参数）
- 所有研究脚本必须验证patch是否对目标函数生效

### 关键教训（2026-04-04更新）

1. 研究脚本monkey-patch了错误的变量（`_TIME_WEIGHTS` vs `session_multiplier`）→ 假阳性结论
2. 事后模拟（乘在最终score上）≠ 真实集成（加进raw_total）→ 结论方向可能有偏差
3. "有益偏差"不应修复（跨日GAP、forming bar Q=0）→ 修复后反而亏钱
4. 现货/期货价格混用是致命bug → 持仓管理必须统一价格源
5. 高相关性 ≠ 有预测力，低相关性 ≠ 无预测力 → 必须用回测数据验证
6. **时间标签对齐影响全部时间相关逻辑**：`resample('15min', label='right')`使15m bar时间偏移15分钟，影响持仓时间/时段权重/exit判断。永远用`label='left', closed='left'`
7. SYMBOL_PROFILES中的参数只有在代码真正读取时才生效，硬编码会静默覆盖配置 → 修复前需确认读取路径
8. **稳健性验证三项指标**：时间分段一致性 + 阈值敏感性(±20%) + 单日贡献<30%。三项全过才实施
