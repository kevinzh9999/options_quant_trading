# 每日交易运营手册

> 每天按这个流程执行，养成纪律。核心目标：系统性地积累数据、验证信号、优化策略。

---

## 一、盘前准备（8:50 - 9:25）

### 执行

```bash
# 1. Morning Briefing（综合方向判断）
python scripts/morning_briefing.py

# 2. 查看昨日报告回顾
cat logs/eod/$(date -v-1d +%Y%m%d).md
cat logs/analysis/$(date -v-1d +%Y%m%d).md

# 3. 象限状态检查
python scripts/quadrant_monitor.py --once
```

### 思考和判断

**确定今日方向 Guidance（多/空/中性）**

需要回答的问题：
- 昨天的趋势是什么？是否有延续的动力？
- 昨日ATM IV是上升还是下降？VRP信号是什么？
- 昨日贴水有没有异常变化（突然加深=做空力量增强）？
- 有没有重大事件（经济数据、政策、外盘大跌）？
- 昨天的signal_log里，哪个品种的信号最强、方向是什么？
- 当前处于哪个象限（A/B/C/D）？象限建议的策略方向是什么？
- 持仓和象限是否匹配？如果不匹配，需要什么调仓？
- 参考 STRATEGY_PLAYBOOK.md 对应象限的操作建议

**确认持仓的风控线**

逐个持仓回答：
- 止损价是多少？距离当前价多远？
- 如果今天跳空到止损线以外怎么办？
- 期权持仓的安全距离够不够？卖方合约距离行权价多远？
- 今天有没有必须平仓的合约（到期、周五不过周末等）？

**写下今日计划**

用一句话写下来，例如：
- "今日偏空，IF找反弹高点平多头，IH等反弹做空，IM观察不动"
- "今日中性，不开新仓，Strangle继续持有吃Theta"

### 想要取得的结论

> 开盘前必须有一个明确的方向判断和操作计划，不能盘中临时起意。

---

## 二、开盘监控（9:25 - 9:45）

### 执行

```bash
# 终端1：日内信号monitor
python -m strategies.intraday.monitor

# 终端2：期权波动率monitor
python scripts/vol_monitor.py

# 终端3：策略象限monitor（每5分钟刷新）
python scripts/quadrant_monitor.py

# 终端4：半自动下单（可选，信号触发时确认下单）
python scripts/order_executor.py          # 实盘
python scripts/order_executor.py --dry-run # 只显示不下单
```

### 观察和记录

**开盘竞价阶段（9:15-9:25）**
- 集合竞价的方向（高开/低开/平开）
- 高开/低开幅度——超过0.5%算大幅
- 竞价方向和你的 Guidance 是否一致？

**前15分钟区间形成（9:30-9:45）**
- 这段时间不交易（系统禁止开仓，09:45后才允许）
- 记录三个品种的开盘区间高低点
- 成交量是放量还是缩量（vs昨日同时段）

### 想要取得的结论

> 开盘区间和竞价方向是否确认了你的盘前 Guidance？如果矛盾，要警惕。

---

## 三、盘中交易时段（9:45 - 14:30）

### 持续关注

**日内 monitor 面板（每5分钟刷新）**
- 信号得分变化趋势——是在升高（趋势加强）还是回落（趋势减弱）？
- 哪个品种的信号最强？方向是什么？
- 情绪乘数有没有异常（>1.2 或 <0.8）？
- 各维度的得分分布——是某一个维度主导还是多维度共振？

**vol_monitor 面板（每5分钟刷新）**
- 市场IV vs 结构IV 的分离程度——分离越大说明恐慌越重
- Skew（RR）的变化方向——扩大=恐慌加重，收窄=恐慌消退
- 持仓安全距离——三基准（Forward/期货/现货）都要看
- Theta累计——你的卖方持仓今天赚了多少时间价值

**象限 monitor 面板（每5分钟刷新）**
- 当前象限是否发生变化？
- 持仓匹配度评估是否出现⚠️或🔴？
- 象限切换预警距离多远？

### 关键决策点

**信号触发时（得分 >= 60）**
- monitor 会显示盘口 bid/ask 和建议限价，提示 [y/n/note]
- 信号和你的 Guidance 一致吗？
- 信号是在趋势中间还是趋势末端？（日内已涨跌>1.5%要警惕）
- 盘口支持吗？（monitor会显示bid/ask和建议限价）

**每次你有交易冲动时**
- 不管有没有执行，记录下来：
  - 时间、品种、方向
  - 触发你这个想法的原因（用自己的话）
  - 当时 monitor 的信号得分是多少
  - 你最终执行了还是放弃了
- 这些记录是将来量化你盘感的原材料

**异常情况处理**
- 标的日内跌/涨超2%：检查卖方持仓安全距离，考虑是否需要对冲
- IV突然飙升5pp+：可能有重大消息，先评估再操作
- 止损线接近：提前准备好平仓单，不要犹豫

### 时段特征（A股实证）

| 时段 | 特征 | 操作建议 |
|------|------|---------|
| 9:30-9:45 | 开盘区间形成 | 不交易，系统禁止开仓 |
| 9:45-10:30 | 突破时段 | 信号最佳入场窗口 |
| 10:30-11:20 | 趋势延续最稳定 | 顺势持仓，不轻易平 |
| 11:20-13:05 | 午休 | 系统禁止开仓 |
| 13:05-14:30 | 下午主时段 | 正常交易 |
| 14:30-15:00 | 尾盘 | 系统禁止开仓，观察不操作 |

### 想要取得的结论

> 每天记录2-3个关键观察：信号发出时的市场状态、你的判断和信号是否一致、最终结果对了还是错了。

---

## 四、收盘操作（15:15 - 15:30）

### 执行

```bash
# 1. 收盘后数据记录
python scripts/daily_record.py eod

# 2. 持仓分析
python scripts/portfolio_analysis.py

# 3. 波动率快照（收盘后用snapshot模式）
python scripts/vol_monitor.py --snapshot

# 4. 象限状态记录
python scripts/quadrant_monitor.py --once
```

### 检查清单

- [ ] eod 输出无报错
- [ ] 账户权益和昨天对比变化多少？
- [ ] 当日成交记录是否完整？
- [ ] 模型输出是否正常（IV、VRP、GARCH、Greeks）？
- [ ] 5分钟线归档成功（应该有48根，每天4小时/5分钟=48根）？
- [ ] 象限判定和持仓匹配度正常？
- [ ] Markdown报告已保存？

---

## 五、盘后复盘（18:30 - 19:00）

> Tushare数据通常18:00后发布。如果eod在15:15跑时Tushare无数据，18:30再跑一次。

### 需要分析的信息

**账户层面**
- 今日账户权益变动——赚了还是亏了，金额多少
- PnL归因：Delta贡献多少、Theta贡献多少、Vega贡献多少
- 哪个持仓赚了、哪个亏了

**信号层面**
```bash
# 查看今日signal_log
python -c "
import sqlite3, pandas as pd
conn = sqlite3.connect('data/storage/trading.db')
df = pd.read_sql('''
    SELECT datetime, symbol, direction, score, signal_version,
           score_breakout, score_vwap, score_multiframe,
           score_volume, score_daily
    FROM signal_log
    WHERE datetime >= date('now', '-1 day')
    AND score > 30
    ORDER BY score DESC LIMIT 20
''', conn)
print(df.to_string())
conn.close()
"
```

- 今天最强的信号是什么？得分多少？方向对了吗？
- 信号触发时你选择了执行还是观察？结果如何？
- 有没有信号系统漏掉的交易机会（你有感觉但系统没给分）？

**波动率层面**
- ATM IV vs 昨天变化了多少？
- VRP信号变了吗？
- Skew变化方向？
- 你的卖方持仓Theta今天赚了多少？

**市场层面**
- 今天中证1000/沪深300/上证50表现如何？
- 有没有异常事件（政策、大资金进出、涨跌停异常）？
- 今天的走势符合你盘前的Guidance吗？

### 写交易笔记

```bash
# 打开今日笔记（如果没有就创建）
mkdir -p logs/notes
nano logs/notes/$(date +%Y%m%d).md
```

必须记录的内容：
1. **盘前 Guidance 是什么** → 实际走势是什么 → 判断对了还是错了
2. **信号 vs 盘感**：今天信号和你的直觉有几次一致、几次矛盾？谁更准？
3. **关键时刻记录**：最重要的1-2个决策时刻，你怎么想的、怎么做的、结果如何
4. **改进想法**：有没有发现信号系统的问题或优化方向

### 想要取得的结论

> 回答一个核心问题：今天的操作有没有按纪律执行？如果偏离了纪律，原因是什么？

---

## 六、周末复盘（每周五收盘后额外30分钟）

### 本周数据汇总

```bash
# 本周的账户权益变化
python -c "
import sqlite3, pandas as pd
conn = sqlite3.connect('data/storage/trading.db')
df = pd.read_sql('''
    SELECT trade_date, balance, float_profit
    FROM account_snapshots
    WHERE trade_date >= strftime('%Y%m%d', 'now', '-7 days')
    ORDER BY trade_date
''', conn)
print(df.to_string())
conn.close()
"
```

### 需要回答的问题

**策略层面**
- 本周哪个策略赚钱了？哪个亏了？
- vol_arb的Theta收入 vs Vega损失——本周净收多少？
- 日内策略的信号准确率——本周发了多少个信号，对了几个？

**信号系统**
- 回顾本周所有 score >= 60 的信号，统计方向正确率
- v2 和 v3 哪个在实盘中更准？
- 情绪乘数有没有起到作用？

**你的盘感 vs 系统**
- 本周有几次你的判断和系统矛盾的情况？谁更准？
- 有没有你看到的规律是系统捕捉不到的？记录下来

**持仓管理**
- 到期合约需要滚动吗？
- Strangle的行权价需要调整吗？
- 下周有没有重大事件需要提前对冲？

### 想要取得的结论

> 本周学到了什么？下周的操作重点是什么？信号系统需要什么调整？

---

## 七、月度回顾（每月最后一个交易日额外1小时）

### 分析内容

- 本月账户权益曲线——画出来看趋势
- 本月各策略贡献分解：贴水收益、Theta收益、方向性盈亏、Gamma Scalping
- 信号系统月度统计：总信号数、准确率、盈亏比
- VRP信号的准确率：VRP信号为"做空波动率"时，后续IV真的下降了吗？
- 你的交易笔记中出现频率最高的"盘感模式"是什么？能量化吗？

### 想要取得的结论

> 回答：这个月的交易系统在进步还是退步？你对市场的理解有没有加深？下个月的重点方向是什么？

---

## 附：常用命令速查

```bash
# === 盘前 ===
cat logs/eod/YYYYMMDD.md                    # 查看某日EOD报告
cat logs/analysis/YYYYMMDD.md               # 查看某日持仓分析
cat logs/notes/YYYYMMDD.md                  # 查看某日交易笔记
python scripts/quadrant_monitor.py --once   # 象限快照

# === 盘中 ===
python -m strategies.intraday.monitor       # 日内信号面板
python scripts/vol_monitor.py               # 波动率面板
python scripts/quadrant_monitor.py          # 象限监控（持续）
python scripts/daily_record.py snapshot     # 盘中持仓快照

# === 收盘后 ===
python scripts/daily_record.py eod          # 完整EOD流程
python scripts/portfolio_analysis.py        # 持仓分析
python scripts/vol_monitor.py --snapshot    # 波动率快照
python scripts/quadrant_monitor.py --once   # 象限快照

# === 回测 ===
python scripts/backtest_signals_day.py --symbol IM --date 20260327              # 单日回测
python scripts/backtest_signals_day.py --symbol IM --date 20260220-20260327 --sensitivity  # 敏感性分析
python scripts/signal_quality_analysis.py --symbol IM                           # 信号质量分析

# === 研究分析 ===
python scripts/momentum_research.py         # 品种动量特征研究
streamlit run dashboard/app.py              # Dashboard可视化

# === 一键启动 ===
make trading-day                    # 全天一键（tmux 4窗口+定时EOD）
make briefing                       # 只跑morning briefing
make eod                            # 只跑EOD
make backtest                       # 回测今天
make dashboard                      # 启动Dashboard

# === tmux操作 ===
tmux attach -t trading_YYYYMMDD     # 进入交易session
# Ctrl-b 0/1/2/3 切换窗口（monitor/vol/quadrant/executor）
# Ctrl-b d 退出session（不关闭）
```
