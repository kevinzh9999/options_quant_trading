# 因子研究框架开发文档（Factor & Operator Framework）

> 版本：1.1 | 2026-04-05（更新：融合04-04的215天敏感分析和regime研究结论）
> 目标：将 SignalGeneratorV2 的信号评分系统因子化、算子化，建立可复用、可扩展的因子研究平台。

---

## 一、架构概览

### 当前系统（V2 硬编码）

```
score_all() 函数内部：
  mom_5m = (close[-1] - close[-13]) / close[-13]     ← 因子计算
  if abs(mom_5m) > 0.003: m = 35                     ← 打分映射
  elif abs(mom_5m) > 0.002: m = 25                   ← （和因子混在一起）
  ...
  raw_total = m + v + q + b                           ← 组合
  total = raw_total * daily_mult * intraday_filter    ← 调节
```

问题：因子计算、打分映射、组合权重全部混在一个函数里。测试新因子需要写完整的monkey-patch脚本。

### 目标架构（分层解耦）

```
Layer 0: Operators（算子库）
  ├── 标准化的数据变换函数
  ├── 所有因子的构建积木
  └── 一次编写，到处复用

Layer 1: Factors（因子计算）
  ├── 每个因子 = Operator 的组合
  ├── 输入 OHLCV → 输出一个连续数值
  └── 独立评估（IC、分组收益、单调性）

Layer 2: Scoring（打分映射）
  ├── 因子值 → 0-N 分
  ├── 阶梯式 / 线性 / 分段线性 / 排名
  └── 可独立于因子替换

Layer 3: Combination（组合）
  ├── 多因子加权求和
  ├── 当前：M+V+Q+B 等权
  └── 未来：优化权重 / 树模型

Layer 4: Adjustment（调节链）
  ├── daily_mult, intraday_filter, time_weight, sentiment_mult
  ├── Z-Score过滤, RSI加分
  └── 保持不变，不在因子框架内
```

### 文件结构

```
models/factors/
├── operators.py          # Layer 0: 算子库
├── base.py               # Factor 基类 + 评估接口
├── catalog_price.py      # Layer 1: 价格/动量类因子（M分替代候选）
├── catalog_vol.py        # Layer 1: 波动率类因子（V分替代候选）
├── catalog_volume.py     # Layer 1: 成交量类因子（Q分替代候选）
├── catalog_structure.py  # Layer 1: 结构类因子（B分替代候选）
├── catalog_alpha101.py   # Layer 1: 101 Alphas 经典因子
├── evaluator.py          # 因子评估器（IC、分组收益、相关性矩阵）
├── registry.py           # 因子注册表（自动发现和管理）
└── README.md             # 使用指南

scripts/
├── factor_research.py    # 交互式因子研究脚本
└── factor_batch_eval.py  # 批量因子评估
```

---

## 二、Layer 0: Operators（算子库）

### 设计原则

1. 每个 Operator 接受 `pd.Series` 或 `pd.DataFrame`，返回 `pd.Series`
2. 全部向量化计算（不用循环），一次算完整个时间序列
3. 命名参考 101 Formulaic Alphas 论文
4. 处理边界情况（NaN、除零等）

### 算子分类与完整清单

#### 2.1 时间序列算子（ts_* 系列）

对单个序列沿时间轴做运算。

```python
def delay(series: pd.Series, n: int) -> pd.Series:
    """
    n期前的值。101 Alphas 基础算子。
    用途：获取历史价格（计算return、delta等的基础）
    示例：delay(close, 1) = 昨天的收盘价
    """
    return series.shift(n)

def delta(series: pd.Series, n: int) -> pd.Series:
    """
    当前值 - n期前的值。
    用途：价格变化量、成交量变化量
    示例：delta(close, 12) = 当前close - 12根bar前的close
    """
    return series - series.shift(n)

def returns(series: pd.Series, n: int = 1) -> pd.Series:
    """
    n期收益率 = (当前 - n期前) / n期前。
    用途：动量因子的基础。当前M分的核心计算。
    示例：returns(close, 12) = 60分钟收益率
    对应当前系统：mom_5m = returns(close, 12)
    """
    return series.pct_change(n)

def ts_max(series: pd.Series, n: int) -> pd.Series:
    """
    过去n期最大值。
    用途：N日最高价、阻力位、Donchian通道上轨
    示例：ts_max(high, 48) = 过去1天最高价
    """
    return series.rolling(n).max()

def ts_min(series: pd.Series, n: int) -> pd.Series:
    """
    过去n期最小值。
    用途：N日最低价、支撑位、Donchian通道下轨
    """
    return series.rolling(n).min()

def ts_argmax(series: pd.Series, n: int) -> pd.Series:
    """
    过去n期中最大值出现的位置（距今多少期）。
    用途：高点是刚出现的还是很久前的（趋势新鲜度）
    101 Alphas: Alpha#14, #29 使用
    """
    return series.rolling(n).apply(lambda x: x.argmax(), raw=True)

def ts_argmin(series: pd.Series, n: int) -> pd.Series:
    """过去n期中最小值出现的位置"""
    return series.rolling(n).apply(lambda x: x.argmin(), raw=True)

def ts_rank(series: pd.Series, n: int) -> pd.Series:
    """
    当前值在过去n期中的分位数（0-1）。
    用途：相对强弱、百分位排名
    101 Alphas: 核心算子，Alpha#9, #32, #45 等大量使用
    示例：ts_rank(close, 48) = 当前价格在过去1天中的百分位
    """
    return series.rolling(n).apply(
        lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False
    )

def ts_stddev(series: pd.Series, n: int) -> pd.Series:
    """
    过去n期标准差。
    用途：波动率度量、布林带宽度
    对应当前系统：布林带的std(20)
    """
    return series.rolling(n).std()

def ts_mean(series: pd.Series, n: int) -> pd.Series:
    """
    过去n期均值。
    用途：均线、均量
    对应当前系统：volume均值mean(volume[-20:])
    """
    return series.rolling(n).mean()

def ts_sum(series: pd.Series, n: int) -> pd.Series:
    """
    过去n期累计和。
    用途：累计成交量、累计资金流
    """
    return series.rolling(n).sum()

def ts_corr(x: pd.Series, y: pd.Series, n: int) -> pd.Series:
    """
    过去n期的滚动Pearson相关系数。
    用途：量价相关性、跨品种相关性
    101 Alphas: Alpha#6, #44 等大量使用
    示例：ts_corr(close, volume, 10) = 10根bar的量价相关性
    """
    return x.rolling(n).corr(y)

def ts_covariance(x: pd.Series, y: pd.Series, n: int) -> pd.Series:
    """
    过去n期的滚动协方差。
    101 Alphas: Alpha#86, #96 使用
    """
    return x.rolling(n).cov(y)

def ts_skewness(series: pd.Series, n: int) -> pd.Series:
    """
    过去n期的偏度。
    用途：收益分布的不对称性（正偏=右尾长=跳涨风险）
    """
    return series.rolling(n).skew()

def ts_kurtosis(series: pd.Series, n: int) -> pd.Series:
    """
    过去n期的峰度。
    用途：收益分布的尾部厚度（高峰度=极端事件多）
    """
    return series.rolling(n).kurt()

def ts_product(series: pd.Series, n: int) -> pd.Series:
    """
    过去n期的连乘积。
    用途：复合收益率
    101 Alphas: Alpha#18 使用 ts_product(1+returns, 5)
    """
    return series.rolling(n).apply(lambda x: x.prod(), raw=True)
```

#### 2.2 加权/衰减算子

```python
def decay_linear(series: pd.Series, n: int) -> pd.Series:
    """
    线性衰减加权均值（近期权重线性递增）。
    权重：[1, 2, 3, ..., n] / sum
    用途：给近期数据更多权重的平滑（比简单MA更灵敏）
    101 Alphas: Alpha#7, #16 等使用
    """
    weights = np.arange(1, n + 1, dtype=float)
    weights /= weights.sum()
    return series.rolling(n).apply(lambda x: np.dot(x, weights), raw=True)

def decay_exp(series: pd.Series, span: int) -> pd.Series:
    """
    指数衰减均值（EMA）。
    用途：EMA均线、EMA动量
    对应当前系统：Z-Score 使用的 EMA20
    """
    return series.ewm(span=span).mean()

def weighted_mean(series: pd.Series, weights: pd.Series, n: int) -> pd.Series:
    """
    自定义权重的滚动加权均值。
    用途：成交量加权价格（VWAP类计算）
    """
    def _wm(vals, w):
        return np.average(vals, weights=w)
    # 需要对齐weights
    return series.rolling(n).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True
    )
```

#### 2.3 截面算子（cross-sectional）

对多品种在同一时间点做运算。

```python
def rank(series: pd.Series) -> pd.Series:
    """
    时序排名标准化到 0-1（逐行）。
    注意：在单品种时间序列上用 ts_rank 更合适。
    截面rank需要多品种DataFrame。
    101 Alphas: 最核心的算子，"万物皆 rank()"
    """
    return series.rank(pct=True)

def cross_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    截面排名：每个时间点，对所有品种排名（0-1）。
    输入：DataFrame，columns=品种，index=时间
    输出：同shape DataFrame，值为排名百分位
    用途：跨品种相对强弱
    """
    return df.rank(axis=1, pct=True)

def scale(series: pd.Series, target: float = 1.0) -> pd.Series:
    """
    缩放使绝对值之和 = target。
    101 Alphas: Alpha#1 等使用
    """
    return series * target / series.abs().sum()

def normalize(series: pd.Series, n: int) -> pd.Series:
    """
    Z-Score 标准化（过去n期的均值和标准差）。
    用途：因子值标准化，去量纲
    """
    mean = ts_mean(series, n)
    std = ts_stddev(series, n)
    return (series - mean) / std.replace(0, np.nan)
```

#### 2.4 数学/逻辑算子

```python
def sign(series: pd.Series) -> pd.Series:
    """符号函数：正=1，负=-1，零=0"""
    return np.sign(series)

def log(series: pd.Series) -> pd.Series:
    """自然对数"""
    return np.log(series.clip(lower=1e-10))

def abs_(series: pd.Series) -> pd.Series:
    """绝对值"""
    return series.abs()

def max_(a, b) -> pd.Series:
    """逐元素取大值"""
    return np.maximum(a, b)

def min_(a, b) -> pd.Series:
    """逐元素取小值"""
    return np.minimum(a, b)

def clamp(series: pd.Series, lower: float, upper: float) -> pd.Series:
    """限幅"""
    return series.clip(lower=lower, upper=upper)

def if_else(condition: pd.Series, true_val, false_val) -> pd.Series:
    """条件选择"""
    return pd.Series(np.where(condition, true_val, false_val), index=condition.index)
```

#### 2.5 K线结构算子（OHLCV 专用）

```python
def typical_price(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """典型价格 = (H+L+C)/3"""
    return (high + low + close) / 3.0

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    真实波幅 = max(H-L, |H-prevC|, |L-prevC|)
    对应当前系统：ATR 的基础计算
    """
    prev_c = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_c).abs(),
        (low - prev_c).abs()
    ], axis=1).max(axis=1)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    """
    平均真实波幅（N期ATR）。
    对应当前系统：ATR_short(5), ATR_long(40)
    """
    tr = true_range(high, low, close)
    return ts_mean(tr, n)

def body_ratio(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    K线实体占比 = (C-O)/(H-L)。
    101 Alphas: Alpha#101
    范围 -1 到 +1。+1=大阳线，-1=大阴线，0=十字星
    """
    return (close - open_) / (high - low + 1e-10)

def upper_shadow(open_: pd.Series, high: pd.Series, close: pd.Series, low: pd.Series) -> pd.Series:
    """上影线占比 = (H - max(O,C)) / (H-L)"""
    return (high - pd.concat([open_, close], axis=1).max(axis=1)) / (high - low + 1e-10)

def lower_shadow(open_: pd.Series, low: pd.Series, close: pd.Series, high: pd.Series) -> pd.Series:
    """下影线占比 = (min(O,C) - L) / (H-L)"""
    return (pd.concat([open_, close], axis=1).min(axis=1) - low) / (high - low + 1e-10)

def vwap_cumulative(typical: pd.Series, volume: pd.Series) -> pd.Series:
    """
    日内累计 VWAP。
    注意：需要按日分组，每天重新累计。
    """
    cum_tp_vol = (typical * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)

def bollinger_band(close: pd.Series, n: int = 20, k: float = 2.0):
    """
    布林带：中轨(MA) ± k*std。
    对应当前系统：B分的布林带逻辑
    返回：(upper, middle, lower, width, %b)
    """
    middle = ts_mean(close, n)
    std = ts_stddev(close, n)
    upper = middle + k * std
    lower = middle - k * std
    width = (upper - lower) / middle
    pct_b = (close - lower) / (upper - lower + 1e-10)
    return upper, middle, lower, width, pct_b

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """
    RSI（相对强弱指标）。
    对应当前系统：Z-Score过滤层的RSI反转加分
    """
    delta_ = close.diff()
    gain = delta_.clip(lower=0)
    loss = (-delta_).clip(lower=0)
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """
    ADX（平均趋向指数）。
    用途：趋势强度度量（不分方向）
    研究结论：ADX>35 是IM的"死亡区间"，但样本太少未实施
    """
    from ta.trend import ADXIndicator
    indicator = ADXIndicator(high, low, close, window=n)
    return indicator.adx()

def linreg_slope(series: pd.Series, n: int) -> pd.Series:
    """
    过去n期线性回归斜率（标准化为收益率）。
    用途：比简单return更稳健的趋势度量（不受单个异常点影响）
    """
    def _slope(vals):
        x = np.arange(len(vals))
        slope = np.polyfit(x, vals, 1)[0]
        return slope / vals[-1] if vals[-1] != 0 else 0
    return series.rolling(n).apply(_slope, raw=True)
```

---

## 三、Layer 1: Factors（因子定义）

### 3.1 Factor 基类

```python
from abc import ABC, abstractmethod
import pandas as pd

class Factor(ABC):
    """因子基类。所有因子继承此类。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """因子唯一标识符，如 'mom_simple_12'"""
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        """因子类别：'momentum', 'volatility', 'volume', 'structure', 'composite'"""
        pass

    @property
    def description(self) -> str:
        """因子描述（可选）"""
        return ""

    @property
    def params(self) -> dict:
        """因子参数（用于grid search）"""
        return {}

    @abstractmethod
    def compute_series(self, bar_5m: pd.DataFrame,
                       bar_15m: pd.DataFrame = None,
                       daily: pd.DataFrame = None) -> pd.Series:
        """
        向量化计算完整时间序列的因子值。
        这是主要接口——比逐bar compute() 快100倍。
        
        输入：
          bar_5m: DataFrame with columns [open, high, low, close, volume]
          bar_15m: 可选，15分钟K线
          daily: 可选，日线数据
        输出：
          pd.Series，index 与 bar_5m 对齐，值为因子值
        """
        pass

    def compute(self, bar_5m: pd.DataFrame, **kwargs) -> float:
        """
        计算最新一根bar的因子值（用于实盘）。
        默认实现：调用 compute_series 取最后一个值。
        子类可覆盖以优化性能。
        """
        series = self.compute_series(bar_5m, **kwargs)
        return series.iloc[-1] if len(series) > 0 else 0.0
```

### 3.2 价格/动量类因子（catalog_price.py）

覆盖当前 M分 的所有计算变体。

```python
# === 当前系统使用的因子（Baseline） ===

class MomSimple(Factor):
    """
    当前M分使用的因子：简单N根bar收益率。
    对应代码：mom_5m = (close[-1] - close[-13]) / close[-13]
    """
    name = "mom_simple"
    category = "momentum"
    description = "简单N期收益率，当前系统M分的基础"

    def __init__(self, lookback=12):
        self.lookback = lookback
        self._name = f"mom_simple_{lookback}"

    @property
    def name(self):
        return self._name

    @property
    def params(self):
        return {"lookback": self.lookback}

    def compute_series(self, bar_5m, **kwargs):
        return returns(bar_5m['close'], self.lookback)


# === 替代候选因子 ===

class MomEMA(Factor):
    """
    EMA交叉度：快EMA vs 慢EMA 的偏离。
    优势：对近期价格赋予更多权重，比简单return更平滑
    """
    name = "mom_ema"
    category = "momentum"

    def __init__(self, fast=5, slow=20):
        self.fast = fast
        self.slow = slow
        self._name = f"mom_ema_{fast}_{slow}"

    @property
    def name(self):
        return self._name

    def compute_series(self, bar_5m, **kwargs):
        ema_f = decay_exp(bar_5m['close'], self.fast)
        ema_s = decay_exp(bar_5m['close'], self.slow)
        return (ema_f - ema_s) / ema_s


class MomLinReg(Factor):
    """
    线性回归斜率：用最小二乘法拟合N根bar的趋势线，取斜率。
    优势：不受lookback两端异常值影响（简单return只看头尾两个点）
    """
    name = "mom_linreg"
    category = "momentum"

    def __init__(self, lookback=12):
        self.lookback = lookback
        self._name = f"mom_linreg_{lookback}"

    @property
    def name(self):
        return self._name

    def compute_series(self, bar_5m, **kwargs):
        return linreg_slope(bar_5m['close'], self.lookback)


class MomDecayLinear(Factor):
    """
    线性衰减加权动量：近期bar权重线性递增。
    优势：综合了多根bar的信息，近期变化影响更大
    101 Alphas 灵感：decay_linear 是高频使用的算子
    """
    name = "mom_decay"
    category = "momentum"

    def __init__(self, lookback=12):
        self.lookback = lookback
        self._name = f"mom_decay_{lookback}"

    @property
    def name(self):
        return self._name

    def compute_series(self, bar_5m, **kwargs):
        ret = returns(bar_5m['close'], 1)
        return decay_linear(ret, self.lookback)


class MomRank(Factor):
    """
    当前价格在过去N期中的百分位排名。
    101 Alphas: ts_rank 是核心算子
    优势：天然标准化到0-1，不受价格绝对水平影响
    """
    name = "mom_rank"
    category = "momentum"

    def __init__(self, lookback=48):
        self.lookback = lookback
        self._name = f"mom_rank_{lookback}"

    @property
    def name(self):
        return self._name

    def compute_series(self, bar_5m, **kwargs):
        return ts_rank(bar_5m['close'], self.lookback)


class MomMultiScale(Factor):
    """
    多尺度动量：5min + 15min 动量的加权组合。
    对应当前系统：M分要求5min和15min方向一致
    这里用连续值替代二值判断
    """
    name = "mom_multiscale"
    category = "momentum"

    def __init__(self, fast=6, slow=18):
        self.fast = fast
        self.slow = slow

    def compute_series(self, bar_5m, **kwargs):
        mom_fast = returns(bar_5m['close'], self.fast)
        mom_slow = returns(bar_5m['close'], self.slow)
        # 方向一致时取平均，不一致时衰减
        agreement = sign(mom_fast) * sign(mom_slow)  # +1=一致, -1=矛盾
        return (mom_fast + mom_slow) / 2 * (0.5 + 0.5 * agreement)
```

### 3.3 波动率类因子（catalog_vol.py）

覆盖当前 V分 的所有计算变体。

```python
# === 当前系统使用的因子 ===

class VolATRRatio(Factor):
    """
    当前V分使用的因子：ATR短期/长期比值。
    对应代码：ratio = ATR_short(5) / ATR_long(40)
    V分是逆向指标：ratio低=蓄势待发=高分
    """
    name = "vol_atr_ratio"
    category = "volatility"

    def __init__(self, short=5, long=40):
        self.short = short
        self.long = long
        self._name = f"vol_atr_ratio_{short}_{long}"

    @property
    def name(self):
        return self._name

    def compute_series(self, bar_5m, **kwargs):
        atr_s = atr(bar_5m['high'], bar_5m['low'], bar_5m['close'], self.short)
        atr_l = atr(bar_5m['high'], bar_5m['low'], bar_5m['close'], self.long)
        return atr_s / atr_l.replace(0, np.nan)


# === 替代候选 ===

class VolATRTrend(Factor):
    """
    ATR变化趋势：ATR在扩张还是收缩。
    优势：捕捉波动率的方向（V分只看水平）
    扩张中入场 > 已经扩张完毕
    """
    name = "vol_atr_trend"
    category = "volatility"

    def __init__(self, short=3, long=8):
        self.short = short
        self.long = long

    def compute_series(self, bar_5m, **kwargs):
        tr = true_range(bar_5m['high'], bar_5m['low'], bar_5m['close'])
        atr_now = ts_mean(tr, self.short)
        atr_prev = ts_mean(tr, self.long)
        return (atr_now - atr_prev) / atr_prev.replace(0, np.nan)


class VolParkinson(Factor):
    """
    Parkinson波动率：只用High-Low估计波动率（比Close-Close更高效）。
    理论上效率是Close-Close的5倍。
    """
    name = "vol_parkinson"
    category = "volatility"

    def __init__(self, lookback=20):
        self.lookback = lookback

    def compute_series(self, bar_5m, **kwargs):
        hl_ratio = log(bar_5m['high'] / bar_5m['low'])
        return (hl_ratio ** 2).rolling(self.lookback).mean().apply(
            lambda x: np.sqrt(x / (4 * np.log(2)))
        )


class VolReturnStd(Factor):
    """
    收益率标准差比值：短期std / 长期std。
    和ATR Ratio类似但用return而非true range
    """
    name = "vol_return_std"
    category = "volatility"

    def __init__(self, short=10, long=40):
        self.short = short
        self.long = long

    def compute_series(self, bar_5m, **kwargs):
        ret = returns(bar_5m['close'], 1)
        std_s = ts_stddev(ret, self.short)
        std_l = ts_stddev(ret, self.long)
        return std_s / std_l.replace(0, np.nan)


class VolBBWidth(Factor):
    """
    布林带宽度：(上轨-下轨)/中轨。
    对应当前系统：B分中的窄带突破判断
    """
    name = "vol_bb_width"
    category = "volatility"

    def __init__(self, n=20, k=2.0):
        self.n = n
        self.k = k

    def compute_series(self, bar_5m, **kwargs):
        _, _, _, width, _ = bollinger_band(bar_5m['close'], self.n, self.k)
        return width
```

### 3.4 成交量类因子（catalog_volume.py）

覆盖当前 Q分 的所有计算变体。

```python
# === 当前系统使用的因子 ===

class VolRatio(Factor):
    """
    当前Q分使用的因子：当前bar量 / 20根均量。
    对应代码：ratio = volume[-1] / mean(volume[-20:])
    """
    name = "vol_ratio"
    category = "volume"

    def __init__(self, lookback=20):
        self.lookback = lookback
        self._name = f"vol_ratio_{lookback}"

    @property
    def name(self):
        return self._name

    def compute_series(self, bar_5m, **kwargs):
        return bar_5m['volume'] / ts_mean(bar_5m['volume'], self.lookback)


# === 替代候选 ===

class VolTrend(Factor):
    """
    成交量趋势：短期均量 / 长期均量。
    优势：连续放量比单根放量更可靠（Q分只看单根bar）
    """
    name = "vol_trend"
    category = "volume"

    def __init__(self, short=3, long=10):
        self.short = short
        self.long = long

    def compute_series(self, bar_5m, **kwargs):
        vol_s = ts_mean(bar_5m['volume'], self.short)
        vol_l = ts_mean(bar_5m['volume'], self.long)
        return vol_s / vol_l.replace(0, np.nan)


class VolPriceCorr(Factor):
    """
    量价相关性：价格和成交量的滚动相关系数。
    正相关=量价齐升（健康趋势），负相关=量价背离（趋势可能反转）
    101 Alphas: Alpha#6 = -ts_corr(open, volume, 10)
    之前研究结论：作为独立信号跑shadow观察，不急于接入
    """
    name = "vol_price_corr"
    category = "volume"

    def __init__(self, lookback=10):
        self.lookback = lookback

    def compute_series(self, bar_5m, **kwargs):
        return ts_corr(bar_5m['close'], bar_5m['volume'], self.lookback)


class VolSignedFlow(Factor):
    """
    带方向的资金流：sign(return) * volume。
    粗略估计主动买入/卖出方向（没有tick数据时的OFI近似）
    """
    name = "vol_signed_flow"
    category = "volume"

    def __init__(self, lookback=10):
        self.lookback = lookback

    def compute_series(self, bar_5m, **kwargs):
        ret = returns(bar_5m['close'], 1)
        signed_vol = sign(ret) * bar_5m['volume']
        return ts_sum(signed_vol, self.lookback) / ts_sum(bar_5m['volume'], self.lookback)
```

### 3.5 结构类因子（catalog_structure.py）

覆盖当前 B分 的逻辑以及K线内部结构。

```python
class BollBreakout(Factor):
    """
    布林带突破强度：价格相对布林带的位置（%B）。
    对应当前系统：B分的布林带中轨突破
    %B > 1 = 在上轨以上，%B < 0 = 在下轨以下
    """
    name = "boll_breakout"
    category = "structure"

    def __init__(self, n=20, k=2.0):
        self.n = n
        self.k = k

    def compute_series(self, bar_5m, **kwargs):
        _, _, _, _, pct_b = bollinger_band(bar_5m['close'], self.n, self.k)
        return pct_b


class BodyRatio(Factor):
    """
    K线实体占比。
    101 Alphas: Alpha#101 = (close-open)/((high-low)+.001)
    研究结论：IM有效IC有害，未实施。但作为因子可供组合使用
    """
    name = "body_ratio"
    category = "structure"

    def compute_series(self, bar_5m, **kwargs):
        return body_ratio(bar_5m['open'], bar_5m['high'],
                         bar_5m['low'], bar_5m['close'])


class UpperShadowRatio(Factor):
    """上影线占比：(H-max(O,C))/(H-L)"""
    name = "upper_shadow"
    category = "structure"

    def compute_series(self, bar_5m, **kwargs):
        return upper_shadow(bar_5m['open'], bar_5m['high'],
                           bar_5m['close'], bar_5m['low'])


class PricePosition(Factor):
    """
    价格在N日范围中的位置（0=最低, 1=最高）。
    对应当前系统：morning_briefing 中的20日范围位置
    """
    name = "price_position"
    category = "structure"

    def __init__(self, lookback=48*5):  # 5天
        self.lookback = lookback

    def compute_series(self, bar_5m, **kwargs):
        high_n = ts_max(bar_5m['high'], self.lookback)
        low_n = ts_min(bar_5m['low'], self.lookback)
        return (bar_5m['close'] - low_n) / (high_n - low_n + 1e-10)
```

### 3.6 101 Alphas 经典因子（catalog_alpha101.py）

从论文中选取与日内动量策略最相关的因子。

```python
class Alpha001(Factor):
    """
    Alpha#1: (rank(ts_argmax(sign(delta(close,1))^2, 5)) - 0.5) * (-sign(delta(close,1)))
    含义：如果最近5天有大涨，且今天收跌，做空
    """
    name = "alpha001"
    category = "composite"

    def compute_series(self, bar_5m, **kwargs):
        c = bar_5m['close']
        inner = sign(delta(c, 1)) ** 2
        return (ts_rank(ts_argmax(inner, 5), 5) - 0.5) * (-sign(delta(c, 1)))


class Alpha002(Factor):
    """
    Alpha#2: -ts_corr(rank(delta(log(volume),2)), rank((close-open)/open), 6)
    含义：成交量变化和价格变化的负相关性
    """
    name = "alpha002"
    category = "composite"

    def compute_series(self, bar_5m, **kwargs):
        c, o, v = bar_5m['close'], bar_5m['open'], bar_5m['volume']
        d_log_vol = delta(log(v), 2)
        price_move = (c - o) / o
        return -ts_corr(d_log_vol.rank(pct=True), price_move.rank(pct=True), 6)


class Alpha006(Factor):
    """
    Alpha#6: -ts_corr(open, volume, 10)
    含义：开盘价和成交量的负相关性。简单但有效
    """
    name = "alpha006"
    category = "composite"

    def compute_series(self, bar_5m, **kwargs):
        return -ts_corr(bar_5m['open'], bar_5m['volume'], 10)


class Alpha012(Factor):
    """
    Alpha#12: sign(delta(volume,1)) * (-delta(close,1))
    含义：量增价跌→做多（吸筹），量增价涨→做空（出货）
    """
    name = "alpha012"
    category = "composite"

    def compute_series(self, bar_5m, **kwargs):
        return sign(delta(bar_5m['volume'], 1)) * (-delta(bar_5m['close'], 1))


class Alpha018(Factor):
    """
    Alpha#18: -rank(ts_stddev(abs(close-open), 5) + (close-open) + ts_corr(close,open,10))
    含义：波动率+趋势+开收相关性的组合
    """
    name = "alpha018"
    category = "composite"

    def compute_series(self, bar_5m, **kwargs):
        c, o = bar_5m['close'], bar_5m['open']
        part1 = ts_stddev(abs_(c - o), 5)
        part2 = c - o
        part3 = ts_corr(c, o, 10)
        return -(part1 + part2 + part3).rank(pct=True)


class Alpha041(Factor):
    """
    Alpha#41: ((high * low)^0.5) - vwap
    含义：几何均价 vs VWAP 的偏离
    注意：需要VWAP数据或计算
    """
    name = "alpha041"
    category = "composite"

    def compute_series(self, bar_5m, **kwargs):
        geo_mean = (bar_5m['high'] * bar_5m['low']) ** 0.5
        tp = typical_price(bar_5m['high'], bar_5m['low'], bar_5m['close'])
        vwap = vwap_cumulative(tp, bar_5m['volume'])
        return geo_mean - vwap


class Alpha101(Factor):
    """
    Alpha#101: (close - open) / ((high - low) + .001)
    含义：K线实体占比（和BodyRatio相同但这是论文原始定义）
    """
    name = "alpha101"
    category = "composite"

    def compute_series(self, bar_5m, **kwargs):
        return (bar_5m['close'] - bar_5m['open']) / (bar_5m['high'] - bar_5m['low'] + 0.001)
```

---

## 四、Layer 1.5: Evaluator（因子评估器）

### 4.1 评估指标

| 指标 | 计算方式 | 好的标准 | 用途 |
|------|---------|---------|------|
| **IC (Information Coefficient)** | Spearman(因子值, 未来N期收益) | \|IC\| > 0.03 | 预测力 |
| **IC_IR** | mean(IC) / std(IC) | \|IC_IR\| > 0.5 | IC稳定性 |
| **分组收益单调性** | Q1到Q5的收益是否单调 | \|单调性相关\| > 0.8 | 预测方向正确 |
| **因子间相关性** | Pearson(因子A, 因子B) | < 0.5 才有独立价值 | 去冗余 |
| **换手率** | 因子排名变化频率 | 适中 | 交易成本估计 |

### 4.2 Evaluator 类

```python
class FactorEvaluator:
    """
    因子评估器。
    支持三种target variable：
    1. price_return: 未来N根bar的价格收益率（通用）
    2. iv_change: 未来IV变化（期权因子评估）
    3. realized_vrp: 已实现VRP（波动率因子评估）
    """

    def __init__(self, bar_5m: pd.DataFrame,
                 target_type: str = 'price_return',
                 forward_periods: list = [1, 3, 5, 10],
                 daily_data: pd.DataFrame = None):
        self.bar_5m = bar_5m
        self.target_type = target_type
        self.forward_periods = forward_periods
        self.daily_data = daily_data

        # 预计算 target
        self.targets = {}
        if target_type == 'price_return':
            for n in forward_periods:
                self.targets[n] = bar_5m['close'].pct_change(n).shift(-n)

    def evaluate(self, factor: Factor) -> dict:
        """评估单个因子，返回完整的评估报告"""
        values = factor.compute_series(self.bar_5m)
        values = values.replace([np.inf, -np.inf], np.nan).dropna()

        result = {
            'name': factor.name,
            'category': factor.category,
            'stats': self._calc_stats(values),
            'ic': self._calc_ic(values),
            'group_returns': self._calc_group_returns(values),
            'monotonicity': self._calc_monotonicity(values),
        }
        return result

    def _calc_stats(self, values):
        return {
            'count': len(values),
            'mean': values.mean(),
            'std': values.std(),
            'skew': values.skew(),
            'min': values.min(),
            'max': values.max(),
            'pct_nan': values.isna().mean(),
        }

    def _calc_ic(self, values):
        """计算每个forward period的IC和IC_IR"""
        ics = {}
        for n in self.forward_periods:
            target = self.targets[n].reindex(values.index)
            valid = pd.concat([values, target], axis=1).dropna()
            if len(valid) < 30:
                ics[f'IC_{n}bar'] = np.nan
                continue
            # Rank IC (Spearman)
            ic = valid.iloc[:, 0].corr(valid.iloc[:, 1], method='spearman')
            ics[f'IC_{n}bar'] = round(ic, 4)

            # Rolling IC for IC_IR
            rolling_ic = valid.iloc[:, 0].rolling(60).corr(valid.iloc[:, 1])
            ic_mean = rolling_ic.mean()
            ic_std = rolling_ic.std()
            ics[f'IC_IR_{n}bar'] = round(ic_mean / ic_std, 4) if ic_std > 0 else np.nan

        return ics

    def _calc_group_returns(self, values, n_groups=5):
        """按因子值分组，看各组的平均未来收益"""
        groups = {}
        for n in self.forward_periods:
            target = self.targets[n].reindex(values.index)
            valid = pd.concat([values.rename('factor'), target.rename('target')], axis=1).dropna()
            if len(valid) < 50:
                continue
            valid['group'] = pd.qcut(valid['factor'], n_groups,
                                     labels=[f'Q{i+1}' for i in range(n_groups)],
                                     duplicates='drop')
            gr = valid.groupby('group')['target'].mean()
            groups[f'{n}bar'] = gr.to_dict()
        return groups

    def _calc_monotonicity(self, values):
        """检验分组收益是否单调"""
        mono = {}
        for n in self.forward_periods:
            target = self.targets[n].reindex(values.index)
            valid = pd.concat([values.rename('factor'), target.rename('target')], axis=1).dropna()
            if len(valid) < 50:
                continue
            valid['group'] = pd.qcut(valid['factor'], 5,
                                     labels=range(5), duplicates='drop')
            gr = valid.groupby('group')['target'].mean()
            from scipy.stats import spearmanr
            corr, pval = spearmanr(range(len(gr)), gr.values)
            mono[f'{n}bar'] = {'corr': round(corr, 3), 'pval': round(pval, 4)}
        return mono

    def batch_evaluate(self, factors: list) -> tuple:
        """批量评估+相关性矩阵"""
        results = []
        series_dict = {}

        for f in factors:
            result = self.evaluate(f)
            results.append(result)
            series_dict[f.name] = f.compute_series(self.bar_5m)

        # 因子间相关性
        corr_df = pd.DataFrame(series_dict).corr()

        return results, corr_df

    def print_report(self, factors: list):
        """打印可读的评估报告"""
        results, corr = self.batch_evaluate(factors)

        print("=" * 90)
        print(f"  FACTOR EVALUATION REPORT | {len(factors)} factors | {len(self.bar_5m)} bars")
        print("=" * 90)

        # IC汇总表
        print("\n--- Information Coefficient (Rank IC) ---")
        header = f"{'Factor':<25} {'Category':<12}"
        for n in self.forward_periods:
            header += f" {'IC_'+str(n)+'bar':>10}"
        print(header)
        print("-" * len(header))

        for r in results:
            row = f"{r['name']:<25} {r['category']:<12}"
            for n in self.forward_periods:
                ic = r['ic'].get(f'IC_{n}bar', np.nan)
                row += f" {ic:>10.4f}" if not np.isnan(ic) else f" {'N/A':>10}"
            print(row)

        # 单调性
        print("\n--- Monotonicity (Q1→Q5) ---")
        for r in results:
            mono = r['monotonicity']
            mono_str = ", ".join([f"{k}: {v['corr']:.2f}(p={v['pval']:.3f})"
                                  for k, v in mono.items()])
            print(f"  {r['name']:<25} {mono_str}")

        # 相关性矩阵
        print("\n--- Factor Correlation Matrix ---")
        print(corr.round(3).to_string())

        # 分组收益（只打印最有前景的因子）
        best = sorted(results, key=lambda r: abs(r['ic'].get(f'IC_{self.forward_periods[1]}bar', 0)), reverse=True)
        print(f"\n--- Top Factor Group Returns ({self.forward_periods[1]}bar) ---")
        for r in best[:3]:
            gr = r['group_returns'].get(f'{self.forward_periods[1]}bar', {})
            if gr:
                print(f"  {r['name']}: {gr}")
```

---

## 五、因子开发指南

### 5.1 添加新因子的步骤

```
1. 确定类别 → 放入对应的 catalog_xxx.py
2. 继承 Factor 基类，实现 compute_series()
3. 用 operator 组合定义（不要手写循环）
4. 在 factor_research.py 中加入评估
5. 跑 evaluator.print_report() 看 IC 和分组收益
6. IC > 0.03 且和现有因子相关性 < 0.5 → 值得进一步回测
7. 完整回测验证 → 决定是否替换现有因子
```

### 5.2 因子命名规范

```
{category}_{method}_{param1}_{param2}
例：
  mom_simple_12        # 动量_简单收益率_12根lookback
  vol_atr_ratio_5_40   # 波动率_ATR比值_短5长40
  vol_price_corr_10    # 成交量_量价相关性_10根窗口
  alpha006             # 101 Alphas 原始编号
```

### 5.3 因子质量标准

| 等级 | IC绝对值 | 单调性 | 和现有因子相关性 | 行动 |
|------|---------|--------|----------------|------|
| ⭐⭐⭐ | > 0.05 | > 0.8 | < 0.3 | 立即回测验证 |
| ⭐⭐ | 0.03-0.05 | > 0.6 | < 0.5 | 值得进一步研究 |
| ⭐ | 0.02-0.03 | > 0.4 | < 0.5 | 放入候选池观察 |
| ❌ | < 0.02 | < 0.4 | > 0.5 | 放弃 |

### 5.4 避坑清单

1. **前瞻偏差**：compute_series 不能用未来数据。forward_periods 的 shift(-n) 只在 evaluator 中用，因子本身不能 shift 负值
2. **存活偏差**：如果因子用到的数据（如期权行情）只有部分日期有，需要标注覆盖率
3. **过拟合**：一个因子如果在IC grid search中只有特定参数好，其他参数都很差→可能过拟合
4. **交易成本**：高换手率的因子（每根bar排名剧烈变化）在实盘中成本高
5. **和现有因子的关系**：新因子必须和M/V/Q的相关性 < 0.5 才有独立价值。之前验证VWAP(r=0.67)被排除就是这个原因

---

## 六、和现有系统的对应关系

### 当前 M/V/Q/B 的因子化分解

| 当前维度 | 底层因子 | 打分映射 | 因子化后 |
|---------|---------|---------|---------|
| **M分(0-50)** | `returns(close, 12)` = 0.35% | 阶梯：>0.3%→35, >0.2%→25, >0.1%→15 | `MomSimple(12)` |
| M分(方向) | `sign(mom_5m) == sign(mom_15m)` | 二值：一致→+15, 不一致→M=0 | `MomMultiScale(6, 18)` |
| **V分(0-30)** | `atr(5) / atr(40)` = 0.8 | 逆向阶梯：<0.7→30, <0.9→25... | `VolATRRatio(5, 40)` |
| **Q分(0-20)** | `volume / mean(volume, 20)` = 1.6 | 阶梯：>1.5→20, >0.5→10 | `VolRatio(20)` |
| **B分(0-20)** | `close穿越bollinger中轨` | 条件叠加：基础10 + 放量+2 + 窄带+3 + 15m确认+5 | `BollBreakout(20, 2)` |

### 乘数链（不在因子框架内，保持现状）

| 乘数 | 数据来源 | 当前值 | 因子化？ |
|------|---------|--------|---------|
| daily_mult | 日线5日动量 | dm=1.1/0.9（04-04从1.2/0.8调整，215天最优） | 否——regime调节 |
| intraday_filter | 当日涨跌幅 | 顺势1.0/逆势0.5~0.8 | 否——风控层面 |
| time_weight | 时段 | per-symbol session_multiplier | 否——固定规则 |
| sentiment_mult | 期权IV/RR/VRP | 0.5~1.5 | **不建议因子化**（04-04验证：期权指标与PnL偏相关<0.15，均为振幅代理） |
| Z-Score过滤 | 日线EMA20/STD20 | 4层Z-Score | 否——安全阀 |
| **振幅过滤** | 开盘30min振幅 | <0.4%不开仓 | 否——04-04新增，振幅是策略PnL最强预测因子(r=0.43) |

### 04-04研究结论对因子框架的影响

1. **日内振幅是策略PnL的最强预测因子(r=0.43)**——远超任何单因子IC。因子框架应优先研究能预测"振幅会不会大"的因子，而非直接预测"涨跌方向"
2. **期权指标（IV/RV/VRP/RR/Term Structure）对日内PnL几乎无独立预测力**——偏相关全部<0.15，均为振幅代理。不建议在因子框架中花时间做期权因子（除非有5分钟级别的期权tick数据）
3. **dm=1.1/0.9最优 = 逆势交易也是盈利的**——因子评估时不应只看顺势IC，逆势信号同样需要评估
4. **低振幅日亏-27pt/天、高振幅日赚+49pt/天**——因子在高/低振幅日的表现可能完全不同。evaluator应支持按振幅regime分组评估

---

## 七、实施路线图

### 前置条件（已完成 2026-04-04）

```
已完成:
✅ 215天数据基础设施（index_min 10295根5分钟bar，2025-05~2026-04）
✅ 7参数敏感分析框架（scripts/sensitivity_215d.py）
✅ 190天IV/RV/VRP/RR历史数据回算入库（scripts/backfill_iv_history.py）
✅ 振幅过滤器实现（check_low_amplitude + monitor/backtest集成）
✅ 期权regime指标验证（结论：无独立增量）

研究结论（影响因子框架设计）:
- 日内振幅r=0.43是最强预测因子，远超所有因子IC
- 期权指标偏相关<0.15，不值得因子化
- dm=1.1/0.9最优，逆势交易也盈利
```

### Phase 1：建框架 + 评估M分替代（2天）

```
1. 创建 models/factors/ 目录结构
2. 实现 operators.py（全部算子）
3. 实现 base.py（Factor基类）
4. 实现 catalog_price.py（6个M分候选因子）
5. 实现 evaluator.py（IC + 分组收益 + 相关性）
   → 新增：支持按振幅regime分组评估（高/低振幅日分别算IC）
6. 跑 M分 6个候选的评估报告
7. 如果有 IC 显著高于 MomSimple 的候选 → Phase 2 回测
```

### Phase 2：评估V/Q/B替代 + 101 Alphas（2天）

```
1. 实现 catalog_vol.py（4个V分候选）
2. 实现 catalog_volume.py（4个Q分候选）
3. 实现 catalog_structure.py（4个B分候选）
4. 实现 catalog_alpha101.py（8-10个经典因子）
5. 批量评估所有因子（~25个），输出完整报告
6. 识别 IC top-5 因子，检查和现有因子的独立性
注意：跳过期权因子（已验证无增量）
```

### Phase 3：因子替换回测（1-2天）

```
1. 对 IC top 因子，在 backtest 中替换对应维度
2. 用 215天数据验证 PnL 变化（直接用 sensitivity_215d.py 框架）
3. 稳健性验证（时间分段、参数邻域）
4. 决定是否替换
```

### Phase 4：因子组合优化（未来）

```
1. 多因子等权组合 vs 优化权重
2. 因子正交化（去除共线性）
3. LightGBM 非线性组合（需要500+笔交易数据）
注意：当前215天约1100笔交易，勉强够训练简单模型
```

---

## 八、使用示例

### 快速评估一批因子

```python
# scripts/factor_research.py

import pandas as pd
from models.factors.catalog_price import *
from models.factors.catalog_vol import *
from models.factors.catalog_volume import *
from models.factors.catalog_alpha101 import *
from models.factors.evaluator import FactorEvaluator

# 加载215天5分钟K线数据
import sqlite3
conn = sqlite3.connect('data/storage/trading.db')
bar_5m = pd.read_sql("""
    SELECT datetime, open, high, low, close, volume
    FROM index_min WHERE symbol = '000852'
    ORDER BY datetime
""", conn)
bar_5m['datetime'] = pd.to_datetime(bar_5m['datetime'])
bar_5m.set_index('datetime', inplace=True)

# 创建评估器
evaluator = FactorEvaluator(bar_5m, forward_periods=[1, 3, 5, 10])

# 定义候选因子
factors = [
    # M分候选
    MomSimple(12),        # baseline
    MomEMA(5, 20),
    MomLinReg(12),
    MomDecayLinear(12),
    MomRank(48),

    # V分候选
    VolATRRatio(5, 40),   # baseline
    VolATRTrend(3, 8),
    VolReturnStd(10, 40),

    # Q分候选
    VolRatio(20),         # baseline
    VolTrend(3, 10),
    VolPriceCorr(10),

    # 101 Alphas
    Alpha006(),
    Alpha012(),
    Alpha101(),
]

# 一键评估
evaluator.print_report(factors)
```

### 添加自定义因子

```python
# 只需要3步：

# 1. 定义因子类
class MyNewFactor(Factor):
    name = "my_factor"
    category = "momentum"

    def compute_series(self, bar_5m, **kwargs):
        # 用operator组合
        mom = decay_linear(returns(bar_5m['close'], 1), 10)
        vol = ts_stddev(returns(bar_5m['close'], 1), 20)
        return mom / vol  # 动量/波动率 = 风险调整动量

# 2. 加入评估列表
factors.append(MyNewFactor())

# 3. 跑评估
evaluator.print_report(factors)
```
