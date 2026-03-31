"""
signal.py
---------
多维度日内信号评分系统。每根K线更新一次，对 IF/IH/IM 分别独立评分。

维度（满分100）：
  1. 开盘区间突破  （25分）
  2. VWAP偏离回归  （21分）
  3. 多周期趋势一致 （16分）
  4. 成交量确认     （11分）— 含量能趋势 + 关键位置量价配合
  5. 日线支撑阻力   （7分）
  6. 盘口压力       （8分）— 仅实时模式，回测返回0
  7. 布林带多周期   （12分）— 5m/15m/日线三周期布林带
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# UTC 交易时段边界（北京时间 = UTC + 8h）
# ---------------------------------------------------------------------------
# 上午 09:30-11:30 北京 = 01:30-03:30 UTC
# 下午 13:00-15:00 北京 = 05:00-07:00 UTC
MORNING_START = "01:30"
MORNING_END = "03:30"
AFTERNOON_START = "05:00"
AFTERNOON_END = "07:00"
OPENING_RANGE_END = "02:00"  # 10:00 北京 = 开盘后30分钟

NO_TRADE_BEFORE = "01:35"           # 09:35 北京
NO_NEW_INTRADAY_AFTER = "06:50"     # 14:50 北京
NO_OVERNIGHT_THU_AFTER = "06:30"    # 14:30 北京（周四）
OVERNIGHT_CONSIDER_AFTER = "06:00"  # 14:00 北京


# ---------------------------------------------------------------------------
# 信号数据结构
# ---------------------------------------------------------------------------

@dataclass
class IntradaySignal:
    """日内交易信号"""
    symbol: str          # IF / IH / IM
    datetime: str        # 信号产生时间（UTC）
    direction: str       # LONG / SHORT
    score: int           # 0-100 信号强度
    entry_price: float   # 建议入场价
    stop_loss: float     # 止损价（距入场≤100基点）
    signal_type: str     # INTRADAY / OVERNIGHT
    components: Dict     # 各维度得分明细
    reason: str          # 信号描述


# ---------------------------------------------------------------------------
# 信号生成器
# ---------------------------------------------------------------------------

class IntradaySignalGenerator:
    """多维度信号评分系统。"""

    def __init__(self, config: Dict | None = None):
        cfg = config or {}
        self.opening_range_minutes: int = cfg.get("opening_range_minutes", 30)
        self.vwap_deviation_threshold: float = cfg.get("vwap_deviation_threshold", 0.002)
        self.volume_surge_ratio: float = cfg.get("volume_surge_ratio", 1.5)
        self.atr_period: int = cfg.get("atr_period", 20)
        self.trend_fast_period: int = cfg.get("trend_fast_period", 10)
        self.trend_slow_period: int = cfg.get("trend_slow_period", 30)
        self.min_signal_score: int = cfg.get("min_signal_score", 55)

        # 每个5分钟区间多少根K线组成开盘区间
        # 300s(5m) → 6根，900s(15m) → 2根
        self._opening_bars: int = cfg.get("opening_bars", 6)

        # debug 模式：在 monitor 中打印维度计算细节
        self.debug: bool = cfg.get("debug", False)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def update(
        self,
        symbol: str,
        bar_5m: pd.DataFrame,
        bar_15m: pd.DataFrame | None,
        daily_bar: pd.DataFrame | None,
        quote_data: Dict | None = None,
    ) -> Optional[IntradaySignal]:
        """
        输入最新K线数据，输出信号（或 None）。

        bar_5m  : 当日5分钟K线（至少包含开盘区间+当前bar）
        bar_15m : 当日15分钟K线
        daily_bar: 最近日线（≥20根）
        quote_data: 盘口数据（仅实时模式），回测时为 None
        """
        if bar_5m is None or len(bar_5m) < 2:
            return None

        current_bar = bar_5m.iloc[-1]
        current_time = self._bar_dt_str(bar_5m, -1)
        time_str = current_time.split(" ")[-1][:5]  # "HH:MM"

        # 时间窗口过滤
        if time_str < NO_TRADE_BEFORE:
            return None
        if time_str >= NO_NEW_INTRADAY_AFTER:
            return None

        # 开盘区间
        opening_range = self._calc_opening_range(bar_5m)
        if opening_range is None:
            return None

        # 各维度评分
        s1, d1 = self._score_opening_range_breakout(bar_5m, opening_range)
        s2, d2 = self._score_vwap(bar_5m)
        s3, d3 = self._score_multi_timeframe(bar_5m, bar_15m)
        s4 = self._score_volume(bar_5m, opening_range)
        s5, d5 = self._score_daily_levels(bar_5m, daily_bar)
        s6, d6 = self._score_orderbook(quote_data)
        s7, d7 = self._score_bollinger(bar_5m, bar_15m, daily_bar)

        scores: List[Tuple[int, str, str]] = [
            (s1, d1, "opening_breakout"),
            (s2, d2, "vwap"),
            (s3, d3, "multi_tf"),
            (s4, "", "volume"),
            (s5, d5, "daily_levels"),
            (s6, d6, "orderbook"),
            (s7, d7, "bollinger"),
        ]

        total_score, direction = self._aggregate_scores(
            [(s, d) for s, d, _ in scores]
        )
        total_score = max(0, min(100, total_score))

        if total_score < self.min_signal_score or not direction:
            return None

        entry_price = float(current_bar["close"])
        stop_loss = self._calc_stop_loss(direction, entry_price, bar_5m, symbol)
        signal_type = self._determine_hold_type(bar_5m, total_score, direction)

        components = {name: {"score": s, "direction": d} for s, d, name in scores}

        parts = []
        if s1 > 0:
            parts.append(f"开盘突破+{s1}")
        if s2 > 0:
            parts.append(f"VWAP+{s2}")
        if s3 > 0:
            parts.append(f"多周期+{s3}")
        if s4 > 0:
            parts.append(f"量能+{s4}")
        if s5 > 0:
            parts.append(f"日线+{s5}")
        if s6 > 0:
            parts.append(f"盘口+{s6}")
        if s7 > 0:
            parts.append(f"BOLL+{s7}")

        return IntradaySignal(
            symbol=symbol,
            datetime=current_time,
            direction=direction,
            score=total_score,
            entry_price=entry_price,
            stop_loss=stop_loss,
            signal_type=signal_type,
            components=components,
            reason=", ".join(parts),
        )

    def score_all(
        self,
        symbol: str,
        bar_5m: pd.DataFrame,
        bar_15m: pd.DataFrame | None,
        daily_bar: pd.DataFrame | None,
        quote_data: Dict | None = None,
    ) -> Dict | None:
        """返回所有维度得分（不受 min_signal_score 过滤），供面板显示用。"""
        if bar_5m is None or len(bar_5m) < 2:
            return None

        opening_range = self._calc_opening_range(bar_5m)

        # VWAP
        today = self._today_bars(bar_5m)
        vwap_val = 0.0
        vwap_dev = 0.0
        if len(today) >= 3:
            closes = today["close"].values.astype(float)
            volumes = today["volume"].values.astype(float)
            cum_vol = np.cumsum(volumes)
            cum_pv = np.cumsum(closes * volumes)
            if cum_vol[-1] > 0:
                vwap_val = cum_pv[-1] / cum_vol[-1]
                vwap_dev = (closes[-1] - vwap_val) / vwap_val if vwap_val else 0

        # 各维度评分
        s1, d1 = self._score_opening_range_breakout(bar_5m, opening_range) \
            if opening_range else (0, "")
        s2, d2 = self._score_vwap(bar_5m)
        s3, d3 = self._score_multi_timeframe(bar_5m, bar_15m)
        s4 = self._score_volume(bar_5m, opening_range)
        s5, d5 = self._score_daily_levels(bar_5m, daily_bar)
        s6, d6 = self._score_orderbook(quote_data)
        s7, d7 = self._score_bollinger(bar_5m, bar_15m, daily_bar)

        total, direction = self._aggregate_scores([
            (s1, d1), (s2, d2), (s3, d3), (s4, ""), (s5, d5), (s6, d6),
            (s7, d7),
        ])
        total = max(0, min(100, total))

        # 趋势方向
        dir_5m = self._trend_direction(bar_5m, self.trend_fast_period, self.trend_slow_period)
        dir_15m = ""
        if bar_15m is not None and len(bar_15m) >= self.trend_slow_period:
            dir_15m = self._trend_direction(bar_15m, self.trend_fast_period, self.trend_slow_period)

        # 布林带状态
        boll_5m = self._boll_position_label(bar_5m)
        boll_15m = self._boll_position_label(bar_15m) if bar_15m is not None else ""
        boll_daily = self._boll_position_label(daily_bar) if daily_bar is not None else ""

        return {
            "total": total,
            "direction": direction,
            "s_breakout": s1, "s_vwap": s2, "s_multi_tf": s3,
            "s_volume": s4, "s_daily": s5, "s_orderbook": s6,
            "s_bollinger": s7,
            "vwap": vwap_val,
            "vwap_dev": vwap_dev,
            "or_high": opening_range["high"] if opening_range else 0,
            "or_low": opening_range["low"] if opening_range else 0,
            "trend_5m": dir_5m,
            "trend_15m": dir_15m,
            "boll_5m": boll_5m,
            "boll_15m": boll_15m,
            "boll_daily": boll_daily,
        }

    # ------------------------------------------------------------------
    # 维度1：开盘区间突破（25分）
    # ------------------------------------------------------------------

    @staticmethod
    def _bar_dt_str(bars: pd.DataFrame, idx: int = -1) -> str:
        """从 bars 中获取指定行的日期时间字符串，兼容 DatetimeIndex 和 TQ 整数 index。"""
        if isinstance(bars.index, pd.DatetimeIndex):
            return str(bars.index[idx])
        # TQ 返回的 DataFrame：index 是整数，日期在 'datetime' 列（纳秒 epoch）
        if "datetime" in bars.columns:
            return str(pd.to_datetime(bars["datetime"].iloc[idx], unit="ns"))
        return str(bars.iloc[idx].name)

    @staticmethod
    def _today_bars(bars: pd.DataFrame) -> pd.DataFrame:
        """从可能跨日的rolling bars中提取当日bars。"""
        if len(bars) == 0:
            return bars
        if isinstance(bars.index, pd.DatetimeIndex):
            last_date = bars.index[-1].strftime("%Y-%m-%d")
            mask = bars.index.strftime("%Y-%m-%d") == last_date
        elif "datetime" in bars.columns:
            dt_col = pd.to_datetime(bars["datetime"], unit="ns")
            last_date = dt_col.iloc[-1].date()
            mask = dt_col.dt.date == last_date
        else:
            return bars  # fallback: 无法判断日期，返回全部
        return bars[mask]

    def _calc_opening_range(self, bars: pd.DataFrame) -> Optional[Dict]:
        n = self._opening_bars
        today = self._today_bars(bars)
        if len(today) <= n:
            return None  # 还在开盘区间内，不产生信号
        opening = today.iloc[:n]
        high = float(opening["high"].max())
        low = float(opening["low"].min())
        return {"high": high, "low": low, "mid": (high + low) / 2,
                "width": high - low}

    def _score_opening_range_breakout(
        self, bars: pd.DataFrame, opening_range: Dict
    ) -> Tuple[int, str]:
        score = 0
        direction = ""
        close = float(bars.iloc[-1]["close"])
        or_high = opening_range["high"]
        or_low = opening_range["low"]

        # 下午衰减：开盘区间突破在下午信号减弱（需要更强的突破）
        current_time = self._bar_dt_str(bars, -1)
        time_str = current_time.split(" ")[-1][:5]
        afternoon_decay = 0.5 if time_str >= AFTERNOON_START else 1.0

        # ATR for context
        n_atr = min(self.atr_period, len(bars) - 1)
        if n_atr >= 2:
            tr = bars["high"].iloc[-n_atr:].astype(float) - bars["low"].iloc[-n_atr:].astype(float)
            atr = float(tr.mean())
        else:
            atr = opening_range["width"] or 1.0

        # 检查今日所有K线的极值（突破"记忆"）
        today = self._today_bars(bars)
        intraday_high = float(today["high"].max()) if len(today) > 0 else close
        intraday_low = float(today["low"].min()) if len(today) > 0 else close
        ever_broke_high = intraday_high > or_high
        ever_broke_low = intraday_low < or_low

        if close > or_high:
            # 当前仍在突破状态
            direction = "LONG"
            score += 20
            if atr > 0 and (close - or_high) > atr * 0.3:
                score += 5
            if len(bars) >= 2:
                prev_close = float(bars.iloc[-2]["close"])
                prev_low = float(bars.iloc[-2]["low"])
                if prev_close > or_high and prev_low >= or_high:
                    score += 5  # 回踩确认

        elif close < or_low:
            # 当前仍在突破状态
            direction = "SHORT"
            score += 20
            if atr > 0 and (or_low - close) > atr * 0.3:
                score += 5
            if len(bars) >= 2:
                prev_close = float(bars.iloc[-2]["close"])
                prev_high = float(bars.iloc[-2]["high"])
                if prev_close < or_low and prev_high <= or_low:
                    score += 5

        elif ever_broke_high and close >= opening_range["mid"]:
            # 曾经突破上轨，价格回到区间内但仍在中轨以上
            direction = "LONG"
            score += 10  # 曾经突破，给部分分
        elif ever_broke_low and close <= opening_range["mid"]:
            # 曾经突破下轨，价格回到区间内但仍在中轨以下
            direction = "SHORT"
            score += 10

        # 下午衰减：开盘区间突破在下午的意义减弱
        score = int(score * afternoon_decay)
        return min(25, score), direction

    # ------------------------------------------------------------------
    # 维度2：VWAP偏离与回归（21分）
    # ------------------------------------------------------------------

    def _score_vwap(self, bars: pd.DataFrame) -> Tuple[int, str]:
        today = self._today_bars(bars)
        if len(today) < 3:
            return 0, ""

        closes = today["close"].values.astype(float)
        volumes = today["volume"].values.astype(float)

        cum_vol = np.cumsum(volumes)
        cum_pv = np.cumsum(closes * volumes)
        mask = cum_vol > 0
        if not mask.any():
            return 0, ""

        vwap = np.where(mask, cum_pv / cum_vol, closes)
        current_close = closes[-1]
        current_vwap = vwap[-1]
        if current_vwap == 0:
            return 0, ""

        deviation = (current_close - current_vwap) / current_vwap
        vwap_slope = (vwap[-1] - vwap[max(0, len(vwap) - 5)]) if len(vwap) >= 5 else 0

        score = 0
        direction = ""
        threshold = self.vwap_deviation_threshold

        if self.debug:
            print(f"    [VWAP] price={current_close:.1f}  vwap={current_vwap:.1f}"
                  f"  偏离={deviation*100:+.3f}%  阈值={threshold*100:.2f}%"
                  f"  slope={vwap_slope:.2f}")

        # --- 均值回归 ---
        if deviation < -threshold and len(closes) >= 4:
            recent_lows = bars["low"].iloc[-3:].values.astype(float)
            if self.debug:
                print(f"    [VWAP] 均值回归(空→多): 低点序列={recent_lows}"
                      f"  抬高={recent_lows[-1] > recent_lows[0]}")
            if recent_lows[-1] > recent_lows[0]:  # 低点抬高
                direction = "LONG"
                score += 10
                slope = (closes[-1] - closes[-3]) / abs(current_vwap) * 100
                score += min(10, max(0, int(slope * 20)))
                if vwap_slope > 0:
                    score += 5

        elif deviation > threshold and len(closes) >= 4:
            recent_highs = bars["high"].iloc[-3:].values.astype(float)
            if self.debug:
                print(f"    [VWAP] 均值回归(多→空): 高点序列={recent_highs}"
                      f"  压低={recent_highs[-1] < recent_highs[0]}")
            if recent_highs[-1] < recent_highs[0]:
                direction = "SHORT"
                score += 10
                slope = (closes[-3] - closes[-1]) / abs(current_vwap) * 100
                score += min(10, max(0, int(slope * 20)))
                if vwap_slope < 0:
                    score += 5

        # --- 趋势跟随（价格加速远离VWAP） ---
        trend_threshold = threshold * 1.5  # 趋势跟随用1.5倍偏离阈值
        if score == 0 and len(vwap) >= 3:
            if self.debug and abs(deviation) > threshold:
                prev_dev = (closes[-3] - vwap[-3]) / vwap[-3] if vwap[-3] != 0 else 0
                print(f"    [VWAP] 趋势跟随检查: dev={deviation*100:+.3f}%"
                      f"  trend_th={trend_threshold*100:.2f}%"
                      f"  prev_dev={prev_dev*100:+.3f}%  加速={abs(deviation)>abs(prev_dev)}")
            if deviation > trend_threshold:
                prev_dev = (closes[-3] - vwap[-3]) / vwap[-3] if vwap[-3] != 0 else 0
                if deviation > prev_dev:
                    direction = "LONG"
                    score += 10
                    avg_vol = volumes[max(0, len(volumes)-5):].mean()
                    if avg_vol > 0 and volumes[-1] > avg_vol * 1.3:
                        score += 10
                    if vwap_slope > 0:
                        score += 5
            elif deviation < -trend_threshold:
                prev_dev = (closes[-3] - vwap[-3]) / vwap[-3] if vwap[-3] != 0 else 0
                if deviation < prev_dev:
                    direction = "SHORT"
                    score += 10
                    avg_vol = volumes[max(0, len(volumes)-5):].mean()
                    if avg_vol > 0 and volumes[-1] > avg_vol * 1.3:
                        score += 10
                    if vwap_slope < 0:
                        score += 5

        final_score = min(21, score)
        if self.debug:
            print(f"    [VWAP] → score={final_score}  dir={direction or 'N/A'}")
        return final_score, direction

    # ------------------------------------------------------------------
    # 维度3：多周期趋势一致性（16分）
    # ------------------------------------------------------------------

    def _score_multi_timeframe(
        self, bar_5m: pd.DataFrame, bar_15m: pd.DataFrame | None
    ) -> Tuple[int, str]:
        dir_5m = self._trend_direction(bar_5m, self.trend_fast_period, self.trend_slow_period)
        dir_15m = ""
        if bar_15m is not None and len(bar_15m) >= self.trend_slow_period:
            dir_15m = self._trend_direction(bar_15m, self.trend_fast_period, self.trend_slow_period)

        # 15m回测模式：bar_5m 实际是15m数据，bar_15m 可能是5m或None
        # 此时只看主周期趋势方向
        if self._opening_bars == 2:
            # 15分钟模式：主周期有方向即给基础分12
            if dir_5m:
                if dir_15m and dir_15m == dir_5m:
                    return 16, dir_5m
                elif dir_15m and dir_15m != dir_5m:
                    return 0, ""  # 矛盾
                return 12, dir_5m
            return 0, ""

        # 5分钟模式
        if dir_5m:
            score = 8  # 5m有方向即给基础分8
            if dir_15m:
                if dir_15m == dir_5m:
                    score += 8  # 15m同向额外+8
                else:
                    return 0, ""  # 矛盾 → 不交易
            return score, dir_5m
        return 0, ""

    @staticmethod
    def _trend_direction(bars: pd.DataFrame, fast: int, slow: int) -> str:
        if len(bars) < slow:
            return ""
        closes = bars["close"].astype(float)
        sma_fast = closes.rolling(fast).mean()
        sma_slow = closes.rolling(slow).mean()
        if pd.isna(sma_fast.iloc[-1]) or pd.isna(sma_slow.iloc[-1]):
            return ""
        current = float(closes.iloc[-1])
        f_val = float(sma_fast.iloc[-1])
        s_val = float(sma_slow.iloc[-1])
        if f_val > s_val and current > f_val:
            return "LONG"
        if f_val < s_val and current < f_val:
            return "SHORT"
        return ""

    # ------------------------------------------------------------------
    # 维度4：成交量确认（11分）
    # ------------------------------------------------------------------

    def _score_volume(
        self, bars: pd.DataFrame, opening_range: Dict | None = None,
    ) -> int:
        if len(bars) < 2:
            return 0
        volumes = bars["volume"].astype(float)
        current_vol = float(volumes.iloc[-1])
        lookback = min(20, len(volumes) - 1)
        if lookback < 1:
            return 0
        avg_vol = float(volumes.iloc[-lookback - 1:-1].mean())
        if avg_vol <= 0:
            return 0

        score = 0

        # --- 子维度a：相对量能 ---
        ratio = current_vol / avg_vol
        if ratio >= 2.0:
            score += 8
        elif ratio >= self.volume_surge_ratio:
            score += 5
        elif ratio < 0.5:
            score -= 5

        # --- 子维度b：量能趋势（最近5根K线成交量线性斜率） ---
        n_trend = min(5, len(volumes))
        if n_trend >= 3:
            recent_vols = volumes.iloc[-n_trend:].values.astype(float)
            x = np.arange(n_trend, dtype=float)
            slope = np.polyfit(x, recent_vols, 1)[0]
            if slope > 0:
                score += 5  # 量能递增
            elif slope < 0:
                score -= 3  # 量能递减

        # --- 子维度c：关键位置量价配合 ---
        if opening_range is not None and len(bars) >= 1:
            close = float(bars.iloc[-1]["close"])
            open_ = float(bars.iloc[-1]["open"])
            or_high = opening_range["high"]
            or_low = opening_range["low"]
            or_width = opening_range["width"] or 1.0
            tol = or_width * 0.15  # 15% of range width as tolerance

            is_bullish = close > open_
            is_bearish = close < open_
            is_high_vol = ratio >= self.volume_surge_ratio
            is_low_vol = ratio < 0.8

            # 价格在开盘区间上沿 + 放量阳线 → 突破确认
            if abs(close - or_high) < tol or close > or_high:
                if is_bullish and is_high_vol:
                    score += 5
                # 价格在开盘区间上沿 + 缩量阴线 → 假突破嫌疑
                elif is_bearish and is_low_vol:
                    score -= 5

            # 价格在开盘区间下沿 + 放量阳线 → 支撑有效
            if abs(close - or_low) < tol or close < or_low:
                if is_bullish and is_high_vol:
                    score += 5

        return max(-5, min(11, score))

    # ------------------------------------------------------------------
    # 维度5：日线级支撑阻力（7分）
    # ------------------------------------------------------------------

    def _score_daily_levels(
        self, bar_5m: pd.DataFrame, daily_bar: pd.DataFrame | None
    ) -> Tuple[int, str]:
        if daily_bar is None or len(daily_bar) < 2:
            return 0, ""

        score = 0
        direction = ""
        current_close = float(bar_5m.iloc[-1]["close"])

        prev_day = daily_bar.iloc[-1]
        prev_high = float(prev_day["high"])
        prev_low = float(prev_day["low"])
        tol = current_close * 0.002

        n_sma = min(20, len(daily_bar))
        sma20 = float(daily_bar["close"].iloc[-n_sma:].astype(float).mean())

        rising = (len(bar_5m) >= 2 and
                  float(bar_5m.iloc[-1]["close"]) > float(bar_5m.iloc[-2]["close"]))
        falling = (len(bar_5m) >= 2 and
                   float(bar_5m.iloc[-1]["close"]) < float(bar_5m.iloc[-2]["close"]))

        # 做多：从前日低点附近反弹 / 站上20日均线
        if abs(current_close - prev_low) < tol and rising:
            score += 5
            direction = "LONG"
        if current_close > sma20:
            if direction == "LONG":
                score += 5
            elif not direction:
                score += 5
                direction = "LONG"

        # 做空：从前日高点附近回落 / 跌破20日均线
        if abs(current_close - prev_high) < tol and falling:
            if not direction:
                score += 5
                direction = "SHORT"
        if current_close < sma20:
            if direction == "SHORT":
                score += 5
            elif not direction:
                score += 5
                direction = "SHORT"

        return min(7, score), direction

    # ------------------------------------------------------------------
    # 维度6：盘口压力（8分）— 仅实时模式
    # ------------------------------------------------------------------

    def _score_orderbook(
        self, quote_data: Dict | None,
    ) -> Tuple[int, str]:
        """
        盘口压力评分。仅实时模式可用，回测时 quote_data=None 返回 (0, "")。

        quote_data keys: bid_price1, ask_price1, bid_volume1, ask_volume1,
                         last_price
        """
        if quote_data is None:
            return 0, ""

        bid_vol = float(quote_data.get("bid_volume1", 0))
        ask_vol = float(quote_data.get("ask_volume1", 0))
        bid_price = float(quote_data.get("bid_price1", 0))
        ask_price = float(quote_data.get("ask_price1", 0))
        last_price = float(quote_data.get("last_price", 0))

        total_vol = bid_vol + ask_vol
        if total_vol <= 0 or bid_price <= 0 or ask_price <= 0:
            return 0, ""

        score = 0
        direction = ""

        # --- 买卖压力 ---
        pressure = bid_vol / total_vol
        if pressure > 0.7:
            score += 8
            direction = "LONG"
        elif pressure > 0.6:
            score += 5
            direction = "LONG"
        elif pressure < 0.3:
            score += 8
            direction = "SHORT"
        elif pressure < 0.4:
            score += 5
            direction = "SHORT"

        # --- 价差异常（流动性下降） ---
        mid_price = (bid_price + ask_price) / 2
        spread_bps = (ask_price - bid_price) / mid_price * 10000
        # 股指期货正常价差约 0.2-1 个指数点，约 0.5-3 bps
        if spread_bps > 6:  # 价差明显扩大
            score -= 3

        # --- 成交方向推断 ---
        if last_price > 0 and direction:
            if last_price >= ask_price:
                if direction == "LONG":
                    score += 2  # 主动买入，加强做多
            elif last_price <= bid_price:
                if direction == "SHORT":
                    score += 2  # 主动卖出，加强做空

        return max(-3, min(8, score)), direction

    # ------------------------------------------------------------------
    # 维度7：布林带多周期（12分）
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_bollinger(
        df: pd.DataFrame, period: int = 20, num_std: float = 2.0,
    ) -> Optional[Dict]:
        """计算布林带上中下轨，返回 {upper, middle, lower} 序列。"""
        if df is None or len(df) < period:
            return None
        closes = df["close"].astype(float)
        middle = closes.rolling(period).mean()
        std = closes.rolling(period).std()
        upper = middle + num_std * std
        lower = middle - num_std * std
        if pd.isna(middle.iloc[-1]):
            return None
        return {
            "upper": upper.values,
            "middle": middle.values,
            "lower": lower.values,
        }

    @staticmethod
    def _boll_position_label(df: pd.DataFrame, period: int = 20) -> str:
        """返回当前价格相对布林带的位置标签。"""
        if df is None or len(df) < period:
            return ""
        closes = df["close"].astype(float)
        middle = closes.rolling(period).mean()
        std = closes.rolling(period).std()
        if pd.isna(middle.iloc[-1]) or pd.isna(std.iloc[-1]):
            return ""
        m = float(middle.iloc[-1])
        s = float(std.iloc[-1])
        c = float(closes.iloc[-1])
        upper = m + 2 * s
        lower = m - 2 * s
        if c > upper:
            return "上轨上"
        elif c > m:
            return "中轨上"
        elif c > lower:
            return "中轨下"
        else:
            return "下轨下"

    def _score_bollinger(
        self,
        bar_5m: pd.DataFrame,
        bar_15m: pd.DataFrame | None,
        daily_bar: pd.DataFrame | None,
    ) -> Tuple[int, str]:
        """
        布林带多周期评分（上限12分）。

        三个时间周期独立判断布林带状态，然后叠加评分。
        方向冲突时取较高周期方向。
        """
        long_score = 0
        short_score = 0
        long_tfs = 0
        short_tfs = 0

        for label, df, base_break, base_cross in [
            ("5m", bar_5m, 5, 3),
            ("15m", bar_15m, 4, 3),
            ("daily", daily_bar, 2, 1),
        ]:
            boll = self._calc_bollinger(df)
            if boll is None or df is None:
                continue

            closes = df["close"].astype(float).values
            upper = boll["upper"]
            middle = boll["middle"]
            lower = boll["lower"]
            n = len(closes)

            # 跌破下轨（超卖）→ 做多
            if n >= 2 and closes[-1] < lower[-1] and closes[-2] < lower[-2]:
                long_score += base_break
                long_tfs += 1
            # 向上突破中轨（趋势确认）→ 做多
            elif n >= 3 and closes[-1] > middle[-1] and closes[-2] > middle[-2]:
                # 确认是"突破"：前面有K线在中轨下方
                if any(closes[max(0, n - 6):n - 2] < middle[max(0, n - 6):n - 2]):
                    long_score += base_cross
                    long_tfs += 1

            # 抵达上轨（超买）→ 做空
            if n >= 2 and closes[-1] > upper[-1] and closes[-2] > upper[-2]:
                short_score += base_break
                short_tfs += 1
            # 跌破中轨（趋势反转）→ 做空
            elif n >= 3 and closes[-1] < middle[-1] and closes[-2] < middle[-2]:
                if any(closes[max(0, n - 6):n - 2] > middle[max(0, n - 6):n - 2]):
                    short_score += base_cross
                    short_tfs += 1

        # 多周期共振加分
        if long_tfs >= 2:
            long_score += 2
        if short_tfs >= 2:
            short_score += 2

        # 方向判定
        if long_score > 0 and short_score > 0:
            # 冲突：取高周期方向，得分减半
            if long_score >= short_score:
                return min(12, long_score // 2), "LONG"
            else:
                return min(12, short_score // 2), "SHORT"

        if long_score > 0:
            return min(12, long_score), "LONG"
        if short_score > 0:
            return min(12, short_score), "SHORT"
        return 0, ""

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------

    def _aggregate_scores(self, scores: List[Tuple[int, str]]) -> Tuple[int, str]:
        long_score = short_score = neutral_score = 0
        long_dims = short_dims = 0

        for s, d in scores:
            if d == "LONG":
                long_score += s
                long_dims += 1
            elif d == "SHORT":
                short_score += s
                short_dims += 1
            else:
                neutral_score += s

        if long_dims > 0 and short_dims > 0:
            if long_score > short_score * 1.5:
                return long_score + neutral_score - short_score, "LONG"
            if short_score > long_score * 1.5:
                return short_score + neutral_score - long_score, "SHORT"
            return 0, ""

        if long_dims > 0:
            return long_score + neutral_score, "LONG"
        if short_dims > 0:
            return short_score + neutral_score, "SHORT"
        return 0, ""

    # ------------------------------------------------------------------
    # 止损
    # ------------------------------------------------------------------

    def _calc_stop_loss(
        self, direction: str, entry_price: float,
        bars: pd.DataFrame, symbol: str,
    ) -> float:
        max_dist = entry_price * 0.01  # 100 bps
        lookback = min(3, len(bars) - 1)
        if lookback < 1:
            return (entry_price - max_dist) if direction == "LONG" \
                else (entry_price + max_dist)

        recent = bars.iloc[-lookback:]
        buffer = entry_price * 0.0005

        if direction == "LONG":
            smart = float(recent["low"].min()) - buffer
            return max(smart, entry_price - max_dist)
        else:
            smart = float(recent["high"].max()) + buffer
            return min(smart, entry_price + max_dist)

    # ------------------------------------------------------------------
    # 日内 / 隔日判定
    # ------------------------------------------------------------------

    def _determine_hold_type(
        self, bars: pd.DataFrame, score: int, direction: str,
    ) -> str:
        if score < 75:
            return "INTRADAY"

        current_time = self._bar_dt_str(bars, -1)
        time_str = current_time.split(" ")[-1][:5]

        if time_str < OVERNIGHT_CONSIDER_AFTER:
            return "INTRADAY"

        try:
            dt = pd.Timestamp(current_time)
            wd = dt.weekday()
            if wd == 3 and time_str >= NO_OVERNIGHT_THU_AFTER:
                return "INTRADAY"
            if wd >= 4:
                return "INTRADAY"
        except Exception:
            pass

        return "OVERNIGHT"
