"""
momentum_research.py
--------------------
A股股指期货动量/均值回归特征实证研究。

数据来源：
- futures_daily: IF.CFX / IH.CFX / IM.CFX 日线
- futures_min:   IF / IH / IM  5分钟线（period=300）

输出：终端打印 + logs/research/momentum_research_YYYYMMDD.md
"""

from __future__ import annotations

import sys
import os
from datetime import date, datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

# ── project imports ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.config_loader import ConfigLoader
from data.storage.db_manager import DBManager

# ── globals ──────────────────────────────────────────────────────
TODAY = date.today().strftime("%Y%m%d")
SYMBOLS_DAILY = {"IF": "IF.CFX", "IH": "IH.CFX", "IM": "IM.CFX"}
SYMBOLS_MIN = ["IF", "IH", "IM"]
LOOKBACKS = [1, 3, 5, 10, 20, 40, 60]
THRESHOLDS = [0.01, 0.015, 0.02]  # 1%, 1.5%, 2%
OUTPUT_BUF = StringIO()


def tee(text: str = ""):
    """Print to terminal and capture for markdown."""
    print(text)
    OUTPUT_BUF.write(text + "\n")


# ═══════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════

def load_daily(db: DBManager) -> dict[str, pd.DataFrame]:
    """Load daily bars for IF/IH/IM, sorted by date."""
    data = {}
    for name, ts_code in SYMBOLS_DAILY.items():
        df = db.query_df(
            "SELECT trade_date, open, high, low, close, volume "
            "FROM futures_daily WHERE ts_code = ? AND close > 0 "
            "ORDER BY trade_date",
            params=(ts_code,),
        )
        df["trade_date"] = df["trade_date"].astype(str)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        df = df.reset_index(drop=True)
        data[name] = df
        tee(f"  {name}: {len(df)} 日线  ({df['trade_date'].iloc[0]} ~ {df['trade_date'].iloc[-1]})")
    return data


def load_min5(db: DBManager) -> dict[str, pd.DataFrame]:
    """Load 5-min bars for IF/IH/IM."""
    data = {}
    for sym in SYMBOLS_MIN:
        df = db.query_df(
            "SELECT symbol, datetime, open, high, low, close, volume "
            "FROM futures_min WHERE symbol = ? AND period = 300 "
            "ORDER BY datetime",
            params=(sym,),
        )
        df["datetime"] = pd.to_datetime(df["datetime"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        df["date"] = df["datetime"].dt.date
        data[sym] = df
        n_days = df["date"].nunique()
        tee(f"  {sym}: {len(df)} 根5分钟K线  ({n_days} 交易日)")
    return data


# ═══════════════════════════════════════════════════════════════
#  研究1：各品种的动量特征差异
# ═══════════════════════════════════════════════════════════════

def research_1(daily: dict[str, pd.DataFrame]):
    tee("\n" + "═" * 60)
    tee("  研究1：各品种的动量特征差异")
    tee("═" * 60)

    # ── 1a: 动量因子收益 ──
    tee("\n── 1a: 不同回看期的动量因子收益 ──\n")
    tee(f"  {'品种':>4}  {'回看期':>6}  {'年化收益':>8}  {'夏普':>6}  {'胜率':>6}  {'最大回撤':>8}")
    tee("  " + "─" * 52)

    best_by_sym: dict[str, tuple] = {}  # sym -> (lookback, sharpe)

    for sym, df in daily.items():
        closes = df["close"].values
        # daily returns
        daily_ret = np.diff(closes) / closes[:-1]  # ret[i] = close[i+1]/close[i] - 1

        for lb in LOOKBACKS:
            if len(closes) < lb + 2:
                continue
            # momentum signal: past lb days return
            signals = []
            returns = []
            for i in range(lb, len(daily_ret)):
                past_ret = (closes[i] - closes[i - lb]) / closes[i - lb]
                sign = 1.0 if past_ret > 0 else -1.0
                signals.append(sign)
                returns.append(daily_ret[i])  # next day return

            signals = np.array(signals)
            returns = np.array(returns)
            strat_ret = signals * returns

            ann_ret = float(np.mean(strat_ret)) * 242
            ann_std = float(np.std(strat_ret, ddof=1)) * np.sqrt(242)
            sharpe = ann_ret / ann_std if ann_std > 0 else 0
            win_rate = float(np.mean(strat_ret > 0)) * 100

            # max drawdown
            cum = np.cumsum(strat_ret)
            peak = np.maximum.accumulate(cum)
            dd = peak - cum
            max_dd = float(np.max(dd)) * 100 if len(dd) > 0 else 0

            tee(f"  {sym:>4}  {lb:>4}天  {ann_ret*100:>+7.1f}%  {sharpe:>6.2f}  {win_rate:>5.1f}%  {max_dd:>7.1f}%")

            if sym not in best_by_sym or sharpe > best_by_sym[sym][1]:
                best_by_sym[sym] = (lb, sharpe)

        tee("")

    tee("  最优回看期：")
    for sym, (lb, sharpe) in best_by_sym.items():
        tee(f"    {sym}: {lb}天 (夏普={sharpe:.2f})")

    # ── 1b: 动量 vs 均值回归 ──
    tee("\n── 1b: 动量 vs 均值回归 ──\n")
    tee(f"  {'品种':>4}  {'回看期':>6}  {'P(涨|过去涨)':>12}  {'P(跌|过去跌)':>12}  {'判断':>8}")
    tee("  " + "─" * 56)

    for sym, df in daily.items():
        closes = df["close"].values
        daily_ret = np.diff(closes) / closes[:-1]

        for lb in LOOKBACKS:
            if len(closes) < lb + 2:
                continue
            up_then_up = 0
            up_count = 0
            down_then_down = 0
            down_count = 0

            for i in range(lb, len(daily_ret)):
                past_ret = (closes[i] - closes[i - lb]) / closes[i - lb]
                next_ret = daily_ret[i]

                if past_ret > 0:
                    up_count += 1
                    if next_ret > 0:
                        up_then_up += 1
                elif past_ret < 0:
                    down_count += 1
                    if next_ret < 0:
                        down_then_down += 1

            p_up = up_then_up / up_count * 100 if up_count > 0 else 50
            p_down = down_then_down / down_count * 100 if down_count > 0 else 50
            avg_p = (p_up + p_down) / 2

            if avg_p > 52:
                label = "动量"
            elif avg_p < 48:
                label = "均值回归"
            else:
                label = "中性"

            tee(f"  {sym:>4}  {lb:>4}天  {p_up:>10.1f}%  {p_down:>10.1f}%  {label:>8}")

        tee("")


# ═══════════════════════════════════════════════════════════════
#  研究2：日内动量特征（5分钟线）
# ═══════════════════════════════════════════════════════════════

def research_2(min5: dict[str, pd.DataFrame]):
    tee("\n" + "═" * 60)
    tee("  研究2：日内动量特征（5分钟线）")
    tee("═" * 60 + "\n")

    # Bars per time window: 30min=6bars, 60min=12bars, 90min=18bars
    windows = [(30, 6), (60, 12), (90, 18)]

    tee(f"  {'品种':>4}  {'时段':>16}  {'一致性概率':>10}  {'样本数':>6}")
    tee("  " + "─" * 46)

    for sym, df in min5.items():
        dates = sorted(df["date"].unique())
        for win_min, n_bars in windows:
            consistent = 0
            total = 0
            for d in dates:
                day_df = df[df["date"] == d].sort_values("datetime")
                if len(day_df) < n_bars + 2:
                    continue
                open_price = float(day_df.iloc[0]["open"])
                early_close = float(day_df.iloc[n_bars - 1]["close"])
                day_close = float(day_df.iloc[-1]["close"])

                early_dir = early_close - open_price
                full_dir = day_close - open_price

                if early_dir == 0 or full_dir == 0:
                    continue

                total += 1
                if (early_dir > 0 and full_dir > 0) or (early_dir < 0 and full_dir < 0):
                    consistent += 1

            pct = consistent / total * 100 if total > 0 else 0
            tee(f"  {sym:>4}  前{win_min}分钟→全天  {pct:>9.1f}%  {total:>6}")
        tee("")

    tee("  解读：> 55% 表示日内动量较强，< 45% 偏均值回归")


# ═══════════════════════════════════════════════════════════════
#  研究3：日内涨跌幅反转效应
# ═══════════════════════════════════════════════════════════════

def research_3(min5: dict[str, pd.DataFrame]):
    tee("\n" + "═" * 60)
    tee("  研究3：日内涨跌幅反转效应")
    tee("═" * 60 + "\n")

    tee("── 超跌后反弹 ──\n")
    tee(f"  {'品种':>4}  {'阈值':>8}  {'触发次数':>8}  {'后续平均收益':>12}  {'反弹概率':>8}")
    tee("  " + "─" * 50)

    for sym, df in min5.items():
        dates = sorted(df["date"].unique())
        for thresh in THRESHOLDS:
            bounce_returns = []
            for d in dates:
                day_df = df[df["date"] == d].sort_values("datetime")
                if len(day_df) < 3:
                    continue
                open_price = float(day_df.iloc[0]["open"])
                if open_price <= 0:
                    continue
                triggered = False
                for idx in range(1, len(day_df) - 1):
                    cur_close = float(day_df.iloc[idx]["close"])
                    intra_ret = (cur_close - open_price) / open_price
                    if not triggered and intra_ret < -thresh:
                        triggered = True
                        day_close = float(day_df.iloc[-1]["close"])
                        subsequent_ret = (day_close - cur_close) / cur_close
                        bounce_returns.append(subsequent_ret)
                        break  # one trigger per day

            n_triggers = len(bounce_returns)
            if n_triggers > 0:
                avg_ret = np.mean(bounce_returns) * 100
                bounce_prob = np.mean(np.array(bounce_returns) > 0) * 100
            else:
                avg_ret = 0
                bounce_prob = 0
            tee(f"  {sym:>4}  跌>{thresh*100:.1f}%  {n_triggers:>8}  {avg_ret:>+11.3f}%  {bounce_prob:>7.1f}%")
        tee("")

    tee("── 超涨后回调 ──\n")
    tee(f"  {'品种':>4}  {'阈值':>8}  {'触发次数':>8}  {'后续平均收益':>12}  {'回调概率':>8}")
    tee("  " + "─" * 50)

    for sym, df in min5.items():
        dates = sorted(df["date"].unique())
        for thresh in THRESHOLDS:
            pullback_returns = []
            for d in dates:
                day_df = df[df["date"] == d].sort_values("datetime")
                if len(day_df) < 3:
                    continue
                open_price = float(day_df.iloc[0]["open"])
                if open_price <= 0:
                    continue
                triggered = False
                for idx in range(1, len(day_df) - 1):
                    cur_close = float(day_df.iloc[idx]["close"])
                    intra_ret = (cur_close - open_price) / open_price
                    if not triggered and intra_ret > thresh:
                        triggered = True
                        day_close = float(day_df.iloc[-1]["close"])
                        subsequent_ret = (day_close - cur_close) / cur_close
                        pullback_returns.append(subsequent_ret)
                        break

            n_triggers = len(pullback_returns)
            if n_triggers > 0:
                avg_ret = np.mean(pullback_returns) * 100
                pullback_prob = np.mean(np.array(pullback_returns) < 0) * 100
            else:
                avg_ret = 0
                pullback_prob = 0
            tee(f"  {sym:>4}  涨>{thresh*100:.1f}%  {n_triggers:>8}  {avg_ret:>+11.3f}%  {pullback_prob:>7.1f}%")
        tee("")


# ═══════════════════════════════════════════════════════════════
#  研究4：波动率收缩后突破
# ═══════════════════════════════════════════════════════════════

def _calc_atr(highs, lows, closes, period: int) -> np.ndarray:
    """Vectorized ATR calculation, returns array same length as input (NaN padded)."""
    n = len(highs)
    tr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    # simple moving average of TR
    atr = np.full(n, np.nan)
    for i in range(period, n):
        atr[i] = np.nanmean(tr[i - period + 1:i + 1])
    return atr


def research_4_daily(daily: dict[str, pd.DataFrame]):
    tee("\n" + "═" * 60)
    tee("  研究4：波动率收缩后突破")
    tee("═" * 60)

    tee("\n── 日线级别：ATR(5) / ATR(20) ──\n")
    tee(f"  {'品种':>4}  {'收缩次数':>8}  {'后5天|涨跌幅|':>14}  {'正常|涨跌幅|':>14}  {'突破概率(>2%)':>14}")
    tee("  " + "─" * 60)

    for sym, df in daily.items():
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        n = len(closes)

        atr5 = _calc_atr(highs, lows, closes, 5)
        atr20 = _calc_atr(highs, lows, closes, 20)

        squeeze_abs_rets = []
        normal_abs_rets = []
        squeeze_breakouts = 0
        squeeze_count = 0

        for i in range(20, n - 5):
            if atr20[i] <= 0 or np.isnan(atr5[i]) or np.isnan(atr20[i]):
                continue
            ratio = atr5[i] / atr20[i]
            fwd_ret = (closes[i + 5] - closes[i]) / closes[i]
            abs_fwd = abs(fwd_ret)

            if ratio < 0.7:
                squeeze_abs_rets.append(abs_fwd)
                squeeze_count += 1
                if abs_fwd > 0.02:
                    squeeze_breakouts += 1
            else:
                normal_abs_rets.append(abs_fwd)

        avg_sq = np.mean(squeeze_abs_rets) * 100 if squeeze_abs_rets else 0
        avg_nm = np.mean(normal_abs_rets) * 100 if normal_abs_rets else 0
        brk_prob = squeeze_breakouts / squeeze_count * 100 if squeeze_count > 0 else 0

        tee(f"  {sym:>4}  {squeeze_count:>8}  {avg_sq:>12.2f}%  {avg_nm:>12.2f}%  {brk_prob:>12.1f}%")

    tee("")


def research_4_intraday(min5: dict[str, pd.DataFrame]):
    tee("── 5分钟级别：ATR(5) / ATR(40)  (跨日滚动窗口) ──\n")
    tee(f"  {'品种':>4}  {'收缩次数':>8}  {'后20根|涨跌幅|':>16}  {'正常|涨跌幅|':>14}  {'突破概率(>0.5%)':>16}")
    tee("  " + "─" * 66)

    for sym, df_sym in min5.items():
        # Use full continuous series (cross-day rolling window)
        df_sorted = df_sym.sort_values("datetime").reset_index(drop=True)
        h = df_sorted["high"].values
        l = df_sorted["low"].values
        c = df_sorted["close"].values
        n = len(c)

        atr5 = _calc_atr(h, l, c, 5)
        atr40 = _calc_atr(h, l, c, 40)

        squeeze_abs = []
        normal_abs = []
        squeeze_brk = 0
        squeeze_n = 0

        for i in range(40, n - 20):
            if atr40[i] <= 0 or np.isnan(atr5[i]) or np.isnan(atr40[i]):
                continue
            ratio = atr5[i] / atr40[i]
            fwd_ret = (c[i + 20] - c[i]) / c[i]
            abs_fwd = abs(fwd_ret)

            if ratio < 0.7:
                squeeze_abs.append(abs_fwd)
                squeeze_n += 1
                if abs_fwd > 0.005:
                    squeeze_brk += 1
            else:
                normal_abs.append(abs_fwd)

        avg_sq = np.mean(squeeze_abs) * 100 if squeeze_abs else 0
        avg_nm = np.mean(normal_abs) * 100 if normal_abs else 0
        brk_prob = squeeze_brk / squeeze_n * 100 if squeeze_n > 0 else 0

        tee(f"  {sym:>4}  {squeeze_n:>8}  {avg_sq:>14.3f}%  {avg_nm:>12.3f}%  {brk_prob:>14.1f}%")

    tee("")


# ═══════════════════════════════════════════════════════════════
#  研究5：IF vs IH vs IM 品种特征对比
# ═══════════════════════════════════════════════════════════════

def research_5(daily: dict[str, pd.DataFrame], min5: dict[str, pd.DataFrame]):
    tee("\n" + "═" * 60)
    tee("  研究5：IF vs IH vs IM 品种特征对比")
    tee("═" * 60)

    # ── 5a: 日均波动率 ──
    tee("\n── 5a: 日均波动率 (ATR/Close) ──\n")
    tee(f"  {'品种':>4}  {'日均ATR/Close':>14}  {'年化波动率':>10}")
    tee("  " + "─" * 34)
    for sym, df in daily.items():
        h, l, c = df["high"].values, df["low"].values, df["close"].values
        atr = _calc_atr(h, l, c, 20)
        valid = ~np.isnan(atr) & (c > 0)
        if valid.any():
            avg_atr_pct = np.mean(atr[valid] / c[valid]) * 100
            daily_ret = np.diff(c) / c[:-1]
            ann_vol = np.std(daily_ret, ddof=1) * np.sqrt(242) * 100
        else:
            avg_atr_pct = 0
            ann_vol = 0
        tee(f"  {sym:>4}  {avg_atr_pct:>12.3f}%  {ann_vol:>9.1f}%")

    # ── 5b: 日均成交量 ──
    tee("\n── 5b: 日均成交量 ──\n")
    tee(f"  {'品种':>4}  {'日均成交量(手)':>14}  {'近1年均量':>12}")
    tee("  " + "─" * 36)
    for sym, df in daily.items():
        vol = df["volume"].values
        avg_all = np.mean(vol)
        recent = vol[-242:] if len(vol) >= 242 else vol
        avg_recent = np.mean(recent)
        tee(f"  {sym:>4}  {avg_all:>14,.0f}  {avg_recent:>12,.0f}")

    # ── 5c: 趋势延续性（自相关系数）──
    tee("\n── 5c: 日收益率自相关系数 ──\n")
    lags = list(range(1, 11))
    header = f"  {'品种':>4}" + "".join(f"  {'lag'+str(l):>7}" for l in lags)
    tee(header)
    tee("  " + "─" * (4 + 9 * len(lags)))
    for sym, df in daily.items():
        c = df["close"].values
        rets = pd.Series(np.diff(c) / c[:-1])
        row = f"  {sym:>4}"
        for lag in lags:
            ac = rets.autocorr(lag=lag)
            row += f"  {ac:>+7.3f}"
        tee(row)

    tee("\n  解读：正值=动量延续，负值=均值回归")

    # ── 5d: 跳空频率和幅度 ──
    tee("\n── 5d: 跳空统计 ──\n")
    tee(f"  {'品种':>4}  {'跳空>0.5%频率':>14}  {'平均跳空幅度':>12}  {'最大跳空':>10}")
    tee("  " + "─" * 46)
    for sym, df in daily.items():
        opens = df["open"].values
        prev_closes = df["close"].values
        gaps = []
        for i in range(1, len(opens)):
            if prev_closes[i - 1] > 0:
                gap = (opens[i] - prev_closes[i - 1]) / prev_closes[i - 1]
                gaps.append(gap)
        gaps = np.array(gaps)
        if len(gaps) > 0:
            freq = np.mean(np.abs(gaps) > 0.005) * 100
            avg_gap = np.mean(np.abs(gaps)) * 100
            max_gap = np.max(np.abs(gaps)) * 100
        else:
            freq = avg_gap = max_gap = 0
        tee(f"  {sym:>4}  {freq:>12.1f}%  {avg_gap:>10.3f}%  {max_gap:>8.2f}%")

    # ── 5e: 尾盘反转频率 ──
    tee("\n── 5e: 尾盘反转频率 (14:00-15:00 vs 全天) ──\n")
    tee(f"  {'品种':>4}  {'不一致率':>10}  {'样本数':>6}")
    tee("  " + "─" * 26)
    # Note: datetime in DB is UTC (Beijing - 8h), so 14:00 BJ = 06:00 UTC
    for sym, df_min in min5.items():
        dates = sorted(df_min["date"].unique())
        inconsistent = 0
        total = 0
        for d in dates:
            day_df = df_min[df_min["date"] == d].sort_values("datetime")
            if len(day_df) < 10:
                continue
            open_price = float(day_df.iloc[0]["open"])
            day_close = float(day_df.iloc[-1]["close"])
            full_dir = day_close - open_price

            # 14:00-15:00 BJ = 06:00-07:00 UTC
            tail = day_df[day_df["datetime"].dt.hour >= 6]
            if len(tail) < 2:
                continue

            tail_open = float(tail.iloc[0]["open"])
            tail_close = float(tail.iloc[-1]["close"])
            tail_dir = tail_close - tail_open

            if full_dir == 0 or tail_dir == 0:
                continue

            total += 1
            if (full_dir > 0 and tail_dir < 0) or (full_dir < 0 and tail_dir > 0):
                inconsistent += 1

        pct = inconsistent / total * 100 if total > 0 else 0
        tee(f"  {sym:>4}  {pct:>9.1f}%  {total:>6}")

    # ── 综合判断 ──
    tee("\n── 品种适用策略判断 ──\n")
    # Compute summary scores
    for sym, df in daily.items():
        c = df["close"].values
        rets = pd.Series(np.diff(c) / c[:-1])
        ac1 = rets.autocorr(lag=1)
        ac5 = rets.autocorr(lag=5)
        vol = np.std(rets, ddof=1) * np.sqrt(242) * 100

        if ac1 > 0.02 and ac5 > 0:
            trend_label = "趋势跟踪 ✓"
        elif ac1 < -0.02 and ac5 < 0:
            trend_label = "均值回归 ✓"
        else:
            trend_label = "混合型"

        tee(f"  {sym}: AC(1)={ac1:+.3f}, AC(5)={ac5:+.3f}, 年化波动={vol:.1f}% → {trend_label}")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    tee("=" * 60)
    tee("  A股股指期货动量特征实证研究")
    tee(f"  日期：{TODAY}")
    tee("=" * 60)

    db_path = ConfigLoader().get_db_path()
    db = DBManager(db_path)

    tee("\n加载数据...")
    daily = load_daily(db)
    min5 = load_min5(db)

    research_1(daily)
    research_2(min5)
    research_3(min5)
    research_4_daily(daily)
    research_4_intraday(min5)
    research_5(daily, min5)

    tee("\n" + "=" * 60)
    tee("  研究完成")
    tee("=" * 60)

    # Save markdown
    out_dir = Path(__file__).resolve().parents[1] / "logs" / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"momentum_research_{TODAY}.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# A股股指期货动量特征实证研究 ({TODAY})\n\n")
        f.write("```\n")
        f.write(OUTPUT_BUF.getvalue())
        f.write("```\n")
    print(f"\n结果已保存至: {out_path}")


if __name__ == "__main__":
    main()
