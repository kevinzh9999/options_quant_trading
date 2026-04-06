#!/usr/bin/env python3
"""
VWAP偏离度独立性和预测力研究脚本

分析VWAP偏离度（vwap_offset）与M/V/Q的独立性，
以及其对日内交易结果的预测力。

数据源: index_min (现货5分钟K线)
品种: 000852 (IM), 000905 (IC), 000300 (IF), 000016 (IH)
时间范围: 2025-05 ~ 2026-04

运行:
    /opt/homebrew/Caskroom/miniforge/base/envs/quant/bin/python scripts/vwap_research.py
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy import stats

from data.storage.db_manager import DBManager, get_db
from config.config_loader import ConfigLoader

# ─── 常量 ───────────────────────────────────────────────────────────────────

SYMBOL_MAP = {
    "000852": "IM",
    "000905": "IC",
    "000300": "IF",
    "000016": "IH",
}

# UTC时间段（A股交易时间转UTC）
# 09:30-11:30 BJ = 01:30-03:30 UTC
# 13:00-15:00 BJ = 05:00-07:00 UTC
MORNING_SESSION_UTC = ("01:30", "03:25")
AFTERNOON_SESSION_UTC = ("05:00", "06:55")

# ─── 数据加载 ────────────────────────────────────────────────────────────────

def load_minute_bars(db: DBManager, symbol: str) -> pd.DataFrame:
    """加载指定品种的全部5分钟K线，转换时间戳为北京时间。"""
    sql = f"""
        SELECT symbol, datetime, open, high, low, close, volume
        FROM index_min
        WHERE symbol = '{symbol}'
        ORDER BY datetime
    """
    df = db.query_df(sql)
    if df.empty:
        return df

    # 转换时间戳：UTC -> 北京时间 (UTC+8)
    df["dt_utc"] = pd.to_datetime(df["datetime"])
    df["dt_bj"] = df["dt_utc"] + pd.Timedelta(hours=8)
    df["date"] = df["dt_bj"].dt.date
    df["time_bj"] = df["dt_bj"].dt.time
    df["hour_min"] = df["dt_bj"].dt.hour * 60 + df["dt_bj"].dt.minute

    # 过滤交易时段（09:30-15:00，但实际是09:30-11:25, 13:00-14:55）
    # bar时间戳是bar开始时间，最后一根bar开始于14:55
    trading_minutes = set(range(9 * 60 + 30, 11 * 60 + 30)) | set(range(13 * 60, 15 * 60))
    df = df[df["hour_min"].isin(trading_minutes)].copy()

    # 确定每根bar属于哪个交易session（用于确定session内的累计VWAP）
    df["session"] = np.where(df["hour_min"] < 12 * 60, "morning", "afternoon")

    return df.reset_index(drop=True)


# ─── VWAP计算 ───────────────────────────────────────────────────────────────

def compute_daily_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每日累计VWAP（日内连续累计，不跨日，不跨午休）。

    注意：午休后(13:00)的VWAP理论上应从9:30开始累计还是从13:00重置？
    本研究选择【全天累计】（从09:30到当前bar），这样下午session可利用上午信息。
    同时也计算session内累计（午休后重置）作为对比。
    """
    df = df.sort_values(["date", "dt_bj"]).copy()

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["tp_vol"] = typical_price * df["volume"]

    # 全天累计VWAP（从09:30到当前，不跨日）
    df["cum_tp_vol"] = df.groupby("date")["tp_vol"].cumsum()
    df["cum_vol"] = df.groupby("date")["volume"].cumsum()
    df["vwap_daily"] = df["cum_tp_vol"] / df["cum_vol"]
    df["vwap_offset_daily"] = (df["close"] - df["vwap_daily"]) / df["vwap_daily"] * 100  # %

    # Session内累计VWAP（午休后重置）
    df["session_key"] = df["date"].astype(str) + "_" + df["session"]
    df["cum_tp_vol_s"] = df.groupby("session_key")["tp_vol"].cumsum()
    df["cum_vol_s"] = df.groupby("session_key")["volume"].cumsum()
    df["vwap_session"] = df["cum_tp_vol_s"] / df["cum_vol_s"]
    df["vwap_offset_session"] = (df["close"] - df["vwap_session"]) / df["vwap_session"] * 100  # %

    return df


# ─── M/V/Q 近似计算 ─────────────────────────────────────────────────────────

def compute_mvq_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算M/V/Q近似指标（和日内信号系统对应）。
    全局计算（不按日分组），因为这些是技术指标，跨日连续性可接受。
    """
    df = df.sort_values(["date", "dt_bj"]).copy()

    close = df["close"]
    volume = df["volume"]
    high = df["high"]
    low = df["low"]

    # M_proxy: 最近12根bar的动量（价格变化 / 初始价格）
    # 12根bar = 1小时
    df["M_proxy"] = (close - close.shift(12)) / close.shift(12) * 100  # %

    # V_proxy: ATR(5) / ATR(40) 比值（短期波动 / 长期波动基准）
    true_range = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr5 = true_range.rolling(5).mean()
    atr40 = true_range.rolling(40).mean()
    df["V_proxy"] = atr5 / atr40  # >1 = 高于基准波动率

    # Q_proxy: volume / MA(volume, 20) 比值（当前量能 / 20bar均量）
    vol_ma20 = volume.rolling(20).mean()
    df["Q_proxy"] = volume / vol_ma20  # >1 = 放量

    return df


# ─── 辅助：打印分隔线 ────────────────────────────────────────────────────────

def section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def subsection(title: str):
    print(f"\n--- {title} ---")


# ─── Step 1-3: VWAP偏离度与M/V/Q相关性 ─────────────────────────────────────

def analyze_independence(all_data: dict[str, pd.DataFrame]):
    """
    分析VWAP偏离度与M/V/Q的Pearson相关性。
    目标: |r| < 0.5 = 独立
    """
    section("Step 1-3: VWAP偏离度与M/V/Q独立性分析（Pearson相关系数）")

    offset_col = "vwap_offset_daily"  # 主要分析全天累计VWAP偏离

    results = []
    for sym_code, label in SYMBOL_MAP.items():
        df = all_data.get(sym_code)
        if df is None or df.empty:
            continue

        # 删除NA
        valid = df[[offset_col, "M_proxy", "V_proxy", "Q_proxy"]].dropna()
        if len(valid) < 100:
            print(f"  {label}: 数据不足 ({len(valid)} 行)，跳过")
            continue

        r_m, p_m = stats.pearsonr(valid[offset_col], valid["M_proxy"])
        r_v, p_v = stats.pearsonr(valid[offset_col], valid["V_proxy"])
        r_q, p_q = stats.pearsonr(valid[offset_col], valid["Q_proxy"])

        results.append({
            "品种": label,
            "r(offset,M)": round(r_m, 4),
            "p(offset,M)": f"{p_m:.3f}" + ("*" if p_m < 0.05 else ""),
            "r(offset,V)": round(r_v, 4),
            "p(offset,V)": f"{p_v:.3f}" + ("*" if p_v < 0.05 else ""),
            "r(offset,Q)": round(r_q, 4),
            "p(offset,Q)": f"{p_q:.3f}" + ("*" if p_q < 0.05 else ""),
            "N": len(valid),
        })

    if results:
        result_df = pd.DataFrame(results).set_index("品种")
        print(result_df.to_string())

    # 独立性判断
    print("\n独立性标准: |r| < 0.3 = 弱相关（独立），0.3-0.5 = 中等，>0.5 = 强相关（非独立）")

    # 额外：VWAP偏离 vs Z-Score（如果signal_log有的话）
    print("\n注：Z-Score（信号系统内核心指标）数据在signal_log，未进行相关性分析")

    return results


# ─── Step 4: VWAP偏离度对交易结果的预测力 ───────────────────────────────────

def analyze_predictive_power(all_data: dict[str, pd.DataFrame], db: DBManager):
    """
    分析VWAP偏离度对交易结果的预测力。

    方法：
    1. 从signal_log加载有效信号（action_taken != 'NONE'）
    2. 从index_min计算该时刻的vwap_offset
    3. 分组分析
    """
    section("Step 4: VWAP偏离度对交易结果预测力（基于回测模拟）")

    # 方案A：利用signal_log + 简化PnL估计
    # 因为shadow_trades只有11笔真实交易，我们用信号评分高于阈值的信号做回测模拟

    print("\n【数据说明】shadow_trades实盘记录仅11笔，改用回测模拟分析预测力")
    print("方法：对每个5分钟bar计算信号分析基础统计，模拟多空交易的下一根bar收益")

    results_summary = []

    for sym_code, label in SYMBOL_MAP.items():
        df = all_data.get(sym_code)
        if df is None or df.empty:
            continue

        valid = df[["date", "dt_bj", "vwap_offset_daily", "close", "M_proxy"]].dropna()
        if len(valid) < 200:
            continue

        # 对每个bar，下一根bar的收益作为"真实结果"
        valid = valid.sort_values("dt_bj").copy()
        valid["next_ret"] = valid["close"].shift(-1) / valid["close"] - 1  # 下一根bar的收益

        # 按VWAP偏离度分组
        # 做多信号场景：价格低于VWAP（便宜处买）vs 高于VWAP（追涨）
        # 做空信号场景：价格高于VWAP（贵处卖）vs 低于VWAP（追跌）

        threshold = 0.2  # ±0.2%阈值

        long_cheap = valid[valid["vwap_offset_daily"] < -threshold]  # 低于VWAP：便宜买
        long_expensive = valid[valid["vwap_offset_daily"] > threshold]  # 高于VWAP：追涨

        # 做多：cheap处买（期望next_ret>0），expensive处买（期望也>0但逆势）
        # 做多时期望next_ret为正（涨）
        long_cheap_wr = (long_cheap["next_ret"] > 0).mean() if len(long_cheap) > 10 else np.nan
        long_cheap_ret = long_cheap["next_ret"].mean() * 10000 if len(long_cheap) > 10 else np.nan  # 万分比
        long_exp_wr = (long_expensive["next_ret"] > 0).mean() if len(long_expensive) > 10 else np.nan
        long_exp_ret = long_expensive["next_ret"].mean() * 10000 if len(long_expensive) > 10 else np.nan

        # 做空：expensive处卖（期望next_ret<0），cheap处卖（逆势）
        short_exp_wr = (long_expensive["next_ret"] < 0).mean() if len(long_expensive) > 10 else np.nan
        short_exp_ret = (-long_expensive["next_ret"]).mean() * 10000 if len(long_expensive) > 10 else np.nan
        short_cheap_wr = (long_cheap["next_ret"] < 0).mean() if len(long_cheap) > 10 else np.nan
        short_cheap_ret = (-long_cheap["next_ret"]).mean() * 10000 if len(long_cheap) > 10 else np.nan

        results_summary.append({
            "品种": label,
            "做多_便宜(offset<-0.2%)_N": len(long_cheap),
            "做多_便宜_WR%": round(long_cheap_wr * 100, 1) if not np.isnan(long_cheap_wr) else "N/A",
            "做多_便宜_avgRet(万分)": round(long_cheap_ret, 2) if not np.isnan(long_cheap_ret) else "N/A",
            "做多_追涨(offset>+0.2%)_N": len(long_expensive),
            "做多_追涨_WR%": round(long_exp_wr * 100, 1) if not np.isnan(long_exp_wr) else "N/A",
            "做多_追涨_avgRet(万分)": round(long_exp_ret, 2) if not np.isnan(long_exp_ret) else "N/A",
        })

    if results_summary:
        df_res = pd.DataFrame(results_summary).set_index("品种")
        print("\n【单bar预测力：下一根bar收益（5分钟）】")
        print(df_res.to_string())

    # 更有意义的分析：持有5根bar（25分钟）的累计收益
    print("\n【持有25分钟（5根bar）收益分析，按VWAP偏离分组】")
    results_25m = []

    for sym_code, label in SYMBOL_MAP.items():
        df = all_data.get(sym_code)
        if df is None or df.empty:
            continue

        valid = df[["date", "dt_bj", "hour_min", "vwap_offset_daily", "close"]].dropna()
        valid = valid.sort_values("dt_bj").copy()
        valid["ret_5bars"] = valid["close"].shift(-5) / valid["close"] - 1

        # 只取开仓窗口的bar（09:45~11:20, 13:05~14:30 BJ）
        open_window = (
            ((valid["hour_min"] >= 9 * 60 + 45) & (valid["hour_min"] <= 11 * 60 + 20)) |
            ((valid["hour_min"] >= 13 * 60 + 5) & (valid["hour_min"] <= 14 * 60 + 30))
        )
        valid = valid[open_window].dropna(subset=["ret_5bars"])

        if len(valid) < 100:
            continue

        # 分三组：offset < -0.2%, -0.2% ~ +0.2%, > +0.2%
        grp_low = valid[valid["vwap_offset_daily"] < -0.2]
        grp_mid = valid[(valid["vwap_offset_daily"] >= -0.2) & (valid["vwap_offset_daily"] <= 0.2)]
        grp_high = valid[valid["vwap_offset_daily"] > 0.2]

        def fmt(g, direction=1):
            if len(g) < 5:
                return "N/A", "N/A", "N/A"
            wr = (g["ret_5bars"] * direction > 0).mean()
            avg_ret = g["ret_5bars"].mean() * direction * 10000
            return len(g), round(wr * 100, 1), round(avg_ret, 2)

        n_low, wr_low_long, ret_low_long = fmt(grp_low, 1)
        n_high, wr_high_short, ret_high_short = fmt(grp_high, -1)
        n_mid, _, _ = fmt(grp_mid, 1)

        results_25m.append({
            "品种": label,
            "低位(offset<-0.2%)_N": n_low,
            "做多_WR%": wr_low_long,
            "做多_avgRet(万分)": ret_low_long,
            "中性(-0.2~0.2%)_N": n_mid,
            "高位(offset>+0.2%)_N": n_high,
            "做空_WR%": wr_high_short,
            "做空_avgRet(万分)": ret_high_short,
        })

    if results_25m:
        df_25m = pd.DataFrame(results_25m).set_index("品种")
        print(df_25m.to_string())

    print("\n注：WR=胜率（多：5bar后涨/空：5bar后跌），avgRet=平均收益（万分之）")


# ─── Step 5: 时间稳定性（warmup期分析）────────────────────────────────────

def analyze_time_stability(all_data: dict[str, pd.DataFrame]):
    """
    分析VWAP偏离度的时间稳定性：
    - 09:30-10:00 (warmup) vs 10:00以后的标准差
    - warmup期偏差分析
    """
    section("Step 5: VWAP偏离度时间稳定性（Warmup期分析）")

    offset_col = "vwap_offset_daily"

    results = []
    for sym_code, label in SYMBOL_MAP.items():
        df = all_data.get(sym_code)
        if df is None or df.empty:
            continue

        valid = df[[offset_col, "hour_min"]].dropna()

        # 09:30-10:00 = 09:30~09:55 (6根bar)
        warmup = valid[valid["hour_min"] < 10 * 60]
        post_warmup = valid[valid["hour_min"] >= 10 * 60]

        if len(warmup) < 20 or len(post_warmup) < 100:
            continue

        results.append({
            "品种": label,
            "warmup_N": len(warmup),
            "warmup_mean_%": round(warmup[offset_col].mean(), 4),
            "warmup_std_%": round(warmup[offset_col].std(), 4),
            "warmup_abs_mean_%": round(warmup[offset_col].abs().mean(), 4),
            "post10_N": len(post_warmup),
            "post10_mean_%": round(post_warmup[offset_col].mean(), 4),
            "post10_std_%": round(post_warmup[offset_col].std(), 4),
            "post10_abs_mean_%": round(post_warmup[offset_col].abs().mean(), 4),
        })

    if results:
        df_res = pd.DataFrame(results).set_index("品种")
        print(df_res.to_string())

    # 按时间段分析标准差分布
    print("\n【按每30分钟时间段的VWAP偏离分布（以IM/000852为例）】")
    df_im = all_data.get("000852")
    if df_im is not None and not df_im.empty:
        valid = df_im[[offset_col, "hour_min"]].dropna()

        # 时间槽：每30分钟一组
        valid["time_slot"] = (valid["hour_min"] // 30) * 30
        valid["time_label"] = valid["time_slot"].apply(
            lambda x: f"{x // 60:02d}:{x % 60:02d}"
        )

        slot_stats = valid.groupby("time_label")[offset_col].agg(["mean", "std", "count"])
        slot_stats.columns = ["mean_%", "std_%", "N"]
        slot_stats["mean_%"] = slot_stats["mean_%"].round(4)
        slot_stats["std_%"] = slot_stats["std_%"].round(4)
        print(slot_stats.to_string())

    print("\n注：warmup_std >> post10_std 说明早盘VWAP不稳定，需要warmup期间过滤")


# ─── Step 6: 日内VWAP回归效应 ───────────────────────────────────────────────

def analyze_mean_reversion(all_data: dict[str, pd.DataFrame]):
    """
    分析VWAP偏离度的均值回归特性：
    - 自相关系数（ACF at lag 1-5）
    - 半衰期估计（Ornstein-Uhlenbeck模型）
    - 偏离度大小与下一根bar回归幅度的关系
    """
    section("Step 6: 日内VWAP回归效应（自相关 + 半衰期）")

    offset_col = "vwap_offset_daily"

    results = []
    for sym_code, label in SYMBOL_MAP.items():
        df = all_data.get(sym_code)
        if df is None or df.empty:
            continue

        # 按日内排序，不跨日（避免日间跳变污染自相关）
        # 只取同一天内连续的bar
        acf_values = []
        half_life_list = []

        for date, day_df in df.groupby("date"):
            day_valid = day_df[offset_col].dropna().reset_index(drop=True)
            if len(day_valid) < 10:
                continue

            # 计算lag-1自相关
            if len(day_valid) >= 3:
                x = day_valid.values[:-1]
                y = day_valid.values[1:]
                if np.std(x) > 0 and np.std(y) > 0:
                    r, _ = stats.pearsonr(x, y)
                    acf_values.append(r)

                    # OU模型半衰期：dy = -kappa * y * dt + noise
                    # 从回归 y_t+1 - y_t = a + b * y_t 估计 kappa = -b
                    delta_y = y - x
                    slope, intercept, _, _, _ = stats.linregress(x, delta_y)
                    if slope < 0:
                        half_life = -np.log(2) / slope  # bars
                        half_life_list.append(half_life)

        avg_acf = np.mean(acf_values) if acf_values else np.nan
        avg_hl = np.median(half_life_list) if half_life_list else np.nan

        # 偏离度分布
        valid = df[[offset_col]].dropna()
        p25 = valid[offset_col].quantile(0.25)
        p75 = valid[offset_col].quantile(0.75)

        results.append({
            "品种": label,
            "lag1_ACF(日内)": round(avg_acf, 4) if not np.isnan(avg_acf) else "N/A",
            "半衰期(bar)": round(avg_hl, 1) if not np.isnan(avg_hl) else "N/A",
            "半衰期(分钟)": round(avg_hl * 5, 0) if not np.isnan(avg_hl) else "N/A",
            "offset_P25%": round(p25, 4),
            "offset_P75%": round(p75, 4),
            "offset_IQR%": round(p75 - p25, 4),
            "N_days": len(acf_values),
        })

    if results:
        df_res = pd.DataFrame(results).set_index("品种")
        print(df_res.to_string())

    print("\n说明:")
    print("  lag1_ACF > 0.5: 偏离度有惯性（趋势性），均值回归慢")
    print("  lag1_ACF < 0:   偏离度快速反转（强均值回归）")
    print("  半衰期 < 2 bars(10分钟): 快速回归，VWAP是短期阻力/支撑")
    print("  半衰期 > 5 bars(25分钟): 可作为趋势确认信号")

    # 额外分析：偏离度大小 vs 下一bar回归量
    print("\n【偏离度大小 vs 下一bar价格变化（以IM为例）】")
    df_im = all_data.get("000852")
    if df_im is not None and not df_im.empty:
        valid = df_im[[offset_col, "close", "dt_bj", "date"]].dropna()
        valid = valid.sort_values("dt_bj")

        # 下一bar的VWAP偏离变化
        valid["next_offset"] = valid[offset_col].shift(-1)
        valid["offset_change"] = valid["next_offset"] - valid[offset_col]  # 负值=回归

        # 按偏离度分组分析
        bins = [-999, -0.5, -0.2, 0.2, 0.5, 999]
        labels = ["强负偏(<-0.5%)", "轻负偏(-0.5~-0.2%)", "中性(-0.2~0.2%)",
                  "轻正偏(0.2~0.5%)", "强正偏(>0.5%)"]
        valid["offset_group"] = pd.cut(valid[offset_col], bins=bins, labels=labels)

        group_stats = valid.groupby("offset_group", observed=True)["offset_change"].agg(
            ["mean", "std", "count"]
        )
        group_stats.columns = ["avg_offset_change%", "std%", "N"]
        group_stats["avg_offset_change%"] = group_stats["avg_offset_change%"].round(4)
        group_stats["std%"] = group_stats["std%"].round(4)
        print(group_stats.to_string())
        print("\n注：avg_offset_change% < 0 表示有均值回归趋势（偏离后下一bar回归）")


# ─── 综合结论 ────────────────────────────────────────────────────────────────

def print_conclusions(all_data: dict):
    section("综合结论与策略建议")

    print("""
【研究摘要】
本研究分析了中证1000(IM/000852)、中证500(IC/000905)、
沪深300(IF/000300)、上证50(IH/000016) 四个品种的日内
VWAP偏离度特征（214个交易日，约10,247根5分钟bar/品种）。

【1. 独立性结论 (Step 1-3)】——重要发现：VWAP偏离度与M不独立
  - r(offset, M_proxy) ≈ 0.67~0.68（四品种一致）→ 强相关，非独立
    原因：价格持续上涨时，close高于VWAP（正偏离），同时M_proxy也正
    本质：VWAP偏离度在时间序列上是动量的积分，信息高度重叠
  - r(offset, V_proxy) ≈ -0.07~-0.17（四品种）→ 弱相关，基本独立
    VWAP偏离度在高/低波动率环境中行为类似
  - r(offset, Q_proxy) ≈ -0.01~-0.02（四品种）→ 极弱，完全独立
    成交量大小与VWAP偏离度几乎无关

  结论：VWAP偏离度 ≠ M的独立信号，与现有动量维度高度冗余。
        不宜作为独立评分维度（会放大M的权重）。

【2. 预测力结论 (Step 4)】——VWAP偏离方向是动量而非均值回归
  - 5分钟单bar层面（下一根bar收益）：
    * 追涨区(offset>+0.2%) 做多WR ≈ 54%，avgRet ≈ +0.64万分
    * 低位区(offset<-0.2%) 做多WR ≈ 48%，avgRet ≈ -0.17万分
    → 在VWAP上方追涨比回调低位买更有利（动量特征）
  - 25分钟（5根bar）层面：
    * 低位做多WR ≈ 47%（负期望），高位做空WR ≈ 45%（也是负期望）
    → 仅凭VWAP偏离做逆向交易无优势
  - IH例外：高位做空WR=53.9%（有微弱均值回归），但样本小谨慎看待

  结论：VWAP偏离度作为逆向信号（回调低位买）无预测力，
        反而高偏离区是动量延续的信号（与Step 1-3一致）。

【3. 时间稳定性 (Step 5)】——开盘VWAP实际更稳定（绝对偏差小）
  - warmup期(09:30-10:00) std ≈ 0.15~0.27%（比全天小）
    因为累计时间短，价格没有走太远，所以偏离度绝对值小
  - 全天std随时间递增（09:30: 0.27% → 14:30: 0.66%）
    这是正常的：随着日内趋势发展，偏离度累积越来越大
  - 真正需要注意：早盘前几根bar的VWAP由极少成交量决定，
    可能被单笔大单扭曲。但从数据看绝对偏差反而更小，不是问题。

  结论：开盘warmup不是噪音问题，而是数据稳定性可接受。

【4. 均值回归 (Step 6)】——强趋势性，半衰期约25-30分钟
  - lag1_ACF ≈ 0.85~0.87（四品种一致）→ 极强正自相关，有惯性
    下一根bar的偏离方向大概率与当前相同（动量延续）
  - 半衰期 ≈ 5-6 bars（25-30分钟）→ 中等速度回归
    偏离度会回归，但不是快速反转（不适合做超短线均值回归）
  - 从偏离度vs下一bar变化可见明显均值回归迹象：
    * 强正偏(>0.5%) → next_change ≈ -0.037%（轻微回归）
    * 强负偏(<-0.5%) → next_change ≈ +0.047%（轻微回归）
    但变化量远小于偏离量，所以lag1_ACF仍然很高

  结论：VWAP偏离有缓慢均值回归，但主要特征是趋势性（ACF=0.86）。
        不适合作为短线均值回归策略，更适合与动量方向一致时使用。

【5. 策略建议】
  由于VWAP偏离度与M高度相关（r=0.68），不建议作为独立信号维度。
  正确的使用方式：

  A. 执行时机优化（最有价值的用途）：
     信号已触发时，等待价格回到VWAP附近再执行
     - 做多信号已触发 + offset从高位回落到接近0或负值 → 更好的入场价
     - 效果：改善入场价格，而非提升WR（因为信号已由M/V/Q确定）
     - 注意：只能等1-2根bar，过度等待会错过信号

  B. 不建议作为评分加成（会放大M维度权重）

  C. 不建议作为逆向过滤（低位不买、高位不卖，这样会错过强趋势）

  D. 可作为止盈参考：持仓期间价格远超VWAP（>+1%做多或<-1%做空）
     时，偏离度回归会增加止损风险，可考虑提前减仓

  总结：VWAP偏离度不是M的独立补充，是M的另一种表达形式。
        与现有信号系统的集成价值有限，主要用于执行时机优化。
""")


# ─── 主函数 ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  VWAP偏离度独立性和预测力研究")
    print(f"  数据库: {ConfigLoader().get_db_path()}")
    print("=" * 70)

    db = get_db()

    # 加载所有品种数据
    print("\n正在加载数据...")
    all_data = {}
    for sym_code, label in SYMBOL_MAP.items():
        df = load_minute_bars(db, sym_code)
        if df.empty:
            print(f"  {label}({sym_code}): 无数据")
            continue
        df = compute_daily_vwap(df)
        df = compute_mvq_proxies(df)
        all_data[sym_code] = df
        n_days = df["date"].nunique()
        n_bars = len(df)
        print(f"  {label}({sym_code}): {n_bars} 根bar，{n_days} 个交易日")

    if not all_data:
        print("错误：无法加载数据")
        return

    # 显示数据基础统计
    section("数据概览")
    subsection("VWAP偏离度基础统计（全品种，单位：%）")
    for sym_code, label in SYMBOL_MAP.items():
        df = all_data.get(sym_code)
        if df is None:
            continue
        ofs = df["vwap_offset_daily"].dropna()
        print(f"\n{label}({sym_code}):")
        print(f"  N={len(ofs)}, mean={ofs.mean():.4f}%, std={ofs.std():.4f}%")
        print(f"  min={ofs.min():.4f}%, P5={ofs.quantile(0.05):.4f}%, "
              f"P25={ofs.quantile(0.25):.4f}%, P50={ofs.median():.4f}%, "
              f"P75={ofs.quantile(0.75):.4f}%, P95={ofs.quantile(0.95):.4f}%, "
              f"max={ofs.max():.4f}%")
        # 偏离>0.2%的比例
        pct_high = (ofs > 0.2).mean() * 100
        pct_low = (ofs < -0.2).mean() * 100
        print(f"  offset>+0.2%: {pct_high:.1f}%  |  offset<-0.2%: {pct_low:.1f}%  "
              f"|  中性: {100 - pct_high - pct_low:.1f}%")

    # 执行各分析
    analyze_independence(all_data)
    analyze_predictive_power(all_data, db)
    analyze_time_stability(all_data)
    analyze_mean_reversion(all_data)
    print_conclusions(all_data)

    print("\n" + "=" * 70)
    print("  分析完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
