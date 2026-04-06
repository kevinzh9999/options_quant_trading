#!/usr/bin/env python3
"""
cross_rank_research.py
----------------------
跨品种相对强弱rank作为信号维度的可行性分析。

分析步骤：
  Step 1: 计算跨品种rank（每根5分钟bar的bar-to-bar return排名）
  Step 2: rank与M/V/Q proxy的相关性分析
  Step 3: rank对交易结果的预测力（基于回测框架）
  Step 4: IM-IH剪刀差（style spread）分析
  Step 5: 模拟接入rank乘数效果（方案A、方案B）

Usage:
    python scripts/cross_rank_research.py
    python scripts/cross_rank_research.py --dates 20260324,20260325
    python scripts/cross_rank_research.py --no-backtest   # 只做相关性分析，跳过耗时回测
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.storage.db_manager import DBManager, get_db
from config.config_loader import ConfigLoader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYMBOLS = ["IM", "IC", "IF", "IH"]
SPOT_MAP = {
    "IM": "000852",
    "IC": "000905",
    "IF": "000300",
    "IH": "000016",
}
SPOT_IDX_MAP = {
    "IM": "000852.SH",
    "IC": "000905.SH",
    "IF": "000300.SH",
    "IH": "000016.SH",
}

# IM/IC回测日期（34天）
DEFAULT_DATES = (
    "20260204,20260205,20260206,20260209,20260210,20260211,20260212,20260213,"
    "20260225,20260226,20260227,20260302,20260303,20260304,20260305,20260306,"
    "20260309,20260310,20260311,20260312,20260313,20260316,20260317,20260318,"
    "20260319,20260320,20260323,20260324,20260325,20260326,20260327,20260328,"
    "20260401,20260402"
)

# 乘数方案A: 方向无感（只看rank绝对值）
RANK_MULT_A = {1: 1.10, 2: 1.00, 3: 1.00, 4: 0.85}

# 乘数方案B: 方向感知
#   做多时: rank1(最强)=1.1, rank4(最弱)=0.85
#   做空时: rank4(最弱=做空最强)=1.1, rank1(最强=做空矛盾)=0.85
def rank_mult_b(rank: int, direction: str) -> float:
    if direction == "LONG":
        return {1: 1.10, 2: 1.00, 3: 1.00, 4: 0.85}.get(rank, 1.0)
    elif direction == "SHORT":
        return {1: 0.85, 2: 1.00, 3: 1.00, 4: 1.10}.get(rank, 1.0)
    return 1.0


# ---------------------------------------------------------------------------
# Step 0: 加载全量index_min数据
# ---------------------------------------------------------------------------

def load_all_spot_bars(db: DBManager, dates: List[str]) -> pd.DataFrame:
    """加载4品种5分钟现货K线，仅含指定日期。"""
    date_strs = [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in dates]
    placeholders = ",".join(f"'{d}'" for d in date_strs)
    query = f"""
        SELECT symbol, datetime, open, high, low, close, volume
        FROM index_min
        WHERE period=300
          AND symbol IN ('000852','000905','000300','000016')
          AND substr(datetime,1,10) IN ({placeholders})
        ORDER BY symbol, datetime
    """
    df = db.query_df(query)
    if df is None or df.empty:
        raise ValueError("index_min 无数据，请检查日期范围或数据入库状态")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df


# ---------------------------------------------------------------------------
# Step 1: 计算跨品种rank
# ---------------------------------------------------------------------------

def compute_cross_rank(all_bars: pd.DataFrame) -> pd.DataFrame:
    """
    对每根5分钟bar计算4品种bar-to-bar return排名。

    返回DataFrame，index=datetime，列=["000852","000905","000300","000016"]
    值为1(最强)~4(最弱)的rank，以及各品种return。
    防lookahead: rank用completed bar的return（上一根bar close → 当前bar close）。
    """
    # Pivot: symbol × datetime → close
    pivot = all_bars.pivot_table(index="datetime", columns="symbol", values="close")
    pivot = pivot.sort_index()

    # bar-to-bar return（防lookahead：rank计算用上一根bar close，即shift(1)→当前）
    returns = pivot.pct_change()  # return[t] = (close[t] - close[t-1]) / close[t-1]

    # 同理volume pivot
    vol_pivot = all_bars.pivot_table(index="datetime", columns="symbol", values="volume")
    vol_pivot = vol_pivot.sort_index()

    # rank: 1=最强(return最大), 4=最弱
    rank = returns.rank(axis=1, ascending=False, method="min")

    # Combine
    result = pd.DataFrame(index=pivot.index)
    for sym_code in pivot.columns:
        result[f"return_{sym_code}"] = returns[sym_code]
        result[f"rank_{sym_code}"] = rank[sym_code]
        result[f"close_{sym_code}"] = pivot[sym_code]
        result[f"volume_{sym_code}"] = vol_pivot.get(sym_code, pd.Series(dtype=float))

    # IM-IH剪刀差（style spread）: 小盘 - 大盘
    result["style_spread"] = returns.get("000852", pd.Series()) - returns.get("000016", pd.Series())

    return result


# ---------------------------------------------------------------------------
# Step 2: rank与M/V/Q proxy的相关性
# ---------------------------------------------------------------------------

def compute_mvq_proxy(all_bars: pd.DataFrame) -> pd.DataFrame:
    """
    对每个品种、每根bar计算M/V/Q proxy。

    M_proxy: 近12根bar的close变化率（动量）
    V_proxy: ATR(5)/ATR(40)（波动率相对水平）
    Q_proxy: volume/MA(volume,20)（成交量相对水平）
    """
    results = []
    sym_map = {v: k for k, v in SPOT_MAP.items()}

    for sym_code, grp in all_bars.groupby("symbol"):
        grp = grp.sort_values("datetime").reset_index(drop=True)
        closes = grp["close"]
        highs = grp["high"]
        lows = grp["low"]
        vols = grp["volume"]

        # M_proxy: 12根bar变化率（约1小时动量）
        m_proxy = closes.pct_change(12)

        # ATR
        tr = pd.concat([
            highs - lows,
            (highs - closes.shift(1)).abs(),
            (lows - closes.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr5 = tr.rolling(5).mean()
        atr40 = tr.rolling(40).mean()
        v_proxy = atr5 / atr40.replace(0, np.nan)

        # Q_proxy
        vol_ma20 = vols.rolling(20).mean()
        q_proxy = vols / vol_ma20.replace(0, np.nan)

        tmp = grp[["datetime", "close"]].copy()
        tmp["symbol"] = sym_code
        tmp["fut_sym"] = sym_map.get(sym_code, sym_code)
        tmp["m_proxy"] = m_proxy
        tmp["v_proxy"] = v_proxy
        tmp["q_proxy"] = q_proxy
        results.append(tmp)

    return pd.concat(results, ignore_index=True)


def step2_correlation(rank_df: pd.DataFrame, mvq_df: pd.DataFrame) -> None:
    """Step 2: rank与M/V/Q proxy相关性分析。"""
    print("\n" + "=" * 70)
    print(" Step 2: rank与M/V/Q proxy的Pearson相关系数")
    print("=" * 70)

    # 按日期过滤，排除同一天第一根bar（return=NaN）
    sym_map = {v: k for k, v in SPOT_MAP.items()}

    header = f"  {'品种':<8} {'corr(rank,M)':<16} {'corr(rank,V)':<16} {'corr(rank,Q)':<16} {'N'}"
    print(header)
    print("  " + "-" * 65)

    for sym_code in ["000852", "000905", "000300", "000016"]:
        fut_sym = sym_map.get(sym_code, sym_code)
        rank_col = f"rank_{sym_code}"
        if rank_col not in rank_df.columns:
            continue

        # 合并rank和MVQ
        mvq_sub = mvq_df[mvq_df["symbol"] == sym_code][["datetime", "m_proxy", "v_proxy", "q_proxy"]]
        merged = rank_df[[rank_col]].merge(mvq_sub, left_index=True, right_on="datetime", how="inner")
        merged = merged.dropna()

        n = len(merged)
        if n < 20:
            print(f"  {fut_sym}({sym_code}): 数据不足 (N={n})")
            continue

        r_rank = merged[rank_col].values
        r_m = merged["m_proxy"].values
        r_v = merged["v_proxy"].values
        r_q = merged["q_proxy"].values

        # rank越小=越强，M越大=越强，预期corr为负
        c_m, p_m = pearsonr(r_rank, r_m)
        c_v, p_v = pearsonr(r_rank, r_v)
        c_q, p_q = pearsonr(r_rank, r_q)

        def _fmt(c, p):
            star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            return f"{c:+.3f}{star:<3}"

        print(f"  {fut_sym}({sym_code}): {_fmt(c_m,p_m):<15} {_fmt(c_v,p_v):<15} {_fmt(c_q,p_q):<15} N={n}")

    print("  (* p<0.05, ** p<0.01, *** p<0.001)")


# ---------------------------------------------------------------------------
# Step 3: rank对交易结果的预测力
# ---------------------------------------------------------------------------

def _utc_to_bj(utc_str: str) -> str:
    h = int(utc_str[:2]) + 8
    if h >= 24:
        h -= 24
    return f"{h:02d}:{utc_str[3:5]}"


def _bj_to_utc(bj_str: str) -> str:
    h = int(bj_str[:2]) - 8
    if h < 0:
        h += 24
    return f"{h:02d}:{bj_str[3:5]}"


def get_rank_at_time(rank_df: pd.DataFrame, date_str: str, bj_time: str, sym: str) -> Optional[int]:
    """
    查询指定日期+时间点该品种的rank。
    bj_time格式: "HH:MM"，已转换为UTC后在rank_df中查找。
    使用上一根完成bar的rank（和信号评分的lookahead-free逻辑一致）。
    """
    sym_code = SPOT_MAP.get(sym)
    if not sym_code:
        return None
    rank_col = f"rank_{sym_code}"
    if rank_col not in rank_df.columns:
        return None

    date_dash = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    # entry_time是BJ时间，转UTC
    utc_hm = _bj_to_utc(bj_time)
    utc_dt = f"{date_dash} {utc_hm}:00"

    # 找到entry_time对应的bar，取上一根（完成bar）
    day_rank = rank_df[rank_df.index.str.startswith(date_dash)]
    if day_rank.empty:
        return None

    # 找entry bar的前一根
    idx_list = day_rank.index.tolist()
    try:
        pos = next(i for i, x in enumerate(idx_list) if x >= utc_dt)
        if pos == 0:
            return None
        prev_idx = idx_list[pos - 1]
        val = day_rank.loc[prev_idx, rank_col]
        if pd.isna(val):
            return None
        return int(val)
    except StopIteration:
        # entry_time在最后一根bar之后，取最后一根
        val = day_rank.iloc[-1][rank_col]
        return int(val) if not pd.isna(val) else None


def step3_rank_predictability(
    all_trades_by_sym: Dict[str, List],
    rank_df: pd.DataFrame,
    dates: List[str],
) -> None:
    """Step 3: rank对交易结果的预测力分析。"""
    print("\n" + "=" * 70)
    print(" Step 3: rank对交易结果的预测力")
    print("=" * 70)

    for sym in ["IM", "IC"]:
        trades_all = all_trades_by_sym.get(sym, [])
        if not trades_all:
            print(f"\n  {sym}: 无交易记录")
            continue

        # 给每笔trade附加rank
        enriched = []
        for td, trade in trades_all:
            if trade.get("partial"):
                continue
            entry_time = trade.get("entry_time", "")
            rank_val = get_rank_at_time(rank_df, td, entry_time, sym)
            t = dict(trade)
            t["rank"] = rank_val
            t["date"] = td
            enriched.append(t)

        df = pd.DataFrame(enriched)
        if df.empty or "rank" not in df.columns:
            continue

        df_valid = df.dropna(subset=["rank"])
        if df_valid.empty:
            continue

        print(f"\n  {sym} | 总交易 {len(df)} 笔, 成功匹配rank {len(df_valid)} 笔")

        # 按rank分组
        print(f"\n  {'Rank':<6} {'N':>5} {'WR%':>7} {'AvgPnL':>9} {'TotPnL':>9} {'说明'}")
        print("  " + "-" * 55)
        for r in [1, 2, 3, 4]:
            sub = df_valid[df_valid["rank"] == r]
            n = len(sub)
            if n == 0:
                print(f"  {r:<6} {'0':>5}")
                continue
            wr = (sub["pnl_pts"] > 0).mean() * 100
            avg = sub["pnl_pts"].mean()
            tot = sub["pnl_pts"].sum()
            desc = {1: "最强", 2: "次强", 3: "次弱", 4: "最弱"}.get(r, "")
            print(f"  {r}({desc}){'':<2} {n:>5} {wr:>6.1f}% {avg:>+8.1f}pt {tot:>+8.1f}pt")

        # 按方向×rank拆分
        print(f"\n  {sym} 方向×rank交叉分析:")
        print(f"  {'Direction':<8} {'Rank':<6} {'N':>4} {'WR%':>7} {'AvgPnL':>8}")
        print("  " + "-" * 40)
        for direction in ["LONG", "SHORT"]:
            d_sub = df_valid[df_valid["direction"] == direction]
            for r in [1, 2, 3, 4]:
                sub = d_sub[d_sub["rank"] == r]
                n = len(sub)
                if n == 0:
                    continue
                wr = (sub["pnl_pts"] > 0).mean() * 100
                avg = sub["pnl_pts"].mean()
                rank_desc = {1: "最强", 2: "次强", 3: "次弱", 4: "最弱"}.get(r, "")
                print(f"  {direction:<8} {r}({rank_desc}){'':<2} {n:>4} {wr:>6.1f}% {avg:>+7.1f}pt")

        # Pearson相关: rank vs pnl_pts
        if len(df_valid) >= 5:
            c, p = pearsonr(df_valid["rank"].values, df_valid["pnl_pts"].values)
            star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            print(f"\n  {sym} rank vs pnl_pts: Pearson r={c:+.3f} (p={p:.3f} {star})")

            # 做多子集
            long_sub = df_valid[df_valid["direction"] == "LONG"]
            if len(long_sub) >= 5:
                c2, p2 = pearsonr(long_sub["rank"].values, long_sub["pnl_pts"].values)
                star2 = "***" if p2 < 0.001 else ("**" if p2 < 0.01 else ("*" if p2 < 0.05 else "ns"))
                print(f"  {sym} LONG rank vs pnl_pts: Pearson r={c2:+.3f} (p={p2:.3f} {star2})")

            # 做空子集（理论: rank越高=越弱=做空越好 → 预期正相关）
            short_sub = df_valid[df_valid["direction"] == "SHORT"]
            if len(short_sub) >= 5:
                c3, p3 = pearsonr(short_sub["rank"].values, short_sub["pnl_pts"].values)
                star3 = "***" if p3 < 0.001 else ("**" if p3 < 0.01 else ("*" if p3 < 0.05 else "ns"))
                print(f"  {sym} SHORT rank vs pnl_pts: Pearson r={c3:+.3f} (p={p3:.3f} {star3})")


# ---------------------------------------------------------------------------
# Step 4: IM-IH剪刀差（style spread）分析
# ---------------------------------------------------------------------------

def step4_style_spread(
    all_trades_by_sym: Dict[str, List],
    rank_df: pd.DataFrame,
) -> None:
    """Step 4: style_spread（IM-IH return差）与IM交易结果的关系。"""
    print("\n" + "=" * 70)
    print(" Step 4: IM-IH剪刀差（style_spread）分析")
    print("=" * 70)

    sym = "IM"
    trades_all = all_trades_by_sym.get(sym, [])
    if not trades_all:
        print("  IM无交易记录")
        return

    enriched = []
    for td, trade in trades_all:
        if trade.get("partial"):
            continue
        date_dash = f"{td[:4]}-{td[4:6]}-{td[6:]}"
        entry_time = trade.get("entry_time", "")
        utc_hm = _bj_to_utc(entry_time)
        utc_dt = f"{date_dash} {utc_hm}:00"

        day_rank = rank_df[rank_df.index.str.startswith(date_dash)]
        if day_rank.empty:
            continue

        idx_list = day_rank.index.tolist()
        try:
            pos = next(i for i, x in enumerate(idx_list) if x >= utc_dt)
            if pos == 0:
                continue
            prev_idx = idx_list[pos - 1]
            spread_val = day_rank.loc[prev_idx, "style_spread"]
        except StopIteration:
            spread_val = day_rank.iloc[-1]["style_spread"]

        if pd.isna(spread_val):
            continue

        t = dict(trade)
        t["style_spread"] = float(spread_val)
        t["date"] = td
        enriched.append(t)

    if not enriched:
        print("  无法匹配style_spread数据")
        return

    df = pd.DataFrame(enriched)
    print(f"\n  样本量: {len(df)} 笔")

    # 相关性
    c, p = pearsonr(df["style_spread"].values, df["pnl_pts"].values)
    star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"  style_spread vs pnl_pts: Pearson r={c:+.3f} (p={p:.3f} {star})")

    # 按spread分三档
    q33 = df["style_spread"].quantile(0.33)
    q67 = df["style_spread"].quantile(0.67)
    print(f"\n  style_spread分位数: Q33={q33:.4%}, Q67={q67:.4%}")
    print(f"\n  {'分组':<20} {'N':>4} {'WR%':>7} {'AvgPnL':>9} {'TotPnL':>9}")
    print("  " + "-" * 55)

    for label, mask in [
        (f"负spread (<{q33:.3%})", df["style_spread"] < q33),
        (f"中性spread", (df["style_spread"] >= q33) & (df["style_spread"] <= q67)),
        (f"正spread (>{q67:.3%})", df["style_spread"] > q67),
    ]:
        sub = df[mask]
        n = len(sub)
        if n == 0:
            print(f"  {label:<20} {n:>4}")
            continue
        wr = (sub["pnl_pts"] > 0).mean() * 100
        avg = sub["pnl_pts"].mean()
        tot = sub["pnl_pts"].sum()
        print(f"  {label:<20} {n:>4} {wr:>6.1f}% {avg:>+8.1f}pt {tot:>+8.1f}pt")

    # 做多/做空×spread
    print(f"\n  做多时style_spread预测力:")
    long_df = df[df["direction"] == "LONG"]
    if len(long_df) >= 5:
        c_l, p_l = pearsonr(long_df["style_spread"].values, long_df["pnl_pts"].values)
        star_l = "***" if p_l < 0.001 else ("**" if p_l < 0.01 else ("*" if p_l < 0.05 else "ns"))
        print(f"  LONG  style_spread vs pnl_pts: r={c_l:+.3f} (p={p_l:.3f} {star_l})")

    print(f"\n  做空时style_spread预测力:")
    short_df = df[df["direction"] == "SHORT"]
    if len(short_df) >= 5:
        c_s, p_s = pearsonr(short_df["style_spread"].values, short_df["pnl_pts"].values)
        star_s = "***" if p_s < 0.001 else ("**" if p_s < 0.01 else ("*" if p_s < 0.05 else "ns"))
        print(f"  SHORT style_spread vs pnl_pts: r={c_s:+.3f} (p={p_s:.3f} {star_s})")


# ---------------------------------------------------------------------------
# Step 5: 模拟接入rank乘数
# ---------------------------------------------------------------------------

def run_day_with_rank_mult(
    sym: str,
    td: str,
    db: DBManager,
    rank_df: pd.DataFrame,
    mult_scheme: str = "A",
    version: str = "auto",
) -> List[Dict]:
    """
    在回测循环中，score_all之后、阈值判断之前，乘以rank乘数。
    复用backtest_signals_day.py的run_day逻辑，仅改动score应用部分。
    """
    import importlib
    import scripts.backtest_signals_day as bsd

    from strategies.intraday.A_share_momentum_signal_v2 import (
        SignalGeneratorV2, SignalGeneratorV3, SentimentData, check_exit,
        is_open_allowed, SIGNAL_ROUTING, SYMBOL_PROFILES, _DEFAULT_PROFILE,
        STOP_LOSS_PCT, TRAILING_STOP_HIVOL, TRAILING_STOP_NORMAL,
    )

    date_dash = f"{td[:4]}-{td[4:6]}-{td[6:]}"

    _SPOT_SYM = {"IM": "000852", "IF": "000300", "IH": "000016", "IC": "000905"}
    spot_sym = _SPOT_SYM.get(sym)
    all_bars = db.query_df(
        f"SELECT datetime, open, high, low, close, volume "
        f"FROM index_min WHERE symbol='{spot_sym}' AND period=300 "
        f"ORDER BY datetime"
    ) if spot_sym else None
    if all_bars is None or all_bars.empty:
        return []

    for c in ["open", "high", "low", "close", "volume"]:
        all_bars[c] = all_bars[c].astype(float)

    today_mask = all_bars["datetime"].str.startswith(date_dash)
    today_indices = all_bars.index[today_mask].tolist()
    if not today_indices:
        return []

    _SPOT_IDX = {"IM": "000852.SH", "IF": "000300.SH", "IH": "000016.SH", "IC": "000905.SH"}
    idx_code = _SPOT_IDX.get(sym, f"{sym}.CFX")
    daily_all = db.query_df(
        f"SELECT trade_date, close as open, close as high, close as low, close, 0 as volume "
        f"FROM index_daily WHERE ts_code='{idx_code}' ORDER BY trade_date"
    )
    if daily_all is not None and not daily_all.empty:
        daily_all["close"] = daily_all["close"].astype(float)
    else:
        daily_all = None

    spot_all = db.query_df(
        f"SELECT trade_date, close FROM index_daily WHERE ts_code='{idx_code}' ORDER BY trade_date"
    )
    if spot_all is not None:
        spot_all["close"] = spot_all["close"].astype(float)

    def _zscore_for_date(target_date: str):
        if spot_all is None or spot_all.empty:
            return 0.0, 0.0
        sub = spot_all[spot_all["trade_date"] < target_date].tail(30)
        if len(sub) < 20:
            return 0.0, 0.0
        closes = sub["close"].values
        ema = float(pd.Series(closes).ewm(span=20).mean().iloc[-1])
        std = float(pd.Series(closes).rolling(20).std().iloc[-1])
        return ema, std

    ema20, std20 = _zscore_for_date(td)

    is_high_vol = True
    dmo = db.query_df(
        "SELECT garch_forecast_vol FROM daily_model_output "
        "WHERE underlying='IM' AND garch_forecast_vol > 0 "
        f"AND trade_date < '{td}' "
        "ORDER BY trade_date DESC LIMIT 1"
    )
    if dmo is not None and not dmo.empty:
        is_high_vol = (float(dmo.iloc[0].iloc[0]) * 100 / 24.9) > 1.2

    sentiment = None
    try:
        sdf = db.query_df(
            "SELECT atm_iv, atm_iv_market, vrp, term_structure_shape, rr_25d "
            "FROM daily_model_output WHERE underlying='IM' "
            f"AND trade_date < '{td}' "
            "ORDER BY trade_date DESC LIMIT 2"
        )
        if sdf is not None and len(sdf) >= 1:
            cur = sdf.iloc[0]
            prev = sdf.iloc[1] if len(sdf) >= 2 else sdf.iloc[0]
            sentiment = SentimentData(
                atm_iv=float(cur.get("atm_iv_market") or cur.get("atm_iv") or 0),
                atm_iv_prev=float(prev.get("atm_iv_market") or prev.get("atm_iv") or 0),
                rr_25d=float(cur.get("rr_25d") or 0),
                rr_25d_prev=float(prev.get("rr_25d") or 0),
                vrp=float(cur.get("vrp") or 0),
                term_structure=str(cur.get("term_structure_shape") or ""),
            )
    except Exception:
        pass

    d_override = None
    try:
        briefing_row = db.query_df(
            "SELECT d_override_long, d_override_short FROM morning_briefing "
            f"WHERE trade_date = '{td}' LIMIT 1"
        )
        if briefing_row is not None and len(briefing_row) > 0:
            d_long = briefing_row.iloc[0].get("d_override_long")
            d_short = briefing_row.iloc[0].get("d_override_short")
            if d_long is not None and d_short is not None:
                d_override = {"LONG": float(d_long), "SHORT": float(d_short)}
    except Exception:
        pass

    _ver = version if version != "auto" else SIGNAL_ROUTING.get(sym, "v2")
    gen = SignalGeneratorV3({"min_signal_score": 60}) if _ver == "v3" else SignalGeneratorV2({"min_signal_score": 60})

    _sym_prof = SYMBOL_PROFILES.get(sym, _DEFAULT_PROFILE)
    effective_threshold = _sym_prof.get("signal_threshold", 60)

    # day_rank: 当天rank数据，按datetime索引
    day_rank = rank_df[rank_df.index.str.startswith(date_dash)]
    rank_idx_list = day_rank.index.tolist() if not day_rank.empty else []
    sym_code = SPOT_MAP.get(sym, "")
    rank_col = f"rank_{sym_code}"

    def _get_prev_bar_rank(utc_dt_str: str) -> Optional[int]:
        """获取当前bar的上一根完成bar的rank（防lookahead）。"""
        if not rank_idx_list or rank_col not in day_rank.columns:
            return None
        try:
            pos = next(i for i, x in enumerate(rank_idx_list) if x >= utc_dt_str)
            if pos == 0:
                return None
            prev_idx = rank_idx_list[pos - 1]
            val = day_rank.loc[prev_idx, rank_col]
            return int(val) if not pd.isna(val) else None
        except StopIteration:
            return None

    def _build_15m_from_5m(bar_5m: pd.DataFrame) -> pd.DataFrame:
        return bsd._build_15m_from_5m(bar_5m)

    position = None
    completed_trades = []
    last_exit_utc = ""
    last_exit_dir = ""
    COOLDOWN_MINUTES = 15

    daily_df = None
    if daily_all is not None:
        daily_df = daily_all[daily_all["trade_date"] < td].tail(30).reset_index(drop=True)
        if daily_df.empty:
            daily_df = None

    prev_c = 0.0
    if daily_df is not None and len(daily_df) >= 2:
        prev_rows = daily_df[daily_df["trade_date"] < td]
        if len(prev_rows) > 0:
            prev_c = float(prev_rows.iloc[-1]["close"])

    _today_open = 0.0
    _gap_pct = 0.0
    if prev_c > 0 and today_indices:
        _today_open = float(all_bars.loc[today_indices[0], "open"])
        _gap_pct = (_today_open - prev_c) / prev_c

    for idx in today_indices:
        bar_5m = all_bars.loc[:idx].tail(200).copy()
        if len(bar_5m) < 15:
            continue

        bar_5m_signal = bar_5m.iloc[:-1]
        if len(bar_5m_signal) < 15:
            continue

        price = float(bar_5m.iloc[-1]["close"])
        high = float(bar_5m.iloc[-1]["high"])
        low = float(bar_5m.iloc[-1]["low"])
        signal_price = float(bar_5m_signal.iloc[-1]["close"])
        dt_str = str(all_bars.loc[idx, "datetime"])
        utc_hm = dt_str[11:16]
        bj_time = _utc_to_bj(utc_hm)

        z_val = (signal_price - ema20) / std20 if std20 > 0 else None
        bar_15m = _build_15m_from_5m(bar_5m_signal)

        result = gen.score_all(
            sym, bar_5m_signal, bar_15m, daily_df, None, sentiment,
            zscore=z_val, is_high_vol=is_high_vol, d_override=d_override,
        )

        score = result["total"] if result else 0
        direction = result["direction"] if result else ""

        # --- rank乘数介入点（score_all之后、阈值判断之前）---
        utc_dt_str = f"{date_dash} {utc_hm}:00"
        cur_rank = _get_prev_bar_rank(utc_dt_str)

        rank_mult = 1.0
        if cur_rank is not None and direction:
            if mult_scheme == "A":
                rank_mult = RANK_MULT_A.get(cur_rank, 1.0)
            elif mult_scheme == "B":
                rank_mult = rank_mult_b(cur_rank, direction)

        adj_score = int(round(min(100, max(0, score * rank_mult))))

        # 检查持仓退出（不受rank影响）
        action_str = ""
        if position is not None:
            stop_price = position.get("stop_loss", 0)
            bar_stopped = False
            if stop_price > 0:
                if position["direction"] == "LONG" and low <= stop_price:
                    bar_stopped = True
                elif position["direction"] == "SHORT" and high >= stop_price:
                    bar_stopped = True

            if bar_stopped:
                entry_p = position["entry_price"]
                exit_p = stop_price
                pnl_pts = (exit_p - entry_p) if position["direction"] == "LONG" else (entry_p - exit_p)
                elapsed = bsd._calc_minutes(position["entry_time_utc"], utc_hm)
                completed_trades.append({
                    "entry_time": _utc_to_bj(position["entry_time_utc"]),
                    "entry_price": entry_p,
                    "exit_time": bj_time,
                    "exit_price": exit_p,
                    "direction": position["direction"],
                    "pnl_pts": pnl_pts,
                    "reason": "STOP_LOSS",
                    "minutes": elapsed,
                    "entry_score": position.get("entry_score", 0),
                    "entry_rank": position.get("entry_rank"),
                    "entry_rank_mult": position.get("entry_rank_mult", 1.0),
                })
                last_exit_utc = utc_hm
                last_exit_dir = position["direction"]
                position = None
                action_str = "STOP_LOSS"
            else:
                if position["direction"] == "LONG":
                    position["highest_since"] = max(position["highest_since"], high)
                else:
                    position["lowest_since"] = min(position["lowest_since"], low)

                reverse_score = 0
                if result and direction and direction != position["direction"]:
                    reverse_score = adj_score

                exit_info = check_exit(
                    position, price, bar_5m_signal,
                    bar_15m if not bar_15m.empty else None,
                    utc_hm, reverse_score, is_high_vol=is_high_vol,
                    symbol=sym,
                )

                if exit_info["should_exit"]:
                    exit_vol = exit_info["exit_volume"]
                    reason = exit_info["exit_reason"]
                    entry_p = position["entry_price"]
                    exit_p = price
                    pnl_pts = (exit_p - entry_p) if position["direction"] == "LONG" else (entry_p - exit_p)
                    elapsed = bsd._calc_minutes(position["entry_time_utc"], utc_hm)

                    if exit_vol >= position["volume"]:
                        completed_trades.append({
                            "entry_time": _utc_to_bj(position["entry_time_utc"]),
                            "entry_price": entry_p,
                            "exit_time": bj_time,
                            "exit_price": exit_p,
                            "direction": position["direction"],
                            "pnl_pts": pnl_pts,
                            "reason": reason,
                            "minutes": elapsed,
                            "entry_score": position.get("entry_score", 0),
                            "entry_rank": position.get("entry_rank"),
                            "entry_rank_mult": position.get("entry_rank_mult", 1.0),
                        })
                        last_exit_utc = utc_hm
                        last_exit_dir = position["direction"]
                        position = None
                        action_str = f"EXIT {reason}"
                    else:
                        position["volume"] -= exit_vol
                        position["half_closed"] = True
                        completed_trades.append({
                            "entry_time": _utc_to_bj(position["entry_time_utc"]),
                            "entry_price": entry_p,
                            "exit_time": bj_time,
                            "exit_price": exit_p,
                            "direction": position["direction"],
                            "pnl_pts": pnl_pts,
                            "reason": reason + "(半仓)",
                            "minutes": elapsed,
                            "partial": True,
                            "entry_score": position.get("entry_score", 0),
                            "entry_rank": position.get("entry_rank"),
                            "entry_rank_mult": position.get("entry_rank_mult", 1.0),
                        })
                        action_str = f"PARTIAL {reason}"

        # 入场检查（用adj_score）
        in_cooldown = False
        if last_exit_utc and direction == last_exit_dir:
            cd_elapsed = bsd._calc_minutes(last_exit_utc, utc_hm)
            if 0 < cd_elapsed < COOLDOWN_MINUTES:
                in_cooldown = True

        if (position is None and not action_str and result and not in_cooldown
                and adj_score >= effective_threshold and direction and is_open_allowed(utc_hm)):
            entry_p = price
            stop = entry_p * (1 - STOP_LOSS_PCT) if direction == "LONG" else entry_p * (1 + STOP_LOSS_PCT)
            position = {
                "entry_price": entry_p,
                "direction": direction,
                "entry_time_utc": utc_hm,
                "highest_since": high,
                "lowest_since": low,
                "stop_loss": stop,
                "volume": 1,
                "half_closed": False,
                "bars_below_mid": 0,
                "entry_score": adj_score,
                "entry_rank": cur_rank,
                "entry_rank_mult": rank_mult,
            }

    # EOD强平
    if position is not None and today_indices:
        last_price = float(all_bars.loc[today_indices[-1], "close"])
        entry_p = position["entry_price"]
        exit_p = last_price
        pnl = (exit_p - entry_p) if position["direction"] == "LONG" else (entry_p - exit_p)
        elapsed = bsd._calc_minutes(
            position["entry_time_utc"],
            str(all_bars.loc[today_indices[-1], "datetime"])[11:16]
        )
        completed_trades.append({
            "entry_time": _utc_to_bj(position["entry_time_utc"]),
            "entry_price": entry_p,
            "exit_time": _utc_to_bj(str(all_bars.loc[today_indices[-1], "datetime"])[11:16]),
            "exit_price": exit_p,
            "direction": position["direction"],
            "pnl_pts": pnl,
            "reason": "EOD_FORCE",
            "minutes": elapsed,
            "entry_score": position.get("entry_score", 0),
            "entry_rank": position.get("entry_rank"),
            "entry_rank_mult": position.get("entry_rank_mult", 1.0),
        })

    return completed_trades


def step5_rank_mult_simulation(
    dates: List[str],
    db: DBManager,
    rank_df: pd.DataFrame,
    baseline_by_sym: Dict[str, Dict],
) -> None:
    """Step 5: 模拟rank乘数效果，对比baseline。"""
    print("\n" + "=" * 70)
    print(" Step 5: rank乘数模拟接入效果")
    print("=" * 70)

    CONTRACT_MULT = {"IM": 200, "IC": 200}

    for sym in ["IM", "IC"]:
        print(f"\n  {'─'*60}")
        print(f"  {sym} | baseline vs 方案A(rank无感) vs 方案B(rank方向感知)")
        print(f"  {'─'*60}")

        baseline = baseline_by_sym.get(sym, {})

        for scheme_name, scheme_key in [("方案A(无感)", "A"), ("方案B(方向感知)", "B")]:
            all_trades = []
            for td in dates:
                try:
                    trades = run_day_with_rank_mult(sym, td, db, rank_df,
                                                    mult_scheme=scheme_key)
                    all_trades.extend(trades)
                except Exception as e:
                    print(f"    {td} {sym} {scheme_key} error: {e}")
                    continue

            full_trades = [t for t in all_trades if not t.get("partial")]
            n = len(full_trades)
            wins = len([t for t in full_trades if t["pnl_pts"] > 0])
            wr = wins / n * 100 if n > 0 else 0
            tot_pnl = sum(t["pnl_pts"] for t in full_trades)
            avg_pnl = tot_pnl / n if n > 0 else 0
            mult = CONTRACT_MULT.get(sym, 200)

            # 和baseline比较
            b_n = baseline.get("n", 0)
            b_wr = baseline.get("wr", 0)
            b_pnl = baseline.get("tot_pnl", 0)

            print(f"\n  {scheme_name}:")
            print(f"    笔数: {n} ({b_n}→{n}, {'↑' if n >= b_n else '↓'}{abs(n-b_n)}笔)")
            print(f"    WR:   {wr:.1f}% ({b_wr:.1f}%→{wr:.1f}%, {wr-b_wr:+.1f}%)")
            print(f"    总PnL:{tot_pnl:+.0f}pt ({b_pnl:+.0f}→{tot_pnl:+.0f}, {tot_pnl-b_pnl:+.0f}pt)")
            print(f"    均PnL:{avg_pnl:+.1f}pt")
            print(f"    总收益:{tot_pnl*mult:+,.0f}元(1手)")

            # rank分布统计
            if full_trades:
                rank_counts = {}
                for t in full_trades:
                    r = t.get("entry_rank")
                    if r is not None:
                        rank_counts[r] = rank_counts.get(r, 0) + 1
                total_with_rank = sum(rank_counts.values())
                if total_with_rank > 0:
                    dist = " | ".join(
                        f"rank{r}:{cnt}笔({cnt/total_with_rank:.0%})"
                        for r, cnt in sorted(rank_counts.items())
                    )
                    print(f"    rank分布: {dist}")


# ---------------------------------------------------------------------------
# Baseline回测（复用backtest_signals_day逻辑）
# ---------------------------------------------------------------------------

def run_baseline(dates: List[str], db: DBManager) -> Dict[str, Dict]:
    """运行baseline回测，返回各品种汇总。"""
    from scripts.backtest_signals_day import run_day, _patch_threshold

    results = {}
    CONTRACT_MULT = {"IM": 200, "IC": 200}

    for sym in ["IM", "IC"]:
        _patch_threshold(65 if sym == "IC" else 60)
        all_trades = []
        for td in dates:
            try:
                trades = run_day(sym, td, db, verbose=False)
                all_trades.extend([(td, t) for t in trades])
            except Exception as e:
                print(f"  [WARN] baseline {sym} {td}: {e}")

        full_trades = [t for _, t in all_trades if not t.get("partial")]
        n = len(full_trades)
        wins = len([t for t in full_trades if t["pnl_pts"] > 0])
        wr = wins / n * 100 if n > 0 else 0
        tot_pnl = sum(t["pnl_pts"] for t in full_trades)
        avg_pnl = tot_pnl / n if n > 0 else 0
        mult = CONTRACT_MULT.get(sym, 200)

        results[sym] = {
            "n": n, "wr": wr, "tot_pnl": tot_pnl, "avg_pnl": avg_pnl,
            "yuan": tot_pnl * mult,
            "trades": all_trades,
        }
        print(f"  [Baseline] {sym}: N={n}, WR={wr:.1f}%, PnL={tot_pnl:+.0f}pt "
              f"({tot_pnl*mult:+,.0f}元), 均={avg_pnl:+.1f}pt")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="跨品种相对强弱rank可行性分析")
    parser.add_argument("--dates", default=DEFAULT_DATES,
                        help="逗号分隔的回测日期(YYYYMMDD)")
    parser.add_argument("--no-backtest", action="store_true",
                        help="跳过回测部分（Step 3/5），只做相关性分析（更快）")
    parser.add_argument("--version", default="auto", choices=["v2", "v3", "auto"])
    args = parser.parse_args()

    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    print(f"\n{'='*70}")
    print(f" 跨品种相对强弱rank分析")
    print(f" 日期: {dates[0]} ~ {dates[-1]} ({len(dates)}天)")
    print(f"{'='*70}")

    db = get_db()

    # ---------------------------------------------------------------------------
    # 加载数据
    # ---------------------------------------------------------------------------
    print("\n[*] 加载index_min数据...")
    all_bars = load_all_spot_bars(db, dates)
    print(f"    共 {len(all_bars):,} 行，4品种×{len(dates)}天")

    # ---------------------------------------------------------------------------
    # Step 1
    # ---------------------------------------------------------------------------
    print("\n[*] Step 1: 计算跨品种rank...")
    rank_df = compute_cross_rank(all_bars)
    print(f"    rank_df shape: {rank_df.shape}")

    # 显示几行示例
    print("\n  前5行示例（含rank）:")
    rank_cols = [c for c in rank_df.columns if "rank_" in c or "return_" in c or c == "style_spread"]
    sample = rank_df[rank_cols].dropna().head(5)
    for i, (idx, row) in enumerate(sample.iterrows()):
        parts = []
        for sym_code, sym_name in [("000852","IM"),("000905","IC"),("000300","IF"),("000016","IH")]:
            r = row.get(f"rank_{sym_code}", "?")
            ret = row.get(f"return_{sym_code}", 0)
            parts.append(f"{sym_name}:rank{r:.0f}({ret:.3%})")
        spread = row.get("style_spread", 0)
        print(f"  {str(idx)[:16]}  {' | '.join(parts)}  style_spread={spread:.4%}")

    # rank分布统计
    print("\n  rank分布（各品种各rank的频次）:")
    print(f"  {'品种':<8} {'rank1':>8} {'rank2':>8} {'rank3':>8} {'rank4':>8} {'total':>8}")
    print("  " + "-" * 50)
    sym_map_rev = {v: k for k, v in SPOT_MAP.items()}
    for sym_code in ["000852", "000905", "000300", "000016"]:
        fut_sym = sym_map_rev.get(sym_code, sym_code)
        col = f"rank_{sym_code}"
        if col not in rank_df.columns:
            continue
        counts = rank_df[col].value_counts().sort_index()
        total = counts.sum()
        row_parts = [f"{counts.get(r, 0):>8}" for r in [1, 2, 3, 4]]
        print(f"  {fut_sym}({sym_code}): {''.join(row_parts)} {total:>8}")

    # ---------------------------------------------------------------------------
    # Step 2
    # ---------------------------------------------------------------------------
    print("\n[*] Step 2: 相关性分析...")
    mvq_df = compute_mvq_proxy(all_bars)
    step2_correlation(rank_df, mvq_df)

    # ---------------------------------------------------------------------------
    # Step 3 & 5 (需要回测)
    # ---------------------------------------------------------------------------
    if args.no_backtest:
        print("\n[--no-backtest] 跳过Step 3/4/5回测部分")
    else:
        print("\n[*] 运行Baseline回测（IM/IC）...")
        baseline_by_sym = run_baseline(dates, db)

        # 整理trades供Step3/4使用
        all_trades_by_sym = {
            sym: baseline_by_sym[sym]["trades"]
            for sym in ["IM", "IC"]
            if sym in baseline_by_sym
        }

        # Step 3
        step3_rank_predictability(all_trades_by_sym, rank_df, dates)

        # Step 4
        step4_style_spread(all_trades_by_sym, rank_df)

        # Step 5
        print("\n[*] Step 5: rank乘数模拟回测...")
        print("    注意：方案A/B在score_all结果上乘以rank_mult，阈值保持不变")
        step5_rank_mult_simulation(
            dates, db, rank_df,
            {sym: baseline_by_sym[sym] for sym in ["IM", "IC"] if sym in baseline_by_sym}
        )

    # ---------------------------------------------------------------------------
    # 综合结论
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" 综合结论")
    print("=" * 70)
    print("""
  [Step 1] rank计算
  - 每根5分钟bar的4品种return排名（rank1=最强，rank4=最弱）
  - 防lookahead：信号rank用当前bar的上一根完成bar的return

  [Step 2] rank与MVQ相关性
  - rank与M_proxy（动量）理论上负相关（rank低=强=M大）
  - 若|corr|>0.2且统计显著(*)，说明rank是M的有效代理
  - V_proxy/Q_proxy相关性揭示rank是否反映波动/成交行为

  [Step 3] rank的交易预测力
  - 核心问题：高rank(最强)做多，或低rank(最弱)做空时，胜率/均PnL是否更好？
  - 若rank1做多比rank4做多明显更好 → rank有用，方向感知方案B有价值
  - 若各rank表现接近(p>0.1) → rank无显著预测力

  [Step 4] style spread（IM-IH）
  - style_spread>0表示小盘(IM)跑赢大盘(IH)
  - 若做多时正spread对应更好结果，验证风格共振的价值

  [Step 5] rank乘数模拟
  - 方案A（无感）：最强/最弱分别+10%/-15%得分调整
  - 方案B（方向感知）：做多时最强+10%，做空时最弱+10%
  - 若总PnL提升且交易笔数未大幅减少，说明rank过滤有增量价值

  决策建议：
  - 若Step3 Pearson p<0.05 + Step5两方案均正增量 → 可考虑接入monitor
  - 若rank无显著预测力(p>0.1) → 继续观察，不接入正式系统
    """)


if __name__ == "__main__":
    main()
