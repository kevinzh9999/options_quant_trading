#!/usr/bin/env python3
"""
discount_convergence_research.py
--------------------------------
研究IM期货贴水在到期日前的收敛规律，以及现货在到期前的表现。

核心假设：量化基金持有IM空头对冲，到期前可能通过砸现货来减少基差收敛损失。

用法:
    python scripts/discount_convergence_research.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(ROOT) / ".env")
except ImportError:
    pass

from data.storage.db_manager import DBManager, get_db
from config.config_loader import ConfigLoader

# ── 常量 ──────────────────────────────────────────────
LOOKBACK_DAYS = 30          # 到期前研究天数
MIN_VOLUME = 1000           # 最低日成交量过滤
QUARTER_MONTHS = {3, 6, 9, 12}
OUTPUT_DIR = Path(ROOT) / "logs" / "research"


# ======================================================================
# 数据下载
# ======================================================================

def download_im_specific_contracts(db: DBManager) -> pd.DataFrame:
    """
    从 Tushare 下载所有IM具体月份合约的日线数据并写入 futures_daily 表。
    返回合约清单 DataFrame。
    """
    import tushare as tus
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        print("ERROR: TUSHARE_TOKEN not set in .env")
        sys.exit(1)

    pro = tus.pro_api(token)

    # 获取合约列表
    fut_basic = pro.fut_basic(
        exchange="CFFEX", fut_type="1",
        fields="ts_code,symbol,name,list_date,delist_date,multiplier"
    )
    im_contracts = fut_basic[
        fut_basic["ts_code"].str.startswith("IM")
        & ~fut_basic["ts_code"].str.contains("IML|IMC")
    ].sort_values("ts_code").reset_index(drop=True)

    print(f"共 {len(im_contracts)} 个IM具体合约")

    # 检查哪些已经在数据库中
    existing = db.query_df(
        "SELECT DISTINCT ts_code FROM futures_daily "
        "WHERE ts_code LIKE 'IM2%' AND ts_code NOT LIKE 'IML%'"
    )
    existing_set = set()
    if existing is not None and not existing.empty:
        existing_set = set(existing["ts_code"].tolist())

    to_download = im_contracts[~im_contracts["ts_code"].isin(existing_set)]
    print(f"已有 {len(existing_set)} 个，需下载 {len(to_download)} 个")

    for _, row in to_download.iterrows():
        ts_code = row["ts_code"]
        try:
            df = pro.fut_daily(
                ts_code=ts_code,
                fields="ts_code,trade_date,open,high,low,close,vol,oi,settle,pre_close,pre_settle"
            )
            if df is not None and not df.empty:
                df = df.rename(columns={"vol": "volume"})
                db.upsert_dataframe("futures_daily", df)
                print(f"  ✓ {ts_code}: {len(df)} rows")
            else:
                print(f"  - {ts_code}: no data")
            time.sleep(0.35)  # rate limit
        except Exception as e:
            print(f"  ✗ {ts_code}: {e}")
            time.sleep(1)

    return im_contracts


# ======================================================================
# 数据准备
# ======================================================================

def build_expiry_cycles(
    db: DBManager,
    contracts: pd.DataFrame,
) -> List[Dict]:
    """
    为每个到期周期构建对齐的贴水时间序列。

    Returns list of dicts:
        {contract, delist_date, expiry_month, is_quarter,
         data: DataFrame[T_minus, trade_date, futures_close, spot_close,
                         discount_pts, discount_pct, futures_ret, spot_ret]}
    """
    # 加载现货
    spot_df = db.query_df(
        "SELECT trade_date, close as spot_close FROM index_daily "
        "WHERE ts_code='000852.SH' ORDER BY trade_date"
    )
    if spot_df is None or spot_df.empty:
        print("ERROR: 无 000852.SH 数据")
        return []
    spot_map = dict(zip(spot_df["trade_date"], spot_df["spot_close"].astype(float)))

    cycles = []

    for _, row in contracts.iterrows():
        ts_code = row["ts_code"]
        delist_date = str(row["delist_date"])

        if not delist_date or delist_date == "None":
            continue

        # 提取到期月份
        # IM2309.CFX → month=9
        month_str = ts_code[4:6] if ts_code[2:4].isdigit() else ts_code[4:6]
        try:
            expiry_month = int(ts_code[4:6])
        except ValueError:
            continue

        is_quarter = expiry_month in QUARTER_MONTHS

        # 读取该合约日线
        fut_df = db.query_df(
            f"SELECT trade_date, close, volume, oi FROM futures_daily "
            f"WHERE ts_code='{ts_code}' AND close > 0 ORDER BY trade_date"
        )
        if fut_df is None or fut_df.empty:
            continue

        fut_df["close"] = fut_df["close"].astype(float)
        fut_df["volume"] = fut_df["volume"].astype(float)

        # 过滤低成交量
        fut_df = fut_df[fut_df["volume"] >= MIN_VOLUME].copy()
        if fut_df.empty:
            continue

        # 只取到期日之前（含）的数据
        fut_df = fut_df[fut_df["trade_date"] <= delist_date].copy()
        if len(fut_df) < 5:
            continue

        # 按距到期日天数对齐
        # 获取到期日前的交易日列表
        all_dates = fut_df["trade_date"].tolist()
        expiry_idx = len(all_dates) - 1  # 最后一天即到期日或最后交易日

        records = []
        for i, (_, frow) in enumerate(fut_df.iterrows()):
            td = frow["trade_date"]
            t_minus = expiry_idx - i  # T-N
            if t_minus > LOOKBACK_DAYS:
                continue

            spot = spot_map.get(td)
            if spot is None or spot <= 0:
                continue

            futures_close = float(frow["close"])
            discount_pts = futures_close - spot
            discount_pct = discount_pts / spot * 100

            records.append({
                "T_minus": t_minus,
                "trade_date": td,
                "futures_close": futures_close,
                "spot_close": spot,
                "discount_pts": discount_pts,
                "discount_pct": discount_pct,
            })

        if len(records) < 5:
            continue

        df_cycle = pd.DataFrame(records).sort_values("T_minus", ascending=False).reset_index(drop=True)

        cycles.append({
            "contract": ts_code,
            "delist_date": delist_date,
            "expiry_month": expiry_month,
            "is_quarter": is_quarter,
            "year": 2000 + int(ts_code[2:4]),
            "data": df_cycle,
        })

    return cycles


# ======================================================================
# 研究1：贴水收敛时间规律
# ======================================================================

def research_convergence(cycles: List[Dict]) -> str:
    """贴水收敛时间规律分析"""
    lines = []
    lines.append("# 研究1：贴水收敛的时间规律\n")

    if not cycles:
        lines.append("无有效到期周期数据\n")
        return "\n".join(lines)

    lines.append(f"共 {len(cycles)} 个到期周期\n")

    # 汇总所有周期的T-N数据
    all_records = []
    for c in cycles:
        for _, r in c["data"].iterrows():
            all_records.append({
                "T_minus": int(r["T_minus"]),
                "discount_pct": r["discount_pct"],
                "discount_pts": r["discount_pts"],
                "contract": c["contract"],
                "is_quarter": c["is_quarter"],
                "year": c["year"],
            })

    df_all = pd.DataFrame(all_records)

    # a. 全量平均贴水曲线
    lines.append("## a. 全量平均贴水收敛曲线\n")
    lines.append(f"{'距到期':<8} {'平均贴水率':>10} {'中位贴水率':>10} {'平均贴水点数':>12} {'样本数':>6}")
    lines.append("-" * 55)

    for t in range(LOOKBACK_DAYS, -1, -1):
        subset = df_all[df_all["T_minus"] == t]
        if subset.empty:
            continue
        mean_pct = subset["discount_pct"].mean()
        median_pct = subset["discount_pct"].median()
        mean_pts = subset["discount_pts"].mean()
        n = len(subset)
        lines.append(f"T-{t:<5} {mean_pct:>+9.2f}%  {median_pct:>+9.2f}%  {mean_pts:>+10.0f}点  {n:>5}")

    # 收敛特征分析
    convergence_by_t = {}
    for t in range(LOOKBACK_DAYS, -1, -1):
        subset = df_all[df_all["T_minus"] == t]
        if not subset.empty:
            convergence_by_t[t] = subset["discount_pct"].mean()

    if convergence_by_t:
        t30 = convergence_by_t.get(30, convergence_by_t.get(max(convergence_by_t.keys())))
        t0 = convergence_by_t.get(0, convergence_by_t.get(min(convergence_by_t.keys())))

        lines.append("")
        if t30 is not None and t0 is not None:
            lines.append(f"**T-30到T-0总收敛幅度: {t30:.2f}% → {t0:.2f}% (变化{t0-t30:+.2f}pp)**\n")

        # 找收敛加速点
        speeds = {}
        sorted_t = sorted(convergence_by_t.keys(), reverse=True)
        for i in range(len(sorted_t) - 1):
            t1, t2 = sorted_t[i], sorted_t[i + 1]
            if t1 - t2 <= 2:
                speed = (convergence_by_t[t2] - convergence_by_t[t1]) / (t1 - t2)
                speeds[t1] = speed

        if speeds:
            fastest_t = max(speeds, key=lambda k: abs(speeds[k]))
            lines.append(f"**收敛速度最快区间: T-{fastest_t}附近 (日均收敛{speeds[fastest_t]:+.3f}pp)**\n")

        # T-0残余贴水
        if 0 in convergence_by_t:
            lines.append(f"**到期日平均残余贴水: {convergence_by_t[0]:+.2f}%**\n")

    # d. 季月 vs 非季月
    lines.append("\n## d. 季月 vs 非季月\n")
    for label, is_q in [("季月(03/06/09/12)", True), ("非季月", False)]:
        subset = df_all[df_all["is_quarter"] == is_q]
        count = len(set(subset["contract"]))
        lines.append(f"\n### {label} ({count}个周期)")
        lines.append(f"{'距到期':<8} {'平均贴水率':>10} {'样本数':>6}")
        lines.append("-" * 30)
        for t in [30, 20, 15, 10, 5, 3, 1, 0]:
            sub = subset[subset["T_minus"] == t]
            if not sub.empty:
                lines.append(f"T-{t:<5} {sub['discount_pct'].mean():>+9.2f}%  {len(sub):>5}")

    # e. 按年份分组
    lines.append("\n## e. 按年份分组\n")
    for year in sorted(df_all["year"].unique()):
        subset = df_all[df_all["year"] == year]
        count = len(set(subset["contract"]))
        lines.append(f"\n### {year}年 ({count}个周期)")
        lines.append(f"{'距到期':<8} {'平均贴水率':>10}")
        lines.append("-" * 25)
        for t in [30, 20, 10, 5, 0]:
            sub = subset[subset["T_minus"] == t]
            if not sub.empty:
                lines.append(f"T-{t:<5} {sub['discount_pct'].mean():>+9.2f}%")

    return "\n".join(lines)


# ======================================================================
# 研究2：到期前现货表现
# ======================================================================

def research_spot_behavior(cycles: List[Dict]) -> str:
    """到期前现货表现分析"""
    lines = []
    lines.append("\n# 研究2：到期前现货的表现\n")

    if not cycles:
        lines.append("无有效数据\n")
        return "\n".join(lines)

    # 计算每个周期在各时段的现货/期货收益率
    periods = [(30, 0), (20, 0), (10, 0), (5, 0), (3, 0)]
    cycle_details = []

    for c in cycles:
        df = c["data"]
        detail = {"contract": c["contract"], "is_quarter": c["is_quarter"], "year": c["year"]}

        for t_start, t_end in periods:
            start_row = df[df["T_minus"] == t_start]
            end_row = df[df["T_minus"] == t_end]
            if start_row.empty or end_row.empty:
                continue

            spot_start = float(start_row.iloc[0]["spot_close"])
            spot_end = float(end_row.iloc[0]["spot_close"])
            fut_start = float(start_row.iloc[0]["futures_close"])
            fut_end = float(end_row.iloc[0]["futures_close"])
            disc_start = float(start_row.iloc[0]["discount_pts"])
            disc_end = float(end_row.iloc[0]["discount_pts"])

            spot_ret = (spot_end - spot_start) / spot_start * 100
            fut_ret = (fut_end - fut_start) / fut_start * 100

            # 收敛贡献分析
            disc_change = disc_end - disc_start  # 贴水变化（负变正 = 收敛）
            spot_contribution = -(spot_end - spot_start)  # 现货下跌贡献
            fut_contribution = fut_end - fut_start  # 期货上涨贡献

            if abs(disc_change) > 0.1:
                spot_contrib_pct = abs(spot_contribution) / (abs(spot_contribution) + abs(fut_contribution)) * 100
            else:
                spot_contrib_pct = 50.0

            key = f"T{t_start}_T{t_end}"
            detail[f"{key}_spot_ret"] = spot_ret
            detail[f"{key}_fut_ret"] = fut_ret
            detail[f"{key}_spot_contrib"] = spot_contrib_pct

        cycle_details.append(detail)

    df_details = pd.DataFrame(cycle_details)

    # 明细表
    lines.append("## 每个周期明细 (T-10到T-0)\n")
    lines.append(f"{'到期周期':<12} {'合约':<14} {'现货T-10':>10} {'期货T-10':>10} {'收敛来源':>12}")
    lines.append("-" * 65)

    for _, r in df_details.iterrows():
        spot_r = r.get("T10_T0_spot_ret")
        fut_r = r.get("T10_T0_fut_ret")
        contrib = r.get("T10_T0_spot_contrib")
        if spot_r is None:
            continue
        if contrib > 60:
            source = f"现货跌({contrib:.0f}%)"
        elif contrib < 40:
            source = f"期货涨({100-contrib:.0f}%)"
        else:
            source = "混合"
        year = int(r["year"])
        contract = str(r["contract"])
        lines.append(f"{year:<12} {contract:<14} {spot_r:>+9.1f}%  {fut_r:>+9.1f}%  {source:>12}")

    # 汇总统计
    lines.append("\n## 汇总统计\n")
    lines.append(f"{'时段':<14} {'现货均值':>10} {'现货跌概率':>10} {'期货均值':>10} {'现货贡献比':>10}")
    lines.append("-" * 60)

    for t_start, t_end in periods:
        key = f"T{t_start}_T{t_end}"
        spot_col = f"{key}_spot_ret"
        fut_col = f"{key}_fut_ret"
        contrib_col = f"{key}_spot_contrib"

        if spot_col not in df_details.columns:
            continue

        spots = df_details[spot_col].dropna()
        futs = df_details[fut_col].dropna()
        contribs = df_details[contrib_col].dropna()

        if spots.empty:
            continue

        spot_mean = spots.mean()
        spot_down_pct = (spots < 0).mean() * 100
        fut_mean = futs.mean() if not futs.empty else 0
        contrib_mean = contribs.mean() if not contribs.empty else 50

        lines.append(
            f"T-{t_start}到T-{t_end}  {spot_mean:>+9.2f}%  {spot_down_pct:>9.0f}%  "
            f"{fut_mean:>+9.2f}%  {contrib_mean:>9.0f}%"
        )

    # 分季月/非季月
    for label, is_q in [("\n### 季月", True), ("\n### 非季月", False)]:
        lines.append(f"{label}")
        sub = df_details[df_details["is_quarter"] == is_q]
        lines.append(f"{'时段':<14} {'现货均值':>10} {'现货跌概率':>10} {'样本数':>6}")
        lines.append("-" * 45)
        for t_start, t_end in periods:
            col = f"T{t_start}_T{t_end}_spot_ret"
            if col not in sub.columns:
                continue
            vals = sub[col].dropna()
            if vals.empty:
                continue
            lines.append(
                f"T-{t_start}到T-{t_end}  {vals.mean():>+9.2f}%  "
                f"{(vals < 0).mean()*100:>9.0f}%  {len(vals):>5}"
            )

    return "\n".join(lines)


# ======================================================================
# 研究3：统计检验
# ======================================================================

def research_statistical_tests(cycles: List[Dict], db: DBManager) -> str:
    """到期日效应的统计检验"""
    lines = []
    lines.append("\n# 研究3：到期日效应的统计检验\n")

    # 加载现货日线
    spot_df = db.query_df(
        "SELECT trade_date, close FROM index_daily "
        "WHERE ts_code='000852.SH' ORDER BY trade_date"
    )
    if spot_df is None or spot_df.empty:
        lines.append("无现货数据\n")
        return "\n".join(lines)

    spot_df["close"] = spot_df["close"].astype(float)
    spot_df["daily_ret"] = spot_df["close"].pct_change() * 100
    spot_df = spot_df.dropna(subset=["daily_ret"])

    # 标记到期日前10天
    expiry_dates = set()
    for c in cycles:
        df = c["data"]
        near_expiry = df[df["T_minus"] <= 10]["trade_date"].tolist()
        expiry_dates.update(near_expiry)

    expiry_5d_dates = set()
    for c in cycles:
        df = c["data"]
        near = df[df["T_minus"] <= 5]["trade_date"].tolist()
        expiry_5d_dates.update(near)

    spot_df["is_expiry_10d"] = spot_df["trade_date"].isin(expiry_dates)
    spot_df["is_expiry_5d"] = spot_df["trade_date"].isin(expiry_5d_dates)

    # a. T检验：到期前10天 vs 非到期前
    expiry_rets = spot_df[spot_df["is_expiry_10d"]]["daily_ret"]
    normal_rets = spot_df[~spot_df["is_expiry_10d"]]["daily_ret"]

    lines.append("## a. 到期前10天 vs 非到期前10天\n")
    lines.append(f"到期前10天: 均值={expiry_rets.mean():+.4f}% 样本={len(expiry_rets)}")
    lines.append(f"非到期前  : 均值={normal_rets.mean():+.4f}% 样本={len(normal_rets)}")

    if len(expiry_rets) > 5 and len(normal_rets) > 5:
        t_stat, t_pval = stats.ttest_ind(expiry_rets, normal_rets)
        u_stat, u_pval = stats.mannwhitneyu(expiry_rets, normal_rets, alternative="two-sided")
        lines.append(f"t检验: t={t_stat:.3f}, p={t_pval:.4f} {'*显著*' if t_pval < 0.05 else '不显著'}")
        lines.append(f"Mann-Whitney U: U={u_stat:.0f}, p={u_pval:.4f} {'*显著*' if u_pval < 0.05 else '不显著'}")

    # b. 到期前5天累计收益
    lines.append("\n## b. 到期前5天现货累计收益率\n")
    cum_rets_5d = []
    for c in cycles:
        df = c["data"]
        t5 = df[df["T_minus"] == 5]
        t0 = df[df["T_minus"] == 0]
        if not t5.empty and not t0.empty:
            s5 = float(t5.iloc[0]["spot_close"])
            s0 = float(t0.iloc[0]["spot_close"])
            cum_rets_5d.append((s0 - s5) / s5 * 100)

    if cum_rets_5d:
        arr = np.array(cum_rets_5d)
        lines.append(f"均值:    {arr.mean():+.3f}%")
        lines.append(f"中位数:  {np.median(arr):+.3f}%")
        lines.append(f"标准差:  {arr.std():.3f}%")
        lines.append(f"下跌比例: {(arr < 0).mean()*100:.0f}%")
        ci = stats.t.interval(0.95, len(arr) - 1, loc=arr.mean(), scale=stats.sem(arr))
        lines.append(f"95%置信区间: [{ci[0]:+.3f}%, {ci[1]:+.3f}%]")

    # c. 季月 vs 非季月
    lines.append("\n## c. 季月 vs 非季月到期前10天\n")
    quarter_dates = set()
    non_quarter_dates = set()
    for c in cycles:
        df = c["data"]
        dates = df[df["T_minus"] <= 10]["trade_date"].tolist()
        if c["is_quarter"]:
            quarter_dates.update(dates)
        else:
            non_quarter_dates.update(dates)

    q_rets = spot_df[spot_df["trade_date"].isin(quarter_dates)]["daily_ret"]
    nq_rets = spot_df[spot_df["trade_date"].isin(non_quarter_dates)]["daily_ret"]

    lines.append(f"季月:   均值={q_rets.mean():+.4f}% 样本={len(q_rets)}")
    lines.append(f"非季月: 均值={nq_rets.mean():+.4f}% 样本={len(nq_rets)}")

    if len(q_rets) > 5 and len(nq_rets) > 5:
        t_stat, t_pval = stats.ttest_ind(q_rets, nq_rets)
        lines.append(f"差异t检验: t={t_stat:.3f}, p={t_pval:.4f}")

    # d. 到期日当天
    lines.append("\n## d. 到期日当天现货表现\n")
    expiry_day_dates = set()
    for c in cycles:
        df = c["data"]
        t0 = df[df["T_minus"] == 0]
        if not t0.empty:
            expiry_day_dates.add(t0.iloc[0]["trade_date"])

    exp_day_rets = spot_df[spot_df["trade_date"].isin(expiry_day_dates)]["daily_ret"]
    non_exp_rets = spot_df[~spot_df["trade_date"].isin(expiry_day_dates)]["daily_ret"]

    if not exp_day_rets.empty:
        lines.append(f"到期日收益率均值: {exp_day_rets.mean():+.4f}%")
        lines.append(f"到期日下跌概率:   {(exp_day_rets < 0).mean()*100:.0f}%")
        lines.append(f"非到期日均值:     {non_exp_rets.mean():+.4f}%")
        lines.append(f"样本数: 到期日{len(exp_day_rets)} 非到期日{len(non_exp_rets)}")

    return "\n".join(lines)


# ======================================================================
# 研究4：当前周期定位
# ======================================================================

def research_current_cycle(cycles: List[Dict], db: DBManager) -> str:
    """当前周期定位分析"""
    lines = []
    lines.append("\n# 研究4：当前周期的定位\n")

    # IM2604 到期日 2026-04-17
    target_contract = "IM2604.CFX"
    delist_date = "20260417"
    today = "20260320"

    # 当前数据
    fut_row = db.query_df(
        f"SELECT close FROM futures_daily WHERE ts_code='{target_contract}' "
        f"ORDER BY trade_date DESC LIMIT 1"
    )
    spot_row = db.query_df(
        "SELECT close FROM index_daily WHERE ts_code='000852.SH' "
        "ORDER BY trade_date DESC LIMIT 1"
    )

    if fut_row is None or fut_row.empty:
        # Try IML1 as proxy
        fut_row = db.query_df(
            "SELECT close FROM futures_daily WHERE ts_code='IML1.CFX' "
            "ORDER BY trade_date DESC LIMIT 1"
        )

    if fut_row is not None and not fut_row.empty and spot_row is not None and not spot_row.empty:
        fut_price = float(fut_row["close"].iloc[0])
        spot_price = float(spot_row["close"].iloc[0])
        current_disc_pts = fut_price - spot_price
        current_disc_pct = current_disc_pts / spot_price * 100

        # 距到期天数
        t_today = pd.Timestamp(today)
        t_expiry = pd.Timestamp(delist_date)
        dte_cal = (t_expiry - t_today).days
        # 估算交易日
        dte_trade = int(dte_cal * 5 / 7)

        lines.append(f"## 当前持仓状态\n")
        lines.append(f"合约: {target_contract}")
        lines.append(f"到期日: {delist_date}")
        lines.append(f"今日: {today} (距到期约T-{dte_trade}交易日)")
        lines.append(f"期货价: {fut_price:.2f}")
        lines.append(f"现货价: {spot_price:.2f}")
        lines.append(f"当前贴水: {current_disc_pts:+.1f}点 ({current_disc_pct:+.2f}%)\n")

        # 历史同期百分位
        historical_disc_at_t = []
        for c in cycles:
            df = c["data"]
            nearby = df[(df["T_minus"] >= dte_trade - 2) & (df["T_minus"] <= dte_trade + 2)]
            if not nearby.empty:
                historical_disc_at_t.append(nearby["discount_pct"].mean())

        if historical_disc_at_t:
            hist_arr = np.array(historical_disc_at_t)
            percentile = np.mean(hist_arr <= current_disc_pct) * 100
            lines.append(f"## b. 历史同期(T-{dte_trade}±2)贴水率\n")
            lines.append(f"历史均值: {hist_arr.mean():+.2f}%")
            lines.append(f"历史中位: {np.median(hist_arr):+.2f}%")
            lines.append(f"当前贴水率 {current_disc_pct:+.2f}% 在历史百分位: {percentile:.0f}%")
            if current_disc_pct < hist_arr.mean():
                lines.append("→ 当前贴水偏深（低于历史均值）\n")
            else:
                lines.append("→ 当前贴水偏浅（高于历史均值）\n")

        # 预测收敛
        lines.append("## c. 收敛预测\n")
        convergence_t0 = []
        for c in cycles:
            df = c["data"]
            t0 = df[df["T_minus"] == 0]
            if not t0.empty:
                convergence_t0.append(float(t0.iloc[0]["discount_pct"]))

        if convergence_t0:
            avg_t0_disc = np.mean(convergence_t0)
            lines.append(f"到期日平均残余贴水: {avg_t0_disc:+.2f}%")
            expected_convergence = current_disc_pct - avg_t0_disc
            lines.append(f"预期收敛幅度: {expected_convergence:+.2f}pp")
            lines.append(f"对应约 {abs(expected_convergence / 100 * spot_price):.0f} 点\n")

        # T-10到T-0期间现货预期
        lines.append("## d. T-10到T-0期间预期 (约4月3日-4月17日)\n")
        spot_rets_t10 = []
        for c in cycles:
            df = c["data"]
            t10 = df[df["T_minus"] == 10]
            t0 = df[df["T_minus"] == 0]
            if not t10.empty and not t0.empty:
                s10 = float(t10.iloc[0]["spot_close"])
                s0 = float(t0.iloc[0]["spot_close"])
                spot_rets_t10.append((s0 - s10) / s10 * 100)

        if spot_rets_t10:
            arr = np.array(spot_rets_t10)
            lines.append(f"T-10到T-0现货历史均值收益: {arr.mean():+.3f}%")
            lines.append(f"下跌概率: {(arr < 0).mean()*100:.0f}%")
            lines.append(f"最大跌幅: {arr.min():+.2f}%")
            lines.append(f"最大涨幅: {arr.max():+.2f}%")

            if arr.mean() < -0.3:
                lines.append(f"\n→ 历史上到期前10天现货平均下跌{abs(arr.mean()):.2f}%，")
                lines.append(f"  对P-7800持仓有一定保护但也增加触及风险")
                lines.append(f"  7800距当前现货{spot_price:.0f}约{spot_price-7800:.0f}点 ({(spot_price-7800)/spot_price*100:.1f}%)")
            elif arr.mean() > 0.3:
                lines.append(f"\n→ 历史上到期前10天现货平均上涨{arr.mean():.2f}%")
            else:
                lines.append(f"\n→ 历史上到期前10天现货无明显方向偏移")

    else:
        lines.append("无法获取当前IM2604或现货数据\n")
        lines.append("（IM2604具体合约可能需要下载）\n")

    return "\n".join(lines)


# ======================================================================
# 综合结论
# ======================================================================

def generate_conclusions(cycles: List[Dict], db: DBManager) -> str:
    """生成综合结论"""
    lines = []
    lines.append("\n# 综合结论\n")

    if not cycles:
        lines.append("数据不足，无法得出结论\n")
        return "\n".join(lines)

    # 计算关键指标
    spot_down_ratios = []
    quarter_down = []
    non_quarter_down = []

    for c in cycles:
        df = c["data"]
        t10 = df[df["T_minus"] == 10]
        t0 = df[df["T_minus"] == 0]
        if not t10.empty and not t0.empty:
            s10 = float(t10.iloc[0]["spot_close"])
            s0 = float(t0.iloc[0]["spot_close"])
            ret = (s0 - s10) / s10 * 100
            spot_down_ratios.append(ret < 0)
            if c["is_quarter"]:
                quarter_down.append(ret)
            else:
                non_quarter_down.append(ret)

    if spot_down_ratios:
        down_pct = np.mean(spot_down_ratios) * 100
    else:
        down_pct = 50

    q_mean = np.mean(quarter_down) if quarter_down else 0
    nq_mean = np.mean(non_quarter_down) if non_quarter_down else 0

    lines.append(f"1. **到期前砸盘效应**: 到期前10天现货下跌概率={down_pct:.0f}%")
    if down_pct > 60:
        lines.append(f"   → 存在一定的到期前下跌倾向，但需结合统计检验判断显著性")
    else:
        lines.append(f"   → 下跌概率接近50%，到期前砸盘效应不明显")

    lines.append(f"\n2. **季月 vs 非季月**: 季月T-10到T-0现货均值={q_mean:+.2f}%, 非季月={nq_mean:+.2f}%")
    if q_mean < nq_mean - 0.3:
        lines.append(f"   → 季月效应更明显（量化基金集中到期效应）")
    else:
        lines.append(f"   → 季月和非季月差异不大")

    lines.append(f"\n3. **收敛方式**: 贴水收敛主要通过{'现货下跌' if down_pct > 55 else '期货上涨'}完成")

    lines.append(f"\n4. **对当前持仓的启示**:")
    lines.append(f"   - IM2604到期日 2026-04-17")
    lines.append(f"   - 关注4月上旬（T-10附近）现货走势")
    lines.append(f"   - P-7800距现货约安全距离，但到期前波动可能加大")

    return "\n".join(lines)


# ======================================================================
# 主函数
# ======================================================================

def main():
    print("=" * 60)
    print("  IM贴水到期收敛规律研究")
    print("=" * 60)
    print()

    db = get_db()

    # Step 1: 下载具体合约数据
    print(">>> Step 1: 下载IM具体合约日线数据...")
    contracts = download_im_specific_contracts(db)

    # Step 2: 构建到期周期
    print("\n>>> Step 2: 构建到期周期...")
    cycles = build_expiry_cycles(db, contracts)
    print(f"共构建 {len(cycles)} 个有效到期周期")

    quarter_count = sum(1 for c in cycles if c["is_quarter"])
    non_quarter_count = len(cycles) - quarter_count
    print(f"  季月: {quarter_count}个, 非季月: {non_quarter_count}个")

    if not cycles:
        print("ERROR: 无有效到期周期，退出")
        return

    # Step 3: 运行研究
    print("\n>>> Step 3: 运行研究...\n")
    report_parts = []

    report_parts.append(f"# IM贴水到期收敛规律研究")
    report_parts.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report_parts.append(f"> 数据范围: IM上市(2022-07-22)至今, {len(cycles)}个到期周期\n")

    r1 = research_convergence(cycles)
    print(r1)
    report_parts.append(r1)

    r2 = research_spot_behavior(cycles)
    print(r2)
    report_parts.append(r2)

    r3 = research_statistical_tests(cycles, db)
    print(r3)
    report_parts.append(r3)

    r4 = research_current_cycle(cycles, db)
    print(r4)
    report_parts.append(r4)

    conclusions = generate_conclusions(cycles, db)
    print(conclusions)
    report_parts.append(conclusions)

    # 保存报告
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"discount_convergence_{datetime.now().strftime('%Y%m%d')}.md"
    output_path.write_text("\n".join(report_parts), encoding="utf-8")
    print(f"\n报告已保存: {output_path}")


if __name__ == "__main__":
    main()
