"""
position_sizing_research.py
----------------------------
对比不同仓位管理方法的效果（基于日内信号回测数据）。

用法：
    python scripts/position_sizing_research.py --symbol IM --date 20260220-20260327
    python scripts/position_sizing_research.py --symbol IM --account 6400000 --risk-pct 0.005
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.config_loader import ConfigLoader
from data.storage.db_manager import DBManager
from scripts.backtest_signals_day import run_day

IM_MULT = 200
STOP_LOSS_PCT = 0.005  # 0.5% 止损


# ---------------------------------------------------------------------------
# 日期工具
# ---------------------------------------------------------------------------

def _get_dates(db: DBManager, sym: str, date_spec: str) -> list[str]:
    """解析日期参数（支持 YYYYMMDD-YYYYMMDD 范围或逗号分隔）。"""
    _SPOT = {"IM": "000852", "IF": "000300", "IH": "000016", "IC": "000905"}
    spot = _SPOT.get(sym, "000852")

    if "-" in date_spec and len(date_spec) == 17:
        start, end = date_spec.split("-")
        start_dash = f"{start[:4]}-{start[4:6]}-{start[6:]}"
        end_dash = f"{end[:4]}-{end[4:6]}-{end[6:]}"
        df = db.query_df(
            f"SELECT DISTINCT substr(datetime, 1, 10) as d "
            f"FROM index_min WHERE symbol='{spot}' AND period=300 "
            f"AND d >= '{start_dash}' AND d <= '{end_dash}' "
            f"ORDER BY d"
        )
    else:
        df = db.query_df(
            f"SELECT DISTINCT substr(datetime, 1, 10) as d "
            f"FROM index_min WHERE symbol='{spot}' AND period=300 "
            f"ORDER BY d DESC LIMIT 30"
        )
    if df is None or df.empty:
        return []
    dates = sorted([d.replace("-", "") for d in df["d"].tolist()])
    return dates


# ---------------------------------------------------------------------------
# 波动率计算
# ---------------------------------------------------------------------------

def _calc_daily_vol(db: DBManager) -> float:
    """从 index_daily 计算 ATR(20) 基础的年化波动率。"""
    df = db.query_df(
        "SELECT close FROM index_daily WHERE ts_code='000852.SH' "
        "ORDER BY trade_date DESC LIMIT 25"
    )
    if df is None or len(df) < 20:
        return 25.0  # fallback
    closes = df["close"].astype(float).values[::-1]
    log_rets = np.diff(np.log(closes))
    return float(np.std(log_rets[-20:], ddof=1) * np.sqrt(252) * 100)


# ---------------------------------------------------------------------------
# Sizing 方法
# ---------------------------------------------------------------------------

def _size_fixed(trade: dict, **kw) -> float:
    """固定1手。"""
    return 1.0


def _size_fixed_risk(trade: dict, account: float, risk_pct: float, **kw) -> float:
    """固定风险金额。"""
    max_loss = account * risk_pct
    entry_p = trade["entry_price"]
    risk_per_lot = entry_p * IM_MULT * STOP_LOSS_PCT
    if risk_per_lot <= 0:
        return 1.0
    lots = max_loss / risk_per_lot
    return max(1.0, float(int(lots)))


def _size_half_kelly(trade: dict, account: float, kelly_f: float, **kw) -> float:
    """Half Kelly（max 10手）。"""
    if kelly_f <= 0:
        return 1.0
    half_k = kelly_f / 2
    entry_p = trade["entry_price"]
    risk_per_lot = entry_p * IM_MULT * STOP_LOSS_PCT
    if risk_per_lot <= 0:
        return 1.0
    lots = account * half_k / risk_per_lot
    return max(1.0, min(10.0, float(int(lots))))


def _size_vol_adjusted(trade: dict, current_vol: float, target_vol: float, **kw) -> float:
    """波动率调整。"""
    if current_vol <= 0:
        return 1.0
    ratio = target_vol / current_vol
    lots = max(1.0, ratio)
    return float(int(lots))


def _size_score_weighted(trade: dict, **kw) -> float:
    """信号强度加权。"""
    score = trade.get("entry_score", 60)
    if score >= 80:
        return 2.0
    return 1.0


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

def _evaluate(
    trades: list[dict],
    sizing_fn,
    account: float,
    dates: list[str],
    **kw,
) -> dict:
    """用 sizing 方法评估交易序列。"""
    if not trades:
        return {"total_pnl": 0, "max_dd": 0, "max_dd_pct": 0,
                "sharpe": 0, "calmar": 0, "avg_lots": 0, "max_loss": 0}

    # 按日期分组计算每日PnL
    daily_pnl: dict[str, float] = {d: 0.0 for d in dates}
    all_lots = []
    all_trade_pnl = []

    for t in trades:
        lots = sizing_fn(t, **kw)
        all_lots.append(lots)
        pnl_yuan = t["pnl_pts"] * IM_MULT * lots
        all_trade_pnl.append(pnl_yuan)

        # 归属到日期
        entry_date = None
        for d in dates:
            dd = f"{d[4:6]}:{d[6:]}"
            # entry_time 是 "HH:MM" 北京时间
            # 简化：按交易顺序归属
            pass
        # 用 entry_time 推算日期（trades 是按日期顺序的）
        # 简单方法：按顺序累加

    # 重新按日期跑一次（更准确）
    daily_pnl_list = []
    trade_idx = 0
    for d in dates:
        day_pnl = 0.0
        while trade_idx < len(trades):
            t = trades[trade_idx]
            # 检查这笔交易是否属于这一天
            # trades 是按日期顺序排列的
            lots = sizing_fn(t, **kw)
            day_pnl += t["pnl_pts"] * IM_MULT * lots
            trade_idx += 1
            # 看下一笔是否还是同一天（简化：我们按run_day的顺序）
            if trade_idx < len(trades):
                # 检查是否跨天了
                next_t = trades[trade_idx]
                if next_t.get("_date") != t.get("_date"):
                    break
        daily_pnl_list.append(day_pnl)

    # 重算：给每笔交易标记日期
    total_pnl = sum(t["pnl_pts"] * IM_MULT * sizing_fn(t, **kw) for t in trades)

    # 最大单笔亏损
    trade_pnls = [t["pnl_pts"] * IM_MULT * sizing_fn(t, **kw) for t in trades]
    max_loss = min(trade_pnls) if trade_pnls else 0

    # 计算日频统计
    daily_vals = np.array(daily_pnl_list) if daily_pnl_list else np.array([0.0])

    # 最大回撤
    cumsum = np.cumsum(daily_vals)
    running_max = np.maximum.accumulate(cumsum)
    drawdowns = cumsum - running_max
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0
    max_dd_pct = abs(max_dd) / account * 100 if account > 0 else 0

    # Sharpe (日频，年化)
    if len(daily_vals) > 1 and np.std(daily_vals) > 0:
        sharpe = float(np.mean(daily_vals) / np.std(daily_vals) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Calmar
    annual_ret = total_pnl / account * (252 / max(len(dates), 1))
    calmar = abs(annual_ret / (max_dd_pct / 100)) if max_dd_pct > 0 else 0

    avg_lots = float(np.mean([sizing_fn(t, **kw) for t in trades])) if trades else 0

    return {
        "total_pnl": total_pnl,
        "max_dd": max_dd,
        "max_dd_pct": max_dd_pct,
        "sharpe": sharpe,
        "calmar": calmar,
        "avg_lots": avg_lots,
        "max_loss": max_loss,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run(sym: str, date_spec: str, account: float, risk_pct: float, target_vol: float):
    db = DBManager(ConfigLoader().get_db_path())
    dates = _get_dates(db, sym, date_spec)
    if not dates:
        print("没有可用数据")
        return

    print(f"  加载 {sym} | {dates[0]}~{dates[-1]} | {len(dates)} 天")

    # 收集所有交易（标记日期）
    all_trades: list[dict] = []
    daily_trade_groups: dict[str, list] = {}
    for td in dates:
        trades = run_day(sym, td, db, verbose=False)
        completed = [t for t in trades if not t.get("partial")]
        for t in completed:
            t["_date"] = td
        all_trades.extend(completed)
        daily_trade_groups[td] = completed

    if not all_trades:
        print("  无交易")
        return

    # Kelly 参数
    pts = [t["pnl_pts"] for t in all_trades]
    wins = [p for p in pts if p > 0]
    losses = [p for p in pts if p < 0]
    wr = len(wins) / len(pts) if pts else 0
    avg_win = float(np.mean(wins)) if wins else 0
    avg_loss = abs(float(np.mean(losses))) if losses else 1
    payoff = avg_win / avg_loss if avg_loss > 0 else 1
    kelly_f = wr - (1 - wr) / payoff if payoff > 0 else 0

    # 当前波动率
    current_vol = _calc_daily_vol(db)

    print(f"  交易: {len(all_trades)} 笔 | WR={wr*100:.1f}%"
          f" | 均盈={avg_win:.1f}pt 均亏={avg_loss:.1f}pt"
          f" | Kelly={kelly_f*100:.1f}% | Vol={current_vol:.1f}%")

    # 按日期重组trades列表（保持日期顺序，用于日频统计）
    ordered_trades = []
    for td in dates:
        ordered_trades.extend(daily_trade_groups.get(td, []))

    # 评估各方法
    methods = [
        ("Fixed 1 lot", _size_fixed, {}),
        (f"Fixed risk {risk_pct*100:.1f}%", _size_fixed_risk,
         {"account": account, "risk_pct": risk_pct}),
        ("Half Kelly", _size_half_kelly,
         {"account": account, "kelly_f": kelly_f}),
        (f"Vol adj {target_vol:.0f}%", _size_vol_adjusted,
         {"current_vol": current_vol, "target_vol": target_vol}),
        ("Score weighted", _size_score_weighted, {}),
    ]

    # 计算每种方法的日PnL序列
    results = []
    for name, fn, kw_extra in methods:
        # 计算日PnL序列
        daily_pnls = []
        for td in dates:
            day_trades = daily_trade_groups.get(td, [])
            day_pnl = sum(
                t["pnl_pts"] * IM_MULT * fn(t, **kw_extra)
                for t in day_trades
            )
            daily_pnls.append(day_pnl)

        daily_arr = np.array(daily_pnls)
        total_pnl = float(np.sum(daily_arr))

        # 最大回撤
        cumsum = np.cumsum(daily_arr)
        running_max = np.maximum.accumulate(cumsum)
        drawdowns = cumsum - running_max
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0
        max_dd_pct = abs(max_dd) / account * 100

        # Sharpe
        if len(daily_arr) > 1 and np.std(daily_arr) > 0:
            sharpe = float(np.mean(daily_arr) / np.std(daily_arr) * np.sqrt(252))
        else:
            sharpe = 0.0

        # Calmar
        n_days = len(dates)
        annual_ret_pct = total_pnl / account * (252 / n_days) * 100
        calmar = annual_ret_pct / max_dd_pct if max_dd_pct > 0 else 0

        # 平均手数
        lots_list = [fn(t, **kw_extra) for t in all_trades]
        avg_lots = float(np.mean(lots_list)) if lots_list else 0

        # 最大单笔亏损
        trade_pnls = [t["pnl_pts"] * IM_MULT * fn(t, **kw_extra) for t in all_trades]
        max_loss = min(trade_pnls) if trade_pnls else 0

        results.append({
            "name": name,
            "total_pnl": total_pnl,
            "max_dd": max_dd,
            "max_dd_pct": max_dd_pct,
            "sharpe": sharpe,
            "calmar": calmar,
            "avg_lots": avg_lots,
            "max_loss": max_loss,
        })

    # 输出
    print(f"\n{'=' * 100}")
    print(f" Position Sizing Research | {sym} | {len(dates)} days"
          f" | Account={account/10000:.0f}万")
    print(f"{'=' * 100}")
    print(f" {'Method':<20s} | {'Total PnL':>10s} | {'MaxDD':>9s}"
          f" | {'MaxDD%':>6s} | {'Sharpe':>6s} | {'Calmar':>6s}"
          f" | {'AvgLots':>7s} | {'MaxLoss':>9s}")
    print(f" {'-'*20}-+-{'-'*10}-+-{'-'*9}-+-{'-'*6}-+-{'-'*6}"
          f"-+-{'-'*6}-+-{'-'*7}-+-{'-'*9}")

    for r in results:
        tag = "⭐" if r["sharpe"] == max(x["sharpe"] for x in results) else "  "
        print(f" {r['name']:<20s} | {r['total_pnl']:>+10,.0f}"
              f" | {r['max_dd']:>+9,.0f} | {r['max_dd_pct']:>5.2f}%"
              f" | {r['sharpe']:>6.2f} | {r['calmar']:>6.1f}"
              f" | {r['avg_lots']:>7.1f} | {r['max_loss']:>+9,.0f} {tag}")

    print(f"\n Kelly parameters: WR={wr*100:.1f}%"
          f"  avg_win={avg_win:.1f}pt  avg_loss={avg_loss:.1f}pt"
          f"  payoff={payoff:.2f}  kelly_f={kelly_f*100:.1f}%"
          f"  half_kelly={kelly_f/2*100:.1f}%")
    print(f" Current Vol: {current_vol:.1f}%  Target Vol: {target_vol:.1f}%"
          f"  Vol ratio: {target_vol/current_vol:.2f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Position Sizing Research")
    parser.add_argument("--symbol", default="IM")
    parser.add_argument("--date", default="20260220-20260327",
                        help="YYYYMMDD-YYYYMMDD range")
    parser.add_argument("--account", type=float, default=6_400_000,
                        help="Account size in yuan (default 6.4M)")
    parser.add_argument("--risk-pct", type=float, default=0.005,
                        help="Fixed risk per trade (default 0.5%%)")
    parser.add_argument("--target-vol", type=float, default=15.0,
                        help="Target annualized vol (default 15%%)")
    args = parser.parse_args()
    run(args.symbol, args.date, args.account, args.risk_pct, args.target_vol)


if __name__ == "__main__":
    main()
