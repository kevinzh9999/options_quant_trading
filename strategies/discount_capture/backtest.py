"""
backtest.py
-----------
IM 贴水捕获策略历史回测模块。

回测方法论：
  1. 对历史每个月的第一个交易日，检查 IML2（当季）贴水率
  2. 如果年化贴水率 > 阈值（默认 8%），建立多头期货仓位
  3. 持有至到期前 5 日，平仓
  4. 计算 P&L：(exit_price - entry_price) * 200 * lots
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CONTRACT_MULT = 200


class DiscountBacktest:
    """
    贴水捕获策略回测。

    Parameters
    ----------
    db_manager : DBManager
        数据库管理器
    """

    def __init__(self, db_manager):
        self.db = db_manager

    def run(
        self,
        contract_type: str = "IML2",
        min_discount_rate: float = 0.08,
        start_date: str = "20220722",
        initial_capital: float = 1_000_000.0,
        futures_lots: int = 1,
    ) -> dict:
        """
        运行回测。

        Parameters
        ----------
        contract_type : str
            连续合约类型（IML1 或 IML2）
        min_discount_rate : float
            最低年化贴水率入场阈值
        start_date : str
            回测起始日期，格式 YYYYMMDD
        initial_capital : float
            初始资金（元）
        futures_lots : int
            每次建仓手数

        Returns
        -------
        dict
            trades, total_return, annualized_return, max_drawdown, sharpe, win_rate, equity_curve
        """
        from strategies.discount_capture.signal import DiscountSignal

        signal_gen = DiscountSignal(self.db)

        # 获取历史数据
        ts_code = f"{contract_type}.CFX" if not contract_type.endswith(".CFX") else contract_type

        try:
            df_hist = self.db.query_df(
                f"SELECT trade_date, close "
                f"FROM futures_daily "
                f"WHERE ts_code='{ts_code}' "
                f"AND trade_date >= '{start_date}' "
                f"ORDER BY trade_date ASC"
            )

            df_spot = self.db.query_df(
                f"SELECT trade_date, close as spot "
                f"FROM futures_daily "
                f"WHERE ts_code='IM.CFX' "
                f"AND trade_date >= '{start_date}' "
                f"ORDER BY trade_date ASC"
            )
        except Exception as e:
            logger.error("回测数据加载失败: %s", e)
            return self._empty_result()

        if df_hist is None or df_hist.empty:
            logger.warning("无 %s 历史数据", ts_code)
            return self._empty_result()

        if df_spot is None or df_spot.empty:
            logger.warning("无 IM.CFX 历史数据")
            return self._empty_result()

        # 合并
        df = pd.merge(
            df_hist.rename(columns={"close": "futures_close"}),
            df_spot,
            on="trade_date",
            how="inner",
        ).sort_values("trade_date").reset_index(drop=True)

        if len(df) < 20:
            return self._empty_result()

        # 计算每日贴水率
        df["discount"] = df["futures_close"] - df["spot"]
        df["discount_rate"] = df["discount"] / df["spot"]

        # 月度采样：每月第一个交易日
        df["ym"] = df["trade_date"].str[:6]
        monthly_first = df.groupby("ym").first().reset_index()

        # 回测逻辑
        trades = []
        capital = initial_capital
        equity_curve_rows = []

        i = 0
        while i < len(monthly_first):
            entry_row = monthly_first.iloc[i]
            entry_date = str(entry_row["trade_date"])
            entry_futures = float(entry_row["futures_close"])
            entry_spot = float(entry_row["spot"])

            # 检查贴水率
            disc = entry_futures - entry_spot
            if entry_spot > 0:
                T_approx = 45 / 365.0  # 近似剩余天数
                ann_rate = abs(disc) / entry_spot * (365 / 45)
            else:
                ann_rate = 0.0

            if disc >= 0 or ann_rate < min_discount_rate:
                i += 1
                continue  # 不满足条件，跳过

            # 寻找出场日：约45天后（或当月末）
            ym = str(entry_row["ym"])
            next_ym = _next_ym(ym)

            # 在下个月初前5个交易日出场
            exit_candidates = df[
                (df["trade_date"] > entry_date) &
                (df["ym"] <= next_ym)
            ].head(45)

            if exit_candidates.empty:
                i += 1
                continue

            # 取最后一行（最近的出场机会）
            exit_row = exit_candidates.iloc[-1]
            exit_date = str(exit_row["trade_date"])
            exit_futures = float(exit_row["futures_close"])

            # P&L
            pnl = (exit_futures - entry_futures) * CONTRACT_MULT * futures_lots
            pnl_pct = pnl / capital

            capital += pnl
            trades.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_futures,
                "exit_price": exit_futures,
                "entry_spot": entry_spot,
                "annualized_discount_rate": ann_rate,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "capital": capital,
            })

            equity_curve_rows.append({"date": exit_date, "capital": capital})
            i += 1

        if not trades:
            return self._empty_result()

        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve_rows).set_index("date")

        # 绩效统计
        total_return = (capital - initial_capital) / initial_capital
        n_days = (
            pd.Timestamp(trades_df["exit_date"].iloc[-1]) -
            pd.Timestamp(trades_df["entry_date"].iloc[0])
        ).days
        if n_days > 0:
            ann_return = (1 + total_return) ** (365 / n_days) - 1
        else:
            ann_return = 0.0

        # 最大回撤
        peak = initial_capital
        max_dd = 0.0
        for cap in trades_df["capital"].values:
            if cap > peak:
                peak = cap
            dd = (peak - cap) / peak
            if dd > max_dd:
                max_dd = dd

        # Sharpe
        returns = trades_df["pnl_pct"].values
        if len(returns) > 1 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(12)  # 月度调整
        else:
            sharpe = 0.0

        win_rate = float((trades_df["pnl"] > 0).mean())

        return {
            "trades": trades_df,
            "total_return": total_return,
            "annualized_return": ann_return,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "equity_curve": equity_df,
            "n_trades": len(trades_df),
        }

    def print_report(self, results: dict) -> None:
        """打印格式化的回测报告。"""
        sep = "─" * 60
        print()
        print(sep)
        print("  IM 贴水捕获策略  |  回测报告")
        print(sep)

        if not results or "trades" not in results or results["trades"].empty:
            print("  无有效回测结果（可能数据不足或无满足条件的信号）")
            print(sep)
            return

        print(f"  交易笔数    : {results['n_trades']}")
        print(f"  总收益率    : {results['total_return']*100:>+8.2f}%")
        print(f"  年化收益率  : {results['annualized_return']*100:>+8.2f}%")
        print(f"  最大回撤    : {results['max_drawdown']*100:>8.2f}%")
        print(f"  Sharpe 比率 : {results['sharpe']:>8.3f}")
        print(f"  胜率        : {results['win_rate']*100:>8.1f}%")
        print()

        trades = results["trades"]
        print("  近期交易记录 (最新5笔):")
        print(f"  {'入场日期':<12} {'出场日期':<12} {'入场价':>8} {'出场价':>8} "
              f"{'贴水率':>8} {'P&L':>10}")
        print(f"  {'─'*60}")
        for _, row in trades.tail(5).iterrows():
            print(
                f"  {row['entry_date']:<12} {row['exit_date']:<12}"
                f"  {row['entry_price']:>8.0f}  {row['exit_price']:>8.0f}"
                f"  {row['annualized_discount_rate']*100:>7.1f}%"
                f"  {row['pnl']:>+10,.0f}"
            )
        print(sep)
        print()

    def _empty_result(self) -> dict:
        return {
            "trades": pd.DataFrame(),
            "total_return": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "equity_curve": pd.DataFrame(),
            "n_trades": 0,
        }


def _next_ym(ym: str) -> str:
    """返回下一个月的 YYYYMM 字符串。"""
    y = int(ym[:4])
    m = int(ym[4:])
    m += 1
    if m > 12:
        m = 1
        y += 1
    return f"{y}{m:02d}"
