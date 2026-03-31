"""
report.py
---------
回测绩效报告。

根据逐日账户状态和成交记录计算收益指标，输出格式化报告。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.broker import AccountState, Trade

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
RF_ANNUAL = 0.02  # 2% annual risk-free rate


class BacktestReport:
    """
    Backtest performance report.

    Parameters
    ----------
    daily_states : list[AccountState]
        One AccountState per trading day.
    trades : list[Trade]
        All executed trades.
    initial_capital : float
        Starting capital.
    benchmark_returns : pd.Series, optional
        Daily returns of benchmark index.
    strategy_name : str
        Display name for the strategy.
    """

    def __init__(
        self,
        daily_states: List[AccountState],
        trades: List[Trade],
        initial_capital: float,
        benchmark_returns: Optional[pd.Series] = None,
        strategy_name: str = "",
    ) -> None:
        self.daily_states = daily_states
        self.trades = trades
        self.initial_capital = initial_capital
        self.benchmark_returns = benchmark_returns
        self.strategy_name = strategy_name
        self._metrics: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def calculate_metrics(self) -> Dict:
        """
        Compute and return all performance metrics.

        Returns dict with:
            total_return, annualized_return, max_drawdown, max_drawdown_duration,
            sharpe_ratio, sortino_ratio, calmar_ratio, win_rate, profit_loss_ratio,
            total_trades, avg_holding_days, total_commission,
            avg_monthly_return, monthly_win_rate, best_month, worst_month
        """
        if not self.daily_states:
            return {}

        equity = self.get_equity_curve()
        if equity.empty:
            return {}

        balances = equity["balance"]
        daily_returns = equity["daily_return"].dropna()

        # ---- Return metrics ----
        total_return = (balances.iloc[-1] - self.initial_capital) / self.initial_capital

        n_days = len(balances)
        years = n_days / TRADING_DAYS_PER_YEAR
        annualized_return = (1 + total_return) ** (1 / max(years, 1e-9)) - 1

        # ---- Drawdown ----
        drawdown_series = equity["drawdown"]
        max_drawdown = float(drawdown_series.min())

        # Max drawdown duration: longest streak where balance < rolling peak
        is_in_dd = drawdown_series < 0
        max_dd_dur = 0
        cur_dur = 0
        for in_dd in is_in_dd:
            if in_dd:
                cur_dur += 1
                max_dd_dur = max(max_dd_dur, cur_dur)
            else:
                cur_dur = 0

        # ---- Sharpe ----
        rf_daily = RF_ANNUAL / TRADING_DAYS_PER_YEAR
        excess = daily_returns - rf_daily
        std_daily = float(daily_returns.std())
        sharpe = float(excess.mean() / std_daily * np.sqrt(TRADING_DAYS_PER_YEAR)) if std_daily > 0 else 0.0

        # ---- Sortino ----
        downside = daily_returns[daily_returns < rf_daily] - rf_daily
        downside_std = float(np.sqrt((downside ** 2).mean())) if len(downside) > 0 else 0.0
        sortino = float(excess.mean() / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR)) if downside_std > 0 else 0.0

        # ---- Calmar ----
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

        # ---- Trade stats ----
        total_trades = len(self.trades)
        total_commission = sum(t.commission for t in self.trades)

        # Win rate and profit/loss ratio from paired trades
        trade_summary = self.get_trade_summary()
        win_rate = 0.0
        profit_loss_ratio = 0.0
        avg_holding_days = 0.0

        if not trade_summary.empty and "pnl" in trade_summary.columns:
            pnl_series = trade_summary["pnl"].dropna()
            if len(pnl_series) > 0:
                winners = pnl_series[pnl_series > 0]
                losers = pnl_series[pnl_series < 0]
                win_rate = len(winners) / len(pnl_series)
                avg_win = float(winners.mean()) if len(winners) > 0 else 0.0
                avg_loss = float(losers.abs().mean()) if len(losers) > 0 else 0.0
                profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

            if "holding_days" in trade_summary.columns:
                avg_holding_days = float(trade_summary["holding_days"].mean())

        # ---- Monthly stats ----
        monthly_df = self.get_monthly_returns()
        monthly_win_rate = 0.0
        avg_monthly_return = 0.0
        best_month = 0.0
        worst_month = 0.0

        if not monthly_df.empty:
            # Flatten monthly values (exclude Annual column)
            month_cols = [c for c in monthly_df.columns if c != "Annual"]
            month_vals = monthly_df[month_cols].values.flatten()
            month_vals = month_vals[~np.isnan(month_vals)]
            if len(month_vals) > 0:
                avg_monthly_return = float(np.mean(month_vals))
                monthly_win_rate = float(np.mean(month_vals > 0))
                best_month = float(np.max(month_vals))
                worst_month = float(np.min(month_vals))

        metrics = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "max_drawdown": max_drawdown,
            "max_drawdown_duration": max_dd_dur,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "total_trades": total_trades,
            "avg_holding_days": avg_holding_days,
            "total_commission": total_commission,
            "avg_monthly_return": avg_monthly_return,
            "monthly_win_rate": monthly_win_rate,
            "best_month": best_month,
            "worst_month": worst_month,
        }
        self._metrics = metrics
        return metrics

    # ------------------------------------------------------------------
    # Equity curve
    # ------------------------------------------------------------------

    def get_equity_curve(self) -> pd.DataFrame:
        """
        Build equity curve DataFrame.

        Returns
        -------
        pd.DataFrame with columns:
            trade_date, balance, daily_return, cumulative_return, drawdown
        """
        if not self.daily_states:
            return pd.DataFrame()

        rows = [
            {"trade_date": s.trade_date, "balance": s.balance}
            for s in self.daily_states
        ]
        df = pd.DataFrame(rows)
        df = df.sort_values("trade_date").reset_index(drop=True)

        df["daily_return"] = df["balance"].pct_change()
        df["cumulative_return"] = df["balance"] / self.initial_capital - 1

        # Drawdown: (balance - rolling_peak) / rolling_peak
        rolling_peak = df["balance"].cummax()
        df["drawdown"] = (df["balance"] - rolling_peak) / rolling_peak

        return df

    # ------------------------------------------------------------------
    # Monthly returns
    # ------------------------------------------------------------------

    def get_monthly_returns(self) -> pd.DataFrame:
        """
        Return pivot table of monthly returns.

        Rows = year, columns = month (1-12) + Annual.
        Values are monthly return fractions (e.g. 0.05 = 5%).
        """
        equity = self.get_equity_curve()
        if equity.empty:
            return pd.DataFrame()

        df = equity[["trade_date", "balance"]].copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df = df.set_index("trade_date")

        # Resample to month-end
        monthly = df["balance"].resample("ME").last()

        # Monthly returns
        monthly_ret = monthly.pct_change().dropna()

        pivot = pd.DataFrame(index=sorted(monthly_ret.index.year.unique()))
        pivot.index.name = "Year"

        for dt, ret in monthly_ret.items():
            year = dt.year
            month = dt.month
            pivot.loc[year, month] = ret

        # Annual column
        for year in pivot.index:
            year_vals = pivot.loc[year, [c for c in pivot.columns if isinstance(c, int)]].dropna()
            if len(year_vals) > 0:
                pivot.loc[year, "Annual"] = float(np.prod(1 + year_vals) - 1)
            else:
                pivot.loc[year, "Annual"] = np.nan

        # Ensure columns 1-12 are present
        for m in range(1, 13):
            if m not in pivot.columns:
                pivot[m] = np.nan

        col_order = list(range(1, 13)) + ["Annual"]
        pivot = pivot[[c for c in col_order if c in pivot.columns]]
        return pivot

    # ------------------------------------------------------------------
    # Trade summary (paired OPEN/CLOSE)
    # ------------------------------------------------------------------

    def get_trade_summary(self) -> pd.DataFrame:
        """
        Pair OPEN trades with CLOSE trades by symbol.

        Returns DataFrame with:
            symbol, open_date, close_date, direction, volume,
            entry_price, exit_price, pnl, commission, holding_days
        """
        if not self.trades:
            return pd.DataFrame()

        opens: Dict[str, List[Trade]] = {}
        results = []

        for trade in self.trades:
            sym = trade.symbol
            if trade.offset == "OPEN":
                opens.setdefault(sym, []).append(trade)
            elif trade.offset == "CLOSE":
                open_queue = opens.get(sym, [])
                if open_queue:
                    open_trade = open_queue.pop(0)

                    # Determine direction from open trade
                    if open_trade.direction == "BUY":
                        pos_dir = "LONG"
                        mult_sign = 1
                    else:
                        pos_dir = "SHORT"
                        mult_sign = -1

                    # Estimate PnL (without knowing multiplier here, use price diff)
                    price_diff = trade.price - open_trade.price
                    pnl_per_unit = price_diff * mult_sign * trade.volume
                    total_commission = open_trade.commission + trade.commission

                    try:
                        open_dt = pd.to_datetime(open_trade.trade_date, format="%Y%m%d")
                        close_dt = pd.to_datetime(trade.trade_date, format="%Y%m%d")
                        holding_days = (close_dt - open_dt).days
                    except Exception:
                        holding_days = 0

                    results.append({
                        "symbol": sym,
                        "open_date": open_trade.trade_date,
                        "close_date": trade.trade_date,
                        "direction": pos_dir,
                        "volume": trade.volume,
                        "entry_price": open_trade.price,
                        "exit_price": trade.price,
                        "pnl": pnl_per_unit - total_commission,
                        "commission": total_commission,
                        "holding_days": holding_days,
                    })

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        """Print a formatted terminal report."""
        metrics = self._metrics or self.calculate_metrics()

        if not self.daily_states:
            print("No data to report.")
            return

        start_date = self.daily_states[0].trade_date
        end_date = self.daily_states[-1].trade_date

        # Format date for display: YYYYMMDD → YYYY-MM-DD
        def fmt_date(d: str) -> str:
            if len(d) == 8:
                return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            return d

        start_str = fmt_date(start_date)
        end_str = fmt_date(end_date)
        name_str = self.strategy_name or "Unknown"

        sep = "═" * 62
        print(f"\n{sep}")
        print(f" 回测报告 | 策略: {name_str} | {start_str} ~ {end_str}")
        print(f"{sep}")

        def pct(v: float) -> str:
            sign = "+" if v >= 0 else ""
            return f"{sign}{v*100:.1f}%"

        def ratio(v: float) -> str:
            return f"{v:.2f}"

        benchmark_str = "N/A"
        if self.benchmark_returns is not None and not self.benchmark_returns.empty:
            bm_total = float((1 + self.benchmark_returns).prod() - 1)
            benchmark_str = pct(bm_total)

        print("【收益】")
        print(f"  累计收益率    : {pct(metrics.get('total_return', 0)):>10}")
        print(f"  年化收益率    : {pct(metrics.get('annualized_return', 0)):>10}")
        print(f"  基准收益率    : {benchmark_str:>10}")
        print(f"  最佳月份      : {pct(metrics.get('best_month', 0)):>10}")
        print(f"  最差月份      : {pct(metrics.get('worst_month', 0)):>10}")
        print(f"  平均月收益    : {pct(metrics.get('avg_monthly_return', 0)):>10}")
        print()
        print("【风险】")
        print(f"  最大回撤      : {pct(metrics.get('max_drawdown', 0)):>10}")
        print(f"  最大回撤时长  : {metrics.get('max_drawdown_duration', 0):>9}天")
        print(f"  年化波动率    : ", end="")
        equity = self.get_equity_curve()
        if not equity.empty:
            ann_vol = float(equity["daily_return"].std() * np.sqrt(TRADING_DAYS_PER_YEAR))
            print(f"{pct(ann_vol):>10}")
        else:
            print("       N/A")
        print()
        print("【绩效】")
        print(f"  夏普比率      : {ratio(metrics.get('sharpe_ratio', 0)):>10}")
        print(f"  索提诺比率    : {ratio(metrics.get('sortino_ratio', 0)):>10}")
        print(f"  卡玛比率      : {ratio(metrics.get('calmar_ratio', 0)):>10}")
        print(f"  月胜率        : {pct(metrics.get('monthly_win_rate', 0)):>10}")
        print()
        print("【交易】")
        print(f"  总交易次数    : {metrics.get('total_trades', 0):>10}")
        print(f"  胜率          : {pct(metrics.get('win_rate', 0)):>10}")
        print(f"  盈亏比        : {ratio(metrics.get('profit_loss_ratio', 0)):>10}")
        print(f"  平均持仓天数  : {metrics.get('avg_holding_days', 0):>9.1f}天")
        print(f"  总手续费      : {metrics.get('total_commission', 0):>10,.0f}元")
        print()
        print(f"  初始资金      : {self.initial_capital:>10,.0f}元")
        if self.daily_states:
            final_balance = self.daily_states[-1].balance
            print(f"  期末净值      : {final_balance:>10,.0f}元")
        print(f"{sep}\n")

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def save_to_csv(self, output_dir: str) -> None:
        """
        Save equity curve, monthly returns, and trades to CSV files.

        Parameters
        ----------
        output_dir : str
            Directory path to save files.
        """
        os.makedirs(output_dir, exist_ok=True)

        equity = self.get_equity_curve()
        if not equity.empty:
            equity.to_csv(os.path.join(output_dir, "equity_curve.csv"), index=False)
            logger.info("Saved equity_curve.csv to %s", output_dir)

        monthly = self.get_monthly_returns()
        if not monthly.empty:
            monthly.to_csv(os.path.join(output_dir, "monthly_returns.csv"))
            logger.info("Saved monthly_returns.csv to %s", output_dir)

        trade_summary = self.get_trade_summary()
        if not trade_summary.empty:
            trade_summary.to_csv(os.path.join(output_dir, "trades.csv"), index=False)
            logger.info("Saved trades.csv to %s", output_dir)
