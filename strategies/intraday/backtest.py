"""
backtest.py
-----------
日内策略回测。

用法：
    python -m strategies.intraday.backtest --start 20250509 --end 20260318
    python -m strategies.intraday.backtest --period 15m --start 20230815

数据来源：futures_min 表（5分钟线 或 15分钟线）
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from strategies.intraday.strategy import IntradayStrategy, IntradayConfig
from strategies.intraday.position import CONTRACT_MULTIPLIERS
from strategies.intraday.A_share_momentum_signal_v2 import (
    SignalGeneratorV2, SignalGeneratorV3,
)

DB_PATH = os.path.join(ROOT, "data", "storage", "trading.db")

# ---------------------------------------------------------------------------
# 手续费计算
# ---------------------------------------------------------------------------

COMMISSION_OPEN = 0.000023
COMMISSION_INTRADAY_CLOSE = 0.000345
COMMISSION_OVERNIGHT_CLOSE = 0.000023


def _commission(symbol: str, price: float, action: str) -> float:
    """
    计算单手手续费。
    action: "open" / "intraday_close" / "overnight_close" / "lock_open" / "unlock_close"
    """
    mult = CONTRACT_MULTIPLIERS.get(symbol, 300)
    notional = price * mult
    if action in ("open", "lock_open"):
        return notional * COMMISSION_OPEN
    elif action == "intraday_close":
        return notional * COMMISSION_INTRADAY_CLOSE
    else:  # overnight_close, unlock_close
        return notional * COMMISSION_OVERNIGHT_CLOSE


# ---------------------------------------------------------------------------
# 回测器
# ---------------------------------------------------------------------------

class IntradayBacktester:
    """日内策略回测器。按K线逐根推进。"""

    def __init__(
        self,
        db_path: str = DB_PATH,
        symbols: List[str] | None = None,
        period: int = 300,
        initial_capital: float = 5_000_000,
        config: IntradayConfig | None = None,
        signal_version: str = "v1",
    ):
        self.db_path = db_path
        self.symbols = symbols or ["IF", "IH", "IM"]
        self.period = period
        self.initial_capital = initial_capital
        self.signal_version = signal_version
        self.config = config or IntradayConfig(universe=self.symbols)

        # 如果用15m线，调整开盘区间根数
        if period == 900:
            self.config.opening_range_minutes = 30  # still 30 min
            # 2 bars of 15m = 30 minutes

        self.strategy = IntradayStrategy(self.config)
        if period == 900:
            self.strategy.signal_gen._opening_bars = 2

        # v2/v3: 替换信号生成器
        if signal_version == "v2":
            self.strategy.signal_gen = SignalGeneratorV2({
                "min_signal_score": self.config.min_signal_score,
            })
        elif signal_version == "v3":
            self.strategy.signal_gen = SignalGeneratorV3({
                "min_signal_score": self.config.min_signal_score,
            })

        self.capital = initial_capital
        self.equity_curve: List[Dict] = []
        self.trade_log: List[Dict] = []
        self.daily_stats: List[Dict] = []
        self.total_commission: float = 0.0
        self.lock_count: int = 0
        self.direct_close_count: int = 0
        self.commission_saved: float = 0.0

        # Data storage
        self._min_data: Dict[str, pd.DataFrame] = {}
        self._min_15m_data: Dict[str, pd.DataFrame] = {}
        self._daily_data: Dict[str, pd.DataFrame] = {}
        self._trading_dates: List[str] = []

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def load_data(self, start_date: str, end_date: str) -> None:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")

        # 分钟线
        for sym in self.symbols:
            df = pd.read_sql_query(
                "SELECT datetime, open, high, low, close, volume "
                "FROM futures_min "
                "WHERE symbol = ? AND period = ? "
                "  AND datetime >= ? AND datetime <= ? "
                "ORDER BY datetime",
                conn,
                params=(sym, self.period,
                        f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}",
                        f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]} 23:59:59"),
            )
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
            # 提取交易日 (UTC date = Beijing date for daytime futures)
            df["trade_date"] = df.index.strftime("%Y%m%d")
            self._min_data[sym] = df

        # 15分钟线（用于多周期趋势一致性维度）
        alt_period = 900 if self.period == 300 else 300
        for sym in self.symbols:
            df = pd.read_sql_query(
                "SELECT datetime, open, high, low, close, volume "
                "FROM futures_min "
                "WHERE symbol = ? AND period = ? "
                "  AND datetime >= ? AND datetime <= ? "
                "ORDER BY datetime",
                conn,
                params=(sym, alt_period,
                        f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}",
                        f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]} 23:59:59"),
            )
            if len(df) > 0:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")
                df["trade_date"] = df.index.strftime("%Y%m%d")
                self._min_15m_data[sym] = df

        # 日线（用于日线级支撑阻力）
        for sym in self.symbols:
            ts_code = f"{sym}.CFX"
            df = pd.read_sql_query(
                "SELECT trade_date, open, high, low, close, volume "
                "FROM futures_daily "
                "WHERE ts_code = ? "
                "ORDER BY trade_date",
                conn,
                params=(ts_code,),
            )
            self._daily_data[sym] = df

        conn.close()

        # 提取所有交易日
        all_dates = set()
        for sym, df in self._min_data.items():
            all_dates.update(df["trade_date"].unique())
        self._trading_dates = sorted(all_dates)

        if start_date:
            self._trading_dates = [d for d in self._trading_dates if d >= start_date]
        if end_date:
            self._trading_dates = [d for d in self._trading_dates if d <= end_date]

        print(f"  加载完成: {len(self._trading_dates)} 个交易日")
        for sym in self.symbols:
            n = len(self._min_data.get(sym, []))
            print(f"    {sym}: {n} 根K线")

    # ------------------------------------------------------------------
    # 主回测循环
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        print(f"\n{'═' * 60}")
        print(f"  日内策略回测 | {self.period // 60}分钟线 | "
              f"{self._trading_dates[0]} ~ {self._trading_dates[-1]}")
        print(f"  品种: {', '.join(self.symbols)} | "
              f"初始资金: {self.initial_capital:,.0f}")
        print(f"{'═' * 60}\n")

        peak = self.capital
        max_dd = 0.0
        prev_date = ""

        for date_idx, trade_date in enumerate(self._trading_dates):
            # 下一交易日
            next_td = self._trading_dates[date_idx + 1] \
                if date_idx + 1 < len(self._trading_dates) else ""

            # 日线数据（当日之前）
            daily_slices: Dict[str, pd.DataFrame] = {}
            for sym in self.symbols:
                dd = self._daily_data.get(sym)
                if dd is not None and len(dd) > 0:
                    mask = dd["trade_date"] < trade_date
                    daily_slices[sym] = dd[mask].tail(30)

            # 开盘处理
            opening_prices: Dict[str, float] = {}
            for sym in self.symbols:
                md = self._min_data.get(sym)
                if md is not None:
                    day_bars = md[md["trade_date"] == trade_date]
                    if len(day_bars) > 0:
                        opening_prices[sym] = float(day_bars.iloc[0]["open"])

            if prev_date:
                unlock_actions = self.strategy.on_daily_open(
                    trade_date, opening_prices
                )
                for act in unlock_actions:
                    sym = act["symbol"]
                    price = opening_prices.get(sym, 0)
                    # 双平手续费
                    comm = _commission(sym, price, "unlock_close") * 2
                    self.total_commission += comm
                    self.capital += act["pnl"] - comm
                    self.trade_log.append({
                        **act, "trade_date": trade_date,
                        "commission": comm,
                    })
            else:
                self.strategy.on_daily_open(trade_date, opening_prices)

            # 逐K线推进
            day_actions: List[Dict] = []
            day_pnl = 0.0

            # 收集所有品种当日K线的时间戳并排序
            all_times = set()
            day_bars_cache: Dict[str, pd.DataFrame] = {}
            for sym in self.symbols:
                md = self._min_data.get(sym)
                if md is not None:
                    db = md[md["trade_date"] == trade_date]
                    day_bars_cache[sym] = db
                    all_times.update(db.index.tolist())

            sorted_times = sorted(all_times)

            # DEBUG: 第一个月的每个交易日打印开盘区间信息
            first_month_end = self._trading_dates[0][:6] + "31"
            if trade_date <= first_month_end:
                n_or = self.strategy.signal_gen._opening_bars
                for sym in self.symbols:
                    db = day_bars_cache.get(sym)
                    if db is not None and len(db) >= n_or:
                        or_bars = db.iloc[:n_or]
                        or_high = float(or_bars["high"].max())
                        or_low = float(or_bars["low"].min())
                        or_width = or_high - or_low
                        first_signal_time = str(db.index[n_or]) if len(db) > n_or else "N/A"
                        print(f"  [DEBUG] {trade_date} {sym}: "
                              f"开盘区间 H={or_high:.1f} L={or_low:.1f} "
                              f"宽={or_width:.1f}  "
                              f"首个信号K线={first_signal_time.split(' ')[-1][:5]}  "
                              f"当日共{len(db)}根K线")

            # 15m数据：使用滚动窗口（跨日，最近50根），让SMA计算有效
            day_15m_cache: Dict[str, pd.DataFrame] = {}
            for sym in self.symbols:
                md15 = self._min_15m_data.get(sym)
                if md15 is not None:
                    # 取当天及之前的所有15m bars
                    end_of_day = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]} 23:59:59"
                    up_to_today = md15[md15.index <= end_of_day]
                    if len(up_to_today) > 0:
                        day_15m_cache[sym] = up_to_today.tail(60)  # 最近60根

            for bar_time in sorted_times:
                current_time_str = str(bar_time)

                # 构建到当前bar为止的数据（rolling多日窗口，让SMA等指标有效）
                bar_data: Dict[str, pd.DataFrame] = {}
                for sym in self.symbols:
                    md = self._min_data.get(sym)
                    if md is not None:
                        up_to = md[md.index <= bar_time]
                        if len(up_to) > 0:
                            bar_data[sym] = up_to.tail(60)  # 最近60根

                # 15分钟数据（到当前时间为止）
                bar_15m_data: Dict[str, pd.DataFrame] = {}
                for sym in self.symbols:
                    db15 = day_15m_cache.get(sym)
                    if db15 is not None:
                        up_to_15 = db15[db15.index <= bar_time]
                        if len(up_to_15) > 0:
                            bar_15m_data[sym] = up_to_15

                # 执行策略
                actions = self.strategy.on_bar(
                    bar_data, bar_15m_data, daily_slices,
                    current_time_str, next_trade_date=next_td,
                )

                # 处理操作
                for act in actions:
                    sym = act.get("symbol", "")
                    price = act.get("price", 0)

                    if act["action"] == "OPEN":
                        comm = _commission(sym, price, "open")
                        self.total_commission += comm
                        self.capital -= comm
                        act["commission"] = comm

                    elif act["action"] == "LOCK":
                        comm = _commission(sym, price if price else
                                           opening_prices.get(sym, 0),
                                           "lock_open")
                        self.total_commission += comm
                        self.capital += act["pnl"] - comm
                        day_pnl += act["pnl"]
                        act["commission"] = comm
                        self.lock_count += 1
                        # 计算节省的手续费
                        saved = _commission(sym, price if price else
                                            opening_prices.get(sym, 0),
                                            "intraday_close") - comm
                        self.commission_saved += saved

                    elif act["action"] == "CLOSE":
                        comm = _commission(sym, price if price else
                                           opening_prices.get(sym, 0),
                                           "overnight_close")
                        self.total_commission += comm
                        self.capital += act["pnl"] - comm
                        day_pnl += act["pnl"]
                        act["commission"] = comm
                        self.direct_close_count += 1

                    act["trade_date"] = trade_date
                    act["bar_time"] = current_time_str
                    day_actions.extend([act])
                    self.trade_log.append(act)

            # 日结
            equity = self.capital
            # 加上浮盈
            eod_prices: Dict[str, float] = {}
            for sym in self.symbols:
                db = day_bars_cache.get(sym)
                if db is not None and len(db) > 0:
                    eod_prices[sym] = float(db.iloc[-1]["close"])

            for pos in self.strategy.position_mgr.positions.values():
                if pos.is_lock:
                    continue
                p = eod_prices.get(pos.symbol)
                if p is None:
                    continue
                mult = CONTRACT_MULTIPLIERS.get(pos.symbol, 300)
                if pos.direction == "LONG":
                    equity += (p - pos.entry_price) * pos.volume * mult
                else:
                    equity += (pos.entry_price - p) * pos.volume * mult

            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

            n_opens = sum(1 for a in day_actions if a["action"] == "OPEN")
            n_closes = sum(1 for a in day_actions
                          if a["action"] in ("LOCK", "CLOSE", "UNLOCK"))

            self.equity_curve.append({
                "trade_date": trade_date,
                "equity": equity,
                "capital": self.capital,
                "daily_pnl": day_pnl,
                "drawdown": dd,
                "opens": n_opens,
                "closes": n_closes,
            })

            self.daily_stats.append({
                "trade_date": trade_date,
                "pnl": day_pnl,
                "trades": n_opens,
                "equity": equity,
            })

            prev_date = trade_date

        return self._compile_results(max_dd)

    # ------------------------------------------------------------------
    # 结果汇总
    # ------------------------------------------------------------------

    def _compile_results(self, max_dd: float) -> Dict:
        eq = pd.DataFrame(self.equity_curve)
        if eq.empty:
            return {}

        total_pnl = eq["equity"].iloc[-1] - self.initial_capital
        n_days = len(eq)
        annual_return = total_pnl / self.initial_capital / n_days * 252 \
            if n_days > 0 else 0

        # 交易统计
        opens = [t for t in self.trade_log if t["action"] == "OPEN"]
        closes = [t for t in self.trade_log
                  if t["action"] in ("LOCK", "CLOSE")]
        n_trades = len(closes)
        wins = [t for t in closes if t.get("pnl", 0) > 0]
        losses = [t for t in closes if t.get("pnl", 0) <= 0]
        win_rate = len(wins) / n_trades if n_trades > 0 else 0
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t["pnl"] for t in losses])) if losses else 1
        pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        # 夏普
        daily_rets = eq["daily_pnl"] / self.initial_capital
        sharpe = (daily_rets.mean() / daily_rets.std() * np.sqrt(252)
                  if daily_rets.std() > 0 else 0)

        # 月度统计
        eq["month"] = eq["trade_date"].str[:6]
        monthly = eq.groupby("month")["daily_pnl"].sum()
        monthly_wins = (monthly > 0).sum()
        monthly_total = len(monthly)

        return {
            "total_pnl": total_pnl,
            "annual_return": annual_return,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "n_trades": n_trades,
            "win_rate": win_rate,
            "pl_ratio": pl_ratio,
            "commission": self.total_commission,
            "lock_count": self.lock_count,
            "direct_close_count": self.direct_close_count,
            "commission_saved": self.commission_saved,
            "monthly_win_rate": monthly_wins / monthly_total
            if monthly_total > 0 else 0,
            "n_days": n_days,
        }

    # ------------------------------------------------------------------
    # 报告
    # ------------------------------------------------------------------

    def generate_report(self, results: Dict) -> None:
        print(f"\n{'═' * 60}")
        print("  日内策略回测报告")
        print(f"{'═' * 60}")

        print(f"\n【总体表现】")
        print(f"  总盈亏       : {results['total_pnl']:>+12,.0f} 元")
        print(f"  年化收益     : {results['annual_return']:>10.2%}")
        print(f"  最大回撤     : {results['max_drawdown']:>10.2%}")
        print(f"  夏普比率     : {results['sharpe']:>10.2f}")
        print(f"  月度胜率     : {results['monthly_win_rate']:>10.1%}")

        print(f"\n【交易统计】")
        print(f"  总交易笔数   : {results['n_trades']:>6d}")
        print(f"  胜率         : {results['win_rate']:>10.1%}")
        print(f"  盈亏比       : {results['pl_ratio']:>10.2f}")
        print(f"  交易天数     : {results['n_days']:>6d}")
        print(f"  月均交易     : {results['n_trades'] / max(1, results['n_days']) * 21:>10.1f} 笔")

        print(f"\n【手续费与锁仓】")
        print(f"  总手续费     : {results['commission']:>12,.0f} 元")
        print(f"  锁仓次数     : {results['lock_count']:>6d}")
        print(f"  直接平仓次数 : {results['direct_close_count']:>6d}")
        print(f"  锁仓节省费用 : {results['commission_saved']:>12,.0f} 元")

        # 按品种统计
        print(f"\n【按品种统计】")
        by_sym: Dict[str, List] = defaultdict(list)
        for t in self.trade_log:
            if t["action"] in ("LOCK", "CLOSE"):
                by_sym[t.get("symbol", "?")].append(t)

        print(f"  {'品种':>4}  {'交易数':>6}  {'胜率':>6}  {'盈亏比':>6}  {'总盈亏':>12}")
        for sym in sorted(by_sym.keys()):
            trades = by_sym[sym]
            n = len(trades)
            w = sum(1 for t in trades if t.get("pnl", 0) > 0)
            wr = w / n if n > 0 else 0
            avg_w = np.mean([t["pnl"] for t in trades if t.get("pnl", 0) > 0]) or 0
            avg_l = abs(np.mean([t["pnl"] for t in trades
                                 if t.get("pnl", 0) <= 0]) or 1)
            plr = avg_w / avg_l if avg_l > 0 else 0
            total = sum(t.get("pnl", 0) for t in trades)
            print(f"  {sym:>4}  {n:>6d}  {wr:>5.1%}  {plr:>6.2f}  {total:>+12,.0f}")

        # 按信号强度统计
        print(f"\n【按信号强度统计】")
        opens = [t for t in self.trade_log if t["action"] == "OPEN"]
        score_bins = [(80, 100), (60, 79)]
        print(f"  {'强度':>8}  {'交易数':>6}  {'平均得分':>8}")
        for lo, hi in score_bins:
            in_bin = [t for t in opens if lo <= t.get("score", 0) <= hi]
            n = len(in_bin)
            avg = np.mean([t.get("score", 0) for t in in_bin]) if in_bin else 0
            print(f"  {lo}-{hi:>3}    {n:>6d}  {avg:>8.1f}")

        # 按时段统计
        print(f"\n【按时段统计（UTC→北京时间）】")
        time_bins = [
            ("01:35", "02:30", "09:35-10:30"),
            ("02:30", "03:30", "10:30-11:30"),
            ("05:00", "06:00", "13:00-14:00"),
            ("06:00", "06:50", "14:00-14:50"),
        ]
        print(f"  {'时段':>12}  {'开仓数':>6}")
        for start, end, label in time_bins:
            in_period = [t for t in opens
                         if start <= (t.get("bar_time", "").split(" ")[-1][:5]
                                      if t.get("bar_time") else "") < end]
            print(f"  {label:>12}  {len(in_period):>6d}")

        # 止损分析
        print(f"\n【止损分析】")
        stops = [t for t in self.trade_log if t.get("reason") == "STOP_LOSS"]
        print(f"  止损触发次数 : {len(stops)}")
        if stops:
            avg_sl = np.mean([abs(t.get("pnl", 0)) for t in stops])
            print(f"  平均止损金额 : {avg_sl:,.0f} 元")

        print(f"\n{'═' * 60}")

    # ------------------------------------------------------------------
    # 打印每日明细
    # ------------------------------------------------------------------

    def print_trade_details(self, max_rows: int = 30) -> None:
        print(f"\n【交易明细（前{max_rows}笔）】")
        print(f"  {'日期':>10}  {'时间(UTC)':>12}  {'操作':>6}  "
              f"{'品种':>4}  {'方向':>5}  {'价格':>8}  "
              f"{'盈亏':>10}  {'原因':>12}")
        count = 0
        for t in self.trade_log:
            if count >= max_rows:
                break
            action = t.get("action", "")
            sym = t.get("symbol", "")
            direction = t.get("direction", "")
            price = t.get("price", 0)
            pnl = t.get("pnl", 0)
            reason = t.get("reason", "")
            td = t.get("trade_date", "")
            bt = t.get("bar_time", "")
            time_part = bt.split(" ")[-1][:8] if bt else ""

            print(f"  {td:>10}  {time_part:>12}  {action:>6}  "
                  f"{sym:>4}  {direction:>5}  {price:>8.1f}  "
                  f"{pnl:>+10,.0f}  {reason:>12}")
            count += 1


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def _run_single(args, signal_version: str = "v1") -> Dict:
    """运行单个版本的回测，返回结果字典。"""
    period = 300 if args.period == "5m" else 900
    symbols = args.symbols.split(",")

    config = IntradayConfig(
        universe=symbols,
        min_signal_score=args.min_score,
    )

    bt = IntradayBacktester(
        symbols=symbols,
        period=period,
        initial_capital=args.capital,
        config=config,
        signal_version=signal_version,
    )

    print(f"加载数据 ({args.period}, {signal_version})...")
    bt.load_data(args.start, args.end)

    if not bt._trading_dates:
        print("无数据，退出")
        return {}

    results = bt.run()
    return {"bt": bt, "results": results}


def _print_comparison(r1: Dict, r2: Dict) -> None:
    """打印 v1 vs v2 对比表。"""
    _print_comparison_v3({"v1": r1, "v2": r2})


def _print_comparison_v3(results_map: Dict[str, Dict]) -> None:
    """打印 v1/v2/v3 对比表，含按品种明细。"""
    versions = [v for v in ["v1", "v2", "v3"] if v in results_map]

    print(f"\n{'═' * 74}")
    print(f"  信号系统对比")
    print(f"{'═' * 74}")

    header = (f"  {'版本':>4}  {'交易数':>6}  {'胜率':>6}  {'盈亏比':>6}"
              f"  {'总盈亏':>12}  {'夏普':>6}  {'最大回撤':>8}")
    print(header)
    print(f"  {'─' * 68}")

    for ver in versions:
        r = results_map.get(ver)
        if not r or not r.get("results"):
            print(f"  {ver:>4}  {'无数据':>6}")
            continue
        res = r["results"]
        print(f"  {ver:>4}  {res['n_trades']:>6d}  {res['win_rate']:>5.1%}"
              f"  {res['pl_ratio']:>6.2f}  {res['total_pnl']:>+12,.0f}"
              f"  {res['sharpe']:>6.2f}  {res['max_drawdown']:>7.2%}")

    print(f"{'═' * 74}")

    # 按品种对比
    print(f"\n{'═' * 74}")
    print(f"  按品种对比")
    print(f"{'═' * 74}")
    header2 = (f"  {'品种':>4}  {'版本':>4}  {'交易数':>6}  {'胜率':>6}"
               f"  {'盈亏比':>6}  {'总盈亏':>12}  {'夏普':>6}")
    print(header2)
    print(f"  {'─' * 60}")

    all_syms = set()
    for ver in versions:
        r = results_map.get(ver)
        if r and r.get("bt"):
            for t in r["bt"].trade_log:
                if t["action"] in ("LOCK", "CLOSE"):
                    all_syms.add(t.get("symbol", ""))

    for sym in sorted(all_syms):
        for ver in versions:
            r = results_map.get(ver)
            if not r or not r.get("bt"):
                continue
            trades = [t for t in r["bt"].trade_log
                      if t["action"] in ("LOCK", "CLOSE")
                      and t.get("symbol") == sym]
            n = len(trades)
            if n == 0:
                continue
            w = sum(1 for t in trades if t.get("pnl", 0) > 0)
            wr = w / n
            pnls = [t.get("pnl", 0) for t in trades]
            total = sum(pnls)
            avg_w = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
            avg_l = abs(np.mean([p for p in pnls if p <= 0])) if any(p <= 0 for p in pnls) else 1
            plr = avg_w / avg_l if avg_l > 0 else 0

            # per-symbol daily returns for sharpe
            bt = r["bt"]
            daily_pnls = defaultdict(float)
            for t in trades:
                daily_pnls[t.get("trade_date", "")] += t.get("pnl", 0)
            if daily_pnls:
                dr = np.array(list(daily_pnls.values())) / bt.initial_capital
                sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0
            else:
                sharpe = 0

            print(f"  {sym:>4}  {ver:>4}  {n:>6d}  {wr:>5.1%}"
                  f"  {plr:>6.2f}  {total:>+12,.0f}  {sharpe:>6.2f}")
        print(f"  {'─' * 60}")

    print(f"{'═' * 74}")


def main():
    parser = argparse.ArgumentParser(description="日内策略回测")
    parser.add_argument("--start", default="20250509", help="起始日期 YYYYMMDD")
    parser.add_argument("--end", default="20260318", help="结束日期 YYYYMMDD")
    parser.add_argument("--period", default="5m",
                        choices=["5m", "15m"], help="K线周期")
    parser.add_argument("--capital", type=float, default=5_000_000,
                        help="初始资金")
    parser.add_argument("--symbols", default="IF,IH,IM",
                        help="交易品种（逗号分隔）")
    parser.add_argument("--min-score", type=int, default=60,
                        help="最低信号分数")
    parser.add_argument("--signal-version", default="v1",
                        choices=["v1", "v2", "v3", "compare"],
                        help="信号版本: v1/v2/v3/compare")
    args = parser.parse_args()

    if args.signal_version == "compare":
        # 对比模式：依次运行 v1, v2, v3
        results_map: Dict[str, Dict] = {}
        for ver in ["v1", "v2", "v3"]:
            print("\n" + "=" * 70)
            print(f"  运行 {ver} 回测...")
            print("=" * 70)
            r = _run_single(args, ver)
            if r and r.get("results"):
                r["bt"].generate_report(r["results"])
            results_map[ver] = r

        _print_comparison_v3(results_map)
    else:
        r = _run_single(args, args.signal_version)
        if r and r.get("results"):
            r["bt"].generate_report(r["results"])
            r["bt"].print_trade_details(max_rows=20)


if __name__ == "__main__":
    main()
