"""
gamma_backtest.py
-----------------
贴水+Gamma Scalping 回测引擎。

数据需求：
- futures_daily: IM.CFX, IML1/2/3.CFX 日线（计算贴水、选合约）
- index_daily: 000852.SH（现货价格，计算真实贴水）
- options_daily + options_contracts: MO期权日线（选Put、计算IV和Greeks）
- futures_min: IM 5分钟线（Gamma Scalping的对冲模拟）

用法：
    python -m strategies.discount_capture.gamma_backtest \\
        --start 20220722 --end 20260319 --capital 5000000

    python -m strategies.discount_capture.gamma_backtest \\
        --start 20250509 --end 20260319 --capital 5000000 --with-scalping
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models.pricing.greeks import calc_delta, calc_gamma, calc_theta
from models.pricing.implied_vol import bs_price, calc_implied_vol
from strategies.discount_capture.gamma_scalper import (
    GammaScalper,
    FUTURES_MULTIPLIER,
    OPTION_MULTIPLIER,
)
from strategies.discount_capture.gamma_strategy import (
    DiscountGammaStrategy,
    _parse_mo_ts_code,
)

logger = logging.getLogger(__name__)

MARGIN_RATE = 0.15  # 期货保证金率


# ======================================================================
# Backtester
# ======================================================================

class DiscountGammaBacktester:
    """
    贴水+Gamma Scalping回测引擎。

    Parameters
    ----------
    db_manager : DBManager
    initial_capital : float
    futures_lots : int
    strategy_config : dict
    """

    def __init__(
        self,
        db_manager,
        initial_capital: float = 5_000_000.0,
        futures_lots: int = 1,
        strategy_config: Dict = None,
    ):
        self.db = db_manager
        self.initial_capital = initial_capital
        self.futures_lots = futures_lots
        self.strategy = DiscountGammaStrategy(db_manager, strategy_config)
        self.risk_free_rate = (strategy_config or {}).get("risk_free_rate", 0.02)

    # ------------------------------------------------------------------
    # 日线级回测
    # ------------------------------------------------------------------

    def run_daily_only(
        self,
        start_date: str = "20220722",
        end_date: str = "20260319",
    ) -> Dict:
        """
        日线级回测：贴水收割 + Put保护（不含Gamma Scalping）。
        """
        logger.info("开始日线级回测 %s ~ %s", start_date, end_date)

        # 加载数据
        trade_dates = self._get_trade_dates(start_date, end_date)
        if len(trade_dates) < 20:
            logger.warning("交易日不足")
            return self._empty_result()

        capital = self.initial_capital
        equity_curve = []
        trades = []
        daily_pnl_records = []

        # 当前持仓状态
        position = None  # {contract_month, futures_price, spot_price, iml_code, ...}
        put_position = None  # {strike, entry_price, lots, iv, expire_date, ...}
        entry_date = None

        prev_futures_close = None
        prev_put_close = None

        # 收益分解累计
        total_discount_pnl = 0.0
        total_direction_pnl = 0.0
        total_put_cost = 0.0
        total_put_payout = 0.0

        for td in trade_dates:
            # 1. 检查是否需要建仓/换月
            if position is None:
                best = self.strategy.select_best_futures_contract(td)
                if best is None or best["annualized_rate"] < 0.05:
                    equity_curve.append({"date": td, "capital": capital})
                    continue

                position = best
                entry_date = td
                prev_futures_close = best["futures_price"]

                # 选Put
                put_info = self.strategy.select_optimal_put(
                    best["futures_price"], td, best["contract_month"]
                )
                if put_info:
                    put_lots = self.strategy.calc_initial_put_lots(
                        self.futures_lots, put_info["delta"]
                    )
                    put_cost = put_info["put_price"] * OPTION_MULTIPLIER * put_lots
                    capital -= put_cost
                    total_put_cost += put_cost
                    put_position = {
                        **put_info,
                        "lots": put_lots,
                        "entry_price": put_info["put_price"],
                    }
                    prev_put_close = put_info["put_price"]
                else:
                    put_position = None
                    prev_put_close = None

                equity_curve.append({"date": td, "capital": capital})
                continue

            # 2. 获取今日价格
            futures_close = self._get_futures_close(td, position["iml_code"])
            spot_close = self._get_spot_close(td)

            if futures_close is None:
                equity_curve.append({"date": td, "capital": capital})
                continue

            # 3. 计算每日P&L
            if prev_futures_close is not None:
                daily_futures_pnl = (futures_close - prev_futures_close) * FUTURES_MULTIPLIER * self.futures_lots
            else:
                daily_futures_pnl = 0.0

            daily_put_pnl = 0.0
            if put_position is not None:
                put_close = self._get_put_close(td, put_position)
                if put_close is not None and prev_put_close is not None:
                    daily_put_pnl = (put_close - prev_put_close) * OPTION_MULTIPLIER * put_position["lots"]
                    prev_put_close = put_close

            daily_total = daily_futures_pnl + daily_put_pnl
            capital += daily_total

            daily_pnl_records.append({
                "date": td,
                "futures_pnl": daily_futures_pnl,
                "put_pnl": daily_put_pnl,
                "total_pnl": daily_total,
            })

            prev_futures_close = futures_close

            # 4. 检查到期/换月
            dte_remaining = position["days_to_expiry"] - (
                pd.Timestamp(td) - pd.Timestamp(entry_date)
            ).days

            need_roll = dte_remaining <= self.strategy.roll_days

            # Put滚动检查
            if put_position is not None:
                put_dte = put_position["dte"] - (
                    pd.Timestamp(td) - pd.Timestamp(entry_date)
                ).days
                if put_dte <= self.strategy.put_roll_days:
                    # 平掉旧Put
                    if prev_put_close is not None and prev_put_close > 0:
                        residual = prev_put_close * OPTION_MULTIPLIER * put_position["lots"]
                        capital += residual
                        total_put_payout += residual
                    put_position = None
                    prev_put_close = None

                    if not need_roll:
                        # 滚动到新Put
                        put_info = self.strategy.select_optimal_put(
                            futures_close, td, position["contract_month"]
                        )
                        if put_info:
                            put_lots = self.strategy.calc_initial_put_lots(
                                self.futures_lots, put_info["delta"]
                            )
                            put_cost = put_info["put_price"] * OPTION_MULTIPLIER * put_lots
                            capital -= put_cost
                            total_put_cost += put_cost
                            put_position = {
                                **put_info,
                                "lots": put_lots,
                                "entry_price": put_info["put_price"],
                            }
                            prev_put_close = put_info["put_price"]

            if need_roll:
                # 记录这笔交易
                if spot_close and position.get("spot_price"):
                    # 贴水收益 = 现货变动 - 期货变动 的修正
                    # 简化：贴水收益 ≈ 入场贴水点数 × 合约乘数（如果期货向现货收敛）
                    discount_convergence = abs(position["discount_points"]) * FUTURES_MULTIPLIER * self.futures_lots
                    direction_pnl = (futures_close - position["futures_price"]) * FUTURES_MULTIPLIER * self.futures_lots - discount_convergence
                else:
                    discount_convergence = 0
                    direction_pnl = (futures_close - position["futures_price"]) * FUTURES_MULTIPLIER * self.futures_lots

                total_pnl = (futures_close - position["futures_price"]) * FUTURES_MULTIPLIER * self.futures_lots
                total_discount_pnl += discount_convergence
                total_direction_pnl += direction_pnl

                trades.append({
                    "entry_date": entry_date,
                    "exit_date": td,
                    "contract": position["symbol"],
                    "entry_price": position["futures_price"],
                    "exit_price": futures_close,
                    "annualized_discount": position["annualized_rate"],
                    "discount_pnl": discount_convergence,
                    "direction_pnl": direction_pnl,
                    "futures_pnl": total_pnl,
                    "put_strike": put_position["strike"] if put_position else None,
                    "capital": capital,
                })

                # 平掉Put
                if put_position is not None and prev_put_close is not None:
                    residual = prev_put_close * OPTION_MULTIPLIER * put_position["lots"]
                    capital += residual
                    total_put_payout += residual

                position = None
                put_position = None
                prev_put_close = None
                prev_futures_close = None

            equity_curve.append({"date": td, "capital": capital})

        # 汇总
        return self._build_results(
            trades, equity_curve, daily_pnl_records,
            total_discount_pnl, total_direction_pnl,
            total_put_cost, total_put_payout,
            gamma_pnl=0.0, theta_cost=0.0, commission_cost=0.0,
        )

    # ------------------------------------------------------------------
    # 含Gamma Scalping的回测
    # ------------------------------------------------------------------

    def run_with_scalping(
        self,
        start_date: str = "20250509",
        end_date: str = "20260319",
        scalp_configs: List[Dict] = None,
    ) -> Dict:
        """
        日线+分钟线回测：贴水 + Put保护 + Gamma Scalping。

        Parameters
        ----------
        scalp_configs : list[dict]
            多组对冲参数，用于敏感性分析。
            默认测试4种阈值。
        """
        if scalp_configs is None:
            scalp_configs = [
                {"rehedge_method": "price", "rehedge_threshold_pct": 0.002, "label": "0.2%"},
                {"rehedge_method": "price", "rehedge_threshold_pct": 0.003, "label": "0.3%"},
                {"rehedge_method": "price", "rehedge_threshold_pct": 0.005, "label": "0.5%"},
                {"rehedge_method": "time", "time_interval_min": 30, "label": "30min"},
            ]

        logger.info("开始含Gamma Scalping回测 %s ~ %s（%d组参数）",
                     start_date, end_date, len(scalp_configs))

        # 加载分钟数据
        min_data = self._load_minute_data(start_date, end_date)
        min_dates = set()
        if min_data is not None and not min_data.empty:
            min_data["date"] = min_data["datetime"].str[:10].str.replace("-", "")
            min_dates = set(min_data["date"].unique())

        results_by_config = {}

        for cfg in scalp_configs:
            label = cfg.pop("label", str(cfg))
            result = self._run_single_scalping(
                start_date, end_date, cfg, min_data, min_dates
            )
            result["label"] = label
            results_by_config[label] = result
            cfg["label"] = label  # restore

        # 找最优
        best_label = max(
            results_by_config,
            key=lambda k: results_by_config[k].get("net_gamma_pnl", 0)
        )

        return {
            "best_config": best_label,
            "best_result": results_by_config[best_label],
            "all_results": results_by_config,
            "sensitivity": self._build_sensitivity_table(results_by_config),
        }

    def _run_single_scalping(
        self,
        start_date: str,
        end_date: str,
        scalp_config: Dict,
        min_data: Optional[pd.DataFrame],
        min_dates: set,
    ) -> Dict:
        """单组参数的Gamma Scalping回测"""

        trade_dates = self._get_trade_dates(start_date, end_date)
        if len(trade_dates) < 5:
            return self._empty_result()

        capital = self.initial_capital
        equity_curve = []
        trades = []
        daily_pnl_records = []

        position = None
        put_position = None
        entry_date = None
        prev_futures_close = None
        prev_put_close = None

        total_discount_pnl = 0.0
        total_direction_pnl = 0.0
        total_put_cost = 0.0
        total_put_payout = 0.0
        total_gamma_pnl = 0.0
        total_theta_cost = 0.0
        total_commission = 0.0

        scalper = GammaScalper(scalp_config)

        for td in trade_dates:
            # 建仓逻辑（同daily_only）
            if position is None:
                best = self.strategy.select_best_futures_contract(td)
                if best is None or best["annualized_rate"] < 0.05:
                    equity_curve.append({"date": td, "capital": capital})
                    continue

                position = best
                entry_date = td
                prev_futures_close = best["futures_price"]

                put_info = self.strategy.select_optimal_put(
                    best["futures_price"], td, best["contract_month"]
                )
                if put_info:
                    put_lots = self.strategy.calc_initial_put_lots(
                        self.futures_lots, put_info["delta"]
                    )
                    put_cost = put_info["put_price"] * OPTION_MULTIPLIER * put_lots
                    capital -= put_cost
                    total_put_cost += put_cost
                    put_position = {
                        **put_info,
                        "lots": put_lots,
                        "entry_price": put_info["put_price"],
                    }
                    prev_put_close = put_info["put_price"]
                    scalper.initialize(best["futures_price"])
                else:
                    put_position = None
                    prev_put_close = None

                equity_curve.append({"date": td, "capital": capital})
                continue

            # 获取日线价格
            futures_close = self._get_futures_close(td, position["iml_code"])
            if futures_close is None:
                equity_curve.append({"date": td, "capital": capital})
                continue

            # Gamma Scalping（分钟级）
            day_gamma_pnl = 0.0
            day_theta_cost = 0.0
            day_commission = 0.0
            day_rehedge_count = 0

            if put_position is not None:
                scalper.reset_daily()

                td_formatted = f"{td[:4]}-{td[4:6]}-{td[6:]}"

                if td in min_dates and min_data is not None:
                    # 有分钟数据：逐bar模拟
                    day_bars = min_data[min_data["date"] == td].sort_values("datetime")
                    day_gamma_pnl, day_commission, day_rehedge_count = self._simulate_minute_scalping(
                        scalper, day_bars, put_position, futures_close
                    )
                else:
                    # 无分钟数据：用日线High-Low估算
                    day_gamma_pnl = self._estimate_daily_gamma_pnl(
                        td, put_position, position["iml_code"]
                    )

                # Theta成本
                dte_put = put_position["dte"] - (pd.Timestamp(td) - pd.Timestamp(entry_date)).days
                if dte_put > 0:
                    T_put = dte_put / 365.0
                    theta_per_lot = calc_theta(
                        futures_close, put_position["strike"], T_put,
                        self.risk_free_rate, put_position.get("iv", 0.25), "P"
                    )
                    day_theta_cost = abs(theta_per_lot) * OPTION_MULTIPLIER * put_position["lots"]
                else:
                    day_theta_cost = 0.0

                total_gamma_pnl += day_gamma_pnl
                total_theta_cost += day_theta_cost
                total_commission += day_commission

            # 日线P&L
            daily_futures_pnl = (futures_close - prev_futures_close) * FUTURES_MULTIPLIER * self.futures_lots
            daily_put_pnl = 0.0
            if put_position is not None:
                put_close = self._get_put_close(td, put_position)
                if put_close is not None and prev_put_close is not None:
                    daily_put_pnl = (put_close - prev_put_close) * OPTION_MULTIPLIER * put_position["lots"]
                    prev_put_close = put_close

            # Scalping净利润记入capital
            scalp_net = day_gamma_pnl - day_commission
            daily_total = daily_futures_pnl + daily_put_pnl + scalp_net
            capital += daily_total

            daily_pnl_records.append({
                "date": td,
                "futures_pnl": daily_futures_pnl,
                "put_pnl": daily_put_pnl,
                "gamma_pnl": day_gamma_pnl,
                "theta_cost": day_theta_cost,
                "commission": day_commission,
                "scalp_net": scalp_net,
                "total_pnl": daily_total,
                "rehedge_count": day_rehedge_count,
            })

            prev_futures_close = futures_close

            # 换月/滚动检查
            dte_remaining = position["days_to_expiry"] - (
                pd.Timestamp(td) - pd.Timestamp(entry_date)
            ).days

            # Put滚动
            if put_position is not None:
                put_dte = put_position["dte"] - (
                    pd.Timestamp(td) - pd.Timestamp(entry_date)
                ).days
                if put_dte <= self.strategy.put_roll_days:
                    if prev_put_close is not None and prev_put_close > 0:
                        residual = prev_put_close * OPTION_MULTIPLIER * put_position["lots"]
                        capital += residual
                        total_put_payout += residual
                    put_position = None
                    prev_put_close = None

                    if dte_remaining > self.strategy.roll_days:
                        put_info = self.strategy.select_optimal_put(
                            futures_close, td, position["contract_month"]
                        )
                        if put_info:
                            put_lots = self.strategy.calc_initial_put_lots(
                                self.futures_lots, put_info["delta"]
                            )
                            cost = put_info["put_price"] * OPTION_MULTIPLIER * put_lots
                            capital -= cost
                            total_put_cost += cost
                            put_position = {
                                **put_info,
                                "lots": put_lots,
                                "entry_price": put_info["put_price"],
                            }
                            prev_put_close = put_info["put_price"]
                            scalper.initialize(futures_close)

            if dte_remaining <= self.strategy.roll_days:
                total_pnl = (futures_close - position["futures_price"]) * FUTURES_MULTIPLIER * self.futures_lots
                discount_convergence = abs(position["discount_points"]) * FUTURES_MULTIPLIER * self.futures_lots
                direction_pnl = total_pnl - discount_convergence
                total_discount_pnl += discount_convergence
                total_direction_pnl += direction_pnl

                trades.append({
                    "entry_date": entry_date,
                    "exit_date": td,
                    "contract": position["symbol"],
                    "entry_price": position["futures_price"],
                    "exit_price": futures_close,
                    "annualized_discount": position["annualized_rate"],
                    "discount_pnl": discount_convergence,
                    "direction_pnl": direction_pnl,
                    "futures_pnl": total_pnl,
                    "capital": capital,
                })

                if put_position is not None and prev_put_close is not None:
                    residual = prev_put_close * OPTION_MULTIPLIER * put_position["lots"]
                    capital += residual
                    total_put_payout += residual

                position = None
                put_position = None
                prev_put_close = None
                prev_futures_close = None
                scalper.initialize(0)

            equity_curve.append({"date": td, "capital": capital})

        return self._build_results(
            trades, equity_curve, daily_pnl_records,
            total_discount_pnl, total_direction_pnl,
            total_put_cost, total_put_payout,
            total_gamma_pnl, total_theta_cost, total_commission,
        )

    def _simulate_minute_scalping(
        self,
        scalper: GammaScalper,
        day_bars: pd.DataFrame,
        put_position: Dict,
        day_close: float,
    ) -> Tuple[float, float, int]:
        """
        用分钟数据模拟Gamma Scalping。

        Returns (gamma_pnl, commission, rehedge_count)
        """
        total_gamma_pnl = 0.0
        total_commission = 0.0

        for _, bar in day_bars.iterrows():
            price = float(bar["close"])
            ts = str(bar["datetime"])

            # 重算Put delta
            dte = put_position.get("dte", 30)
            T = max(dte / 365.0, 1 / 365.0)
            put_delta = calc_delta(
                price, put_position["strike"], T,
                self.risk_free_rate, put_position.get("iv", 0.25), "P"
            )

            # 计算组合delta
            portfolio_delta = scalper.calc_portfolio_delta(
                self.futures_lots, put_position["lots"], put_delta
            )

            # 检查是否需要对冲
            signal = scalper.check_rehedge(price, ts, portfolio_delta)
            if signal:
                record = scalper.execute_rehedge(
                    price, ts, signal["action"], signal["volume"],
                    portfolio_delta,
                )
                total_gamma_pnl += record.gamma_pnl
                total_commission += record.commission

        return total_gamma_pnl, total_commission, scalper.today_rehedge_count

    def _estimate_daily_gamma_pnl(
        self,
        trade_date: str,
        put_position: Dict,
        iml_code: str,
    ) -> float:
        """
        无分钟数据时，用日线High-Low估算Gamma P&L。

        Gamma P&L ≈ 0.5 × Gamma × (daily_range)² / N
        N = 估算的日内对冲次数
        """
        try:
            df = self.db.query_df(
                f"SELECT high, low, close FROM futures_daily "
                f"WHERE ts_code='{iml_code}' AND trade_date='{trade_date}'"
            )
            if df is None or df.empty:
                return 0.0

            high = float(df["high"].iloc[0])
            low = float(df["low"].iloc[0])
            close_price = float(df["close"].iloc[0])
            daily_range = high - low

            if daily_range <= 0:
                return 0.0

            # 估算Gamma
            dte = max(put_position.get("dte", 30), 1)
            T = dte / 365.0
            gamma = calc_gamma(
                close_price, put_position["strike"], T,
                self.risk_free_rate, put_position.get("iv", 0.25)
            )

            # 估算: 假设日内对冲5次，每次捕获 range/sqrt(N) 的波动
            N_hedges = 5
            segment = daily_range / np.sqrt(N_hedges)
            gamma_pnl = 0.5 * gamma * segment ** 2 * N_hedges
            gamma_pnl *= OPTION_MULTIPLIER * put_position["lots"]

            return gamma_pnl

        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # 数据加载helpers
    # ------------------------------------------------------------------

    def _get_trade_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取区间内交易日列表"""
        try:
            df = self.db.query_df(
                f"SELECT DISTINCT trade_date FROM futures_daily "
                f"WHERE ts_code='IM.CFX' "
                f"AND trade_date >= '{start_date}' AND trade_date <= '{end_date}' "
                f"ORDER BY trade_date"
            )
            if df is not None and not df.empty:
                return df["trade_date"].tolist()
        except Exception:
            pass
        return []

    def _get_futures_close(self, trade_date: str, iml_code: str) -> Optional[float]:
        """获取期货收盘价"""
        try:
            r = self.db.query_df(
                f"SELECT close FROM futures_daily "
                f"WHERE ts_code='{iml_code}' AND trade_date='{trade_date}'"
            )
            if r is not None and not r.empty:
                return float(r["close"].iloc[0])
        except Exception:
            pass
        return None

    def _get_spot_close(self, trade_date: str) -> Optional[float]:
        """获取现货收盘价"""
        try:
            r = self.db.query_df(
                f"SELECT close FROM index_daily "
                f"WHERE ts_code='000852.SH' AND trade_date='{trade_date}'"
            )
            if r is not None and not r.empty:
                return float(r["close"].iloc[0])
        except Exception:
            pass
        return None

    def _get_put_close(self, trade_date: str, put_position: Dict) -> Optional[float]:
        """获取Put期权的当日收盘价"""
        ts_code = put_position.get("ts_code", "")
        if ts_code:
            try:
                r = self.db.query_df(
                    f"SELECT close FROM options_daily "
                    f"WHERE ts_code='{ts_code}' AND trade_date='{trade_date}'"
                )
                if r is not None and not r.empty:
                    val = float(r["close"].iloc[0])
                    if val > 0:
                        return val
            except Exception:
                pass

        # fallback: 用BS模型估算
        try:
            futures_close = self._get_futures_close(trade_date, "IM.CFX")
            if futures_close is None:
                return None
            dte = max(put_position.get("dte", 30), 1)
            T = dte / 365.0
            return bs_price(
                futures_close, put_position["strike"], T,
                self.risk_free_rate, put_position.get("iv", 0.25), "P"
            )
        except Exception:
            return None

    def _load_minute_data(
        self, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """加载IM 5分钟数据"""
        try:
            # 转换日期格式
            sd = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
            ed = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
            df = self.db.query_df(
                f"SELECT datetime, open, high, low, close, volume "
                f"FROM futures_min WHERE symbol='IM' "
                f"AND datetime >= '{sd}' AND datetime <= '{ed} 23:59:59' "
                f"ORDER BY datetime"
            )
            if df is not None and not df.empty:
                logger.info("加载 %d 条IM分钟数据", len(df))
                return df
        except Exception as e:
            logger.warning("加载分钟数据失败: %s", e)
        return None

    # ------------------------------------------------------------------
    # 结果构建
    # ------------------------------------------------------------------

    def _build_results(
        self,
        trades: List[Dict],
        equity_curve: List[Dict],
        daily_pnl: List[Dict],
        discount_pnl: float,
        direction_pnl: float,
        put_cost: float,
        put_payout: float,
        gamma_pnl: float,
        theta_cost: float,
        commission_cost: float,
    ) -> Dict:
        """构建回测结果字典"""
        if not equity_curve:
            return self._empty_result()

        eq_df = pd.DataFrame(equity_curve)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        final_capital = eq_df["capital"].iloc[-1]
        total_return = (final_capital - self.initial_capital) / self.initial_capital

        # 年化收益
        n_days = (pd.Timestamp(eq_df["date"].iloc[-1]) - pd.Timestamp(eq_df["date"].iloc[0])).days
        ann_return = (1 + total_return) ** (365 / max(n_days, 1)) - 1

        # 最大回撤
        peak = eq_df["capital"].cummax()
        drawdown = (peak - eq_df["capital"]) / peak
        max_dd = float(drawdown.max())

        # Sharpe
        if daily_pnl:
            daily_df = pd.DataFrame(daily_pnl)
            daily_returns = daily_df["total_pnl"] / self.initial_capital
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
            else:
                sharpe = 0.0
            max_daily_loss = float(daily_df["total_pnl"].min())
        else:
            sharpe = 0.0
            max_daily_loss = 0.0

        # 胜率
        if not trades_df.empty and "futures_pnl" in trades_df.columns:
            win_rate = float((trades_df["futures_pnl"] > 0).mean())
        else:
            win_rate = 0.0

        put_net_cost = put_cost - put_payout
        net_gamma_pnl = gamma_pnl - theta_cost - commission_cost
        theta_coverage = (gamma_pnl / theta_cost * 100) if theta_cost > 0 else 0.0

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "max_daily_loss": max_daily_loss,
            "n_trades": len(trades),
            "trades": trades_df,
            "equity_curve": eq_df,
            "daily_pnl": pd.DataFrame(daily_pnl) if daily_pnl else pd.DataFrame(),
            # 收益分解
            "discount_pnl": discount_pnl,
            "direction_pnl": direction_pnl,
            "put_cost": put_cost,
            "put_payout": put_payout,
            "put_net_cost": put_net_cost,
            "gamma_pnl": gamma_pnl,
            "theta_cost": theta_cost,
            "commission_cost": commission_cost,
            "net_gamma_pnl": net_gamma_pnl,
            "theta_coverage": theta_coverage,
        }

    def _build_sensitivity_table(self, results_by_config: Dict) -> pd.DataFrame:
        """构建对冲频率敏感性分析表"""
        rows = []
        for label, r in results_by_config.items():
            daily_df = r.get("daily_pnl", pd.DataFrame())
            total_hedges = 0
            if not daily_df.empty and "rehedge_count" in daily_df.columns:
                total_hedges = int(daily_df["rehedge_count"].sum())
            rows.append({
                "阈值": label,
                "总对冲次数": total_hedges,
                "Gamma利润": r.get("gamma_pnl", 0),
                "Theta成本": r.get("theta_cost", 0),
                "手续费": r.get("commission_cost", 0),
                "净Gamma利润": r.get("net_gamma_pnl", 0),
                "Theta覆盖率": f"{r.get('theta_coverage', 0):.1f}%",
                "总收益率": f"{r.get('total_return', 0) * 100:.2f}%",
            })
        return pd.DataFrame(rows)

    def _empty_result(self) -> Dict:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "max_daily_loss": 0.0,
            "n_trades": 0,
            "trades": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "daily_pnl": pd.DataFrame(),
            "discount_pnl": 0.0,
            "direction_pnl": 0.0,
            "put_cost": 0.0,
            "put_payout": 0.0,
            "put_net_cost": 0.0,
            "gamma_pnl": 0.0,
            "theta_cost": 0.0,
            "commission_cost": 0.0,
            "net_gamma_pnl": 0.0,
            "theta_coverage": 0.0,
        }

    # ------------------------------------------------------------------
    # 报告输出
    # ------------------------------------------------------------------

    def generate_report(self, results: Dict) -> None:
        """打印回测报告"""
        sep = "═" * 60
        thin = "─" * 60

        print()
        print(sep)
        print("  贴水+Gamma Scalping 回测报告")
        print(sep)

        if results.get("n_trades", 0) == 0:
            print("  无有效回测结果")
            print(sep)
            return

        # 总体绩效
        print()
        print(f"  总收益率      : {results['total_return']*100:>+8.2f}%")
        print(f"  年化收益率    : {results['annualized_return']*100:>+8.2f}%")
        print(f"  最大回撤      : {results['max_drawdown']*100:>8.2f}%")
        print(f"  Sharpe比率    : {results['sharpe']:>8.3f}")
        print(f"  胜率          : {results['win_rate']*100:>8.1f}%")
        print(f"  最大单日亏损  : {results['max_daily_loss']:>+10,.0f} 元")
        print(f"  交易笔数      : {results['n_trades']}")

        # 收益分解
        print()
        print(thin)
        print("  【收益分解】")
        print(thin)

        total_pnl = results["total_return"] * self.initial_capital
        print(f"  总收益           : {total_pnl:>+12,.0f} 元")
        print(f"  ├─ 贴水收敛       : {results['discount_pnl']:>+12,.0f} 元")
        print(f"  ├─ 方向性盈亏     : {results['direction_pnl']:>+12,.0f} 元")

        if results.get("gamma_pnl", 0) != 0 or results.get("theta_cost", 0) != 0:
            print(f"  ├─ Gamma Scalping : {results.get('net_gamma_pnl', 0):>+12,.0f} 元")
            print(f"  │   ├─ Gamma利润   : {results.get('gamma_pnl', 0):>+12,.0f} 元")
            print(f"  │   ├─ Theta成本   : {-results.get('theta_cost', 0):>+12,.0f} 元")
            print(f"  │   └─ 对冲手续费  : {-results.get('commission_cost', 0):>+12,.0f} 元")

        print(f"  └─ Put保护净成本  : {-results.get('put_net_cost', 0):>+12,.0f} 元")
        print(f"      ├─ 权利金支出  : {-results.get('put_cost', 0):>+12,.0f} 元")
        print(f"      └─ 到期/平仓回收: {results.get('put_payout', 0):>+12,.0f} 元")

        if results.get("theta_cost", 0) > 0:
            print()
            print(f"  Gamma/Theta 覆盖率: {results.get('theta_coverage', 0):.1f}%")
            print(f"  （100%表示Gamma利润完全覆盖Theta成本，Put保护免费）")

        # 交易记录
        trades = results.get("trades", pd.DataFrame())
        if not trades.empty:
            print()
            print(thin)
            print("  【合约选择统计】")
            print(thin)
            contracts_used = trades["contract"].unique()
            print(f"  使用过的期货合约: {', '.join(contracts_used)}")
            if len(trades) > 1:
                hold_days = []
                for _, t in trades.iterrows():
                    d = (pd.Timestamp(t["exit_date"]) - pd.Timestamp(t["entry_date"])).days
                    hold_days.append(d)
                print(f"  平均持仓天数    : {np.mean(hold_days):.0f}天")
            print(f"  换月次数        : {len(trades)}次")
            if "annualized_discount" in trades.columns:
                print(f"  平均年化贴水率  : {trades['annualized_discount'].mean()*100:.1f}%")

            print()
            print("  近期交易记录 (最新5笔):")
            print(f"  {'入场':<10} {'出场':<10} {'合约':<8} {'入场价':>7} {'出场价':>7} {'贴水率':>6} {'P&L':>10}")
            print(f"  {thin}")
            for _, row in trades.tail(5).iterrows():
                print(
                    f"  {row['entry_date']:<10} {row['exit_date']:<10} "
                    f"{row['contract']:<8} {row['entry_price']:>7.0f} {row['exit_price']:>7.0f} "
                    f"{row['annualized_discount']*100:>5.1f}% "
                    f"{row['futures_pnl']:>+10,.0f}"
                )

        # 敏感性分析
        if "sensitivity" in results:
            print()
            print(thin)
            print("  【对冲频率敏感性分析】")
            print(thin)
            sens = results["sensitivity"]
            if not sens.empty:
                print(sens.to_string(index=False))

        print()
        print(sep)
        print()


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="贴水+Gamma Scalping 回测"
    )
    parser.add_argument("--start", default="20220722", help="起始日期 YYYYMMDD")
    parser.add_argument("--end", default="20260319", help="结束日期 YYYYMMDD")
    parser.add_argument("--capital", type=float, default=5_000_000, help="初始资金")
    parser.add_argument("--lots", type=int, default=1, help="期货手数")
    parser.add_argument("--with-scalping", action="store_true", help="启用Gamma Scalping")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from config.config_loader import ConfigLoader
    from data.storage.db_manager import DBManager, get_db

    db = get_db()
    bt = DiscountGammaBacktester(
        db,
        initial_capital=args.capital,
        futures_lots=args.lots,
    )

    if args.with_scalping:
        results = bt.run_with_scalping(args.start, args.end)
        # 打印最优结果
        best = results.get("best_result", {})
        bt.generate_report(best)
        # 打印敏感性
        if "sensitivity" in results:
            print("对冲频率敏感性分析：")
            print(results["sensitivity"].to_string(index=False))
            print()
    else:
        results = bt.run_daily_only(args.start, args.end)
        bt.generate_report(results)


if __name__ == "__main__":
    main()
