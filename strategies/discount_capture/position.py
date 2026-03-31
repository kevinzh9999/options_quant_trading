"""
position.py
-----------
IM 贴水捕获策略仓位管理模块。

核心逻辑：
  - 做多 IM 期货（捕获贴水收益）
  - 买入 OTM Put 进行下行保护
  - 计算情景 P&L 分析
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DiscountPosition:
    """
    贴水捕获策略仓位管理器。

    Parameters
    ----------
    account_equity : float
        账户权益（元）
    max_allocation : float
        最大资金分配比例（默认 30%）
    max_loss_ratio : float
        最大可接受亏损比例（相对账户权益，默认 5%）
    """

    def __init__(
        self,
        account_equity: float,
        max_allocation: float = 0.3,
        max_loss_ratio: float = 0.05,
    ):
        self.equity = account_equity
        self.max_allocation = max_allocation
        self.max_loss_ratio = max_loss_ratio

    def calculate_futures_lots(
        self,
        futures_price: float,
        contract_multiplier: int = 200,
    ) -> int:
        """
        计算期货建仓手数。

        策略：
          allocated_capital = equity * max_allocation
          margin_per_lot = futures_price * contract_multiplier * 0.15  (15% 保证金)
          max_lots = int(allocated_capital * 0.70 / margin_per_lot)  (70% 保证金利用率)

        Parameters
        ----------
        futures_price : float
            期货价格
        contract_multiplier : int
            合约乘数（IM = 200）

        Returns
        -------
        int
            建议手数（最小 1 手）
        """
        if futures_price <= 0 or contract_multiplier <= 0:
            return 1

        allocated_capital = self.equity * self.max_allocation
        margin_per_lot = futures_price * contract_multiplier * 0.15
        if margin_per_lot <= 0:
            return 1

        max_lots_by_margin = int(allocated_capital * 0.70 / margin_per_lot)
        return max(1, max_lots_by_margin)

    def select_protective_put(
        self,
        futures_price: float,
        options_chain: pd.DataFrame,
        target_max_loss: Optional[float] = None,
        preferred_delta: float = -0.15,
        num_lots: int = 1,
    ) -> dict:
        """
        从期权链中选择最优保护性 Put。

        Parameters
        ----------
        futures_price : float
            期货入场价格
        options_chain : pd.DataFrame
            期权链，需含列: exercise_price, call_put, close
            可选列: volume, delta, iv
        target_max_loss : float, optional
            目标最大亏损金额（元），None 表示不限制
        preferred_delta : float
            目标 Delta（负值，如 -0.15 表示 OTM 15%）
        num_lots : int
            期货手数（用于计算总保护成本）

        Returns
        -------
        dict
            best Put 候选信息：strike, premium, protection_cost,
            max_loss, protection_ratio, notes
        """
        if options_chain is None or options_chain.empty:
            return {
                "strike": None,
                "premium": 0.0,
                "protection_cost": 0.0,
                "max_loss": None,
                "protection_ratio": 0.0,
                "notes": "无期权链数据",
            }

        CONTRACT_MULT = 200

        try:
            # 过滤 Put
            put_df = options_chain[
                options_chain["call_put"].str.upper().isin(["P", "PUT"])
            ].copy()

            # 过滤低成交量
            if "volume" in put_df.columns:
                put_df = put_df[put_df["volume"] >= 100]

            if put_df.empty:
                return {
                    "strike": None,
                    "premium": 0.0,
                    "protection_cost": 0.0,
                    "max_loss": None,
                    "protection_ratio": 0.0,
                    "notes": "无足够流动性的 Put 期权",
                }

            # 计算各行权价的最大亏损
            # 多头期货 + 保护性 Put：
            #   期货亏损下限 = (K - futures_price) * multiplier
            #   put_cost = put_premium * multiplier * num_lots
            #   当价格跌至 K 时（Put 平值），总亏损 = (K - futures_price)*mult + put_cost
            #   即 = -(futures_price - K) * mult - put_cost
            put_df = put_df.copy()
            put_df["strike"] = put_df["exercise_price"].astype(float)
            put_df["premium"] = put_df["close"].astype(float)
            put_df["protection_cost_per_lot"] = put_df["premium"] * CONTRACT_MULT
            put_df["total_protection_cost"] = put_df["protection_cost_per_lot"] * num_lots

            # 最大亏损（在行权价处，put 平值不盈利）
            put_df["floor_loss_per_lot"] = (
                (futures_price - put_df["strike"]) * CONTRACT_MULT
                + put_df["protection_cost_per_lot"]
            )
            # 只考虑行权价 < 期货价的 OTM Put
            put_df = put_df[put_df["strike"] < futures_price]

            if put_df.empty:
                return {
                    "strike": None,
                    "premium": 0.0,
                    "protection_cost": 0.0,
                    "max_loss": None,
                    "protection_ratio": 0.0,
                    "notes": "无 OTM Put 选项",
                }

            # 按目标最大亏损过滤
            if target_max_loss is not None and target_max_loss > 0:
                eligible = put_df[
                    put_df["floor_loss_per_lot"] * num_lots <= target_max_loss
                ]
                if not eligible.empty:
                    put_df = eligible

            # 如果有 delta 列，优先选最接近目标 delta 的
            if "delta" in put_df.columns and put_df["delta"].notna().any():
                put_df["delta_dist"] = abs(put_df["delta"] - preferred_delta)
                best_row = put_df.loc[put_df["delta_dist"].idxmin()]
            else:
                # 否则按行权价选：目标是 futures_price * (1 + preferred_delta)
                target_strike = futures_price * (1 + preferred_delta)
                put_df["strike_dist"] = abs(put_df["strike"] - target_strike)
                best_row = put_df.loc[put_df["strike_dist"].idxmin()]

            strike = float(best_row["strike"])
            premium = float(best_row["premium"])
            protection_cost = float(best_row["total_protection_cost"])
            floor_loss = float(best_row["floor_loss_per_lot"]) * num_lots

            # 估算贴水收益（粗略：用年化贴水率 * 期货名义价值估算）
            # 这里作为占位符，调用方可以传入更精确的值
            expected_discount_pnl = abs(futures_price * 0.05) * CONTRACT_MULT * num_lots

            protection_ratio = (
                expected_discount_pnl / protection_cost
                if protection_cost > 0 else 0.0
            )

            iv_str = ""
            if "iv" in put_df.columns and pd.notna(best_row.get("iv")):
                iv_str = f"  IV={best_row['iv']*100:.1f}%" if best_row["iv"] > 0 else ""

            return {
                "strike": strike,
                "premium": premium,
                "protection_cost": protection_cost,
                "max_loss": floor_loss,
                "protection_ratio": protection_ratio,
                "notes": f"MO-P-{strike:.0f}  成本={premium:.1f}元/张{iv_str}",
                "floor_loss_per_lot": float(best_row["floor_loss_per_lot"]),
                "all_candidates": put_df,
            }

        except Exception as e:
            logger.warning("选择保护 Put 失败: %s", e)
            return {
                "strike": None,
                "premium": 0.0,
                "protection_cost": 0.0,
                "max_loss": None,
                "protection_ratio": 0.0,
                "notes": f"计算失败: {e}",
            }

    def build_protection_comparison(
        self,
        put_index: dict,
        futures_price: float,
        disc_pnl_per_lot: float,
        candidate_strikes: list,
        spread_widths: list = None,
        option_mult: int = 100,
        futures_mult: int = 200,
    ) -> list:
        """
        为指定行权价列表生成裸 Put 和 Put Spread 两种方案的比较数据。

        Parameters
        ----------
        put_index : dict
            {exercise_price (float): premium (float)}，仅包含 Put
        futures_price : float
            期货入场价格
        disc_pnl_per_lot : float
            预期贴水收益（元/手），正值
        candidate_strikes : list of int/float
            买入行权价候选列表（K1）
        spread_widths : list of int
            Price spread widths for K2 = K1 - width。默认 [400, 600]
        option_mult : int
            期权合约乘数（MO = 100）
        futures_mult : int
            期货合约乘数（IM = 200）

        Returns
        -------
        list of dict
            每条记录含：scheme, buy_strike, sell_strike, buy_premium,
            sell_premium, net_cost, max_protection, max_loss,
            net_disc_pnl, protection_ratio, available
        """
        if spread_widths is None:
            spread_widths = [400, 600]

        rows = []

        for k1 in candidate_strikes:
            k1 = float(k1)
            prem1 = put_index.get(k1)

            # ── 裸 Put ──────────────────────────────────────────────────────
            if prem1 is not None and prem1 > 0:
                net_cost = prem1 * option_mult
                max_loss = (k1 - futures_price) * futures_mult - net_cost  # 负值=亏损
                rows.append({
                    "scheme":           f"裸Put P-{k1:.0f}",
                    "buy_strike":       k1,
                    "sell_strike":      None,
                    "buy_premium":      prem1,
                    "sell_premium":     0.0,
                    "net_cost":         net_cost,
                    "max_protection":   None,          # 无上限
                    "max_loss":         max_loss,
                    "net_disc_pnl":     disc_pnl_per_lot - net_cost,
                    "protection_ratio": disc_pnl_per_lot / net_cost if net_cost > 0 else 0.0,
                    "available":        True,
                })
            else:
                rows.append({
                    "scheme":       f"裸Put P-{k1:.0f}",
                    "buy_strike":   k1,
                    "sell_strike":  None,
                    "available":    False,
                })

            # ── Put Spread ───────────────────────────────────────────────────
            for width in spread_widths:
                k2 = k1 - float(width)
                prem2 = put_index.get(k2)

                if prem1 is None or prem1 <= 0 or prem2 is None or prem2 <= 0:
                    rows.append({
                        "scheme":       f"Spread {k1:.0f}/{k2:.0f}",
                        "buy_strike":   k1,
                        "sell_strike":  k2,
                        "available":    False,
                    })
                    continue

                net_cost = (prem1 - prem2) * option_mult
                if net_cost <= 0:
                    continue   # 价差为负（极罕见），跳过
                max_protection = (k1 - k2) * futures_mult   # spread 封顶保护金额
                max_loss = (k1 - futures_price) * futures_mult - net_cost  # 负值=亏损
                rows.append({
                    "scheme":           f"Spread {k1:.0f}/{k2:.0f}",
                    "buy_strike":       k1,
                    "sell_strike":      k2,
                    "buy_premium":      prem1,
                    "sell_premium":     prem2,
                    "net_cost":         net_cost,
                    "max_protection":   max_protection,
                    "max_loss":         max_loss,
                    "net_disc_pnl":     disc_pnl_per_lot - net_cost,
                    "protection_ratio": disc_pnl_per_lot / net_cost if net_cost > 0 else 0.0,
                    "available":        True,
                })

        return rows

    def calculate_strategy_pnl_scenarios(
        self,
        futures_price: float,
        put_strike: float,
        put_premium: float,
        spot_price: float,
        futures_lots: int = 1,
        put_lots: int = 1,
    ) -> pd.DataFrame:
        """
        计算策略到期情景 P&L。

        Parameters
        ----------
        futures_price : float
            期货建仓价格
        put_strike : float
            保护 Put 行权价
        put_premium : float
            保护 Put 权利金（元/张）
        spot_price : float
            当前现货价格
        futures_lots : int
            期货手数
        put_lots : int
            Put 手数

        Returns
        -------
        pd.DataFrame
            columns: spot_at_expiry, futures_pnl, put_pnl, total_pnl,
                     total_return_pct
        """
        CONTRACT_MULT = 200

        # 情景区间：当前现货 ±20%，每 1% 一个点
        spot_range = spot_price * np.linspace(0.80, 1.20, 41)

        futures_pnl = (spot_range - futures_price) * CONTRACT_MULT * futures_lots

        # Put 到期 P&L = max(K - S_T, 0) * mult * lots - premium * mult * lots
        put_payoff = np.maximum(put_strike - spot_range, 0.0) * CONTRACT_MULT * put_lots
        put_cost = put_premium * CONTRACT_MULT * put_lots
        put_pnl = put_payoff - put_cost

        total_pnl = futures_pnl + put_pnl
        notional = futures_price * CONTRACT_MULT * futures_lots
        total_return_pct = total_pnl / notional * 100 if notional > 0 else np.zeros(len(spot_range))

        return pd.DataFrame({
            "spot_at_expiry": spot_range,
            "futures_pnl": futures_pnl,
            "put_pnl": put_pnl,
            "total_pnl": total_pnl,
            "total_return_pct": total_return_pct,
        })
