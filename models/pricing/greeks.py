"""
greeks.py
---------
职责：计算期权 Greeks（风险敞口指标）。
- Delta：期权价值对标的价格的一阶偏导
- Gamma：Delta 对标的价格的偏导（凸性）
- Theta：期权价值对时间的偏导（时间衰减）
- Vega：期权价值对波动率的偏导
- Rho：期权价值对利率的偏导

所有 Greeks 基于 Black-Scholes 解析公式计算（欧式期权）。
对 Portfolio 级别提供汇总功能。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from typing import Dict, List, Optional

from models.pricing.implied_vol import OptionType, bs_d1_d2

logger = logging.getLogger(__name__)

# A 股股指期权合约乘数
DEFAULT_MULTIPLIER = 100  # IO / MO 均为 100 元/点


@dataclass
class Greeks:
    """单个期权的 Greeks"""
    ts_code: str
    delta: float       # 无量纲，Call ∈ (0,1)，Put ∈ (-1,0)
    gamma: float       # 单位：1/点位（每点价格变动的 Delta 变化量）
    theta: float       # 单位：点位/天（每日时间衰减，通常为负）
    vega: float        # 单位：点位/（1%波动率变动）
    rho: float         # 单位：点位/（1%利率变动）


@dataclass
class PortfolioGreeks:
    """投资组合级别的 Greeks 汇总"""
    net_delta: float           # 净 Delta（已乘合约乘数）
    net_gamma: float           # 净 Gamma
    net_theta: float           # 净 Theta（每日时间价值损耗，正数为收益）
    net_vega: float            # 净 Vega（每1%波动率变动的盈亏）
    delta_dollars: float       # Delta 名义敞口（元）= net_delta × 标的价格 × 合约乘数
    vega_dollars: float        # Vega 名义敞口（元/1%波动率）


# ======================================================================
# 单合约 Greeks 计算
# ======================================================================

def calc_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType | str,
) -> float:
    """
    计算期权 Delta。

    Call: N(d1)       ∈ (0, 1)
    Put:  N(d1) - 1   ∈ (-1, 0)

    Parameters
    ----------
    S, K, T, r, sigma : float
        标的价、行权价、到期时间（年）、无风险利率、波动率
    option_type : OptionType | str
        'C' 或 'P'

    Returns
    -------
    float
        Delta 值
    """
    if T <= 0:
        # 到期时 Delta 为 0 或 ±1
        if str(option_type) == OptionType.CALL or option_type == "C":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1, _ = bs_d1_d2(S, K, T, r, sigma)
    nd1 = float(stats.norm.cdf(d1))

    if str(option_type) == OptionType.CALL or option_type == "C":
        return nd1
    else:
        return nd1 - 1.0


def calc_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    计算期权 Gamma（Call 和 Put 相同）。

    Gamma = N'(d1) / (S·σ·√T)

    Returns
    -------
    float
        Gamma 值（永远为正）
    """
    if T <= 0:
        return 0.0

    d1, _ = bs_d1_d2(S, K, T, r, sigma)
    return float(stats.norm.pdf(d1)) / (S * sigma * np.sqrt(T))


def calc_theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType | str,
    trading_days: bool = True,
) -> float:
    """
    计算期权 Theta（时间衰减）。

    Call: -(S·N'(d1)·σ)/(2·√T) - r·K·e^{-rT}·N(d2)
    Put:  -(S·N'(d1)·σ)/(2·√T) + r·K·e^{-rT}·N(-d2)

    Parameters
    ----------
    S, K, T, r, sigma : float
    option_type : OptionType | str
    trading_days : bool
        True → 以交易日为单位（÷252）；False → 以自然日为单位（÷365）

    Returns
    -------
    float
        Theta 值（通常为负）
    """
    if T <= 0:
        return 0.0

    d1, d2 = bs_d1_d2(S, K, T, r, sigma)
    n_prime_d1 = stats.norm.pdf(d1)
    discount = np.exp(-r * T)
    divisor = 252.0 if trading_days else 365.0

    # 公共项（总为负）
    common = -(S * n_prime_d1 * sigma) / (2.0 * np.sqrt(T))

    if str(option_type) == OptionType.CALL or option_type == "C":
        theta_annual = common - r * K * discount * stats.norm.cdf(d2)
    else:
        theta_annual = common + r * K * discount * stats.norm.cdf(-d2)

    return float(theta_annual / divisor)


def calc_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    计算期权 Vega（Call 和 Put 相同）。

    Vega = S·√T·N'(d1) / 100   （每1%波动率变动的期权价值变动）

    Returns
    -------
    float
        Vega 值（永远为正）
    """
    if T <= 0:
        return 0.0

    d1, _ = bs_d1_d2(S, K, T, r, sigma)
    vega_raw = S * np.sqrt(T) * float(stats.norm.pdf(d1))
    return vega_raw / 100.0  # 标准化为每 1% 波动率变动


def calc_rho(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType | str,
) -> float:
    """
    计算期权 Rho（利率敏感度）。

    Call: K·T·e^{-rT}·N(d2) / 100
    Put:  -K·T·e^{-rT}·N(-d2) / 100

    Returns
    -------
    float
        Rho 值，以 1% 利率变动为基准
    """
    if T <= 0:
        return 0.0

    _, d2 = bs_d1_d2(S, K, T, r, sigma)
    discount = np.exp(-r * T)

    if str(option_type) == OptionType.CALL or option_type == "C":
        rho_raw = K * T * discount * stats.norm.cdf(d2)
    else:
        rho_raw = -K * T * discount * stats.norm.cdf(-d2)

    return float(rho_raw / 100.0)


def calc_all_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType | str,
    ts_code: str = "",
) -> Greeks:
    """
    一次性计算所有 Greeks（复用 d1/d2 避免重复计算）。

    Parameters
    ----------
    ts_code : str
        合约代码（用于标识，可选）

    Returns
    -------
    Greeks
        包含 delta, gamma, theta, vega, rho 的数据类
    """
    return Greeks(
        ts_code=ts_code,
        delta=calc_delta(S, K, T, r, sigma, option_type),
        gamma=calc_gamma(S, K, T, r, sigma),
        theta=calc_theta(S, K, T, r, sigma, option_type),
        vega=calc_vega(S, K, T, r, sigma),
        rho=calc_rho(S, K, T, r, sigma, option_type),
    )


# ======================================================================
# 组合级别 Greeks 汇总
# ======================================================================

def calc_portfolio_greeks(
    positions: pd.DataFrame,
    spot_price: float,
    risk_free_rate: float,
    contract_multiplier: int = DEFAULT_MULTIPLIER,
) -> PortfolioGreeks:
    """
    计算投资组合的净 Greeks。

    Parameters
    ----------
    positions : pd.DataFrame
        持仓数据，需含 ts_code, strike_price, call_put, expire_date,
        net_position（净持仓手数，正=多，负=空）, iv 列
    spot_price : float
        标的现价
    risk_free_rate : float
        无风险利率
    contract_multiplier : int
        合约乘数（IO/MO 均为 100）

    Returns
    -------
    PortfolioGreeks

    Notes
    -----
    - 净 Delta = sum(delta_i × net_position_i × multiplier)
    - 做空期权时 net_position 为负，Vega 贡献为负
    """
    if positions.empty:
        return PortfolioGreeks(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    ref_date = pd.Timestamp.today()
    net_delta = net_gamma = net_theta = net_vega = 0.0

    for _, row in positions.iterrows():
        try:
            expire_ts = pd.Timestamp(str(row["expire_date"]))
            T = max((expire_ts - ref_date).days / 365.0, 0.0)
            sigma = float(row.get("iv", 0.20))
            if sigma <= 0 or np.isnan(sigma):
                sigma = 0.20
            qty = float(row.get("net_position", 0))

            g = calc_all_greeks(
                S=spot_price,
                K=float(row["strike_price"]),
                T=T,
                r=risk_free_rate,
                sigma=sigma,
                option_type=str(row["call_put"]),
                ts_code=str(row.get("ts_code", "")),
            )
            net_delta += g.delta * qty * contract_multiplier
            net_gamma += g.gamma * qty * contract_multiplier
            net_theta += g.theta * qty * contract_multiplier
            net_vega += g.vega * qty * contract_multiplier
        except Exception as e:
            logger.warning("Greeks 计算异常 %s: %s", row.get("ts_code", ""), e)

    delta_dollars = net_delta * spot_price
    vega_dollars = net_vega * spot_price  # Vega per 1% vol move in dollar terms

    return PortfolioGreeks(
        net_delta=net_delta,
        net_gamma=net_gamma,
        net_theta=net_theta,
        net_vega=net_vega,
        delta_dollars=delta_dollars,
        vega_dollars=vega_dollars,
    )


# ======================================================================
# GreeksCalculator：面向策略层的组合 Greeks 接口
# ======================================================================

_DEFAULT_IV = 0.20   # 无 IV 时的缺省波动率（20%）


class GreeksCalculator:
    """
    计算期权持仓组合的汇总 Greeks。

    Parameters
    ----------
    trade_date : str, optional
        当前交易日 YYYYMMDD；为 None 时以今天为基准计算剩余到期时间。
    dividend_yield : float
        标的连续分红率，默认 0。
    """

    def __init__(
        self,
        trade_date: Optional[str] = None,
        dividend_yield: float = 0.0,
    ) -> None:
        self._ref_date = (
            pd.Timestamp(trade_date) if trade_date else pd.Timestamp.today().normalize()
        )
        self.dividend_yield = dividend_yield

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def calculate_position_greeks(
        self,
        positions: List[Dict],
        underlying_price: float,
        risk_free_rate: float = 0.02,
    ) -> Dict:
        """
        计算期权持仓组合的 Greeks。

        Parameters
        ----------
        positions : list[dict]
            持仓列表，每个元素须含：
                strike_price : float   行权价
                call_put     : str     "C" 或 "P"
                expire_date  : str     到期日 YYYYMMDD
                volume       : int     净持仓手数（正=多头，负=空头）
                contract_unit: int     合约乘数（IO=100）
            可选字段：
                iv           : float   该合约的隐含波动率，缺省用 0.20
        underlying_price : float
            标的当前价格
        risk_free_rate : float
            无风险利率（年化），默认 0.02

        Returns
        -------
        dict
            net_delta        : float  — 组合净 Delta（标的价每变动1点的组合盈亏，元）
            net_gamma        : float  — 组合净 Gamma（元/点²）
            net_theta        : float  — 组合净 Theta（元/天，通常为负）
            net_vega         : float  — 组合净 Vega（元/1%波动率变动）
            positions_detail : list   — 每条持仓的明细 Greeks（见下）

        positions_detail 每元素字段：
            strike_price, call_put, expire_date, volume, contract_unit,
            T (年化剩余期), iv,
            delta, gamma, theta, vega, rho,
            position_delta (= delta × volume × contract_unit),
            position_gamma, position_theta, position_vega
        """
        from models.pricing.black_scholes import BlackScholes

        net_delta = net_gamma = net_theta = net_vega = 0.0
        details: List[Dict] = []

        for pos in positions:
            try:
                detail = self._calc_single(pos, underlying_price, risk_free_rate, BlackScholes)
            except Exception as e:
                logger.warning(
                    "Greeks 计算异常 K=%s %s: %s",
                    pos.get("strike_price"), pos.get("call_put"), e,
                )
                detail = self._zero_detail(pos)

            net_delta += detail["position_delta"]
            net_gamma += detail["position_gamma"]
            net_theta += detail["position_theta"]
            net_vega += detail["position_vega"]
            details.append(detail)

        return {
            "net_delta": net_delta,
            "net_gamma": net_gamma,
            "net_theta": net_theta,
            "net_vega": net_vega,
            "positions_detail": details,
        }

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _calc_single(
        self,
        pos: Dict,
        S: float,
        r: float,
        BlackScholes,
    ) -> Dict:
        """计算单条持仓的 Greeks 明细。"""
        # 允许每条持仓覆盖标的价格（如同到期月份的期货合约价格）
        S = float(pos.get("underlying_price", S))
        K = float(pos["strike_price"])
        cp = str(pos["call_put"]).upper()
        expire_ts = pd.Timestamp(str(pos["expire_date"]))
        T = max((expire_ts - self._ref_date).days / 365.0, 0.0)
        iv = float(pos.get("iv") or _DEFAULT_IV)
        if np.isnan(iv) or iv <= 0:
            iv = _DEFAULT_IV
        volume = int(pos.get("volume", 0))
        unit = int(pos.get("contract_unit", DEFAULT_MULTIPLIER))

        delta = BlackScholes.delta(S, K, T, r, iv, self.dividend_yield, cp)
        gamma = BlackScholes.gamma(S, K, T, r, iv, self.dividend_yield)
        theta = BlackScholes.theta(S, K, T, r, iv, self.dividend_yield, cp)   # 每日
        vega = BlackScholes.vega(S, K, T, r, iv, self.dividend_yield)          # per 1%
        rho = calc_rho(S, K, T, r, iv, cp)

        pos_delta = delta * volume * unit
        pos_gamma = gamma * volume * unit
        pos_theta = theta * volume * unit
        pos_vega = vega * volume * unit

        return {
            "strike_price": K,
            "call_put": cp,
            "expire_date": str(pos["expire_date"]),
            "volume": volume,
            "contract_unit": unit,
            "T": T,
            "iv": iv,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho,
            "position_delta": pos_delta,
            "position_gamma": pos_gamma,
            "position_theta": pos_theta,
            "position_vega": pos_vega,
        }

    @staticmethod
    def _zero_detail(pos: Dict) -> Dict:
        """返回全零的明细（计算异常时的安全降级）。"""
        return {
            "strike_price": float(pos.get("strike_price", 0)),
            "call_put": str(pos.get("call_put", "")),
            "expire_date": str(pos.get("expire_date", "")),
            "volume": int(pos.get("volume", 0)),
            "contract_unit": int(pos.get("contract_unit", DEFAULT_MULTIPLIER)),
            "T": 0.0, "iv": _DEFAULT_IV,
            "delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0,
            "position_delta": 0.0, "position_gamma": 0.0,
            "position_theta": 0.0, "position_vega": 0.0,
        }
