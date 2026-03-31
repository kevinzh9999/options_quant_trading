"""
black_scholes.py
----------------
Black-Scholes-Merton 期权定价模型（类封装版本）。

相比 implied_vol.py 的函数接口，本模块：
- 增加连续分红率 q 参数支持（e.g. 股指期货隐含分红）
- 提供统一的 BlackScholes 类，方便策略层调用
- IV 反推使用 Newton-Raphson + 二分法兜底

公式（含连续分红率）：
    d1 = [ln(S/K) + (r - q + σ²/2)·T] / (σ·√T)
    d2 = d1 - σ·√T
    Call = S·e^{-qT}·N(d1) - K·e^{-rT}·N(d2)
    Put  = K·e^{-rT}·N(-d2) - S·e^{-qT}·N(-d1)
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.stats import norm

# IV 搜索边界
_IV_MIN = 1e-6
_IV_MAX = 5.0
_IV_INIT = 0.3       # 初始猜测 30%


class BlackScholes:
    """
    Black-Scholes-Merton 期权定价模型。

    假设：
    - 标的资产价格服从几何布朗运动
    - 欧式期权（A 股股指期权 IO/MO 都是欧式）
    - 支持连续分红率 q（股指期货可用隐含分红率近似）

    所有方法为 @staticmethod，无需实例化。

    核心公式：
        d1 = [ln(S/K) + (r - q + σ²/2)·T] / (σ·√T)
        d2 = d1 - σ·√T
        Call = S·e^{-qT}·N(d1) - K·e^{-rT}·N(d2)
        Put  = K·e^{-rT}·N(-d2) - S·e^{-qT}·N(-d1)
    """

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _d1_d2(
        S: float, K: float, T: float, r: float, sigma: float, q: float
    ) -> Tuple[float, float]:
        """计算 d1, d2（含分红率 q）。"""
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        return d1, d2

    @staticmethod
    def _intrinsic(S: float, K: float, r: float, q: float, T: float,
                   option_type: str) -> float:
        """到期内在价值（含折现）：用于 T≤0 或 sigma≤0 的边界处理。"""
        if T <= 0:
            if option_type == "C":
                return max(S - K, 0.0)
            else:
                return max(K - S, 0.0)
        # sigma=0 时：折现内在价值
        if option_type == "C":
            return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
        else:
            return max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)

    # ------------------------------------------------------------------
    # 定价
    # ------------------------------------------------------------------

    @staticmethod
    def price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "C",
    ) -> float:
        """
        计算期权理论价格（BSM 公式）。

        Parameters
        ----------
        S : float
            标的当前价格（指数点位）
        K : float
            行权价
        T : float
            到期时间（年化，如 30天 = 30/365）
        r : float
            无风险利率（年化，如 0.02 表示 2%）
        sigma : float
            波动率（年化，如 0.20 表示 20%）
        q : float
            连续分红率（年化，默认 0）
        option_type : str
            "C" 认购 / "P" 认沽

        Returns
        -------
        float
            期权理论价格

        边界处理：
        - S <= 0 或 K <= 0 → 返回 0.0
        - T <= 0 → 返回内在价值 max(S-K, 0) 或 max(K-S, 0)
        - sigma <= 0 → 返回折现内在价值
        """
        if S <= 0 or K <= 0:
            return 0.0
        if T <= 0 or sigma <= 0:
            return BlackScholes._intrinsic(S, K, r, q, T, option_type)

        d1, d2 = BlackScholes._d1_d2(S, K, T, r, sigma, q)
        disc_S = S * math.exp(-q * T)
        disc_K = K * math.exp(-r * T)

        if option_type == "C":
            return disc_S * norm.cdf(d1) - disc_K * norm.cdf(d2)
        else:
            return disc_K * norm.cdf(-d2) - disc_S * norm.cdf(-d1)

    # ------------------------------------------------------------------
    # Greeks
    # ------------------------------------------------------------------

    @staticmethod
    def delta(
        S: float, K: float, T: float, r: float, sigma: float,
        q: float = 0.0, option_type: str = "C",
    ) -> float:
        """
        Delta = ∂V/∂S。

        Call: e^{-qT}·N(d1)    ∈ (0, 1)
        Put:  -e^{-qT}·N(-d1)  ∈ (-1, 0)
        """
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            if option_type == "C":
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0

        d1, _ = BlackScholes._d1_d2(S, K, T, r, sigma, q)
        e_qT = math.exp(-q * T)
        if option_type == "C":
            return e_qT * norm.cdf(d1)
        else:
            return -e_qT * norm.cdf(-d1)

    @staticmethod
    def gamma(
        S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
    ) -> float:
        """
        Gamma = ∂²V/∂S²（认购认沽相同）。

        Gamma = e^{-qT}·N'(d1) / (S·σ·√T)
        """
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return 0.0

        d1, _ = BlackScholes._d1_d2(S, K, T, r, sigma, q)
        return math.exp(-q * T) * norm.pdf(d1) / (S * sigma * math.sqrt(T))

    @staticmethod
    def theta(
        S: float, K: float, T: float, r: float, sigma: float,
        q: float = 0.0, option_type: str = "C",
    ) -> float:
        """
        Theta = ∂V/∂T（每日衰减）。

        返回**每天**的 theta（已除以 365），通常为负。
        即"持有 1 天后期权价值变化量"。

        Call: -[S·e^{-qT}·N'(d1)·σ / (2√T)] + q·S·e^{-qT}·N(d1)
              - r·K·e^{-rT}·N(d2)
        Put:  -[S·e^{-qT}·N'(d1)·σ / (2√T)] - q·S·e^{-qT}·N(-d1)
              + r·K·e^{-rT}·N(-d2)
        """
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return 0.0

        d1, d2 = BlackScholes._d1_d2(S, K, T, r, sigma, q)
        sqrt_T = math.sqrt(T)
        disc_S = S * math.exp(-q * T)
        disc_K = K * math.exp(-r * T)

        # 公共项（时间价值的时间导数，总为负）
        common = -disc_S * norm.pdf(d1) * sigma / (2.0 * sqrt_T)

        if option_type == "C":
            theta_annual = (
                common
                + q * disc_S * norm.cdf(d1)
                - r * disc_K * norm.cdf(d2)
            )
        else:
            theta_annual = (
                common
                - q * disc_S * norm.cdf(-d1)
                + r * disc_K * norm.cdf(-d2)
            )

        return theta_annual / 365.0  # 转换为每日 theta

    @staticmethod
    def vega(
        S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
    ) -> float:
        """
        Vega = ∂V/∂σ（认购认沽相同）。

        返回波动率变化 **1%（即 0.01）** 对应的期权价格变化：
        vega_per_1pct = S·e^{-qT}·N'(d1)·√T / 100
        """
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return 0.0

        d1, _ = BlackScholes._d1_d2(S, K, T, r, sigma, q)
        vega_raw = S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T)
        return vega_raw / 100.0

    # ------------------------------------------------------------------
    # 隐含波动率反推
    # ------------------------------------------------------------------

    @staticmethod
    def implied_volatility(
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
        option_type: str = "C",
        max_iter: int = 100,
        tol: float = 1e-8,
    ) -> float:
        """
        用 Newton-Raphson + 二分法 从市场价格反推隐含波动率。

        Parameters
        ----------
        market_price : float
            期权市场价格
        S, K, T, r, q : float
            同 price()
        option_type : str
            "C" / "P"
        max_iter : int
            Newton 法最大迭代次数
        tol : float
            收敛精度

        Returns
        -------
        float
            隐含波动率（年化），如 0.25 表示 25%。
            无法计算时返回 float('nan')。

        边界处理：
        - market_price < 内在价值（无套利下界） → NaN
        - T <= 0 → NaN
        - S <= 0 或 K <= 0 → NaN
        - 深度虚值（价格接近 0）可能无法收敛 → NaN
        - Newton 不收敛 → 自动切换二分法
        """
        # 基本边界检查
        if T <= 0 or S <= 0 or K <= 0:
            return float("nan")
        if market_price < 0:
            return float("nan")

        # 内在价值下界检查
        intrinsic = BlackScholes._intrinsic(S, K, r, q, T, option_type)
        if market_price < intrinsic - tol:
            return float("nan")

        # 价格极小（深度虚值）时直接返回 NaN 避免无意义的迭代
        if market_price < 1e-10:
            return float("nan")

        def objective(sigma: float) -> float:
            return BlackScholes.price(S, K, T, r, sigma, q, option_type) - market_price

        def vega_val(sigma: float) -> float:
            return BlackScholes.vega(S, K, T, r, sigma, q) * 100.0  # 恢复 raw vega

        # Newton-Raphson
        sigma = _IV_INIT
        for _ in range(max_iter):
            price_diff = objective(sigma)
            if abs(price_diff) < tol:
                return sigma
            v = vega_val(sigma)
            if abs(v) < 1e-12:
                # Vega 太小，切换二分法
                break
            sigma_new = sigma - price_diff / v
            if sigma_new <= _IV_MIN or sigma_new >= _IV_MAX:
                # 越界，切换二分法
                break
            sigma = sigma_new

        # 二分法兜底
        return BlackScholes._bisect_iv(objective, tol, max_iter * 10)

    @staticmethod
    def _bisect_iv(
        objective,
        tol: float,
        max_iter: int,
    ) -> float:
        """二分法求解 IV，搜索范围 [_IV_MIN, _IV_MAX]。"""
        lo, hi = _IV_MIN, _IV_MAX

        # 检查端点符号
        f_lo = objective(lo)
        f_hi = objective(hi)
        if f_lo * f_hi > 0:
            return float("nan")

        for _ in range(max_iter):
            mid = (lo + hi) / 2.0
            f_mid = objective(mid)
            if abs(f_mid) < tol:
                return mid
            if f_lo * f_mid <= 0:
                hi = mid
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid

        return (lo + hi) / 2.0
