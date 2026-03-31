"""
ou_process.py
-------------
职责：Ornstein-Uhlenbeck (OU) 过程参数估计。

OU 过程是均值回归价差的连续时间模型：
  dX_t = κ(θ - X_t)dt + σ_OU · dW_t

其中：
  κ（kappa）：均值回归速度
  θ（theta_ou）：长期均值
  σ_OU：扩散系数

半衰期 τ₁/₂ = ln(2) / κ，表示偏离均值后恢复到一半偏离的期望时间。

应用场景：
- 价差套利持仓周期选择（半衰期决定持仓时间）
- 均值回归策略入场条件评估
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OUParams:
    """
    OU 过程参数估计结果。

    Attributes
    ----------
    kappa : float
        均值回归速度（年化）
    theta : float
        长期均值
    sigma : float
        扩散系数（OU 波动率）
    half_life : float
        均值回归半衰期（交易日）= ln(2) / kappa × (1/252)
    r_squared : float
        OLS 拟合 R²（衡量 OU 模型对价差的解释力）
    is_mean_reverting : bool
        是否显著均值回归（kappa > 0 且统计显著）
    """
    kappa: float
    theta: float
    sigma: float
    half_life: float
    r_squared: float = 0.0
    is_mean_reverting: bool = True


def fit_ou_process(
    spread: pd.Series,
    dt: float = 1 / 252,
) -> OUParams:
    """
    通过 OLS 回归估计 OU 过程参数（离散化方法）。

    Parameters
    ----------
    spread : pd.Series
        价差序列（已对齐的时间序列）
    dt : float
        时间步长（年），默认 1/252（一个交易日）

    Returns
    -------
    OUParams
        OU 过程参数估计结果

    Notes
    -----
    离散化：ΔX_t = κ·θ·Δt - κ·X_{t-1}·Δt + ε_t
    等价于：ΔX_t = a + b·X_{t-1} + ε_t
    其中 a = κ·θ·Δt，b = -κ·Δt
    → κ = -b/Δt，θ = a / (κ·Δt) = -a/b

    通过 OLS 对 ΔX_t 关于 X_{t-1} 回归估计 a 和 b。
    """
    x = spread.dropna().values
    if len(x) < 3:
        raise ValueError("价差序列长度不足（至少需要 3 个有效观测值）")

    delta_x = np.diff(x)
    x_lag = x[:-1]

    # OLS: ΔX = a + b * X_{t-1}
    X_mat = np.column_stack([np.ones(len(x_lag)), x_lag])
    coeffs, _, _, _ = np.linalg.lstsq(X_mat, delta_x, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])

    # 估计参数
    # b = -κ·Δt → κ = -b/Δt
    kappa = max(-b / dt, 1e-8)  # 强制非负

    # θ = -a/b（当 b ≠ 0 时）
    if abs(b) < 1e-10:
        theta_ou = float(x.mean())
    else:
        theta_ou = -a / b

    # 残差标准差 → σ_OU
    x_hat = a + b * x_lag
    residuals = delta_x - x_hat
    sigma_dt = float(np.std(residuals, ddof=2))
    sigma = sigma_dt / np.sqrt(dt) if dt > 0 else sigma_dt

    # R²
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((delta_x - delta_x.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    half_life = ou_half_life(kappa, dt)
    is_mr = (b < 0)  # 均值回归要求 b < 0（即 kappa > 0）

    return OUParams(
        kappa=kappa,
        theta=theta_ou,
        sigma=sigma,
        half_life=half_life,
        r_squared=r2,
        is_mean_reverting=is_mr,
    )


def ou_half_life(kappa: float, dt: float = 1 / 252) -> float:
    """
    计算 OU 过程半衰期（交易日数）。

    Parameters
    ----------
    kappa : float
        均值回归速度（年化）
    dt : float
        时间步长（年）

    Returns
    -------
    float
        半衰期（交易日数），kappa ≤ 0 时返回 inf
    """
    if kappa <= 0:
        return float("inf")
    return np.log(2) / kappa / dt


def simulate_ou(
    kappa: float,
    theta: float,
    sigma: float,
    x0: float,
    n_steps: int = 252,
    n_paths: int = 1000,
    dt: float = 1 / 252,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    蒙特卡洛模拟 OU 过程路径（用于回测和情景分析）。

    Parameters
    ----------
    kappa : float
        均值回归速度（年化）
    theta : float
        长期均值
    sigma : float
        扩散系数
    x0 : float
        初始价差值
    n_steps : int
        模拟步数（交易日数）
    n_paths : int
        模拟路径数
    dt : float
        时间步长（年）
    seed : int, optional
        随机种子

    Returns
    -------
    np.ndarray
        形状为 (n_paths, n_steps+1) 的路径数组
    """
    rng = np.random.default_rng(seed)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = x0
    sqrt_dt = np.sqrt(dt)

    for t in range(n_steps):
        dW = rng.standard_normal(n_paths) * sqrt_dt
        paths[:, t + 1] = (
            paths[:, t]
            + kappa * (theta - paths[:, t]) * dt
            + sigma * dW
        )

    return paths
