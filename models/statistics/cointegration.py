"""
cointegration.py
----------------
职责：协整关系检验和对冲比例估计。

支持：
- Engle-Granger 两步法（单方程协整检验）
- Johansen 迹检验（多变量系统）
- OLS 和 TLS（总最小二乘）对冲比例估计

应用场景：
- 价差交易配对选择（IF/IH、IC/IM）
- 统计套利信号生成

依赖：statsmodels
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CointegrationResult:
    """
    协整检验结果。

    Attributes
    ----------
    pair_id : str
        配对标识符，如 'IF-IH'
    method : str
        检验方法：'engle_granger' 或 'johansen'
    is_cointegrated : bool
        是否通过协整检验（p_value < significance）
    p_value : float
        检验 p 值
    test_statistic : float
        检验统计量
    critical_values : dict
        临界值字典（1%、5%、10% 置信水平）
    hedge_ratio : float
        OLS 估计的对冲比例（β in y = α + β×x + ε）
    spread_mean : float
        价差均值
    spread_std : float
        价差标准差
    adf_residual_pvalue : float
        残差 ADF 检验 p 值（越小越好，< 0.05 表示残差平稳）
    """
    pair_id: str
    method: str
    is_cointegrated: bool
    p_value: float
    test_statistic: float
    critical_values: dict = field(default_factory=dict)
    hedge_ratio: float = 1.0
    spread_mean: float = 0.0
    spread_std: float = 0.0
    adf_residual_pvalue: float = 1.0


def cointegration_test(
    series1: pd.Series,
    series2: pd.Series,
    method: str = "engle_granger",
    significance: float = 0.05,
) -> CointegrationResult:
    """
    对两个价格序列进行协整检验。

    Parameters
    ----------
    series1 : pd.Series
        第一个价格序列（被解释变量，如 IF 收盘价）
    series2 : pd.Series
        第二个价格序列（解释变量，如 IH 收盘价）
    method : str
        检验方法：'engle_granger'（默认）/ 'johansen'
    significance : float
        显著性水平，默认 0.05

    Returns
    -------
    CointegrationResult
        协整检验结果

    Notes
    -----
    - 使用 statsmodels.tsa.stattools.coint 实现 Engle-Granger 检验
    - 要求两个序列长度相同且索引对齐
    - 序列应为价格水平（非收益率）
    """
    from statsmodels.tsa.stattools import coint, adfuller
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    # 对齐并去除 NaN
    df = pd.concat([series1, series2], axis=1).dropna()
    s1 = df.iloc[:, 0]
    s2 = df.iloc[:, 1]
    pair_id = f"{series1.name or 'S1'}-{series2.name or 'S2'}"

    hedge_ratio, intercept = estimate_hedge_ratio(s1, s2, "ols")
    spread = s1 - hedge_ratio * s2 - intercept

    # ADF 检验价差平稳性
    adf_res = adfuller(spread.dropna(), autolag="AIC")
    adf_pvalue = float(adf_res[1])

    if method == "engle_granger":
        stat, p_value, crit = coint(s1, s2)
        crit_vals = {
            "1%": float(crit[0]),
            "5%": float(crit[1]),
            "10%": float(crit[2]),
        }
        return CointegrationResult(
            pair_id=pair_id,
            method="engle_granger",
            is_cointegrated=(float(p_value) < significance),
            p_value=float(p_value),
            test_statistic=float(stat),
            critical_values=crit_vals,
            hedge_ratio=hedge_ratio,
            spread_mean=float(spread.mean()),
            spread_std=float(spread.std()),
            adf_residual_pvalue=adf_pvalue,
        )

    elif method == "johansen":
        jres = coint_johansen(df, det_order=0, k_ar_diff=1)
        # lr1: trace statistics; cvt: critical values (rows=ranks, cols=[90%,95%,99%])
        trace_stat = float(jres.lr1[0])
        crit_90, crit_95, crit_99 = float(jres.cvt[0, 0]), float(jres.cvt[0, 1]), float(jres.cvt[0, 2])

        # Approximate p-value based on critical values
        if trace_stat > crit_99:
            p_approx = 0.01
        elif trace_stat > crit_95:
            p_approx = 0.05
        elif trace_stat > crit_90:
            p_approx = 0.10
        else:
            p_approx = 0.20

        return CointegrationResult(
            pair_id=pair_id,
            method="johansen",
            is_cointegrated=(trace_stat > crit_95),
            p_value=p_approx,
            test_statistic=trace_stat,
            critical_values={"90%": crit_90, "95%": crit_95, "99%": crit_99},
            hedge_ratio=hedge_ratio,
            spread_mean=float(spread.mean()),
            spread_std=float(spread.std()),
            adf_residual_pvalue=adf_pvalue,
        )
    else:
        raise ValueError(f"未知检验方法: {method}，支持 'engle_granger' 和 'johansen'")


def estimate_hedge_ratio(
    series1: pd.Series,
    series2: pd.Series,
    method: str = "ols",
) -> tuple[float, float]:
    """
    估计最优对冲比例。

    Parameters
    ----------
    series1 : pd.Series
        被解释变量（持有做多端）
    series2 : pd.Series
        解释变量（持有做空端）
    method : str
        估计方法：'ols'（普通最小二乘）/ 'tls'（总最小二乘）

    Returns
    -------
    tuple[float, float]
        (hedge_ratio, intercept)
        即 series1 = intercept + hedge_ratio × series2 + ε
    """
    df = pd.concat([series1, series2], axis=1).dropna()
    y = df.iloc[:, 0].values
    x = df.iloc[:, 1].values

    if method == "ols":
        X = np.column_stack([np.ones(len(x)), x])
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        intercept, beta = float(coeffs[0]), float(coeffs[1])
        return beta, intercept

    elif method == "tls":
        # 总最小二乘（SVD 方法）
        x_c = x - x.mean()
        y_c = y - y.mean()
        data = np.column_stack([x_c, y_c])
        _, _, Vt = np.linalg.svd(data, full_matrices=False)
        # 最小奇异值对应的右奇异向量给出零空间方向
        v = Vt[-1]
        if abs(v[1]) < 1e-10:
            return estimate_hedge_ratio(series1, series2, "ols")
        beta = -v[0] / v[1]
        intercept = float(y.mean() - beta * x.mean())
        return float(beta), intercept

    else:
        raise ValueError(f"未知方法: {method}，支持 'ols' 和 'tls'")


def rolling_cointegration(
    series1: pd.Series,
    series2: pd.Series,
    window: int = 60,
    step: int = 5,
) -> pd.DataFrame:
    """
    滚动协整检验（用于监测协整关系稳定性）。

    Parameters
    ----------
    window : int
        滚动窗口长度（交易日）
    step : int
        滚动步长（交易日）

    Returns
    -------
    pd.DataFrame
        列：date, p_value, hedge_ratio, spread_mean, spread_std
    """
    n = len(series1)
    rows = []
    for start in range(0, n - window + 1, step):
        end = start + window
        s1 = series1.iloc[start:end]
        s2 = series2.iloc[start:end]
        try:
            result = cointegration_test(s1, s2)
            rows.append({
                "date": series1.index[end - 1],
                "p_value": result.p_value,
                "is_cointegrated": result.is_cointegrated,
                "hedge_ratio": result.hedge_ratio,
                "spread_mean": result.spread_mean,
                "spread_std": result.spread_std,
            })
        except Exception as e:
            logger.warning("滚动协整检验窗口 [%d, %d] 失败: %s", start, end, e)

    if not rows:
        return pd.DataFrame(columns=["date", "p_value", "is_cointegrated",
                                     "hedge_ratio", "spread_mean", "spread_std"])
    return pd.DataFrame(rows)
