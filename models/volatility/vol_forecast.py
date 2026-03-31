"""
vol_forecast.py
---------------
综合波动率预测器，整合 RV 和 GARCH 预测，
提供面向策略层的统一接口。

支持：
- 纯 GARCH 预测
- HAR-RV（Heterogeneous Autoregressive RV）预测
- GARCH + RV 混合预测（GARCH-X）
- 集成预测（GARCH + HAR-RV 等权平均）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from .garch_model import GJRGARCHModel
from .realized_vol import compute_realized_vol, RVEstimator

logger = logging.getLogger(__name__)

# HAR-RV 窗口参数
HAR_DAILY_LAG = 1     # 日度分量：前 1 天
HAR_WEEKLY_LAG = 5    # 周度分量：前 5 天均值
HAR_MONTHLY_LAG = 22  # 月度分量：前 22 天均值

# 最小样本量
MIN_GARCH_SAMPLES = 60
MIN_HAR_SAMPLES = 30


class ForecastMethod(str, Enum):
    """波动率预测方法"""
    GARCH = "garch"          # 纯 GJR-GARCH 预测
    HAR_RV = "har_rv"        # HAR-RV 预测
    GARCH_X = "garch_x"     # GARCH-X（引入 RV 作为外生变量，简化为混合）
    ENSEMBLE = "ensemble"    # 加权集成（GARCH + HAR-RV 等权均值）


@dataclass
class VolForecastResult:
    """
    波动率预测结果。

    Attributes
    ----------
    trade_date : str
        预测基准日期
    underlying : str
        预测标的
    forecast_vol : float
        预测波动率（年化小数）
    horizon : int
        预测期限（交易日）
    method : ForecastMethod
        使用的预测方法
    conf_interval_lower : float
        85% 置信区间下限
    conf_interval_upper : float
        85% 置信区间上限
    garch_vol : float
        GARCH 预测分量（仅集成方法下有意义）
    har_rv_vol : float
        HAR-RV 预测分量（仅集成方法下有意义）
    """
    trade_date: str
    underlying: str
    forecast_vol: float
    horizon: int = 5
    method: ForecastMethod = ForecastMethod.GARCH
    conf_interval_lower: float = 0.0
    conf_interval_upper: float = 0.0
    garch_vol: float = 0.0
    har_rv_vol: float = 0.0


class VolForecaster:
    """
    综合波动率预测器。

    Parameters
    ----------
    method : ForecastMethod
        预测方法，默认 ENSEMBLE
    garch_model : GJRGARCHModel, optional
        已拟合的 GARCH 模型（method=GARCH/GARCH_X/ENSEMBLE 时可提供；
        未提供时自动在 forecast() 内重新拟合）

    Examples
    --------
    >>> forecaster = VolForecaster(method=ForecastMethod.ENSEMBLE)
    >>> result = forecaster.forecast("20240101", "IC.CFX", returns)
    >>> result.forecast_vol  # 年化波动率，小数形式
    0.2134
    """

    def __init__(
        self,
        method: ForecastMethod = ForecastMethod.ENSEMBLE,
        garch_model: Optional[GJRGARCHModel] = None,
    ) -> None:
        self.method = method
        self.garch_model = garch_model

        # HAR-RV 回归系数（fit 后缓存）
        self._har_betas: Optional[np.ndarray] = None
        self._har_residual_std: float = 0.0

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def forecast(
        self,
        trade_date: str,
        underlying: str,
        returns: pd.Series,
        min_df: Optional[pd.DataFrame] = None,
        horizon: int = 5,
    ) -> VolForecastResult:
        """
        执行波动率预测。

        Parameters
        ----------
        trade_date : str
            预测基准日期，格式 YYYYMMDD
        underlying : str
            标的代码
        returns : pd.Series
            日度对数收益率序列（小数形式，如 0.01 表示 1%）
        min_df : pd.DataFrame, optional
            分钟线数据（datetime, close 列）；若提供则用于计算更精确的日内 RV
        horizon : int
            预测期限（交易日），默认 5（一周）

        Returns
        -------
        VolForecastResult
        """
        returns_clean = returns.dropna()
        if len(returns_clean) < MIN_GARCH_SAMPLES:
            raise ValueError(
                f"返回率序列长度 {len(returns_clean)} 不足，"
                f"至少需要 {MIN_GARCH_SAMPLES} 个有效观测值"
            )

        # 计算日度已实现波动率序列（用于 HAR-RV）
        daily_rv = self._compute_daily_rv(returns_clean, min_df)

        garch_vol = 0.0
        har_vol = 0.0

        if self.method in (ForecastMethod.GARCH, ForecastMethod.GARCH_X,
                           ForecastMethod.ENSEMBLE):
            garch_vol = self._garch_forecast(returns_clean, horizon)

        if self.method in (ForecastMethod.HAR_RV, ForecastMethod.ENSEMBLE):
            har_vol = self._har_rv_forecast(daily_rv, horizon)

        if self.method == ForecastMethod.GARCH:
            forecast_vol = garch_vol

        elif self.method == ForecastMethod.HAR_RV:
            forecast_vol = har_vol

        elif self.method == ForecastMethod.ENSEMBLE:
            forecast_vol = 0.5 * garch_vol + 0.5 * har_vol

        elif self.method == ForecastMethod.GARCH_X:
            # 简化 GARCH-X：以 GARCH 为主，最新 RV 提供实时修正
            last_rv = float(daily_rv.dropna().iloc[-1]) if len(daily_rv.dropna()) > 0 else garch_vol
            forecast_vol = 0.7 * garch_vol + 0.3 * last_rv

        else:
            raise ValueError(f"未知预测方法: {self.method}")

        forecast_vol = max(forecast_vol, 0.01)

        # 置信区间（基于历史 RV 波动或 HAR-RV 残差标准差）
        ci_lower, ci_upper = self._compute_confidence_interval(
            forecast_vol, daily_rv
        )

        return VolForecastResult(
            trade_date=trade_date,
            underlying=underlying,
            forecast_vol=forecast_vol,
            horizon=horizon,
            method=self.method,
            conf_interval_lower=ci_lower,
            conf_interval_upper=ci_upper,
            garch_vol=garch_vol,
            har_rv_vol=har_vol,
        )

    def fit_har(self, rv_series: pd.Series) -> None:
        """
        单独拟合 HAR-RV 回归并缓存系数，避免每次 forecast() 重复拟合。

        Parameters
        ----------
        rv_series : pd.Series
            日度已实现波动率序列（年化，小数）
        """
        betas, residual_std = self._fit_har_ols(rv_series)
        self._har_betas = betas
        self._har_residual_std = residual_std

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _compute_daily_rv(
        self,
        returns: pd.Series,
        min_df: Optional[pd.DataFrame],
    ) -> pd.Series:
        """
        计算日度已实现波动率。

        优先使用分钟线（更精确）；无分钟线时用绝对收益率作为日度 RV 代理。
        RV_proxy[t] = |r_t| * sqrt(252)  （年化绝对收益）
        """
        if min_df is not None and not min_df.empty:
            try:
                return compute_realized_vol(
                    min_df,
                    estimator=RVEstimator.SIMPLE,
                    freq_minutes=5,
                    annualize=True,
                )
            except Exception as e:
                logger.warning("分钟线 RV 计算失败，回退到绝对收益代理: %s", e)

        # 绝对收益率代理（年化）
        rv_proxy = returns.abs() * np.sqrt(252)
        rv_proxy.name = "rv"
        return rv_proxy

    def _garch_forecast(self, returns: pd.Series, horizon: int) -> float:
        """
        使用 GJR-GARCH 预测期内平均波动率。

        若 garch_model 未拟合，自动重新拟合后预测。
        """
        model = self.garch_model
        if model is None:
            model = GJRGARCHModel()

        if not model.is_fitted:
            model.fit(returns)

        return float(model.forecast_period_avg(horizon=horizon))

    def _har_rv_forecast(
        self,
        rv_series: pd.Series,
        horizon: int = 5,
    ) -> float:
        """
        HAR-RV 模型预测。

        Corsi (2009) 模型：
            RV_{t+1} = β₀ + β_d·RV_d_t + β_w·RV_w_t + β_m·RV_m_t + ε

        其中：
            RV_d_t = RV_t                       （日度）
            RV_w_t = mean(RV_{t-4}, …, RV_t)   （5天均值，周度）
            RV_m_t = mean(RV_{t-21}, …, RV_t)  （22天均值，月度）

        Parameters
        ----------
        rv_series : pd.Series
            日度已实现波动率序列（年化，小数）
        horizon : int
            预测期限（1-day ahead 预测用作 horizon 期均值近似）

        Returns
        -------
        float
            预测波动率（年化，小数）
        """
        rv = rv_series.dropna()
        if len(rv) < MIN_HAR_SAMPLES:
            # 样本不足时，回退到移动平均
            logger.warning("HAR-RV 样本不足 (%d < %d)，回退到 22 日移动平均", len(rv), MIN_HAR_SAMPLES)
            return float(rv.iloc[-min(22, len(rv)):].mean())

        # 使用缓存系数（若已通过 fit_har 预先拟合）
        if self._har_betas is not None:
            betas = self._har_betas
        else:
            betas, _ = self._fit_har_ols(rv)

        # HAR-RV 通过迭代预测多步平均：以 1-step 预测为基础，
        # 利用模型持续性对 horizon 步做简单滚动平均近似
        rv_arr = rv.values.copy()
        preds = []
        for _ in range(horizon):
            p = self._har_predict(rv_arr, betas)
            preds.append(p)
            rv_arr = np.append(rv_arr, p)
        return float(np.mean(preds))

    def _fit_har_ols(self, rv_series: pd.Series) -> tuple[np.ndarray, float]:
        """
        拟合 HAR-RV OLS 回归，返回 (betas, residual_std)。

        HAR 特征矩阵：X = [1, RV_d, RV_w, RV_m]
        被解释变量：y = RV_{t+1}（次日 RV）
        """
        rv = rv_series.dropna().values
        n = len(rv)

        rv_d = rv
        rv_w = pd.Series(rv).rolling(HAR_WEEKLY_LAG).mean().values
        rv_m = pd.Series(rv).rolling(HAR_MONTHLY_LAG).mean().values

        # y[i] = RV_{t+1}，X[i] 为 t 时刻特征
        y = rv_d[1:]
        X = np.column_stack([np.ones(n - 1), rv_d[:-1], rv_w[:-1], rv_m[:-1]])

        # 过滤 NaN（rolling 产生的前缘 NaN）
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_c, y_c = X[mask], y[mask]

        if len(y_c) < 10:
            # 极端情况：直接返回简单均值预测
            return np.array([float(rv.mean()), 0.0, 0.0, 0.0]), float(rv.std())

        betas, residuals, _, _ = np.linalg.lstsq(X_c, y_c, rcond=None)

        # 残差标准差（用于置信区间）
        y_hat = X_c @ betas
        res = y_c - y_hat
        residual_std = float(np.std(res, ddof=len(betas)))

        return betas, residual_std

    def _har_predict(self, rv: np.ndarray, betas: np.ndarray) -> float:
        """用已拟合系数预测下一期 RV。"""
        last_rv_d = rv[-1]
        last_rv_w = float(np.mean(rv[-HAR_WEEKLY_LAG:])) if len(rv) >= HAR_WEEKLY_LAG else last_rv_d
        last_rv_m = float(np.mean(rv[-HAR_MONTHLY_LAG:])) if len(rv) >= HAR_MONTHLY_LAG else float(np.mean(rv))

        x_pred = np.array([1.0, last_rv_d, last_rv_w, last_rv_m])
        pred = float(np.dot(betas, x_pred))
        return max(pred, 0.01)

    def _compute_confidence_interval(
        self,
        forecast_vol: float,
        daily_rv: pd.Series,
    ) -> tuple[float, float]:
        """
        计算 85% 置信区间。

        使用历史 RV 波动率构造对称的对数正态近似：
        CI = [exp(log(μ) ± 1.44 * σ_log)]
        其中 σ_log = std(log(rv)) 为波动率的对数标准差。
        """
        rv_clean = daily_rv.dropna()
        if len(rv_clean) < 5:
            # 无法估计，使用 ±30% 经验值
            return max(forecast_vol * 0.7, 0.01), forecast_vol * 1.3

        log_rv = np.log(rv_clean.clip(lower=1e-6))
        sigma_log = float(log_rv.std())
        z_85 = 1.44  # Normal z for 85% CI (one-sided 92.5%)

        log_forecast = np.log(max(forecast_vol, 1e-6))
        ci_lower = float(np.exp(log_forecast - z_85 * sigma_log))
        ci_upper = float(np.exp(log_forecast + z_85 * sigma_log))

        return max(ci_lower, 0.01), ci_upper


# ======================================================================
# VolForecast：面向策略层的轻量波动率预测接口
# ======================================================================

class VolForecast:
    """
    波动率预测统一接口。

    封装 GJR-GARCH 和 EWMA 两种方法，提供统一的调用方式。
    比 VolForecaster 更轻量：直接接受收盘价序列，内部计算对数收益率。

    Parameters
    ----------
    method : str
        预测方法：
        - "garch"  — GJR-GARCH（默认，精度高，计算较慢）
        - "ewma"   — 指数加权移动平均（RiskMetrics，快速基准）
        - "har"    — HAR-RV（委托给 VolForecaster，需要足够历史数据）
    **kwargs
        传递给底层模型的额外参数（如 ewma 的 decay）。
    """

    def __init__(self, method: str = "garch", **kwargs) -> None:
        valid = {"garch", "ewma", "har"}
        if method not in valid:
            raise ValueError(f"method 须为 {valid}，收到: {method!r}")
        self.method = method
        self._kwargs = kwargs

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def fit_and_predict(
        self,
        close_prices: pd.Series,
        horizon: int = 5,
    ) -> dict:
        """
        用历史收盘价序列拟合模型并预测未来波动率。

        Parameters
        ----------
        close_prices : pd.Series
            收盘价序列（至少 62 个有效值）
        horizon : int
            预测期限（交易日），默认 5

        Returns
        -------
        dict
            method       : str    — 使用的方法名
            current_vol  : float  — 当前波动率（年化，基于最近 20 日 HV）
            forecast_vol : float  — 预测的未来平均波动率（年化）
            model_params : dict   — 模型参数（GARCH: ω/α/β/γ；EWMA: decay/last_var）
            diagnostics  : dict   — 诊断信息（AIC/BIC 等，EWMA 时为空 dict）
        """
        close_clean = close_prices.dropna()
        if len(close_clean) < 2:
            raise ValueError("close_prices 至少需要 2 个有效值")

        returns = np.log(close_clean / close_clean.shift(1)).dropna()

        # 当前波动率：20 日历史波动率
        current_vol = float(
            returns.iloc[-20:].std() * np.sqrt(252)
        ) if len(returns) >= 20 else float(returns.std() * np.sqrt(252))

        if self.method == "ewma":
            decay = self._kwargs.get("decay", 0.94)
            forecast_vol = self.ewma_forecast(returns, decay)
            model_params = {"decay": decay, "last_var": float(returns.iloc[-1] ** 2)}
            diagnostics: dict = {}

        elif self.method == "garch":
            model = GJRGARCHModel()
            model.fit(returns)
            forecast_vol = float(model.forecast_period_avg(horizon=horizon))
            result = model.fit_result
            if result is not None:
                params = result.params  # dict[str, float]
                model_params = {
                    "omega": float(params.get("omega", 0.0)),
                    "alpha": float(params.get("alpha", 0.0)),
                    "beta":  float(params.get("beta",  0.0)),
                    "gamma": float(params.get("gamma", 0.0)),
                }
                diagnostics = {
                    "aic": float(result.aic),
                    "bic": float(result.bic),
                }
            else:
                model_params = {}
                diagnostics = {}

        else:  # "har"
            forecaster = VolForecaster(method=ForecastMethod.HAR_RV)
            result_obj = forecaster.forecast(
                trade_date="", underlying="", returns=returns, horizon=horizon
            )
            forecast_vol = result_obj.forecast_vol
            model_params = {}
            diagnostics = {
                "ci_lower": result_obj.conf_interval_lower,
                "ci_upper": result_obj.conf_interval_upper,
            }

        return {
            "method": self.method,
            "current_vol": current_vol,
            "forecast_vol": max(float(forecast_vol), 0.01),
            "model_params": model_params,
            "diagnostics": diagnostics,
        }

    def ewma_forecast(self, returns: pd.Series, decay: float = 0.94) -> float:
        """
        EWMA 波动率预测（RiskMetrics λ 方法）。

        σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}

        初始化：σ²_0 = 方差估计（历史均值）。
        返回当前时刻的年化波动率。

        Parameters
        ----------
        returns : pd.Series
            日度对数收益率
        decay : float
            衰减因子 λ，默认 0.94（RiskMetrics 推荐值）

        Returns
        -------
        float
            当前 EWMA 年化波动率
        """
        r = returns.dropna().values
        if len(r) == 0:
            return 0.20  # 缺省 20%

        # 初始化：使用前 20 日历史方差（或全部数据）
        init_window = min(20, len(r))
        var = float(np.var(r[:init_window], ddof=1))
        if var <= 0:
            var = (r[0] ** 2) or 1e-6

        for ret in r[init_window:]:
            var = decay * var + (1.0 - decay) * (ret ** 2)

        return float(np.sqrt(var * 252))
