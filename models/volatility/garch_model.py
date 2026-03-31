"""
garch_model.py
--------------
GJR-GARCH(1,1) 波动率模型封装。

条件方差方程：
    σ²_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·I(ε_{t-1}<0) + β·σ²_{t-1}

其中：
    ω > 0        常数项
    α ≥ 0        ARCH 项（昨日冲击）
    γ ≥ 0        GJR 杠杆项（下跌对波动率的额外影响，股指期货通常 γ>0）
    β ≥ 0        GARCH 项（波动率持续性）
    α+γ/2+β < 1  弱平稳条件

提供两套接口：
  GARCHModel      — 新接口（dict 返回，预测/诊断一体化）
  GJRGARCHModel   — 向后兼容旧接口（GARCHFitResult 返回）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


# ======================================================================
# GARCHFitResult — 向后兼容数据类
# ======================================================================

@dataclass
class GARCHFitResult:
    """GJR-GARCH 模型拟合结果（向后兼容接口）"""
    params: dict[str, float]    # 参数: omega, alpha, gamma, beta（均在模型内部尺度）
    aic: float
    bic: float
    log_likelihood: float
    persistence: float          # α + γ/2 + β（应 < 1）
    conditional_vol: pd.Series  # 历史条件波动率序列（年化，小数）
    std_resid: pd.Series        # 标准化残差序列
    converged: bool


# ======================================================================
# 内部工具
# ======================================================================

def _detect_scale(returns: pd.Series) -> float:
    """
    自动检测收益率序列的尺度。
    abs.mean() < 0.5 → 视为小数形式（如 0.01 = 1%），返回 100.0
    否则视为百分比形式（如 1.0 = 1%），返回 1.0
    """
    return 100.0 if returns.abs().mean() < 0.5 else 1.0


def _extract_arch_params(res_params: pd.Series) -> dict[str, float]:
    """从 arch 拟合结果的 params Series 中提取命名参数"""
    d: dict[str, float] = {}

    d["omega"] = float(res_params.get("omega", np.nan))

    alpha_keys = [k for k in res_params.index if k.startswith("alpha")]
    d["alpha"] = float(res_params[alpha_keys[0]]) if alpha_keys else 0.0

    gamma_keys = [k for k in res_params.index if k.startswith("gamma")]
    d["gamma"] = float(res_params[gamma_keys[0]]) if gamma_keys else 0.0

    beta_keys = [k for k in res_params.index if k.startswith("beta")]
    d["beta"] = float(res_params[beta_keys[0]]) if beta_keys else 0.0

    # t 分布自由度
    for nu_key in ("nu", "eta"):
        if nu_key in res_params.index:
            d["nu"] = float(res_params[nu_key])
            break

    return d


# ======================================================================
# GARCHModel — 新接口（dict-based）
# ======================================================================

class GARCHModel:
    """
    GJR-GARCH(p,o,q) 模型。

    面向策略层的简洁接口：fit → dict，predict → dict，diagnose → dict。
    内部使用 arch 库，自动处理收益率尺度转换（小数/百分比）。

    Parameters
    ----------
    p : int     GARCH 阶数，默认 1
    q : int     ARCH 阶数，默认 1
    o : int     杠杆效应阶数（0=标准 GARCH，1=GJR-GARCH），默认 1
    dist : str  残差分布：'normal' / 't' / 'ged' / 'skewt'，默认 't'
    rescale : bool  arch 库自动缩放，推荐 True

    Notes
    -----
    - 输入收益率为小数（0.01 = 1%）或百分比（1.0 = 1%）均可，自动检测
    - 输出波动率统一为年化小数（0.20 = 20%）
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        o: int = 1,
        dist: str = "t",
        rescale: bool = True,
    ) -> None:
        self.p = p
        self.q = q
        self.o = o
        self.dist = dist
        self.rescale = rescale

        self._arch_result = None   # arch ARCHModelResult
        self._scale: float = 100.0
        self._clean_returns: Optional[pd.Series] = None
        self._fit_params: Optional[dict] = None  # dict from fit()

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        returns: pd.Series,
        show_summary: bool = False,
    ) -> dict:
        """
        拟合 GJR-GARCH 模型。

        Parameters
        ----------
        returns : pd.Series
            日对数收益率（小数或百分比，自动检测）
        show_summary : bool
            是否打印 arch 拟合摘要

        Returns
        -------
        dict
            omega, alpha, gamma, beta, persistence, long_run_var,
            long_run_vol（年化），nu（仅 dist='t'），
            log_likelihood, aic, bic
        """
        from arch import arch_model as _arch_model_fn

        clean = returns.dropna()
        if len(clean) < 50:
            raise ValueError(f"收益率序列太短：{len(clean)} < 50")

        self._scale = _detect_scale(clean)
        self._clean_returns = clean
        scaled = clean * self._scale

        am = _arch_model_fn(
            scaled,
            vol="Garch",
            p=self.p,
            o=self.o,
            q=self.q,
            dist=self.dist,
            mean="Constant",
            rescale=self.rescale,
        )
        res = am.fit(disp="off", update_freq=0)
        self._arch_result = res

        if show_summary:
            print(res.summary())

        raw_params = _extract_arch_params(res.params)
        alpha      = raw_params["alpha"]
        gamma      = raw_params["gamma"]
        beta       = raw_params["beta"]
        omega_scaled = raw_params["omega"]          # ω in (scale_unit)²
        persistence  = alpha + gamma / 2.0 + beta

        # Convert omega back to daily decimal² for long-run calculations
        omega_decimal = omega_scaled / (self._scale ** 2)

        if 0.0 < persistence < 1.0:
            long_run_var = omega_decimal / (1.0 - persistence)
            long_run_vol = float(np.sqrt(long_run_var) * np.sqrt(TRADING_DAYS_PER_YEAR))
        else:
            long_run_var = float("inf")
            long_run_vol = float("inf")

        if persistence >= 1.0:
            logger.warning("GARCHModel: persistence=%.4f ≥ 1，模型非平稳", persistence)

        result = {
            "omega":          omega_decimal,
            "alpha":          alpha,
            "gamma":          gamma,
            "beta":           beta,
            "persistence":    persistence,
            "long_run_var":   long_run_var,
            "long_run_vol":   long_run_vol,
            "log_likelihood": float(res.loglikelihood),
            "aic":            float(res.aic),
            "bic":            float(res.bic),
        }
        if "nu" in raw_params:
            result["nu"] = raw_params["nu"]

        self._fit_params = result
        return result

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(
        self,
        horizon: int = 5,
        returns: Optional[pd.Series] = None,
    ) -> dict:
        """
        预测未来波动率。

        多步递推公式（analytic 法）：
            σ²_{t+h} = ω + (α+γ/2+β) * σ²_{t+h-1}
        最终收敛到长期方差 ω/(1-α-γ/2-β)。

        Parameters
        ----------
        horizon : int
            预测天数
        returns : pd.Series, optional
            如果提供则先用新数据重新拟合，否则使用上次 fit 结果

        Returns
        -------
        dict
            daily_vol  : list[float]   未来每天年化条件波动率
            mean_vol   : float         horizon 天平均年化波动率
            current_vol: float         当前最新一天年化条件波动率
        """
        self._require_fitted()

        if returns is not None:
            self.fit(returns)

        fc = self._arch_result.forecast(horizon=horizon, method="analytic")
        # variance in (scale_unit)², shape (nobs, horizon), take last row
        var_forecast = fc.variance.iloc[-1].values
        daily_vols = (
            np.sqrt(var_forecast) / self._scale * np.sqrt(TRADING_DAYS_PER_YEAR)
        ).tolist()

        current_vol = float(
            np.sqrt(self._arch_result.conditional_volatility.iloc[-1])
            / self._scale
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        )

        return {
            "daily_vol":   [float(v) for v in daily_vols],
            "mean_vol":    float(np.mean(daily_vols)),
            "current_vol": current_vol,
        }

    # ------------------------------------------------------------------
    # get_conditional_vol
    # ------------------------------------------------------------------

    def get_conditional_vol(
        self,
        returns: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        获取历史条件波动率序列（年化，小数）。

        Returns
        -------
        pd.Series
            与输入 returns 等长，index 与 returns 一致
        """
        self._require_fitted()

        if returns is not None:
            self.fit(returns)

        # conditional_volatility from arch is in scale_unit (daily std)
        cond_vol_annual = (
            self._arch_result.conditional_volatility.values
            / self._scale
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
        return pd.Series(cond_vol_annual, index=self._clean_returns.index)

    # ------------------------------------------------------------------
    # get_standardized_residuals
    # ------------------------------------------------------------------

    def get_standardized_residuals(self) -> pd.Series:
        """
        获取标准化残差序列 ε_t / σ_t。

        用于模型诊断：正确的模型残差应接近 iid，无自相关，
        残差平方也不应有 ARCH 效应。

        Returns
        -------
        pd.Series
            与输入 returns 等长
        """
        self._require_fitted()
        return pd.Series(
            self._arch_result.std_resid.values,
            index=self._clean_returns.index,
        )

    # ------------------------------------------------------------------
    # diagnose
    # ------------------------------------------------------------------

    def diagnose(self) -> dict:
        """
        模型诊断统计量。

        Returns
        -------
        dict
            ljung_box_p  : Ljung-Box 检验 p 值（标准化残差，lag=10）
                           > 0.05 → 无显著自相关，模型 OK
            arch_test_p  : ARCH-LM 检验 p 值（标准化残差，lag=5）
                           > 0.05 → 无残余 ARCH 效应，模型 OK
            jarque_bera_p: Jarque-Bera 正态性检验 p 值
            skewness     : 标准化残差偏度
            kurtosis     : 标准化残差超额峰度（Fisher 定义，正态=0）
            is_stationary: α+γ/2+β < 1
            half_life    : 波动率冲击半衰期（天），ln(0.5)/ln(persistence)
        """
        self._require_fitted()

        from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
        from scipy import stats as scipy_stats

        std_resid = self._arch_result.std_resid.dropna()

        # Ljung-Box on standardized residuals (lag 10)
        lb = acorr_ljungbox(std_resid, lags=10, return_df=True)
        lb_p = float(lb["lb_pvalue"].iloc[-1])

        # ARCH-LM test on standardized residuals (lag 5)
        lm_stat, lm_p, _, _ = het_arch(std_resid, nlags=5)

        # Jarque-Bera normality test
        _, jb_p = scipy_stats.jarque_bera(std_resid)

        persistence = self._fit_params["persistence"]

        if 0.0 < persistence < 1.0:
            half_life = float(np.log(0.5) / np.log(persistence))
        else:
            half_life = float("inf")

        return {
            "ljung_box_p":   lb_p,
            "arch_test_p":   float(lm_p),
            "jarque_bera_p": float(jb_p),
            "skewness":      float(scipy_stats.skew(std_resid)),
            "kurtosis":      float(scipy_stats.kurtosis(std_resid, fisher=True)),
            "is_stationary": bool(persistence < 1.0),
            "half_life":     half_life,
        }

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if self._arch_result is None:
            raise RuntimeError("模型尚未拟合，请先调用 fit()")

    @property
    def is_fitted(self) -> bool:
        return self._arch_result is not None


# ======================================================================
# GJRGARCHModel — 向后兼容旧接口（基于 GARCHModel）
# ======================================================================

class GJRGARCHModel:
    """
    GJR-GARCH(1,1) 模型（向后兼容接口）。

    内部委托给 GARCHModel，对外暴露旧接口：
    fit() → GARCHFitResult，forecast/forecast_next_day/forecast_period_avg，
    ljung_box_test / arch_lm_test。

    Parameters
    ----------
    dist : str
        残差分布假设：'normal' / 'skewt' / 't'，默认 'skewt'
    lookback : int
        最少需要的历史数据长度（交易日），默认 252（仅作为文档用途）
    """

    def __init__(self, dist: str = "skewt", lookback: int = 252) -> None:
        self.dist = dist
        self.lookback = lookback
        self._core = GARCHModel(p=1, q=1, o=1, dist=dist)
        self._fit_result: Optional[GARCHFitResult] = None

    # ------------------------------------------------------------------
    # 拟合
    # ------------------------------------------------------------------

    def fit(
        self,
        returns: pd.Series,
        update_freq: int = 0,
    ) -> GARCHFitResult:
        """
        拟合 GJR-GARCH(1,1) 模型。

        Parameters
        ----------
        returns : pd.Series
            日度对数收益率序列（百分比或小数均可）
        update_freq : int
            优化进度打印频率（0=不打印）

        Returns
        -------
        GARCHFitResult
        """
        fit_dict = self._core.fit(returns, show_summary=(update_freq > 0))
        res = self._core._arch_result
        clean = self._core._clean_returns
        scale = self._core._scale

        # 参数 dict（omega 保留 arch 内部尺度，方便与 arch 直接对比）
        raw = _extract_arch_params(res.params)

        cond_vol_annual = pd.Series(
            (res.conditional_volatility.values / scale * np.sqrt(TRADING_DAYS_PER_YEAR)),
            index=clean.index,
        )
        std_resid = pd.Series(res.std_resid.values, index=clean.index)

        persistence = fit_dict["persistence"]
        if persistence >= 1.0:
            logger.warning(
                "GJRGARCHModel: persistence=%.4f ≥ 1，模型非平稳", persistence
            )

        result = GARCHFitResult(
            params=raw,
            aic=fit_dict["aic"],
            bic=fit_dict["bic"],
            log_likelihood=fit_dict["log_likelihood"],
            persistence=persistence,
            conditional_vol=cond_vol_annual,
            std_resid=std_resid,
            converged=(res.convergence_flag == 0),
        )
        self._fit_result = result
        return result

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------

    def forecast(
        self,
        horizon: int = 5,
        method: str = "analytic",
        n_simulations: int = 1000,
    ) -> pd.DataFrame:
        """
        预测未来多步条件波动率。

        Parameters
        ----------
        horizon : int    预测步数（交易日）
        method : str     'analytic' / 'simulation' / 'bootstrap'

        Returns
        -------
        pd.DataFrame
            单行 DataFrame，列为 h.1, h.2, ..., h.{horizon}，
            值为年化条件波动率（小数）
        """
        self._require_fitted_legacy()
        res = self._core._arch_result
        scale = self._core._scale

        fc = res.forecast(horizon=horizon, method=method,
                          simulations=n_simulations if method == "simulation" else None)
        var_forecast = fc.variance.iloc[-1].values  # (horizon,) in scale²
        vol_annual = np.sqrt(var_forecast) / scale * np.sqrt(TRADING_DAYS_PER_YEAR)

        last_date = self._fit_result.conditional_vol.index[-1]
        cols = [f"h.{i+1}" for i in range(horizon)]
        return pd.DataFrame([vol_annual], index=[last_date], columns=cols)

    def forecast_next_day(self) -> float:
        """
        快捷方法：预测下一交易日年化条件波动率（小数）。
        """
        self._require_fitted_legacy()
        res = self._core._arch_result
        scale = self._core._scale

        fc = res.forecast(horizon=1, method="analytic")
        var_h1 = float(fc.variance.iloc[-1, 0])
        return float(np.sqrt(var_h1) / scale * np.sqrt(TRADING_DAYS_PER_YEAR))

    def forecast_period_avg(self, horizon: int = 5) -> float:
        """
        预测未来 horizon 个交易日的平均年化波动率（小数）。
        用于与期权隐含波动率（代表持仓期平均波动率）对比。
        """
        self._require_fitted_legacy()
        res = self._core._arch_result
        scale = self._core._scale

        fc = res.forecast(horizon=horizon, method="analytic")
        var_all = fc.variance.iloc[-1].values
        daily_vols = np.sqrt(var_all) / scale * np.sqrt(TRADING_DAYS_PER_YEAR)
        return float(np.mean(daily_vols))

    # ------------------------------------------------------------------
    # 诊断
    # ------------------------------------------------------------------

    def ljung_box_test(self, lags: int = 10) -> dict:
        """
        对标准化残差进行 Ljung-Box 自相关检验。

        Returns
        -------
        dict  statistic, p_value；p_value > 0.05 → 无显著自相关
        """
        self._require_fitted_legacy()
        from statsmodels.stats.diagnostic import acorr_ljungbox

        std_resid = self._fit_result.std_resid.dropna()
        lb = acorr_ljungbox(std_resid, lags=lags, return_df=True)
        return {
            "statistic": float(lb["lb_stat"].iloc[-1]),
            "p_value":   float(lb["lb_pvalue"].iloc[-1]),
        }

    def arch_lm_test(self, lags: int = 5) -> dict:
        """
        ARCH-LM 检验：判断是否还有剩余 ARCH 效应。

        Returns
        -------
        dict  statistic, p_value；p_value > 0.05 → 无残余 ARCH 效应
        """
        self._require_fitted_legacy()
        from statsmodels.stats.diagnostic import het_arch

        std_resid = self._fit_result.std_resid.dropna()
        lm_stat, lm_p, _, _ = het_arch(std_resid, nlags=lags)
        return {
            "statistic": float(lm_stat),
            "p_value":   float(lm_p),
        }

    # ------------------------------------------------------------------
    # 委托给 GARCHModel 的新接口方法
    # ------------------------------------------------------------------

    def get_conditional_vol(self) -> pd.Series:
        """历史条件波动率序列（年化，小数）"""
        self._require_fitted_legacy()
        return self._fit_result.conditional_vol

    def get_standardized_residuals(self) -> pd.Series:
        """标准化残差序列"""
        self._require_fitted_legacy()
        return self._fit_result.std_resid

    def diagnose(self) -> dict:
        """
        模型诊断（委托给 GARCHModel.diagnose）。

        Returns
        -------
        dict  ljung_box_p, arch_test_p, jarque_bera_p,
              skewness, kurtosis, is_stationary, half_life
        """
        self._require_fitted_legacy()
        return self._core.diagnose()

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._fit_result is not None

    @property
    def fit_result(self) -> GARCHFitResult:
        if not self.is_fitted:
            raise RuntimeError("模型尚未拟合，请先调用 fit() 方法")
        return self._fit_result  # type: ignore

    def _require_fitted_legacy(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("模型尚未拟合，请先调用 fit()")
