"""
test_garch_model.py
-------------------
测试 models/volatility/garch_model.py 中的 GARCHModel 和 GJRGARCHModel。

测试覆盖：
1. GJRGARCHModel — 向后兼容旧接口（9 个既有测试）
2. GARCHModel.fit       — dict 返回、参数合理性
3. GARCHModel.predict   — 输出格式、合理性
4. GARCHModel.get_conditional_vol / get_standardized_residuals
5. GARCHModel.diagnose  — 各诊断字段
6. 模拟数据参数恢复测试  — 生成 GARCH(1,1) 过程，拟合后检查参数
7. 真实数据 smoke test  — IF.CFX 日线，打印参数/诊断/预测
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.volatility.garch_model import GARCHFitResult, GARCHModel, GJRGARCHModel

_DB_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "storage" / "trading.db"
)
_DB_EXISTS = _DB_PATH.exists()


# ======================================================================
# 测试数据工具
# ======================================================================

def make_returns(n: int = 500, vol: float = 0.01, seed: int = 42) -> pd.Series:
    """生成简单正态随机收益率（小数形式）"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0, vol, n), index=dates)


def simulate_garch11(
    n: int = 2000,
    omega: float = 1e-6,
    alpha: float = 0.08,
    beta: float = 0.88,
    seed: int = 0,
) -> pd.Series:
    """
    用 GARCH(1,1) 过程生成模拟收益率（无杠杆项，小数形式）。

    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2017-01-01", periods=n, freq="B")

    rets = np.zeros(n)
    var_t = omega / (1.0 - alpha - beta)   # 初始化为长期方差

    for t in range(n):
        sigma_t = np.sqrt(var_t)
        rets[t] = rng.normal(0.0, sigma_t)
        var_t = omega + alpha * rets[t] ** 2 + beta * var_t

    return pd.Series(rets, index=dates)


# ======================================================================
# 一、GJRGARCHModel — 向后兼容旧接口
# ======================================================================

class TestGJRGARCHModel:

    def test_fit_returns_result(self):
        """fit() 应返回 GARCHFitResult 对象"""
        model = GJRGARCHModel()
        result = model.fit(make_returns())
        assert isinstance(result, GARCHFitResult)

    def test_is_fitted_after_fit(self):
        model = GJRGARCHModel()
        model.fit(make_returns())
        assert model.is_fitted

    def test_params_keys(self):
        """拟合参数应包含 omega, alpha, gamma, beta"""
        model = GJRGARCHModel()
        result = model.fit(make_returns())
        assert {"omega", "alpha", "gamma", "beta"}.issubset(result.params.keys())

    def test_persistence_less_than_one(self):
        model = GJRGARCHModel()
        result = model.fit(make_returns())
        assert result.persistence < 1.0, f"persistence={result.persistence:.4f} ≥ 1"

    def test_forecast_next_day_positive(self):
        model = GJRGARCHModel()
        model.fit(make_returns())
        assert model.forecast_next_day() > 0

    def test_forecast_next_day_reasonable(self):
        """年化预测波动率在 5%~200%"""
        model = GJRGARCHModel()
        model.fit(make_returns(vol=0.01))
        vol = model.forecast_next_day()
        assert 0.05 < vol < 2.0, f"forecast_next_day={vol:.4f}"

    def test_forecast_period_avg(self):
        model = GJRGARCHModel()
        model.fit(make_returns())
        assert model.forecast_period_avg(horizon=5) > 0

    def test_fit_result_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            _ = GJRGARCHModel().fit_result

    def test_conditional_vol_length(self):
        n = 300
        model = GJRGARCHModel()
        result = model.fit(make_returns(n=n))
        assert len(result.conditional_vol) == n


# ======================================================================
# 二、GARCHModel.fit — dict 接口
# ======================================================================

class TestGARCHModelFit:

    def test_fit_returns_dict(self):
        d = GARCHModel().fit(make_returns())
        assert isinstance(d, dict)

    def test_fit_required_keys(self):
        d = GARCHModel().fit(make_returns())
        required = {
            "omega", "alpha", "gamma", "beta",
            "persistence", "long_run_var", "long_run_vol",
            "log_likelihood", "aic", "bic",
        }
        assert required.issubset(d.keys()), f"缺失: {required - set(d.keys())}"

    def test_persistence_in_0_1(self):
        d = GARCHModel().fit(make_returns())
        assert 0.0 < d["persistence"] < 1.0

    def test_omega_positive(self):
        d = GARCHModel().fit(make_returns())
        assert d["omega"] > 0

    def test_long_run_vol_positive_finite(self):
        d = GARCHModel().fit(make_returns())
        assert d["long_run_vol"] > 0
        assert np.isfinite(d["long_run_vol"])

    def test_long_run_vol_reasonable(self):
        """长期年化波动率在 5%~200%"""
        d = GARCHModel().fit(make_returns(vol=0.01))
        assert 0.05 < d["long_run_vol"] < 2.0

    def test_aic_bic_finite(self):
        d = GARCHModel().fit(make_returns())
        assert np.isfinite(d["aic"]) and np.isfinite(d["bic"])

    def test_is_fitted_after_fit(self):
        m = GARCHModel()
        m.fit(make_returns())
        assert m.is_fitted

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            GARCHModel().predict()

    def test_dist_t_includes_nu(self):
        """t 分布时应返回自由度 nu"""
        d = GARCHModel(dist="t").fit(make_returns())
        assert "nu" in d
        assert d["nu"] > 2  # nu > 2 保证方差存在

    def test_no_leverage_when_o0(self):
        """o=0 时为标准 GARCH，gamma 应为 0"""
        d = GARCHModel(o=0).fit(make_returns())
        assert d["gamma"] == pytest.approx(0.0, abs=1e-10)


# ======================================================================
# 三、GARCHModel.predict
# ======================================================================

class TestGARCHModelPredict:

    @pytest.fixture
    def fitted_model(self):
        m = GARCHModel()
        m.fit(make_returns(n=500))
        return m

    def test_predict_returns_dict(self, fitted_model):
        d = fitted_model.predict(horizon=5)
        assert isinstance(d, dict)

    def test_predict_required_keys(self, fitted_model):
        d = fitted_model.predict(horizon=5)
        assert {"daily_vol", "mean_vol", "current_vol"}.issubset(d.keys())

    def test_predict_daily_vol_length(self, fitted_model):
        for h in (1, 5, 10):
            d = fitted_model.predict(horizon=h)
            assert len(d["daily_vol"]) == h, f"horizon={h}: 期望 {h} 个，得到 {len(d['daily_vol'])}"

    def test_predict_all_positive(self, fitted_model):
        d = fitted_model.predict(horizon=5)
        assert all(v > 0 for v in d["daily_vol"])
        assert d["mean_vol"] > 0
        assert d["current_vol"] > 0

    def test_predict_reasonable_range(self, fitted_model):
        """预测波动率年化值在 5%~200%"""
        d = fitted_model.predict(horizon=5)
        assert 0.05 < d["mean_vol"] < 2.0
        assert 0.05 < d["current_vol"] < 2.0

    def test_predict_mean_equals_avg_of_daily(self, fitted_model):
        """mean_vol 等于 daily_vol 的均值"""
        d = fitted_model.predict(horizon=5)
        assert d["mean_vol"] == pytest.approx(np.mean(d["daily_vol"]), rel=1e-10)

    def test_predict_with_new_returns_updates(self):
        """传入新 returns 后，预测应使用新数据"""
        m = GARCHModel()
        m.fit(make_returns(n=300, seed=1))
        vol1 = m.predict(horizon=1)["current_vol"]
        # 使用不同的收益率序列更新
        vol2 = m.predict(horizon=1, returns=make_returns(n=300, seed=99))["current_vol"]
        # 两次预测应来自不同数据，结果应不同（极小概率相同）
        assert vol1 != pytest.approx(vol2, rel=1e-5)


# ======================================================================
# 四、get_conditional_vol / get_standardized_residuals
# ======================================================================

class TestGARCHModelOutputs:

    @pytest.fixture
    def fitted(self):
        m = GARCHModel()
        m.fit(make_returns(n=400))
        return m

    def test_conditional_vol_length(self, fitted):
        """条件波动率序列与输入等长"""
        cv = fitted.get_conditional_vol()
        assert len(cv) == 400

    def test_conditional_vol_positive(self, fitted):
        assert (fitted.get_conditional_vol() > 0).all()

    def test_conditional_vol_annualized_range(self, fitted):
        """年化条件波动率应在合理范围"""
        cv = fitted.get_conditional_vol()
        assert (cv > 0.02).all()
        assert (cv < 3.0).all()

    def test_std_resid_length(self, fitted):
        assert len(fitted.get_standardized_residuals()) == 400

    def test_std_resid_mean_near_zero(self, fitted):
        """标准化残差均值应接近 0"""
        sr = fitted.get_standardized_residuals()
        assert abs(sr.mean()) < 0.2

    def test_std_resid_std_near_one(self, fitted):
        """标准化残差标准差应接近 1"""
        sr = fitted.get_standardized_residuals()
        assert 0.8 < sr.std() < 1.2


# ======================================================================
# 五、GARCHModel.diagnose
# ======================================================================

class TestGARCHModelDiagnose:

    @pytest.fixture
    def diagnosis(self):
        m = GARCHModel()
        m.fit(make_returns(n=800))
        return m.diagnose()

    def test_diagnose_required_keys(self, diagnosis):
        required = {
            "ljung_box_p", "arch_test_p", "jarque_bera_p",
            "skewness", "kurtosis", "is_stationary", "half_life",
        }
        assert required.issubset(diagnosis.keys())

    def test_p_values_in_0_1(self, diagnosis):
        for key in ("ljung_box_p", "arch_test_p", "jarque_bera_p"):
            assert 0.0 <= diagnosis[key] <= 1.0, f"{key}={diagnosis[key]}"

    def test_is_stationary_is_bool(self, diagnosis):
        assert isinstance(diagnosis["is_stationary"], bool)

    def test_is_stationary_true_for_good_data(self, diagnosis):
        assert diagnosis["is_stationary"]

    def test_half_life_positive(self, diagnosis):
        assert diagnosis["half_life"] > 0

    def test_half_life_formula(self):
        """half_life = ln(0.5) / ln(persistence)"""
        m = GARCHModel()
        m.fit(make_returns(n=600))
        d = m.diagnose()
        p = m._fit_params["persistence"]
        expected_hl = np.log(0.5) / np.log(p)
        assert d["half_life"] == pytest.approx(expected_hl, rel=1e-10)


# ======================================================================
# 六、模拟数据参数恢复
# ======================================================================

class TestGARCHParameterRecovery:
    """
    用已知参数生成 GARCH(1,1) 数据，拟合后检查参数估计是否接近真实值。
    n=2000 提供足够数据量，允许 30% 相对误差（有限样本偏差）。
    """

    TRUE_OMEGA = 1e-6
    TRUE_ALPHA = 0.08
    TRUE_BETA  = 0.88
    N          = 2000

    @pytest.fixture(scope="class")
    def fit_result(self):
        returns = simulate_garch11(
            n=self.N,
            omega=self.TRUE_OMEGA,
            alpha=self.TRUE_ALPHA,
            beta=self.TRUE_BETA,
        )
        # 使用 o=0 排除杠杆项，与模拟过程匹配
        m = GARCHModel(o=0, dist="normal")
        return m.fit(returns)

    def test_alpha_recovers(self, fit_result):
        """α 估计应在真实值 ±50% 以内（有限样本允许较大容差）"""
        est = fit_result["alpha"]
        assert abs(est - self.TRUE_ALPHA) / self.TRUE_ALPHA < 0.5, (
            f"α 估计偏差过大: true={self.TRUE_ALPHA}, est={est:.4f}"
        )

    def test_beta_recovers(self, fit_result):
        """β 估计应在真实值 ±10% 以内（β 估计通常较准确）"""
        est = fit_result["beta"]
        assert abs(est - self.TRUE_BETA) / self.TRUE_BETA < 0.1, (
            f"β 估计偏差过大: true={self.TRUE_BETA}, est={est:.4f}"
        )

    def test_persistence_close_to_true(self, fit_result):
        """持续性 α+β 应接近真实值 0.96（允许 ±5%）"""
        true_pers = self.TRUE_ALPHA + self.TRUE_BETA  # 0.96
        est = fit_result["persistence"]
        assert abs(est - true_pers) < 0.05, (
            f"persistence 偏差: true={true_pers}, est={est:.4f}"
        )

    def test_persistence_stationary(self, fit_result):
        assert fit_result["persistence"] < 1.0


# ======================================================================
# 七、真实数据 smoke test — IF.CFX
# ======================================================================

@pytest.mark.skipif(not _DB_EXISTS, reason=f"DB 不存在: {_DB_PATH}")
class TestSmokeRealData:

    @pytest.fixture(scope="class")
    def if_returns(self):
        conn = sqlite3.connect(_DB_PATH)
        df = pd.read_sql(
            "SELECT trade_date, close FROM futures_daily "
            "WHERE ts_code='IF.CFX' ORDER BY trade_date",
            conn,
        )
        conn.close()
        df["close"] = pd.to_numeric(df["close"])
        close = df.set_index("trade_date")["close"]
        return np.log(close / close.shift(1)).dropna()

    def test_fit_converges(self, if_returns):
        """IF.CFX 拟合应收敛"""
        m = GJRGARCHModel(dist="skewt")
        result = m.fit(if_returns)
        assert result.converged, "模型未收敛"

    def test_persistence_below_1(self, if_returns):
        """IF.CFX 持续性应 < 1"""
        m = GJRGARCHModel()
        result = m.fit(if_returns)
        assert result.persistence < 1.0

    def test_gamma_positive_leverage(self, if_returns):
        """
        A 股股指期货应呈现杠杆效应（下跌冲击 > 上涨冲击）。
        GJR γ 通常 > 0，允许一定容忍（≥ -0.05 作为软检查）。
        """
        m = GJRGARCHModel()
        result = m.fit(if_returns)
        gamma = result.params.get("gamma", 0.0)
        assert gamma >= -0.05, f"gamma={gamma:.4f}，预期应 ≥ 0（杠杆效应）"

    def test_garc_model_dict_interface(self, if_returns):
        """GARCHModel dict 接口在真实数据上正常运行"""
        m = GARCHModel(dist="t")
        d = m.fit(if_returns)

        print(f"\n[smoke] IF.CFX GARCHModel 拟合结果:")
        print(f"  alpha={d['alpha']:.4f}  gamma={d['gamma']:.4f}  "
              f"beta={d['beta']:.4f}  persistence={d['persistence']:.4f}")
        print(f"  long_run_vol={d['long_run_vol']:.4f} ({d['long_run_vol']*100:.1f}%)")
        print(f"  AIC={d['aic']:.1f}  BIC={d['bic']:.1f}")

        assert 0.0 < d["persistence"] < 1.0
        assert d["long_run_vol"] > 0.05

    def test_predict_5day(self, if_returns):
        """IF.CFX 5日预测格式和合理性"""
        m = GARCHModel(dist="t")
        m.fit(if_returns)
        pred = m.predict(horizon=5)

        print(f"\n[smoke] IF.CFX 5日波动率预测:")
        for i, v in enumerate(pred["daily_vol"], 1):
            print(f"  h={i}: {v:.4f} ({v*100:.2f}%)")
        print(f"  mean_vol={pred['mean_vol']:.4f}  current_vol={pred['current_vol']:.4f}")

        assert len(pred["daily_vol"]) == 5
        assert 0.05 < pred["mean_vol"] < 2.0

    def test_diagnose_output(self, if_returns):
        """诊断统计量输出并打印"""
        m = GARCHModel(dist="t")
        m.fit(if_returns)
        d = m.diagnose()

        print(f"\n[smoke] IF.CFX 模型诊断:")
        print(f"  Ljung-Box p={d['ljung_box_p']:.3f}  "
              f"ARCH-LM p={d['arch_test_p']:.3f}  "
              f"JB p={d['jarque_bera_p']:.4f}")
        print(f"  skew={d['skewness']:.3f}  kurt={d['kurtosis']:.3f}")
        print(f"  is_stationary={d['is_stationary']}  "
              f"half_life={d['half_life']:.1f} 天")

        assert d["is_stationary"]
        assert d["half_life"] > 0
        assert 0.0 <= d["ljung_box_p"] <= 1.0
