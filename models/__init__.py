"""
models 包：模型层（重构后结构）

子包：
  models.volatility   - 波动率计算和预测（GARCH、RV、HAR-RV）
  models.pricing      - 期权定价和 Greeks（BS、IV、波动率曲面）
  models.statistics   - 统计模型（协整、OU 过程、HMM 状态检测）
  models.indicators   - 技术指标（趋势/动量/波动率/成交量）

向后兼容：原 models.X 路径的导入仍然有效（通过当前文件重新导出）。
"""

# 向后兼容：从新路径重新导出，保持旧代码不需改动
from .volatility.realized_vol import compute_realized_vol, compute_rolling_rv, RVEstimator
from .volatility.garch_model import GJRGARCHModel, GARCHFitResult
from .pricing.implied_vol import calc_implied_vol, bs_price, OptionType
from .pricing.vol_surface import VolSurface, VolSmile
from .pricing.greeks import Greeks, PortfolioGreeks, calc_all_greeks, calc_portfolio_greeks

__all__ = [
    # volatility
    "compute_realized_vol", "compute_rolling_rv", "RVEstimator",
    "GJRGARCHModel", "GARCHFitResult",
    # pricing
    "calc_implied_vol", "bs_price", "OptionType",
    "VolSurface", "VolSmile",
    "Greeks", "PortfolioGreeks", "calc_all_greeks", "calc_portfolio_greeks",
]
