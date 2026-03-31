"""
regime_detection.py
-------------------
职责：使用隐马尔可夫模型（HMM）检测市场状态（低波动/高波动/趋势）。

市场状态可用于：
- 策略切换（高波动时暂停趋势策略，切换到波动率套利）
- 调整持仓规模（高波动期降低仓位）
- 期权策略参数调整（高波动期提高 VRP 阈值）

依赖：hmmlearn
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeState(str, Enum):
    """市场状态"""
    LOW_VOL = "low_vol"         # 低波动（均值回归市场，适合期权卖方）
    HIGH_VOL = "high_vol"       # 高波动（趋势市场，适合趋势策略）
    TRANSITION = "transition"   # 状态转换中（不确定期，降低仓位）


@dataclass
class HMMRegimeResult:
    """
    HMM 状态检测结果。

    Attributes
    ----------
    trade_date : str
        检测日期
    current_state : RegimeState
        当前市场状态
    state_probability : dict[str, float]
        各状态的概率
    state_sequence : pd.Series
        历史状态序列（用于可视化）
    transition_matrix : np.ndarray
        状态转移矩阵
    """
    trade_date: str
    current_state: RegimeState
    state_probability: dict[str, float] = field(default_factory=dict)
    state_sequence: Optional[pd.Series] = None
    transition_matrix: Optional[np.ndarray] = None


class HMMRegimeDetector:
    """
    隐马尔可夫模型市场状态检测器。

    Parameters
    ----------
    n_states : int
        隐状态数量，默认 2（低波动/高波动）
    n_iter : int
        EM 算法最大迭代次数
    features : list[str]
        用于状态检测的特征列名（默认：收益率、RV、波动率变化）

    Examples
    --------
    >>> detector = HMMRegimeDetector(n_states=2)
    >>> detector.fit(feature_df)
    >>> result = detector.predict("20240101", feature_df)
    >>> result.current_state
    RegimeState.LOW_VOL
    """

    def __init__(
        self,
        n_states: int = 2,
        n_iter: int = 100,
        features: Optional[list[str]] = None,
    ) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.features = features or ["log_return", "rv", "rv_change"]
        self._model = None   # hmmlearn GaussianHMM 实例
        self._state_labels: dict[int, RegimeState] = {}
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(self, feature_df: pd.DataFrame) -> "HMMRegimeDetector":
        """
        拟合 HMM 模型。

        Parameters
        ----------
        feature_df : pd.DataFrame
            特征 DataFrame，列包含 self.features 中的特征名

        Returns
        -------
        HMMRegimeDetector
            self（支持链式调用）

        Notes
        -----
        - 使用 hmmlearn.hmm.GaussianHMM
        - 拟合后自动标记状态（波动率低的状态 → LOW_VOL，高的 → HIGH_VOL）
        - 特征在拟合前标准化（zero-mean, unit-variance）
        """
        from hmmlearn.hmm import GaussianHMM

        # 提取特征并去除 NaN
        avail_features = [f for f in self.features if f in feature_df.columns]
        if not avail_features:
            raise ValueError(f"特征列 {self.features} 均不在 DataFrame 中")

        df = feature_df[avail_features].dropna()
        if len(df) < self.n_states * 10:
            raise ValueError(
                f"样本量 {len(df)} 不足，HMM 拟合至少需要 {self.n_states * 10} 行有效数据"
            )

        X = df.values.astype(float)

        # 标准化
        self._feature_means = X.mean(axis=0)
        self._feature_stds = X.std(axis=0) + 1e-8
        X_scaled = (X - self._feature_means) / self._feature_stds
        self._fitted_features = avail_features

        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=42,
        )
        model.fit(X_scaled)
        self._model = model

        # 按波动率特征均值对状态排序（低 → HIGH，高 → HIGH_VOL）
        # 优先使用 rv 特征，否则用绝对均值
        if "rv" in avail_features:
            rv_idx = avail_features.index("rv")
        elif "log_return" in avail_features:
            rv_idx = avail_features.index("log_return")
        else:
            rv_idx = 0

        state_vol = [float(model.means_[s, rv_idx]) for s in range(self.n_states)]
        sorted_states = np.argsort(state_vol)

        if self.n_states == 2:
            self._state_labels = {
                int(sorted_states[0]): RegimeState.LOW_VOL,
                int(sorted_states[1]): RegimeState.HIGH_VOL,
            }
        else:
            self._state_labels = {}
            self._state_labels[int(sorted_states[0])] = RegimeState.LOW_VOL
            self._state_labels[int(sorted_states[-1])] = RegimeState.HIGH_VOL
            for s in sorted_states[1:-1]:
                self._state_labels[int(s)] = RegimeState.TRANSITION

        logger.info(
            "HMM 拟合完成：%d 状态，样本量=%d，收敛=%s",
            self.n_states, len(df), model.monitor_.converged
        )
        return self

    def predict(
        self,
        trade_date: str,
        feature_df: pd.DataFrame,
    ) -> HMMRegimeResult:
        """
        预测当日市场状态。

        Parameters
        ----------
        trade_date : str
            预测日期，格式 YYYYMMDD
        feature_df : pd.DataFrame
            特征数据（含历史特征，最后一行为当日）

        Returns
        -------
        HMMRegimeResult
        """
        if not self.is_fitted:
            raise RuntimeError("HMMRegimeDetector 未拟合，请先调用 fit()")

        df = feature_df[self._fitted_features].dropna()
        X = df.values.astype(float)
        X_scaled = (X - self._feature_means) / self._feature_stds

        states = self._model.predict(X_scaled)
        proba = self._model.predict_proba(X_scaled)

        current_raw = int(states[-1])
        current_state = self._state_labels.get(current_raw, RegimeState.TRANSITION)

        state_probability = {
            label.value: float(proba[-1, raw])
            for raw, label in self._state_labels.items()
        }

        state_sequence = pd.Series(
            [self._state_labels.get(int(s), RegimeState.TRANSITION) for s in states],
            index=df.index,
            name="regime",
        )

        return HMMRegimeResult(
            trade_date=trade_date,
            current_state=current_state,
            state_probability=state_probability,
            state_sequence=state_sequence,
            transition_matrix=self._model.transmat_.copy(),
        )

    def get_state_statistics(self) -> dict[str, dict]:
        """
        获取各状态的特征均值和标准差（反标准化回原始空间）。

        Returns
        -------
        dict
            {state_name: {feature: {"mean": float, "std": float}}}
        """
        if not self.is_fitted:
            raise RuntimeError("HMMRegimeDetector 未拟合，请先调用 fit()")

        result = {}
        for raw_state, label in self._state_labels.items():
            stats: dict[str, dict] = {}
            for i, feature in enumerate(self._fitted_features):
                # 反标准化均值
                mean_scaled = float(self._model.means_[raw_state, i])
                mean_orig = mean_scaled * self._feature_stds[i] + self._feature_means[i]

                # 反标准化方差 → std
                var_scaled = float(self._model.covars_[raw_state][i, i])
                std_orig = float(np.sqrt(max(var_scaled, 0))) * self._feature_stds[i]

                stats[feature] = {"mean": float(mean_orig), "std": float(std_orig)}
            result[label.value] = stats

        return result
