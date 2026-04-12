"""v1_ic评分模块：IC专用的统一映射。

V分U型 + 更细的M分档 + 时段加分 + Gap加分。
剔除B分和S分（诊断显示无预测力）。
"""
from strategies.intraday.experimental.score_components_new import (
    compute_m_score, compute_v_score, compute_q_score,
    compute_session_bonus, compute_gap_bonus, compute_total_score,
)

# IC v1 最优 threshold
THRESHOLD = 55
SIGNAL_VERSION = "v1_ic"


def score(raw_mom_5m: float, raw_atr_ratio: float,
          raw_vol_pct: float = -1.0, raw_vol_ratio: float = -1.0,
          entry_hour_bj: int = 13, gap_aligned: bool = False) -> dict:
    """计算v1_ic评分。返回包含所有子分量的dict。"""
    return compute_total_score(
        abs_mom_5m=abs(raw_mom_5m),
        raw_atr_ratio=raw_atr_ratio,
        raw_vol_pct=raw_vol_pct,
        raw_vol_ratio=raw_vol_ratio,
        entry_hour_bj=entry_hour_bj,
        gap_aligned=gap_aligned,
    )
