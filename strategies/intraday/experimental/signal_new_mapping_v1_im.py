"""v1_im评分模块：IM专用的数据驱动映射。

基于IM 900天PnL数据驱动设计（Phase C）。
M/V分用10桶按PnL排序分配分数，时段加分基于IM实际时段PnL。
Q分和Gap保持统一逻辑。
"""
from strategies.intraday.experimental.score_components_new import (
    compute_q_score, compute_gap_bonus,
)

# IM v1 最优 threshold
THRESHOLD = 60
SIGNAL_VERSION = "v1_im"

# IM专属M分映射（10桶，按PnL从低到高分配0-50分）
_IM_M_THRESHOLDS = [
    (0.0, 0.00122, 0), (0.00122, 0.00191, 17), (0.00191, 0.00247, 11),
    (0.00247, 0.00319, 33), (0.00319, 0.00393, 28), (0.00393, 0.00497, 6),
    (0.00497, 0.00632, 22), (0.00632, 0.00872, 50), (0.00872, 0.0128, 39),
    (0.0128, 1.0, 44),
]

# IM专属V分映射（10桶，不规则多峰）
_IM_V_THRESHOLDS = [
    (0.0, 0.666, 30), (0.666, 0.776, 20), (0.776, 0.864, 3),
    (0.864, 0.958, 13), (0.958, 1.065, 10), (1.065, 1.215, 17),
    (1.215, 1.424, 0), (1.424, 1.655, 27), (1.655, 1.987, 7),
    (1.987, 100.0, 23),
]

# IM专属时段加分
_IM_SESSION = {9: 10, 10: -10, 11: 4, 13: 0, 14: 8}


def _im_m_score(abs_mom: float) -> int:
    for lo, hi, s in _IM_M_THRESHOLDS:
        if lo <= abs_mom < hi:
            return s
    return _IM_M_THRESHOLDS[-1][2]


def _im_v_score(atr_ratio: float) -> int:
    for lo, hi, s in _IM_V_THRESHOLDS:
        if lo <= atr_ratio < hi:
            return s
    return _IM_V_THRESHOLDS[-1][2]


def _im_session_bonus(hour_bj: int) -> int:
    return _IM_SESSION.get(hour_bj, 0)


def score(raw_mom_5m: float, raw_atr_ratio: float,
          raw_vol_pct: float = -1.0, raw_vol_ratio: float = -1.0,
          entry_hour_bj: int = 13, gap_aligned: bool = False) -> dict:
    """计算v1_im评分。返回包含所有子分量的dict。"""
    m = _im_m_score(abs(raw_mom_5m))
    v = _im_v_score(raw_atr_ratio)
    q = compute_q_score(raw_vol_pct, raw_vol_ratio)
    session = _im_session_bonus(entry_hour_bj)
    gap = compute_gap_bonus(gap_aligned)

    return {
        'total_score': m + v + q + session + gap,
        'm_score': m,
        'v_score': v,
        'q_score': q,
        'session_bonus': session,
        'gap_bonus': gap,
    }
