"""v1_ic评分模块：IC专用映射。

当前参数值与v1_im相同（两品种市场结构相似），
但独立存储，未来可独立优化IC的映射而不影响IM。

900天backtest(IM参数集): 单笔效率2.07x v2, 总PnL -29%(信号-66%)
"""
from strategies.intraday.experimental.score_components_new import (
    compute_q_score, compute_gap_bonus,
)

# IC v1 threshold
THRESHOLD = 60
SIGNAL_VERSION = "v1_ic"

# IC专属M分映射（当前值=IM，未来可独立调整）
_IC_M_THRESHOLDS = [
    (0.0, 0.00122, 0), (0.00122, 0.00191, 17), (0.00191, 0.00247, 11),
    (0.00247, 0.00319, 33), (0.00319, 0.00393, 28), (0.00393, 0.00497, 6),
    (0.00497, 0.00632, 22), (0.00632, 0.00872, 50), (0.00872, 0.0128, 39),
    (0.0128, 1.0, 44),
]

# IC专属V分映射（当前值=IM，未来可独立调整）
_IC_V_THRESHOLDS = [
    (0.0, 0.666, 30), (0.666, 0.776, 20), (0.776, 0.864, 3),
    (0.864, 0.958, 13), (0.958, 1.065, 10), (1.065, 1.215, 17),
    (1.215, 1.424, 0), (1.424, 1.655, 27), (1.655, 1.987, 7),
    (1.987, 100.0, 23),
]

# IC专属时段加分（当前值=IM，未来可独立调整）
_IC_SESSION = {9: 10, 10: -10, 11: 4, 13: 0, 14: 8}


def _ic_m_score(abs_mom: float) -> int:
    for lo, hi, s in _IC_M_THRESHOLDS:
        if lo <= abs_mom < hi:
            return s
    return _IC_M_THRESHOLDS[-1][2]


def _ic_v_score(atr_ratio: float) -> int:
    for lo, hi, s in _IC_V_THRESHOLDS:
        if lo <= atr_ratio < hi:
            return s
    return _IC_V_THRESHOLDS[-1][2]


def _ic_session_bonus(hour_bj: int) -> int:
    return _IC_SESSION.get(hour_bj, 0)


def score(raw_mom_5m: float, raw_atr_ratio: float,
          raw_vol_pct: float = -1.0, raw_vol_ratio: float = -1.0,
          entry_hour_bj: int = 13, gap_aligned: bool = False) -> dict:
    """计算v1_ic评分。"""
    m = _ic_m_score(abs(raw_mom_5m))
    v = _ic_v_score(raw_atr_ratio)
    q = compute_q_score(raw_vol_pct, raw_vol_ratio)
    session = _ic_session_bonus(entry_hour_bj)
    gap = compute_gap_bonus(gap_aligned)

    return {
        'total_score': m + v + q + session + gap,
        'm_score': m,
        'v_score': v,
        'q_score': q,
        'session_bonus': session,
        'gap_bonus': gap,
    }
