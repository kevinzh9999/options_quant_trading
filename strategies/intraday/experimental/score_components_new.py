"""new_mapping_v1 子分量映射函数。

每个函数独立实现一个子分量的映射逻辑，便于后续单独调整。
"""


def compute_m_score(abs_mom_5m: float) -> int:
    """M分映射（max 50）。

    基于MVQB诊断：v2的0.3%边界处M=25和M=35 PnL几乎一样(+2.6/+2.6)，
    说明0.3%不是有效断点。v1用更细的7档映射捕捉动量强度的渐进效应。

    Args:
        abs_mom_5m: 5分钟动量绝对值百分比(如0.0024=0.24%)
    """
    if abs_mom_5m < 0.0005:
        return 0
    elif abs_mom_5m < 0.001:
        return 5
    elif abs_mom_5m < 0.002:
        return 15
    elif abs_mom_5m < 0.003:
        return 25
    elif abs_mom_5m < 0.005:
        return 30
    elif abs_mom_5m < 0.010:
        return 40
    else:
        return 50


def compute_v_score(raw_atr_ratio: float) -> int:
    """V分U型映射（max 30）。

    基于MVQB诊断关键发现：v2假设"ATR ratio越高越差"是错的。
    实际数据显示U型——ratio<0.7(盘整待启动)和ratio>1.5(大趋势中)都表现好，
    中间状态(0.8-1.2)最差。

    Args:
        raw_atr_ratio: ATR(5)/ATR(40)比值
    """
    if raw_atr_ratio < 0.6:
        return 30
    elif raw_atr_ratio < 0.8:
        return 25
    elif raw_atr_ratio < 1.0:
        return 10
    elif raw_atr_ratio < 1.2:
        return 5
    elif raw_atr_ratio < 1.5:
        return 15
    elif raw_atr_ratio < 2.0:
        return 25
    else:
        return 30


def compute_q_score(raw_vol_pct: float = -1.0, raw_vol_ratio: float = -1.0) -> int:
    """Q分映射（max 15）。

    保持v2的三档结构，最大值从20降到15。
    优先用percentile法，fallback到ratio法。

    Args:
        raw_vol_pct: 成交量百分位(0-1)，-1表示不可用
        raw_vol_ratio: 成交量/20bar均值比，-1表示不可用
    """
    if raw_vol_pct >= 0:
        # percentile法
        if raw_vol_pct > 0.75:
            return 15
        elif raw_vol_pct > 0.25:
            return 8
        else:
            return 0
    elif raw_vol_ratio > 0:
        # ratio法fallback
        if raw_vol_ratio > 1.5:
            return 15
        elif raw_vol_ratio > 0.5:
            return 8
        else:
            return 0
    else:
        return 8  # 数据不可用时中性值


def compute_session_bonus(entry_hour_bj: int) -> int:
    """时段加分（-10到+10）。

    基于MVQB诊断任务6：
    - 09:00-10:00和14:00-15:00是两品种最好的时段(AvgPnL +3-4pt)
    - 10:00-11:00是最差时段(AvgPnL +0-1pt)
    - 两品种完全一致

    Args:
        entry_hour_bj: 北京时间小时(9-14)
    """
    if entry_hour_bj == 9:
        return 10
    elif entry_hour_bj == 10:
        return -10
    elif entry_hour_bj == 11:
        return -5
    elif entry_hour_bj == 13:
        return 0
    elif entry_hour_bj == 14:
        return 10
    else:
        return 0


def compute_gap_bonus(gap_aligned: bool) -> int:
    """Gap对齐加分（0到+5）。

    基于MVQB诊断：gap_aligned=True时AvgPnL比False高约1pt(两品种一致)。

    Args:
        gap_aligned: 开盘gap方向与持仓方向是否一致
    """
    return 5 if gap_aligned else 0


def compute_total_score(
    abs_mom_5m: float,
    raw_atr_ratio: float,
    raw_vol_pct: float = -1.0,
    raw_vol_ratio: float = -1.0,
    entry_hour_bj: int = 13,
    gap_aligned: bool = False,
) -> dict:
    """组合总分。

    total = M(0-50) + V(0-30) + Q(0-15) + session(-10~+10) + gap(0~+5)
    范围: -10 到 110

    Returns:
        dict with total_score and all components
    """
    m = compute_m_score(abs_mom_5m)
    v = compute_v_score(raw_atr_ratio)
    q = compute_q_score(raw_vol_pct, raw_vol_ratio)
    session = compute_session_bonus(entry_hour_bj)
    gap = compute_gap_bonus(gap_aligned)

    return {
        'total_score': m + v + q + session + gap,
        'm_score': m,
        'v_score': v,
        'q_score': q,
        'session_bonus': session,
        'gap_bonus': gap,
    }
