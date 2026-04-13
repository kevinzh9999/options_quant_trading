"""v1_im评分模块：IM专用的per-symbol数据驱动映射。

基于IM 220天v2 trade数据（20250516-20260413），10桶PnL-ranked M/V映射。
Q分反直觉：低成交量百分位更好（IC/IM一致）。
Session: 14点最好(+10)，11点最差(-10)。
Gap: IM premium仅+0.51pt，不给bonus。

用途：v2的高置信度过滤器（Intersection信号加仓，V2\V1信号减仓/不交易）。
"""

THRESHOLD = 60
SIGNAL_VERSION = "v1_im"

# IM per-symbol M映射（10桶，按PnL排序分配0-50分）
# 最佳: bucket4 (0.0032-0.0038) +9.97pt → 50分
# 最差: bucket2 (0.0017-0.0026) -3.26pt → 0分
_IM_M_THRESHOLDS = [
    (0.0, 0.0017, 23),
    (0.0017, 0.0026, 0),
    (0.0026, 0.0032, 19),
    (0.0032, 0.0038, 50),
    (0.0038, 0.0047, 16),
    (0.0047, 0.0055, 13),
    (0.0055, 0.0072, 35),
    (0.0072, 0.0097, 40),
    (0.0097, 0.0139, 11),
    (0.0139, 1.0, 37),
]

# IM per-symbol V映射（7桶，非单调）
# 最佳: bucket6 ATR 1.55-2.02 +8.91pt → 30分
# 最差: bucket5 ATR 1.25-1.55 -2.12pt → 0分
_IM_V_THRESHOLDS = [
    (0.0, 0.7527, 5),
    (0.7527, 0.9031, 12),
    (0.9031, 1.0400, 25),
    (1.0400, 1.2434, 16),
    (1.2434, 1.5461, 0),
    (1.5461, 2.0152, 30),
    (2.0152, 100.0, 15),
]

# IM per-symbol Q映射（3档，反直觉：低量最好）
# 低量(0-0.60) +4.24pt → 15分, 中量(0.60-0.90) +3.92pt → 13分, 高量(0.90-1.0) +1.60pt → 0分
_IM_Q_TIERS = [
    (0.0, 0.60, 15),
    (0.60, 0.90, 13),
    (0.90, 1.01, 0),
]

# IM per-symbol Session bonus
# 14点最好(+5.95pt→+10), 9点次好(+5.29pt→+8), 11点最差(-0.14pt→-10)
_IM_SESSION = {9: 8, 10: -2, 11: -10, 13: -2, 14: 10}

# IM gap bonus: premium仅+0.51pt，不给bonus
_IM_GAP_BONUS = 0


def _lookup(value: float, thresholds: list) -> int:
    for lo, hi, s in thresholds:
        if lo <= value < hi:
            return s
    return thresholds[-1][2]


def score(raw_mom_5m: float, raw_atr_ratio: float,
          raw_vol_pct: float = -1.0, raw_vol_ratio: float = -1.0,
          entry_hour_bj: int = 13, gap_aligned: bool = False) -> dict:
    """计算v1_im per-symbol评分。返回包含所有子分量的dict。"""
    m = _lookup(abs(raw_mom_5m), _IM_M_THRESHOLDS)
    v = _lookup(raw_atr_ratio, _IM_V_THRESHOLDS)

    # Q分：per-symbol反直觉映射（低量好）
    if raw_vol_pct >= 0:
        q = _lookup(raw_vol_pct, _IM_Q_TIERS)
    elif raw_vol_ratio > 0:
        # ratio fallback（保持旧逻辑兼容）
        if raw_vol_ratio > 1.5:
            q = 0   # 高量 = 差
        elif raw_vol_ratio > 0.5:
            q = 13
        else:
            q = 15  # 低量 = 好
    else:
        q = 13  # 不可用时中性

    session = _IM_SESSION.get(entry_hour_bj, 0)
    gap = _IM_GAP_BONUS if gap_aligned else 0

    return {
        'total_score': m + v + q + session + gap,
        'm_score': m,
        'v_score': v,
        'q_score': q,
        'session_bonus': session,
        'gap_bonus': gap,
    }
