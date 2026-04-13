"""v1_ic评分模块：IC专用的per-symbol数据驱动映射。

基于IC 220天v2 trade数据（20250516-20260413），10桶PnL-ranked M/V映射。
IC与IM因子特征显著不同：
  - M: 非线性，低动量(0-0.001)和高动量(0.008+)都好，中间最差
  - V: ATR 1.35-1.74最佳，0.71-0.84最差
  - Q: 反直觉，低成交量百分位最好
  - Session: 9点最好(+10)，10点最差(-10)（IM是14点最好11点最差）
  - Gap: +4.76pt premium → bonus=5（IM几乎无premium → bonus=0）

用途：v2的高置信度过滤器（Intersection信号加仓，V2\V1信号减仓/不交易）。
"""

THRESHOLD = 60
SIGNAL_VERSION = "v1_ic"

# IC per-symbol M映射（10桶，按PnL排序分配0-50分）
# 最佳: bucket9 (0.0082-0.0121) +7.61pt → 50分
# 最差: bucket5 (0.0030-0.0036) -0.85pt → 0分
_IC_M_THRESHOLDS = [
    (0.0, 0.0011, 33),
    (0.0011, 0.0016, 11),
    (0.0016, 0.0022, 17),
    (0.0022, 0.0030, 6),
    (0.0030, 0.0036, 0),
    (0.0036, 0.0047, 22),
    (0.0047, 0.0061, 39),
    (0.0061, 0.0080, 28),
    (0.0080, 0.0121, 50),
    (0.0121, 1.0, 44),
]

# IC per-symbol V映射（7桶，非单调）
# 最佳: bucket6 ATR 1.35-1.74 +8.76pt → 30分
# 最差: bucket2 ATR 0.71-0.84 -0.40pt → 0分
_IC_V_THRESHOLDS = [
    (0.0, 0.7057, 25),
    (0.7057, 0.8409, 0),
    (0.8409, 0.9872, 20),
    (0.9872, 1.1464, 5),
    (1.1464, 1.3453, 15),
    (1.3453, 1.7412, 30),
    (1.7412, 100.0, 10),
]

# IC per-symbol Q映射（3档，反直觉：低量最好）
_IC_Q_TIERS = [
    (0.0, 0.45, 15),
    (0.45, 0.80, 8),
    (0.80, 1.01, 0),
]

# IC per-symbol Session bonus
_IC_SESSION = {9: 10, 10: -10, 11: 1, 13: -2, 14: 4}

# IC gap bonus
_IC_GAP_BONUS = 5


def _lookup(value: float, thresholds: list) -> int:
    for lo, hi, s in thresholds:
        if lo <= value < hi:
            return s
    return thresholds[-1][2]


def score(raw_mom_5m: float, raw_atr_ratio: float,
          raw_vol_pct: float = -1.0, raw_vol_ratio: float = -1.0,
          entry_hour_bj: int = 13, gap_aligned: bool = False) -> dict:
    """计算v1_ic per-symbol评分。返回包含所有子分量的dict。"""
    m = _lookup(abs(raw_mom_5m), _IC_M_THRESHOLDS)
    v = _lookup(raw_atr_ratio, _IC_V_THRESHOLDS)

    if raw_vol_pct >= 0:
        q = _lookup(raw_vol_pct, _IC_Q_TIERS)
    elif raw_vol_ratio > 0:
        if raw_vol_ratio > 1.5:
            q = 0
        elif raw_vol_ratio > 0.5:
            q = 8
        else:
            q = 15
    else:
        q = 8

    session = _IC_SESSION.get(entry_hour_bj, 0)
    gap = _IC_GAP_BONUS if gap_aligned else 0

    return {
        'total_score': m + v + q + session + gap,
        'm_score': m,
        'v_score': v,
        'q_score': q,
        'session_bonus': session,
        'gap_bonus': gap,
    }
