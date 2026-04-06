"""
quadrant_monitor.py
-------------------
策略象限监控面板。每5分钟刷新，显示当前市场四象限位置（A/B/C/D），
与存量持仓对比，给出调仓建议。象限切换时重点 alert。

四象限框架（参考 STRATEGY_PLAYBOOK.md）：
  - A: 高IV + 偏多 → 卖Strangle
  - B: 高IV + 偏空 → B1(恐慌进行中)买Put / B2(恐慌见顶)卖Vega
  - C: 低IV + 偏多 → 买Call / IM+Put吃贴水
  - D: 低IV + 偏空 → 卖Call Spread

用法：
    python scripts/quadrant_monitor.py              # 持续监控（每5分钟刷新）
    python scripts/quadrant_monitor.py --once        # 只刷新一次
"""

from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.config_loader import ConfigLoader
from data.storage.db_manager import DBManager, get_db

W = 80  # 面板宽度


# ---------------------------------------------------------------------------
# Hurst 指数
# ---------------------------------------------------------------------------

def _calc_hurst_rs(prices: np.ndarray) -> float:
    """R/S分析法计算Hurst指数。H>0.5趋势，H<0.5均值回归。"""
    log_returns = np.diff(np.log(np.maximum(prices, 1e-10)))
    n = len(log_returns)
    if n < 8:
        return 0.5
    rs_list, sizes = [], []
    for s in [2, 3, 4, 6, 8, 12, 16]:
        w = int(n / s)
        if w < 4:
            continue
        rs_values = []
        for start in range(0, n - w + 1, w):
            chunk = log_returns[start:start + w]
            if len(chunk) < 4:
                continue
            m = np.mean(chunk)
            cumdev = np.cumsum(chunk - m)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(chunk, ddof=1)
            if S > 1e-10 and R > 0:
                rs_values.append(R / S)
        if rs_values:
            rs_list.append(np.mean(rs_values))
            sizes.append(w)
    if len(rs_list) < 2:
        return 0.5
    H = np.polyfit(np.log(sizes), np.log(rs_list), 1)[0]
    return float(np.clip(H, 0.01, 0.99))


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def _load_vol_snapshot(db: DBManager) -> dict:
    """最新 vol_monitor_snapshots。"""
    df = db.query_df(
        "SELECT * FROM vol_monitor_snapshots ORDER BY datetime DESC LIMIT 1"
    )
    if df is None or df.empty:
        return {}
    return df.iloc[0].to_dict()


def _load_model_output(db: DBManager) -> dict:
    """最新 daily_model_output（IM）。"""
    df = db.query_df(
        "SELECT * FROM daily_model_output "
        "WHERE underlying='IM' ORDER BY trade_date DESC LIMIT 1"
    )
    if df is None or df.empty:
        return {}
    return df.iloc[0].to_dict()


def _load_daily_prices(db: DBManager, n: int = 25) -> pd.DataFrame:
    """000852.SH 最近 n 天日线。"""
    df = db.query_df(
        "SELECT trade_date, close FROM index_daily "
        "WHERE ts_code='000852.SH' ORDER BY trade_date DESC LIMIT ?",
        (n,),
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["close"] = df["close"].astype(float)
    return df


def _load_positions(db: DBManager) -> list[dict]:
    """最新持仓快照。"""
    # 先取最新日期
    dt_df = db.query_df(
        "SELECT MAX(trade_date) as td FROM position_snapshots"
    )
    if dt_df is None or dt_df.empty or not dt_df.iloc[0]["td"]:
        return []
    td = dt_df.iloc[0]["td"]
    df = db.query_df(
        "SELECT * FROM position_snapshots WHERE trade_date = ?", (td,)
    )
    if df is None or df.empty:
        return []
    return df.to_dict("records")


def _load_account(db: DBManager) -> dict:
    """最新账户快照（DB fallback）。"""
    df = db.query_df(
        "SELECT * FROM account_snapshots ORDER BY trade_date DESC LIMIT 1"
    )
    if df is None or df.empty:
        return {}
    return df.iloc[0].to_dict()


def _load_tq_account_and_positions() -> tuple[dict, list[dict]]:
    """从TQ实时读取账户和持仓。失败返回空。"""
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(ROOT) / ".env")
    except ImportError:
        pass
    try:
        from data.sources.tq_client import TqClient
        import time as _time
        creds = {
            "auth_account": os.getenv("TQ_ACCOUNT", ""),
            "auth_password": os.getenv("TQ_PASSWORD", ""),
            "broker_id": os.getenv("TQ_BROKER", ""),
            "account_id": os.getenv("TQ_ACCOUNT_ID", ""),
            "broker_password": os.getenv("TQ_BROKER_PASSWORD", ""),
        }
        if not creds["auth_account"]:
            return {}, []
        client = TqClient(**creds)
        client.connect()
        api = client._api

        # 等待数据到达
        api.wait_update(deadline=_time.time() + 5)

        # 账户
        account_obj = api.get_account()
        account = {
            "balance": float(getattr(account_obj, "balance", 0) or 0),
            "margin": float(getattr(account_obj, "margin", 0) or 0),
            "float_profit": float(getattr(account_obj, "float_profit", 0) or 0),
            "available": float(getattr(account_obj, "available", 0) or 0),
            "source": "TQ实时",
        }

        # 持仓
        positions = []
        tq_pos = api.get_position()
        for sym_key in tq_pos:
            pos_obj = tq_pos[sym_key]
            vol_long = int(getattr(pos_obj, "pos_long", 0) or 0)
            vol_short = int(getattr(pos_obj, "pos_short", 0) or 0)
            if vol_long > 0:
                positions.append({
                    "symbol": sym_key.replace("CFFEX.", ""),
                    "direction": "多",
                    "volume": vol_long,
                })
            if vol_short > 0:
                positions.append({
                    "symbol": sym_key.replace("CFFEX.", ""),
                    "direction": "空",
                    "volume": vol_short,
                })

        client.disconnect()
        return account, positions
    except Exception as e:
        print(f"  [WARN] TQ实时数据读取失败: {e}")
        return {}, []


# ---------------------------------------------------------------------------
# IV 分位计算
# ---------------------------------------------------------------------------

def _calc_iv_percentile(db: DBManager, atm_iv: float) -> float:
    """用 volatility_history 计算 IV 历史分位。"""
    if atm_iv <= 0:
        return 50.0
    hist = db.query_df(
        "SELECT atm_iv FROM volatility_history "
        "WHERE atm_iv IS NOT NULL AND atm_iv > 0"
    )
    if hist is None or len(hist) < 20:
        return 50.0
    vals = hist["atm_iv"].astype(float).values
    # volatility_history 存百分比(28.85)，atm_iv 可能是小数(0.2885)
    iv_pct = atm_iv * 100 if atm_iv < 1 else atm_iv
    vals_pct = vals if vals.mean() > 1 else vals * 100
    return float(np.mean(vals_pct <= iv_pct) * 100)


# ---------------------------------------------------------------------------
# 象限判定
# ---------------------------------------------------------------------------

def _calc_direction_score(daily: pd.DataFrame) -> tuple[int, dict]:
    """计算方向得分（-4~+4）和细节。"""
    if daily.empty or len(daily) < 6:
        return 0, {}

    closes = daily["close"].values
    current = float(closes[-1])

    # 5天动量
    mom_5d = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 else 0

    # 连涨跌天数
    streak = 0
    streak_dir = ""
    for i in range(len(closes) - 1, 0, -1):
        if closes[i] > closes[i - 1]:
            if streak_dir == "" or streak_dir == "up":
                streak += 1
                streak_dir = "up"
            else:
                break
        elif closes[i] < closes[i - 1]:
            if streak_dir == "" or streak_dir == "down":
                streak += 1
                streak_dir = "down"
            else:
                break
        else:
            break

    # 20日范围位置
    recent_20 = closes[-20:] if len(closes) >= 20 else closes
    high_20 = float(np.max(recent_20))
    low_20 = float(np.min(recent_20))
    range_pos = (current - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5

    # 综合得分
    score = 0
    if mom_5d > 0.01:
        score += 1
    if mom_5d > 0.03:
        score += 1
    if streak >= 2 and streak_dir == "up":
        score += 1
    if range_pos > 0.5:
        score += 1

    if mom_5d < -0.01:
        score -= 1
    if mom_5d < -0.03:
        score -= 1
    if streak >= 2 and streak_dir == "down":
        score -= 1
    if range_pos < 0.3:
        score -= 1

    detail = {
        "mom_5d": mom_5d,
        "streak": streak,
        "streak_dir": streak_dir,
        "range_pos": range_pos,
        "current": current,
        "high_20": high_20,
        "low_20": low_20,
    }
    return score, detail


def determine_quadrant(
    iv_pctile: float, direction_score: int, vrp: float,
    term_structure: str = "", iv_change_pp: float = 0,
    hurst: float = 0.5,
) -> tuple[str, str]:
    """返回 (象限名, 子状态描述)。综合IV/VRP/Hurst/期限结构/IV变动判定。

    高IV象限按Hurst细分：
      A1: 高IV+VRP>0+震荡 — 卖方最佳甜点区
      A2: 高IV+VRP>0+趋势 — 顺势卖方可以，逆势危险
      B1: 高IV+VRP<0+震荡 — 等待VRP转正，可小仓卖方
      B2: 高IV+VRP<0+趋势 — 最危险，禁止卖方
    """
    iv_high = iv_pctile >= 70
    iv_low = iv_pctile <= 30
    bullish = direction_score >= 2
    bearish = direction_score <= -2
    is_inverted = "倒挂" in term_structure
    vrp_negative = vrp < 0
    iv_surging = iv_change_pp > 3
    trending = hurst > 0.55

    if iv_high:
        # B象限特征：VRP<0 或 期限倒挂 或 IV急升
        b_signal = vrp_negative or is_inverted or iv_surging

        if b_signal and vrp_negative:
            if trending:
                return "B2", "高IV+VRP<0+趋势（最危险，禁止卖方，纯买方或对冲）"
            else:
                return "B1", "高IV+VRP<0+震荡（等VRP转正，可小仓卖方）"
        elif b_signal:
            # 期限倒挂或IV急升但VRP>0
            if trending:
                return "A2", "高IV+VRP>0+趋势+倒挂/IV急升（顺势卖方可以，注意方向）"
            else:
                return "A1", "高IV+VRP>0+震荡+倒挂/IV急升（卖方可以但谨慎）"
        else:
            # 纯A象限：IV高+VRP>0+正常期限结构
            if trending:
                return "A2", "高IV+VRP>0+趋势（顺势卖方可以，逆势危险）"
            else:
                return "A1", "高IV+VRP>0+震荡（卖方最佳甜点区）"
    elif iv_low:
        if bullish:
            return "C", "低IV+偏多（买Call / IM+Put吃贴水）"
        elif bearish:
            return "D", "低IV+偏空（卖Call Spread / 减少暴露）"
        else:
            return "C/D边界", "低IV+方向不明确（轻仓观望）"
    else:
        if bullish:
            return "过渡区→A", f"IV中位(P{iv_pctile:.0f})+偏多（等IV升高再卖方）"
        elif bearish:
            return "过渡区→D", f"IV中位(P{iv_pctile:.0f})+偏空（警惕向B跃迁）"
        else:
            return "中性区", f"IV中位(P{iv_pctile:.0f})+方向不明（等待信号）"


# ---------------------------------------------------------------------------
# 持仓评估
# ---------------------------------------------------------------------------

_MO_RE = re.compile(r'MO(\d{4})-([CP])-(\d+)')


def _parse_positions(positions: list[dict]) -> dict:
    """解析持仓，返回策略类型标签。"""
    has_short_put = False
    has_short_call = False
    has_long_put = False
    has_long_call = False
    has_long_futures = False
    has_short_futures = False
    mo_positions = []

    for pos in positions:
        symbol = str(pos.get("symbol", pos.get("ts_code", "")))
        direction = str(pos.get("direction", ""))
        volume = int(pos.get("volume", pos.get("vol", 0)) or 0)
        if volume <= 0:
            continue

        m = _MO_RE.search(symbol)
        if m:
            cp = m.group(2)
            dir_lower = direction.lower()
            if cp == "P" and dir_lower in ("sell", "short", "空"):
                has_short_put = True
            elif cp == "P" and dir_lower in ("buy", "long", "多"):
                has_long_put = True
            elif cp == "C" and dir_lower in ("sell", "short", "空"):
                has_short_call = True
            elif cp == "C" and dir_lower in ("buy", "long", "多"):
                has_long_call = True
            mo_positions.append({
                "symbol": symbol, "cp": cp, "direction": direction,
                "volume": volume,
            })
        elif "IM" in symbol.upper():
            dir_lower = direction.lower()
            if dir_lower in ("buy", "long", "多"):
                has_long_futures = True
            elif dir_lower in ("sell", "short", "空"):
                has_short_futures = True

    return {
        "has_short_put": has_short_put,
        "has_short_call": has_short_call,
        "has_long_put": has_long_put,
        "has_long_call": has_long_call,
        "has_long_futures": has_long_futures,
        "has_short_futures": has_short_futures,
        "is_strangle": has_short_put and has_short_call,
        "is_selling_vol": has_short_put or has_short_call,
        "mo_positions": mo_positions,
    }


def evaluate_position_match(quadrant: str, pos_info: dict, vrp: float = 0) -> tuple[str, str]:
    """评估持仓与象限的匹配度。返回 (status_icon, description)。"""
    is_sv = pos_info["is_selling_vol"]
    is_str = pos_info["is_strangle"]
    has_sp = pos_info["has_short_put"]
    has_sc = pos_info["has_short_call"]
    has_lp = pos_info["has_long_put"]
    has_lf = pos_info["has_long_futures"]

    if not any(pos_info[k] for k in [
        "has_short_put", "has_short_call", "has_long_put",
        "has_long_call", "has_long_futures", "has_short_futures"
    ]):
        return "⚪", "无持仓"

    q = quadrant.split("/")[0].split("→")[0].strip()  # 提取主象限

    if q == "A1":
        # 震荡+VRP>0 — 卖方最佳
        if is_str:
            return "✅", "最佳匹配（震荡市卖方Strangle赚Theta）"
        if is_sv:
            return "✅", "匹配（震荡市卖方有利）"
        if not is_sv:
            return "⚠️", "不匹配（A1是卖方甜点区，应该做卖方）"

    elif q == "A2":
        # 趋势+VRP>0 — 顺势卖方可以
        if is_sv:
            return "🟡", "注意：趋势市卖方需确保持仓方向和趋势一致，逆势端加保护"
        if not is_sv:
            return "🟡", "中性（趋势市可顺势卖方，但不如A1安全）"

    elif q == "A" or q == "A(VRP警告)":
        if vrp < 0 and is_sv:
            return "⚠️", "注意: 卖方持仓但VRP<0，卖的是便宜保险，不要加仓"
        if is_str:
            return "✅", "匹配（卖方Strangle在高IV中赚钱）"
        if not is_sv:
            return "⚠️", "不匹配（应该做卖方）"

    elif q == "B1":
        # 震荡+VRP<0 — 等待
        if is_sv:
            return "🔴", "危险！B1 VRP<0卖方在亏钱，但震荡期风险可控，减仓观望"
        if has_lp:
            return "✅", "匹配（买方对冲）"
        return "🟡", "中性（等VRP转正再开卖方）"

    elif q == "B2":
        # 趋势+VRP<0 — 最危险
        if is_sv:
            return "🔴", "极度危险！B2趋势市+VRP<0，禁止卖方，建议立即减仓或买保护"
        if has_lp:
            return "✅", "匹配（买方在单边市赚Gamma）"
        return "⚠️", "B2最危险象限（考虑买Put对冲或离场）"

    elif q == "C":
        if is_sv:
            return "⚠️", "不匹配（IV太低，Theta不值得风险）"
        if has_lf and has_lp:
            return "✅", "匹配（IM+Put贴水策略）"
        if has_lp or pos_info["has_long_call"]:
            return "✅", "匹配（低IV买方策略）"
        return "🟡", "中性（考虑IM+Put或买远月Call）"

    elif q == "D":
        if has_sp:
            return "🔴", "危险！阴跌中卖Put极其危险"
        if has_sc and not has_sp:
            return "✅", "匹配（顺势卖Call Spread）"
        return "🟡", "中性（考虑减少方向暴露）"

    # 过渡区/边界
    if is_sv:
        return "🟡", "过渡区持仓需监控，注意象限切换"
    return "⚪", "过渡区，等待方向明确"


# ---------------------------------------------------------------------------
# 象限切换检测
# ---------------------------------------------------------------------------

_SWITCH_ALERTS = {
    ("A", "B1"): ("🚨", "暴跌！立即止损卖Put端"),
    ("A", "B2"): ("🚨", "暴跌！检查卖Put端风险"),
    ("D", "B1"): ("🚨", "雪球敲入跃迁！远月Put应该在赚钱"),
    ("D", "B2"): ("🚨", "快速下跌+IV飙升，检查Put端"),
    ("B2", "B1"): ("⚠️", "波动重新加剧，暂停卖方操作"),
    ("B1", "B2"): ("📢", "恐慌见顶迹象，准备建立卖方"),
    ("B2", "A"): ("📢", "恐慌消退，可以开始卖方仓位"),
    ("B1", "A"): ("📢", "恐慌消退+反弹，可以开始卖方仓位"),
    ("A", "C"): ("📢", "IV回落到低位，考虑从卖方切换到买方"),
    ("C", "D"): ("📋", "方向转弱，减少做多暴露"),
    ("D", "C"): ("📋", "方向好转，可考虑做多"),
}


def check_switch(prev_q: str, curr_q: str) -> tuple[str, str] | None:
    """检查象限切换，返回 (icon, message) 或 None。"""
    if prev_q == curr_q:
        return None
    # 精确匹配
    key = (prev_q, curr_q)
    if key in _SWITCH_ALERTS:
        return _SWITCH_ALERTS[key]
    # 模糊匹配（如 A/B边界 → B1）
    prev_base = prev_q.split("/")[0].split("→")[0].strip()
    curr_base = curr_q.split("/")[0].split("→")[0].strip()
    if prev_base == curr_base:
        return None
    key2 = (prev_base, curr_base)
    if key2 in _SWITCH_ALERTS:
        return _SWITCH_ALERTS[key2]
    return ("📋", f"象限切换: {prev_q} → {curr_q}")


# ---------------------------------------------------------------------------
# 面板输出
# ---------------------------------------------------------------------------

def print_panel(
    quadrant: str,
    quad_desc: str,
    iv_pctile: float,
    atm_iv: float,
    vrp: float,
    direction_score: int,
    dir_detail: dict,
    pos_icon: str,
    pos_desc: str,
    pos_info: dict,
    discount: dict,
    account: dict,
    switch_alert: tuple[str, str] | None,
    term_structure: str = "",
    iv_change_pp: float = 0,
    db: DBManager = None,
    hurst: float = 0.5,
    hurst_pctile: float = None,
):
    """输出完整面板。"""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'═' * W}")
    print(f" 策略象限监控 | {now_str}")
    print(f"{'═' * W}")

    # 象限切换 alert
    if switch_alert:
        icon, msg = switch_alert
        print(f"\n {icon} {icon} {icon}  {msg}")
        print()

    # 当前象限
    print(f"\n{'【当前象限】'}")
    print(f"  象限: {quadrant}  —  {quad_desc}")

    # IV 状态
    print(f"\n{'【波动率环境】'}")
    iv_tag = "高位" if iv_pctile >= 70 else ("低位" if iv_pctile <= 30 else "中位")
    iv_chg_s = f"  日变: {iv_change_pp:+.1f}pp" if iv_change_pp != 0 else ""
    iv_warn = " ⚠" if abs(iv_change_pp) > 3 else ""
    print(f"  ATM IV        : {atm_iv*100:.1f}%  (P{iv_pctile:.0f}, {iv_tag}){iv_chg_s}{iv_warn}")
    vrp_tag = "IV>RV(卖方有利)" if vrp > 0 else "RV>IV(卖方不利)"
    print(f"  VRP           : {vrp*100:+.1f}%  ({vrp_tag})")
    if term_structure:
        ts_warn = " ⚠" if "倒挂" in term_structure else ""
        print(f"  期限结构      : {term_structure}{ts_warn}")
    h_tag = "强趋势期" if hurst > 0.6 else ("趋势期" if hurst > 0.55
             else "震荡期" if hurst < 0.45 else "中性")
    hp_s = f"P{hurst_pctile:.0f}, " if hurst_pctile is not None else ""
    print(f"  Hurst(60d)    : {hurst:.2f} ({hp_s}{h_tag})")

    # 方向状态
    mom = dir_detail.get("mom_5d", 0)
    streak = dir_detail.get("streak", 0)
    streak_dir = dir_detail.get("streak_dir", "")
    rp = dir_detail.get("range_pos", 0.5)
    dir_label = ("偏多" if direction_score >= 2
                 else "偏空" if direction_score <= -2
                 else "中性")
    print(f"\n{'【方向判断】'} 得分={direction_score:+d} ({dir_label})")
    print(f"  5日动量       : {mom*100:+.2f}%")
    streak_s = f"{streak}天{'涨' if streak_dir == 'up' else '跌'}" if streak > 0 else "无连续"
    print(f"  连涨跌        : {streak_s}")
    print(f"  20日范围位置  : {rp*100:.0f}%"
          f"  ({dir_detail.get('low_20', 0):.0f} ~ {dir_detail.get('high_20', 0):.0f})")

    # 贴水（val是年化百分比）
    if discount:
        print(f"\n{'【贴水】'}")
        _disc_contracts = ["IML1", "IML2", "IML3"]
        # DTE 近似：IML1~45天, IML2~90天, IML3~180天
        _dte_approx = [45, 90, 180]
        for i, (label, val) in enumerate(discount.items()):
            pct_s = ""
            if db and i < len(_disc_contracts):
                try:
                    from strategies.discount_capture.signal import DiscountSignal
                    ds = DiscountSignal(db)
                    # 年化百分比→日贴水率绝对值（用近似DTE反推）
                    dte = _dte_approx[i] if i < len(_dte_approx) else 90
                    daily_rate_abs = abs(val) / 100 * dte / 365
                    pct = ds.get_discount_percentile(
                        daily_rate_abs, contract_type=_disc_contracts[i])
                    tag = ("历史极深" if pct >= 90 else "偏深" if pct >= 70
                           else "偏浅" if pct <= 10 else "")
                    pct_s = f"  P{pct:.0f}" + (f"({tag})" if tag else "")
                except Exception:
                    pass
            print(f"  {label:16s}: {val:+.1f}%{pct_s}")

    # 持仓匹配
    print(f"\n{'【持仓评估】'}")
    print(f"  {pos_icon} {pos_desc}")
    mo_pos = pos_info.get("mo_positions", [])
    if mo_pos:
        for p in mo_pos:
            d_cn = "空" if p["direction"].lower() in ("sell", "short", "空") else "多"
            print(f"    {p['symbol']} {d_cn} {p['volume']}手")

    # 账户
    if account:
        equity = float(account.get("balance", 0) or account.get("equity", 0) or 0)
        margin = float(account.get("margin", 0) or 0)
        margin_pct = margin / equity * 100 if equity > 0 else 0
        src = account.get("source", "DB快照")
        print(f"\n{'【账户】'} ({src})")
        print(f"  权益: {equity:,.0f}  保证金: {margin:,.0f} ({margin_pct:.1f}%)")

    print(f"\n{'═' * W}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_once(db: DBManager, prev_quadrant: str = "",
             tq_account: dict = None, tq_positions: list = None) -> str:
    """刷新一次面板。返回当前象限名称。"""
    # 加载数据
    snap = _load_vol_snapshot(db)
    model = _load_model_output(db)
    daily = _load_daily_prices(db, 25)

    # 持仓和账户：优先TQ实时，fallback DB快照
    if tq_positions is not None:
        positions_raw = tq_positions
    else:
        positions_raw = _load_positions(db)
    if tq_account is not None:
        account = tq_account
    else:
        account = _load_account(db)
    positions = positions_raw

    # IV 分位
    atm_iv = float(snap.get("atm_iv", 0) or model.get("atm_iv", 0) or 0)
    iv_pctile_snap = float(snap.get("iv_percentile", 0) or 0)
    if iv_pctile_snap > 0:
        iv_pctile = iv_pctile_snap
    else:
        iv_pctile = _calc_iv_percentile(db, atm_iv)

    # VRP
    vrp = float(snap.get("vrp", 0) or model.get("vrp", 0) or 0)

    # 期限结构
    term_structure = str(snap.get("term_structure_shape", "") or "")

    # IV日变动（从最近2条vol_monitor_snapshots或daily_model_output）
    iv_change_pp = 0.0
    prev_iv_df = db.query_df(
        "SELECT atm_iv FROM vol_monitor_snapshots ORDER BY datetime DESC LIMIT 2"
    )
    if prev_iv_df is not None and len(prev_iv_df) >= 2:
        iv_now = float(prev_iv_df.iloc[0].get("atm_iv", 0) or 0)
        iv_prev = float(prev_iv_df.iloc[1].get("atm_iv", 0) or 0)
        if iv_now > 0 and iv_prev > 0:
            iv_change_pp = (iv_now - iv_prev) * 100

    # 方向
    direction_score, dir_detail = _calc_direction_score(daily)

    # Hurst指数（60天窗口，日频更新）
    hurst_val = 0.5
    hurst_pctile = None
    try:
        hdf = db.query_df(
            "SELECT close FROM index_daily WHERE ts_code='000852.SH' "
            "ORDER BY trade_date DESC LIMIT 300")
        if hdf is not None and len(hdf) >= 60:
            _closes = hdf["close"].astype(float).values[::-1]
            hurst_val = _calc_hurst_rs(_closes[-60:])
            if len(_closes) >= 120:
                _hist = [_calc_hurst_rs(_closes[i-60:i])
                         for i in range(60, len(_closes))]
                hurst_pctile = float(
                    sum(1 for x in _hist if x <= hurst_val)
                    / len(_hist) * 100)
    except Exception:
        pass

    # 象限（综合VRP+期限结构+IV变动+Hurst）
    quadrant, quad_desc = determine_quadrant(
        iv_pctile, direction_score, vrp,
        term_structure=term_structure, iv_change_pp=iv_change_pp,
        hurst=hurst_val)

    # 持仓
    pos_info = _parse_positions(positions)
    pos_icon, pos_desc = evaluate_position_match(quadrant, pos_info, vrp=vrp)

    # 贴水：优先vol_monitor_snapshots实时年化%，fallback daily_model_output日贴水率
    discount = {}
    for label, snap_col, model_col in [
        ("次月(IML1)", "discount_iml1", "discount_rate_iml1"),
        ("当季(IML2)", "discount_iml2", "discount_rate_iml2"),
        ("隔季(IML3)", None,            "discount_rate_iml3"),
    ]:
        # 优先实时（snap已是年化百分比）
        val = snap.get(snap_col) if snap_col else None
        if val is not None and val != "" and not (isinstance(val, float) and np.isnan(val)):
            discount[label] = float(val)
        else:
            # fallback: daily_model_output 存的是价格偏离率小数（非年化）
            # 需要除以DTE再乘365转年化
            val2 = model.get(model_col)
            if val2 is not None and val2 != "":
                _dte_map = {"discount_rate_iml1": 45,
                            "discount_rate_iml2": 90,
                            "discount_rate_iml3": 180}
                dte = _dte_map.get(model_col, 90)
                discount[label] = float(val2) / dte * 365 * 100

    # 象限切换
    switch_alert = None
    if prev_quadrant:
        switch_alert = check_switch(prev_quadrant, quadrant)

    print_panel(
        quadrant, quad_desc, iv_pctile, atm_iv, vrp,
        direction_score, dir_detail,
        pos_icon, pos_desc, pos_info,
        discount, account, switch_alert,
        term_structure=term_structure, iv_change_pp=iv_change_pp,
        db=db, hurst=hurst_val, hurst_pctile=hurst_pctile,
    )

    return quadrant


def main():
    db = get_db()

    if "--once" in sys.argv:
        tq_acc, tq_pos = _load_tq_account_and_positions()
        run_once(db, tq_account=tq_acc or None, tq_positions=tq_pos or None)
        return

    print("  策略象限监控启动 | 每5分钟刷新（TQ实时账户+持仓）")
    prev_q = ""
    while True:
        try:
            tq_acc, tq_pos = _load_tq_account_and_positions()
            prev_q = run_once(db, prev_q,
                              tq_account=tq_acc or None,
                              tq_positions=tq_pos or None)
        except Exception as e:
            print(f"\n  [ERROR] {e}")
        try:
            now = datetime.now()
            wait = (5 - now.minute % 5) * 60 - now.second
            if wait <= 0:
                wait = 300
            time.sleep(wait)
        except KeyboardInterrupt:
            print("\n  退出")
            break


if __name__ == "__main__":
    main()
