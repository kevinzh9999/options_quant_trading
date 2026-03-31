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
from data.storage.db_manager import DBManager

W = 80  # 面板宽度


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
    """最新账户快照。"""
    df = db.query_df(
        "SELECT * FROM account_snapshots ORDER BY trade_date DESC LIMIT 1"
    )
    if df is None or df.empty:
        return {}
    return df.iloc[0].to_dict()


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
) -> tuple[str, str]:
    """返回 (象限名, 子状态描述)。综合IV/VRP/期限结构/IV变动判定。"""
    iv_high = iv_pctile >= 70
    iv_low = iv_pctile <= 30
    bullish = direction_score >= 2
    bearish = direction_score <= -2
    is_inverted = "倒挂" in term_structure
    vrp_negative = vrp < 0
    iv_surging = iv_change_pp > 3

    if iv_high:
        # 检查B象限特征：VRP<0 或 期限倒挂 或 IV急升
        b_signal = vrp_negative or is_inverted or iv_surging

        if b_signal:
            if vrp_negative:
                if bearish:
                    return "B1", "高IV+偏空+RV>IV（恐慌进行中，禁止卖方）"
                elif bullish:
                    return "A(VRP警告)", "高IV+偏多但VRP<0（卖方需保护，不加仓）"
                else:
                    return "B1", "高IV+VRP<0（RV>IV，禁止卖方）"
            else:
                # 期限倒挂或IV急升但VRP>0
                if bearish:
                    return "B2", "高IV+偏空+IV>RV（恐慌见顶，可卖Vega）"
                elif bullish:
                    return "A", "高IV+偏多（卖Strangle，注意期限结构）"
                else:
                    return "B2", "高IV+方向中性+倒挂/IV急升（谨慎卖方）"
        else:
            # 纯A象限：IV高+VRP>0+正常期限结构
            if bullish:
                return "A", "高IV+偏多（卖Strangle最佳窗口）"
            elif bearish:
                return "B2", "高IV+偏空+IV>RV（恐慌见顶，可卖Vega）"
            else:
                return "A/B边界", "高IV+方向不明确（观望或小仓位卖方）"
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

    if q == "A" or q == "A(VRP警告)":
        if vrp < 0 and is_sv:
            return "⚠️", "注意: 卖方持仓但VRP<0，卖的是便宜保险，不要加仓"
        if is_str:
            return "✅", "匹配（卖方Strangle在高IV+偏多中赚钱）"
        if has_sp and not has_sc:
            return "🟡", "部分匹配（可考虑加Call端构成Strangle）"
        if has_sc and not has_sp:
            return "🟡", "部分匹配（可考虑加Put端构成Strangle）"
        if not is_sv:
            return "⚠️", "不匹配（应该做卖方）"

    elif q == "B1":
        if is_sv:
            return "🔴", "危险！B1禁止卖方，VRP<0意味着RV>IV，卖方在亏钱"
        if has_lp:
            return "✅", "匹配（买方在恐慌中赚Gamma）"
        return "🟡", "中性（考虑买Put对冲）"

    elif q == "B2":
        if is_str:
            return "✅", "匹配（恐慌见顶卖Vega）"
        if is_sv:
            return "🟡", "部分匹配（检查VRP是否已转正）"
        return "⚠️", "偏空（可开始建立卖方仓位）"

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

    # 贴水
    if discount:
        print(f"\n{'【贴水】'}")
        # 计算分位数（从daily_model_output历史）
        cols = ["discount_rate_iml1", "discount_rate_iml2", "discount_rate_iml3"]
        for i, (label, val) in enumerate(discount.items()):
            pct_s = ""
            if i < len(cols):
                try:
                    hist = db.query_df(
                        f"SELECT {cols[i]} FROM daily_model_output "
                        f"WHERE {cols[i]} IS NOT NULL")
                    if hist is not None and len(hist) >= 5:
                        vals = hist[cols[i]].astype(float).dropna().values
                        rate = val / 100
                        pct = float(np.mean(vals <= rate) * 100)
                        tag = ("历史极深" if pct <= 10 else "偏深" if pct <= 30
                               else "偏浅" if pct >= 70 else "")
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
        print(f"\n{'【账户】'}")
        print(f"  权益: {equity:,.0f}  保证金: {margin:,.0f} ({margin_pct:.1f}%)")

    print(f"\n{'═' * W}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_once(db: DBManager, prev_quadrant: str = "") -> str:
    """刷新一次面板。返回当前象限名称。"""
    # 加载数据
    snap = _load_vol_snapshot(db)
    model = _load_model_output(db)
    daily = _load_daily_prices(db, 25)
    positions = _load_positions(db)
    account = _load_account(db)

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

    # 象限（综合VRP+期限结构+IV变动）
    quadrant, quad_desc = determine_quadrant(
        iv_pctile, direction_score, vrp,
        term_structure=term_structure, iv_change_pp=iv_change_pp)

    # 持仓
    pos_info = _parse_positions(positions)
    pos_icon, pos_desc = evaluate_position_match(quadrant, pos_info, vrp=vrp)

    # 贴水
    discount = {}
    for label, col in [("次月(IML1)", "discount_rate_iml1"),
                        ("当季(IML2)", "discount_rate_iml2"),
                        ("隔季(IML3)", "discount_rate_iml3")]:
        val = model.get(col)
        if val is not None and val != "":
            discount[label] = float(val) * 100

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
        db=db,
    )

    return quadrant


def main():
    db = DBManager(ConfigLoader().get_db_path())

    if "--once" in sys.argv:
        run_once(db)
        return

    print("  策略象限监控启动 | 每5分钟刷新")
    prev_q = ""
    while True:
        try:
            prev_q = run_once(db, prev_q)
        except Exception as e:
            print(f"\n  [ERROR] {e}")
        try:
            # 等待到下一个整5分钟
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
