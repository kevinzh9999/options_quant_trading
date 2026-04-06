"""
vol_monitor.py
--------------
期权波动率实时监控面板。盘中每5分钟刷新，覆盖：
  - ATM IV / RV / VRP / GARCH
  - IV期限结构
  - IV Skew + 25D RR/BF
  - 持仓Greeks实时更新
  - 贴水监控
  - 综合信号汇总

用法：
    python scripts/vol_monitor.py              # 盘中实时模式
    python scripts/vol_monitor.py --snapshot   # 只刷新一次
"""

from __future__ import annotations

import os
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(ROOT) / ".env")
except ImportError:
    pass

from config.config_loader import ConfigLoader
from data.storage.db_manager import DBManager, get_db
from models.pricing.forward_price import calc_implied_forward
from models.pricing.implied_vol import calc_implied_vol
from models.pricing.greeks import calc_all_greeks
from utils.cffex_calendar import active_im_months, get_im_futures_prices

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RISK_FREE = 0.02
MO_MULT = 100      # MO合约乘数
IM_MULT = 200       # IM合约乘数
_MO_RE = re.compile(r'^MO(\d{4})-([CP])-(\d+)\.CFX$')
W = 82              # 面板宽度


def _parse_mo(ts_code: str):
    m = _MO_RE.match(str(ts_code))
    return (m.group(1), m.group(2), float(m.group(3))) if m else None


def _dte(expire_date: str, today: str) -> int:
    """剩余天数。"""
    try:
        ed = datetime.strptime(expire_date, "%Y%m%d")
        td = datetime.strptime(today, "%Y%m%d")
        return max((ed - td).days, 1)
    except Exception:
        return 30


def _T(expire_date: str, today: str) -> float:
    return _dte(expire_date, today) / 365.0


def _find_near_month(months: list, expire_map: dict, today: str, min_dte: int = 7) -> str:
    """找到 DTE >= min_dte 的最近到期月份。"""
    for m in months:
        ed = expire_map.get(m, "")
        if ed and _dte(ed, today) >= min_dte:
            return m
    # 如果都不够，用chain中最近的
    for em, ed in sorted(expire_map.items()):
        if _dte(ed, today) >= min_dte:
            return em
    return months[0] if months else ""


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _open_db():
    try:
        return get_db()
    except Exception as e:
        print(f"  [警告] DB: {e}")
        return None


def _calc_discount_percentile(db, daily_rate: float, contract_idx: int = 0) -> float | None:
    """计算日贴水率在 futures_daily 完整历史中的百分位。

    daily_rate: 日贴水率小数（如 -0.016 = -1.6%），(futures-spot)/spot
    contract_idx: 0=IML1(次月), 1=IML2(当季), 2=IML3(隔季)
    返回 0-100，越高表示当前贴水越深。
    """
    try:
        from strategies.discount_capture.signal import DiscountSignal
        ds = DiscountSignal(db)
        contract_type = ["IML1", "IML2", "IML3"][contract_idx]
        # get_discount_percentile 接受贴水幅度绝对值
        pct = ds.get_discount_percentile(
            abs(daily_rate), contract_type=contract_type)
        return pct
    except Exception:
        return None


def _discount_pct_label(pct: float | None) -> str:
    """贴水分位标签。"""
    if pct is None:
        return ""
    if pct <= 10:
        return f"P{pct:.0f}（历史极深）"
    elif pct <= 30:
        return f"P{pct:.0f}（偏深）"
    elif pct >= 90:
        return f"P{pct:.0f}（历史极浅）"
    elif pct >= 70:
        return f"P{pct:.0f}（偏浅）"
    return f"P{pct:.0f}"


def _build_expire_map(db, today: str = "") -> dict:
    """{'2604': '20260417', ...}  只返回尚未到期的合约月份。"""
    if not today:
        today = date.today().strftime("%Y%m%d")
    df = db.query_df(
        "SELECT DISTINCT ts_code, expire_date "
        "FROM options_contracts WHERE ts_code LIKE 'MO%' "
        "AND expire_date >= ?",
        params=(today,),
    )
    if df is None or df.empty:
        return {}
    result = {}
    for _, row in df.iterrows():
        p = _parse_mo(str(row["ts_code"]))
        if p and row["expire_date"]:
            result[p[0]] = str(row["expire_date"])
    return result


def _get_mo_chain(db, trade_date: str, expire_map: dict) -> pd.DataFrame:
    df = db.query_df(
        f"SELECT ts_code, close, settle, volume, oi "
        f"FROM options_daily "
        f"WHERE ts_code LIKE 'MO%' AND trade_date='{trade_date}' AND close > 0"
    )
    if df is None or df.empty:
        return pd.DataFrame()
    records = []
    for _, row in df.iterrows():
        p = _parse_mo(str(row["ts_code"]))
        if p is None:
            continue
        em, cp, strike = p
        ed = expire_map.get(em, "")
        if not ed:
            continue
        records.append({
            "ts_code": row["ts_code"], "expire_month": em, "expire_date": ed,
            "call_put": cp, "exercise_price": strike,
            "close": float(row["close"]),
            "volume": float(row["volume"]) if row["volume"] else 0.0,
            "oi": float(row["oi"]) if row["oi"] else 0.0,
        })
    return pd.DataFrame(records) if records else pd.DataFrame()


def _load_positions(db) -> list:
    """从 position_snapshots 加载最新 MO 持仓。"""
    try:
        latest = db.query_df("SELECT MAX(trade_date) as dt FROM position_snapshots")
        if latest is None or latest.empty or latest["dt"].iloc[0] is None:
            print("  [持仓] position_snapshots 表为空")
            return []
        td = str(latest["dt"].iloc[0])
        rows = db.query_df(
            "SELECT symbol, direction, volume, open_price_avg "
            "FROM position_snapshots WHERE trade_date = ?",
            params=(td,),
        )
        if rows is None or rows.empty:
            print(f"  [持仓] {td} 无持仓记录")
            return []
    except Exception as e:
        print(f"  [持仓] 读取 position_snapshots 失败: {e}")
        return []

    tq_re = re.compile(r'^CFFEX\.(MO\d{4}-[CP]-\d+)$')
    positions = []
    for _, row in rows.iterrows():
        sym = str(row["symbol"])
        m = tq_re.match(sym)
        if not m:
            continue
        ts_code = m.group(1) + ".CFX"
        vol = int(row["volume"])
        if str(row["direction"]).upper() == "SHORT":
            vol = -abs(vol)
        else:
            vol = abs(vol)
        positions.append({
            "ts_code": ts_code, "volume": vol,
            "open_price": float(row.get("open_price_avg", 0) or 0),
        })
    return positions


def _get_yesterday_model(db, today: str) -> dict:
    """从 daily_model_output 读取昨日数据。"""
    df = db.query_df(
        "SELECT * FROM daily_model_output "
        "WHERE trade_date < ? AND underlying = 'IM' "
        "ORDER BY trade_date DESC LIMIT 1",
        (today,),
    )
    if df is None or df.empty:
        return {}
    return df.iloc[0].to_dict()


def _get_iv_history(db, days: int = 0) -> pd.DataFrame:
    """ATM IV 和 VRP 历史序列。优先从 volatility_history 读取(886天)，
    fallback 到 daily_model_output。days=0 表示全量。"""
    # 优先 volatility_history（完整回算序列）
    limit_clause = f"LIMIT {days}" if days > 0 else ""
    df = db.query_df(
        f"SELECT trade_date, atm_iv, vrp_rv as vrp FROM volatility_history "
        f"WHERE atm_iv IS NOT NULL AND atm_iv > 0 "
        f"ORDER BY trade_date DESC {limit_clause}"
    )
    if df is not None and len(df) > 10:
        return df

    # fallback: daily_model_output
    df = db.query_df(
        f"SELECT trade_date, atm_iv, vrp FROM daily_model_output "
        f"WHERE underlying = 'IM' ORDER BY trade_date DESC {limit_clause}"
    )
    return df if df is not None else pd.DataFrame()


# ---------------------------------------------------------------------------
# Volatility calculations
# ---------------------------------------------------------------------------

def calc_atm_iv_for_month(
    chain: pd.DataFrame, expire_month: str, forward: float,
    expire_date: str, today: str,
) -> float:
    """计算指定到期月的 ATM IV。"""
    sub = chain[chain["expire_month"] == expire_month].copy()
    if sub.empty:
        return 0.0
    # 找最接近 forward 的行权价
    sub["dist"] = (sub["exercise_price"] - forward).abs()
    atm_strike = float(sub.loc[sub["dist"].idxmin(), "exercise_price"])
    T_val = _T(expire_date, today)

    ivs = []
    for cp in ["C", "P"]:
        row = sub[(sub["exercise_price"] == atm_strike) & (sub["call_put"] == cp)]
        if row.empty:
            continue
        price = float(row.iloc[0]["close"])
        iv = calc_implied_vol(price, forward, atm_strike, T_val, RISK_FREE, cp)
        if iv and 0 < iv < 5:
            ivs.append(iv)
    return float(np.mean(ivs)) if ivs else 0.0


def calc_market_atm_iv(
    chain: pd.DataFrame, expire_month: str, futures_price: float,
    expire_date: str, today: str,
) -> float:
    """计算 Market ATM IV：直接用期货价格（非隐含Forward）作为标的。

    与 calc_atm_iv_for_month() 的区别：
    - 本函数用 IM主力期货收盘价作为 S，找最近行权价
    - calc_atm_iv_for_month 用 PCP 隐含 Forward Price
    - 市场IV 用于 VRP/情绪指标（避免 Forward-based 循环论证）
    - 结构IV（Forward-based）用于 Greeks 定价
    """
    sub = chain[chain["expire_month"] == expire_month].copy()
    if sub.empty or futures_price <= 0:
        return 0.0
    # 找最接近期货价格的行权价
    sub["dist"] = (sub["exercise_price"] - futures_price).abs()
    atm_strike = float(sub.loc[sub["dist"].idxmin(), "exercise_price"])
    T_val = _T(expire_date, today)

    ivs = []
    for cp in ["C", "P"]:
        row = sub[(sub["exercise_price"] == atm_strike) & (sub["call_put"] == cp)]
        if row.empty:
            continue
        price = float(row.iloc[0]["close"])
        # 用期货价格作为 S（不是Forward）
        iv = calc_implied_vol(price, futures_price, atm_strike, T_val, RISK_FREE, cp)
        if iv and 0 < iv < 5:
            ivs.append(iv)
    return float(np.mean(ivs)) if ivs else 0.0


def calc_forwards_by_expiry(
    chain: pd.DataFrame, expire_map: dict, futures_prices: dict,
    today: str,
) -> dict:
    """每个到期月的隐含Forward。"""
    # 用最近月的期货价作为无直接合约月份的 fallback
    fallback_price = 0.0
    if futures_prices:
        fallback_price = next(iter(
            v for _, v in sorted(futures_prices.items()) if v > 0
        ), 0.0)

    forwards = {}
    for em in chain["expire_month"].unique():
        ed = expire_map.get(em, "")
        if not ed:
            continue
        T_val = _T(ed, today)
        sub = chain[chain["expire_month"] == em]
        fut_price = futures_prices.get(em, fallback_price)
        if fut_price <= 0:
            continue
        fwd, n = calc_implied_forward(sub, T_val, RISK_FREE, fut_price)
        forwards[em] = fwd
    return forwards


def calc_skew_table(
    chain: pd.DataFrame, expire_month: str, forward: float,
    expire_date: str, today: str,
) -> pd.DataFrame:
    """计算指定月份的 IV Skew 表。"""
    sub = chain[chain["expire_month"] == expire_month].copy()
    if sub.empty:
        return pd.DataFrame()

    T_val = _T(expire_date, today)
    strikes = sorted(sub["exercise_price"].unique())

    # ATM ±5 档
    atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - forward))
    lo = max(0, atm_idx - 5)
    hi = min(len(strikes), atm_idx + 6)
    selected = strikes[lo:hi]

    rows = []
    for k in selected:
        for cp in ["C", "P"]:
            r = sub[(sub["exercise_price"] == k) & (sub["call_put"] == cp)]
            if r.empty:
                continue
            price = float(r.iloc[0]["close"])
            vol = float(r.iloc[0].get("volume", 0))
            iv = calc_implied_vol(price, forward, k, T_val, RISK_FREE, cp)
            if iv is None or iv <= 0 or iv > 5:
                continue
            g = calc_all_greeks(forward, k, T_val, RISK_FREE, iv, cp)
            rows.append({
                "strike": k, "cp": cp, "iv": iv, "delta": g.delta,
                "volume": vol, "price": price,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _interp_iv_at_delta(options: pd.DataFrame, target_delta: float) -> float:
    """在delta两侧的行权价之间线性插值IV，避免行权价跳变导致的假信号。

    options: 同类型（C或P）的行权价DataFrame，含delta和iv列
    target_delta: 目标delta值（如+0.25或-0.25）
    返回: 插值后的IV（小数），失败返回0.0
    """
    if options.empty or len(options) < 2:
        if not options.empty:
            return float(options.iloc[0]["iv"])
        return 0.0

    df = options.sort_values("delta").reset_index(drop=True)
    deltas = df["delta"].values
    ivs = df["iv"].values

    # 找到target_delta两侧的行权价
    # delta单调递减（Put: 深OTM→ATM = -0.01→-0.50）或递增（Call: OTM→ATM = 0.01→0.50）
    for i in range(len(deltas) - 1):
        d_lo, d_hi = deltas[i], deltas[i + 1]
        if (d_lo <= target_delta <= d_hi) or (d_hi <= target_delta <= d_lo):
            # 线性插值
            span = d_hi - d_lo
            if abs(span) < 1e-10:
                return float(ivs[i])
            w = (target_delta - d_lo) / span
            return float(ivs[i] * (1 - w) + ivs[i + 1] * w)

    # target_delta在范围外，用最近的
    idx = (df["delta"] - target_delta).abs().idxmin()
    return float(df.loc[idx, "iv"])


def calc_rr_bf(skew: pd.DataFrame) -> Tuple[float, float]:
    """25D Risk Reversal 和 Butterfly。

    使用delta线性插值（非最近邻），避免MO行权价间距大（200点）
    导致的行权价跳变假信号。
    """
    if skew.empty:
        return 0.0, 0.0

    puts = skew[skew["cp"] == "P"].copy()
    calls = skew[skew["cp"] == "C"].copy()

    # 25D Put: delta插值到-0.25
    put_25d_iv = _interp_iv_at_delta(puts, -0.25)

    # 25D Call: delta插值到+0.25
    call_25d_iv = _interp_iv_at_delta(calls, 0.25)

    # ATM IV: delta插值到+0.50 (call)
    atm_iv = _interp_iv_at_delta(calls, 0.50)

    rr = put_25d_iv - call_25d_iv  # 正值 = Put skew（看跌偏向）
    bf = (put_25d_iv + call_25d_iv) / 2 - atm_iv if atm_iv > 0 else 0.0
    return rr, bf


def calc_position_greeks(
    positions: list, chain: pd.DataFrame, forwards: dict,
    expire_map: dict, today: str,
    live_prices: dict | None = None,
) -> Tuple[list, dict]:
    """计算持仓Greeks。返回 (详细列表, 组合汇总)。"""
    details = []
    totals = {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}

    for pos in positions:
        p = _parse_mo(pos["ts_code"])
        if not p:
            continue
        em, cp, strike = p
        ed = expire_map.get(em, "")
        if not ed:
            continue

        T_val = _T(ed, today)
        fwd = forwards.get(em, 0)
        if fwd <= 0:
            continue

        # 市场价：优先用实时价格
        mkt_price = 0.0
        if live_prices and pos["ts_code"] in live_prices:
            mkt_price = live_prices[pos["ts_code"]]
        else:
            # 从 chain 查找
            cr = chain[chain["ts_code"] == pos["ts_code"]]
            if not cr.empty:
                mkt_price = float(cr.iloc[0]["close"])

        iv = calc_implied_vol(mkt_price, fwd, strike, T_val, RISK_FREE, cp) \
            if mkt_price > 0 else None
        if iv is None or iv <= 0 or iv > 5:
            iv = 0.25  # fallback

        g = calc_all_greeks(fwd, strike, T_val, RISK_FREE, iv, cp)
        vol = pos["volume"]
        mult = MO_MULT

        details.append({
            "ts_code": pos["ts_code"], "volume": vol,
            "market_price": mkt_price,
            "open_price": pos.get("open_price", 0),
            "iv": iv, "delta": g.delta * vol * mult,
            "theta": g.theta * vol * mult,
            "vega": g.vega * vol * mult,
            "gamma": g.gamma * vol * mult,
        })
        totals["delta"] += g.delta * vol * mult
        totals["gamma"] += g.gamma * vol * mult
        totals["theta"] += g.theta * vol * mult
        totals["vega"] += g.vega * vol * mult

    return details, totals


# ---------------------------------------------------------------------------
# Panel display
# ---------------------------------------------------------------------------

def print_panel(
    today: str,
    now_str: str,
    im_forward: float,
    im_futures: float,
    spot_price: float,
    # vol overview
    atm_iv: float,
    yesterday: dict,
    rv_20d: float,
    rv_5d: float,
    vrp: float,
    garch_sigma: float,
    iv_pctile: float,
    vrp_pctile: float,
    # term structure
    term_data: list,  # [(month, dte, iv, forward, iv_chg)]
    ts_shape: str,
    # skew
    skew_df: pd.DataFrame,
    skew_month: str,
    rr: float,
    bf: float,
    rr_chg: float,
    bf_chg: float,
    # positions
    pos_details: list,
    pos_totals: dict,
    # discount
    discounts: list,  # [(label, price, abs_disc, annual)]
    # signals
    signals: list,    # [(label, text)]
    safety: list,     # [dict with label/fwd/fut/spot/status]
    atm_iv_market: float = 0.0,
    zscore: float | None = None,
    spot_source: str = "",       # "实时" or "估算"
    db=None,                     # for discount percentile
    im_futures_label: str = "IM期货",
    spot_prev_close: float = 0.0,
):
    """输出完整面板。"""
    fwd_warning = im_forward > 0 and im_futures > 0 and im_forward < im_futures

    spot_mark = "=" if spot_source == "实时" else "≈"
    spot_tag = f"({spot_source})" if spot_source else ""
    print(f"\n{'═' * W}")
    print(f" 期权波动率监控 | {now_str} | IM Forward={im_forward:.0f}"
          f"  {im_futures_label}={im_futures:.0f}  现货{spot_mark}{spot_price:.0f}{spot_tag}")
    print(f"{'═' * W}")

    # --- 波动率概览 ---
    prev_iv = float(yesterday.get("atm_iv", 0) or 0)
    iv_chg = (atm_iv - prev_iv) * 100 if prev_iv > 0 else 0

    print(f"\n{'【波动率概览】'}")
    iv_note = ""
    if fwd_warning:
        iv_note = "  (Forward偏低，IV可能偏高)"
    print(f"  结构IV (Forward-based)  :  {atm_iv*100:.2f}%"
          f"    昨日: {prev_iv*100:.2f}%  变动: {iv_chg:+.2f}pp{iv_note}")
    if atm_iv_market > 0:
        diff_pp = (atm_iv_market - atm_iv) * 100
        print(f"  市场IV (期货价格based)  :  {atm_iv_market*100:.2f}%"
              f"    vs结构IV: {diff_pp:+.2f}pp  (用于VRP/情绪)")
    print(f"  20日已实现RV            :  {rv_20d*100:.2f}%"
          f"    5日RV: {rv_5d*100:.1f}%")
    # RV趋势
    if rv_5d > 0 and rv_20d > 0:
        rv_trend = "回落" if rv_5d < rv_20d else "上升"
        print(f"  RV趋势                 :  5d={rv_5d*100:.1f}% → 20d={rv_20d*100:.1f}%"
              f"（波动率在{rv_trend}）")
    iv_src = "市场IV" if atm_iv_market > 0 else "结构IV"
    # Blended RV + GARCH 可靠性
    blended_rv, garch_reliable, blended_label = _calc_blended_rv(rv_5d, rv_20d, garch_sigma)
    garch_tag = "  (可靠)" if garch_reliable else f"   ⚠ 偏高(>RV×1.4)，已降权"
    print(f"  GARCH条件σ              :  {garch_sigma*100:.2f}%{garch_tag}")
    print(f"  Blended RV              :  {blended_rv*100:.1f}%"
          f"    ({blended_label})")
    vrp_blended = (atm_iv_market if atm_iv_market > 0 else atm_iv) - blended_rv \
        if blended_rv > 0 else vrp
    vrp_warn = "   ⚠ GARCH已降权，仅参考" if not garch_reliable else ""
    print(f"  VRP ({iv_src} - BlendedRV):  {vrp_blended*100:+.2f}%"
          f"    (辅助参考){vrp_warn}")
    if fwd_warning:
        print(f"  ⚠ Forward({im_forward:.0f}) < 期货({im_futures:.0f})"
              f"，Put超买可能导致Forward失真")
    print()
    print(f"  IV分位（全历史）        :  {iv_pctile:.0f}%"
          f"       ({_pctile_label(iv_pctile)})")
    print(f"  VRP分位（全历史）       :  {vrp_pctile:.0f}%"
          f"       ({_pctile_label(vrp_pctile)})")

    # --- IV 期限结构 ---
    print(f"\n{'【IV期限结构】'}")
    print(f"  {'到期月':>6}  {'剩余天数':>8}  {'ATM IV':>8}"
          f"  {'隐含Forward':>10}  {'vs昨日':>8}")
    for month, dte_val, iv_val, fwd, chg in term_data:
        chg_str = f"{chg:+.2f}" if chg != 0 else "--"
        print(f"  MO{month:>4}  {dte_val:>8}  {iv_val*100:>7.2f}%"
              f"  {fwd:>10.0f}  {chg_str:>8}")
    print(f"\n  期限结构形态: {ts_shape}")

    # --- IV Skew ---
    print(f"\n[IV Skew] MO{skew_month}")
    if not skew_df.empty:
        print(f"  STRIKE | C/P |      IV | DELTA  |    VOL |   PRICE")
        print(f"  -------+-----+---------+--------+--------+---------")
        for _, r in skew_df.iterrows():
            print(f"  {r['strike']:>6.0f} |  {r['cp']:1s}  | {r['iv']*100:>6.2f}% |"
                  f" {r['delta']:>+5.2f} | {r['volume']:>6.0f} | {r['price']:>7.1f}")
    print()
    print(f"  25D RR  : {rr*100:+5.1f}pp  (>0 = put skew)")
    print(f"  25D BF  : {bf*100:+5.1f}pp  (>0 = tail premium)")
    if rr_chg != 0 or bf_chg != 0:
        print(f"  vs prev : RR {rr_chg*100:+.1f}pp  BF {bf_chg*100:+.1f}pp")

    # --- 持仓 Greeks ---
    print(f"\n[Greeks]")
    if pos_details:
        print(f"  CONTRACT                | DIR | QTY |  PRICE |     IV |  DELTA |  THETA |    VEGA")
        print(f"  ------------------------+-----+-----+--------+--------+--------+--------+--------")
        for d in pos_details:
            ds = " S" if d["volume"] < 0 else " L"
            print(f"  {d['ts_code']:<24s} | {ds:>3s} | {abs(d['volume']):>3d}"
                  f" | {d['market_price']:>6.1f}"
                  f" | {d['iv']*100:>5.1f}%"
                  f" | {d['delta']:>+6.0f} | {d['theta']:>+6.0f}"
                  f" | {d['vega']:>+7.0f}")
        print()
        print(f"  Total: D={pos_totals['delta']:>+,.0f}"
              f"  G={pos_totals['gamma']:>+.4f}"
              f"  T={pos_totals['theta']:>+,.0f}"
              f"  V={pos_totals['vega']:>+,.0f}")
    else:
        print(f"  no MO positions")

    # --- 贴水 ---
    print(f"\n{'【贴水监控】'}")
    print(f"  现货(000852.SH)  :  {spot_price:.2f}")
    for i, (label, price, abs_d, ann) in enumerate(discounts):
        daily_rate = abs_d / spot_price if spot_price > 0 else 0
        pct = _calc_discount_percentile(db, daily_rate, i) if db else None
        pct_s = f"  分位: {_discount_pct_label(pct)}" if pct is not None else ""
        print(f"  {label:<16} :  {price:.2f}"
              f"  贴水: {abs_d:+.0f}点  年化: {ann:+.2f}%{pct_s}")
    if len(discounts) >= 2:
        near_ann = discounts[0][3]
        far_ann = discounts[-1][3]
        diff = near_ann - far_ann
        print(f"  近远月年化差: {diff:+.1f}pp"
              f" (近月{near_ann:+.1f}% vs 远月{far_ann:+.1f}%)")

    # --- Z-Score ---
    if zscore is not None:
        z_label = "极度超卖" if zscore < -2.5 else "超卖" if zscore < -2.0 else \
                  "偏低" if zscore < -1.0 else "中性" if zscore < 1.0 else \
                  "偏高" if zscore < 2.0 else "超买" if zscore < 2.5 else "极度超买"
        print(f"\n{'【Z-Score (EMA20)】'}")
        print(f"  IM Z-Score             :  {zscore:+.2f}    ({z_label})")

    # --- 信号汇总 ---
    print(f"\n{'【波动率交易信号汇总】'}")
    for label, text in signals:
        print(f"  {label:<16} : {text}")

    # 综合建议
    if vrp_blended > 0 and iv_pctile >= 50:
        print(f"  {'综合建议':<16} : 卖方有利 (VRP={vrp_blended*100:+.1f}%, IV P{iv_pctile:.0f})")
    elif vrp_blended > 0:
        print(f"  {'综合建议':<16} : 卖方略有利 (VRP>0但IV分位偏低P{iv_pctile:.0f})")
    elif vrp_blended <= 0 and iv_pctile <= 25:
        print(f"  {'综合建议':<16} : 买方有利 (VRP={vrp_blended*100:+.1f}%, IV P{iv_pctile:.0f})")
    else:
        print(f"  {'综合建议':<16} : 中性 (VRP={vrp_blended*100:+.1f}%, IV P{iv_pctile:.0f})")

    if safety:
        print(f"  持仓安全距离:")
        for s in safety:
            fd, fp = s["fwd"]
            utd, utp = s["fut"]
            sd, sp = s["spot"]
            print(f"    {s['label']}  "
                  f"Forward: {fd:.0f}点({fp:.1f}%){_safety_icon(fd)} | "
                  f"期货: {utd:.0f}点({utp:.1f}%){_safety_icon(utd)} | "
                  f"现货: {sd:.0f}点({sp:.1f}%){_safety_icon(sd)}")

    # IV飙升/回落预警
    if spot_prev_close > 0 and spot_price > 0:
        intraday_chg = (spot_price - spot_prev_close) / spot_prev_close * 100
        net_vega = pos_totals.get("vega", 0)
        if intraday_chg < -1.5:
            est_iv_surge = 1.5  # 保守估计pp
            est_vega_loss = abs(net_vega) * est_iv_surge
            print(f"\n  ⚠️  日内跌幅 {intraday_chg:+.1f}%，IV可能飙升{est_iv_surge}pp+")
            print(f"     Vega={net_vega:+,.0f}元/pp  预估Vega亏损: {-est_vega_loss:+,.0f}元")
        elif intraday_chg > 2.0:
            print(f"\n  ✅ 日内涨幅 {intraday_chg:+.1f}%，IV可能回落0.5-1pp，卖方受益")
            if net_vega < 0:
                est_benefit = abs(net_vega) * 0.75
                print(f"     Vega={net_vega:+,.0f}元/pp  预估Vega收益: +{est_benefit:,.0f}元")

    print(f"\n{'═' * W}")


def _calc_im_zscore(db) -> float | None:
    """从日线数据计算IM当前Z-Score(EMA20)。"""
    try:
        df = db.query_df(
            "SELECT close FROM futures_daily WHERE ts_code='IM.CFX' "
            "AND close > 0 ORDER BY trade_date DESC LIMIT 25"
        )
        if df is None or len(df) < 20:
            return None
        closes = df["close"].astype(float).iloc[::-1].reset_index(drop=True)
        ema20 = float(closes.ewm(span=20).mean().iloc[-1])
        std20 = float(closes.rolling(20).std().iloc[-1])
        if std20 <= 0:
            return None
        return (float(closes.iloc[-1]) - ema20) / std20
    except Exception:
        return None


def _calc_blended_rv(rv_5d: float, rv_20d: float, garch: float) -> tuple[float, bool, str]:
    """计算 Blended RV 和 GARCH 可靠性。

    Returns: (blended_rv, garch_reliable, blended_label)
    """
    rv_max = max(rv_5d, rv_20d) if rv_5d > 0 and rv_20d > 0 else rv_20d
    garch_reliable = True
    if garch > 0 and rv_max > 0 and garch > rv_max * 1.4:
        garch_reliable = False

    if garch_reliable and garch > 0 and rv_5d > 0 and rv_20d > 0:
        blended = 0.4 * rv_5d + 0.4 * rv_20d + 0.2 * garch
        label = "0.4×5d + 0.4×20d + 0.2×GARCH"
    elif rv_5d > 0 and rv_20d > 0:
        blended = 0.5 * rv_5d + 0.5 * rv_20d
        label = "0.5×5d + 0.5×20d, GARCH已降权"
    elif rv_20d > 0:
        blended = rv_20d
        label = "仅RV20d"
    else:
        blended = garch if garch > 0 else 0.0
        label = "仅GARCH"
    return blended, garch_reliable, label


def _vrp_signal_text(vrp: float) -> str:
    """Legacy VRP signal (kept for compatibility)."""
    if vrp > 0.02:
        return "做空波动率"
    elif vrp > 0:
        return "NEUTRAL"
    else:
        return "做多波动率"


def _vol_signal(
    iv_pctile: float, vrp: float, garch: float, garch_prev: float,
    rv_20d: float = 0.0, garch_reliable: bool = True,
) -> list[tuple[str, str]]:
    """
    主信号：IV历史分位（不依赖GARCH）。
    辅助：VRP = IV - BlendedRV（GARCH不可靠时降级）。

    Returns list of (label, text) tuples.
    """
    signals = []

    # 主信号：IV分位（5档）
    if iv_pctile >= 90:
        signals.append(("IV分位信号", f"做空波动率 (IV P{iv_pctile:.0f}, 历史高位)"))
    elif iv_pctile >= 75:
        signals.append(("IV分位信号", f"偏空波动率 (IV P{iv_pctile:.0f}, 偏高)"))
    elif iv_pctile <= 10:
        signals.append(("IV分位信号", f"做多波动率 (IV P{iv_pctile:.0f}, 历史低位)"))
    elif iv_pctile <= 25:
        signals.append(("IV分位信号", f"偏多波动率 (IV P{iv_pctile:.0f}, 偏低)"))
    else:
        signals.append(("IV分位信号", f"NEUTRAL (IV P{iv_pctile:.0f})"))

    # 辅助：VRP（基于 Blended RV）
    if not garch_reliable:
        signals.append(("VRP参考", f"⚠ GARCH偏高(>RV×1.4)，暂不提供方向信号"))
    else:
        garch_change_pp = abs(garch - garch_prev) * 100 if garch > 0 and garch_prev > 0 else 0
        if garch_change_pp > 5:
            signals.append(("VRP参考", f"忽略 (GARCH日变{garch_change_pp:.1f}pp，剧烈波动期)"))
        elif garch_change_pp > 2:
            signals.append(("VRP参考", f"{_vrp_signal_text(vrp)} (GARCH变动{garch_change_pp:.1f}pp，谨慎)"))
        else:
            signals.append(("VRP参考", f"{_vrp_signal_text(vrp)} (GARCH稳定)"))

    return signals


def _pctile_label(p: float) -> str:
    if p >= 80:
        return "偏高"
    elif p >= 60:
        return "中偏高"
    elif p >= 40:
        return "中等水平"
    elif p >= 20:
        return "中偏低"
    else:
        return "偏低"


def _safety_icon(dist: float) -> str:
    if dist > 300:
        return "✓"
    elif dist > 150:
        return "⚠"
    else:
        return "🔴"


def _calc_safety_triple(
    positions: list, forwards: dict, im_prices: dict,
    spot_price: float, im_forward: float,
) -> list:
    """计算三基准安全距离：Forward / 期货 / 现货。"""
    # 构建MO月份→最近IM期货价格的映射
    from utils.cffex_calendar import map_expiry_to_futures_price
    pos_months = set()
    for pos in positions:
        p = _parse_mo(pos.get("ts_code", ""))
        if p:
            pos_months.add(p[0])
    mapped = map_expiry_to_futures_price(
        sorted(pos_months), im_prices, im_forward
    ) if pos_months and im_prices else {}

    results = []
    for pos in positions:
        p = _parse_mo(pos["ts_code"])
        if not p or pos["volume"] >= 0:
            continue  # 只看空头
        em, cp, strike = p
        fwd = forwards.get(em, im_forward)
        # 用映射后的期货价格（MO2605→IM2606）
        _, fut = mapped.get(em, ("", 0))
        if fut <= 0:
            fut = im_prices.get(em, 0)

        if cp == "P":
            fwd_dist = fwd - strike if fwd > 0 else 0
            fut_dist = fut - strike if fut > 0 else 0
            spot_dist = spot_price - strike if spot_price > 0 else 0
            label = f"P-{strike:.0f}"
        else:
            fwd_dist = strike - fwd if fwd > 0 else 0
            fut_dist = strike - fut if fut > 0 else 0
            spot_dist = strike - spot_price if spot_price > 0 else 0
            label = f"C-{strike:.0f}"

        # 预警级别用三个基准中最保守的（最小正距离）
        valid_dists = [d for d in [fwd_dist, fut_dist, spot_dist] if d > 0]
        min_dist = min(valid_dists) if valid_dists else 0
        status = _safety_icon(min_dist)

        results.append({
            "label": label,
            "fwd": (fwd_dist, fwd_dist / fwd * 100 if fwd > 0 else 0),
            "fut": (fut_dist, fut_dist / fut * 100 if fut > 0 else 0),
            "spot": (spot_dist, spot_dist / spot_price * 100 if spot_price > 0 else 0),
            "status": status,
        })
    return results


def _ts_shape(ivs: list) -> str:
    """判断期限结构形态。ivs = [(dte, iv), ...]"""
    if len(ivs) < 2:
        return "数据不足"
    vals = [iv for _, iv in sorted(ivs) if iv > 0]
    if len(vals) < 2:
        return "数据不足"
    # 检查单调性
    increasing = all(vals[i] <= vals[i+1] for i in range(len(vals)-1))
    decreasing = all(vals[i] >= vals[i+1] for i in range(len(vals)-1))
    if decreasing:
        return "倒挂（近月>远月）"
    elif increasing:
        return "正常升水（远月>近月）"
    else:
        return "驼峰/非单调"


# ---------------------------------------------------------------------------
# Snapshot mode (from DB)
# ---------------------------------------------------------------------------

def run_snapshot(db):
    """从数据库读取最新数据，输出一次面板。"""
    today = date.today().strftime("%Y%m%d")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    expire_map = _build_expire_map(db, today)
    if not expire_map:
        print("  [错误] 无法获取到期日映射")
        return
    print(f"  活跃到期月: {sorted(expire_map.keys())}")

    # 尝试今天的数据，fallback 到最近交易日
    chain = _get_mo_chain(db, today, expire_map)
    data_date = today
    if chain.empty:
        latest = db.query_df(
            "SELECT MAX(trade_date) as dt FROM options_daily "
            "WHERE ts_code LIKE 'MO%'"
        )
        if latest is not None and not latest.empty and latest["dt"].iloc[0]:
            data_date = str(latest["dt"].iloc[0])
            chain = _get_mo_chain(db, data_date, expire_map)
            print(f"  [注意] 使用 {data_date} 的期权数据")

    if chain.empty:
        print("  [错误] 无期权链数据")
        return

    # 期货价格
    im_prices = get_im_futures_prices(db, data_date)
    months = active_im_months(data_date)
    near_month = _find_near_month(months, expire_map, data_date)
    im_futures = im_prices.get(near_month, im_prices.get(months[0], 0) if months else 0)
    im_futures_label = f"IM{near_month}" if near_month else "IM期货"

    # 隐含 Forward
    forwards = calc_forwards_by_expiry(chain, expire_map, im_prices, data_date)
    im_forward = forwards.get(near_month, im_futures)

    # 昨日模型数据
    yesterday = _get_yesterday_model(db, data_date)

    # 当日模型数据（eod已跑的情况）
    today_model = {}
    df_tm = db.query_df(
        "SELECT * FROM daily_model_output "
        "WHERE trade_date = ? AND underlying = 'IM'",
        (data_date,),
    )
    if df_tm is not None and not df_tm.empty:
        today_model = df_tm.iloc[0].to_dict()

    # ATM IV（结构IV = Forward-based，用于Greeks定价）
    near_ed = expire_map.get(near_month, "")
    atm_iv = calc_atm_iv_for_month(chain, near_month, im_forward, near_ed, data_date)
    if atm_iv == 0 and today_model.get("atm_iv"):
        atm_iv = float(today_model["atm_iv"])

    # 市场ATM IV（期货价格based，用于VRP/情绪指标）
    atm_iv_market = calc_market_atm_iv(chain, near_month, im_futures, near_ed, data_date)
    if atm_iv_market == 0 and today_model.get("atm_iv_market"):
        atm_iv_market = float(today_model["atm_iv_market"])

    # RV / GARCH / VRP
    rv_20d = float(today_model.get("realized_vol_20d", 0) or
                   yesterday.get("realized_vol_20d", 0) or 0)
    rv_5d = float(today_model.get("realized_vol_60d", 0) or
                  yesterday.get("realized_vol_60d", 0) or 0)
    # 5日RV：用 rv_5d_actual 或简单估算
    rv5_val = float(today_model.get("rv_5d_actual", 0) or 0)
    if rv5_val <= 0:
        rv5_val = rv_20d * 0.8  # rough estimate

    garch = float(today_model.get("garch_forecast_vol", 0) or
                  yesterday.get("garch_forecast_vol", 0) or 0)
    # Blended RV + GARCH sanity check
    blended_rv, garch_reliable, _ = _calc_blended_rv(rv5_val, rv_20d, garch)
    iv_for_vrp = atm_iv_market if atm_iv_market > 0 else atm_iv
    vrp_val = iv_for_vrp - blended_rv if blended_rv > 0 else 0

    # IV & VRP 分位（从 volatility_history 读取完整历史）
    iv_hist = _get_iv_history(db)
    iv_pctile = 50.0
    vrp_pctile = 50.0
    if not iv_hist.empty:
        iv_vals = iv_hist["atm_iv"].dropna().values
        if len(iv_vals) > 5 and atm_iv > 0:
            # volatility_history 存的是百分比(28.85)，atm_iv 是小数(0.2885)
            atm_iv_pct = atm_iv * 100 if atm_iv < 1 else atm_iv
            iv_vals_pct = iv_vals if iv_vals.mean() > 1 else iv_vals * 100
            iv_pctile = float(np.mean(iv_vals_pct <= atm_iv_pct) * 100)
        vrp_vals = iv_hist["vrp"].dropna().values
        if len(vrp_vals) > 5:
            # vrp_rv 在 volatility_history 是百分比，vrp_val 是小数
            vrp_val_pct = vrp_val * 100 if abs(vrp_val) < 1 else vrp_val
            vrp_vals_pct = vrp_vals if abs(vrp_vals.mean()) > 0.5 else vrp_vals * 100
            vrp_pctile = float(np.mean(vrp_vals_pct <= vrp_val_pct) * 100)

    # 期限结构
    term_data = []
    ts_ivs = []
    for em in sorted(chain["expire_month"].unique()):
        ed = expire_map.get(em, "")
        if not ed:
            continue
        dte_val = _dte(ed, data_date)
        fwd = forwards.get(em, im_futures)
        iv_val = calc_atm_iv_for_month(chain, em, fwd, ed, data_date)
        # vs 昨日：简单用 atm_iv 差异估算（无逐月昨日数据时）
        chg = 0.0
        if em == near_month:
            prev_iv = float(yesterday.get("atm_iv", 0) or 0)
            chg = (iv_val - prev_iv) * 100 if prev_iv > 0 else 0
        term_data.append((em, dte_val, iv_val, fwd, chg))
        if iv_val > 0:
            ts_ivs.append((dte_val, iv_val))

    shape = _ts_shape(ts_ivs)

    # Skew
    skew_month = near_month
    # 如果近月剩余<14天，用次月
    if near_ed and _dte(near_ed, data_date) < 14 and len(months) > 1:
        skew_month = months[1]
    skew_ed = expire_map.get(skew_month, "")
    skew_fwd = forwards.get(skew_month, im_forward)
    skew_df = calc_skew_table(chain, skew_month, skew_fwd, skew_ed, data_date)
    rr, bf = calc_rr_bf(skew_df)

    # vs 前一个快照的 RR/BF（优先盘中快照，fallback到daily_model_output）
    prev_rr = 0.0
    prev_bf = 0.0
    try:
        prev_snap = db.query_df(
            "SELECT rr_25d, bf_25d FROM vol_monitor_snapshots "
            "ORDER BY datetime DESC LIMIT 1")
        if prev_snap is not None and not prev_snap.empty:
            prev_rr = float(prev_snap.iloc[0].get("rr_25d", 0) or 0)
            prev_bf = float(prev_snap.iloc[0].get("bf_25d", 0) or 0)
    except Exception:
        pass
    if prev_rr == 0:
        prev_rr = float(yesterday.get("rr_25d", 0) or 0)
    if prev_bf == 0:
        prev_bf = float(yesterday.get("bf_25d", 0) or 0)
    rr_chg = rr - prev_rr if prev_rr != 0 else 0
    bf_chg = bf - prev_bf if prev_bf != 0 else 0

    # 持仓
    positions = _load_positions(db)
    pos_details, pos_totals = calc_position_greeks(
        positions, chain, forwards, expire_map, data_date)

    # 贴水
    spot_price = 0.0
    spot_df = db.query_df(
        "SELECT close FROM index_daily "
        "WHERE ts_code = '000852.SH' ORDER BY trade_date DESC LIMIT 1"
    )
    if spot_df is not None and not spot_df.empty:
        spot_price = float(spot_df.iloc[0]["close"])

    discounts = []
    for i, em in enumerate(months[:3]):
        if i == 0:
            continue  # 当月跳过（就是主力合约本身）
        price = im_prices.get(em, 0)
        if price > 0 and spot_price > 0:
            abs_d = price - spot_price
            ed = expire_map.get(em, "")
            dte_val = _dte(ed, data_date) if ed else 90
            ann = abs_d / spot_price * (365 / dte_val) * 100
            label = f"IM{em}期货"
            discounts.append((label, price, abs_d, ann))

    # 信号（主信号：IV分位，辅助：VRP）
    garch_prev = float(yesterday.get("garch_forecast_vol", 0) or 0)
    signals = _vol_signal(iv_pctile, vrp_val, garch, garch_prev,
                          rv_20d=rv_20d, garch_reliable=garch_reliable)

    if shape.startswith("倒挂"):
        signals.append(("期限结构信号", "近月恐慌溢价，考虑日历价差"))
    else:
        signals.append(("期限结构信号", "正常"))

    rr_pp = rr * 100
    if rr_pp > 5:
        signals.append(("Skew信号", "看跌偏向极强，卖Put有吸引力但风险大"))
    elif rr_pp > 2:
        signals.append(("Skew信号", "正常看跌偏向"))
    else:
        signals.append(("Skew信号", "看跌偏向弱，市场不担心下跌"))

    # 安全距离（三基准）
    safety = _calc_safety_triple(
        positions, forwards, im_prices, spot_price, im_forward)

    # Z-Score
    im_zscore = _calc_im_zscore(db)

    print_panel(
        data_date, now_str, im_forward, im_futures, spot_price,
        atm_iv, yesterday, rv_20d, rv5_val, vrp_val, garch,
        iv_pctile, vrp_pctile,
        term_data, shape,
        skew_df, skew_month, rr, bf, rr_chg, bf_chg,
        pos_details, pos_totals,
        discounts,
        signals, safety,
        atm_iv_market=atm_iv_market,
        zscore=im_zscore,
        im_futures_label=im_futures_label,
        db=db,
    )

    # 写入数据库
    _save_snapshot(db, now_str, atm_iv, rv_20d, rv5_val, vrp_val, garch,
                   iv_pctile, vrp_pctile, term_data, shape, rr, bf,
                   rr_chg, bf_chg, pos_totals, spot_price, discounts)

    # 写入 Markdown
    _save_markdown(data_date, now_str, im_forward, im_futures, atm_iv,
                   yesterday, rv_20d, rv5_val, vrp_val, garch,
                   iv_pctile, vrp_pctile, term_data, shape,
                   skew_df, skew_month, rr, bf, rr_chg, bf_chg,
                   pos_details, pos_totals, spot_price, discounts,
                   signals, safety)


def _save_snapshot(db, dt, atm_iv, rv_20d, rv_5d, vrp, garch,
                   iv_pct, vrp_pct, term_data, shape, rr, bf,
                   rr_chg, bf_chg, totals, spot, discounts):
    """写入 vol_monitor_snapshots。"""
    try:
        iv_m1 = term_data[0][2] if len(term_data) > 0 else None
        iv_m2 = term_data[1][2] if len(term_data) > 1 else None
        iv_m3 = term_data[2][2] if len(term_data) > 2 else None
        disc1 = discounts[0][3] if len(discounts) > 0 else None
        disc2 = discounts[1][3] if len(discounts) > 1 else None

        db._conn.execute(
            "INSERT OR REPLACE INTO vol_monitor_snapshots VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (dt, atm_iv, None, rv_20d, rv_5d, vrp, garch,
             iv_pct, vrp_pct, iv_m1, iv_m2, iv_m3, shape,
             rr, bf, rr_chg, bf_chg,
             totals.get("delta", 0), totals.get("gamma", 0),
             totals.get("theta", 0), totals.get("vega", 0),
             None, spot, disc1, disc2),
        )
        db._conn.commit()
    except Exception as e:
        print(f"  [警告] 写入 vol_monitor_snapshots 失败: {e}")


def _save_markdown(data_date, now_str, im_forward, im_futures, atm_iv,
                   yesterday, rv_20d, rv_5d, vrp, garch,
                   iv_pct, vrp_pct, term_data, shape,
                   skew_df, skew_month, rr, bf, rr_chg, bf_chg,
                   pos_details, pos_totals, spot, discounts,
                   signals, safety):
    """追加写入 logs/vol_monitor/YYYYMMDD.md。"""
    log_dir = os.path.join(ROOT, "logs", "vol_monitor")
    os.makedirs(log_dir, exist_ok=True)
    fpath = os.path.join(log_dir, f"{data_date}.md")

    prev_iv = float(yesterday.get("atm_iv", 0) or 0)
    iv_chg = (atm_iv - prev_iv) * 100 if prev_iv > 0 else 0

    lines = [
        f"\n## {now_str}\n",
        f"IM Forward={im_forward:.0f}  IM期货={im_futures:.0f}\n",
        f"- ATM IV: {atm_iv*100:.2f}% (vs昨日 {iv_chg:+.2f}pp)",
        f"- RV20d: {rv_20d*100:.2f}%  RV5d: {rv_5d*100:.1f}%",
        f"- VRP(Blended): {vrp*100:+.2f}%  GARCH: {garch*100:.2f}%",
        f"- IV分位: {iv_pct:.0f}%  VRP分位: {vrp_pct:.0f}%",
        f"- 期限结构: {shape}",
        f"- 25D RR: {rr*100:+.1f}pp  BF: {bf*100:+.1f}pp",
    ]
    if pos_totals:
        lines.append(
            f"- Greeks: D={pos_totals['delta']:+.0f}"
            f" G={pos_totals['gamma']:+.4f}"
            f" T={pos_totals['theta']:+.0f}"
            f" V={pos_totals['vega']:+.0f}"
        )
    for label, text in signals:
        lines.append(f"- {label}: {text}")
    for s in safety:
        fd, fp = s["fwd"]
        utd, utp = s["fut"]
        sd, sp = s["spot"]
        lines.append(
            f"- {s['label']}  "
            f"Fwd:{fd:.0f}({fp:.1f}%) "
            f"Fut:{utd:.0f}({utp:.1f}%) "
            f"Spot:{sd:.0f}({sp:.1f}%)")
    lines.append("")

    with open(fpath, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Live mode (TQ real-time)
# ---------------------------------------------------------------------------

def run_live(db):
    """盘中实时模式：连接TQ，每5分钟刷新。"""
    from data.sources.tq_client import TqClient

    creds = {
        "auth_account":    os.getenv("TQ_ACCOUNT", ""),
        "auth_password":   os.getenv("TQ_PASSWORD", ""),
        "broker_id":       os.getenv("TQ_BROKER", ""),
        "account_id":      os.getenv("TQ_ACCOUNT_ID", ""),
        "broker_password": os.getenv("TQ_BROKER_PASSWORD", ""),
    }
    if not creds["auth_account"] or not creds["auth_password"]:
        print("请在 .env 中设置 TQ_ACCOUNT 和 TQ_PASSWORD")
        return

    today = date.today().strftime("%Y%m%d")
    expire_map = _build_expire_map(db, today)
    months = active_im_months(today)

    if not expire_map:
        print("  [错误] 无法获取到期日映射")
        return

    print(f"  活跃到期月: {sorted(expire_map.keys())}")

    # 确定要订阅的期货和期权合约
    im_contracts = [f"CFFEX.IM{m}" for m in months]

    # 确定ATM附近行权价（基于数据库最新 IM 价格取整百）
    im_prices = get_im_futures_prices(db, today)
    approx_atm = im_prices.get(months[0], 8000) if months else 8000
    atm_base = round(approx_atm / 100) * 100
    print(f"  IM 参考价: {approx_atm:.0f}  ATM基准: {atm_base}")

    # 订阅月份：取未过期的前2个月份
    opt_months_to_sub = []
    for m in sorted(expire_map.keys()):
        ed = expire_map.get(m, "")
        if ed and _dte(ed, today) >= 1:
            opt_months_to_sub.append(m)
        if len(opt_months_to_sub) >= 2:
            break

    # DB持仓预扫描（用于初始订阅范围，TQ连接后会用实时持仓补充）
    pos_db = _load_positions(db)
    pos_months_extra = set()
    pos_strikes_extra: dict[str, set] = {}  # {month: {strike1, strike2, ...}}
    for pos in pos_db:
        p = _parse_mo(pos.get("ts_code", ""))
        if p:
            em = p[0]
            if em not in opt_months_to_sub:
                pos_months_extra.add(em)
            pos_strikes_extra.setdefault(em, set()).add(int(p[2]))

    if pos_months_extra:
        opt_months_to_sub = sorted(set(opt_months_to_sub) | pos_months_extra)
        print(f"  持仓额外月份: {sorted(pos_months_extra)}")

    # 确保持仓月份对应的IM期货也被订阅
    for em in pos_months_extra:
        # MO月份→最近IM月份
        matched = None
        for im_m in sorted(months):
            if im_m >= em:
                matched = im_m
                break
        if not matched and months:
            matched = months[-1]
        if matched:
            tq_fut = f"CFFEX.IM{matched}"
            if tq_fut not in im_contracts:
                im_contracts.append(tq_fut)
                print(f"  额外期货: {tq_fut} (for MO{em})")

    # 行权价: ATM ± 5档（间距100）
    base_strikes = sorted(atm_base + i * 100 for i in range(-5, 7))

    opt_syms = []
    for m in opt_months_to_sub:
        # 每月份用基础行权价 + 该月份的持仓行权价（不混入其他月份的）
        month_strikes = sorted(set(base_strikes) | pos_strikes_extra.get(m, set()))
        for k in month_strikes:
            for cp in ["C", "P"]:
                opt_syms.append(f"CFFEX.MO{m}-{cp}-{k}")
    strikes = base_strikes  # 面板显示用

    print(f"  订阅期货: {', '.join(im_contracts)}")
    print(f"  订阅期权月份: {opt_months_to_sub}  行权价: {min(strikes)}~{max(strikes)}")
    print(f"  期权合约数: {len(opt_syms)}")

    client = TqClient(**creds)
    client.connect()
    api = client._api

    try:
        # 订阅期货
        fut_quotes = {}
        for sym in im_contracts:
            try:
                fut_quotes[sym] = api.get_quote(sym)
                print(f"  ✓ 期货订阅成功: {sym}")
            except Exception as e:
                print(f"  ✗ 期货订阅失败: {sym} → {e}")

        # 订阅现货指数（中证1000）
        spot_quote = None
        for spot_sym in ["SSE.000852", "KQ.i@SSE.000852", "SZSE.399852"]:
            try:
                spot_quote = api.get_quote(spot_sym)
                print(f"  ✓ 现货指数订阅成功: {spot_sym}")
                break
            except Exception as e:
                print(f"  - 现货指数 {spot_sym} 不可用: {e}")
        if spot_quote is None:
            print(f"  ⚠ 现货指数订阅失败，将使用期货估算")

        # 从TQ实时持仓中发现额外的MO月份和合约
        try:
            tq_positions = api.get_position()
            tq_mo_re = re.compile(r'^CFFEX\.MO(\d{4})-([CP])-(\d+)$')
            tq_extra_months = set()
            tq_extra_strikes: dict[str, set] = {}
            tq_pos_list = []  # 用于打印
            for sym_key in tq_positions:
                pos_obj = tq_positions[sym_key]
                m_match = tq_mo_re.match(sym_key)
                if not m_match:
                    continue
                pos_long = int(getattr(pos_obj, "pos_long", 0))
                pos_short = int(getattr(pos_obj, "pos_short", 0))
                has_pos = pos_long > 0 or pos_short > 0
                if not has_pos:
                    continue
                d = f"多{pos_long}" if pos_long > 0 else f"空{pos_short}"
                tq_pos_list.append(f"    {sym_key:<28s} {d}手")
                em = m_match.group(1)
                strike = int(m_match.group(3))
                if em not in opt_months_to_sub:
                    tq_extra_months.add(em)
                tq_extra_strikes.setdefault(em, set()).add(strike)
            # 打印TQ实时持仓汇总（无论是否有额外月份都打印）
            if tq_pos_list:
                print(f"  [TQ实时持仓] {len(tq_pos_list)} 个MO合约:")
                for line in tq_pos_list:
                    print(line)
            if tq_extra_months:
                opt_months_to_sub = sorted(set(opt_months_to_sub) | tq_extra_months)
                # 为新月份展开完整行权价链
                for em in tq_extra_months:
                    for k in strikes:
                        for cp in ["C", "P"]:
                            sym = f"CFFEX.MO{em}-{cp}-{k}"
                            if sym not in opt_syms:
                                opt_syms.append(sym)
                    # 确保持仓行权价也在列表中
                    for k in tq_extra_strikes.get(em, set()):
                        for cp in ["C", "P"]:
                            sym = f"CFFEX.MO{em}-{cp}-{k}"
                            if sym not in opt_syms:
                                opt_syms.append(sym)
                    # 对应IM期货也要订阅
                    matched = None
                    for im_m in sorted(months):
                        if im_m >= em:
                            matched = im_m
                            break
                    if not matched and months:
                        matched = months[-1]
                    if matched:
                        tq_fut = f"CFFEX.IM{matched}"
                        if tq_fut not in im_contracts:
                            im_contracts.append(tq_fut)
                            try:
                                fut_quotes[tq_fut] = api.get_quote(tq_fut)
                            except Exception:
                                pass
                print(f"  TQ持仓额外月份: {sorted(tq_extra_months)}"
                      f"  行权价: {tq_extra_strikes}")
            # 追加持仓中已有但可能不在标准行权价范围的合约
            for em, ks in tq_extra_strikes.items():
                if em in set(opt_months_to_sub) - tq_extra_months:
                    for k in ks:
                        for cp in ["C", "P"]:
                            sym = f"CFFEX.MO{em}-{cp}-{k}"
                            if sym not in opt_syms:
                                opt_syms.append(sym)
        except Exception as e:
            print(f"  [WARN] TQ持仓读取失败: {e}")

        print(f"  最终订阅期权月份: {opt_months_to_sub}  合约数: {len(opt_syms)}")

        # 订阅期权（先用ATM合约测试每月可用性，再批量订阅）
        opt_quotes = {}
        sub_ok = 0
        sub_fail = 0
        sub_ok_by_month = {}
        skip_months = set()

        # 先测试每月ATM合约是否可订阅
        for m in opt_months_to_sub:
            test_sym = f"CFFEX.MO{m}-C-{atm_base}"
            try:
                opt_quotes[test_sym] = api.get_quote(test_sym)
                sub_ok += 1
                sub_ok_by_month[m] = 1
                print(f"  ✓ MO{m} ATM可用（{test_sym}）")
            except Exception as e:
                skip_months.add(m)
                print(f"  ✗ MO{m} 不可用（{test_sym} → {e}），跳过")

        # 批量订阅可用月份的剩余合约
        for sym in opt_syms:
            if sym in opt_quotes:  # ATM测试已订阅
                continue
            m_match = re.match(r'CFFEX\.MO(\d{4})', sym)
            em = m_match.group(1) if m_match else ""
            if em in skip_months:
                sub_fail += 1
                continue
            try:
                opt_quotes[sym] = api.get_quote(sym)
                sub_ok += 1
                sub_ok_by_month[em] = sub_ok_by_month.get(em, 0) + 1
            except Exception as e:
                sub_fail += 1
                if sub_fail <= 5:
                    print(f"  ✗ {sym} → {e}")

        print(f"  期权订阅: 成功 {sub_ok} / 失败 {sub_fail}")

        # 如果某个月份合约全部失败，从订阅列表中移除，fallback到下个月
        failed_months = []
        for m in list(opt_months_to_sub):
            if sub_ok_by_month.get(m, 0) == 0:
                failed_months.append(m)
        if failed_months:
            for m in failed_months:
                opt_months_to_sub.remove(m)
            print(f"  ⚠ 月份 {failed_months} 订阅全部失败，已移除。剩余: {opt_months_to_sub}")
            # 尝试补充下一个可用月份
            for m in sorted(expire_map.keys()):
                if m not in opt_months_to_sub and expire_map.get(m, "") and _dte(expire_map[m], today) >= 1:
                    # 尝试订阅这个月份的ATM合约验证可用性
                    test_sym = f"CFFEX.MO{m}-C-{atm_base}"
                    try:
                        opt_quotes[test_sym] = api.get_quote(test_sym)
                        opt_months_to_sub.append(m)
                        opt_months_to_sub.sort()
                        # 补充这个月份的全部行权价
                        for k in strikes:
                            for cp in ["C", "P"]:
                                s = f"CFFEX.MO{m}-{cp}-{k}"
                                if s not in opt_quotes:
                                    try:
                                        opt_quotes[s] = api.get_quote(s)
                                    except Exception:
                                        pass
                        print(f"  ✓ 补充月份 {m}，当前订阅: {opt_months_to_sub}")
                        break
                    except Exception:
                        continue
        print(f"\n{'═' * W}")
        print(f"  波动率监控已启动 | 每5分钟刷新")
        print(f"{'═' * W}\n")

        # 对齐到整5分钟时点
        _now = datetime.now()
        seconds_to_wait = (5 - _now.minute % 5) * 60 - _now.second
        if 0 < seconds_to_wait < 300:
            print(f"  等待{seconds_to_wait}秒对齐到整5分钟...")
            time.sleep(seconds_to_wait)

        last_refresh = 0
        while True:
            try:
                api.wait_update()
            except Exception as e:
                now_h = datetime.now().hour
                now_m = datetime.now().minute
                if now_h >= 15 and now_m >= 5:
                    print(f"\n  收盘后正常退出")
                    break
                print(f"\n  [TQ] 连接异常: {e}")
                print(f"  10秒后重试...")
                time.sleep(10)
                continue

            # 对齐到整5分钟：只在分钟数为0/5的整数倍时刷新
            _now = datetime.now()
            if _now.minute % 5 != 0:
                continue
            now = time.time()
            if now - last_refresh < 240:
                continue  # 防止同一个5分钟窗口内重复刷新
            last_refresh = now

            try:
                _refresh_live_panel(
                    db, today, expire_map, months,
                    im_contracts, fut_quotes, opt_quotes,
                    spot_quote=spot_quote,
                    tq_api=api,
                )
            except Exception as e:
                print(f"\n  [ERROR] 面板刷新失败: {e}")

    except KeyboardInterrupt:
        print("\n监控已停止")
    finally:
        client.disconnect()


def _load_positions_from_tq(api) -> list:
    """从TQ账户读取当前MO期权实时持仓（多空分开记录）。"""
    tq_re = re.compile(r'^CFFEX\.(MO\d{4}-[CP]-\d+)$')
    positions = []
    try:
        all_pos = api.get_position()
        for sym, pos in all_pos.items():
            m = tq_re.match(sym)
            if not m:
                continue
            ts_code = m.group(1) + ".CFX"
            long_vol = int(pos.pos_long) if hasattr(pos, "pos_long") else 0
            short_vol = int(pos.pos_short) if hasattr(pos, "pos_short") else 0
            if long_vol > 0:
                positions.append({
                    "ts_code": ts_code,
                    "volume": long_vol,
                    "open_price": float(pos.open_price_long),
                    "float_profit": float(pos.float_profit_long)
                        if hasattr(pos, "float_profit_long") else 0.0,
                })
            if short_vol > 0:
                positions.append({
                    "ts_code": ts_code,
                    "volume": -short_vol,
                    "open_price": float(pos.open_price_short),
                    "float_profit": float(pos.float_profit_short)
                        if hasattr(pos, "float_profit_short") else 0.0,
                })
    except Exception as e:
        print(f"  [POS] TQ read failed: {e}")
        return []

    if positions:
        print(f"  [POS] TQ live: {len(positions)} MO legs")
        for p in positions:
            d = "S" if p["volume"] < 0 else "L"
            pnl = p.get("float_profit", 0)
            print(f"    {p['ts_code']:<24s} {d}{abs(p['volume']):>2d}"
                  f"  pnl={pnl:+,.0f}")
    else:
        print(f"  [POS] no MO positions")
    return positions


def _refresh_live_panel(
    db, today, expire_map, months,
    im_contracts, fut_quotes, opt_quotes,
    spot_quote=None,
    tq_api=None,
):
    """实时刷新一次面板。"""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 读取期货价格
    im_prices_live = {}
    for sym in im_contracts:
        q = fut_quotes.get(sym)
        if q and float(q.last_price) > 0:
            # 从 CFFEX.IM2604 提取月份 2604
            m = sym.replace("CFFEX.IM", "")
            im_prices_live[m] = float(q.last_price)

    # 近月合约价格（跳过已到期月份）
    near_month = _find_near_month(months, expire_map, today)
    im_futures = im_prices_live.get(near_month, 0)
    if im_futures <= 0 and months:
        # fallback: 取第一个有价格的月份
        for m in months:
            if im_prices_live.get(m, 0) > 0:
                im_futures = im_prices_live[m]
                near_month = m
                break
    im_futures_label = f"IM{near_month}" if near_month else "IM期货"

    # 构建实时期权链 DataFrame
    records = []
    live_prices = {}
    no_price_count = 0
    for sym, q in opt_quotes.items():
        if q is None:
            continue
        last = float(q.last_price) if hasattr(q, "last_price") else 0
        bid = float(q.bid_price1) if hasattr(q, "bid_price1") else 0
        ask = float(q.ask_price1) if hasattr(q, "ask_price1") else 0
        vol = int(q.volume) if hasattr(q, "volume") else 0

        # 用中间价（如有），其次 last，再次跳过
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
        if mid <= 0:
            no_price_count += 1
            continue

        # 解析 CFFEX.MO2604-C-8000
        m_match = re.match(r'CFFEX\.(MO(\d{4})-([CP])-(\d+))', sym)
        if not m_match:
            continue
        ts_code = m_match.group(1) + ".CFX"
        em = m_match.group(2)
        cp = m_match.group(3)
        strike = float(m_match.group(4))
        ed = expire_map.get(em, "")

        records.append({
            "ts_code": ts_code, "expire_month": em, "expire_date": ed,
            "call_put": cp, "exercise_price": strike,
            "close": mid, "volume": float(vol), "oi": 0,
        })
        live_prices[ts_code] = mid

    chain = pd.DataFrame(records) if records else pd.DataFrame()

    if chain.empty:
        print(f"  {now_str} | 期权数据不足（{no_price_count}个合约无报价），跳过")
        return

    print(f"  {now_str} | 有效期权报价: {len(records)} 个"
          f"  无报价: {no_price_count} 个"
          f"  月份: {sorted(chain['expire_month'].unique())}")

    # 隐含 Forward
    near_month = _find_near_month(months, expire_map, today)
    forwards = calc_forwards_by_expiry(chain, expire_map, im_prices_live, today)
    im_forward = forwards.get(near_month, im_futures)

    print(f"  近月: MO{near_month}  IM期货: {im_futures:.0f}"
          f"  Forwards: {', '.join(f'{k}={v:.0f}' for k,v in sorted(forwards.items()))}")

    # 昨日数据
    yesterday = _get_yesterday_model(db, today)
    near_ed = expire_map.get(near_month, "")
    atm_iv = calc_atm_iv_for_month(chain, near_month, im_forward, near_ed, today)

    if atm_iv <= 0:
        near_chain = chain[chain["expire_month"] == near_month]
        print(f"  [诊断] ATM IV=0 | near_month={near_month}"
              f"  forward={im_forward:.0f}  expire_date={near_ed}"
              f"  near_chain行数={len(near_chain)}")
        if not near_chain.empty:
            strikes_avail = sorted(near_chain["exercise_price"].unique())
            print(f"  [诊断] 可用行权价: {strikes_avail[:10]}...")

    # 市场ATM IV（期货价格based，用于VRP）
    atm_iv_market = calc_market_atm_iv(chain, near_month, im_futures, near_ed, today)

    # RV / GARCH（盘中用昨日的值）
    rv_20d = float(yesterday.get("realized_vol_20d", 0) or 0)
    rv_5d = float(yesterday.get("rv_5d_actual", 0) or rv_20d * 0.8)
    garch = float(yesterday.get("garch_forecast_vol", 0) or 0)
    # Blended RV + GARCH sanity check
    blended_rv, garch_reliable, _ = _calc_blended_rv(rv_5d, rv_20d, garch)
    iv_for_vrp = atm_iv_market if atm_iv_market > 0 else atm_iv
    vrp_val = iv_for_vrp - blended_rv if blended_rv > 0 else 0

    # 分位（从 volatility_history 读取完整历史）
    iv_hist = _get_iv_history(db)
    iv_pctile = 50.0
    vrp_pctile = 50.0
    if not iv_hist.empty:
        iv_vals = iv_hist["atm_iv"].dropna().values
        if len(iv_vals) > 5 and atm_iv > 0:
            atm_iv_pct = atm_iv * 100 if atm_iv < 1 else atm_iv
            iv_vals_pct = iv_vals if iv_vals.mean() > 1 else iv_vals * 100
            iv_pctile = float(np.mean(iv_vals_pct <= atm_iv_pct) * 100)
        vrp_vals = iv_hist["vrp"].dropna().values
        if len(vrp_vals) > 5:
            vrp_val_pct = vrp_val * 100 if abs(vrp_val) < 1 else vrp_val
            vrp_vals_pct = vrp_vals if abs(vrp_vals.mean()) > 0.5 else vrp_vals * 100
            vrp_pctile = float(np.mean(vrp_vals_pct <= vrp_val_pct) * 100)

    # 期限结构
    term_data = []
    ts_ivs = []
    for em in sorted(chain["expire_month"].unique()):
        ed = expire_map.get(em, "")
        if not ed:
            continue
        dte_val = _dte(ed, today)
        fwd = forwards.get(em, im_futures)
        iv_val = calc_atm_iv_for_month(chain, em, fwd, ed, today)
        chg = 0.0
        if em == near_month:
            prev_iv = float(yesterday.get("atm_iv", 0) or 0)
            chg = (iv_val - prev_iv) * 100 if prev_iv > 0 else 0
        term_data.append((em, dte_val, iv_val, fwd, chg))
        if iv_val > 0:
            ts_ivs.append((dte_val, iv_val))

    shape = _ts_shape(ts_ivs)

    # Skew
    skew_month = near_month
    if near_ed and _dte(near_ed, today) < 14 and len(months) > 1:
        skew_month = months[1]
    skew_ed = expire_map.get(skew_month, "")
    skew_fwd = forwards.get(skew_month, im_forward)
    skew_df = calc_skew_table(chain, skew_month, skew_fwd, skew_ed, today)
    rr, bf = calc_rr_bf(skew_df)

    # vs 前一个快照的 RR/BF
    prev_rr = 0.0
    prev_bf = 0.0
    try:
        prev_snap = db.query_df(
            "SELECT rr_25d, bf_25d FROM vol_monitor_snapshots "
            "ORDER BY datetime DESC LIMIT 1")
        if prev_snap is not None and not prev_snap.empty:
            prev_rr = float(prev_snap.iloc[0].get("rr_25d", 0) or 0)
            prev_bf = float(prev_snap.iloc[0].get("bf_25d", 0) or 0)
    except Exception:
        pass
    if prev_rr == 0:
        prev_rr = float(yesterday.get("rr_25d", 0) or 0)
    if prev_bf == 0:
        prev_bf = float(yesterday.get("bf_25d", 0) or 0)
    rr_chg = rr - prev_rr if prev_rr != 0 else 0
    bf_chg = bf - prev_bf if prev_bf != 0 else 0

    # 持仓：实时模式从TQ读取，fallback到DB
    if tq_api is not None:
        positions = _load_positions_from_tq(tq_api)
        # 动态订阅持仓中新开的合约（盘中新开仓不在初始订阅列表中）
        for pos in positions:
            ts = pos.get("ts_code", "")
            p = _parse_mo(ts)
            if not p:
                continue
            tq_sym = f"CFFEX.{ts.replace('.CFX', '')}"
            if tq_sym not in opt_quotes:
                try:
                    opt_quotes[tq_sym] = tq_api.get_quote(tq_sym)
                    q = opt_quotes[tq_sym]
                    mid = (float(q.bid_price1) + float(q.ask_price1)) / 2 \
                        if float(q.bid_price1) > 0 and float(q.ask_price1) > 0 \
                        else float(q.last_price)
                    if mid > 0:
                        live_prices[ts] = mid
                except Exception:
                    pass
    else:
        positions = _load_positions(db)
    pos_details, pos_totals = calc_position_greeks(
        positions, chain, forwards, expire_map, today, live_prices)

    # 昨日现货收盘（用于日内涨跌幅预警）
    spot_prev_close = 0.0
    spot_df = db.query_df(
        "SELECT close FROM index_daily "
        "WHERE ts_code = '000852.SH' ORDER BY trade_date DESC LIMIT 1"
    )
    if spot_df is not None and not spot_df.empty:
        spot_prev_close = float(spot_df.iloc[0]["close"])

    # 现货价格：优先用TQ实时指数行情，fallback到期货估算
    spot_price = 0.0
    spot_source = ""
    if spot_quote is not None:
        try:
            sp = float(spot_quote.last_price)
            if sp > 0:
                spot_price = sp
                spot_source = "实时"
        except Exception:
            pass

    if spot_price <= 0:
        # fallback: 昨日现货 × (今日IM实时 / 昨日IM收盘)
        im_yesterday_close = 0.0
        im_yest_df = db.query_df(
            "SELECT close FROM futures_daily "
            "WHERE ts_code = 'IM.CFX' AND close > 0 "
            "ORDER BY trade_date DESC LIMIT 1"
        )
        if im_yest_df is not None and not im_yest_df.empty:
            im_yesterday_close = float(im_yest_df.iloc[0]["close"])

        spot_price = spot_prev_close
        if im_futures > 0 and im_yesterday_close > 0 and spot_prev_close > 0:
            spot_price = spot_prev_close * (im_futures / im_yesterday_close)
        spot_source = "估算"

    discounts = []
    for i, em in enumerate(months[:3]):
        if i == 0:
            continue
        price = im_prices_live.get(em, 0)
        if price > 0 and spot_price > 0:
            abs_d = price - spot_price
            ed = expire_map.get(em, "")
            dte_val = _dte(ed, today) if ed else 90
            ann = abs_d / spot_price * (365 / dte_val) * 100
            discounts.append((f"IM{em}期货", price, abs_d, ann))

    # 信号（主信号：IV分位，辅助：VRP）
    garch_prev = float(yesterday.get("garch_forecast_vol", 0) or 0)
    signals = _vol_signal(iv_pctile, vrp_val, garch, garch_prev,
                          rv_20d=rv_20d, garch_reliable=garch_reliable)
    if shape.startswith("倒挂"):
        signals.append(("期限结构信号", "近月恐慌溢价，考虑日历价差"))
    else:
        signals.append(("期限结构信号", "正常"))

    rr_pp = rr * 100
    if rr_pp > 5:
        signals.append(("Skew信号", "看跌偏向极强"))
    elif rr_pp > 2:
        signals.append(("Skew信号", "正常看跌偏向"))
    else:
        signals.append(("Skew信号", "看跌偏向弱"))

    # 安全距离（三基准）
    safety = _calc_safety_triple(
        positions, forwards, im_prices_live, spot_price, im_forward)

    # Z-Score (盘中用实时期货价估算)
    im_zscore = None
    try:
        df_hist = db.query_df(
            "SELECT close FROM futures_daily WHERE ts_code='IM.CFX' "
            "AND close > 0 ORDER BY trade_date DESC LIMIT 25"
        )
        if df_hist is not None and len(df_hist) >= 20:
            closes = df_hist["close"].astype(float).iloc[::-1].reset_index(drop=True)
            ema20 = float(closes.ewm(span=20).mean().iloc[-1])
            std20 = float(closes.rolling(20).std().iloc[-1])
            if std20 > 0 and im_futures > 0:
                im_zscore = (im_futures - ema20) / std20
    except Exception:
        pass

    print_panel(
        today, now_str, im_forward, im_futures, spot_price,
        atm_iv, yesterday, rv_20d, rv_5d, vrp_val, garch,
        iv_pctile, vrp_pctile,
        term_data, shape,
        skew_df, skew_month, rr, bf, rr_chg, bf_chg,
        pos_details, pos_totals,
        discounts,
        signals, safety,
        atm_iv_market=atm_iv_market,
        zscore=im_zscore,
        spot_source=spot_source,
        im_futures_label=im_futures_label,
        spot_prev_close=spot_prev_close,
        db=db,
    )

    # 记录到DB和Markdown
    _save_snapshot(db, now_str, atm_iv, rv_20d, rv_5d, vrp_val, garch,
                   iv_pctile, vrp_pctile, term_data, shape, rr, bf,
                   rr_chg, bf_chg, pos_totals, spot_price, discounts)

    _save_markdown(today, now_str, im_forward, im_futures, atm_iv,
                   yesterday, rv_20d, rv_5d, vrp_val, garch,
                   iv_pctile, vrp_pctile, term_data, shape,
                   skew_df, skew_month, rr, bf, rr_chg, bf_chg,
                   pos_details, pos_totals, spot_price, discounts,
                   signals, safety)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    db = _open_db()
    if db is None:
        print("无法连接数据库")
        return

    # 确保表存在
    try:
        from data.storage.schemas import VOL_MONITOR_SNAPSHOTS_SQL, \
            DAILY_MODEL_OUTPUT_ALTER_SQLS
        for stmt in VOL_MONITOR_SNAPSHOTS_SQL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    db._conn.execute(stmt)
                except Exception:
                    pass
        # 新增 daily_model_output 的列
        for sql in DAILY_MODEL_OUTPUT_ALTER_SQLS:
            try:
                db._conn.execute(sql)
            except Exception:
                pass
        db._conn.commit()
    except Exception:
        pass

    if "--snapshot" in sys.argv:
        print("  [snapshot 模式] 从数据库读取最新数据...")
        run_snapshot(db)
    else:
        print("  [实时模式] 连接天勤...")
        run_live(db)


if __name__ == "__main__":
    main()
