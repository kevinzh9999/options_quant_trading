"""
morning_briefing.py
-------------------
盘前市场情绪综合判断。综合跨市场信号、市场宽度、成交额、
波动率环境、价格位置五个维度，输出当日方向Guidance。

用法：
    python scripts/morning_briefing.py                # 用最近交易日
    python scripts/morning_briefing.py --date 20260327 # 指定日期
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(ROOT) / ".env")
except ImportError:
    pass

from config.config_loader import ConfigLoader
from data.storage.db_manager import DBManager

W = 70  # panel width
BRIEFING_JSON = "/tmp/morning_briefing.json"


# ---------------------------------------------------------------------------
# Tushare helpers
# ---------------------------------------------------------------------------

def _get_pro():
    import tushare as ts
    token = os.getenv("TUSHARE_TOKEN", "")
    return ts.pro_api(token)


def _safe_call(fn, **kwargs):
    """安全调用 Tushare，失败返回空 DataFrame。"""
    try:
        import time
        time.sleep(0.3)
        df = fn(**kwargs)
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        print(f"  [WARN] Tushare调用失败: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 数据获取
# ---------------------------------------------------------------------------

def _get_recent_trade_dates(db: DBManager, target_date: str, n: int = 2) -> list[str]:
    """获取target_date及之前最近的n个交易日（含target_date本身如果是交易日）。"""
    df = db.query_df(
        "SELECT trade_date FROM index_daily "
        "WHERE ts_code='000852.SH' AND trade_date <= ? "
        "ORDER BY trade_date DESC LIMIT ?",
        (target_date, n),
    )
    if df is not None and not df.empty:
        return [str(d) for d in df["trade_date"].tolist()]
    return [target_date]


def _get_global_indices(pro, target_date: str, db: DBManager = None) -> dict:
    result = {"a50_pct": None, "spx_pct": None, "ixic_pct": None, "hsi_pct": None}
    # 优先从本地DB读取
    if db is not None:
        for ts_code, key in [("XIN9", "a50_pct"), ("SPX", "spx_pct"),
                              ("IXIC", "ixic_pct"), ("HSI", "hsi_pct")]:
            df = db.query_df(
                "SELECT pct_chg FROM global_index_daily "
                "WHERE ts_code=? AND trade_date<=? ORDER BY trade_date DESC LIMIT 1",
                (ts_code, target_date))
            if df is not None and not df.empty:
                result[key] = float(df.iloc[0]["pct_chg"])
        if all(v is not None for v in result.values()):
            return result

    # Fallback: Tushare实时
    dt = datetime.strptime(target_date, "%Y%m%d")
    start = (dt - timedelta(days=10)).strftime("%Y%m%d")
    for ts_code, key in [("XIN9", "a50_pct"), ("SPX", "spx_pct"),
                          ("IXIC", "ixic_pct"), ("HSI", "hsi_pct")]:
        if result[key] is not None:
            continue
        df = _safe_call(pro.index_global, ts_code=ts_code,
                        start_date=start, end_date=target_date,
                        fields="trade_date,close,pct_chg")
        if not df.empty:
            df = df.sort_values("trade_date", ascending=False)
            result[key] = float(df.iloc[0]["pct_chg"])
    return result


def _get_market_breadth(pro, trade_date: str, db: DBManager = None) -> dict:
    result = {"advance": 0, "decline": 0, "ad_ratio": 1.0,
              "limit_up": 0, "limit_down": 0}
    # 优先本地DB
    if db is not None:
        df = db.query_df(
            "SELECT * FROM market_breadth WHERE trade_date=?", (trade_date,))
        if df is not None and not df.empty:
            r = df.iloc[0]
            return {"advance": int(r.get("advance_count", 0)),
                    "decline": int(r.get("decline_count", 0)),
                    "ad_ratio": float(r.get("ad_ratio", 1.0)),
                    "limit_up": int(r.get("limit_up", 0)),
                    "limit_down": int(r.get("limit_down", 0))}
    # Fallback
    df = _safe_call(pro.daily, trade_date=trade_date, fields="ts_code,pct_chg")
    if df.empty:
        return result
    df["pct_chg"] = df["pct_chg"].astype(float)
    result["advance"] = int((df["pct_chg"] > 0).sum())
    result["decline"] = int((df["pct_chg"] < 0).sum())
    result["ad_ratio"] = result["advance"] / max(result["decline"], 1)
    result["limit_up"] = int((df["pct_chg"] >= 9.9).sum())
    result["limit_down"] = int((df["pct_chg"] <= -9.9).sum())
    return result


def _get_market_volume(pro, trade_date: str, db: DBManager = None) -> dict:
    """沪深合计成交额 + 5日均值 + 20日区间位置。"""
    result = {"today_amount": 0, "avg_5d_amount": 0, "amount_ratio": 1.0,
              "min_20d": 0, "max_20d": 0, "range_pct": 0.5}

    # 优先本地DB
    amounts = []
    if db is not None:
        dt_obj = datetime.strptime(trade_date, "%Y%m%d")
        start40 = (dt_obj - timedelta(days=60)).strftime("%Y%m%d")
        tdf = db.query_df(
            "SELECT trade_date, total_amount FROM market_turnover "
            "WHERE trade_date>=? AND trade_date<=? ORDER BY trade_date",
            (start40, trade_date))
        if tdf is not None and len(tdf) >= 5:
            amounts = tdf["total_amount"].astype(float).tolist()

    if not amounts:
        # Fallback: Tushare
        dt_obj = datetime.strptime(trade_date, "%Y%m%d")
        start = (dt_obj - timedelta(days=40)).strftime("%Y%m%d")
        combined = {}
        for idx_code in ["000001.SH", "399001.SZ"]:
            df = _safe_call(pro.index_daily, ts_code=idx_code,
                            start_date=start, end_date=trade_date,
                            fields="trade_date,amount")
            if df.empty:
                continue
            for _, row in df.iterrows():
                td = row["trade_date"]
                amt = float(row["amount"]) / 1e5
                combined[td] = combined.get(td, 0) + amt
        if not combined:
            return result
        dates_sorted = sorted(combined.keys())
        amounts = [combined[d] for d in dates_sorted]

    if amounts:
        result["today_amount"] = amounts[-1]
    if len(amounts) >= 5:
        result["avg_5d_amount"] = float(np.mean(amounts[-5:]))
        if result["avg_5d_amount"] > 0:
            result["amount_ratio"] = result["today_amount"] / result["avg_5d_amount"]
    # 20日区间
    recent_20 = amounts[-20:] if len(amounts) >= 20 else amounts
    result["min_20d"] = float(np.min(recent_20))
    result["max_20d"] = float(np.max(recent_20))
    rng = result["max_20d"] - result["min_20d"]
    result["range_pct"] = (result["today_amount"] - result["min_20d"]) / rng if rng > 0 else 0.5
    return result


def _get_vol_environment(db: DBManager) -> dict:
    result = {"atm_iv": None, "iv_percentile": None, "vrp": None,
              "term_structure": "", "garch_reliable": True}
    dmo = db.query_df(
        "SELECT atm_iv, vrp, garch_reliable, iv_percentile_hist "
        "FROM daily_model_output WHERE underlying='IM' "
        "ORDER BY trade_date DESC LIMIT 1")
    if dmo is not None and not dmo.empty:
        r = dmo.iloc[0]
        result["atm_iv"] = float(r.get("atm_iv") or 0) or None
        result["vrp"] = float(r.get("vrp") or 0) or None
        gr = r.get("garch_reliable")
        result["garch_reliable"] = bool(gr) if gr is not None else True
        ip = r.get("iv_percentile_hist")
        if ip is not None and ip != "":
            result["iv_percentile"] = float(ip)

    snap = db.query_df(
        "SELECT iv_percentile, term_structure_shape "
        "FROM vol_monitor_snapshots ORDER BY datetime DESC LIMIT 1")
    if snap is not None and not snap.empty:
        r = snap.iloc[0]
        if result["iv_percentile"] is None:
            ip = r.get("iv_percentile")
            if ip is not None and ip != "":
                result["iv_percentile"] = float(ip)
        ts_shape = r.get("term_structure_shape")
        if ts_shape:
            result["term_structure"] = str(ts_shape)
    return result


def _get_price_position(db: DBManager, trade_date: str) -> dict:
    result = {"current": 0, "high_20": 0, "low_20": 0,
              "range_pos": 0.5, "mom_5d": 0, "streak": 0, "streak_dir": ""}
    df = db.query_df(
        "SELECT trade_date, close FROM index_daily "
        "WHERE ts_code='000852.SH' AND trade_date <= ? "
        "ORDER BY trade_date DESC LIMIT 25",
        (trade_date,))
    if df is None or len(df) < 6:
        return result
    df = df.sort_values("trade_date").reset_index(drop=True)
    closes = df["close"].astype(float).values
    result["current"] = float(closes[-1])
    recent_20 = closes[-20:] if len(closes) >= 20 else closes
    result["high_20"] = float(np.max(recent_20))
    result["low_20"] = float(np.min(recent_20))
    rng = result["high_20"] - result["low_20"]
    result["range_pos"] = (result["current"] - result["low_20"]) / rng if rng > 0 else 0.5
    result["mom_5d"] = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 else 0

    streak = 0
    streak_dir = ""
    for i in range(len(closes) - 1, 0, -1):
        if closes[i] > closes[i - 1]:
            if streak_dir in ("", "up"):
                streak += 1; streak_dir = "up"
            else:
                break
        elif closes[i] < closes[i - 1]:
            if streak_dir in ("", "down"):
                streak += 1; streak_dir = "down"
            else:
                break
        else:
            break
    result["streak"] = streak
    result["streak_dir"] = streak_dir
    return result


# ---------------------------------------------------------------------------
# P2 指标
# ---------------------------------------------------------------------------

def _get_north_money(pro, trade_date: str, db: DBManager = None) -> dict:
    """北向资金。"""
    result = {"north_money": None, "north_5d_avg": None}
    # 本地DB优先
    if db is not None:
        dt = datetime.strptime(trade_date, "%Y%m%d")
        start = (dt - timedelta(days=12)).strftime("%Y%m%d")
        ndf = db.query_df(
            "SELECT north_money FROM northbound_flow "
            "WHERE trade_date>=? AND trade_date<=? ORDER BY trade_date",
            (start, trade_date))
        if ndf is not None and not ndf.empty:
            vals = ndf["north_money"].astype(float).tolist()
            result["north_money"] = vals[-1]
            if len(vals) >= 5:
                result["north_5d_avg"] = float(np.mean(vals[-5:]))
            return result
    # Fallback
    dt = datetime.strptime(trade_date, "%Y%m%d")
    start = (dt - timedelta(days=12)).strftime("%Y%m%d")
    df = _safe_call(pro.moneyflow_hsgt, start_date=start, end_date=trade_date,
                    fields="trade_date,north_money")
    if df.empty:
        return result
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["north_money"] = df["north_money"].astype(float)
    result["north_money"] = float(df.iloc[-1]["north_money"]) / 10000
    if len(df) >= 5:
        result["north_5d_avg"] = float(df["north_money"].tail(5).mean()) / 10000
    return result


def _get_margin(pro, trade_date: str, db: DBManager = None) -> dict:
    """融资余额。"""
    result = {"margin_balance": None, "margin_change": None}
    # 本地DB优先
    if db is not None:
        mdf = db.query_df(
            "SELECT rzye, rzye_chg FROM margin_data WHERE trade_date=?",
            (trade_date,))
        if mdf is not None and not mdf.empty:
            result["margin_balance"] = float(mdf.iloc[0]["rzye"])
            result["margin_change"] = float(mdf.iloc[0]["rzye_chg"])
            return result
    # Fallback
    dt = datetime.strptime(trade_date, "%Y%m%d")
    start = (dt - timedelta(days=5)).strftime("%Y%m%d")
    total_today = 0.0
    total_prev = 0.0
    for exch in ["SSE", "SZSE"]:
        df = _safe_call(pro.margin, start_date=start, end_date=trade_date,
                        exchange_id=exch, fields="trade_date,rzye")
        if df.empty:
            continue
        df = df.sort_values("trade_date").reset_index(drop=True)
        df["rzye"] = df["rzye"].astype(float)
        total_today += float(df.iloc[-1]["rzye"])
        if len(df) >= 2:
            total_prev += float(df.iloc[-2]["rzye"])
    if total_today > 0:
        result["margin_balance"] = total_today / 1e8
        if total_prev > 0:
            result["margin_change"] = (total_today - total_prev) / 1e8
    return result


def _get_pcr(db: DBManager, trade_date: str) -> dict:
    """从 option_pcr_daily 读取MO/IO/HO的PCR数据。"""
    result = {"pcr": None, "option_vol_ratio": None, "pcr_detail": []}
    dt = datetime.strptime(trade_date, "%Y%m%d")
    start = (dt - timedelta(days=10)).strftime("%Y%m%d")

    # 优先读 option_pcr_daily 表
    df = db.query_df(
        "SELECT * FROM option_pcr_daily "
        "WHERE trade_date >= ? AND trade_date <= ? ORDER BY trade_date, product",
        (start, trade_date),
    )
    if df is not None and not df.empty:
        today = df[df["trade_date"] == trade_date]
        # MO 的成交量PCR作为主PCR
        mo_today = today[today["product"] == "MO"]
        if not mo_today.empty:
            vpcr = mo_today.iloc[0].get("volume_pcr")
            result["pcr"] = float(vpcr) if vpcr is not None and vpcr == vpcr else None

        # 各品种明细（含5日均值）
        for prod in ["MO", "IO", "HO"]:
            prod_df = df[df["product"] == prod].copy()
            if prod_df.empty:
                continue
            prod_today = prod_df[prod_df["trade_date"] == trade_date]
            if prod_today.empty:
                continue
            r = prod_today.iloc[0]
            tv = float(r.get("total_volume", 0) or 0)
            toi = float(r.get("total_oi", 0) or 0)
            vpcr = r.get("volume_pcr")
            opcr = r.get("oi_pcr")
            # 5日均值
            avg5_vol = float(prod_df["total_volume"].tail(5).mean()) if len(prod_df) >= 2 else tv
            avg5_oi = float(prod_df["total_oi"].tail(5).mean()) if len(prod_df) >= 2 else toi
            result["pcr_detail"].append({
                "product": prod,
                "total_vol": tv, "avg5_vol": avg5_vol,
                "vol_ratio": tv / avg5_vol if avg5_vol > 0 else 1.0,
                "volume_pcr": float(vpcr) if vpcr is not None and vpcr == vpcr else None,
                "total_oi": toi, "avg5_oi": avg5_oi,
                "oi_ratio": toi / avg5_oi if avg5_oi > 0 else 1.0,
                "oi_pcr": float(opcr) if opcr is not None and opcr == opcr else None,
            })

        # MO成交量比（量比）
        mo_hist = df[df["product"] == "MO"]
        if len(mo_hist) >= 2:
            today_v = float(mo_hist.iloc[-1]["total_volume"])
            avg_v = float(mo_hist["total_volume"].tail(5).mean()) if len(mo_hist) >= 5 \
                else float(mo_hist["total_volume"].mean())
            if avg_v > 0:
                result["option_vol_ratio"] = today_v / avg_v
        return result

    # Fallback: 直接从 options_daily 计算（兼容无PCR表的情况）
    fdf = db.query_df(
        "SELECT "
        "  CASE WHEN ts_code LIKE '%%MO%%-C-%%' THEN 'C' "
        "       WHEN ts_code LIKE '%%MO%%-P-%%' THEN 'P' END as cp, "
        "  SUM(volume) as vol "
        "FROM options_daily WHERE trade_date = ? AND ts_code LIKE 'MO%%' "
        "GROUP BY cp", (trade_date,))
    if fdf is not None and len(fdf) >= 2:
        cv = float(fdf[fdf["cp"] == "C"]["vol"].sum())
        pv = float(fdf[fdf["cp"] == "P"]["vol"].sum())
        if cv > 0:
            result["pcr"] = pv / cv
    return result


def _get_fut_holding(pro, trade_date: str) -> dict:
    """IM期货前20大席位多空持仓。"""
    result = {"net_holding": None, "net_change": None}
    df = _safe_call(pro.fut_holding, trade_date=trade_date, exchange="CFFEX")
    if df.empty:
        return result
    # 过滤IM合约
    im_df = df[df["symbol"].astype(str).str.startswith("IM")].copy()
    if im_df.empty:
        return result
    for col in ["long_hld", "short_hld", "long_chg", "short_chg"]:
        if col in im_df.columns:
            im_df[col] = pd.to_numeric(im_df[col], errors="coerce").fillna(0)
    total_long = float(im_df["long_hld"].sum())
    total_short = float(im_df["short_hld"].sum())
    result["net_holding"] = total_long - total_short
    if "long_chg" in im_df.columns and "short_chg" in im_df.columns:
        long_chg = float(im_df["long_chg"].sum())
        short_chg = float(im_df["short_chg"].sum())
        result["net_change"] = long_chg - short_chg
    return result


def _get_etf_share(pro, trade_date: str) -> dict:
    """ETF份额变化。"""
    result = {"etf_300_change": None, "etf_1000_change": None}
    dt = datetime.strptime(trade_date, "%Y%m%d")
    start = (dt - timedelta(days=5)).strftime("%Y%m%d")
    for ts_code, key in [("510300.SH", "etf_300_change"),
                          ("560010.SH", "etf_1000_change")]:
        df = _safe_call(pro.fund_share, ts_code=ts_code,
                        start_date=start, end_date=trade_date)
        if df.empty or len(df) < 2:
            continue
        df = df.sort_values("trade_date").reset_index(drop=True)
        df["fd_share"] = df["fd_share"].astype(float)
        today_s = float(df.iloc[-1]["fd_share"])
        prev_s = float(df.iloc[-2]["fd_share"])
        result[key] = (today_s - prev_s) / 10000  # 万份→亿份
    return result


# ---------------------------------------------------------------------------
# 评分
# ---------------------------------------------------------------------------

def _calc_score(global_idx, breadth, volume, vol_env, price_pos,
                north=None, margin=None, pcr_data=None, etf=None,
                fut_hold=None):
    score = 0
    reasons = []

    # --- 跨市场（±30）---
    a50 = global_idx.get("a50_pct")
    if a50 is not None:
        if a50 > 1.0:   score += 15; reasons.append(f"A50涨{a50:+.1f}%")
        elif a50 > 0.3:  score += 8
        elif a50 < -1.0: score -= 15; reasons.append(f"A50跌{a50:+.1f}%")
        elif a50 < -0.3: score -= 8

    spx = global_idx.get("spx_pct")
    ixic = global_idx.get("ixic_pct")
    if spx is not None and ixic is not None:
        us = (spx + ixic) / 2
        if us > 1.0:   score += 10
        elif us > 0.3:  score += 5
        elif us < -1.0: score -= 10; reasons.append(f"美股跌{us:+.1f}%")
        elif us < -0.3: score -= 5

    hsi = global_idx.get("hsi_pct")
    if hsi is not None:
        if hsi > 1.0: score += 5
        elif hsi < -1.0: score -= 5

    # --- 市场宽度（±25）---
    ad = breadth.get("ad_ratio", 1.0)
    if ad > 3.0:   score += 15; reasons.append(f"涨跌比{ad:.1f}")
    elif ad > 2.0:  score += 10
    elif ad > 1.2:  score += 5
    elif ad < 0.33: score -= 15; reasons.append(f"涨跌比{ad:.2f}")
    elif ad < 0.5:  score -= 10
    elif ad < 0.8:  score -= 5

    lu, ld = breadth.get("limit_up", 0), breadth.get("limit_down", 0)
    if lu > 80: score += 10; reasons.append(f"涨停{lu}家")
    elif lu > 40: score += 5
    if ld > 80: score -= 10; reasons.append(f"跌停{ld}家")
    elif ld > 40: score -= 5

    # --- 成交额（±10）---
    vol_r = volume.get("amount_ratio", 1.0)
    vol_rp = volume.get("range_pct", 0.5)
    mom = price_pos.get("mom_5d", 0)
    if vol_r > 1.3 and vol_rp > 0.7:
        score += 5; reasons.append("显著放量")
    elif vol_r < 0.7 and vol_rp < 0.3:
        score -= 5; reasons.append("显著缩量")
        if mom < -0.01: score -= 5

    # --- 波动率环境（±15）---
    ip = vol_env.get("iv_percentile")
    vrp = vol_env.get("vrp")
    term = vol_env.get("term_structure", "")
    if ip is not None and ip > 85:
        score -= 5; reasons.append(f"IV P{ip:.0f}偏高")
    if vrp is not None:
        vp = vrp * 100 if abs(vrp) < 1 else vrp
        if vp < -5: score -= 5; reasons.append(f"VRP={vp:.1f}%卖方不利")
        elif vp > 5: score += 5
    if "倒挂" in term:
        score -= 5; reasons.append("期限倒挂")

    # --- 价格位置（±20）---
    if mom > 0.03:   score += 10; reasons.append(f"5日涨{mom*100:.1f}%")
    elif mom > 0.01:  score += 5
    elif mom < -0.03: score -= 10; reasons.append(f"5日跌{mom*100:.1f}%")
    elif mom < -0.01: score -= 5

    rp = price_pos.get("range_pos", 0.5)
    if rp > 0.7: score += 5
    elif rp < 0.2: score -= 5; reasons.append("20日范围底部")

    streak = price_pos.get("streak", 0)
    sd = price_pos.get("streak_dir", "")
    if streak >= 3 and sd == "up": score += 5
    elif streak >= 3 and sd == "down": score -= 5; reasons.append(f"连跌{streak}天")

    # --- P2: 北向资金（±10）---
    if north:
        nm = north.get("north_money")
        if nm is not None:
            if nm > 100: score += 10; reasons.append(f"北向+{nm:.0f}亿")
            elif nm > 50: score += 5
            elif nm < -100: score -= 10; reasons.append(f"北向{nm:.0f}亿")
            elif nm < -50: score -= 5

    # --- P2: 融资余额（±3, 含逆向+5）---
    if margin:
        mc = margin.get("margin_change")
        if mc is not None:
            if mc > 50: score += 3
            elif mc < -50: score -= 3

    # --- P2: PCR（±5, 综合持仓PCR逆向指标）---
    if pcr_data:
        detail = pcr_data.get("pcr_detail", [])
        oi_pcrs = [d["oi_pcr"] for d in detail if d.get("oi_pcr") is not None]
        if oi_pcrs:
            avg_oi_pcr = float(np.mean(oi_pcrs))
            if avg_oi_pcr > 1.5: score += 5; reasons.append(f"持仓PCR={avg_oi_pcr:.2f}极悲观逆向")
            elif avg_oi_pcr < 0.7: score -= 5; reasons.append(f"持仓PCR={avg_oi_pcr:.2f}极乐观逆向")
        else:
            pcr_v = pcr_data.get("pcr")
            if pcr_v is not None:
                if pcr_v > 1.5: score += 5; reasons.append(f"PCR={pcr_v:.2f}极悲观逆向")
                elif pcr_v < 0.7: score -= 5; reasons.append(f"PCR={pcr_v:.2f}极乐观逆向")

    # --- P2: ETF份额（±3）---
    if etf:
        e300 = etf.get("etf_300_change")
        e1000 = etf.get("etf_1000_change")
        etf_total = (e300 or 0) + (e1000 or 0)
        if etf_total > 20: score += 3; reasons.append("ETF大幅净申购")
        elif etf_total < -20: score -= 3; reasons.append("ETF大幅净赎回")

    # --- P2: 期货多空持仓（±3）---
    if fut_hold:
        nc = fut_hold.get("net_change")
        if nc is not None:
            if nc > 500: score += 3; reasons.append(f"IM净多增{nc:.0f}手")
            elif nc < -500: score -= 3; reasons.append(f"IM净空增{abs(nc):.0f}手")

    return score, reasons


def _score_to_direction(score):
    if score >= 50: return "偏多", 5
    elif score >= 35: return "偏多", 4
    elif score >= 20: return "偏多", 3
    elif score > 10: return "偏多", 2
    elif score > -10: return "中性", 1
    elif score > -20: return "偏空", 2
    elif score > -35: return "偏空", 3
    elif score > -50: return "偏空", 4
    else: return "偏空", 5


def _calc_d_override(direction, confidence):
    if confidence >= 4 and direction == "偏多":
        return {"LONG": 1.2, "SHORT": 0.5}
    elif confidence >= 3 and direction == "偏多":
        return {"LONG": 1.1, "SHORT": 0.7}
    elif confidence >= 4 and direction == "偏空":
        return {"LONG": 0.5, "SHORT": 1.2}
    elif confidence >= 3 and direction == "偏空":
        return {"LONG": 0.7, "SHORT": 1.1}
    return None


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def _stars(n): return "★" * n + "☆" * (5 - n)


def print_briefing(trade_date, direction, confidence, score,
                   global_idx, breadth, volume, vol_env, price_pos,
                   reasons, d_override, contrarian_note="",
                   original_score=None,
                   north=None, margin=None, pcr_data=None, etf=None,
                   fut_hold=None):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'═' * W}")
    print(f" Morning Briefing | {now_str}")
    print(f"{'═' * W}")
    print(f"【方向判断】{direction} | 置信度: {_stars(confidence)} ({confidence}/5) | 评分: {score:+d}")

    print(f"\n【跨市场】")
    a50 = global_idx.get("a50_pct")
    print(f"  A50夜盘: {a50:+.2f}%" if a50 is not None else "  A50夜盘: N/A")
    spx, ixic = global_idx.get("spx_pct"), global_idx.get("ixic_pct")
    print(f"  美股: SPX {spx:+.1f}%  纳斯达克 {ixic:+.1f}%" if spx and ixic else "  美股: N/A")
    hsi = global_idx.get("hsi_pct")
    print(f"  恒生: {hsi:+.2f}%" if hsi is not None else "  恒生: N/A")

    print(f"\n【市场宽度】(昨日 {trade_date[:4]}/{trade_date[4:6]}/{trade_date[6:]})")
    print(f"  涨跌家数: {breadth['advance']}涨 / {breadth['decline']}跌 = {breadth['ad_ratio']:.1f}")
    print(f"  涨停: {breadth['limit_up']}家  跌停: {breadth['limit_down']}家")

    print(f"\n【成交额】")
    ta = volume["today_amount"]
    aa = volume["avg_5d_amount"]
    vr = volume["amount_ratio"]
    mn = volume.get("min_20d", 0)
    mx = volume.get("max_20d", 0)
    rp = volume.get("range_pct", 0.5)
    rp_tag = "偏放量" if rp > 0.6 else ("偏缩量" if rp < 0.4 else "中等")
    print(f"  沪深合计: {ta:,.0f}亿  5日均: {aa:,.0f}亿  比值: {vr:.2f}")
    print(f"  20日区间: {mn:,.0f}~{mx:,.0f}亿  当前位置: {rp*100:.0f}%（{rp_tag}）")

    print(f"\n【波动率环境】")
    iv = vol_env.get("atm_iv")
    ip = vol_env.get("iv_percentile")
    vrp_v = vol_env.get("vrp")
    iv_s = f"{iv*100:.1f}%" if iv and iv < 1 else (f"{iv:.1f}%" if iv else "N/A")
    ip_s = f"P{ip:.0f}" if ip else ""
    vrp_s = f"{vrp_v*100:.1f}%" if vrp_v and abs(vrp_v) < 1 else (f"{vrp_v:.1f}%" if vrp_v else "N/A")
    print(f"  ATM IV: {iv_s} ({ip_s})  VRP: {vrp_s}")
    term = vol_env.get("term_structure", "")
    if term: print(f"  期限结构: {term}")

    print(f"\n【价格位置】")
    cur, h20, l20 = price_pos["current"], price_pos["high_20"], price_pos["low_20"]
    rp = price_pos["range_pos"]
    print(f"  000852: {cur:.0f}  20日范围: {l20:.0f}~{h20:.0f} ({rp*100:.0f}%)")
    mom = price_pos["mom_5d"]
    streak, sd = price_pos["streak"], price_pos["streak_dir"]
    sd_cn = f"连{'涨' if sd == 'up' else '跌'}{streak}天" if streak > 0 else "无连续"
    print(f"  5日动量: {mom*100:+.1f}%  {sd_cn}")

    # P2 指标
    if north or margin or pcr_data or etf or fut_hold:
        print(f"\n【资金流向】(数据日: {trade_date[:4]}/{trade_date[4:6]}/{trade_date[6:]})")
        if north:
            nm = north.get("north_money")
            na = north.get("north_5d_avg")
            nm_s = f"{nm:+.1f}亿" if nm is not None else "N/A"
            na_s = f"5日均: {na:+.1f}亿" if na is not None else ""
            print(f"  北向资金: {nm_s}  {na_s}")
        if margin:
            mb = margin.get("margin_balance")
            mc = margin.get("margin_change")
            mb_s = f"{mb:,.0f}亿" if mb is not None else "N/A"
            mc_s = f"日变化: {mc:+.0f}亿" if mc is not None else ""
            print(f"  融资余额: {mb_s}  {mc_s}")

        if fut_hold:
            nh = fut_hold.get("net_holding")
            nc = fut_hold.get("net_change")
            nh_s = f"净{'多' if nh and nh > 0 else '空'}{abs(nh or 0):.0f}手" if nh is not None else "N/A"
            nc_s = f"日变{nc:+.0f}手" if nc is not None else ""
            print(f"  IM前20席位: {nh_s}  {nc_s}")

        print(f"\n【期权情绪】(全月份合计)")
        if pcr_data:
            detail = pcr_data.get("pcr_detail", [])
            if detail:
                print(f"  {'品种':>4s} | {'日成交量':>8s} | {'5日均量':>7s} | {'量比':>5s}"
                      f" | {'成交PCR':>7s} | {'总持仓量':>8s} | {'5日均OI':>7s}"
                      f" | {'OI比':>5s} | {'持仓PCR':>7s}")
                print(f"  {'----':>4s}-+-{'-'*8}-+-{'-'*7}-+-{'-'*5}"
                      f"-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}-+-{'-'*5}-+-{'-'*7}")
                for d in detail:
                    tv = d["total_vol"] / 10000
                    a5v = d["avg5_vol"] / 10000
                    vr = d["vol_ratio"]
                    vp = d["volume_pcr"]
                    toi = d["total_oi"] / 10000
                    a5o = d["avg5_oi"] / 10000
                    oir = d["oi_ratio"]
                    op = d["oi_pcr"]
                    print(f"  {d['product']:>4s} | {tv:>7.1f}万 | {a5v:>6.1f}万"
                          f" | {vr:>4.2f}x | {vp:>7.2f}" if vp else f"  {d['product']:>4s} | ... | N/A",
                          end="")
                    print(f" | {toi:>7.1f}万 | {a5o:>6.1f}万"
                          f" | {oir:>4.2f}x | {op:>7.2f}" if op else " | N/A")
            else:
                pcr_v = pcr_data.get("pcr")
                print(f"  PCR(MO): {pcr_v:.2f}" if pcr_v is not None else "  PCR(MO): N/A")
            ovr = pcr_data.get("option_vol_ratio")
            if ovr is not None and not detail:
                print(f"  MO成交量/5日均: {ovr:.2f}x")

        if etf:
            print(f"\n【市场结构】")
            e3 = etf.get("etf_300_change")
            e10 = etf.get("etf_1000_change")
            if e3 is not None:
                print(f"  300ETF份额变化: {e3:+.1f}亿份")
            if e10 is not None:
                print(f"  1000ETF份额变化: {e10:+.1f}亿份")

    if reasons: print(f"\n【关键原因】{'  |  '.join(reasons[:6])}")

    if contrarian_note:
        os_ = original_score if original_score is not None else score
        print(f"\n【逆向修正】{contrarian_note}")
        print(f"  原始评分: {os_:+d} → 修正后置信度: {_stars(confidence)} (封顶3星，不覆盖d值)")

    print(f"\n【操作建议】")
    if direction == "偏多": print(f"  日内方向: 偏多，做多信号可信度高于做空")
    elif direction == "偏空": print(f"  日内方向: 偏空，做空信号可信度高于做多")
    else: print(f"  日内方向: 中性，多空均可，谨慎操作")
    if d_override:
        print(f"  d值建议: LONG d={d_override['LONG']} / SHORT d={d_override['SHORT']}")
    else:
        print(f"  d值建议: 不覆盖，用原有daily_mult")
    if ip and ip > 70: print(f"  期权: IV仍在P{ip:.0f}高位，卖方持仓持有但不加仓")
    if vrp_v and ((abs(vrp_v) < 1 and vrp_v < -0.05) or (abs(vrp_v) >= 1 and vrp_v < -5)):
        print(f"  风险提示: VRP为负，警惕波动率再次扩大")
    print(f"{'═' * W}")


# ---------------------------------------------------------------------------
# 持久化
# ---------------------------------------------------------------------------

def _save_to_db(db_path, trade_date, data):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS morning_briefing (
            trade_date       TEXT PRIMARY KEY,
            direction        TEXT,
            confidence       INT,
            score            INT,
            a50_pct          REAL,
            spx_pct          REAL,
            ixic_pct         REAL,
            hsi_pct          REAL,
            ad_ratio         REAL,
            limit_up         INT,
            limit_down       INT,
            market_amount    REAL,
            amount_ratio     REAL,
            iv_percentile    REAL,
            vrp              REAL,
            daily_5d_mom     REAL,
            range_position   REAL,
            streak           INT,
            d_override_long  REAL,
            d_override_short REAL,
            reasons          TEXT
        )
    """)
    d_ovr = data.get("d_override") or {}
    conn.execute(
        "INSERT OR REPLACE INTO morning_briefing VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (trade_date, data["direction"], data["confidence"], data["score"],
         data.get("a50_pct"), data.get("spx_pct"), data.get("ixic_pct"),
         data.get("hsi_pct"), data.get("ad_ratio"), data.get("limit_up"),
         data.get("limit_down"), data.get("market_amount"),
         data.get("amount_ratio"), data.get("iv_percentile"),
         data.get("vrp"), data.get("mom_5d"), data.get("range_pos"),
         data.get("streak"), d_ovr.get("LONG"), d_ovr.get("SHORT"),
         " | ".join(data.get("reasons", []))))
    conn.commit()
    conn.close()


def _save_json(trade_date, direction, confidence, score, d_override):
    data = {"date": trade_date, "direction": direction,
            "confidence": confidence, "score": score, "d_override": d_override}
    try:
        with open(BRIEFING_JSON, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"  [WARN] 写入JSON失败: {e}")


def _save_markdown(trade_date, direction, confidence, score, reasons):
    log_dir = os.path.join(ROOT, "logs", "briefing")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"{trade_date}.md"), "w") as f:
        f.write(f"# Morning Briefing {trade_date}\n\n")
        f.write(f"方向: {direction}  置信度: {confidence}/5  评分: {score:+d}\n\n")
        f.write(f"关键原因:\n")
        for r in reasons:
            f.write(f"- {r}\n")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_briefing(target_date=None):
    db = DBManager(ConfigLoader().get_db_path())
    if target_date is None:
        target_date = datetime.now().strftime("%Y%m%d")

    # target_date = "今天"（盘前运行的日期）
    # prev_trade_date = 前一个交易日（市场宽度/成交额的数据源，盘前已有）
    # price_date = 前一个交易日（价格位置也用已有数据）
    recent = _get_recent_trade_dates(db, target_date, 2)
    prev_trade_date = recent[0]  # 最近的交易日（可能是昨天）
    # 如果target_date本身是交易日且在recent[0]，prev要往前一天
    if prev_trade_date == target_date and len(recent) >= 2:
        prev_trade_date = recent[1]
    print(f"  Briefing日期: {target_date}  数据日: {prev_trade_date}")

    # P1 数据（优先本地DB，fallback到Tushare）
    print(f"  获取跨市场数据...")
    pro = _get_pro()
    global_idx = _get_global_indices(pro, prev_trade_date, db=db)
    print(f"  获取市场宽度(数据日: {prev_trade_date})...")
    breadth = _get_market_breadth(pro, prev_trade_date, db=db)
    print(f"  获取成交额...")
    volume = _get_market_volume(pro, prev_trade_date, db=db)
    print(f"  读取波动率环境...")
    vol_env = _get_vol_environment(db)
    print(f"  计算价格位置...")
    price_pos = _get_price_position(db, prev_trade_date)

    # P2 数据（优先本地DB，fallback到Tushare）
    print(f"  获取北向资金...")
    north = _get_north_money(pro, prev_trade_date, db=db)
    print(f"  获取融资余额...")
    margin_data = _get_margin(pro, prev_trade_date, db=db)
    print(f"  计算PCR...")
    pcr_data = _get_pcr(db, prev_trade_date)
    print(f"  获取ETF份额...")
    etf = _get_etf_share(pro, prev_trade_date)
    print(f"  获取期货持仓...")
    fut_hold = _get_fut_holding(pro, prev_trade_date)

    score, reasons = _calc_score(
        global_idx, breadth, volume, vol_env, price_pos,
        north=north, margin=margin_data, pcr_data=pcr_data, etf=etf,
        fut_hold=fut_hold)
    direction, confidence = _score_to_direction(score)
    d_override = _calc_d_override(direction, confidence)

    # 极端值逆向修正
    contrarian_note = ""
    original_score = score
    ad = breadth.get("ad_ratio", 1.0)
    if score < -40 and ad < 0.3:
        contrarian_note = "⚠ 极端偏空，恐慌可能过度，反弹概率升高。不建议追空。"
        confidence = min(confidence, 3)
        d_override = None
    elif score > 40 and ad > 4.0:
        contrarian_note = "⚠ 极端偏多，亢奋可能过度，回调概率升高。不建议追多。"
        confidence = min(confidence, 3)
        d_override = None

    print_briefing(prev_trade_date, direction, confidence, score,
                   global_idx, breadth, volume, vol_env, price_pos,
                   reasons, d_override, contrarian_note, original_score,
                   north=north, margin=margin_data, pcr_data=pcr_data, etf=etf,
                   fut_hold=fut_hold)

    db_path = ConfigLoader().get_db_path()
    save_data = {
        "direction": direction, "confidence": confidence, "score": score,
        "a50_pct": global_idx.get("a50_pct"), "spx_pct": global_idx.get("spx_pct"),
        "ixic_pct": global_idx.get("ixic_pct"), "hsi_pct": global_idx.get("hsi_pct"),
        "ad_ratio": breadth.get("ad_ratio"), "limit_up": breadth.get("limit_up"),
        "limit_down": breadth.get("limit_down"),
        "market_amount": volume.get("today_amount"),
        "amount_ratio": volume.get("amount_ratio"),
        "iv_percentile": vol_env.get("iv_percentile"), "vrp": vol_env.get("vrp"),
        "mom_5d": price_pos.get("mom_5d"), "range_pos": price_pos.get("range_pos"),
        "streak": price_pos.get("streak"), "d_override": d_override,
        "reasons": reasons,
        "north_money": north.get("north_money"),
        "margin_balance": margin_data.get("margin_balance"),
        "margin_change": margin_data.get("margin_change"),
        "pcr": pcr_data.get("pcr"),
        "option_vol_ratio": pcr_data.get("option_vol_ratio"),
        "etf_300_change": etf.get("etf_300_change"),
        "etf_1000_change": etf.get("etf_1000_change"),
    }
    _save_to_db(db_path, target_date, save_data)
    _save_json(target_date, direction, confidence, score, d_override)
    _save_markdown(target_date, direction, confidence, score, reasons)
    print(f"\n  已保存: DB + {BRIEFING_JSON} + logs/briefing/{target_date}.md")


def main():
    parser = argparse.ArgumentParser(description="Morning Briefing")
    parser.add_argument("--date", default=None, help="YYYYMMDD (default: today)")
    args = parser.parse_args()
    run_briefing(args.date)


if __name__ == "__main__":
    main()
