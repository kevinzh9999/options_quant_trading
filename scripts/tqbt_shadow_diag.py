#!/usr/bin/env python3
"""TqBT诊断：逐bar对比shadow持仓状态，定位与实盘shadow的差异来源。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from tqsdk import TqApi, TqBacktest, BacktestFinished, TqAuth
from strategies.intraday.monitor import IntradayMonitor
from strategies.intraday.strategy import IntradayConfig
from strategies.intraday.A_share_momentum_signal_v2 import check_exit, SYMBOL_PROFILES

td = "20260417"
y, m, d = 2026, 4, 17
symbols = ["IM"]
SPOT_TQ = {"IM": "SSE.000852"}

# Setup monitor
config = IntradayConfig()
config.universe = set(symbols)
config.tradeable = set(symbols)
monitor = IntradayMonitor(config)
monitor._load_daily_data()
monitor._load_sentiment()

# Override data for correct date
from data.storage.db_manager import get_db as _get_db
from strategies.intraday.A_share_momentum_signal_v2 import SentimentData, compute_volume_profile
_db = _get_db()

_sdf = _db.query_df(
    f"SELECT atm_iv, atm_iv_market, vrp, rr_25d, term_structure_shape "
    f"FROM daily_model_output WHERE underlying='IM' AND trade_date < '{td}' "
    "ORDER BY trade_date DESC LIMIT 2")
if _sdf is not None and len(_sdf) >= 2:
    _cur, _prev = _sdf.iloc[0], _sdf.iloc[1]
    monitor._sentiment = SentimentData(
        atm_iv=float(_cur.get("atm_iv_market") or _cur.get("atm_iv") or 0),
        atm_iv_prev=float(_prev.get("atm_iv_market") or _prev.get("atm_iv") or 0),
        rr_25d=float(_cur.get("rr_25d") or 0),
        rr_25d_prev=float(_prev.get("rr_25d") or 0),
        vrp=float(_cur.get("vrp") or 0),
        term_structure=str(_cur.get("term_structure_shape") or ""))

_bar_all = _db.query_df(
    "SELECT datetime, volume FROM index_min WHERE symbol='000852' AND period=300 ORDER BY datetime")
if _bar_all is not None:
    _bar_all['volume'] = _bar_all['volume'].astype(float)
    monitor._vol_profiles["IM"] = compute_volume_profile(_bar_all, before_date=td, lookback_days=20)

_ddf = _db.query_df(
    f"SELECT trade_date, close as open, close as high, close as low, close, 0 as volume "
    f"FROM index_daily WHERE ts_code='000852.SH' AND trade_date < '{td}' "
    "ORDER BY trade_date DESC LIMIT 30")
if _ddf is not None:
    _ddf = _ddf.sort_values("trade_date").reset_index(drop=True)
    _ddf["close"] = _ddf["close"].astype(float)
    monitor._daily_data["IM"] = _ddf

_spot_df = _db.query_df(
    f"SELECT close FROM index_daily WHERE ts_code='000852.SH' AND trade_date < '{td}' "
    "ORDER BY trade_date DESC LIMIT 30")
if _spot_df is not None and len(_spot_df) >= 20:
    _closes = _spot_df["close"].astype(float).iloc[::-1].reset_index(drop=True)
    monitor._zscore_params["IM"] = {
        "ema20": float(_closes.ewm(span=20).mean().iloc[-1]),
        "std20": float(_closes.rolling(20).std().iloc[-1]),
        "index": "000852.SH"}

# TqBacktest
auth = TqAuth(os.getenv("TQ_ACCOUNT", ""), os.getenv("TQ_PASSWORD", ""))
api = TqApi(backtest=TqBacktest(
    start_dt=datetime(y, m, d, 0, 0, 0),
    end_dt=datetime(y, m, d, 23, 59, 59)), auth=auth)

spot_klines_5m = {"IM": api.get_kline_serial("SSE.000852", 300, 200)}
spot_klines_15m = {"IM": api.get_kline_serial("SSE.000852", 900, 100)}

from utils.cffex_calendar import _near_month_by_expiry
near_month = _near_month_by_expiry()
fut_sym = f"CFFEX.IM{near_month}"
monitor._tq_symbols["IM"] = fut_sym
fut_quotes = {"IM": api.get_quote(fut_sym)}

print(f"Contract: {fut_sym}")

monitor._warmup_done = False
monitor._bars_since_start = 0

# Monkey-patch _on_new_bar的check_exit部分，加诊断输出
_orig_on_new_bar = monitor._on_new_bar

bar_count = 0
def patched_on_new_bar(changed_syms, sk5m, sk15m, fq, fk5m):
    global bar_count
    bar_count += 1

    # 调用前记录shadow状态
    sp_before = dict(monitor._shadow_positions)

    # 调用原始方法
    _orig_on_new_bar(changed_syms, sk5m, sk15m, fq, fk5m)

    # 调用后检查变化
    sp_after = dict(monitor._shadow_positions)

    # 获取当前bar信息
    sk = sk5m.get("IM")
    if sk is None or len(sk) < 2:
        return
    completed = sk.iloc[-2]
    import pandas as pd
    bar_dt = pd.to_datetime(completed["datetime"], unit="ns")
    bar_utc = bar_dt.strftime("%H:%M")
    bj_h = int(bar_utc[:2]) + 8
    bj_time = f"{bj_h:02d}:{bar_utc[3:5]}"
    spot_close = float(completed["close"])

    # 期货价
    fq_im = fq.get("IM") if fq else None
    fut_last = float(fq_im.last_price) if fq_im and float(fq_im.last_price) > 0 else 0
    fut_bid = float(fq_im.bid_price1) if fq_im and hasattr(fq_im, 'bid_price1') and float(fq_im.bid_price1) > 0 else 0
    fut_ask = float(fq_im.ask_price1) if fq_im and hasattr(fq_im, 'ask_price1') and float(fq_im.ask_price1) > 0 else 0

    # 打印shadow持仓状态
    had_pos = "IM" in sp_before
    has_pos = "IM" in sp_after

    if had_pos or has_pos or bar_count <= 15:
        sp = sp_after.get("IM") or sp_before.get("IM")
        pos_str = ""
        if has_pos:
            p = sp_after["IM"]
            entry = p.get("entry_price", 0)
            stop = p.get("stop_loss", 0)
            highest = p.get("highest_since", 0)
            hold = p.get("hold_days", p.get("hold_bars", 0))
            pnl = (spot_close - entry) if p["direction"] == "LONG" else (entry - spot_close)
            pos_str = (f"  POS: {p['direction']} entry={entry:.1f} stop={stop:.1f} "
                      f"hi={highest:.1f} pnl={pnl:+.1f} hold={hold}")

        event = ""
        if not had_pos and has_pos:
            event = " ⚡ OPEN"
        elif had_pos and not has_pos:
            event = " ✖ EXIT"

        print(f"  {bj_time} spot={spot_close:.1f} fut={fut_last:.1f}(b{fut_bid:.0f}/a{fut_ask:.0f})"
              f"{event}{pos_str}")

monitor._on_new_bar = patched_on_new_bar

print(f"\nRunning TqBT 20260417 IM diagnostic...\n")

try:
    while True:
        api.wait_update()
        changed_syms = []
        sk = spot_klines_5m.get("IM")
        if sk is not None and api.is_changing(sk) and len(sk) >= 2:
            completed_dt = int(sk.iloc[-2]["datetime"])
            prev_dt = monitor._last_bar_time.get("IM")
            if prev_dt != completed_dt:
                monitor._last_bar_time["IM"] = completed_dt
                changed_syms.append("IM")
        if changed_syms:
            monitor._on_new_bar(changed_syms, spot_klines_5m, spot_klines_15m, fut_quotes, None)
except BacktestFinished:
    pass
finally:
    api.close()

# 汇总
print(f"\n=== TqBT Shadow Trades ===")
shadow_db = _db.query_df(f"SELECT * FROM shadow_trades WHERE trade_date='{td}' AND symbol='IM' ORDER BY entry_time")
if shadow_db is not None and len(shadow_db) > 0:
    for _, r in shadow_db.iterrows():
        print(f"  {r['direction']} {r['entry_time']}→{r['exit_time']} @{float(r['entry_price']):.0f}→{float(r['exit_price']):.0f} "
              f"{r['exit_reason']} pnl={float(r['pnl_pts']):+.1f}pt score={r['entry_score']}")
print(f"\n=== Live Shadow (should match) ===")
print(f"  LONG 09:55→14:55 @8052→8129 EOD_CLOSE pnl=+77.0pt score=65")
