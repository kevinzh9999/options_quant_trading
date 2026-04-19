#!/usr/bin/env python3
"""用归档数据直接驱动monitor._on_new_bar，不依赖TQ网络。
精确记录每个bar的：entry_price, check_exit结果, reversal状态, shadow变化。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
from data.storage.db_manager import get_db
from strategies.intraday.monitor import IntradayMonitor
from strategies.intraday.strategy import IntradayConfig
from strategies.intraday.A_share_momentum_signal_v2 import (
    SentimentData, compute_volume_profile, check_exit, SYMBOL_PROFILES,
)

td = "20260417"
sym = "IM"
db = get_db()

# 加载归档5分钟现货K线
all_bars = db.query_df(
    "SELECT datetime, open, high, low, close, volume FROM index_min "
    "WHERE symbol='000852' AND period=300 ORDER BY datetime"
)
for c in ["open","high","low","close","volume"]:
    all_bars[c] = all_bars[c].astype(float)

# 构建TQ风格的kline series（RangeIndex + datetime列为纳秒）
today_mask = all_bars["datetime"].str.startswith("2026-04-17")
# 需要包含历史bar（warmup用）
all_bars_ns = all_bars.copy()
all_bars_ns["datetime_ns"] = pd.to_datetime(all_bars_ns["datetime"]).astype(np.int64)

# 取今天+前200根bar
today_indices = all_bars.index[today_mask].tolist()
start_idx = max(0, today_indices[0] - 200)
replay_bars = all_bars.iloc[start_idx:today_indices[-1]+1].reset_index(drop=True)
replay_bars_ns = replay_bars.copy()
replay_bars_ns["datetime"] = pd.to_datetime(replay_bars["datetime"]).astype(np.int64)

# 设置monitor
config = IntradayConfig()
config.universe = {sym}
config.tradeable = {sym}
monitor = IntradayMonitor(config)

# 加载辅助数据（和tq_backtest_monitor一样）
_sdf = db.query_df(
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

_bar_all = db.query_df(
    "SELECT datetime, volume FROM index_min WHERE symbol='000852' AND period=300 ORDER BY datetime")
if _bar_all is not None:
    _bar_all['volume'] = _bar_all['volume'].astype(float)
    monitor._vol_profiles[sym] = compute_volume_profile(_bar_all, before_date=td, lookback_days=20)

_ddf = db.query_df(
    f"SELECT trade_date, close as open, close as high, close as low, close, 0 as volume "
    f"FROM index_daily WHERE ts_code='000852.SH' AND trade_date < '{td}' "
    "ORDER BY trade_date DESC LIMIT 30")
if _ddf is not None:
    _ddf = _ddf.sort_values("trade_date").reset_index(drop=True)
    _ddf["close"] = _ddf["close"].astype(float)
    monitor._daily_data[sym] = _ddf

_spot_df = db.query_df(
    f"SELECT close FROM index_daily WHERE ts_code='000852.SH' AND trade_date < '{td}' "
    "ORDER BY trade_date DESC LIMIT 30")
if _spot_df is not None and len(_spot_df) >= 20:
    _closes = _spot_df["close"].astype(float).iloc[::-1].reset_index(drop=True)
    monitor._zscore_params[sym] = {
        "ema20": float(_closes.ewm(span=20).mean().iloc[-1]),
        "std20": float(_closes.rolling(20).std().iloc[-1]),
        "index": "000852.SH"}

monitor._tq_symbols[sym] = "CFFEX.IM2606"
monitor._warmup_done = False
monitor._bars_since_start = 0

# 禁止写DB（避免污染实盘数据）
class FakeRecorder:
    def __init__(self): self.trades = []
    def record_shadow_trade(self, trade):
        self.trades.append(trade)
        print(f"  [DB] record_shadow_trade: {trade.get('symbol')} "
              f"{trade.get('entry_time')}→{trade.get('exit_time')} "
              f"{trade.get('exit_reason')} pnl={trade.get('pnl_pts')}")
    def record_orderbook(self, *a, **kw): pass
    def record_decision(self, *a, **kw): pass
    def record_signal(self, *a, **kw): pass
monitor.recorder = FakeRecorder()

# 逐bar回放
print(f"{'='*80}")
print(f" Monitor Replay Diagnostic | {sym} | {td}")
print(f"{'='*80}\n")

# 找到今天第一根bar在replay_bars中的位置
today_start = None
for i, row in replay_bars.iterrows():
    if row["datetime"].startswith("2026-04-17"):
        today_start = i
        break

# 模拟TQ的kline serial：每次推进一根bar
# kline serial = 200根bar，最后一根是forming
last_bar_time = None

for bar_idx in range(today_start + 1, len(replay_bars)):
    # 构建fake kline series（像TQ的get_kline_serial返回值）
    # 最后一根=forming(当前bar)，倒数第二根=刚完成
    end = bar_idx + 1  # 包含forming bar
    start = max(0, end - 200)
    k5_slice = replay_bars_ns.iloc[start:end].copy()
    k5_slice = k5_slice[["datetime","open","high","low","close","volume"]].reset_index(drop=True)

    # 去重（模拟TQ的is_changing检查）
    completed_dt = int(k5_slice.iloc[-2]["datetime"])
    if completed_dt == last_bar_time:
        continue
    last_bar_time = completed_dt

    # 只处理今天的bar
    completed_ts = pd.Timestamp(completed_dt, unit="ns")
    if completed_ts.strftime("%Y-%m-%d") != "2026-04-17":
        continue
    bj_h = completed_ts.hour + 8
    bj = f"{bj_h:02d}:{completed_ts.minute:02d}"

    # 在_on_new_bar前后记录shadow状态
    shadow_before = dict(monitor._shadow_positions)
    sp_before = shadow_before.get(sym)

    # 调用_on_new_bar
    try:
        monitor._on_new_bar(
            [sym], {sym: k5_slice}, {},  # 无15m（从5m重采样）
            {}, None,  # fut_quotes空（和之前的TqBT v1一致，先排除期货影响）
        )
    except Exception as e:
        print(f"  {bj} ERROR: {e}")
        continue

    shadow_after = dict(monitor._shadow_positions)
    sp_after = shadow_after.get(sym)

    spot_close = float(replay_bars.iloc[bar_idx - 1]["close"])  # completed bar close

    # 检测变化
    opened = (sp_before is None) and (sp_after is not None)
    closed = (sp_before is not None) and (sp_after is None)
    replaced = (sp_before is not None) and (sp_after is not None) and \
               (sp_before.get("entry_time_utc") != sp_after.get("entry_time_utc"))

    event = ""
    if opened:
        event = f" ⚡OPEN {sp_after['direction']} entry={sp_after['entry_price']:.1f} stop={sp_after.get('stop_loss',0):.1f}"
    elif closed:
        event = f" ✖CLOSED"
    elif replaced:
        event = f" 🔄REPLACED → {sp_after['direction']} entry={sp_after['entry_price']:.1f}"

    # 如果有持仓，显示check_exit关键数据
    pos_info = ""
    if sp_after:
        ep = sp_after["entry_price"]
        loss = (ep - spot_close) / ep if sp_after["direction"] == "LONG" else (spot_close - ep) / ep
        pos_info = f" | pos: entry={ep:.1f} loss={loss*100:.3f}% {'⚠>0.3%' if loss>0.003 else ''}"

    if bj >= "09:30" and (event or pos_info or bj <= "10:30"):
        print(f"  {bj} spot={spot_close:.1f}{event}{pos_info}")

# 最终结果
print(f"\n{'='*80}")
print(f" Replay结果:")
print(f"{'='*80}")
print(f" Shadow trades recorded: {len(monitor.recorder.trades)}")
for t in monitor.recorder.trades:
    print(f"   {t.get('symbol')} {t.get('direction')} {t.get('entry_time')}→{t.get('exit_time')} "
          f"{t.get('exit_reason')} pnl={t.get('pnl_pts')}")

print(f"\n 实盘shadow对比:")
print(f"   IM LONG 09:55→14:55 EOD_CLOSE pnl=+77.0pt")
