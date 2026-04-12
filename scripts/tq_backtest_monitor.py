#!/usr/bin/env python3
"""TqBacktest验证：用TQ回放驱动monitor的完整信号逻辑。

直接复用monitor的_on_new_bar()，只替换TQ连接为TqBacktest模式。
这样信号逻辑、bar处理、shadow系统跟实盘monitor完全一致，零差异。

使用方式：
    python scripts/tq_backtest_monitor.py --date 20260410
    python scripts/tq_backtest_monitor.py --date 20260410 --symbol IM
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

import argparse
from datetime import datetime, date
from typing import Dict, List
import pandas as pd


def run_tq_backtest(td: str, symbols: List[str] = None):
    """用TqBacktest回放指定日期，直接驱动monitor。"""
    from dotenv import load_dotenv
    load_dotenv()

    from tqsdk import TqApi, TqBacktest, BacktestFinished, TqAuth

    if symbols is None:
        symbols = ["IM", "IC"]

    y, m, d = int(td[:4]), int(td[4:6]), int(td[6:8])

    # 现货→TQ代码映射
    SPOT_TQ = {"IM": "SSE.000852", "IF": "SSE.000300", "IH": "SSE.000016", "IC": "SSE.000905"}

    # 创建monitor实例（会加载日线、Z-Score、vol_profile等）
    from strategies.intraday.monitor import IntradayMonitor
    from strategies.intraday.strategy import IntradayConfig

    config = IntradayConfig()
    config.universe = set(symbols)
    config.tradeable = set(symbols)
    monitor = IntradayMonitor(config)

    # 加载日线等初始化数据
    monitor._load_daily_data()
    monitor._load_sentiment()

    # TqBacktest回测历史日期时，覆盖sentiment避免未来数据泄漏
    # monitor._load_sentiment()用datetime.now()取当天日期，回测过去日期时会包含未来数据
    from data.storage.db_manager import get_db as _get_db
    from strategies.intraday.A_share_momentum_signal_v2 import SentimentData
    _db = _get_db()
    _sdf = _db.query_df(
        "SELECT atm_iv, atm_iv_market, vrp, rr_25d, term_structure_shape "
        f"FROM daily_model_output WHERE underlying='IM' AND trade_date < '{td}' "
        "ORDER BY trade_date DESC LIMIT 2"
    )
    if _sdf is not None and len(_sdf) >= 2:
        _cur, _prev = _sdf.iloc[0], _sdf.iloc[1]
        monitor._sentiment = SentimentData(
            atm_iv=float(_cur.get("atm_iv_market") or _cur.get("atm_iv") or 0),
            atm_iv_prev=float(_prev.get("atm_iv_market") or _prev.get("atm_iv") or 0),
            rr_25d=float(_cur.get("rr_25d") or 0),
            rr_25d_prev=float(_prev.get("rr_25d") or 0),
            vrp=float(_cur.get("vrp") or 0),
            term_structure=str(_cur.get("term_structure_shape") or ""),
        )
        print(f"  [TqBT] Sentiment覆盖为 trade_date<{td} 的数据")

    # 同样覆盖vol_profile（_load_daily_data里用datetime.now()取today_str）
    from strategies.intraday.A_share_momentum_signal_v2 import compute_volume_profile
    _SPOT_SYM = {"IM": "000852", "IF": "000300", "IH": "000016", "IC": "000905"}
    for _sym in symbols:
        _spot = _SPOT_SYM.get(_sym)
        if not _spot:
            continue
        _bar_all = _db.query_df(
            f"SELECT datetime, volume FROM index_min "
            f"WHERE symbol='{_spot}' AND period=300 ORDER BY datetime"
        )
        if _bar_all is not None and len(_bar_all) > 0:
            _bar_all['volume'] = _bar_all['volume'].astype(float)
            monitor._vol_profiles[_sym] = compute_volume_profile(
                _bar_all, before_date=td, lookback_days=20)
    print(f"  [TqBT] Vol profile覆盖为 before_date={td}")

    # 覆盖daily_data（最关键！影响daily_mult和intraday_filter）
    _SPOT_IDX = {"IM": "000852.SH", "IF": "000300.SH", "IH": "000016.SH", "IC": "000905.SH"}
    for _sym in symbols:
        _idx_code = _SPOT_IDX.get(_sym)
        if not _idx_code:
            continue
        _ddf = _db.query_df(
            f"SELECT trade_date, close as open, close as high, "
            f"close as low, close, 0 as volume "
            f"FROM index_daily WHERE ts_code = '{_idx_code}' "
            f"AND trade_date < '{td}' "
            f"ORDER BY trade_date DESC LIMIT 30"
        )
        if _ddf is not None and len(_ddf) > 0:
            _ddf = _ddf.sort_values("trade_date").reset_index(drop=True)
            _ddf["close"] = _ddf["close"].astype(float)
            monitor._daily_data[_sym] = _ddf

        # 覆盖zscore
        _spot_df = _db.query_df(
            f"SELECT close FROM index_daily WHERE ts_code = '{_idx_code}' "
            f"AND trade_date < '{td}' ORDER BY trade_date DESC LIMIT 30"
        )
        if _spot_df is not None and len(_spot_df) >= 20:
            _closes = _spot_df["close"].astype(float).iloc[::-1].reset_index(drop=True)
            _ema20 = float(_closes.ewm(span=20).mean().iloc[-1])
            _std20 = float(_closes.rolling(20).std().iloc[-1])
            if _std20 > 0:
                monitor._zscore_params[_sym] = {"ema20": _ema20, "std20": _std20, "index": _idx_code}
    print(f"  [TqBT] Daily data + Z-Score覆盖为 trade_date<{td}")

    # TqBacktest API
    auth = TqAuth(os.getenv("TQ_ACCOUNT", ""), os.getenv("TQ_PASSWORD", ""))
    api = TqApi(backtest=TqBacktest(
        start_dt=datetime(y, m, d, 0, 0, 0),
        end_dt=datetime(y, m, d, 23, 59, 59),
    ), auth=auth)

    # 订阅现货K线和期货行情
    spot_klines_5m = {}
    spot_klines_15m = {}
    fut_quotes = {}

    # 用近月合约作为期货数据源
    from utils.cffex_calendar import _near_month_by_expiry
    near_month = _near_month_by_expiry()

    for sym in symbols:
        spot_sym = SPOT_TQ.get(sym)
        if spot_sym:
            spot_klines_5m[sym] = api.get_kline_serial(spot_sym, 300, 200)
            spot_klines_15m[sym] = api.get_kline_serial(spot_sym, 900, 100)
        fut_sym = f"CFFEX.{sym}{near_month}"
        monitor._tq_symbols[sym] = fut_sym
        fut_quotes[sym] = api.get_quote(fut_sym)

    print(f"  Contracts: {monitor._tq_symbols}")

    # warmup状态
    monitor._warmup_done = False
    monitor._bars_since_start = 0

    print(f"\n  Running TqBacktest replay...\n")

    try:
        while True:
            api.wait_update()

            # 与修复后的monitor主循环完全一致
            changed_syms = []
            for sym in symbols:
                sk = spot_klines_5m.get(sym)
                if sk is not None and api.is_changing(sk):
                    if len(sk) >= 2:
                        completed_dt = int(sk.iloc[-2]["datetime"])
                        prev_dt = monitor._last_bar_time.get(sym)
                        if prev_dt != completed_dt:
                            monitor._last_bar_time[sym] = completed_dt
                            changed_syms.append(sym)

            if changed_syms:
                # fut_quotes传空dict：shadow PnL用spot price（与自研backtest一致）
                # 实盘monitor用期货价算PnL，但TqBacktest验证目的是对齐自研backtest
                # 期货贴水3-4%会导致混合价源PnL偏差数十点/笔
                monitor._on_new_bar(
                    changed_syms, spot_klines_5m, spot_klines_15m,
                    {}, None,
                )

    except BacktestFinished:
        pass
    finally:
        api.close()

    # 收集shadow trades
    from data.storage.db_manager import get_db
    db = get_db()
    # shadow_trades在monitor的recorder里，直接从DB读
    trades_df = db.query_df(
        f"SELECT * FROM shadow_trades WHERE trade_date='{td}' ORDER BY entry_time"
    )
    return trades_df


def run_custom_backtest(td: str, symbols: List[str] = None):
    """运行自研backtest。"""
    from data.storage.db_manager import get_db
    from scripts.backtest_signals_day import run_day

    if symbols is None:
        symbols = ["IM", "IC"]

    db = get_db()
    results = {}
    for sym in symbols:
        trades = run_day(sym, td, db, verbose=False)
        full = [t for t in trades if not t.get("partial")]
        results[sym] = full
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="回测日期 YYYYMMDD")
    parser.add_argument("--symbol", default=None, help="品种 IM/IC, 默认两个都跑")
    args = parser.parse_args()

    td = args.date
    symbols = [args.symbol] if args.symbol else ["IM", "IC"]

    print(f"{'='*70}")
    print(f"  TqBacktest vs 自研Backtest 对比 | {td}")
    print(f"{'='*70}\n")

    # 1. 自研backtest
    print("── 自研Backtest ──")
    custom_results = run_custom_backtest(td, symbols)
    for sym in symbols:
        trades = custom_results.get(sym, [])
        pnl = sum(t["pnl_pts"] for t in trades)
        print(f"  {sym}: {len(trades)}笔 PnL={pnl:+.1f}pt")
        for t in trades:
            et = t.get("entry_time", "?")
            xt = t.get("exit_time", "?")
            print(f"    {t['direction']:5s} {et}→{xt} "
                  f"{t.get('exit_reason','?'):20s} score={t.get('entry_score',0)} "
                  f"pnl={t['pnl_pts']:+.1f}pt")
    print()

    # 2. TqBacktest（直接驱动monitor）
    print("── TqBacktest (直接驱动monitor) ──")

    # 先备份今天的shadow_trades，避免TqBacktest写入污染
    from data.storage.db_manager import get_db
    db = get_db()
    existing_shadow = db.query_df(
        f"SELECT * FROM shadow_trades WHERE trade_date='{td}'"
    )

    # 清空今天的shadow记录（TqBacktest会重新写入）
    db._conn.execute(f"DELETE FROM shadow_trades WHERE trade_date='{td}'")
    db._conn.commit()

    try:
        tq_shadow = run_tq_backtest(td, symbols)
    finally:
        # 恢复原始shadow数据
        db._conn.execute(f"DELETE FROM shadow_trades WHERE trade_date='{td}'")
        if existing_shadow is not None and len(existing_shadow) > 0:
            existing_shadow.to_sql("shadow_trades", db._conn, if_exists="append", index=False)
        db._conn.commit()

    print(f"\n── 汇总 ──")
    for sym in symbols:
        custom_sym = custom_results.get(sym, [])
        custom_pnl = sum(t["pnl_pts"] for t in custom_sym)

        if tq_shadow is not None and len(tq_shadow) > 0:
            tq_sym = tq_shadow[tq_shadow["symbol"] == sym]
            tq_pnl = tq_sym["pnl_pts"].astype(float).sum() if len(tq_sym) > 0 else 0
            tq_n = len(tq_sym)
        else:
            tq_pnl, tq_n = 0, 0

        print(f"  {sym}:")
        print(f"    自研: {len(custom_sym)}笔 PnL={custom_pnl:+.1f}pt")
        print(f"    TqBT: {tq_n}笔 PnL={tq_pnl:+.1f}pt")
        print(f"    Δ笔数={tq_n-len(custom_sym)} ΔPNL={tq_pnl-custom_pnl:+.1f}pt")

        # 逐笔对比
        if tq_shadow is not None and len(tq_shadow) > 0:
            tq_sym = tq_shadow[tq_shadow["symbol"] == sym].reset_index(drop=True)
            max_n = max(len(custom_sym), len(tq_sym))
            if max_n > 0:
                print(f"    {'#':>3s} | {'自研':^30s} | {'TqBT':^30s}")
                for i in range(max_n):
                    c = custom_sym[i] if i < len(custom_sym) else None
                    t = tq_sym.iloc[i] if i < len(tq_sym) else None
                    c_str = f"{c['entry_time']}→{c['exit_time']} {c['pnl_pts']:+.1f}pt" if c else "(无)"
                    t_str = f"{t['entry_time']}→{t['exit_time']} {float(t['pnl_pts']):+.1f}pt" if t is not None else "(无)"
                    print(f"    {i+1:>3d} | {c_str:^30s} | {t_str:^30s}")

    # 3. 实盘shadow对比
    print(f"\n── vs 实盘Shadow (修复前) ──")
    if existing_shadow is not None and len(existing_shadow) > 0:
        for sym in symbols:
            sym_shadow = existing_shadow[existing_shadow["symbol"] == sym]
            shadow_pnl = sym_shadow["pnl_pts"].astype(float).sum()
            print(f"  {sym}: {len(sym_shadow)}笔 PnL={shadow_pnl:+.1f}pt")
    else:
        print("  (无)")


if __name__ == "__main__":
    main()
