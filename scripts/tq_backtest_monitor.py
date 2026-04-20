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
    """用TqBacktest回放指定日期，per-symbol独立跑monitor。
    对每个symbol分别创建TqApi+Monitor实例，避免跨品种干扰。
    """
    from dotenv import load_dotenv
    load_dotenv()

    from tqsdk import TqApi, TqBacktest, BacktestFinished, TqAuth

    if symbols is None:
        symbols = ["IM", "IC"]

    # Per-symbol独立跑，合并结果
    all_trades = []
    for _sym in symbols:
        trades = _run_tq_backtest_single(td, _sym)
        if trades is not None:
            all_trades.append(trades)

    import pandas as pd
    if all_trades:
        return pd.concat(all_trades, ignore_index=True)
    return pd.DataFrame()


def _run_tq_backtest_single(td: str, symbol: str):
    """单品种TqBacktest回放。"""
    from tqsdk import TqApi, TqBacktest, BacktestFinished, TqAuth
    symbols = [symbol]  # 单品种

    y, m, d = int(td[:4]), int(td[4:6]), int(td[6:8])

    # 现货→TQ代码映射
    SPOT_TQ = {"IM": "SSE.000852", "IF": "SSE.000300", "IH": "SSE.000016", "IC": "SSE.000905"}

    # 创建monitor实例，使用独立临时DB（不污染实盘trading.db）
    from strategies.intraday.monitor import IntradayMonitor
    from strategies.intraday.strategy import IntradayConfig
    import tempfile
    _tqbt_db = os.path.join(tempfile.gettempdir(), f"tqbt_{td}.db")

    # Per-symbol monitor（和实盘一致，每品种独立实例）
    # 对每个symbol分别跑一次TqBacktest
    # 这里只创建第一个symbol的monitor，后续在循环中为每个symbol独立跑
    _sym = symbols[0]  # 当前跑的品种（循环外层控制）
    config = IntradayConfig()
    config.universe = {_sym}
    config.tradeable = {_sym} if _sym in {"IM", "IC"} else set()
    config.max_position = 1
    monitor = IntradayMonitor(config, db_path=_tqbt_db)
    # 隔离tmp文件
    _tqbt_tmp = os.path.join(tempfile.gettempdir(), f"tqbt_tmp_{td}_{_sym}")
    os.makedirs(_tqbt_tmp, exist_ok=True)
    monitor._tmp_dir = _tqbt_tmp
    monitor._signal_file = os.path.join(_tqbt_tmp, f"signal_pending_{_sym}.json")

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

    # 用回测日期判断主力合约
    # 优先从futures_daily按OI选主力（和实盘get_main_contract逻辑一致）
    from utils.cffex_calendar import _third_friday, _yymm, active_im_months
    import pandas as _pd
    _bt_date = _pd.Timestamp(f"{td[:4]}-{td[4:6]}-{td[6:]}")
    _active = active_im_months(td)
    # 查前一天的OI确定主力
    _oi_db = _get_db()
    _oi_df = _oi_db.query_df(
        f"SELECT ts_code, oi FROM futures_daily "
        f"WHERE ts_code LIKE 'IM%' AND trade_date <= '{td}' "
        f"ORDER BY trade_date DESC LIMIT 20"
    )
    near_month = None
    if _oi_df is not None and len(_oi_df) > 0:
        # 取最新日期的各合约，找OI最大的（和实盘get_main_contract一致）
        _latest = _oi_df.groupby("ts_code").first().reset_index()
        _latest["oi"] = _latest["oi"].astype(float)
        _latest = _latest.sort_values("oi", ascending=False)
        for _, _r in _latest.iterrows():
            _tc = str(_r["ts_code"])  # e.g. IM2606.CFX
            _m = _tc[2:6]  # 2606
            if _m in _active:
                near_month = _m
                break
    if not near_month:
        # fallback: 到期日逻辑
        _bt_tf = _third_friday(_bt_date.year, _bt_date.month)
        if _bt_date.date() < _bt_tf.date():
            near_month = _yymm(_bt_date.year, _bt_date.month)
        else:
            near_month = _active[1] if len(_active) > 1 else _yymm(_bt_date.year, _bt_date.month + 1)

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
                # 传入真实期货行情，复现实盘monitor的完整价格链路：
                # - entry_price_fut = 期货bid1/ask1（与实际下单一致）
                # - check_exit的current_price = 现货close（monitor内部逻辑）
                # - shadow PnL用期货价（与实盘PnL一致）
                monitor._on_new_bar(
                    changed_syms, spot_klines_5m, spot_klines_15m,
                    fut_quotes, None,
                )

    except BacktestFinished:
        pass
    finally:
        api.close()

    # 从TqBT专用临时DB读取shadow trades
    import sqlite3 as _sql
    _conn = _sql.connect(_tqbt_db)
    import pandas as _pd
    trades_df = _pd.read_sql(
        f"SELECT * FROM shadow_trades WHERE trade_date='{td}' ORDER BY entry_time",
        _conn)
    _conn.close()
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

    # 2. TqBacktest（直接驱动monitor，写入独立临时DB，不污染实盘）
    print("── TqBacktest (直接驱动monitor) ──")
    tq_shadow = run_tq_backtest(td, symbols)

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

    # 3. 实盘shadow对比（从实盘DB读取，不受TqBT影响）
    from data.storage.db_manager import get_db
    db = get_db()
    existing_shadow = db.query_df(
        f"SELECT * FROM shadow_trades WHERE trade_date='{td}'"
    )
    print(f"\n── vs 实盘Shadow ──")
    if existing_shadow is not None and len(existing_shadow) > 0:
        for sym in symbols:
            sym_shadow = existing_shadow[existing_shadow["symbol"] == sym]
            if len(sym_shadow) > 0:
                shadow_pnl = sym_shadow["pnl_pts"].astype(float).sum()
                print(f"  {sym}: {len(sym_shadow)}笔 PnL={shadow_pnl:+.1f}pt")
                for _, r in sym_shadow.iterrows():
                    print(f"    {r['direction']} {r['entry_time']}→{r['exit_time']} "
                          f"{r['exit_reason']} pnl={float(r['pnl_pts']):+.1f}pt")
    else:
        print("  (无)")


if __name__ == "__main__":
    main()
