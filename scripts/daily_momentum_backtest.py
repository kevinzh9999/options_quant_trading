#!/usr/bin/env python3
"""最简日线动量策略回测。

入场：收盘价创N日新高(做多)/新低(做空) + ADX>25 + ER>0.3
出场：ATR跟踪止损 / 时间止损(max_hold天)
成本：期货平昨万分之0.23（日线策略不涉及平今）

用10年index_daily数据回测IM和IC。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from collections import defaultdict
from multiprocessing import Pool


def prepare_data(ts_code):
    """加载日线数据并预计算所有指标。"""
    from data.storage.db_manager import get_db
    db = get_db()
    df = db.query_df(
        f"SELECT trade_date, open, high, low, close, volume FROM index_daily "
        f"WHERE ts_code='{ts_code}' ORDER BY trade_date"
    )
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df.set_index("trade_date")

    # ADX(14)
    n = 14
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    mask = plus_dm < minus_dm
    plus_dm[mask] = 0
    minus_dm[~mask] = 0
    tr = pd.concat([df["high"] - df["low"],
                     (df["high"] - df["close"].shift(1)).abs(),
                     (df["low"] - df["close"].shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(span=n, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=n, adjust=False).mean() / atr14
    minus_di = 100 * minus_dm.ewm(span=n, adjust=False).mean() / atr14
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    df["adx"] = dx.ewm(span=n, adjust=False).mean()
    df["atr14"] = atr14
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    # ER(10)
    df["er10"] = (df["close"] - df["close"].shift(10)).abs() / \
                 (df["close"].diff().abs().rolling(10).sum() + 1e-10)

    # Donchian channels
    for w in [10, 15, 20, 30]:
        df[f"high_{w}"] = df["high"].rolling(w).max()
        df[f"low_{w}"] = df["low"].rolling(w).min()

    # 5日/10日动量方向
    df["mom5"] = df["close"] / df["close"].shift(5) - 1
    df["mom10"] = df["close"] / df["close"].shift(10) - 1

    return df


def run_backtest(df, params):
    """单次回测，返回交易列表。"""
    donchian_n = params.get("donchian_n", 20)
    adx_min = params.get("adx_min", 25)
    er_min = params.get("er_min", 0.3)
    atr_stop_mult = params.get("atr_stop_mult", 2.0)
    max_hold = params.get("max_hold", 10)
    allow_short = params.get("allow_short", True)
    # 成本：开仓+平仓各万分之0.23，单边0.023%
    cost_pct = params.get("cost_pct", 0.00046)  # 双边

    high_col = f"high_{donchian_n}"
    low_col = f"low_{donchian_n}"

    position = None  # {direction, entry_date, entry_price, stop, highest, lowest, hold_days}
    trades = []

    dates = df.index.tolist()
    for i in range(donchian_n + 14, len(dates)):
        td = dates[i]
        row = df.iloc[i]
        prev = df.iloc[i - 1]  # 用前一天的指标判断（避免lookahead）

        price = row["close"]
        adx = prev["adx"]
        er = prev["er10"]
        atr = prev["atr14"]

        # 前一天的Donchian（用到i-2天的high/low，因为high_N包含当天）
        # 修正：prev的high_N已经包含了prev当天，所以entry条件是今天close > 昨天的N日最高
        prev_high_n = prev[high_col]
        prev_low_n = prev[low_col]

        # ── 持仓管理 ──
        if position is not None:
            position["hold_days"] += 1
            d = position["direction"]

            # 更新极值
            if d == "LONG":
                position["highest"] = max(position["highest"], row["high"])
                # 跟踪止损：从最高点回撤 N*ATR
                new_stop = position["highest"] - atr_stop_mult * atr
                position["stop"] = max(position["stop"], new_stop)
                # 检查止损（用当日最低价）
                if row["low"] <= position["stop"]:
                    exit_price = position["stop"]  # 精确到止损价
                    pnl_pct = (exit_price / position["entry_price"] - 1) - cost_pct
                    trades.append({
                        "entry_date": position["entry_date"], "exit_date": td,
                        "direction": d, "entry_price": position["entry_price"],
                        "exit_price": exit_price, "pnl_pct": pnl_pct,
                        "reason": "TRAILING_STOP", "hold_days": position["hold_days"],
                    })
                    position = None
                    continue
            else:  # SHORT
                position["lowest"] = min(position["lowest"], row["low"])
                new_stop = position["lowest"] + atr_stop_mult * atr
                position["stop"] = min(position["stop"], new_stop)
                if row["high"] >= position["stop"]:
                    exit_price = position["stop"]
                    pnl_pct = (position["entry_price"] / exit_price - 1) - cost_pct
                    trades.append({
                        "entry_date": position["entry_date"], "exit_date": td,
                        "direction": d, "entry_price": position["entry_price"],
                        "exit_price": exit_price, "pnl_pct": pnl_pct,
                        "reason": "TRAILING_STOP", "hold_days": position["hold_days"],
                    })
                    position = None
                    continue

            # 时间止损
            if position["hold_days"] >= max_hold:
                exit_price = price
                if d == "LONG":
                    pnl_pct = (exit_price / position["entry_price"] - 1) - cost_pct
                else:
                    pnl_pct = (position["entry_price"] / exit_price - 1) - cost_pct
                trades.append({
                    "entry_date": position["entry_date"], "exit_date": td,
                    "direction": d, "entry_price": position["entry_price"],
                    "exit_price": exit_price, "pnl_pct": pnl_pct,
                    "reason": "TIME_STOP", "hold_days": position["hold_days"],
                })
                position = None
                continue

        # ── 入场信号（无持仓时） ──
        if position is None:
            trend_confirmed = adx > adx_min and er > er_min

            # 做多：今日收盘 > 昨日N日最高
            if trend_confirmed and price > prev_high_n and prev["plus_di"] > prev["minus_di"]:
                entry_price = price
                stop = entry_price - atr_stop_mult * atr
                position = {
                    "direction": "LONG", "entry_date": td,
                    "entry_price": entry_price, "stop": stop,
                    "highest": row["high"], "lowest": row["low"],
                    "hold_days": 0,
                }

            # 做空：今日收盘 < 昨日N日最低
            elif allow_short and trend_confirmed and price < prev_low_n and prev["minus_di"] > prev["plus_di"]:
                entry_price = price
                stop = entry_price + atr_stop_mult * atr
                position = {
                    "direction": "SHORT", "entry_date": td,
                    "entry_price": entry_price, "stop": stop,
                    "highest": row["high"], "lowest": row["low"],
                    "hold_days": 0,
                }

    # EOD最后一天强制平仓
    if position is not None:
        exit_price = df.iloc[-1]["close"]
        d = position["direction"]
        if d == "LONG":
            pnl_pct = (exit_price / position["entry_price"] - 1) - cost_pct
        else:
            pnl_pct = (position["entry_price"] / exit_price - 1) - cost_pct
        trades.append({
            "entry_date": position["entry_date"], "exit_date": df.index[-1],
            "direction": d, "entry_price": position["entry_price"],
            "exit_price": exit_price, "pnl_pct": pnl_pct,
            "reason": "BACKTEST_END", "hold_days": position["hold_days"],
        })

    return trades


def analyze_trades(trades, label=""):
    """分析交易结果。"""
    if not trades:
        print(f"  {label}: 0笔交易")
        return {}

    df = pd.DataFrame(trades)
    n = len(df)
    winners = df[df["pnl_pct"] > 0]
    losers = df[df["pnl_pct"] <= 0]
    wr = len(winners) / n * 100
    avg_win = winners["pnl_pct"].mean() * 100 if len(winners) > 0 else 0
    avg_loss = losers["pnl_pct"].mean() * 100 if len(losers) > 0 else 0
    payoff = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    total_pct = df["pnl_pct"].sum() * 100
    avg_hold = df["hold_days"].mean()

    # 按年分解
    df["year"] = df["entry_date"].str[:4]
    yearly = df.groupby("year").agg(
        n=("pnl_pct", "count"),
        pnl=("pnl_pct", lambda x: x.sum() * 100),
        wr=("pnl_pct", lambda x: (x > 0).mean() * 100),
    )

    # 最大回撤（逐笔累计）
    cum = df["pnl_pct"].cumsum()
    peak = cum.cummax()
    dd = (cum - peak)
    max_dd = dd.min() * 100

    # 多空分解
    longs = df[df["direction"] == "LONG"]
    shorts = df[df["direction"] == "SHORT"]

    print(f"\n  {'='*70}")
    print(f"  {label}")
    print(f"  {'='*70}")
    print(f"  交易: {n}笔  WR={wr:.1f}%  AvgWin={avg_win:+.2f}%  AvgLoss={avg_loss:.2f}%  Payoff={payoff:.2f}")
    print(f"  总收益: {total_pct:+.1f}%  MaxDD: {max_dd:.1f}%  AvgHold: {avg_hold:.1f}天")
    print(f"  做多: {len(longs)}笔 PnL={longs['pnl_pct'].sum()*100:+.1f}%  "
          f"做空: {len(shorts)}笔 PnL={shorts['pnl_pct'].sum()*100:+.1f}%")

    # 退出原因
    reason_stats = df.groupby("reason").agg(
        n=("pnl_pct", "count"), pnl=("pnl_pct", lambda x: x.sum() * 100))
    print(f"\n  退出原因:")
    for reason, row in reason_stats.iterrows():
        print(f"    {reason:20s}: {int(row['n']):>4d}笔  PnL={row['pnl']:+.1f}%")

    print(f"\n  按年:")
    print(f"    {'年份':>6s} {'笔数':>5s} {'PnL%':>8s} {'WR%':>6s}")
    for yr, row in yearly.iterrows():
        print(f"    {yr:>6s} {int(row['n']):>5d} {row['pnl']:>+7.1f}% {row['wr']:>5.1f}%")

    return {
        "n": n, "wr": wr, "total_pct": total_pct, "max_dd": max_dd,
        "payoff": payoff, "avg_hold": avg_hold,
    }


def main():
    print(f"{'='*80}")
    print(f" 日线动量策略回测 | Donchian突破 + ADX/ER确认 + ATR跟踪止损")
    print(f"{'='*80}")

    indices = {"IM": "000852.SH", "IC": "000905.SH"}

    # 参数网格（小规模，先看基本面貌）
    param_sets = [
        {"label": "D20 ATR2.0 H10",  "donchian_n": 20, "adx_min": 25, "er_min": 0.3,
         "atr_stop_mult": 2.0, "max_hold": 10, "allow_short": True},
        {"label": "D20 ATR2.5 H10",  "donchian_n": 20, "adx_min": 25, "er_min": 0.3,
         "atr_stop_mult": 2.5, "max_hold": 10, "allow_short": True},
        {"label": "D20 ATR3.0 H15",  "donchian_n": 20, "adx_min": 25, "er_min": 0.3,
         "atr_stop_mult": 3.0, "max_hold": 15, "allow_short": True},
        {"label": "D15 ATR2.0 H10",  "donchian_n": 15, "adx_min": 25, "er_min": 0.3,
         "atr_stop_mult": 2.0, "max_hold": 10, "allow_short": True},
        {"label": "D10 ATR2.0 H10",  "donchian_n": 10, "adx_min": 25, "er_min": 0.3,
         "atr_stop_mult": 2.0, "max_hold": 10, "allow_short": True},
        {"label": "D20 ATR2.0 LongOnly", "donchian_n": 20, "adx_min": 25, "er_min": 0.3,
         "atr_stop_mult": 2.0, "max_hold": 10, "allow_short": False},
        # 无ADX/ER过滤的baseline
        {"label": "D20 ATR2.0 NoFilter", "donchian_n": 20, "adx_min": 0, "er_min": 0,
         "atr_stop_mult": 2.0, "max_hold": 10, "allow_short": True},
    ]

    for sym, ts_code in indices.items():
        print(f"\n\n{'#'*80}")
        print(f" {sym} ({ts_code})")
        print(f"{'#'*80}")

        df = prepare_data(ts_code)
        print(f" 数据: {df.index[0]} ~ {df.index[-1]} ({len(df)}天)")

        summary = []
        for ps in param_sets:
            label = f"{sym} {ps['label']}"
            params = {k: v for k, v in ps.items() if k != "label"}
            trades = run_backtest(df, params)
            stats = analyze_trades(trades, label)
            stats["label"] = ps["label"]
            summary.append(stats)

        # 汇总表
        print(f"\n\n  {'='*80}")
        print(f"  {sym} 参数对比汇总")
        print(f"  {'='*80}")
        print(f"  {'配置':>25s} {'笔数':>5s} {'WR%':>6s} {'PnL%':>8s} {'MaxDD%':>8s} "
              f"{'Payoff':>7s} {'AvgHold':>7s}")
        print(f"  {'-'*25} {'-'*5} {'-'*6} {'-'*8} {'-'*8} {'-'*7} {'-'*7}")
        for s in summary:
            if not s:
                continue
            print(f"  {s['label']:>25s} {s.get('n',0):>5d} {s.get('wr',0):>5.1f}% "
                  f"{s.get('total_pct',0):>+7.1f}% {s.get('max_dd',0):>7.1f}% "
                  f"{s.get('payoff',0):>7.2f} {s.get('avg_hold',0):>6.1f}d")


if __name__ == "__main__":
    main()
