#!/usr/bin/env python3
"""日线动量策略 v3：趋势强度退出 + Vol Targeting。

v2问题诊断：
  - TRAILING_STOP亏-170%：趋势中被价格回撤震出
  - TIME_STOP赚+320%：持仓不动=大赚
  → 退出条件不应该看价格回撤，应该看趋势强度是否衰减

v3改进：
  1. 趋势强度退出：ADX下降N天 / ER跌破阈值 → 趋势结束，平仓
  2. 硬止损保留：极端情况（如闪崩）仍需价格止损兜底，但设很宽（ATR×4）
  3. Vol Targeting：仓位 = 目标波动率 / 实际波动率
  4. 多品种并行：IM+IC独立信号独立仓位
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from collections import defaultdict


def prepare_data(ts_code):
    """加载日线数据+预计算指标。"""
    from data.storage.db_manager import get_db
    db = get_db()
    df = db.query_df(
        f"SELECT trade_date, open, high, low, close, volume FROM index_daily "
        f"WHERE ts_code='{ts_code}' ORDER BY trade_date"
    )
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df.set_index("trade_date")

    # ATR(14)
    n = 14
    tr = pd.concat([df["high"] - df["low"],
                     (df["high"] - df["close"].shift(1)).abs(),
                     (df["low"] - df["close"].shift(1)).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.ewm(span=n, adjust=False).mean()

    # ADX(14) + DI
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    mask = plus_dm < minus_dm
    plus_dm[mask] = 0
    minus_dm[~mask] = 0
    plus_di = 100 * plus_dm.ewm(span=n, adjust=False).mean() / df["atr14"]
    minus_di = 100 * minus_dm.ewm(span=n, adjust=False).mean() / df["atr14"]
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    df["adx"] = dx.ewm(span=n, adjust=False).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    # ADX变化（正=加速，负=衰减）
    df["adx_chg1"] = df["adx"].diff()
    df["adx_chg3"] = df["adx"] - df["adx"].shift(3)

    # ER(10)
    df["er10"] = (df["close"] - df["close"].shift(10)).abs() / \
                 (df["close"].diff().abs().rolling(10).sum() + 1e-10)

    # Donchian
    for w in [5, 10, 15, 20]:
        df[f"high_{w}"] = df["high"].rolling(w).max()
        df[f"low_{w}"] = df["low"].rolling(w).min()

    # 日收益率波动率（用于Vol Targeting）
    df["ret"] = df["close"].pct_change()
    df["vol_20d"] = df["ret"].rolling(20).std() * np.sqrt(252)  # 年化

    return df


def run_backtest_v3(df, params):
    """v3回测：趋势强度退出 + Vol Targeting + 硬止损兜底。"""

    # 入场参数
    donchian_n = params.get("donchian_n", 10)
    adx_entry = params.get("adx_entry", 20)
    er_entry = params.get("er_entry", 0.2)
    allow_short = params.get("allow_short", True)

    # 退出参数
    exit_mode = params.get("exit_mode", "trend_strength")
    # trend_strength模式参数
    adx_exit_drop = params.get("adx_exit_drop", 3)      # ADX连续下降N天则退出
    er_exit_below = params.get("er_exit_below", 0.15)    # ER跌破此值则退出
    hard_stop_atr = params.get("hard_stop_atr", 4.0)     # 硬止损ATR倍数（兜底）
    max_hold = params.get("max_hold", 30)                 # 最大持仓天数（兜底）
    min_hold = params.get("min_hold", 3)                  # 最小持仓天数（避免噪音退出）

    # Vol Targeting
    use_vol_target = params.get("use_vol_target", False)
    target_vol = params.get("target_vol", 0.15)           # 目标年化波动率15%

    cost_pct = params.get("cost_pct", 0.00046)

    high_col = f"high_{donchian_n}"
    low_col = f"low_{donchian_n}"

    position = None
    trades = []
    cooldown = 0

    dates = df.index.tolist()
    for i in range(max(donchian_n, 30) + 1, len(dates)):
        td = dates[i]
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        price = row["close"]
        adx = prev["adx"]
        er = prev["er10"]
        atr = prev["atr14"]
        adx_chg1 = prev["adx_chg1"]
        adx_chg3 = prev["adx_chg3"]
        vol_20d = prev["vol_20d"]

        if cooldown > 0:
            cooldown -= 1

        # ── 持仓管理 ──
        if position is not None:
            position["hold_days"] += 1
            d = position["direction"]

            # 更新极值（硬止损用）
            if d == "LONG":
                position["highest"] = max(position["highest"], row["high"])
            else:
                position["lowest"] = min(position["lowest"], row["low"])

            should_exit = False
            exit_reason = ""

            # 1. 硬止损兜底（极端情况，设很宽）
            if d == "LONG" and price <= position["entry_price"] * (1 - hard_stop_atr * atr / position["entry_price"]):
                should_exit = True
                exit_reason = "HARD_STOP"
            elif d == "SHORT" and price >= position["entry_price"] * (1 + hard_stop_atr * atr / position["entry_price"]):
                should_exit = True
                exit_reason = "HARD_STOP"

            # 2. 趋势强度退出（核心改进，min_hold天后才激活）
            if not should_exit and position["hold_days"] >= min_hold:
                if exit_mode == "trend_strength":
                    # ADX连续下降：趋势正在减弱
                    adx_declining = 0
                    for j in range(1, adx_exit_drop + 1):
                        if i - j >= 0 and df.iloc[i - j]["adx"] > df.iloc[i - j + 1]["adx"] if i - j + 1 < len(df) else False:
                            pass
                        # 简化：用adx_chg3（3天变化）
                    if adx_chg3 < -3:  # ADX 3天下降超过3点
                        should_exit = True
                        exit_reason = "ADX_DECLINE"

                    # ER跌破阈值：方向效率消失
                    if not should_exit and er < er_exit_below:
                        should_exit = True
                        exit_reason = "ER_LOW"

                    # DI交叉反转：趋势方向翻转
                    if not should_exit:
                        if d == "LONG" and prev["minus_di"] > prev["plus_di"]:
                            should_exit = True
                            exit_reason = "DI_CROSS"
                        elif d == "SHORT" and prev["plus_di"] > prev["minus_di"]:
                            should_exit = True
                            exit_reason = "DI_CROSS"

                elif exit_mode == "trailing":
                    # v2的trailing stop（对照组）
                    trail_atr = params.get("trail_atr", 2.5)
                    if d == "LONG":
                        stop = position["highest"] - trail_atr * atr
                        if price <= stop:
                            should_exit = True
                            exit_reason = "TRAILING_STOP"
                    else:
                        stop = position["lowest"] + trail_atr * atr
                        if price >= stop:
                            should_exit = True
                            exit_reason = "TRAILING_STOP"

            # 3. 最大持仓兜底
            if not should_exit and position["hold_days"] >= max_hold:
                should_exit = True
                exit_reason = "TIME_STOP"

            if should_exit:
                if d == "LONG":
                    pnl_pct = (price / position["entry_price"] - 1) - cost_pct
                else:
                    pnl_pct = (position["entry_price"] / price - 1) - cost_pct

                # Vol Targeting影响PnL
                vol_mult = position.get("vol_mult", 1.0)
                pnl_pct_sized = pnl_pct * vol_mult

                trades.append({
                    "entry_date": position["entry_date"], "exit_date": td,
                    "direction": d, "entry_price": position["entry_price"],
                    "exit_price": price, "pnl_pct": pnl_pct_sized,
                    "pnl_pct_raw": pnl_pct,
                    "reason": exit_reason, "hold_days": position["hold_days"],
                    "vol_mult": vol_mult,
                })
                position = None
                cooldown = 2
                continue

        # ── 入场 ──
        if position is None and cooldown <= 0:
            trend_ok = adx > adx_entry and er > er_entry
            prev_high_n = prev[high_col]
            prev_low_n = prev[low_col]

            direction = None
            if trend_ok and price > prev_high_n and prev["plus_di"] > prev["minus_di"]:
                direction = "LONG"
            elif allow_short and trend_ok and price < prev_low_n and prev["minus_di"] > prev["plus_di"]:
                direction = "SHORT"

            if direction:
                # Vol Targeting
                vol_mult = 1.0
                if use_vol_target and vol_20d > 0:
                    vol_mult = target_vol / vol_20d
                    vol_mult = np.clip(vol_mult, 0.3, 3.0)  # 限幅

                position = {
                    "direction": direction, "entry_date": td,
                    "entry_price": price,
                    "highest": row["high"], "lowest": row["low"],
                    "hold_days": 0, "vol_mult": vol_mult,
                }

    # EOD
    if position is not None:
        d = position["direction"]
        price = df.iloc[-1]["close"]
        pnl_pct = (price / position["entry_price"] - 1) - cost_pct if d == "LONG" \
            else (position["entry_price"] / price - 1) - cost_pct
        vol_mult = position.get("vol_mult", 1.0)
        trades.append({
            "entry_date": position["entry_date"], "exit_date": df.index[-1],
            "direction": d, "entry_price": position["entry_price"],
            "exit_price": price, "pnl_pct": pnl_pct * vol_mult,
            "pnl_pct_raw": pnl_pct,
            "reason": "BACKTEST_END", "hold_days": position["hold_days"],
            "vol_mult": vol_mult,
        })

    return trades


def analyze(trades, label=""):
    if not trades:
        print(f"  {label}: 0笔")
        return {}
    df = pd.DataFrame(trades)
    n = len(df)
    wr = (df["pnl_pct"] > 0).mean() * 100
    winners = df[df["pnl_pct"] > 0]
    losers = df[df["pnl_pct"] <= 0]
    avg_w = winners["pnl_pct"].mean() * 100 if len(winners) else 0
    avg_l = losers["pnl_pct"].mean() * 100 if len(losers) else 0
    payoff = abs(avg_w / avg_l) if avg_l else float("inf")
    total = df["pnl_pct"].sum() * 100
    cum = df["pnl_pct"].cumsum()
    maxdd = (cum - cum.cummax()).min() * 100
    hold = df["hold_days"].mean()

    # 多空分解
    long_pnl = df[df["direction"] == "LONG"]["pnl_pct"].sum() * 100
    short_pnl = df[df["direction"] == "SHORT"]["pnl_pct"].sum() * 100

    # 退出原因
    reasons = df.groupby("reason")["pnl_pct"].agg(["count", "sum"])
    reasons["sum"] = reasons["sum"] * 100

    # 年度
    df["year"] = df["entry_date"].str[:4]
    yearly = df.groupby("year").agg(
        n=("pnl_pct", "count"),
        pnl=("pnl_pct", lambda x: x.sum() * 100),
        wr=("pnl_pct", lambda x: (x > 0).mean() * 100),
    )

    print(f"\n  {'='*70}")
    print(f"  {label}")
    print(f"  {'='*70}")
    print(f"  {n}笔 WR={wr:.0f}% AvgW={avg_w:+.2f}% AvgL={avg_l:.2f}% "
          f"Payoff={payoff:.2f} PnL={total:+.1f}% DD={maxdd:.1f}% Hold={hold:.1f}d")
    print(f"  多={long_pnl:+.1f}% 空={short_pnl:+.1f}%")
    print(f"  退出: ", end="")
    for reason, row in reasons.iterrows():
        print(f"{reason}({int(row['count'])}笔,{row['sum']:+.0f}%) ", end="")
    print()
    print(f"  按年: ", end="")
    for yr, row in yearly.iterrows():
        print(f"{yr}:{row['pnl']:+.0f}% ", end="")
    print()

    return {"n": n, "wr": wr, "total": total, "maxdd": maxdd,
            "payoff": payoff, "hold": hold, "label": label}


def main():
    indices = {"IM": "000852.SH", "IC": "000905.SH"}

    configs = [
        # v2 baseline（ATR trailing，对照组）
        {"label": "v2 baseline: trailing ATR2.5",
         "exit_mode": "trailing", "trail_atr": 2.5,
         "donchian_n": 10, "adx_entry": 20, "er_entry": 0.2,
         "max_hold": 15, "allow_short": True, "use_vol_target": False},

        # v3a: 趋势强度退出（核心改进）
        {"label": "v3a: trend_strength exit",
         "exit_mode": "trend_strength",
         "adx_exit_drop": 3, "er_exit_below": 0.15,
         "hard_stop_atr": 4.0, "min_hold": 3, "max_hold": 30,
         "donchian_n": 10, "adx_entry": 20, "er_entry": 0.2,
         "allow_short": True, "use_vol_target": False},

        # v3b: 趋势强度 + 更松min_hold
        {"label": "v3b: trend_str + min_hold=5",
         "exit_mode": "trend_strength",
         "adx_exit_drop": 3, "er_exit_below": 0.15,
         "hard_stop_atr": 4.0, "min_hold": 5, "max_hold": 30,
         "donchian_n": 10, "adx_entry": 20, "er_entry": 0.2,
         "allow_short": True, "use_vol_target": False},

        # v3c: 趋势强度 + Vol Targeting
        {"label": "v3c: trend_str + VolTarget",
         "exit_mode": "trend_strength",
         "adx_exit_drop": 3, "er_exit_below": 0.15,
         "hard_stop_atr": 4.0, "min_hold": 3, "max_hold": 30,
         "donchian_n": 10, "adx_entry": 20, "er_entry": 0.2,
         "allow_short": True, "use_vol_target": True, "target_vol": 0.15},

        # v3d: 全部改进 + LongOnly
        {"label": "v3d: trend_str + VolTarget + LongOnly",
         "exit_mode": "trend_strength",
         "adx_exit_drop": 3, "er_exit_below": 0.15,
         "hard_stop_atr": 4.0, "min_hold": 3, "max_hold": 30,
         "donchian_n": 10, "adx_entry": 20, "er_entry": 0.2,
         "allow_short": False, "use_vol_target": True, "target_vol": 0.15},

        # v3e: 无ADX/ER过滤 + 趋势退出 + VolTarget（最大捕获率）
        {"label": "v3e: no_filter + trend_str + VolTarget",
         "exit_mode": "trend_strength",
         "adx_exit_drop": 3, "er_exit_below": 0.15,
         "hard_stop_atr": 4.0, "min_hold": 3, "max_hold": 30,
         "donchian_n": 10, "adx_entry": 0, "er_entry": 0,
         "allow_short": True, "use_vol_target": True, "target_vol": 0.15},
    ]

    for sym, ts_code in indices.items():
        print(f"\n{'#'*80}")
        print(f" {sym} ({ts_code}) | 日线动量 v3")
        print(f"{'#'*80}")

        df = prepare_data(ts_code)
        for w in [5]:
            if f"high_{w}" not in df.columns:
                df[f"high_{w}"] = df["high"].rolling(w).max()
                df[f"low_{w}"] = df["low"].rolling(w).min()

        summary = []
        for cfg in configs:
            label = f"{sym} {cfg['label']}"
            params = {k: v for k, v in cfg.items() if k != "label"}
            trades = run_backtest_v3(df, params)
            stats = analyze(trades, label)
            stats["label"] = cfg["label"]
            summary.append(stats)

        print(f"\n  {'='*95}")
        print(f"  {sym} 汇总")
        print(f"  {'='*95}")
        print(f"  {'配置':>40s} {'N':>4s} {'WR':>5s} {'PnL':>8s} {'MaxDD':>7s} {'Pay':>5s} {'Hold':>5s}")
        print(f"  {'-'*40} {'-'*4} {'-'*5} {'-'*8} {'-'*7} {'-'*5} {'-'*5}")
        for s in summary:
            if not s:
                continue
            print(f"  {s['label']:>40s} {s.get('n',0):>4d} {s.get('wr',0):>4.0f}% "
                  f"{s.get('total',0):>+7.1f}% {s.get('maxdd',0):>6.1f}% "
                  f"{s.get('payoff',0):>5.2f} {s.get('hold',0):>4.1f}d")

    # ── 多品种并行模拟 ──
    print(f"\n\n{'#'*80}")
    print(f" IM+IC 并行持仓模拟 (v3c: trend_str + VolTarget)")
    print(f"{'#'*80}")

    combined_pnl = defaultdict(float)
    for sym, ts_code in indices.items():
        df = prepare_data(ts_code)
        params = configs[3]  # v3c
        p = {k: v for k, v in params.items() if k != "label"}
        trades = run_backtest_v3(df, p)
        for t in trades:
            yr = t["entry_date"][:4]
            combined_pnl[yr] += t["pnl_pct"] * 100

    print(f"\n  并行年度PnL（IM+IC各1仓位，Vol Targeting）:")
    total_all = 0
    for yr in sorted(combined_pnl.keys()):
        print(f"    {yr}: {combined_pnl[yr]:+.1f}%")
        total_all += combined_pnl[yr]
    n_years = len(combined_pnl)
    print(f"    总计: {total_all:+.1f}%  年均: {total_all/n_years:+.1f}%")


if __name__ == "__main__":
    main()
