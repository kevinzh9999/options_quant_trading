#!/usr/bin/env python3
"""日线动量策略 v4：多因子入场评分 + 趋势强度退出 + Vol Targeting。

v3问题：Donchian二值突破入场太晚(平均已涨10%)、无区分度、2021假突破多。
v4改进：四维度连续评分，每个维度0-25分，总分0-100，超阈值入场。

评分维度：
  M 动量方向+强度 (0-25)：5日/10日动量幅度+方向一致性
  T 趋势质量     (0-25)：ADX水平+ADX斜率+ER
  P 价格位置     (0-25)：相对MA和N日范围的位置+突破
  Q 成交量确认   (0-25)：量比+量趋势

退出：v3的趋势强度退出（ADX下降+ER跌破+DI交叉+硬止损）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from collections import defaultdict


def prepare_data(ts_code):
    from data.storage.db_manager import get_db
    db = get_db()
    df = db.query_df(
        f"SELECT trade_date, open, high, low, close, volume FROM index_daily "
        f"WHERE ts_code='{ts_code}' ORDER BY trade_date"
    )
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df.set_index("trade_date")

    n = 14
    tr = pd.concat([df["high"] - df["low"],
                     (df["high"] - df["close"].shift(1)).abs(),
                     (df["low"] - df["close"].shift(1)).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.ewm(span=n, adjust=False).mean()

    plus_dm = df["high"].diff(); minus_dm = -df["low"].diff()
    plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm < 0] = 0
    mask = plus_dm < minus_dm; plus_dm[mask] = 0; minus_dm[~mask] = 0
    plus_di = 100 * plus_dm.ewm(span=n, adjust=False).mean() / df["atr14"]
    minus_di = 100 * minus_dm.ewm(span=n, adjust=False).mean() / df["atr14"]
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    df["adx"] = dx.ewm(span=n, adjust=False).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx_chg3"] = df["adx"] - df["adx"].shift(3)
    df["adx_chg1"] = df["adx"].diff()

    df["er10"] = (df["close"] - df["close"].shift(10)).abs() / \
                 (df["close"].diff().abs().rolling(10).sum() + 1e-10)

    # 动量
    df["mom5"] = df["close"].pct_change(5)
    df["mom10"] = df["close"].pct_change(10)
    df["mom3"] = df["close"].pct_change(3)

    # 均线
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()

    # N日高低
    for w in [5, 10, 20]:
        df[f"high_{w}"] = df["high"].rolling(w).max()
        df[f"low_{w}"] = df["low"].rolling(w).min()

    # 量比
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    df["vol_trend"] = df["volume"].rolling(3).mean() / df["volume"].rolling(10).mean()

    # Vol Targeting用
    df["ret"] = df["close"].pct_change()
    df["vol_20d"] = df["ret"].rolling(20).std() * np.sqrt(252)

    return df


def score_entry(row, prev, direction):
    """计算入场评分。direction='LONG'或'SHORT'。返回总分0-100和各维度分。"""

    # ── M 动量方向+强度 (0-25) ──
    mom5 = prev["mom5"]
    mom10 = prev["mom10"]
    mom3 = prev["mom3"]

    if direction == "LONG":
        m_5 = mom5; m_10 = mom10; m_3 = mom3
    else:
        m_5 = -mom5; m_10 = -mom10; m_3 = -mom3

    # 基础分：5日动量幅度
    if m_5 > 0.04:
        m_base = 15
    elif m_5 > 0.02:
        m_base = 10
    elif m_5 > 0.005:
        m_base = 5
    else:
        m_base = 0

    # 10日确认（同向加分）
    m_confirm = 5 if m_10 > 0.01 else (2 if m_10 > 0 else 0)

    # 短期加速（3日动量与5日同向且更强）
    m_accel = 5 if (m_3 > 0 and m_3 > m_5 * 0.5) else 0

    M = min(25, m_base + m_confirm + m_accel)

    # ── T 趋势质量 (0-25) ──
    adx = prev["adx"]
    adx_chg3 = prev["adx_chg3"]
    er = prev["er10"]

    # ADX水平
    if adx > 35:
        t_adx = 10
    elif adx > 25:
        t_adx = 7
    elif adx > 20:
        t_adx = 4
    else:
        t_adx = 0

    # ADX斜率（趋势加速中）
    t_slope = 5 if adx_chg3 > 2 else (3 if adx_chg3 > 0 else 0)

    # ER
    if er > 0.5:
        t_er = 10
    elif er > 0.3:
        t_er = 7
    elif er > 0.2:
        t_er = 4
    else:
        t_er = 0

    T = min(25, t_adx + t_slope + t_er)

    # ── P 价格位置 (0-25) ──
    close = prev["close"]
    ma10 = prev["ma10"]
    ma20 = prev["ma20"]
    ma50 = prev["ma50"]

    if direction == "LONG":
        # 在均线之上加分
        p_ma = 0
        if close > ma10: p_ma += 3
        if close > ma20: p_ma += 3
        if close > ma50: p_ma += 3
        if ma10 > ma20: p_ma += 3  # 均线多头排列

        # 突破N日高点（越高越好）
        range_20 = prev["high_20"] - prev["low_20"]
        if range_20 > 0:
            pos = (close - prev["low_20"]) / range_20
            p_pos = int(pos * 8)  # 0-8分
        else:
            p_pos = 0

        # 创新高bonus
        p_breakout = 5 if close >= prev["high_20"] else (3 if close >= prev["high_10"] else 0)
    else:
        p_ma = 0
        if close < ma10: p_ma += 3
        if close < ma20: p_ma += 3
        if close < ma50: p_ma += 3
        if ma10 < ma20: p_ma += 3

        range_20 = prev["high_20"] - prev["low_20"]
        if range_20 > 0:
            pos = (prev["high_20"] - close) / range_20
            p_pos = int(pos * 8)
        else:
            p_pos = 0

        p_breakout = 5 if close <= prev["low_20"] else (3 if close <= prev["low_10"] else 0)

    P = min(25, p_ma + p_pos + p_breakout)

    # ── Q 成交量确认 (0-25) ──
    vol_ratio = prev["vol_ratio"]
    vol_trend = prev["vol_trend"]

    # 量比
    if vol_ratio > 1.5:
        q_ratio = 10
    elif vol_ratio > 1.0:
        q_ratio = 5
    else:
        q_ratio = 0

    # 量趋势（近3日vs近10日放量）
    if vol_trend > 1.3:
        q_trend = 10
    elif vol_trend > 1.0:
        q_trend = 5
    else:
        q_trend = 0

    # 突破日放量bonus
    q_breakout = 5 if vol_ratio > 1.2 else 0

    Q = min(25, q_ratio + q_trend + q_breakout)

    total = M + T + P + Q
    return total, {"M": M, "T": T, "P": P, "Q": Q}


def run_backtest_v4(df, params):
    threshold = params.get("threshold", 50)
    adx_thr = params.get("adx_exit_thr", -4)
    er_exit = params.get("er_exit", 0.15)
    hard_stop_atr = params.get("hard_stop_atr", 4.0)
    min_hold = params.get("min_hold", 3)
    max_hold = params.get("max_hold", 30)
    allow_short = params.get("allow_short", True)
    use_vol_target = params.get("use_vol_target", False)
    target_vol = params.get("target_vol", 0.15)
    cost_pct = params.get("cost_pct", 0.00046)

    position = None
    trades = []
    cooldown = 0
    dates = df.index.tolist()

    for i in range(60, len(dates)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        price = row["close"]
        atr = prev["atr14"]
        vol_20d = prev["vol_20d"]

        if cooldown > 0:
            cooldown -= 1

        # ── 持仓管理（v3退出逻辑，adx_thr参数化）──
        if position is not None:
            position["hold_days"] += 1
            d = position["direction"]
            if d == "LONG":
                position["highest"] = max(position["highest"], row["high"])
            else:
                position["lowest"] = min(position["lowest"], row["low"])

            should_exit = False
            reason = ""

            # 硬止损
            if d == "LONG" and price <= position["entry_price"] - hard_stop_atr * atr:
                should_exit, reason = True, "HARD_STOP"
            elif d == "SHORT" and price >= position["entry_price"] + hard_stop_atr * atr:
                should_exit, reason = True, "HARD_STOP"

            # 趋势强度退出（min_hold后）
            if not should_exit and position["hold_days"] >= min_hold:
                if prev["adx_chg3"] < adx_thr:
                    should_exit, reason = True, "ADX_DECLINE"
                elif prev["er10"] < er_exit:
                    should_exit, reason = True, "ER_LOW"
                elif d == "LONG" and prev["minus_di"] > prev["plus_di"]:
                    should_exit, reason = True, "DI_CROSS"
                elif d == "SHORT" and prev["plus_di"] > prev["minus_di"]:
                    should_exit, reason = True, "DI_CROSS"

            if not should_exit and position["hold_days"] >= max_hold:
                should_exit, reason = True, "TIME_STOP"

            if should_exit:
                pnl = (price / position["entry_price"] - 1) if d == "LONG" \
                    else (position["entry_price"] / price - 1)
                pnl -= cost_pct
                vm = position.get("vol_mult", 1.0)
                trades.append({
                    "entry_date": position["entry_date"], "exit_date": dates[i],
                    "direction": d, "entry_price": position["entry_price"],
                    "exit_price": price, "pnl_pct": pnl * vm,
                    "reason": reason, "hold_days": position["hold_days"],
                    "entry_score": position.get("entry_score", 0),
                    "score_detail": position.get("score_detail", {}),
                    "vol_mult": vm,
                })
                position = None
                cooldown = 2
                continue

        # ── 多因子入场 ──
        if position is None and cooldown <= 0:
            best_dir = None
            best_score = 0
            best_detail = {}

            # 评估做多
            score_l, detail_l = score_entry(row, prev, "LONG")
            if score_l >= threshold and prev["plus_di"] > prev["minus_di"]:
                if score_l > best_score:
                    best_dir, best_score, best_detail = "LONG", score_l, detail_l

            # 评估做空
            if allow_short:
                score_s, detail_s = score_entry(row, prev, "SHORT")
                if score_s >= threshold and prev["minus_di"] > prev["plus_di"]:
                    if score_s > best_score:
                        best_dir, best_score, best_detail = "SHORT", score_s, detail_s

            if best_dir:
                vol_mult = 1.0
                if use_vol_target and vol_20d > 0:
                    vol_mult = np.clip(target_vol / vol_20d, 0.3, 3.0)

                position = {
                    "direction": best_dir, "entry_date": dates[i],
                    "entry_price": price, "highest": row["high"],
                    "lowest": row["low"], "hold_days": 0,
                    "vol_mult": vol_mult, "entry_score": best_score,
                    "score_detail": best_detail,
                }

    # EOD
    if position is not None:
        d = position["direction"]; price = df.iloc[-1]["close"]
        pnl = (price / position["entry_price"] - 1) if d == "LONG" \
            else (position["entry_price"] / price - 1)
        pnl -= cost_pct
        vm = position.get("vol_mult", 1.0)
        trades.append({
            "entry_date": position["entry_date"], "exit_date": df.index[-1],
            "direction": d, "entry_price": position["entry_price"],
            "exit_price": price, "pnl_pct": pnl * vm,
            "reason": "END", "hold_days": position["hold_days"],
            "entry_score": position.get("entry_score", 0),
            "score_detail": position.get("score_detail", {}),
            "vol_mult": vm,
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
    payoff = abs(avg_w / avg_l) if avg_l else 0
    total = df["pnl_pct"].sum() * 100
    cum = df["pnl_pct"].cumsum()
    maxdd = (cum - cum.cummax()).min() * 100
    hold = df["hold_days"].mean()
    long_pnl = df[df["direction"] == "LONG"]["pnl_pct"].sum() * 100
    short_pnl = df[df["direction"] == "SHORT"]["pnl_pct"].sum() * 100

    reasons = df.groupby("reason")["pnl_pct"].agg(["count", "sum"])
    reasons["sum"] *= 100

    df["year"] = df["entry_date"].str[:4]
    yearly = df.groupby("year")["pnl_pct"].sum() * 100

    print(f"\n  {label}")
    print(f"  {n}笔 WR={wr:.0f}% AvgW={avg_w:+.2f}% AvgL={avg_l:.2f}% "
          f"Payoff={payoff:.2f} PnL={total:+.1f}% DD={maxdd:.1f}% Hold={hold:.1f}d")
    print(f"  多={long_pnl:+.1f}% 空={short_pnl:+.1f}%")
    print(f"  退出: ", end="")
    for r, row in reasons.iterrows():
        print(f"{r}({int(row['count'])},{row['sum']:+.0f}%) ", end="")
    print()

    # 按入场score分组
    if "entry_score" in df.columns:
        df["score_bin"] = pd.cut(df["entry_score"], bins=[0, 40, 50, 60, 70, 80, 100],
                                  labels=["<40", "40-50", "50-60", "60-70", "70-80", "80+"])
        sg = df.groupby("score_bin", observed=True).agg(
            n=("pnl_pct", "count"),
            pnl=("pnl_pct", lambda x: x.mean() * 100),
            wr=("pnl_pct", lambda x: (x > 0).mean() * 100),
        )
        print(f"  按score: ", end="")
        for sb, row in sg.iterrows():
            if row["n"] > 0:
                print(f"{sb}({int(row['n'])}笔,avg{row['pnl']:+.1f}%,WR{row['wr']:.0f}%) ", end="")
        print()

    print(f"  按年: ", end="")
    for yr, pnl in yearly.items():
        print(f"{yr}:{pnl:+.0f}% ", end="")
    print()

    return {"n": n, "wr": wr, "total": total, "maxdd": maxdd,
            "payoff": payoff, "hold": hold, "label": label}


def main():
    indices = {"IM": "000852.SH", "IC": "000905.SH"}

    configs = [
        # v3 baseline (Donchian入场)
        {"label": "v3 Donchian baseline", "mode": "v3"},
        # v4 不同阈值
        {"label": "v4 thr=40", "mode": "v4", "threshold": 40},
        {"label": "v4 thr=50", "mode": "v4", "threshold": 50},
        {"label": "v4 thr=55", "mode": "v4", "threshold": 55},
        {"label": "v4 thr=60", "mode": "v4", "threshold": 60},
        {"label": "v4 thr=70", "mode": "v4", "threshold": 70},
        # v4 + VT
        {"label": "v4 thr=50 +VT", "mode": "v4", "threshold": 50, "use_vol_target": True},
        {"label": "v4 thr=60 +VT", "mode": "v4", "threshold": 60, "use_vol_target": True},
        # v4 LongOnly
        {"label": "v4 thr=50 LongOnly", "mode": "v4", "threshold": 50, "allow_short": False},
    ]

    combined = defaultdict(lambda: defaultdict(float))

    for sym, ts_code in indices.items():
        df = prepare_data(ts_code)
        print(f"\n{'#'*80}")
        print(f" {sym} ({ts_code}) | v3 vs v4 多因子入场")
        print(f"{'#'*80}")

        summary = []
        for cfg in configs:
            label = f"{sym} {cfg['label']}"
            if cfg.get("mode") == "v3":
                # v3 Donchian baseline
                from scripts.daily_momentum_v3 import run_backtest_v3 as _run_v3
                params = {
                    "exit_mode": "trend_strength", "adx_exit_drop": 3,
                    "er_exit_below": 0.15, "hard_stop_atr": 4.0,
                    "min_hold": 3, "max_hold": 30,
                    "donchian_n": 10, "adx_entry": 20, "er_entry": 0.2,
                    "allow_short": True, "use_vol_target": False,
                }
                # 手动用adx_thr=-4
                position = None; trades = []; cd = 0
                dates = df.index.tolist()
                for i in range(60, len(dates)):
                    row2 = df.iloc[i]; prev2 = df.iloc[i-1]
                    price2 = row2["close"]; atr2 = prev2["atr14"]
                    if cd > 0: cd -= 1
                    if position is not None:
                        position["hold_days"] += 1
                        d = position["direction"]
                        if d=="LONG": position["highest"]=max(position["highest"],row2["high"])
                        else: position["lowest"]=min(position["lowest"],row2["low"])
                        se,sr = False,""
                        if d=="LONG" and price2<=position["entry_price"]-4*atr2: se,sr=True,"HARD_STOP"
                        elif d=="SHORT" and price2>=position["entry_price"]+4*atr2: se,sr=True,"HARD_STOP"
                        if not se and position["hold_days"]>=3:
                            if prev2["adx_chg3"]<-4: se,sr=True,"ADX_DECLINE"
                            elif prev2["er10"]<0.15: se,sr=True,"ER_LOW"
                            elif d=="LONG" and prev2["minus_di"]>prev2["plus_di"]: se,sr=True,"DI_CROSS"
                            elif d=="SHORT" and prev2["plus_di"]>prev2["minus_di"]: se,sr=True,"DI_CROSS"
                        if not se and position["hold_days"]>=30: se,sr=True,"TIME_STOP"
                        if se:
                            pnl=(price2/position["entry_price"]-1)if d=="LONG" else(position["entry_price"]/price2-1)
                            pnl-=0.00046
                            trades.append({"entry_date":position["entry_date"],"exit_date":dates[i],
                                "direction":d,"entry_price":position["entry_price"],"exit_price":price2,
                                "pnl_pct":pnl,"reason":sr,"hold_days":position["hold_days"],
                                "entry_score":0,"score_detail":{},"vol_mult":1.0})
                            position=None; cd=2; continue
                    if position is None and cd<=0:
                        if prev2["adx"]>20 and prev2["er10"]>0.2:
                            if price2>prev2["high_10"] and prev2["plus_di"]>prev2["minus_di"]:
                                position={"direction":"LONG","entry_date":dates[i],"entry_price":price2,
                                    "highest":row2["high"],"lowest":row2["low"],"hold_days":0}
                            elif price2<prev2["low_10"] and prev2["minus_di"]>prev2["plus_di"]:
                                position={"direction":"SHORT","entry_date":dates[i],"entry_price":price2,
                                    "highest":row2["high"],"lowest":row2["low"],"hold_days":0}
                if position:
                    d=position["direction"];p=df.iloc[-1]["close"]
                    pnl=(p/position["entry_price"]-1)if d=="LONG" else(position["entry_price"]/p-1)
                    trades.append({"entry_date":position["entry_date"],"exit_date":df.index[-1],
                        "direction":d,"entry_price":position["entry_price"],"exit_price":p,
                        "pnl_pct":pnl-0.00046,"reason":"END","hold_days":position["hold_days"],
                        "entry_score":0,"score_detail":{},"vol_mult":1.0})
            else:
                params = {
                    "threshold": cfg.get("threshold", 50),
                    "adx_exit_thr": -4, "er_exit": 0.15,
                    "hard_stop_atr": 4.0, "min_hold": 3, "max_hold": 30,
                    "allow_short": cfg.get("allow_short", True),
                    "use_vol_target": cfg.get("use_vol_target", False),
                    "target_vol": 0.15,
                }
                trades = run_backtest_v4(df, params)

            stats = analyze(trades, label)
            stats["label"] = cfg["label"]
            summary.append(stats)

            # 并行PnL
            for t in trades:
                yr = t["entry_date"][:4]
                combined[cfg["label"]][yr] += t["pnl_pct"] * 100

        print(f"\n  {'='*95}")
        print(f"  {sym} 汇总")
        print(f"  {'='*95}")
        print(f"  {'配置':>30s} {'N':>4s} {'WR':>5s} {'PnL':>8s} {'MaxDD':>7s} {'Pay':>5s} {'Hold':>5s}")
        print(f"  {'-'*30} {'-'*4} {'-'*5} {'-'*8} {'-'*7} {'-'*5} {'-'*5}")
        for s in summary:
            if not s: continue
            print(f"  {s['label']:>30s} {s.get('n',0):>4d} {s.get('wr',0):>4.0f}% "
                  f"{s.get('total',0):>+7.1f}% {s.get('maxdd',0):>6.1f}% "
                  f"{s.get('payoff',0):>5.2f} {s.get('hold',0):>4.1f}d")

    # 并行汇总
    print(f"\n\n{'#'*80}")
    print(f" IM+IC 并行年度PnL")
    print(f"{'#'*80}")
    years = sorted(set(y for d in combined.values() for y in d.keys()))
    print(f"  {'配置':>25s}", end="")
    for yr in years: print(f" {yr:>6s}", end="")
    print(f" {'总计':>8s} {'年均':>7s}")
    for cfg in configs:
        lb = cfg["label"]
        print(f"  {lb:>25s}", end="")
        total = 0
        for yr in years:
            v = combined[lb].get(yr, 0)
            print(f" {v:>+5.0f}%", end="")
            total += v
        n_yr = len([y for y in years if combined[lb].get(y,0)!=0])
        print(f" {total:>+7.0f}% {total/max(n_yr,1):>+6.1f}%")


if __name__ == "__main__":
    main()
