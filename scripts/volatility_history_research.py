#!/usr/bin/env python3
"""
volatility_history_research.py
------------------------------
回算完整的波动率指标历史序列，建立分位参照系。

RV (Close-to-Close / Yang-Zhang / Parkinson) + GARCH + ATM IV + VRP
+ 波动率区间分类 + Z-Score均值回归验证 + IV分位择时

用法:
    python scripts/volatility_history_research.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(ROOT) / ".env")
except ImportError:
    pass

from data.storage.db_manager import DBManager, get_db
from config.config_loader import ConfigLoader

OUTPUT_DIR = Path(ROOT) / "logs" / "research"
ANNUALIZE = np.sqrt(252)


# ======================================================================
# Part 1: Realized Volatility
# ======================================================================

def calc_cc_rv(close: pd.Series, window: int) -> pd.Series:
    """Close-to-close RV (annualized %)."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * ANNUALIZE * 100


def calc_yang_zhang(df: pd.DataFrame, window: int) -> pd.Series:
    """Yang-Zhang volatility (annualized %)."""
    o = np.log(df["open"] / df["close"].shift(1))  # overnight
    c = np.log(df["close"] / df["open"])            # open-to-close
    h = np.log(df["high"])
    l = np.log(df["low"])
    co = np.log(df["close"])
    op = np.log(df["open"])

    # Rogers-Satchell
    rs = (h - co) * (h - op) + (l - co) * (l - op)

    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    var_o = o.rolling(n).var()
    var_c = c.rolling(n).var()
    mean_rs = rs.rolling(n).mean()

    yz_var = var_o + k * var_c + (1 - k) * mean_rs
    yz_var = yz_var.clip(lower=0)
    return np.sqrt(yz_var) * ANNUALIZE * 100


def calc_parkinson(df: pd.DataFrame, window: int) -> pd.Series:
    """Parkinson volatility (annualized %)."""
    hl = np.log(df["high"] / df["low"])
    factor = 1 / (4 * np.log(2))
    var_p = (hl ** 2).rolling(window).mean() * factor
    var_p = var_p.clip(lower=0)
    return np.sqrt(var_p) * ANNUALIZE * 100


def build_rv_history(db: DBManager, ts_code: str = "IM.CFX") -> pd.DataFrame:
    """Build full RV history for a given futures contract."""
    df = db.query_df(
        f"SELECT trade_date, open, high, low, close, volume "
        f"FROM futures_daily WHERE ts_code='{ts_code}' "
        f"AND open > 0 AND high > 0 AND low > 0 AND close > 0 "
        f"ORDER BY trade_date"
    )
    if df is None or df.empty:
        return pd.DataFrame()

    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)

    result = df[["trade_date", "close"]].copy()

    for w in [5, 10, 20, 60]:
        result[f"rv_{w}d"] = calc_cc_rv(df["close"], w)
        result[f"yz_{w}d"] = calc_yang_zhang(df, w)

    result["parkinson_20d"] = calc_parkinson(df, 20)

    # Daily log return for GARCH
    result["log_ret"] = np.log(df["close"] / df["close"].shift(1)) * 100  # percent

    return result


# ======================================================================
# Part 1b: GARCH conditional volatility
# ======================================================================

def calc_garch_history(rv_df: pd.DataFrame) -> pd.Series:
    """Fit GJR-GARCH on full history and return conditional vol series."""
    from models.volatility.garch_model import GJRGARCHModel

    returns = rv_df["log_ret"].dropna()
    if len(returns) < 252:
        print(f"  WARNING: only {len(returns)} returns for GARCH")

    model = GJRGARCHModel(dist="skewt")
    result = model.fit(returns)
    print(f"  GARCH fit: persistence={result.persistence:.4f} converged={result.converged}")

    # conditional_vol is already annualized percent from GJRGARCHModel.fit
    cond_vol = result.conditional_vol

    # Re-index to trade_date: cond_vol index matches returns (which is rv_df without NaN)
    # Map back to trade_date
    valid_mask = rv_df["log_ret"].notna()
    trade_dates = rv_df.loc[valid_mask, "trade_date"].values
    cond_vol_mapped = pd.Series(cond_vol.values, index=trade_dates, name="garch_sigma")
    return cond_vol_mapped


# ======================================================================
# Part 2: ATM IV history
# ======================================================================

def build_iv_history(db: DBManager) -> pd.DataFrame:
    """Calculate ATM IV for every trading day from MO options."""
    from models.pricing.implied_vol import calc_implied_vol

    # Get all trade dates with MO options
    dates_df = db.query_df(
        "SELECT DISTINCT trade_date FROM options_daily "
        "WHERE ts_code LIKE 'MO%' AND close > 0 ORDER BY trade_date"
    )
    if dates_df is None or dates_df.empty:
        return pd.DataFrame()

    trade_dates = dates_df["trade_date"].tolist()
    print(f"  Processing {len(trade_dates)} trading days for ATM IV...")

    # Preload IM.CFX closes
    im_df = db.query_df(
        "SELECT trade_date, close FROM futures_daily "
        "WHERE ts_code='IM.CFX' ORDER BY trade_date"
    )
    im_map = {}
    if im_df is not None and not im_df.empty:
        im_map = dict(zip(im_df["trade_date"], im_df["close"].astype(float)))

    # Preload all MO option data with exercise_price (join)
    print("  Loading options data...")
    opts_all = db.query_df(
        "SELECT od.ts_code, od.trade_date, od.close as opt_close, od.volume, "
        "oc.exercise_price, oc.call_put, oc.expire_date "
        "FROM options_daily od "
        "JOIN options_contracts oc ON od.ts_code = oc.ts_code "
        "WHERE od.ts_code LIKE 'MO%' AND od.close > 0"
    )
    if opts_all is None or opts_all.empty:
        print("  WARNING: no options data from join")
        return pd.DataFrame()

    opts_all["exercise_price"] = opts_all["exercise_price"].astype(float)
    opts_all["opt_close"] = opts_all["opt_close"].astype(float)
    print(f"  Loaded {len(opts_all)} option-day records")

    records = []
    batch_count = 0

    for td in trade_dates:
        spot = im_map.get(td)
        if spot is None or spot <= 0:
            continue

        day_opts = opts_all[opts_all["trade_date"] == td].copy()
        if day_opts.empty:
            continue

        # Compute DTE for each option
        day_opts["dte"] = (pd.to_datetime(day_opts["expire_date"]) - pd.Timestamp(td)).dt.days

        # Filter: 14-45 DTE (nearest month with enough time)
        valid = day_opts[(day_opts["dte"] >= 7) & (day_opts["dte"] <= 60)]
        if valid.empty:
            valid = day_opts[day_opts["dte"] >= 3]
        if valid.empty:
            continue

        # Pick the nearest expiry with dte >= 14 (or nearest available)
        ideal = valid[valid["dte"] >= 14]
        if not ideal.empty:
            target_dte = ideal["dte"].min()
        else:
            target_dte = valid["dte"].min()

        month_opts = valid[valid["dte"] == target_dte]
        T = target_dte / 365.0

        # Find ATM strike (closest to spot)
        strikes = month_opts["exercise_price"].unique()
        if len(strikes) == 0:
            continue
        atm_strike = min(strikes, key=lambda k: abs(k - spot))

        # Get call and put at ATM
        atm_call = month_opts[(month_opts["exercise_price"] == atm_strike) &
                              (month_opts["call_put"] == "C")]
        atm_put = month_opts[(month_opts["exercise_price"] == atm_strike) &
                             (month_opts["call_put"] == "P")]

        ivs = []
        for opt_df, cp in [(atm_call, "C"), (atm_put, "P")]:
            if opt_df.empty:
                continue
            price = float(opt_df.iloc[0]["opt_close"])
            if price <= 0:
                continue
            iv = calc_implied_vol(
                market_price=price, S=spot, K=atm_strike,
                T=T, r=0.02, option_type=cp,
            )
            if iv is not None and 0.01 < iv < 3.0:
                ivs.append(iv)

        if ivs:
            atm_iv = np.mean(ivs) * 100  # percent
            records.append({"trade_date": td, "atm_iv": atm_iv})

        batch_count += 1
        if batch_count % 200 == 0:
            print(f"    {batch_count}/{len(trade_dates)} days processed...")

    print(f"  Computed ATM IV for {len(records)} days")
    return pd.DataFrame(records)


# ======================================================================
# Part 3: VRP
# ======================================================================

def analyze_vrp(vol_df: pd.DataFrame) -> str:
    """VRP analysis and distribution statistics."""
    lines = []
    lines.append("\n# 第三部分：VRP历史序列和分位\n")

    if "atm_iv" not in vol_df.columns:
        lines.append("无ATM IV数据\n")
        return "\n".join(lines)

    valid = vol_df.dropna(subset=["atm_iv"])

    for vrp_name, base_col, label in [
        ("vrp_garch", "garch_sigma", "VRP(GARCH) = ATM_IV - GARCH_σ"),
        ("vrp_rv", "rv_20d", "VRP(RV20) = ATM_IV - RV20"),
    ]:
        if base_col not in valid.columns:
            continue
        sub = valid.dropna(subset=[base_col]).copy()
        if sub.empty:
            continue

        sub[vrp_name] = sub["atm_iv"] - sub[base_col]
        vrp = sub[vrp_name]

        lines.append(f"## {label}\n")
        lines.append(f"  样本数: {len(vrp)}")
        lines.append(f"  均值:   {vrp.mean():+.2f}%")
        lines.append(f"  中位数: {vrp.median():+.2f}%")
        lines.append(f"  标准差: {vrp.std():.2f}%")
        lines.append(f"  偏度:   {vrp.skew():.2f}")
        lines.append(f"  峰度:   {vrp.kurtosis():.2f}")
        lines.append(f"  10分位: {vrp.quantile(0.10):+.2f}%")
        lines.append(f"  25分位: {vrp.quantile(0.25):+.2f}%")
        lines.append(f"  50分位: {vrp.quantile(0.50):+.2f}%")
        lines.append(f"  75分位: {vrp.quantile(0.75):+.2f}%")
        lines.append(f"  90分位: {vrp.quantile(0.90):+.2f}%")

        # Current
        latest = sub.iloc[-1]
        current_vrp = latest[vrp_name]
        pct = (vrp <= current_vrp).mean() * 100
        lines.append(f"  当前:   {current_vrp:+.2f}%  历史百分位: {pct:.0f}%")
        lines.append("")

    return "\n".join(lines)


# ======================================================================
# Part 4: Volatility regimes
# ======================================================================

def analyze_regimes(vol_df: pd.DataFrame) -> str:
    """Volatility regime classification."""
    lines = []
    lines.append("\n# 第四部分：波动率区间分类\n")

    if "garch_sigma" not in vol_df.columns:
        lines.append("无GARCH数据\n")
        return "\n".join(lines)

    valid = vol_df.dropna(subset=["garch_sigma"]).copy()
    if valid.empty:
        return "\n".join(lines)

    long_run = valid["garch_sigma"].mean()
    valid["ratio"] = valid["garch_sigma"] / long_run
    valid["log_ret_raw"] = np.log(valid["close"] / valid["close"].shift(1)) * 100

    bins = [
        ("低波动 (<0.8x)", valid["ratio"] < 0.8),
        ("正常 (0.8-1.2x)", (valid["ratio"] >= 0.8) & (valid["ratio"] < 1.2)),
        ("高波动 (1.2-1.5x)", (valid["ratio"] >= 1.2) & (valid["ratio"] < 1.5)),
        ("极端 (>1.5x)", valid["ratio"] >= 1.5),
    ]

    lines.append(f"GARCH长期均值: {long_run:.1f}%\n")
    lines.append(f"{'区间':<20} {'交易日数':>8} {'占比':>6} {'日均收益':>10} {'日均波动':>10}")
    lines.append("-" * 60)

    for label, mask in bins:
        subset = valid[mask]
        days = len(subset)
        pct = days / len(valid) * 100
        avg_ret = subset["log_ret_raw"].mean() if not subset.empty else 0
        avg_vol = subset["garch_sigma"].mean() if not subset.empty else 0
        lines.append(f"{label:<20} {days:>7}天  {pct:>5.1f}%  {avg_ret:>+9.3f}%  {avg_vol:>9.1f}%")

    # Current
    current_ratio = valid.iloc[-1]["ratio"]
    current_garch = valid.iloc[-1]["garch_sigma"]
    lines.append(f"\n当前: GARCH={current_garch:.1f}% / 均值{long_run:.1f}% = {current_ratio:.2f}x")

    if current_ratio < 0.8:
        lines.append("→ **低波动区间**")
    elif current_ratio < 1.2:
        lines.append("→ **正常区间**")
    elif current_ratio < 1.5:
        lines.append("→ **高波动区间**")
    else:
        lines.append("→ **极端高波动区间**")

    return "\n".join(lines)


# ======================================================================
# Part 5: Z-Score mean reversion
# ======================================================================

def analyze_zscore(db: DBManager, vol_df: pd.DataFrame) -> str:
    """Z-Score mean reversion analysis across vol regimes."""
    lines = []
    lines.append("\n# 第五部分：Z-Score均值回归验证\n")

    for ts_code, name in [("IM.CFX", "IM"), ("IF.CFX", "IF"), ("IH.CFX", "IH")]:
        df = db.query_df(
            f"SELECT trade_date, close FROM futures_daily "
            f"WHERE ts_code='{ts_code}' AND close > 0 ORDER BY trade_date"
        )
        if df is None or df.empty:
            continue

        df["close"] = df["close"].astype(float)
        df["ema20"] = df["close"].ewm(span=20).mean()
        df["ema50"] = df["close"].ewm(span=50).mean()
        df["std20"] = df["close"].rolling(20).std()
        df["z20"] = (df["close"] - df["ema20"]) / df["std20"]
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1)) * 100

        # Forward returns
        for h in [5, 10, 20]:
            df[f"fwd_{h}d"] = (df["close"].shift(-h) / df["close"] - 1) * 100

        # GARCH regime (if IM, use vol_df; else compute RV20 as proxy)
        if ts_code == "IM.CFX" and "garch_sigma" in vol_df.columns:
            garch_map = dict(zip(vol_df["trade_date"], vol_df["garch_sigma"]))
            df["garch"] = df["trade_date"].map(garch_map)
        else:
            rv20 = calc_cc_rv(df["close"], 20)
            df["garch"] = rv20

        df = df.dropna(subset=["z20", "garch"])
        if df.empty:
            continue

        long_run = df["garch"].mean()
        df["vol_regime"] = pd.cut(
            df["garch"] / long_run,
            bins=[0, 0.8, 1.2, 1.5, 100],
            labels=["低波动", "正常", "高波动", "极端"],
        )

        lines.append(f"\n## {name} ({ts_code})\n")

        for direction, threshold, signal_name in [
            ("多", -2.0, "Z < -2.0 (超卖)"),
            ("空", 2.0, "Z > +2.0 (超买)"),
        ]:
            if direction == "多":
                sig = df[df["z20"] < threshold]
            else:
                sig = df[df["z20"] > threshold]

            lines.append(f"\n### {signal_name}\n")
            lines.append(
                f"{'波动率区间':<10} {'出现次数':>8} {'后5日':>8} {'后10日':>8} "
                f"{'后20日':>8} {'5日正收益%':>10}"
            )
            lines.append("-" * 65)

            for regime in ["低波动", "正常", "高波动", "极端"]:
                sub = sig[sig["vol_regime"] == regime]
                if len(sub) < 3:
                    continue

                for h_col, h_name in [("fwd_5d", "后5日"), ("fwd_10d", "后10日"), ("fwd_20d", "后20日")]:
                    pass  # computed below

                fwd5 = sub["fwd_5d"].dropna()
                fwd10 = sub["fwd_10d"].dropna()
                fwd20 = sub["fwd_20d"].dropna()

                # For short signals, invert returns
                if direction == "空":
                    fwd5 = -fwd5
                    fwd10 = -fwd10
                    fwd20 = -fwd20

                win5 = (fwd5 > 0).mean() * 100 if not fwd5.empty else 0

                lines.append(
                    f"{regime:<10} {len(sub):>7}次  "
                    f"{fwd5.mean() if not fwd5.empty else 0:>+7.2f}%  "
                    f"{fwd10.mean() if not fwd10.empty else 0:>+7.2f}%  "
                    f"{fwd20.mean() if not fwd20.empty else 0:>+7.2f}%  "
                    f"{win5:>9.0f}%"
                )

            # All regimes
            all_sub = sig
            if not all_sub.empty:
                fwd5 = all_sub["fwd_5d"].dropna()
                fwd10 = all_sub["fwd_10d"].dropna()
                fwd20 = all_sub["fwd_20d"].dropna()
                if direction == "空":
                    fwd5 = -fwd5
                    fwd10 = -fwd10
                    fwd20 = -fwd20
                win5 = (fwd5 > 0).mean() * 100 if not fwd5.empty else 0
                lines.append(
                    f"{'全部':<10} {len(all_sub):>7}次  "
                    f"{fwd5.mean() if not fwd5.empty else 0:>+7.2f}%  "
                    f"{fwd10.mean() if not fwd10.empty else 0:>+7.2f}%  "
                    f"{fwd20.mean() if not fwd20.empty else 0:>+7.2f}%  "
                    f"{win5:>9.0f}%"
                )

    return "\n".join(lines)


# ======================================================================
# Part 6: IV percentile and vol_arb timing
# ======================================================================

def analyze_iv_timing(vol_df: pd.DataFrame) -> str:
    """IV percentile analysis and vol_arb timing."""
    lines = []
    lines.append("\n# 第六部分：IV分位和vol_arb择时\n")

    if "atm_iv" not in vol_df.columns:
        lines.append("无ATM IV数据\n")
        return "\n".join(lines)

    valid = vol_df.dropna(subset=["atm_iv"]).copy()
    if len(valid) < 60:
        lines.append("ATM IV数据不足\n")
        return "\n".join(lines)

    # a. Rolling percentiles
    lines.append("## a. ATM IV 滚动分位\n")
    for lookback in [60, 120, 250]:
        def _pct(s):
            return s.rank(pct=True).iloc[-1] * 100 if len(s) >= lookback else np.nan
        recent = valid.tail(lookback)
        if len(recent) < lookback:
            continue
        current_iv = valid["atm_iv"].iloc[-1]
        pct = (recent["atm_iv"] <= current_iv).mean() * 100
        lines.append(f"  当前ATM IV={current_iv:.2f}% 在过去{lookback}天的百分位: {pct:.0f}%")

    # b. Backtest: sell vol when IV > percentile threshold
    lines.append("\n## b. vol_arb择时回测（卖波动率）\n")
    valid["iv_pct_120"] = valid["atm_iv"].rolling(120).rank(pct=True) * 100
    valid["fwd_5d_ret"] = valid["close"].shift(-5) / valid["close"] - 1

    # Simplified: when IV is high, the next-period VRP capture is better
    # Approximate by checking: days when IV_percentile > threshold, what's the avg VRP
    if "rv_20d" in valid.columns:
        valid["vrp_rv"] = valid["atm_iv"] - valid["rv_20d"]
        valid_vrp = valid.dropna(subset=["iv_pct_120", "vrp_rv"])

        lines.append(f"{'IV分位阈值':<12} {'入场天数':>8} {'平均VRP':>8} {'VRP>0占比':>10} {'平均后5日收益':>12}")
        lines.append("-" * 55)

        for threshold in [50, 60, 75, 90]:
            mask = valid_vrp["iv_pct_120"] >= threshold
            sub = valid_vrp[mask]
            if sub.empty:
                continue
            avg_vrp = sub["vrp_rv"].mean()
            vrp_pos = (sub["vrp_rv"] > 0).mean() * 100
            fwd = sub["fwd_5d_ret"].dropna()
            avg_fwd = fwd.mean() * 100 if not fwd.empty else 0
            lines.append(
                f"IV > P{threshold:<6}  {len(sub):>7}天  {avg_vrp:>+7.2f}%  {vrp_pos:>9.0f}%  {avg_fwd:>+11.3f}%"
            )

    # c. VRP threshold
    lines.append("\n## c. VRP阈值择时\n")
    if "vrp_rv" in valid.columns:
        valid_vrp = valid.dropna(subset=["vrp_rv"])
        lines.append(f"{'VRP阈值':<12} {'入场天数':>8} {'平均后5日':>10} {'后5日>0占比':>10}")
        lines.append("-" * 45)

        for threshold in [0, 1, 2, 3, 5]:
            mask = valid_vrp["vrp_rv"] > threshold
            sub = valid_vrp[mask]
            if sub.empty:
                continue
            fwd = sub["fwd_5d_ret"].dropna()
            avg_fwd = fwd.mean() * 100 if not fwd.empty else 0
            win = (fwd > 0).mean() * 100 if not fwd.empty else 0
            lines.append(
                f"VRP > {threshold}%     {len(sub):>7}天  {avg_fwd:>+9.3f}%  {win:>9.0f}%"
            )

    # d. Current state
    lines.append("\n## d. 当前状态定位\n")
    latest = valid.iloc[-1]
    lines.append(f"  日期: {latest['trade_date']}")
    lines.append(f"  ATM IV: {latest['atm_iv']:.2f}%")

    if "garch_sigma" in valid.columns:
        g = latest.get("garch_sigma")
        if g is not None and not np.isnan(g):
            lines.append(f"  GARCH: {g:.2f}%")

    if "rv_20d" in valid.columns:
        rv = latest.get("rv_20d")
        if rv is not None and not np.isnan(rv):
            lines.append(f"  RV20:  {rv:.2f}%")

    # IV percentile
    iv_vals = valid["atm_iv"].dropna()
    current_iv = latest["atm_iv"]
    pct_all = (iv_vals <= current_iv).mean() * 100
    lines.append(f"  ATM IV百分位(全历史): {pct_all:.0f}%")

    if "vrp_rv" in valid.columns and not np.isnan(latest.get("vrp_rv", np.nan)):
        vrp = latest["vrp_rv"]
        vrp_vals = valid["vrp_rv"].dropna()
        vrp_pct = (vrp_vals <= vrp).mean() * 100
        lines.append(f"  VRP(RV): {vrp:+.2f}%  百分位: {vrp_pct:.0f}%")

    lines.append("")
    if current_iv > iv_vals.quantile(0.75):
        lines.append("  **→ IV偏高（>P75），卖波动率环境**")
    elif current_iv < iv_vals.quantile(0.25):
        lines.append("  **→ IV偏低（<P25），买波动率环境**")
    else:
        lines.append("  **→ IV处于正常区间（P25-P75）**")

    return "\n".join(lines)


# ======================================================================
# Database storage
# ======================================================================

def save_to_db(db: DBManager, vol_df: pd.DataFrame):
    """Save volatility_history to database."""
    # Create table (with spot Z-Score columns)
    db._conn.execute("""
        CREATE TABLE IF NOT EXISTS volatility_history (
            trade_date TEXT PRIMARY KEY,
            close REAL,
            rv_5d REAL, rv_10d REAL, rv_20d REAL, rv_60d REAL,
            yz_5d REAL, yz_10d REAL, yz_20d REAL, yz_60d REAL,
            parkinson_20d REAL,
            garch_sigma REAL,
            atm_iv REAL,
            vrp_garch REAL,
            vrp_rv REAL,
            spot_close REAL,
            spot_ema20 REAL,
            spot_std20 REAL,
            spot_zscore REAL
        )
    """)
    # Add columns if missing (for existing tables)
    for col in ["spot_close", "spot_ema20", "spot_std20", "spot_zscore"]:
        try:
            db._conn.execute(f"ALTER TABLE volatility_history ADD COLUMN {col} REAL")
        except Exception:
            pass
    db._conn.commit()

    # Prepare columns
    cols = ["trade_date", "close"]
    for c in ["rv_5d", "rv_10d", "rv_20d", "rv_60d",
              "yz_5d", "yz_10d", "yz_20d", "yz_60d",
              "parkinson_20d", "garch_sigma", "atm_iv",
              "spot_close", "spot_ema20", "spot_std20", "spot_zscore"]:
        if c in vol_df.columns:
            cols.append(c)

    save_df = vol_df[cols].copy()

    # Compute VRP columns
    if "atm_iv" in save_df.columns and "garch_sigma" in save_df.columns:
        save_df["vrp_garch"] = save_df["atm_iv"] - save_df["garch_sigma"]
    if "atm_iv" in save_df.columns and "rv_20d" in save_df.columns:
        save_df["vrp_rv"] = save_df["atm_iv"] - save_df["rv_20d"]

    # Replace NaN with None for SQLite
    save_df = save_df.where(pd.notna(save_df), None)

    db.upsert_dataframe("volatility_history", save_df)
    print(f"  Saved {len(save_df)} rows to volatility_history table")


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 60)
    print("  波动率历史研究")
    print("=" * 60)
    print()

    db = get_db()
    report_parts = []
    report_parts.append("# 波动率历史研究")
    report_parts.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # Part 1: RV
    print(">>> Part 1: 计算RV历史序列...")
    vol_df = build_rv_history(db, "IM.CFX")
    print(f"  RV history: {len(vol_df)} rows, {vol_df.columns.tolist()}")

    report_parts.append("# 第一部分：已实现波动率历史序列\n")
    if not vol_df.empty:
        latest = vol_df.iloc[-1]
        report_parts.append(f"日期: {latest['trade_date']}")
        for col in ["rv_5d", "rv_10d", "rv_20d", "rv_60d",
                     "yz_5d", "yz_10d", "yz_20d", "yz_60d", "parkinson_20d"]:
            if col in vol_df.columns:
                v = latest.get(col)
                if v is not None and not np.isnan(v):
                    report_parts.append(f"  {col}: {v:.2f}%")

    # Part 1b: GARCH
    print("\n>>> Part 1b: GARCH条件波动率...")
    garch_vol = calc_garch_history(vol_df)
    # garch_vol is indexed by trade_date strings
    vol_df["garch_sigma"] = vol_df["trade_date"].map(garch_vol)

    if not vol_df.empty and "garch_sigma" in vol_df.columns:
        g = vol_df["garch_sigma"].dropna()
        if not g.empty:
            report_parts.append(f"  garch_sigma: {g.iloc[-1]:.2f}% (当前)")
            report_parts.append(f"  garch均值: {g.mean():.2f}%")

    # Part 2: ATM IV
    print("\n>>> Part 2: ATM IV历史序列...")
    iv_df = build_iv_history(db)
    if not iv_df.empty:
        vol_df = vol_df.merge(iv_df, on="trade_date", how="left")
        report_parts.append(f"\n# 第二部分：ATM IV历史序列\n")
        report_parts.append(f"ATM IV计算覆盖: {len(iv_df)} 个交易日")
        iv_vals = iv_df["atm_iv"].dropna()
        if not iv_vals.empty:
            report_parts.append(f"  均值: {iv_vals.mean():.2f}%")
            report_parts.append(f"  P10: {iv_vals.quantile(0.10):.2f}%")
            report_parts.append(f"  P25: {iv_vals.quantile(0.25):.2f}%")
            report_parts.append(f"  P50: {iv_vals.quantile(0.50):.2f}%")
            report_parts.append(f"  P75: {iv_vals.quantile(0.75):.2f}%")
            report_parts.append(f"  P90: {iv_vals.quantile(0.90):.2f}%")
            report_parts.append(f"  当前: {iv_vals.iloc[-1]:.2f}%")

    # Also add vrp columns for later analysis
    if "atm_iv" in vol_df.columns and "rv_20d" in vol_df.columns:
        vol_df["vrp_rv"] = vol_df["atm_iv"] - vol_df["rv_20d"]
    if "atm_iv" in vol_df.columns and "garch_sigma" in vol_df.columns:
        vol_df["vrp_garch"] = vol_df["atm_iv"] - vol_df["garch_sigma"]

    # Spot index Z-Score（无换月跳变，用于盘中Z-Score参照）
    print("\n>>> 计算现货指数Z-Score (000852.SH)...")
    spot_df = db.query_df(
        "SELECT trade_date, close as spot_close FROM index_daily "
        "WHERE ts_code='000852.SH' ORDER BY trade_date"
    )
    if spot_df is not None and not spot_df.empty:
        spot_df["spot_close"] = spot_df["spot_close"].astype(float)
        spot_df["spot_ema20"] = spot_df["spot_close"].ewm(span=20).mean()
        spot_df["spot_std20"] = spot_df["spot_close"].rolling(20).std()
        spot_df["spot_zscore"] = (
            (spot_df["spot_close"] - spot_df["spot_ema20"]) / spot_df["spot_std20"]
        )
        vol_df = vol_df.merge(
            spot_df[["trade_date", "spot_close", "spot_ema20", "spot_std20", "spot_zscore"]],
            on="trade_date", how="left",
        )
        z_now = spot_df["spot_zscore"].iloc[-1]
        print(f"  000852.SH Z-Score: {z_now:+.2f} (latest: {spot_df['trade_date'].iloc[-1]})")

    # Part 3: VRP
    print("\n>>> Part 3: VRP分析...")
    r3 = analyze_vrp(vol_df)
    print(r3)
    report_parts.append(r3)

    # Part 4: Regimes
    print("\n>>> Part 4: 波动率区间...")
    r4 = analyze_regimes(vol_df)
    print(r4)
    report_parts.append(r4)

    # Part 5: Z-Score
    print("\n>>> Part 5: Z-Score均值回归...")
    r5 = analyze_zscore(db, vol_df)
    print(r5)
    report_parts.append(r5)

    # Part 6: IV timing
    print("\n>>> Part 6: IV分位择时...")
    r6 = analyze_iv_timing(vol_df)
    print(r6)
    report_parts.append(r6)

    # Save to DB
    print("\n>>> 保存到数据库...")
    save_to_db(db, vol_df)

    # Save report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"volatility_history_{datetime.now().strftime('%Y%m%d')}.md"
    output_path.write_text("\n".join(report_parts), encoding="utf-8")
    print(f"\n报告已保存: {output_path}")


if __name__ == "__main__":
    main()
