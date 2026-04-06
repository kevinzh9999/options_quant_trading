"""
ADX趋势强度指标独立信息量研究
================================
评估ADX(14)作为日内信号新维度的价值。

分析1：ADX与M/V/Q代理指标的相关性
分析2：ADX对交易结果的预测力（用signal_log回测模拟）
分析3：ADX作为过滤器（不同阈值的效果）
分析4：ADX作为评分维度（乘数 vs 额外分数）
附加：其他指标冗余性检查（MACD/CCI/Williams %R）

数据源：
- index_min: 5分钟K线（symbol/datetime/open/high/low/close/volume）
- signal_log: 信号记录（已含s_momentum/s_volatility/s_quality）
- shadow_trades: 实际交易记录
"""

import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

sys.path.insert(0, "/Users/kevinzhao/Library/CloudStorage/GoogleDrive-kevinzh@gmail.com/我的云端硬盘/options_quant_trading")

from data.storage.db_manager import DBManager, get_db
from config.config_loader import ConfigLoader

# ─────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────
SYMBOL_MAP = {"IM": "000852", "IC": "000905", "IF": "000300", "IH": "000016"}
ADX_PERIOD = 14
MOM_5M_LOOKBACK = 12
ATR_SHORT = 5
ATR_LONG = 40
VOLUME_SURGE_RATIO = 1.5
VOLUME_LOW_RATIO = 0.5

# ─────────────────────────────────────────────
# 指标计算函数
# ─────────────────────────────────────────────

def _wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder平滑（等价于alpha=1/period的EWM）。"""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """计算ADX(period)。使用标准Wilder RMA实现，输出范围0~100。
    输入需要 high/low/close 列。
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # +DM, -DM
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index
    )

    # Wilder平滑（EWM alpha=1/period）
    tr_smooth = _wilder_rma(tr, period)
    plus_dm_smooth = _wilder_rma(plus_dm, period)
    minus_dm_smooth = _wilder_rma(minus_dm, period)

    plus_di = 100 * plus_dm_smooth / tr_smooth.replace(0, np.nan)
    minus_di = 100 * minus_dm_smooth / tr_smooth.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = _wilder_rma(dx.fillna(0), period)

    return adx.clip(0, 100)


def calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """ATR (rolling mean of TR)."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_macd_hist(close: pd.Series) -> pd.Series:
    """MACD histogram (12,26,9)."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def calc_cci(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """CCI(period)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * mad.replace(0, np.nan))


def calc_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R(period). Range -100 to 0."""
    highest_high = df["high"].rolling(period).max()
    lowest_low = df["low"].rolling(period).min()
    wr = -100 * (highest_high - df["close"]) / (highest_high - lowest_low).replace(0, np.nan)
    return wr


def calc_proxy_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算M/V/Q代理指标，与score_all逻辑对齐。

    M_proxy：5分钟动量 × 15分钟动量一致性（对应score_momentum逻辑）
    V_proxy：ATR_SHORT/ATR_LONG比值
    Q_proxy：成交量/MA(20)比值
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    # --- M_proxy ---
    # 12根bar前的收盘价（MOM_5M_LOOKBACK=12）
    mom_5m = (close - close.shift(MOM_5M_LOOKBACK)) / close.shift(MOM_5M_LOOKBACK)

    # 15分钟动量（每3根5分钟bar = 1根15分钟bar，MOM_15M_LOOKBACK=6）
    # 用6根15分钟bar相当于18根5分钟bar
    mom_15m = (close - close.shift(18)) / close.shift(18)

    # 方向一致时 M = abs(mom_5m)，不一致时 M = 0（对应逻辑：方向不一致return 0）
    same_dir = np.sign(mom_5m) == np.sign(mom_15m)
    m_proxy = mom_5m.abs() * same_dir.astype(float)
    # 带方向的版本（供相关性分析）
    m_proxy_signed = mom_5m * same_dir.astype(float)

    # --- V_proxy ---
    atr_s = calc_atr(df, ATR_SHORT)
    atr_l = calc_atr(df, ATR_LONG)
    # 比值越小=低波动=高评分；转换为正向（低波动区间分数高）
    v_ratio = atr_s / atr_l.replace(0, np.nan)
    # V_score: ratio<0.7→30, <0.9→25, <1.1→15, <1.5→5, else→0
    # 用连续化版本：score从ratio的倒数关系近似
    v_proxy = 1.0 / v_ratio.replace(0, np.nan)  # ratio越小→v_proxy越大→高分

    # --- Q_proxy ---
    avg_vol = vol.rolling(20).mean()
    q_ratio = vol / avg_vol.replace(0, np.nan)
    # ratio>1.5→20分，ratio>0.5→10分，else→0
    q_proxy = q_ratio

    return pd.DataFrame({
        "m_proxy_signed": m_proxy_signed,
        "m_proxy": m_proxy,
        "v_proxy": v_proxy,
        "v_ratio": v_ratio,
        "q_proxy": q_proxy,
        "q_ratio": q_ratio,
    }, index=df.index)


# ─────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────

def load_index_min(db: DBManager, symbol_code: str) -> pd.DataFrame:
    """加载指定现货指数的5分钟K线并排序。"""
    df = db.query_df(
        "SELECT datetime, open, high, low, close, volume FROM index_min "
        "WHERE symbol = ? ORDER BY datetime",
        [symbol_code]
    )
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    # 过滤A股交易时段（UTC+8: 09:30-11:30, 13:00-15:00 → UTC: 01:30-03:30, 05:00-07:00）
    # index_min存的是UTC时间（实际是北京时间-8h）
    # 检查time范围
    return df


def load_signal_log(db: DBManager) -> pd.DataFrame:
    """加载signal_log中有完整MVQ数据的记录。"""
    df = db.query_df(
        """SELECT datetime, symbol, direction, score, s_momentum, s_volatility, s_quality,
           z_score, rsi, raw_score, filtered_score, intraday_filter, time_mult, sentiment_mult
           FROM signal_log
           WHERE s_momentum IS NOT NULL AND s_momentum > 0
           ORDER BY datetime"""
    )
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def load_shadow_trades(db: DBManager) -> pd.DataFrame:
    """加载shadow_trades。"""
    df = db.query_df("SELECT * FROM shadow_trades ORDER BY entry_time")
    return df


# ─────────────────────────────────────────────
# 分析函数
# ─────────────────────────────────────────────

def print_table(title: str, df: pd.DataFrame, float_fmt: str = ".3f"):
    """打印格式化表格。"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    pd.set_option("display.float_format", lambda x: f"{x:{float_fmt}}")
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    print(df.to_string(index=True))
    pd.reset_option("display.float_format")


def corr_significance(r: float, n: int) -> str:
    """Pearson r显著性标注。"""
    if n < 4:
        return ""
    t_stat = r * np.sqrt(n - 2) / np.sqrt(max(1e-10, 1 - r**2))
    p = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return f"p={p:.3f}{stars}"


# ─────────────────────────────────────────────
# 分析1：ADX与M/V/Q的相关性
# ─────────────────────────────────────────────

def analysis1_correlation(db: DBManager):
    print("\n" + "="*70)
    print("  分析1：ADX与M/V/Q代理指标的相关性")
    print("="*70)

    results = []
    all_data = {}

    for future_sym, idx_code in SYMBOL_MAP.items():
        df = load_index_min(db, idx_code)
        if len(df) < 200:
            print(f"  {future_sym}: 数据不足（{len(df)}行），跳过")
            continue

        # 计算ADX
        df["adx"] = calc_adx(df, ADX_PERIOD)

        # 计算代理指标
        proxies = calc_proxy_scores(df)
        df = pd.concat([df, proxies], axis=1)

        # 计算其他指标
        df["macd_hist"] = calc_macd_hist(df["close"])
        df["cci"] = calc_cci(df, 14)
        df["williams_r"] = calc_williams_r(df, 14)

        # 去除NaN
        cols = ["adx", "m_proxy", "v_proxy", "q_proxy", "v_ratio", "q_ratio",
                "m_proxy_signed", "macd_hist", "cci", "williams_r"]
        df_clean = df[cols].dropna()

        if len(df_clean) < 50:
            print(f"  {future_sym}: 清洗后数据不足，跳过")
            continue

        n = len(df_clean)
        adx_vals = df_clean["adx"]

        row = {"Symbol": future_sym, "N_bars": n}
        for col in ["m_proxy", "v_proxy", "q_proxy", "macd_hist", "cci", "williams_r"]:
            r, pval = stats.pearsonr(adx_vals, df_clean[col].fillna(0))
            row[f"r_ADX_{col}"] = r
            row[f"p_{col}"] = pval

        results.append(row)
        all_data[future_sym] = df_clean
        print(f"  {future_sym} (000{idx_code}): {n}根bar数据加载完成")

    if not results:
        print("  无有效数据")
        return all_data

    # 相关性表
    corr_cols = ["r_ADX_m_proxy", "r_ADX_v_proxy", "r_ADX_q_proxy",
                 "r_ADX_macd_hist", "r_ADX_cci", "r_ADX_williams_r"]
    p_cols = ["p_m_proxy", "p_v_proxy", "p_q_proxy", "p_macd_hist", "p_cci", "p_williams_r"]

    df_results = pd.DataFrame(results).set_index("Symbol")

    # 格式化输出
    print("\n  Pearson相关系数矩阵（ADX vs 各指标）：")
    print(f"  {'Symbol':<6} {'N':>6} | {'r_M_proxy':>10} {'r_V_proxy':>10} {'r_Q_proxy':>10} | {'r_MACD':>10} {'r_CCI':>10} {'r_WR':>10}")
    print(f"  {'-'*6} {'-'*6}-+-{'-'*10} {'-'*10} {'-'*10}-+-{'-'*10} {'-'*10} {'-'*10}")

    independence_rows = []
    for sym, row in df_results.iterrows():
        n = int(row["N_bars"])
        print(f"  {sym:<6} {n:>6} | {row['r_ADX_m_proxy']:>10.3f} {row['r_ADX_v_proxy']:>10.3f} {row['r_ADX_q_proxy']:>10.3f} | {row['r_ADX_macd_hist']:>10.3f} {row['r_ADX_cci']:>10.3f} {row['r_ADX_williams_r']:>10.3f}")

    # 显著性
    print("\n  P值（*<0.05, **<0.01, ***<0.001）：")
    print(f"  {'Symbol':<6} | {'p_M':>10} {'p_V':>10} {'p_Q':>10} | {'p_MACD':>10} {'p_CCI':>10} {'p_WR':>10}")
    print(f"  {'-'*6}-+-{'-'*10} {'-'*10} {'-'*10}-+-{'-'*10} {'-'*10} {'-'*10}")
    for sym, row in df_results.iterrows():
        def mark(p):
            s = f"{p:.4f}"
            if p < 0.001: s += "***"
            elif p < 0.01: s += "**"
            elif p < 0.05: s += "*"
            return s
        print(f"  {sym:<6} | {mark(row['p_m_proxy']):>10} {mark(row['p_v_proxy']):>10} {mark(row['p_q_proxy']):>10} | {mark(row['p_macd_hist']):>10} {mark(row['p_cci']):>10} {mark(row['p_williams_r']):>10}")

    # ADX分布统计（只用IM，代表性最强）
    print("\n  ADX分布统计（IM 000852，5分钟bar）：")
    if "IM" in all_data:
        im_adx = all_data["IM"]["adx"].dropna()
        bins_adx = [0, 15, 25, 35, 200]
        for lo, hi, label in [(0, 15, "<15(弱趋势)"), (15, 25, "15-25(中等)"),
                               (25, 35, "25-35(强趋势)"), (35, 200, ">35(极强)")]:
            cnt = ((im_adx >= lo) & (im_adx < hi)).sum()
            pct = cnt / len(im_adx) * 100
            print(f"    {label}: {cnt:>6}根 ({pct:.1f}%)")
        print(f"    均值={im_adx.mean():.1f}, 中位数={im_adx.median():.1f}, std={im_adx.std():.1f}")

    print("\n  解读准则：|r|>0.7 高度冗余 | 0.5~0.7 部分冗余 | <0.5 有独立信息 | <0.3 高度独立")

    return all_data


# ─────────────────────────────────────────────
# 分析2：ADX对交易结果的预测力
# ─────────────────────────────────────────────

def analysis2_trade_prediction(db: DBManager, all_data: dict):
    print("\n" + "="*70)
    print("  分析2：ADX对交易结果的预测力")
    print("="*70)

    shadow = load_shadow_trades(db)
    signal_log = load_signal_log(db)

    print(f"\n  shadow_trades: {len(shadow)}笔 | signal_log（有分）: {len(signal_log)}行")

    # ── 方案A：用signal_log的bar时间匹配ADX ──
    print("\n  [方案A] signal_log × ADX配对分析")
    print("  （将每个信号bar时间映射到对应的index_min ADX值）\n")

    # signal_log时间是北京时间，index_min是UTC（北京时间-8h）
    TZ_OFFSET = pd.Timedelta(hours=8)

    matched_rows = []
    for future_sym, idx_code in SYMBOL_MAP.items():
        if future_sym not in all_data:
            continue
        df_idx = all_data[future_sym]
        logs = signal_log[signal_log["symbol"] == future_sym].copy()
        if len(logs) == 0:
            continue

        # 用信号时间（北京时间）转UTC后找最近的index_min bar
        for _, log_row in logs.iterrows():
            sig_dt_bj = log_row["datetime"]
            sig_dt_utc = sig_dt_bj - TZ_OFFSET  # 转UTC
            # 找最近的bar（向前最多15分钟UTC）
            window = df_idx[
                (df_idx.index <= sig_dt_utc + pd.Timedelta(minutes=2)) &
                (df_idx.index >= sig_dt_utc - pd.Timedelta(minutes=10))
            ]
            if len(window) == 0:
                continue
            nearest = window.iloc[-1]
            matched_rows.append({
                "datetime": sig_dt_bj,
                "symbol": future_sym,
                "direction": log_row["direction"],
                "score": log_row["score"],
                "s_momentum": log_row["s_momentum"],
                "s_volatility": log_row["s_volatility"],
                "s_quality": log_row["s_quality"],
                "adx": nearest["adx"],
                "m_proxy": nearest.get("m_proxy", np.nan),
                "v_ratio": nearest.get("v_ratio", np.nan),
                "q_ratio": nearest.get("q_ratio", np.nan),
            })

    if not matched_rows:
        print("  无有效配对数据")
        # 退回到用all_data做ADX统计
    else:
        df_matched = pd.DataFrame(matched_rows).dropna(subset=["adx"])
        print(f"  成功配对: {len(df_matched)}条 (signal有方向: {df_matched['direction'].notna().sum()}条)")

        # ADX vs 实际score的关系
        df_scored = df_matched[df_matched["score"] > 0].copy()
        if len(df_scored) > 0:
            r_adx_score, p_adx_score = stats.pearsonr(df_scored["adx"], df_scored["score"])
            r_adx_mom, p_adx_mom = stats.pearsonr(df_scored["adx"].dropna(),
                                                    df_scored["s_momentum"].dropna())
            print(f"\n  ADX vs 总分(score): r={r_adx_score:.3f}, p={p_adx_score:.4f}")
            print(f"  ADX vs s_momentum:  r={r_adx_mom:.3f}, p={p_adx_mom:.4f}")

    # ── 方案B：按ADX分箱分析signal_log的得分分布 ──
    print("\n  [方案B] ADX分箱 × 信号评分分布")
    print("  （探索高ADX区间是否触发更多/更高质量信号）\n")

    bins = [0, 15, 25, 35, 100]
    bin_labels = ["<15", "15-25", "25-35", ">35"]

    if matched_rows:
        df_matched_dir = df_matched[df_matched["direction"].notna()].copy()
        if len(df_matched_dir) > 0:
            bin_ranges_b = [(0, 15, "<15"), (15, 25, "15-25"), (25, 35, "25-35"), (35, 200, ">35")]
            print(f"  {'ADX区间':<10} {'N':>6} {'Avg得分':>9} {'AvgMom':>9} {'AvgVol':>9} {'AvgQty':>9} {'%Score>=60':>11}")
            print(f"  {'-'*10} {'-'*6} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*11}")
            for lo, hi, lbl in bin_ranges_b:
                sub = df_matched_dir[(df_matched_dir["adx"] >= lo) & (df_matched_dir["adx"] < hi)]
                if len(sub) == 0:
                    continue
                print(f"  {lbl:<10} {len(sub):>6} {sub['score'].mean():>9.1f} {sub['s_momentum'].mean():>9.1f} {sub['s_volatility'].mean():>9.1f} {sub['s_quality'].mean():>9.1f} {(sub['score']>=60).mean()*100:>10.1f}%")

    # ── shadow_trades 分析 ──
    print("\n  [方案C] shadow_trades × ADX（实际成交）")
    shadow_complete = shadow[shadow["pnl_pts"].notna()].copy()
    print(f"  有PnL记录的交易: {len(shadow_complete)}笔")

    if len(shadow_complete) >= 3:
        # 从entry_time匹配ADX（entry_time是北京时间，index_min是UTC）
        matched_shadow = []
        for _, tr in shadow_complete.iterrows():
            sym = tr["symbol"]
            if sym not in all_data:
                continue
            df_idx = all_data[sym]
            td = str(tr["trade_date"])
            entry_t = str(tr["entry_time"])
            # 构建datetime（北京时间）→转UTC
            try:
                dt_str = f"{td[:4]}-{td[4:6]}-{td[6:]} {entry_t}"
                dt_bj = pd.to_datetime(dt_str)
                dt_utc = dt_bj - pd.Timedelta(hours=8)  # 转UTC
                window = df_idx[
                    (df_idx.index <= dt_utc + pd.Timedelta(minutes=2)) &
                    (df_idx.index >= dt_utc - pd.Timedelta(minutes=15))
                ]
                if len(window) > 0:
                    adx_val = window["adx"].iloc[-1]
                    matched_shadow.append({
                        "symbol": sym,
                        "direction": tr["direction"],
                        "pnl_pts": tr["pnl_pts"],
                        "exit_reason": tr["exit_reason"],
                        "adx": adx_val,
                    })
            except Exception as e:
                pass

        if matched_shadow:
            df_sh = pd.DataFrame(matched_shadow).dropna()
            print(f"  成功匹配ADX: {len(df_sh)}笔")
            df_sh["win"] = df_sh["pnl_pts"] > 0

            print(f"\n  {'ADX区间':<10} {'N':>4} {'WR':>8} {'Avg PnL':>10} {'Total PnL':>10}")
            print(f"  {'-'*10} {'-'*4} {'-'*8} {'-'*10} {'-'*10}")
            bin_ranges_sh = [(0, 15, "<15"), (15, 25, "15-25"), (25, 35, "25-35"), (35, 200, ">35")]
            for lo, hi, lbl in bin_ranges_sh:
                sub = df_sh[(df_sh["adx"] >= lo) & (df_sh["adx"] < hi)]
                if len(sub) == 0:
                    continue
                wr_str = f"{sub['win'].mean()*100:.1f}%"
                print(f"  {lbl:<10} {len(sub):>4} {wr_str:>8} {sub['pnl_pts'].mean():>10.1f} {sub['pnl_pts'].sum():>10.1f}")
            # 总计
            wr_str = f"{df_sh['win'].mean()*100:.1f}%"
            print(f"  {'合计':<10} {len(df_sh):>4} {wr_str:>8} {df_sh['pnl_pts'].mean():>10.1f} {df_sh['pnl_pts'].sum():>10.1f}")
            print(f"\n  注：shadow_trades仅9笔，统计意义有限，仅供参考")
        else:
            print("  无法匹配shadow_trades到ADX（时区或时间格式问题）")
            print("  改用IM的signal_log数据分析")


# ─────────────────────────────────────────────
# 分析3：ADX作为过滤器
# ─────────────────────────────────────────────

def analysis3_adx_filter(db: DBManager, all_data: dict):
    print("\n" + "="*70)
    print("  分析3：ADX作为过滤器的效果")
    print("="*70)

    # 用signal_log里的信号+ADX，模拟过滤效果
    # 核心思路：信号发出时，如果ADX低于阈值则不执行
    # 用score>=60的信号作为基础候选（对应当前阈值），看ADX过滤掉多少

    signal_log = load_signal_log(db)
    # 只看IM和IC（实盘品种）
    im_logs = signal_log[(signal_log["symbol"].isin(["IM", "IC"])) & (signal_log["score"] >= 60)].copy()

    print(f"\n  基础候选（score>=60，IM+IC）: {len(im_logs)}条")
    print("  注：shadow_trades数据太少，此处只分析信号分布特征\n")

    # 对每条信号找ADX（signal_log是北京时间，index_min是UTC）
    TZ_OFFSET = pd.Timedelta(hours=8)
    matched = []
    for _, log_row in im_logs.iterrows():
        sig_dt_bj = log_row["datetime"]
        sig_dt_utc = sig_dt_bj - TZ_OFFSET
        sym = log_row["symbol"]
        if sym not in all_data:
            continue
        df_idx = all_data[sym]
        window = df_idx[
            (df_idx.index <= sig_dt_utc + pd.Timedelta(minutes=2)) &
            (df_idx.index >= sig_dt_utc - pd.Timedelta(minutes=10))
        ]
        if len(window) == 0:
            continue
        nearest = window.iloc[-1]
        matched.append({
            "datetime": sig_dt_bj,
            "symbol": sym,
            "score": log_row["score"],
            "direction": log_row["direction"],
            "adx": nearest["adx"],
        })

    if not matched:
        print("  无法匹配信号到ADX")
        return

    df_m = pd.DataFrame(matched).dropna(subset=["adx"])
    print(f"  成功配对: {len(df_m)}条")
    n_total = len(df_m)

    # ADX分布（用数值直接分箱）
    print("\n  触发高评分信号时的ADX分布：")
    print(f"  {'ADX区间':<15} {'N':>6} {'占比':>8}")
    print(f"  {'-'*15} {'-'*6} {'-'*8}")
    for lo, hi, lbl in [(0, 15, "[0,15)"), (15, 20, "[15,20)"), (20, 25, "[20,25)"),
                         (25, 30, "[25,30)"), (30, 200, "[30,+∞)")]:
        cnt = ((df_m["adx"] >= lo) & (df_m["adx"] < hi)).sum()
        if cnt > 0:
            print(f"  {lbl:<15} {cnt:>6} {cnt/n_total*100:>7.1f}%")

    # 模拟不同阈值的过滤效果
    print(f"\n  不同ADX阈值过滤效果（假设过滤掉ADX<阈值的信号）：")
    print(f"\n  {'ADX阈值':<10} {'保留N':>8} {'保留率':>8} {'被过滤N':>10} {'过滤率':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")

    for thr in [15, 20, 25, 30]:
        kept = (df_m["adx"] >= thr).sum()
        filtered_n = n_total - kept
        print(f"  ADX>={thr:<4}   {kept:>8} {kept/n_total*100:>7.1f}%  {filtered_n:>10} {filtered_n/n_total*100:>7.1f}%")

    # ADX与score的关系
    r_adx_score, p_adx_score = stats.pearsonr(df_m["adx"], df_m["score"])
    print(f"\n  触发高分信号的ADX: 均值={df_m['adx'].mean():.1f}, 中位数={df_m['adx'].median():.1f}, std={df_m['adx'].std():.1f}")
    print(f"  ADX与得分相关性: r={r_adx_score:.3f}, p={p_adx_score:.4f}")

    # 对比高ADX vs 低ADX的score分布
    high_adx = df_m[df_m["adx"] >= 25]["score"]
    low_adx = df_m[df_m["adx"] < 25]["score"]
    if len(high_adx) > 0 and len(low_adx) > 0:
        t_stat, t_p = stats.ttest_ind(high_adx, low_adx)
        print(f"\n  高ADX(>=25)信号均分: {high_adx.mean():.1f} (n={len(high_adx)})")
        print(f"  低ADX(<25)信号均分:  {low_adx.mean():.1f} (n={len(low_adx)})")
        print(f"  t检验: t={t_stat:.2f}, p={t_p:.4f} {'(显著差异)' if t_p<0.05 else '(无显著差异)'}")

    print("\n  注意：由于缺乏ADX分组的实际PnL数据，无法量化过滤收益。")
    print("  建议在backtest_signals_day.py中集成ADX过滤后跑完整回测。")


# ─────────────────────────────────────────────
# 分析4：ADX作为评分维度的量化建议
# ─────────────────────────────────────────────

def analysis4_scoring_dimension(db: DBManager, all_data: dict):
    print("\n" + "="*70)
    print("  分析4：ADX作为评分维度的量化分析")
    print("="*70)

    print("""
  方案A：乘数方式（类似daily_mult）
  ─────────────────────────────────
    ADX < 15  →  raw_score × 0.7  （弱趋势，动量不可靠，减分）
    ADX 15~25 →  raw_score × 1.0  （中等，中性）
    ADX 25~35 →  raw_score × 1.15 （强趋势，确认信号，加分）
    ADX > 35  →  raw_score × 1.25 （极强趋势，高置信度）

  方案B：额外分数（类似s_breakout的加分机制）
  ─────────────────────────────────────────────
    ADX < 15  →  -5分
    ADX 15~25 →  +0分
    ADX 25~35 →  +5分
    ADX > 35  →  +10分
    上限：总分不超过100

  方案C：过滤器（硬门槛）
  ─────────────────────────────────
    ADX < 20  →  跳过信号（不管评分多高）
    优点：简单可靠，无超参调整
    缺点：可能过滤掉高质量横盘突破信号
    """)

    # 对现有数据做量化分析
    print("  量化分析：各ADX区间下M/V/Q与ADX的关系")
    print()

    for future_sym in ["IM", "IC"]:
        if future_sym not in all_data:
            continue
        df = all_data[future_sym].copy()
        df = df[["adx", "m_proxy", "v_ratio", "q_ratio"]].dropna()

        # 使用数值分箱而非pandas Categorical，避免标签匹配问题
        bin_ranges = [(0, 15, "<15(弱)"), (15, 25, "15-25(中)"),
                      (25, 35, "25-35(强)"), (35, 200, ">35(极强)")]

        print(f"  {future_sym} (000{SYMBOL_MAP[future_sym]}):")
        print(f"  {'ADX区间':<12} {'N':>6} {'ADX均值':>9} {'M代理均值':>11} {'V比值':>8} {'Q比值':>8} {'高M%':>7} {'放量%':>7}")
        print(f"  {'-'*12} {'-'*6} {'-'*9} {'-'*11} {'-'*8} {'-'*8} {'-'*7} {'-'*7}")
        for lo, hi, label in bin_ranges:
            mask = (df["adx"] >= lo) & (df["adx"] < hi)
            sub = df[mask]
            if len(sub) == 0:
                continue
            avg_adx = sub["adx"].mean()
            avg_m = sub["m_proxy"].mean()
            avg_v = sub["v_ratio"].mean()
            avg_q = sub["q_ratio"].mean()
            pct_high_m = (sub["m_proxy"] > 0.002).mean() * 100
            pct_surge = (sub["q_ratio"] > 1.5).mean() * 100
            print(f"  {label:<12} {len(sub):>6} {avg_adx:>9.1f} {avg_m*100:>10.3f}% {avg_v:>8.3f} {avg_q:>8.3f} {pct_high_m:>6.1f}% {pct_surge:>6.1f}%")
        print()

    # 给出方案建议
    print("  方案优缺点对比：")
    print(f"  {'方案':<6} {'复杂度':<8} {'可解释性':<10} {'风险':<20} {'推荐':<6}")
    print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*20} {'-'*6}")
    print(f"  {'A乘数':<6} {'低':<8} {'高':<10} {'低ADX时过度惩罚':<20} {'★★★☆':<6}")
    print(f"  {'B加分':<6} {'低':<8} {'高':<10} {'与其他加分累加复杂':<20} {'★★★☆':<6}")
    print(f"  {'C过滤':<6} {'最低':<8} {'最高':<10} {'漏信号风险':<20} {'★★☆☆':<6}")


# ─────────────────────────────────────────────
# 附加：完整相关性矩阵
# ─────────────────────────────────────────────

def analysis_extra_indicators(all_data: dict):
    print("\n" + "="*70)
    print("  附加：多指标相关性矩阵（M/V/Q代理 + ADX + MACD + CCI + WR）")
    print("="*70)

    for future_sym in ["IM", "IC"]:
        if future_sym not in all_data:
            continue
        df = all_data[future_sym].dropna()
        if len(df) < 100:
            continue

        # 选用于相关性分析的列
        indicator_cols = {
            "M_proxy": "m_proxy",
            "V_proxy": "v_proxy",
            "Q_ratio": "q_ratio",
            "ADX": "adx",
            "MACD_hist": "macd_hist",
            "CCI": "cci",
            "Williams%R": "williams_r",
        }

        available = {k: v for k, v in indicator_cols.items() if v in df.columns}
        sub = df[[v for v in available.values()]].dropna()
        sub.columns = list(available.keys())
        n = len(sub)

        corr_matrix = sub.corr()

        print(f"\n  {future_sym} 相关性矩阵 (n={n}):")
        print(f"  {'':>12}", end="")
        for col in available.keys():
            print(f" {col:>11}", end="")
        print()
        print(f"  {'-'*12}", end="")
        for col in available.keys():
            print(f" {'-'*11}", end="")
        print()

        for row_name in available.keys():
            print(f"  {row_name:>12}", end="")
            for col_name in available.keys():
                r = corr_matrix.loc[row_name, col_name]
                if row_name == col_name:
                    print(f" {'1.000':>11}", end="")
                else:
                    # 显著性标记
                    t_stat = r * np.sqrt(n - 2) / np.sqrt(max(1e-10, 1 - r**2))
                    p = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))
                    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else " "
                    print(f" {r:>8.3f}{star:>3}", end="")
            print()

        # 找出与M/V/Q最独立的指标
        mvq_cols = ["M_proxy", "V_proxy", "Q_ratio"]
        other_cols = ["ADX", "MACD_hist", "CCI", "Williams%R"]
        print(f"\n  各新指标与M/V/Q的最大相关系数（独立性评估）：")
        print(f"  {'指标':<12} {'max|r|_with_MVQ':>16} {'独立性评估':>12}")
        print(f"  {'-'*12} {'-'*16} {'-'*12}")
        for ind in other_cols:
            if ind not in corr_matrix.index:
                continue
            max_r = max(abs(corr_matrix.loc[ind, c]) for c in mvq_cols if c in corr_matrix.columns)
            independence = "高度独立" if max_r < 0.3 else "部分独立" if max_r < 0.5 else "部分冗余" if max_r < 0.7 else "高度冗余"
            print(f"  {ind:<12} {max_r:>16.3f} {independence:>12}")


# ─────────────────────────────────────────────
# 综合结论
# ─────────────────────────────────────────────

def print_conclusion(all_data: dict):
    print("\n" + "="*70)
    print("  综合结论与建议")
    print("="*70)

    # 计算一些统计用于动态结论
    adx_stats = {}
    for sym, df in all_data.items():
        df_clean = df[["adx", "m_proxy", "v_proxy", "q_ratio"]].dropna()
        if len(df_clean) == 0:
            continue
        adx_vals = df_clean["adx"]
        adx_stats[sym] = {
            "pct_low": (adx_vals < 15).mean() * 100,
            "pct_med": ((adx_vals >= 15) & (adx_vals < 25)).mean() * 100,
            "pct_high": (adx_vals >= 25).mean() * 100,
            "r_adx_m": adx_vals.corr(df_clean["m_proxy"]),
            "r_adx_v": adx_vals.corr(df_clean["v_proxy"]),
            "r_adx_q": adx_vals.corr(df_clean["q_ratio"]),
        }

    print("""
  1. ADX独立性结论
  ─────────────────""")
    for sym, s in adx_stats.items():
        max_r_mvq = max(abs(s["r_adx_m"]), abs(s["r_adx_v"]), abs(s["r_adx_q"]))
        if max_r_mvq < 0.3:
            ind_text = "高度独立（|r|<0.3），有显著增量信息"
        elif max_r_mvq < 0.5:
            ind_text = "部分独立（|r|<0.5），有一定增量信息"
        elif max_r_mvq < 0.7:
            ind_text = "部分冗余（|r|<0.7），增量信息有限"
        else:
            ind_text = "高度冗余（|r|>0.7），建议不引入"
        print(f"  {sym}: max|r|={max_r_mvq:.3f} → {ind_text}")
        print(f"      ADX分布: 弱(<15)={s['pct_low']:.1f}% | 中(15-25)={s['pct_med']:.1f}% | 强(≥25)={s['pct_high']:.1f}%")

    print("""
  2. 实际操作建议
  ─────────────────
  A. 如果ADX与M/V/Q的|r|<0.5：
     → 建议引入ADX作为"方案A乘数"，低ADX时轻罚（×0.85），高ADX时轻加（×1.1）
     → 预期作用：过滤横盘震荡中的假突破，减少MOMENTUM_EXHAUSTED类亏损

  B. 如果ADX主要与V_proxy（ATR比值）高度相关（|r|>0.6）：
     → ADX本质上测量的是同一件事（趋势/波动强度）
     → 建议不引入ADX，而是优化V_proxy的参数（ATR_SHORT/LONG窗口）

  C. 不推荐"硬过滤"（ADX<20全部跳过）：
     → 横盘突破瞬间ADX仍低，这些是最好的入场点
     → 破坏回测统计意义（数据太少时过拟合风险极高）

  3. 下一步验证
  ─────────────────
  → 在backtest_signals_day.py添加adx_filter=True参数
  → 用完整30天干净回测对比：无ADX / ADX乘数 / ADX硬过滤
  → 目标：在IM和IC各增加PnL >50pts或WR>2%才值得引入复杂度
    """)


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

def main():
    print("="*70)
    print("  ADX趋势强度指标独立信息量研究")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    db = get_db()

    # 加载所有品种的index_min数据并计算全部指标
    print("\n  加载数据并计算指标...")
    all_data = {}
    for future_sym, idx_code in SYMBOL_MAP.items():
        df = load_index_min(db, idx_code)
        if len(df) < 200:
            print(f"  {future_sym}: 数据不足")
            continue

        df["adx"] = calc_adx(df, ADX_PERIOD)
        proxies = calc_proxy_scores(df)
        df = pd.concat([df, proxies], axis=1)
        df["macd_hist"] = calc_macd_hist(df["close"])
        df["cci"] = calc_cci(df, 14)
        df["williams_r"] = calc_williams_r(df, 14)

        all_data[future_sym] = df
        n_valid = df["adx"].notna().sum()
        print(f"  {future_sym} (000{idx_code}): {len(df)}根bar, {n_valid}根有效ADX")

    # 执行各项分析
    analysis1_correlation(db)
    analysis2_trade_prediction(db, all_data)
    analysis3_adx_filter(db, all_data)
    analysis4_scoring_dimension(db, all_data)
    analysis_extra_indicators(all_data)
    print_conclusion(all_data)

    print("\n" + "="*70)
    print("  分析完成")
    print("="*70)


if __name__ == "__main__":
    main()
