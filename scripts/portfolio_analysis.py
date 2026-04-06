"""
portfolio_analysis.py
---------------------
分析当前 MO（中证1000股指期权）持仓的完整链路：
  数据层 → 模型层（Greeks / IV / GARCH）→ 信号层（VRP）

数据库：data/storage/trading.db
  - futures_daily : IM.CFX 日线 882 条（2022-07-22 至今）
  - options_daily : MO 期权日线 216184 行（ts_code LIKE 'MO%'，
                    exercise_price/call_put/expire_date 均为 NULL，须从 ts_code 解析）
  - options_contracts : MO 合约基本信息（含 exercise_price / call_put / expire_date）

用法：
    python scripts/portfolio_analysis.py
"""

from __future__ import annotations

import logging
import os
import re
import sys
import warnings
from datetime import date

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 路径设置（允许从任意目录运行）
# ---------------------------------------------------------------------------

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

logging.getLogger("arch").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

TODAY_STR: str       = date.today().strftime("%Y%m%d")
RISK_FREE: float     = 0.02    # 无风险利率（年化）
CONTRACT_MULT: int   = 100     # MO 合约乘数（元/点）
DB_PATH: str         = os.path.join(ROOT, "data", "storage", "trading.db")

# MO ts_code 格式：MO2604-P-7200.CFX
_MO_RE = re.compile(r'^MO(\d{4})-([CP])-(\d+)\.CFX$')


def _parse_mo(ts_code: str) -> tuple[str, str, float] | None:
    """从 ts_code 解析 (expire_month='2604', call_put='P', strike=7200.0)，失败返回 None。"""
    m = _MO_RE.match(str(ts_code))
    return (m.group(1), m.group(2), float(m.group(3))) if m else None


# ---------------------------------------------------------------------------
# 持仓定义（expire_date 在启动时从数据库动态填充）
# ---------------------------------------------------------------------------

_POSITIONS_RAW_FALLBACK = [
    {"ts_code": "MO2604-P-7200.CFX", "volume": -4},
    {"ts_code": "MO2605-P-7200.CFX", "volume": -2},
    {"ts_code": "MO2606-C-7800.CFX", "volume": 10},
    {"ts_code": "MO2606-C-8400.CFX", "volume": -12},
]


def _load_positions_from_db(db) -> list[dict] | None:
    """
    从 position_snapshots 表读取最新 trade_date 的持仓记录，
    筛选 MO 期权，转换 TQ 格式 → Tushare 格式。

    返回 [{"ts_code": "MO2604-P-7800.CFX", "volume": -4}, ...] 或 None（表为空）。
    """
    if db is None:
        return None
    try:
        latest = db.query_df(
            "SELECT MAX(trade_date) as dt FROM position_snapshots"
        )
        if latest is None or latest.empty or latest["dt"].iloc[0] is None:
            return None
        trade_date = str(latest["dt"].iloc[0])
        rows = db.query_df(
            "SELECT symbol, direction, volume FROM position_snapshots "
            "WHERE trade_date = ?",
            params=(trade_date,),
        )
        if rows is None or rows.empty:
            return None
    except Exception as e:
        print(f"  [警告] 读取 position_snapshots 失败: {e}")
        return None

    print(f"  position_snapshots 最新日期: {trade_date}，共 {len(rows)} 条记录")

    # TQ 格式: CFFEX.MO2604-P-7800 → Tushare: MO2604-P-7800.CFX
    tq_mo_re = re.compile(r'^CFFEX\.(MO\d{4}-[CP]-\d+)$')
    positions_raw: list[dict] = []
    for _, row in rows.iterrows():
        sym = str(row["symbol"])
        m = tq_mo_re.match(sym)
        if m:
            ts_code = m.group(1) + ".CFX"
            vol = int(row["volume"])
            if str(row["direction"]).upper() == "SHORT":
                vol = -abs(vol)
            else:
                vol = abs(vol)
            positions_raw.append({"ts_code": ts_code, "volume": vol})
        else:
            if "MO" not in sym:
                print(f"  [跳过] 非MO期权: {sym}")
            else:
                print(f"  [警告] 无法解析MO合约: {sym}")

    return positions_raw if positions_raw else None


# ---------------------------------------------------------------------------
# DB 连接
# ---------------------------------------------------------------------------

def _open_db():
    if not os.path.exists(DB_PATH):
        return None
    try:
        from data.storage.db_manager import get_db
        return get_db()
    except Exception as e:
        print(f"  [警告] 无法连接数据库: {e}")
        return None


# ---------------------------------------------------------------------------
# 辅助：从 options_contracts 建立到期月 → 到期日映射
# ---------------------------------------------------------------------------

def _build_expire_map(db) -> dict[str, str]:
    """返回 {'2604': '20260417', '2605': '20260515', ...}"""
    try:
        df = db.query_df(
            "SELECT DISTINCT ts_code, expire_date "
            "FROM options_contracts WHERE ts_code LIKE 'MO%'"
        )
        if df is None or df.empty:
            return {}
        result: dict[str, str] = {}
        for _, row in df.iterrows():
            parsed = _parse_mo(str(row["ts_code"]))
            if parsed and row["expire_date"]:
                result[parsed[0]] = str(row["expire_date"])
        return result
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# 辅助：获取最新 MO 期权链（解析 ts_code，补充 exercise_price/call_put/expire_date）
# ---------------------------------------------------------------------------

def _get_mo_chain(db, trade_date: str, expire_map: dict[str, str]) -> pd.DataFrame:
    """
    返回最新 MO 期权链 DataFrame，列：
      ts_code, call_put, exercise_price, expire_date, expire_month, close, volume, oi
    """
    try:
        df = db.query_df(
            f"SELECT ts_code, close, settle, volume, oi "
            f"FROM options_daily "
            f"WHERE ts_code LIKE 'MO%' AND trade_date='{trade_date}' AND close > 0"
        )
        if df is None or df.empty:
            return pd.DataFrame()
    except Exception as e:
        print(f"  [警告] 读取期权链失败: {e}")
        return pd.DataFrame()

    records = []
    for _, row in df.iterrows():
        parsed = _parse_mo(str(row["ts_code"]))
        if parsed is None:
            continue
        expire_month, cp, strike = parsed
        expire_date = expire_map.get(expire_month, "")
        if not expire_date:
            continue   # 没有到期日则跳过
        records.append({
            "ts_code":        row["ts_code"],
            "expire_month":   expire_month,
            "expire_date":    expire_date,
            "call_put":       cp,
            "exercise_price": strike,
            "close":          float(row["close"]),
            "volume":         float(row["volume"]) if row["volume"] else 0.0,
            "oi":             float(row["oi"])     if row["oi"]     else 0.0,
        })

    return pd.DataFrame(records) if records else pd.DataFrame()


# ---------------------------------------------------------------------------
# 持仓列表构建（从 DB 填充 expire_date / strike / call_put）
# ---------------------------------------------------------------------------

def build_positions(expire_map: dict[str, str], positions_raw: list[dict] | None = None) -> list[dict]:
    """
    基于持仓列表和到期日映射，构建含完整信息的持仓列表。

    Parameters
    ----------
    expire_map : dict
        到期月 → 到期日映射，如 {'2604': '20260417'}
    positions_raw : list[dict] | None
        持仓列表，如 [{"ts_code": "MO2604-P-7800.CFX", "volume": -4}]。
        如果为 None，使用硬编码的 _POSITIONS_RAW_FALLBACK。
    """
    if positions_raw is None:
        positions_raw = _POSITIONS_RAW_FALLBACK
    positions = []
    for raw in positions_raw:
        parsed = _parse_mo(raw["ts_code"])
        if parsed is None:
            print(f"  [警告] 无法解析合约代码: {raw['ts_code']}")
            continue
        expire_month, cp, strike = parsed
        expire_date = expire_map.get(expire_month, "")
        if not expire_date:
            print(f"  [警告] options_contracts 中未找到 {raw['ts_code']} 的到期日，跳过")
            continue
        positions.append({
            "ts_code":       raw["ts_code"],
            "strike_price":  strike,
            "call_put":      cp,
            "expire_date":   expire_date,
            "volume":        raw["volume"],
            "contract_unit": CONTRACT_MULT,
        })
    return positions


# ---------------------------------------------------------------------------
# 辅助：查询各到期月份对应期货合约收盘价（使用 utils.cffex_calendar）
# ---------------------------------------------------------------------------

def get_futures_prices_by_expiry(
    db,
    expire_months: list[str],
    fallback_spot: float,
    trade_date: str | None = None,
) -> dict[str, float]:
    """
    为每个期权到期月份查询对应的 IM 期货收盘价。

    策略：
      1. 通过 utils.cffex_calendar.get_im_futures_prices 查询活跃合约价格
         - 优先查具体合约（IM2603.CFX 等），若无则用 IML 连续合约
         - IML 正确映射：IM.CFX=当月, IML1=次月, IML2=当季, IML3=下季
      2. 将每个 expire_month 映射到最近的（≥ 该月份的）IM 合约
      3. 找不到则回退到 fallback_spot（主力合约价）

    返回：{'2603': 8014.0, '2604': 7951.0, '2605': 7770.8, '2606': 7770.8}
    """
    from utils.cffex_calendar import get_im_futures_prices, map_expiry_to_futures_price

    if not trade_date:
        trade_date = date.today().strftime("%Y%m%d")

    im_prices = get_im_futures_prices(db, trade_date)

    if not im_prices:
        print(f"  [警告] 无法获取 IM 合约价格，全部回退到主力合约价 {fallback_spot:.2f}")
        return {m: fallback_spot for m in expire_months}

    mapped = map_expiry_to_futures_price(expire_months, im_prices, fallback_spot)

    result: dict[str, float] = {}
    for opt_m in expire_months:
        im_month, price = mapped[opt_m]
        # Determine which IML code this corresponds to for display
        from utils.cffex_calendar import active_im_months, _IML_CODES
        active = active_im_months(trade_date)
        if im_month in active:
            idx = active.index(im_month)
            iml_code = _IML_CODES[idx]
        else:
            iml_code = im_month  # specific contract like IM2603.CFX
        result[opt_m] = price
        suffix = f"（≥ MO{opt_m}）" if im_month != opt_m else ""
        print(f"  MO{opt_m} → IM{im_month} → {iml_code} = {price:.2f}{suffix}")

    return result


# ---------------------------------------------------------------------------
# 辅助：为每个到期月份计算 PCP 隐含 Forward Price
# ---------------------------------------------------------------------------

def calc_implied_forwards_by_expiry(
    chain_df: pd.DataFrame,
    futures_prices: dict[str, float],
    spot: float,
) -> dict[str, float]:
    """
    对 chain_df 中每个 expire_month 调用 PCP 反推 Forward Price。

    Parameters
    ----------
    chain_df : pd.DataFrame
        包含所有到期月份的期权链（需要 expire_month、exercise_price、
        call_put、close、volume、expire_date 列）。
    futures_prices : dict[str, float]
        月份 → 期货收盘价（来自 get_futures_prices_by_expiry）。
    spot : float
        主力合约价，作为无对应期货价格时的兜底。

    Returns
    -------
    dict[str, float]
        月份 → 隐含 Forward Price。若 PCP 无法计算则回退到期货收盘价。
    """
    from models.pricing.forward_price import calc_implied_forward

    today_ts = pd.Timestamp.today().normalize()
    implied_forwards: dict[str, float] = {}

    for em in sorted(chain_df["expire_month"].unique()):
        sub = chain_df[chain_df["expire_month"] == em].copy()
        exp_date = sub["expire_date"].iloc[0]
        days_left = (pd.Timestamp(exp_date) - today_ts).days
        T = max(days_left / 365.0, 1.0 / 365)
        futures_close = futures_prices.get(em, spot)

        fwd, n = calc_implied_forward(sub, T, RISK_FREE, futures_close)
        implied_forwards[em] = fwd

        if n > 0:
            diff = fwd - futures_close
            print(f"  MO{em} 隐含Forward={fwd:.2f}"
                  f"  (vs 期货 {futures_close:.2f}, 差异 {diff:+.2f}, {n} 个行权价 PCP 估计)")
        else:
            print(f"  MO{em} PCP 无法计算，使用期货收盘价 {futures_close:.2f}")

    return implied_forwards


# ---------------------------------------------------------------------------
# Step 1：获取 IM.CFX 最新收盘价
# ---------------------------------------------------------------------------

def get_spot_price(db) -> tuple[float, str]:
    """获取现货价格：优先 000852.SH（中证1000指数），回退到 IM.CFX。"""
    # 优先：中证1000现货指数（与 vol_monitor 一致）
    try:
        row = db.query_df(
            "SELECT close, trade_date FROM index_daily "
            "WHERE ts_code='000852.SH' ORDER BY trade_date DESC LIMIT 1"
        )
        if row is not None and not row.empty:
            price = float(row["close"].iloc[0])
            trade_date = str(row["trade_date"].iloc[0])
            return price, f"000852.SH {trade_date}"
    except Exception as e:
        print(f"  [警告] 获取 000852.SH 价格失败: {e}")

    # 回退：IM.CFX 主力期货
    try:
        row = db.query_df(
            "SELECT close, trade_date FROM futures_daily "
            "WHERE ts_code='IM.CFX' ORDER BY trade_date DESC LIMIT 1"
        )
        if row is not None and not row.empty:
            price = float(row["close"].iloc[0])
            trade_date = str(row["trade_date"].iloc[0])
            return price, f"IM.CFX {trade_date}"
    except Exception as e:
        print(f"  [警告] 获取 IM.CFX 价格失败: {e}")
    raise RuntimeError("无法获取现货价格（已尝试 000852.SH 和 IM.CFX），请检查数据库")


# ---------------------------------------------------------------------------
# Step 2：从市场价格计算各持仓 IV
# ---------------------------------------------------------------------------

def get_iv_for_positions(
    positions: list[dict],
    spot: float,
    chain_df: pd.DataFrame,
    futures_prices: dict[str, float] | None = None,
    implied_forwards: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    返回 {ts_code: iv}。
    用 chain_df 中的市场收盘价反算 IV；如某合约无法计算则跳过（打印警告）。

    标的价优先级（由高到低）：
      1. implied_forwards[expire_month]   — PCP 反推隐含 Forward
      2. futures_prices[expire_month]     — 对应到期月份期货收盘价
      3. spot                             — 主力合约价（兜底）
    """
    from models.pricing.implied_vol import calc_implied_vol

    iv_map: dict[str, float] = {}
    today_ts = pd.Timestamp.today().normalize()

    # 用 ts_code 做精确匹配索引
    if not chain_df.empty:
        chain_idx = chain_df.set_index("ts_code")

    for pos in positions:
        ts = pos["ts_code"]
        expire_ts = pd.Timestamp(pos["expire_date"])
        T = max((expire_ts - today_ts).days / 365.0, 1.0 / 365)

        parsed = _parse_mo(ts)
        expire_month = parsed[0] if parsed else None

        # 优先使用 PCP 隐含 Forward，其次期货收盘价，最后主力价
        if implied_forwards and expire_month and expire_month in implied_forwards:
            S = implied_forwards[expire_month]
            s_src = "隐含Forward"
        elif futures_prices and expire_month:
            S = futures_prices.get(expire_month, spot)
            s_src = "期货收盘"
        else:
            S = spot
            s_src = "主力"

        try:
            if chain_df.empty or ts not in chain_idx.index:
                raise ValueError(f"期权链中未找到 {ts}")

            mkt_price = float(chain_idx.loc[ts, "close"])
            if mkt_price <= 0:
                raise ValueError(f"市场价格无效: {mkt_price}")

            iv = calc_implied_vol(
                market_price=mkt_price,
                S=S,
                K=pos["strike_price"],
                T=T,
                r=RISK_FREE,
                option_type=pos["call_put"],
            )
            if iv is None or iv <= 0 or iv > 5.0:
                raise ValueError(f"IV 超出合理范围: {iv}")

            iv_map[ts] = iv
            print(f"  {ts:<28}  [{s_src}]标的={S:>8.2f}  市场价={mkt_price:>7.1f}  IV={iv*100:>6.2f}%  T={T:.3f}y")

        except Exception as e:
            print(f"  [警告] {ts}: IV 计算失败（{e}），跳过该合约")

    return iv_map


# ---------------------------------------------------------------------------
# Step 3：从期权链提取 ATM IV（最近期 + 认购认沽平均）
# ---------------------------------------------------------------------------

def get_atm_iv(
    spot: float,
    chain_df: pd.DataFrame,
    trade_date: str,
    futures_prices: dict[str, float] | None = None,
) -> tuple[float, str]:
    """
    从 chain_df 中选合适的到期月，取最近行权价的认购+认沽 IV 作为 ATM IV。

    选月规则：
    - 排除剩余到期天数 < 14 天的合约（临近到期 Gamma 扭曲 IV）
    - 优先选剩余 14-45 天的最近月（当月合约）
    - 若无，使用距今最近的有效月（次月等）
    C/P 发散处理：
    - Call IV 与 Put IV 差值 > 10pp 时打印警告，仅保留成交量更大的一侧
    futures_prices: expire_month → futures_close
    """
    from models.pricing.implied_vol import calc_implied_vol

    if chain_df.empty:
        raise RuntimeError("期权链为空，无法计算 ATM IV")

    today_ts = pd.Timestamp.today().normalize()

    # ── 1. 为每个到期月计算剩余天数，过滤 < 14 天 ─────────────────────
    sorted_expires = sorted(chain_df["expire_month"].unique())
    valid_months = []
    for em in sorted_expires:
        sub = chain_df[chain_df["expire_month"] == em]
        exp_date = sub["expire_date"].iloc[0]
        days_left = (pd.Timestamp(exp_date) - today_ts).days
        if days_left >= 14:
            valid_months.append((em, exp_date, days_left))

    if not valid_months:
        raise RuntimeError("所有到期月剩余天数均 < 14 天，无法计算 ATM IV")

    # ── 2. 选月：优先 14-45 天，否则取最近有效月 ──────────────────────
    preferred = [(em, ed, d) for em, ed, d in valid_months if d <= 45]
    chosen_month, expire_date, days_left = (preferred[0] if preferred
                                             else valid_months[0])

    print(f"  选用到期月: MO{chosen_month}（{expire_date} 到期，剩余 {days_left} 天）")

    near_df = chain_df[chain_df["expire_month"] == chosen_month].copy()
    T = max(days_left / 365.0, 1.0 / 365)

    # 期货收盘价（初始估计，用于筛选 PCP 候选行权价范围）
    futures_close = (futures_prices.get(chosen_month, spot)
                     if futures_prices else spot)

    # ── 3. Put-Call Parity 反推隐含 Forward Price ─────────────────────
    # PCP（期货标的）: C - P = e^(-rT) * (F - K)  →  F = K + (C-P)*e^(rT)
    # 取多个行权价的 F 估计值，取中位数以排除异常值。
    # 候选行权价：在期货收盘价 ±10% 范围内且同时有 C/P 报价的行权价。
    near_df = near_df.sort_values("exercise_price")

    # ── 3. Put-Call Parity 反推隐含 Forward Price ─────────────────────
    from models.pricing.forward_price import calc_implied_forward
    implied_fwd, n_est = calc_implied_forward(near_df, T, RISK_FREE, futures_close)
    if n_est > 0:
        diff = implied_fwd - futures_close
        print(f"  隐含 Forward Price : {implied_fwd:.2f}"
              f"  （vs 期货收盘价 {futures_close:.2f}，差异 {diff:+.2f} 点，"
              f"来自 {n_est} 个行权价的 PCP 估计）")
    else:
        print(f"  [警告] PCP 无法反推 Forward Price，使用期货收盘价 {futures_close:.2f}")

    # ── 4. 用隐含 Forward 确定真实 ATM 行权价 ─────────────────────────
    strikes = near_df["exercise_price"].unique()
    atm_strike = float(strikes[np.argmin(np.abs(strikes - implied_fwd))])
    atm_rows = near_df[near_df["exercise_price"] == atm_strike].copy()

    # ── 5. 用隐含 Forward 计算 ATM IV（Call + Put 均值）──────────────
    iv_by_cp: dict[str, float] = {}
    vol_by_cp: dict[str, float] = {}
    for _, row in atm_rows.iterrows():
        try:
            mkt = float(row["close"])
            if mkt <= 0:
                continue
            cp = str(row["call_put"])
            iv = calc_implied_vol(
                market_price=mkt,
                S=implied_fwd, K=atm_strike, T=T, r=RISK_FREE,
                option_type=cp,
            )
            if iv and 0.01 < iv < 5.0:
                iv_by_cp[cp] = iv
                vol_by_cp[cp] = float(row.get("volume", 0) or 0)
                print(f"  ATM {cp} K={atm_strike:.0f}  Forward={implied_fwd:.2f}"
                      f"  价格={mkt:.1f}  IV={iv*100:.2f}%"
                      f"  成交量={vol_by_cp[cp]:.0f}")
        except Exception as e:
            print(f"  [警告] ATM IV 单合约失败: {e}")

    if not iv_by_cp:
        raise RuntimeError(f"无法计算 ATM IV（K={atm_strike}, T={T:.3f}）")

    # ── 6. C/P 发散检测（用隐含 Forward 后应已收敛，> 5pp 时警告）─────
    if len(iv_by_cp) == 2:
        c_iv = iv_by_cp.get("C", iv_by_cp.get("CALL", 0))
        p_iv = iv_by_cp.get("P", iv_by_cp.get("PUT", 0))
        diff_pp = abs(c_iv - p_iv) * 100
        if diff_pp > 5.0:
            print(f"  [警告] 使用隐含 Forward 后 C/P IV 仍差 {diff_pp:.1f}pp"
                  f"（Call={c_iv*100:.2f}%  Put={p_iv*100:.2f}%）"
                  f"，可能存在流动性或数据问题")

    atm_iv = float(np.mean(list(iv_by_cp.values())))
    used_cps = "/".join(iv_by_cp.keys())
    return (
        atm_iv,
        f"MO{chosen_month} ATM K={atm_strike:.0f}"
        f"（数据日={trade_date}，{expire_date} 到期，剩余{days_left}天"
        f"，{used_cps}，隐含F={implied_fwd:.2f}）",
    )


# ---------------------------------------------------------------------------
# Step 4：读取 IM.CFX 日线，拟合 GJR-GARCH，预测5日波动率
# ---------------------------------------------------------------------------

def get_garch_forecast(db) -> tuple[float, float, str]:
    """返回 (garch_5d_vol, rv_22d, source_desc)。"""
    from models.volatility.garch_model import GJRGARCHModel

    df = db.query_df(
        "SELECT trade_date, close FROM futures_daily "
        "WHERE ts_code='IM.CFX' ORDER BY trade_date ASC"
    )
    if df is None or len(df) < 60:
        raise RuntimeError(f"IM.CFX 日线数据不足（{len(df) if df is not None else 0} 条）")

    df = df.sort_values("trade_date")
    close = pd.Series(
        df["close"].astype(float).values,
        index=pd.to_datetime(df["trade_date"]),
    ).dropna()

    n_days = len(close)
    date_range = f"{df['trade_date'].iloc[0]} ~ {df['trade_date'].iloc[-1]}"
    source = f"IM.CFX 日线 {n_days} 条（{date_range}）"

    returns = np.log(close / close.shift(1)).dropna()

    # 22 日历史波动率（年化）
    rv_22d = float(returns.rolling(22).std().dropna().iloc[-1]) * np.sqrt(252)

    # GJR-GARCH 拟合
    model = GJRGARCHModel()
    fit = model.fit(returns)
    garch_5d = float(model.forecast_period_avg(horizon=5))
    converged = "已收敛" if fit.converged else "未收敛（供参考）"
    print(f"  GJR-GARCH 拟合完成（{converged}）  "
          f"持续性={fit.persistence:.4f}  "
          f"ω={fit.params.get('omega', float('nan')):.2e}")

    return garch_5d, rv_22d, source


# ---------------------------------------------------------------------------
# Step 5：计算组合 Greeks
# ---------------------------------------------------------------------------

def calc_portfolio(
    spot: float,
    positions: list[dict],
    iv_map: dict[str, float],
    futures_prices: dict[str, float] | None = None,
    implied_forwards: dict[str, float] | None = None,
) -> tuple[dict, list[dict]]:
    """用 GreeksCalculator 计算组合 Greeks，注入各合约市场 IV 及同到期月份标的价格。

    标的价优先级：implied_forwards → futures_prices → spot（主力）。
    """
    from models.pricing.greeks import GreeksCalculator

    calc = GreeksCalculator(trade_date=TODAY_STR)

    positions_with_iv = []
    for pos in positions:
        iv = iv_map.get(pos["ts_code"])
        if iv is None:
            print(f"  [跳过] {pos['ts_code']} 无有效 IV，不纳入 Greeks 计算")
            continue
        parsed = _parse_mo(pos["ts_code"])
        expire_month = parsed[0] if parsed else None
        if implied_forwards and expire_month and expire_month in implied_forwards:
            S = implied_forwards[expire_month]
        elif futures_prices and expire_month:
            S = futures_prices.get(expire_month, spot)
        else:
            S = spot
        positions_with_iv.append({**pos, "iv": iv, "underlying_price": S})

    result = calc.calculate_position_greeks(
        positions=positions_with_iv,
        underlying_price=spot,   # fallback; per-position override takes precedence
        risk_free_rate=RISK_FREE,
    )
    return result, positions_with_iv


# ---------------------------------------------------------------------------
# 报告打印
# ---------------------------------------------------------------------------

def _sep(char: str = "─", width: int = 72) -> str:
    return char * width


def _vrp_signal(vrp: float) -> tuple[str, str]:
    abs_vrp = abs(vrp)
    strength = "弱" if abs_vrp < 0.01 else ("中" if abs_vrp < 0.03 else "强")
    if vrp > 0.01:
        return f"{strength}信号：做空波动率", "IV 溢价，期权定价偏贵，Sell Premium 策略占优"
    elif vrp < -0.01:
        return f"{strength}信号：做多波动率", "IV 折价，期权定价偏便宜，Buy Gamma/Straddle 策略占优"
    return "中性信号", "IV ≈ GARCH 预测，市场定价合理，维持现有仓位"


def print_report(
    spot: float,
    spot_source: str,
    greeks_result: dict,
    positions_with_iv: list[dict],
    garch_vol: float,
    rv_22d: float,
    garch_source: str,
    atm_iv: float,
    atm_iv_source: str,
    latest_date: str,
    futures_prices: dict[str, float] | None = None,
    implied_forwards: dict[str, float] | None = None,
    db=None,
    chain_df: pd.DataFrame | None = None,
) -> None:
    today = date.today().strftime("%Y-%m-%d")
    today_ts = pd.Timestamp.today().normalize()

    print()
    print(_sep("═"))
    print(f"  MO 期权持仓分析报告  |  分析日 {today}  |  行情日 {latest_date}")
    print(_sep("═"))

    # ── 标的行情 ────────────────────────────────────────────────────────
    print()
    print("【标的行情】")
    print(f"  现货指数       : {spot:>10.2f}    来源：{spot_source}")
    if futures_prices:
        for month in sorted(futures_prices):
            fp = futures_prices[month]
            discount = fp - spot
            print(f"  IM{month}.CFX      : {fp:>10.2f}    贴水 {discount:+.2f} vs 主力")
    if implied_forwards:
        print()
        print("  隐含 Forward (PCP):")
        all_months = sorted(set(list(implied_forwards.keys()) + list((futures_prices or {}).keys())))
        for month in all_months:
            fwd = implied_forwards.get(month)
            if fwd is None:
                continue
            fut = (futures_prices or {}).get(month, spot)
            diff = fwd - fut
            print(f"    MO{month} : {fwd:>10.2f}    (vs 期货 {fut:.2f}, 差异 {diff:+.2f})")
    print(f"  无风险利率     : {RISK_FREE*100:.2f}%")
    print(f"  合约乘数       : {CONTRACT_MULT} 元/点")

    # ── 持仓 IV 概览 ────────────────────────────────────────────────────
    print()
    print("【持仓 IV 概览（真实市场 IV，按到期月份使用对应期货价格）】")
    print(f"  {'合约':<28} {'方向':>4} {'手数':>5} {'行权价':>7} {'到期日':>10}  "
          f"{'标的价':>8}  {'T(年)':>6}  {'IV':>7}")
    print(f"  {_sep('-', 78)}")
    for pos in positions_with_iv:
        expire_ts = pd.Timestamp(pos["expire_date"])
        T = max((expire_ts - today_ts).days / 365.0, 0.0)
        direction = "多头" if pos["volume"] > 0 else "空头"
        S_used = pos.get("underlying_price", spot)
        print(f"  {pos['ts_code']:<28} {direction:>4} {pos['volume']:>5}  "
              f"{pos['strike_price']:>7.0f}  {pos['expire_date']:>8}  "
              f"{S_used:>8.2f}  {T:>6.3f}  {pos['iv']*100:>6.2f}%")

    # ── 组合 Greeks ─────────────────────────────────────────────────────
    nd = greeks_result["net_delta"]
    ng = greeks_result["net_gamma"]
    nt = greeks_result["net_theta"]
    nv = greeks_result["net_vega"]
    dd = nd * spot

    print()
    print("【组合 Greeks（含合约乘数 100）】")
    print(f"  净 Delta  : {nd:>+12.4f} 元/点    （标的每涨1点，组合盈亏 {nd:>+.0f} 元）")
    print(f"  净 Gamma  : {ng:>+12.6f} 元/点²   （标的每涨1点，Delta 变化 {ng:>+.6f}）")
    print(f"  净 Theta  : {nt:>+12.4f} 元/天    （每过一天，组合盈亏 {nt:>+.0f} 元）")
    print(f"  净 Vega   : {nv:>+12.4f} 元/1%σ   （波动率每变动1%，组合盈亏 {nv:>+,.0f} 元）")
    print()
    print(f"  Delta 名义敞口 : {dd:>+14,.0f} 元  （净Delta × 标的价，等值现货敞口）")
    print(f"  Vega P&L/1%σ  : {nv:>+14,.0f} 元  （波动率每上升1%的P&L）")

    # ── 逐仓明细 ────────────────────────────────────────────────────────
    print()
    print("【逐仓 Greeks 明细】")
    print(f"  {'合约':<28} {'pos_Δ':>9} {'pos_Γ':>11} {'Θ(元/天)':>10} {'V(元/1%σ)':>11}")
    print(f"  {_sep('-', 72)}")
    for detail, pos in zip(greeks_result["positions_detail"], positions_with_iv):
        print(
            f"  {pos['ts_code']:<28}"
            f"  {detail['position_delta']:>+9.2f}"
            f"  {detail['position_gamma']:>+11.6f}"
            f"  {detail['position_theta']:>+10.2f}"
            f"  {detail['position_vega']:>+11.2f}"
        )

    # ── 波动率分析 ──────────────────────────────────────────────────────
    print()
    print("【波动率分析（真实数据）】")
    print(f"  数据来源           : {garch_source}")
    print(f"  22日历史波动率(RV)  : {rv_22d*100:>7.2f}%  （已实现波动率，年化）")
    print(f"  GJR-GARCH 5日预测  : {garch_vol*100:>7.2f}%  （条件波动率预测，年化）")
    print(f"  ATM IV 来源        : {atm_iv_source}")
    print(f"  ATM IV             : {atm_iv*100:>7.2f}%  （期权隐含波动率）")

    # ── VRP 信号 ────────────────────────────────────────────────────────
    vrp = atm_iv - garch_vol
    vrp_ratio = vrp / garch_vol if garch_vol > 0 else 0.0
    signal_label, rationale = _vrp_signal(vrp)

    print()
    print("【VRP 波动率风险溢价信号】")
    print(f"  VRP = ATM_IV − GARCH_5d  =  {atm_iv*100:.2f}% − {garch_vol*100:.2f}%  "
          f"=  {vrp*100:+.2f}%")
    print(f"  VRP/GARCH (相对溢价)     =  {vrp_ratio*100:+.1f}%")
    print()
    print(f"  ▶  {signal_label}")
    print(f"     {rationale}")

    # ── 持仓结构 ────────────────────────────────────────────────────────
    print()
    print("【当前持仓结构解读】")
    theta_view = (f"正 Theta（每日收取约 {nt:+.0f} 元时间价值）" if nt > 0
                  else f"负 Theta（每日支付约 {nt:+.0f} 元时间价值）")
    vega_view  = (f"空 Vega — 波动率上升不利（{nv:+,.0f} 元/1%σ）" if nv < 0
                  else f"多 Vega — 波动率上升有利（{nv:+,.0f} 元/1%σ）")
    delta_view = (f"近似 Delta 中性（净 Delta={nd:+.2f}）" if abs(nd) < 100 else
                  f"净多 Delta（{nd:+.2f}），对标的上涨有利" if nd > 0 else
                  f"净空 Delta（{nd:+.2f}），对标的下跌有利")

    print(f"  Theta : {theta_view}")
    print(f"  Vega  : {vega_view}")
    print(f"  Delta : {delta_view}")

    # ── 风险提示 ────────────────────────────────────────────────────────
    print()
    print("【风险提示】")
    gamma_pnl_1pct = 0.5 * ng * (spot * 0.01) ** 2
    print(f"  标的涨/跌 1%（约 {spot*0.01:.0f} 点），Gamma P&L ≈ {gamma_pnl_1pct:+.0f} 元")
    print(f"  波动率上升 1%，Vega P&L ≈ {nv:+,.0f} 元")
    # 找最近到期合约的剩余天数
    if positions_with_iv:
        nearest_pos = min(positions_with_iv, key=lambda p: p["expire_date"])
        days_left = (pd.Timestamp(nearest_pos["expire_date"]) - today_ts).days
        print(f"  最近到期合约 {nearest_pos['ts_code']}: 剩余 {days_left} 天，注意 Theta 加速衰减")

    # ── 贴水捕获策略参考 ─────────────────────────────────────────────────────
    if db is not None:
        print_discount_section(
            db=db,
            spot=spot,
            trade_date=latest_date,
            chain_df=chain_df if chain_df is not None else pd.DataFrame(),
        )

    print()
    print(_sep("═"))
    print()

    # 生成 Markdown 报告并持久化
    try:
        md = _generate_analysis_markdown(
            spot=spot,
            spot_source=spot_source,
            greeks_result=greeks_result,
            positions_with_iv=positions_with_iv,
            garch_vol=garch_vol,
            rv_22d=rv_22d,
            atm_iv=atm_iv,
            latest_date=latest_date,
            futures_prices=futures_prices,
            implied_forwards=implied_forwards,
        )
        from utils.report_writer import save_report
        td = latest_date.replace("-", "")
        fp = save_report(td, "analysis", md)
        print(f"报告已保存: {fp}")
    except Exception as e:
        logging.getLogger(__name__).warning("分析报告保存失败: %s", e)


def _generate_analysis_markdown(
    spot: float,
    spot_source: str,
    greeks_result: dict,
    positions_with_iv: list[dict],
    garch_vol: float,
    rv_22d: float,
    atm_iv: float,
    latest_date: str,
    futures_prices: dict | None = None,
    implied_forwards: dict | None = None,
) -> str:
    """生成持仓分析 Markdown 报告。"""
    today = date.today().strftime("%Y-%m-%d")
    today_ts = pd.Timestamp.today().normalize()
    lines: list[str] = []

    lines.append(f"# MO 期权持仓分析 | {today}")
    lines.append(f"行情日: {latest_date}")

    # 标的行情
    lines.append("\n## 标的行情")
    lines.append(f"- 现货指数: {spot:.2f}（来源: {spot_source}）")
    if futures_prices:
        for month in sorted(futures_prices):
            fp = futures_prices[month]
            lines.append(f"- IM{month}.CFX: {fp:.2f}（贴水 {fp - spot:+.2f}）")
    if implied_forwards:
        lines.append("\n隐含 Forward (PCP):")
        for month in sorted(implied_forwards):
            fwd = implied_forwards[month]
            fut = (futures_prices or {}).get(month, spot)
            lines.append(f"- MO{month}: {fwd:.2f}（vs 期货 {fut:.2f}, 差异 {fwd - fut:+.2f}）")

    # 持仓 IV 概览
    lines.append("\n## 持仓 IV 概览")
    lines.append("| 合约 | 方向 | 手数 | 行权价 | 到期日 | 标的价 | T(年) | IV |")
    lines.append("|------|------|------|--------|--------|--------|-------|-----|")
    for pos in positions_with_iv:
        expire_ts = pd.Timestamp(pos["expire_date"])
        T = max((expire_ts - today_ts).days / 365.0, 0.0)
        direction = "多头" if pos["volume"] > 0 else "空头"
        S_used = pos.get("underlying_price", spot)
        lines.append(
            f"| {pos['ts_code']} | {direction} | {pos['volume']} "
            f"| {pos['strike_price']:.0f} | {pos['expire_date']} "
            f"| {S_used:.2f} | {T:.3f} | {pos['iv']*100:.2f}% |"
        )

    # 组合 Greeks
    nd = greeks_result["net_delta"]
    ng = greeks_result["net_gamma"]
    nt = greeks_result["net_theta"]
    nv = greeks_result["net_vega"]

    lines.append("\n## 组合 Greeks")
    lines.append(f"- 净Delta: {nd:+.4f} 元/点")
    lines.append(f"- 净Gamma: {ng:+.6f} 元/点²")
    lines.append(f"- 净Theta: {nt:+.4f} 元/天")
    lines.append(f"- 净Vega: {nv:+.4f} 元/1%σ")
    lines.append(f"- Delta 名义敞口: {nd * spot:+,.0f} 元")

    # 逐仓明细
    lines.append("\n## 逐仓 Greeks 明细")
    lines.append("| 合约 | pos_Δ | pos_Γ | Θ(元/天) | V(元/1%σ) |")
    lines.append("|------|-------|-------|----------|-----------|")
    for detail, pos in zip(greeks_result["positions_detail"], positions_with_iv):
        lines.append(
            f"| {pos['ts_code']} "
            f"| {detail['position_delta']:+.2f} "
            f"| {detail['position_gamma']:+.6f} "
            f"| {detail['position_theta']:+.2f} "
            f"| {detail['position_vega']:+.2f} |"
        )

    # 波动率分析
    lines.append("\n## 波动率分析")
    lines.append(f"- 22日历史波动率(RV): {rv_22d*100:.2f}%")
    lines.append(f"- GJR-GARCH 5日预测: {garch_vol*100:.2f}%")
    lines.append(f"- ATM IV: {atm_iv*100:.2f}%")

    # VRP
    vrp = atm_iv - garch_vol
    vrp_ratio = vrp / garch_vol if garch_vol > 0 else 0.0
    lines.append("\n## VRP 信号")
    lines.append(
        f"- VRP = ATM_IV − GARCH = {atm_iv*100:.2f}% − {garch_vol*100:.2f}% "
        f"= **{vrp*100:+.2f}%**"
    )
    lines.append(f"- VRP/GARCH (相对溢价): {vrp_ratio*100:+.1f}%")

    # 持仓结构
    lines.append("\n## 持仓结构解读")
    theta_view = (f"正 Theta（每日收取约 {nt:+.0f} 元）" if nt > 0
                  else f"负 Theta（每日支付约 {nt:+.0f} 元）")
    vega_view = (f"空 Vega（{nv:+,.0f} 元/1%σ）" if nv < 0
                 else f"多 Vega（{nv:+,.0f} 元/1%σ）")
    lines.append(f"- Theta: {theta_view}")
    lines.append(f"- Vega: {vega_view}")
    lines.append(f"- Delta: 净{nd:+.2f}")

    # 风险提示
    lines.append("\n## 风险提示")
    gamma_pnl_1pct = 0.5 * ng * (spot * 0.01) ** 2
    lines.append(f"- 标的涨/跌 1%（约 {spot*0.01:.0f} 点），Gamma P&L ≈ {gamma_pnl_1pct:+.0f} 元")
    lines.append(f"- 波动率上升 1%，Vega P&L ≈ {nv:+,.0f} 元")
    if positions_with_iv:
        nearest_pos = min(positions_with_iv, key=lambda p: p["expire_date"])
        days_left = (pd.Timestamp(nearest_pos["expire_date"]) - today_ts).days
        lines.append(f"- 最近到期: {nearest_pos['ts_code']}，剩余 {days_left} 天")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 贴水捕获策略参考
# ---------------------------------------------------------------------------

def print_discount_section(
    db,
    spot: float,
    trade_date: str,
    chain_df: pd.DataFrame,
) -> None:
    """
    打印贴水捕割策略参考信息，包含：
    - 各合约年化贴水率
    - 推荐合约（贴水率最优）
    - 推荐 Put 保护方案
    """
    try:
        from strategies.discount_capture.signal import DiscountSignal
        from strategies.discount_capture.position import DiscountPosition
    except ImportError as e:
        print(f"  [警告] 贴水策略模块加载失败: {e}")
        return

    print()
    print("【贴水收割策略参考】")
    print(_sep("-", 60))

    try:
        disc_signal = DiscountSignal(db)
        disc_df = disc_signal.calculate_discount(trade_date)
    except Exception as e:
        print(f"  贴水计算失败: {e}")
        return

    if disc_df is None or disc_df.empty:
        print("  暂无贴水数据（当日行情可能未更新）")
        return

    # 打印各合约贴水率
    for _, row in disc_df.iterrows():
        month = row["contract_month"]
        iml_code = row["iml_code"]
        ann_rate = float(row["annualized_discount_rate"])
        abs_disc = float(row["absolute_discount"])
        dte = int(row["days_to_expiry"])
        disc_type = "贴水" if abs_disc < 0 else "升水"
        print(f"  IM{month} ({iml_code:<12}): 年化{disc_type}率={ann_rate*100:>6.2f}%  "
              f"绝对{disc_type}={abs_disc:+.1f}点  剩余{dte}天")

    # 生成信号
    try:
        sig_result = disc_signal.generate_signal(trade_date)
    except Exception as e:
        print(f"  信号生成失败: {e}")
        return

    signal = sig_result.get("signal", "NONE")
    rec_contract = sig_result.get("recommended_contract", "N/A")
    ann_disc = sig_result.get("annualized_discount", 0.0)
    pct = sig_result.get("discount_percentile", 0.0)
    dte_rec = sig_result.get("days_to_expiry", 0)

    print()
    if signal != "NONE":
        print(f"  推荐合约: {rec_contract}（年化贴水率={ann_disc*100:.2f}%，"
              f"历史{pct:.0f}百分位，剩余{dte_rec}天）")
        print(f"  信号强度: {signal}")
    else:
        print("  当前无贴水信号（贴水率不足或无合适合约）")

    # Put / Put Spread 保护方案比较表
    CANDIDATE_STRIKES = [6600, 6800, 7000, 7200, 7400, 7600]
    if chain_df is not None and not chain_df.empty:
        best_month = rec_contract[2:] if rec_contract and rec_contract.startswith("IM") else None
        if not best_month and not disc_df.empty:
            best_month = disc_df.iloc[0]["contract_month"]
        if best_month:
            sub_chain = (chain_df[chain_df["expire_month"] == best_month].copy()
                         if "expire_month" in chain_df.columns else chain_df)
            put_chain = (sub_chain[sub_chain["call_put"] == "P"]
                         if not sub_chain.empty else pd.DataFrame())
            if not put_chain.empty:
                try:
                    fut_price = (
                        float(disc_df[disc_df["contract_month"] == best_month]["futures_price"].iloc[0])
                        if best_month in disc_df["contract_month"].values else spot
                    )
                    ann_disc = sig_result.get("annualized_discount", 0.0) if signal != "NONE" else 0.0
                    dte      = sig_result.get("days_to_expiry", 90)      if signal != "NONE" else 90
                    disc_pnl_per_lot = fut_price * ann_disc * (dte / 365) * 200

                    put_idx  = put_chain.set_index("exercise_price")["close"].to_dict()
                    pos_mgr  = DiscountPosition(account_equity=1_000_000.0)

                    rows = pos_mgr.build_protection_comparison(
                        put_index=put_idx,
                        futures_price=fut_price,
                        disc_pnl_per_lot=disc_pnl_per_lot,
                        candidate_strikes=CANDIDATE_STRIKES,
                        spread_widths=[400, 600],
                    )

                    print()
                    print(f"  Put 保护方案比较 (MO{best_month}，期货={fut_price:.0f}，"
                          f"贴水收益≈{disc_pnl_per_lot:+.0f}元/手)")
                    hdr = (f"  {'方案':<20}  {'买Put':>5}  {'卖Put':>5}  "
                           f"{'净成本':>8}  {'最大保护':>9}  {'最大亏损':>9}  "
                           f"{'净收益':>9}  {'比率':>6}")
                    sep = (f"  {'─'*20}  {'─'*5}  {'─'*5}  "
                           f"{'─'*8}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*6}")
                    print(hdr)
                    print(sep)

                    prev_buy = None
                    for r in rows:
                        # 在每个新买入行权价前加空行分组
                        if r["buy_strike"] != prev_buy and prev_buy is not None:
                            print()
                        prev_buy = r["buy_strike"]

                        if not r.get("available", False):
                            print(f"  {r['scheme']:<20}  {'─':>5}  {'─':>5}  无数据")
                            continue
                        sell_s   = f"{r['sell_strike']:.0f}" if r["sell_strike"] else "  —"
                        max_prot = ("   无上限"
                                    if r["max_protection"] is None
                                    else f"{r['max_protection']:>9,.0f}")
                        print(f"  {r['scheme']:<20}  {r['buy_strike']:>5.0f}  {sell_s:>5}  "
                              f"{r['net_cost']:>8,.0f}  {max_prot}  "
                              f"{r['max_loss']:>+9,.0f}  "
                              f"{r['net_disc_pnl']:>+9,.0f}  "
                              f"{r['protection_ratio']:>5.1f}x")

                    print()
                    print(f"  说明: 净成本=权利金差×100  最大保护=Spread封顶保护金额(裸Put无上限)")
                    print(f"        最大亏损=跌至买入行权价时的期货亏损+净成本  净收益=贴水收益-净成本")
                except Exception as e:
                    print(f"  Put 比较表计算失败: {e}")


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main() -> None:
    print(_sep("─"))
    print("  Portfolio Analysis  |  MO 期权持仓分析（真实数据）")
    print(_sep("─"))
    print(f"  分析日期 : {date.today()}")
    print(f"  数据库   : {DB_PATH}")
    print()

    # Step 0：连接数据库
    print("[0] 连接数据库...")
    db = _open_db()
    if db is None:
        print("  [错误] 数据库文件不存在，退出")
        sys.exit(1)
    print(f"  已连接")

    # Step 1：获取标的价格
    print("\n[1] 获取现货价格（000852.SH / IM.CFX）...")
    spot, spot_source = get_spot_price(db)
    print(f"  标的价格: {spot:.2f}  来源: {spot_source}")

    # Step 2：从 position_snapshots 动态读取持仓，构建持仓列表
    print("\n[2] 从 position_snapshots 读取最新持仓...")
    positions_raw = _load_positions_from_db(db)
    if positions_raw is None:
        print("  [警告] position_snapshots 为空或不可用，回退到硬编码持仓列表")
        positions_raw = _POSITIONS_RAW_FALLBACK
    for pr in positions_raw:
        print(f"  读取: {pr['ts_code']:<28}  vol={pr['volume']:+d}")

    print("  从 options_contracts 获取到期日...")
    expire_map = _build_expire_map(db)
    print(f"  到期日映射: {expire_map}")
    positions = build_positions(expire_map, positions_raw)
    if not positions:
        print("  [错误] 持仓列表为空，退出")
        sys.exit(1)
    for p in positions:
        print(f"  {p['ts_code']:<28}  到期={p['expire_date']}  vol={p['volume']:+d}")

    # Step 3：获取最新 MO 期权链
    print("\n[3] 读取最新 MO 期权链...")
    latest_row = db.query_df(
        "SELECT MAX(trade_date) as dt FROM options_daily WHERE ts_code LIKE 'MO%'"
    )
    latest_date = str(latest_row["dt"].iloc[0]) if latest_row is not None and not latest_row.empty else ""
    if not latest_date or latest_date == "None":
        print("  [错误] 无法获取最新 MO 期权日期，退出")
        sys.exit(1)
    print(f"  最新交易日: {latest_date}")
    chain_df = _get_mo_chain(db, latest_date, expire_map)
    print(f"  读取期权链: {len(chain_df)} 条（含有效收盘价）")

    # Step 3b：查询各到期月份对应 IM 期货价格
    expire_months = sorted({p["ts_code"][2:6] for p in positions})
    print(f"\n[3b] 查询各到期月份期货标的价格（修正贴水偏差）...")
    futures_prices = get_futures_prices_by_expiry(db, expire_months, spot, trade_date=latest_date)
    for month, fp in sorted(futures_prices.items()):
        discount = fp - spot
        print(f"  IM{month}.CFX = {fp:.2f}  （贴水 {discount:+.2f} vs 主力 {spot:.2f}）")

    # Step 3c：为每个到期月份计算 PCP 隐含 Forward Price
    print("\n[3c] 计算各到期月份 PCP 隐含 Forward Price...")
    implied_forwards = calc_implied_forwards_by_expiry(chain_df, futures_prices, spot)

    # Step 4：计算各持仓 IV
    print("\n[4] 计算持仓合约 IV...")
    iv_map = get_iv_for_positions(positions, spot, chain_df, futures_prices, implied_forwards)
    if not iv_map:
        print("  [错误] 所有持仓 IV 计算失败，退出")
        sys.exit(1)

    # Step 5：拟合 GJR-GARCH
    print("\n[5] 拟合 GJR-GARCH（IM.CFX 日线）...")
    garch_vol, rv_22d, garch_source = get_garch_forecast(db)
    print(f"  22日历史RV : {rv_22d*100:.2f}%")
    print(f"  GARCH 5日预测 : {garch_vol*100:.2f}%")

    # Step 6：提取 ATM IV（用最近到期月对应期货价格）
    print("\n[6] 提取 ATM IV...")
    atm_iv, atm_iv_source = get_atm_iv(spot, chain_df, latest_date, futures_prices)
    print(f"  ATM IV: {atm_iv*100:.2f}%  来源: {atm_iv_source}")

    # Step 7：计算组合 Greeks
    print("\n[7] 计算组合 Greeks...")
    greeks_result, positions_with_iv = calc_portfolio(spot, positions, iv_map, futures_prices, implied_forwards)
    if not positions_with_iv:
        print("  [错误] 无有效持仓可计算 Greeks，退出")
        sys.exit(1)

    # 打印完整报告
    print_report(
        spot=spot,
        spot_source=spot_source,
        greeks_result=greeks_result,
        positions_with_iv=positions_with_iv,
        garch_vol=garch_vol,
        rv_22d=rv_22d,
        garch_source=garch_source,
        atm_iv=atm_iv,
        atm_iv_source=atm_iv_source,
        latest_date=latest_date,
        futures_prices=futures_prices,
        implied_forwards=implied_forwards,
        db=db,
        chain_df=chain_df,
    )


if __name__ == "__main__":
    main()
