"""
schemas.py
----------
职责：定义数据库表结构。
使用纯 SQL 字符串定义每张表（不用 ORM），清晰直接。

表清单：
    市场数据
    - futures_daily        股指期货日线（IF/IH/IC/IM）
    - futures_min          期货分钟线（用于计算已实现波动率）
    - options_daily        期权日线
    - options_contracts    期权合约基本信息
    - commodity_daily      商品期货日线（CU/RB/SC 等）
    - trade_calendar       交易日历

    每日记录（实盘运营）
    - tq_snapshots         盘中行情快照（TQ 抓取，每5分钟或信号触发时）
    - account_snapshots    每日账户快照（权益/可用/保证金/风险度）
    - position_snapshots   每日持仓快照（含浮盈/保证金）
    - trade_records        每日成交流水（完整交易记录）
    - daily_model_output   每日模型输出（GARCH/IV/VRP/Greeks）
    - daily_notes          每日研究笔记（市场观察/交易理由/复盘）

    策略层
    - strategy_signals     策略信号记录
    - strategy_trades      策略成交记录
    - strategy_pnl         策略每日盈亏

命名约定：
    - 所有日期字段用 TEXT，格式 YYYYMMDD（日线）或 YYYY-MM-DD HH:MM:SS（分钟线）
    - 所有价格/数量字段用 REAL
    - 每个高频查询字段都建索引
"""

from __future__ import annotations

from dataclasses import dataclass


# ======================================================================
# 数据类定义（用于类型提示和数据传输）
# ======================================================================

@dataclass
class FuturesDaily:
    """期货日线数据"""
    ts_code: str           # 合约代码，如 IF2406.CFX
    trade_date: str        # 交易日，格式 YYYYMMDD
    open: float            # 开盘价
    high: float            # 最高价
    low: float             # 最低价
    close: float           # 收盘价
    volume: float          # 成交量（手）
    oi: float              # 持仓量（手）
    settle: float          # 结算价
    pre_close: float = 0.0   # 昨收盘价
    pre_settle: float = 0.0  # 昨结算价


@dataclass
class FuturesMin:
    """期货分钟线数据（用于计算已实现波动率）"""
    symbol: str            # 品种代码 IF/IH/IM
    datetime: str          # 时间戳，格式 YYYY-MM-DD HH:MM:SS
    period: int            # 周期秒数 300=5m, 900=15m
    open: float            # 开盘价
    high: float            # 最高价
    low: float             # 最低价
    close: float           # 收盘价
    volume: float          # 成交量（手）


@dataclass
class OptionsDaily:
    """期权日线数据"""
    ts_code: str              # 期权合约代码
    trade_date: str           # 交易日，格式 YYYYMMDD
    exchange: str             # 交易所（CFFEX / SSE / SZSE）
    underlying_code: str      # 标的代码，如 IO / MO / 510050.SH
    exercise_price: float     # 行权价
    call_put: str             # 期权类型：C（认购）/ P（认沽）
    expire_date: str          # 到期日，格式 YYYYMMDD
    close: float              # 收盘价
    settle: float             # 结算价
    volume: float             # 成交量（手）
    oi: float                 # 持仓量（手）
    pre_close: float = 0.0    # 昨收盘价
    pre_settle: float = 0.0   # 昨结算价


@dataclass
class OptionsContracts:
    """期权合约基本信息"""
    ts_code: str              # 期权合约代码
    exchange: str             # 交易所
    underlying_code: str      # 标的代码
    exercise_price: float     # 行权价
    call_put: str             # 期权类型：C / P
    expire_date: str          # 到期日，格式 YYYYMMDD
    list_date: str            # 上市日期
    delist_date: str          # 摘牌日期
    contract_unit: float = 0.0   # 合约乘数（如 IO=100）
    exercise_type: str = "E"     # 行权方式：E=欧式 A=美式


@dataclass
class CommodityDaily:
    """商品期货日线数据（CTP 品种）"""
    ts_code: str           # 合约代码，如 CU2406.SHF
    trade_date: str        # 交易日，格式 YYYYMMDD
    exchange: str          # 交易所（SHFE/DCE/CZCE/INE/GFEX/CFFEX）
    underlying: str        # 品种代码，如 CU
    open: float            # 开盘价
    high: float            # 最高价
    low: float             # 最低价
    close: float           # 收盘价
    volume: float          # 成交量（手）
    oi: float              # 持仓量（手）
    settle: float          # 结算价
    amount: float = 0.0    # 成交额（万元）


@dataclass
class TradeCalendar:
    """交易日历"""
    exchange: str          # 交易所代码（SSE / SZSE / CFFEX）
    trade_date: str        # 日期，格式 YYYYMMDD
    is_open: int           # 是否交易日：1=是，0=否
    pretrade_date: str     # 上一交易日，格式 YYYYMMDD


@dataclass
class StrategySignal:
    """策略信号记录"""
    strategy_name: str     # 策略名称
    trade_date: str        # 信号日期，格式 YYYYMMDD
    symbol: str            # 交易标的代码
    direction: str         # 信号方向（long/short/neutral/close）
    signal_type: str       # 信号类型（entry/exit/rebalance）
    strength: str          # 信号强度（strong/moderate/weak）
    target_volume: int     # 目标持仓手数
    metadata_json: str     # 策略特定元数据（JSON 字符串）
    created_at: str = ""   # 创建时间，格式 YYYY-MM-DD HH:MM:SS


@dataclass
class StrategyTrade:
    """策略成交记录"""
    strategy_name: str     # 策略名称
    trade_date: str        # 成交日期，格式 YYYYMMDD
    symbol: str            # 合约代码
    direction: str         # 成交方向（buy/sell）
    volume: int            # 成交手数
    price: float           # 成交价格
    commission: float      # 手续费（元）
    slippage: float        # 滑点（元）
    created_at: str = ""   # 创建时间


@dataclass
class StrategyPnL:
    """策略每日盈亏记录"""
    strategy_name: str     # 策略名称
    trade_date: str        # 日期，格式 YYYYMMDD
    realized_pnl: float    # 已实现盈亏（元）
    unrealized_pnl: float  # 浮动盈亏（元）
    commission: float      # 当日手续费（元）
    net_pnl: float         # 净盈亏 = realized + unrealized - commission


# ======================================================================
# SQL 建表语句
# ======================================================================

FUTURES_DAILY_SQL = """
CREATE TABLE IF NOT EXISTS futures_daily (
    ts_code      TEXT    NOT NULL,          -- 合约代码，如 IF2406.CFX
    trade_date   TEXT    NOT NULL,          -- 交易日 YYYYMMDD
    open         REAL,
    high         REAL,
    low          REAL,
    close        REAL,
    volume       REAL,                      -- 成交量（手）
    oi           REAL,                      -- 持仓量
    settle       REAL,                      -- 结算价
    pre_close    REAL,                      -- 昨收盘价
    pre_settle   REAL,                      -- 昨结算价
    PRIMARY KEY (ts_code, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_fd_trade_date ON futures_daily (trade_date);
CREATE INDEX IF NOT EXISTS idx_fd_ts_code    ON futures_daily (ts_code);
"""

FUTURES_MIN_SQL = """
CREATE TABLE IF NOT EXISTS futures_min (
    symbol       TEXT    NOT NULL,          -- 品种代码 IF/IH/IM
    datetime     TEXT    NOT NULL,          -- 时间戳 YYYY-MM-DD HH:MM:SS
    period       INT     NOT NULL,          -- 周期秒数 300=5m, 900=15m
    open         REAL,
    high         REAL,
    low          REAL,
    close        REAL,
    volume       REAL,                      -- 成交量（手）
    open_interest REAL,
    PRIMARY KEY (symbol, datetime, period)
);
CREATE INDEX IF NOT EXISTS idx_fm_symbol   ON futures_min (symbol);
CREATE INDEX IF NOT EXISTS idx_fm_datetime ON futures_min (datetime);
"""

OPTIONS_DAILY_SQL = """
CREATE TABLE IF NOT EXISTS options_daily (
    ts_code          TEXT    NOT NULL,      -- 期权合约代码
    trade_date       TEXT    NOT NULL,      -- 交易日 YYYYMMDD
    exchange         TEXT,                  -- 交易所 CFFEX/SSE/SZSE
    underlying_code  TEXT,                  -- 标的代码，如 IO/MO/510050.SH
    exercise_price   REAL,                  -- 行权价
    call_put         TEXT,                  -- 期权类型 C/P
    expire_date      TEXT,                  -- 到期日 YYYYMMDD
    close            REAL,                  -- 收盘价
    settle           REAL,                  -- 结算价
    volume           REAL,                  -- 成交量（手）
    oi               REAL,                  -- 持仓量
    pre_close        REAL,                  -- 昨收盘价
    pre_settle       REAL,                  -- 昨结算价
    PRIMARY KEY (ts_code, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_od_trade_date      ON options_daily (trade_date);
CREATE INDEX IF NOT EXISTS idx_od_underlying_date ON options_daily (underlying_code, trade_date);
CREATE INDEX IF NOT EXISTS idx_od_expire_date     ON options_daily (expire_date);
"""

OPTIONS_CONTRACTS_SQL = """
CREATE TABLE IF NOT EXISTS options_contracts (
    ts_code          TEXT    PRIMARY KEY,   -- 期权合约代码
    exchange         TEXT,                  -- 交易所
    underlying_code  TEXT,                  -- 标的代码
    exercise_price   REAL,                  -- 行权价
    call_put         TEXT,                  -- 期权类型 C/P
    expire_date      TEXT,                  -- 到期日 YYYYMMDD
    list_date        TEXT,                  -- 上市日期 YYYYMMDD
    delist_date      TEXT,                  -- 摘牌日期 YYYYMMDD
    contract_unit    REAL,                  -- 合约乘数（如 IO=100）
    exercise_type    TEXT                   -- 行权方式：E=欧式 A=美式
);
CREATE INDEX IF NOT EXISTS idx_oc_underlying ON options_contracts (underlying_code);
CREATE INDEX IF NOT EXISTS idx_oc_expire     ON options_contracts (expire_date);
"""

COMMODITY_DAILY_SQL = """
CREATE TABLE IF NOT EXISTS commodity_daily (
    ts_code      TEXT    NOT NULL,          -- 合约代码，如 CU2406.SHF
    trade_date   TEXT    NOT NULL,          -- 交易日 YYYYMMDD
    exchange     TEXT,                      -- 交易所 SHFE/DCE/CZCE/INE/GFEX
    underlying   TEXT,                      -- 品种代码，如 CU
    open         REAL,
    high         REAL,
    low          REAL,
    close        REAL,
    volume       REAL,                      -- 成交量（手）
    oi           REAL,                      -- 持仓量
    settle       REAL,                      -- 结算价
    amount       REAL,                      -- 成交额（万元）
    PRIMARY KEY (ts_code, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_cd_trade_date  ON commodity_daily (trade_date);
CREATE INDEX IF NOT EXISTS idx_cd_underlying  ON commodity_daily (underlying, trade_date);
"""

TRADE_CALENDAR_SQL = """
CREATE TABLE IF NOT EXISTS trade_calendar (
    exchange       TEXT    NOT NULL,        -- 交易所 SSE/SZSE/CFFEX
    trade_date     TEXT    NOT NULL,        -- 日期 YYYYMMDD
    is_open        INTEGER,                 -- 是否交易日：1=是 0=否
    pretrade_date  TEXT,                    -- 上一交易日 YYYYMMDD
    PRIMARY KEY (exchange, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_tc_trade_date ON trade_calendar (trade_date);
"""

STRATEGY_SIGNALS_SQL = """
CREATE TABLE IF NOT EXISTS strategy_signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name   TEXT    NOT NULL,       -- 策略名称
    trade_date      TEXT    NOT NULL,       -- 信号日期 YYYYMMDD
    symbol          TEXT    NOT NULL,       -- 交易标的代码
    direction       TEXT,                   -- 方向 long/short/neutral/close
    signal_type     TEXT,                   -- 类型 entry/exit/rebalance
    strength        TEXT,                   -- 强度 strong/moderate/weak
    target_volume   INTEGER,                -- 目标持仓手数
    metadata_json   TEXT,                   -- 策略元数据（JSON）
    created_at      TEXT                    -- 创建时间 YYYY-MM-DD HH:MM:SS
);
CREATE INDEX IF NOT EXISTS idx_ss_strategy_date ON strategy_signals (strategy_name, trade_date);
CREATE INDEX IF NOT EXISTS idx_ss_symbol_date   ON strategy_signals (symbol, trade_date);
"""

STRATEGY_TRADES_SQL = """
CREATE TABLE IF NOT EXISTS strategy_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name   TEXT    NOT NULL,       -- 策略名称
    trade_date      TEXT    NOT NULL,       -- 成交日期 YYYYMMDD
    symbol          TEXT    NOT NULL,       -- 合约代码
    direction       TEXT,                   -- 方向 buy/sell
    volume          INTEGER,                -- 成交手数
    price           REAL,                   -- 成交价格
    commission      REAL,                   -- 手续费（元）
    slippage        REAL,                   -- 滑点（元）
    created_at      TEXT                    -- 创建时间 YYYY-MM-DD HH:MM:SS
);
CREATE INDEX IF NOT EXISTS idx_st_strategy_date ON strategy_trades (strategy_name, trade_date);
CREATE INDEX IF NOT EXISTS idx_st_symbol_date   ON strategy_trades (symbol, trade_date);
"""

STRATEGY_PNL_SQL = """
CREATE TABLE IF NOT EXISTS strategy_pnl (
    strategy_name   TEXT    NOT NULL,       -- 策略名称
    trade_date      TEXT    NOT NULL,       -- 日期 YYYYMMDD
    realized_pnl    REAL,                   -- 已实现盈亏（元）
    unrealized_pnl  REAL,                   -- 浮动盈亏（元）
    commission      REAL,                   -- 当日手续费（元）
    net_pnl         REAL,                   -- 净盈亏 = realized + unrealized - commission
    PRIMARY KEY (strategy_name, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_sp_strategy ON strategy_pnl (strategy_name);
CREATE INDEX IF NOT EXISTS idx_sp_date     ON strategy_pnl (trade_date);
"""

TQ_SNAPSHOTS_SQL = """
CREATE TABLE IF NOT EXISTS tq_snapshots (
    snapshot_time  TEXT    NOT NULL,        -- 快照时间 YYYY-MM-DD HH:MM:SS
    symbol         TEXT    NOT NULL,        -- 天勤合约代码，如 CFFEX.MO2604-P-7200
    last_price     REAL,                    -- 最新成交价
    bid_price1     REAL,                    -- 买一价
    ask_price1     REAL,                    -- 卖一价
    bid_volume1    INTEGER,                 -- 买一量
    ask_volume1    INTEGER,                 -- 卖一量
    volume         INTEGER,                 -- 当日累计成交量
    open_interest  REAL,                    -- 持仓量
    highest        REAL,                    -- 当日最高价
    lowest         REAL,                    -- 当日最低价
    open           REAL,                    -- 开盘价
    PRIMARY KEY (snapshot_time, symbol)
);
CREATE INDEX IF NOT EXISTS idx_tqs_symbol      ON tq_snapshots (symbol, snapshot_time);
CREATE INDEX IF NOT EXISTS idx_tqs_time        ON tq_snapshots (snapshot_time);
"""

ACCOUNT_SNAPSHOTS_SQL = """
CREATE TABLE IF NOT EXISTS account_snapshots (
    trade_date     TEXT    PRIMARY KEY,     -- 交易日 YYYYMMDD（每日唯一）
    snapshot_time  TEXT,                    -- 实际抓取时间 YYYY-MM-DD HH:MM:SS
    balance        REAL,                    -- 账户权益（元）
    available      REAL,                    -- 可用资金（元）
    margin         REAL,                    -- 已用保证金（元）
    margin_ratio   REAL,                    -- 保证金占用比（0~1）
    float_profit   REAL,                    -- 浮动盈亏（元）
    close_profit   REAL,                    -- 当日平仓盈亏（元）
    commission     REAL,                    -- 当日手续费（元）
    risk_ratio     REAL                     -- 风险度（0~1）
);
"""

POSITION_SNAPSHOTS_SQL = """
CREATE TABLE IF NOT EXISTS position_snapshots (
    trade_date       TEXT    NOT NULL,      -- 交易日 YYYYMMDD
    symbol           TEXT    NOT NULL,      -- 天勤合约代码
    direction        TEXT    NOT NULL,      -- 持仓方向：LONG / SHORT
    volume           INTEGER,               -- 持仓手数
    volume_today     INTEGER,               -- 今仓手数
    open_price_avg   REAL,                  -- 平均开仓价
    last_price       REAL,                  -- 最新价
    float_profit     REAL,                  -- 浮动盈亏（元）
    margin           REAL,                  -- 占用保证金（元）
    PRIMARY KEY (trade_date, symbol, direction)
);
CREATE INDEX IF NOT EXISTS idx_pos_date        ON position_snapshots (trade_date);
CREATE INDEX IF NOT EXISTS idx_pos_symbol      ON position_snapshots (symbol);
"""

TRADE_RECORDS_SQL = """
CREATE TABLE IF NOT EXISTS trade_records (
    trade_date     TEXT    NOT NULL,        -- 交易日 YYYYMMDD
    trade_time     TEXT,                    -- 成交时间 HH:MM:SS
    symbol         TEXT    NOT NULL,        -- 天勤合约代码
    direction      TEXT,                    -- 方向：BUY / SELL
    offset         TEXT,                    -- 开平：OPEN / CLOSE / CLOSETODAY
    volume         INTEGER,                 -- 成交手数
    price          REAL,                    -- 成交价格
    commission     REAL,                    -- 手续费（元）
    order_id       TEXT    NOT NULL,        -- 委托单号
    strategy_name  TEXT,                    -- 策略名称（手动填 "manual"）
    notes          TEXT,                    -- 备注
    PRIMARY KEY (trade_date, order_id)
);
CREATE INDEX IF NOT EXISTS idx_tr_date         ON trade_records (trade_date);
CREATE INDEX IF NOT EXISTS idx_tr_symbol_date  ON trade_records (symbol, trade_date);
"""

DAILY_MODEL_OUTPUT_SQL = """
CREATE TABLE IF NOT EXISTS daily_model_output (
    trade_date              TEXT    NOT NULL,   -- 交易日 YYYYMMDD
    underlying              TEXT    NOT NULL,   -- 标的品种，如 "IM" / "IF"
    garch_current_vol       REAL,               -- GARCH 当日条件波动率（年化，如 0.22）
    garch_forecast_vol      REAL,               -- GARCH 预测未来N日波动率（年化）
    realized_vol_20d        REAL,               -- 20日已实现波动率（年化）
    realized_vol_60d        REAL,               -- 60日已实现波动率（年化）
    atm_iv                  REAL,               -- 平值隐含波动率（年化）
    vrp                     REAL,               -- 波动率风险溢价 = ATM_IV - GARCH_forecast
    vrp_percentile          REAL,               -- VRP 在历史中的百分位（0~100）
    signal                  TEXT,               -- 信号建议：SELL_VOL / BUY_VOL / NEUTRAL
    net_delta               REAL,               -- 组合净 Delta（元/点）
    net_gamma               REAL,               -- 组合净 Gamma（元/点²）
    net_theta               REAL,               -- 组合净 Theta（元/天）
    net_vega                REAL,               -- 组合净 Vega（元/1%σ）
    discount_rate_iml1      REAL,               -- IML1 原始贴水率（负值=贴水，如 -0.012）
    discount_rate_iml2      REAL,               -- IML2 原始贴水率
    discount_rate_iml3      REAL,               -- IML3 原始贴水率
    discount_signal         TEXT,               -- 贴水信号强度：STRONG/MEDIUM/WEAK/NONE
    recommended_contract    TEXT,               -- 推荐合约月份，如 IM2606
    garch_5d_forecast_date  TEXT,               -- 本条预测的日期（= trade_date，方便回溯查询）
    rv_5d_actual            REAL,               -- 5交易日后回填的实际5日RV（年化）
    forecast_error          REAL,               -- 预测误差 = garch_forecast_vol - rv_5d_actual
    atm_iv_market           REAL,               -- 市场ATM IV（期货价格based，用于VRP/情绪）
    pnl_total               REAL,               -- 当日总盈亏（元）= 权益变动
    pnl_realized            REAL,               -- 已实现盈亏（平仓）
    pnl_unrealized          REAL,               -- 未实现盈亏变动
    pnl_delta               REAL,               -- Delta 归因（元）= BOD_delta × ΔS
    pnl_gamma               REAL,               -- Gamma 归因（元）= 0.5 × BOD_gamma × ΔS²
    pnl_theta               REAL,               -- Theta 归因（元）= BOD_theta × 1day
    pnl_vega                REAL,               -- Vega 归因（元）= BOD_vega × ΔIV(pp)
    pnl_residual            REAL,               -- 残差 = total - (delta+gamma+theta+vega)
    garch_reliable          INTEGER,            -- GARCH是否可靠（1=可靠，0=偏高，GARCH>max(RV5,RV20)×1.4）
    PRIMARY KEY (trade_date, underlying)
);
CREATE INDEX IF NOT EXISTS idx_dmo_date        ON daily_model_output (trade_date);
CREATE INDEX IF NOT EXISTS idx_dmo_underlying  ON daily_model_output (underlying);
"""

DAILY_MODEL_OUTPUT_ALTER_SQLS = [
    "ALTER TABLE daily_model_output ADD COLUMN discount_rate_iml1      REAL",
    "ALTER TABLE daily_model_output ADD COLUMN discount_rate_iml2      REAL",
    "ALTER TABLE daily_model_output ADD COLUMN discount_rate_iml3      REAL",
    "ALTER TABLE daily_model_output ADD COLUMN discount_signal         TEXT",
    "ALTER TABLE daily_model_output ADD COLUMN recommended_contract    TEXT",
    "ALTER TABLE daily_model_output ADD COLUMN garch_5d_forecast_date TEXT",
    "ALTER TABLE daily_model_output ADD COLUMN rv_5d_actual            REAL",
    "ALTER TABLE daily_model_output ADD COLUMN forecast_error          REAL",
    "ALTER TABLE daily_model_output ADD COLUMN rr_25d                  REAL",
    "ALTER TABLE daily_model_output ADD COLUMN bf_25d                  REAL",
    "ALTER TABLE daily_model_output ADD COLUMN iv_percentile_60d       REAL",
    "ALTER TABLE daily_model_output ADD COLUMN vrp_percentile_60d      REAL",
    "ALTER TABLE daily_model_output ADD COLUMN term_structure_shape    TEXT",
    "ALTER TABLE daily_model_output ADD COLUMN atm_iv_market           REAL",
    "ALTER TABLE daily_model_output ADD COLUMN pnl_total               REAL",
    "ALTER TABLE daily_model_output ADD COLUMN pnl_realized            REAL",
    "ALTER TABLE daily_model_output ADD COLUMN pnl_unrealized          REAL",
    "ALTER TABLE daily_model_output ADD COLUMN pnl_delta               REAL",
    "ALTER TABLE daily_model_output ADD COLUMN pnl_gamma               REAL",
    "ALTER TABLE daily_model_output ADD COLUMN pnl_theta               REAL",
    "ALTER TABLE daily_model_output ADD COLUMN pnl_vega                REAL",
    "ALTER TABLE daily_model_output ADD COLUMN pnl_residual            REAL",
    "ALTER TABLE daily_model_output ADD COLUMN iv_percentile_hist      REAL",
    "ALTER TABLE daily_model_output ADD COLUMN signal_primary          TEXT",
    "ALTER TABLE daily_model_output ADD COLUMN garch_reliable          INTEGER",
    "ALTER TABLE daily_model_output ADD COLUMN hurst_60d               REAL",
    "ALTER TABLE daily_model_output ADD COLUMN iv_term_spread          REAL",
    "ALTER TABLE daily_model_output ADD COLUMN realized_vol_5d         REAL",
]

INDEX_DAILY_SQL = """
CREATE TABLE IF NOT EXISTS index_daily (
    ts_code      TEXT    NOT NULL,          -- 指数代码，如 000852.SH
    trade_date   TEXT    NOT NULL,          -- 交易日 YYYYMMDD
    open         REAL,
    high         REAL,
    low          REAL,
    close        REAL,
    volume       REAL,                      -- 成交量
    amount       REAL,                      -- 成交额（万元）
    PRIMARY KEY (ts_code, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_id_trade_date ON index_daily (trade_date);
CREATE INDEX IF NOT EXISTS idx_id_ts_code    ON index_daily (ts_code);
"""

# ---------------------------------------------------------------------------
# 日报持久化
# ---------------------------------------------------------------------------

DAILY_REPORTS_SQL = """
CREATE TABLE IF NOT EXISTS daily_reports (
    trade_date   TEXT    NOT NULL,
    report_type  TEXT    NOT NULL,           -- 'eod' / 'analysis'
    content      TEXT,                       -- 完整 Markdown 文本
    created_at   TEXT,
    PRIMARY KEY (trade_date, report_type)
);
"""

# ---------------------------------------------------------------------------
# 日内策略盘中记录表
# ---------------------------------------------------------------------------

ORDERBOOK_SNAPSHOTS_SQL = """
CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    symbol       TEXT    NOT NULL,           -- IF / IH / IM
    datetime     TEXT    NOT NULL,           -- UTC 时间戳 YYYY-MM-DD HH:MM:SS
    bid_price1   REAL,
    ask_price1   REAL,
    bid_volume1  REAL,
    ask_volume1  REAL,
    last_price   REAL,
    volume       REAL,                       -- 当根K线成交量
    PRIMARY KEY (symbol, datetime)
);
CREATE INDEX IF NOT EXISTS idx_obs_dt ON orderbook_snapshots (datetime);
"""

SIGNAL_LOG_SQL = """
CREATE TABLE IF NOT EXISTS signal_log (
    datetime          TEXT    NOT NULL,      -- UTC 时间戳
    symbol            TEXT    NOT NULL,      -- IF / IH / IM
    direction         TEXT,                  -- LONG / SHORT / NULL
    score             INT,                   -- 总分（0 = 无信号）
    score_breakout    INT,
    score_vwap        INT,
    score_multiframe  INT,
    score_volume      INT,
    score_daily       INT,
    score_orderbook   INT,
    action_taken      TEXT,                  -- OPEN / SKIP / NONE
    reason            TEXT,
    PRIMARY KEY (datetime, symbol)
);
CREATE INDEX IF NOT EXISTS idx_sl_dt ON signal_log (datetime);
"""

SIGNAL_LOG_ALTER_SQLS = [
    "ALTER TABLE signal_log ADD COLUMN score_v2        INT",
    "ALTER TABLE signal_log ADD COLUMN direction_v2    TEXT",
    "ALTER TABLE signal_log ADD COLUMN score_v3        INT",
    "ALTER TABLE signal_log ADD COLUMN direction_v3    TEXT",
    "ALTER TABLE signal_log ADD COLUMN style_v3        TEXT",
    "ALTER TABLE signal_log ADD COLUMN signal_version  TEXT",
    "ALTER TABLE signal_log ADD COLUMN s_momentum      INT",
    "ALTER TABLE signal_log ADD COLUMN s_volatility    INT",
    "ALTER TABLE signal_log ADD COLUMN s_quality       INT",
    "ALTER TABLE signal_log ADD COLUMN intraday_filter REAL",
    "ALTER TABLE signal_log ADD COLUMN time_mult       REAL",
    "ALTER TABLE signal_log ADD COLUMN sentiment_mult  REAL",
    "ALTER TABLE signal_log ADD COLUMN z_score         REAL",
    "ALTER TABLE signal_log ADD COLUMN rsi             REAL",
    "ALTER TABLE signal_log ADD COLUMN raw_score       INT",
    "ALTER TABLE signal_log ADD COLUMN filtered_score  INT",
    "ALTER TABLE signal_log ADD COLUMN filter_reason   TEXT",
]

TRADE_DECISIONS_SQL = """
CREATE TABLE IF NOT EXISTS trade_decisions (
    datetime          TEXT    NOT NULL PRIMARY KEY,
    symbol            TEXT,
    signal_score      INT,
    signal_direction  TEXT,
    decision          TEXT,                  -- EXECUTED / SKIPPED / MANUAL_OVERRIDE
    manual_note       TEXT,                  -- 手动备注
    created_at        TEXT                   -- 记录时间
);
"""

DAILY_NOTES_SQL = """
CREATE TABLE IF NOT EXISTS daily_notes (
    trade_date          TEXT    PRIMARY KEY,    -- 交易日 YYYYMMDD
    market_observation  TEXT,                   -- 市场观察
    trade_rationale     TEXT,                   -- 交易理由
    deviations          TEXT,                   -- 偏离策略规则的操作及原因
    lessons             TEXT,                   -- 当日复盘总结
    created_at          TEXT                    -- 创建时间 YYYY-MM-DD HH:MM:SS
);
"""

VOL_MONITOR_SNAPSHOTS_SQL = """
CREATE TABLE IF NOT EXISTS vol_monitor_snapshots (
    datetime              TEXT    PRIMARY KEY,
    atm_iv                REAL,
    atm_iv_change         REAL,
    rv_20d                REAL,
    rv_5d                 REAL,
    vrp                   REAL,
    garch_sigma           REAL,
    iv_percentile         REAL,
    vrp_percentile        REAL,
    iv_m1                 REAL,
    iv_m2                 REAL,
    iv_m3                 REAL,
    term_structure_shape  TEXT,
    rr_25d                REAL,
    bf_25d                REAL,
    rr_change             REAL,
    bf_change             REAL,
    net_delta             REAL,
    net_gamma             REAL,
    net_theta             REAL,
    net_vega              REAL,
    pnl_change            REAL,
    spot_price            REAL,
    discount_iml1         REAL,
    discount_iml2         REAL
);
"""

INDEX_MIN_SQL = """
CREATE TABLE IF NOT EXISTS index_min (
    symbol       TEXT    NOT NULL,          -- 指数代码 000852/000300/000016/000905
    datetime     TEXT    NOT NULL,
    period       INT     NOT NULL,          -- 300=5m, 900=15m
    open         REAL,
    high         REAL,
    low          REAL,
    close        REAL,
    volume       REAL,
    open_interest REAL,
    PRIMARY KEY (symbol, datetime, period)
);
CREATE INDEX IF NOT EXISTS idx_im_symbol   ON index_min (symbol);
CREATE INDEX IF NOT EXISTS idx_im_datetime ON index_min (datetime);
"""

# ---------------------------------------------------------------------------
# Morning Briefing 历史数据表
# ---------------------------------------------------------------------------

GLOBAL_INDEX_DAILY_SQL = """
CREATE TABLE IF NOT EXISTS global_index_daily (
    trade_date TEXT,
    ts_code    TEXT,
    close      REAL,
    pct_chg    REAL,
    PRIMARY KEY (trade_date, ts_code)
);
"""

MARKET_BREADTH_SQL = """
CREATE TABLE IF NOT EXISTS market_breadth (
    trade_date    TEXT PRIMARY KEY,
    advance_count INT,
    decline_count INT,
    limit_up      INT,
    limit_down    INT,
    ad_ratio      REAL,
    total_stocks  INT
);
"""

MARKET_TURNOVER_SQL = """
CREATE TABLE IF NOT EXISTS market_turnover (
    trade_date     TEXT PRIMARY KEY,
    sh_amount      REAL,
    sz_amount      REAL,
    total_amount   REAL
);
"""

NORTHBOUND_FLOW_SQL = """
CREATE TABLE IF NOT EXISTS northbound_flow (
    trade_date   TEXT PRIMARY KEY,
    north_money  REAL,
    south_money  REAL
);
"""

MARGIN_DATA_SQL = """
CREATE TABLE IF NOT EXISTS margin_data (
    trade_date TEXT PRIMARY KEY,
    rzye       REAL,
    rzye_chg   REAL
);
"""

ETF_SHARE_SQL = """
CREATE TABLE IF NOT EXISTS etf_share (
    trade_date   TEXT,
    ts_code      TEXT,
    fd_share     REAL,
    fd_share_chg REAL,
    PRIMARY KEY (trade_date, ts_code)
);
"""

FUT_HOLDING_SUMMARY_SQL = """
CREATE TABLE IF NOT EXISTS fut_holding_summary (
    trade_date   TEXT,
    symbol       TEXT,
    total_long   REAL,
    total_short  REAL,
    net_position REAL,
    long_chg     REAL,
    short_chg    REAL,
    net_chg      REAL,
    PRIMARY KEY (trade_date, symbol)
);
"""

OPTION_PCR_DAILY_SQL = """
CREATE TABLE IF NOT EXISTS option_pcr_daily (
    trade_date    TEXT,
    product       TEXT,
    call_volume   REAL,
    put_volume    REAL,
    total_volume  REAL,
    volume_pcr    REAL,
    call_oi       REAL,
    put_oi        REAL,
    total_oi      REAL,
    oi_pcr        REAL,
    PRIMARY KEY (trade_date, product)
);
"""

SHADOW_TRADES_SQL = """
CREATE TABLE IF NOT EXISTS shadow_trades (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date       TEXT,
    symbol           TEXT,
    direction        TEXT,
    entry_time       TEXT,
    entry_price      REAL,
    entry_score      INT,
    entry_dm         REAL,
    entry_f          REAL,
    entry_t          REAL,
    entry_s          REAL,
    entry_m          INT,
    entry_v          INT,
    entry_q          INT,
    exit_time        TEXT,
    exit_price       REAL,
    exit_reason      TEXT,
    pnl_pts          REAL,
    hold_minutes     INT,
    operator_action  TEXT,
    is_executed      INT DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_shadow_date ON shadow_trades (trade_date);
"""

EXECUTOR_LOG_SQL = """
CREATE TABLE IF NOT EXISTS executor_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_time       TEXT,              -- monitor产生信号的时间
    receive_time      TEXT,              -- executor收到JSON的时间
    symbol            TEXT,
    direction         TEXT,              -- LONG / SHORT
    action            TEXT,              -- OPEN / CLOSE
    score             INT,
    reason            TEXT,              -- 信号原因（exit reason for CLOSE）
    limit_price       REAL,
    suggested_lots    INT,               -- monitor建议手数
    actual_lots       INT,               -- 实际下单手数（÷2后）
    operator_response TEXT,              -- Y / N / TIMEOUT
    response_reason   TEXT,              -- 备注
    order_submitted   INT DEFAULT 0,     -- 是否提交了TQ订单
    order_id          TEXT,              -- TQ订单ID
    filled_lots       INT DEFAULT 0,     -- 实际成交手数
    filled_price      REAL,              -- 成交均价
    order_status      TEXT,              -- FILLED/PARTIAL/CANCELLED/TIMEOUT_CANCEL/EXPIRED/NOT_SUBMITTED
    cancel_time       TEXT,
    cancel_reason     TEXT,
    signal_json       TEXT               -- 原始JSON完整内容（留底）
);
CREATE INDEX IF NOT EXISTS idx_execlog_time ON executor_log (receive_time);
CREATE INDEX IF NOT EXISTS idx_execlog_sym ON executor_log (symbol, receive_time);
"""

# ---------------------------------------------------------------------------
# 期权分钟线（MO 5分钟K线）
# ---------------------------------------------------------------------------

OPTIONS_MIN_SQL = """
CREATE TABLE IF NOT EXISTS options_min (
    ts_code      TEXT    NOT NULL,          -- 期权合约代码 MO2605-C-7000.CFX
    datetime     TEXT    NOT NULL,          -- 时间戳 YYYY-MM-DD HH:MM:SS
    period       INT     NOT NULL,          -- 周期秒数 300=5m
    open         REAL,
    high         REAL,
    low          REAL,
    close        REAL,
    volume       REAL,                      -- 成交量（手）
    open_interest REAL,                     -- 持仓量
    PRIMARY KEY (ts_code, datetime, period)
);
CREATE INDEX IF NOT EXISTS idx_om_tscode   ON options_min (ts_code);
CREATE INDEX IF NOT EXISTS idx_om_datetime ON options_min (datetime);
"""

# ---------------------------------------------------------------------------
# 期货 Tick 数据（单独 DB：tick_data.db）
# schema 定义放这里方便统一管理，建表在 download_tick_pro.py 中执行
# ---------------------------------------------------------------------------

FUTURES_TICK_SQL = """
CREATE TABLE IF NOT EXISTS futures_tick (
    symbol       TEXT    NOT NULL,          -- 品种代码 IM/IC/IF/IH
    datetime     TEXT    NOT NULL,          -- 纳秒精度时间 YYYY-MM-DD HH:MM:SS.ffffff
    last_price   REAL,                      -- 最新价
    average      REAL,                      -- 当日均价
    highest      REAL,                      -- 当日最高价
    lowest       REAL,                      -- 当日最低价
    bid_price1   REAL,                      -- 买一价
    bid_volume1  INT,                       -- 买一量
    ask_price1   REAL,                      -- 卖一价
    ask_volume1  INT,                       -- 卖一量
    volume       INT,                       -- 当日累计成交量
    amount       REAL,                      -- 当日累计成交额
    open_interest INT,                      -- 持仓量
    PRIMARY KEY (symbol, datetime)
);
CREATE INDEX IF NOT EXISTS idx_ft_symbol   ON futures_tick (symbol);
CREATE INDEX IF NOT EXISTS idx_ft_datetime ON futures_tick (datetime);
"""

# ---------------------------------------------------------------------------
# ETF 分钟线（独立存放于 etf_data.db）
# ---------------------------------------------------------------------------

ETF_MIN_SQL = """
CREATE TABLE IF NOT EXISTS etf_min (
    symbol       TEXT    NOT NULL,          -- ETF代码 512100/510500/510300/510050
    datetime     TEXT    NOT NULL,
    period       INT     NOT NULL,          -- 周期秒数 300=5m
    open         REAL,
    high         REAL,
    low          REAL,
    close        REAL,
    volume       REAL,                      -- 成交量（股）
    open_interest REAL,
    PRIMARY KEY (symbol, datetime, period)
);
CREATE INDEX IF NOT EXISTS idx_etf_symbol   ON etf_min (symbol);
CREATE INDEX IF NOT EXISTS idx_etf_datetime ON etf_min (datetime);
"""

# ---------------------------------------------------------------------------
# 期权表（独立存放于 options_data.db）
# ---------------------------------------------------------------------------

OPTIONS_TABLES: list[str] = [
    OPTIONS_DAILY_SQL,
    OPTIONS_CONTRACTS_SQL,
    OPTIONS_MIN_SQL,
]

TICK_TABLES: list[str] = [
    FUTURES_TICK_SQL,
]

ETF_TABLES: list[str] = [
    ETF_MIN_SQL,
]

# ---------------------------------------------------------------------------
# 主库表（trading.db）— 不含期权表
# ---------------------------------------------------------------------------

ALL_TABLES: list[str] = [
    FUTURES_DAILY_SQL,
    FUTURES_MIN_SQL,
    COMMODITY_DAILY_SQL,
    TRADE_CALENDAR_SQL,
    INDEX_DAILY_SQL,
    # 每日记录表
    TQ_SNAPSHOTS_SQL,
    ACCOUNT_SNAPSHOTS_SQL,
    POSITION_SNAPSHOTS_SQL,
    TRADE_RECORDS_SQL,
    DAILY_MODEL_OUTPUT_SQL,
    DAILY_NOTES_SQL,
    # 策略层
    STRATEGY_SIGNALS_SQL,
    STRATEGY_TRADES_SQL,
    STRATEGY_PNL_SQL,
    # 日报持久化
    DAILY_REPORTS_SQL,
    # 日内盘中记录
    ORDERBOOK_SNAPSHOTS_SQL,
    SIGNAL_LOG_SQL,
    TRADE_DECISIONS_SQL,
    # 波动率监控
    VOL_MONITOR_SNAPSHOTS_SQL,
    # 现货指数分钟线
    INDEX_MIN_SQL,
    # Morning Briefing 历史数据
    GLOBAL_INDEX_DAILY_SQL,
    MARKET_BREADTH_SQL,
    MARKET_TURNOVER_SQL,
    NORTHBOUND_FLOW_SQL,
    MARGIN_DATA_SQL,
    ETF_SHARE_SQL,
    FUT_HOLDING_SUMMARY_SQL,
    OPTION_PCR_DAILY_SQL,
    # 日内影子交易簿（记录所有信号完整生命周期）
    SHADOW_TRADES_SQL,
    # Executor完整信号记录
    EXECUTOR_LOG_SQL,
]

# 向后兼容别名（db_manager.py 等现有代码使用 ALL_DDL）
ALL_DDL = ALL_TABLES
