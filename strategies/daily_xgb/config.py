"""Daily XGB strategy configuration — 保守模式锁死.

参数来源：walk-forward 验证 (2.9 yr, 273 trades, Sharpe 5.64).
保守模式 = 每信号 1 手 + 同时持仓上限 10 手.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DailyXGBConfig:
    # ── Identity ──
    strategy_id: str = "DAILY_XGB"
    underlying: str = "IM"
    contract_mult: int = 200
    margin_per_lot: float = 260_000.0

    # ── Account ──
    account_equity: float = 6_400_000.0   # 默认假设, runtime 用真实

    # ── Conservative mode (LOCKED — no toggle) ──
    lots_per_signal: int = 1
    concurrent_cap: int = 10

    # ── Signal generation ──
    top_pct: float = 0.20         # P80 LONG threshold (in-sample)
    bot_pct: float = 0.20         # P20 SHORT threshold
    initial_train_days: int = 200
    retrain_every: int = 20

    # ── Hold defaults ──
    hold_days_default: int = 5
    atr_k_default: float = 1.5

    # ── G3s SHORT block (bull regime) ──
    g3s_strict_sma60_thr: float = 1.04
    g3s_strict_sma200_thr: float = 1.05

    # ── N5 LONG enhancement ──
    n5_strict_sma60_thr: float = 1.04
    n5_strict_sma200_thr: float = 1.05
    n5_strict_hold_days: int = 10
    n5_strict_atr_k: float = 4.0

    n5_dip_sma200_thr: float = 1.03
    n5_dip_sma60_thr: float = 1.02
    n5_dip_hold_days: int = 15
    n5_dip_atr_k: float = 4.0

    # ── M7 SHORT enhancement ──
    m7_extended_sma60_thr: float = 0.97
    m7_extended_sma200_thr: float = 0.97
    m7_extended_hold_days: int = 10
    m7_extended_atr_k: float = 4.0

    m7_rip_sma200_thr: float = 0.99
    m7_rip_sma60_thr: float = 0.98
    m7_rip_hold_days: int = 15
    m7_rip_atr_k: float = 4.0

    # ── Risk circuit breaker ──
    weekly_dd_threshold_pct: float = 0.05
    margin_usage_max_pct: float = 0.85
    daily_loss_threshold_pct: float = 0.03

    # ── Execution ──
    slippage_pct: float = 0.0008          # 单边滑点（仅 backtest 用）
    open_window_start_bj: str = "09:25"   # 集合竞价
    open_window_end_bj: str = "09:35"     # 5min after open 不再追单
    eod_check_time_bj: str = "14:55"      # EOD SL 检查时间
    order_aggressive_offset_pts: float = 2.0   # 限价单激进价偏移

    # ── DB / IO paths ──
    db_path: str = field(default_factory=lambda: str(PROJECT_ROOT / "data/storage/trading.db"))
    pending_signal_path: str = field(default_factory=lambda: str(PROJECT_ROOT / "tmp/daily_xgb_pending.json"))
    positions_path: str = field(default_factory=lambda: str(PROJECT_ROOT / "tmp/daily_xgb_positions.json"))
    kill_switch_file: str = field(default_factory=lambda: str(PROJECT_ROOT / "tmp/daily_xgb_kill.flag"))
    log_dir: str = field(default_factory=lambda: str(PROJECT_ROOT / "logs/daily_xgb"))

    # ── XGBoost hyperparams (from validated factor pipeline) ──
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.03
    xgb_min_child_weight: int = 10
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.7
    xgb_reg_alpha: float = 0.5
    xgb_reg_lambda: float = 2.0
    xgb_random_state: int = 42

    # ── TQ executor ──
    tq_order_prefix: str = "DXGB_"
    tq_open_fill_timeout_s: int = 60
    tq_close_fill_timeout_s: int = 30


def default_config() -> DailyXGBConfig:
    """Return conservative-mode config (singleton-style for production)."""
    return DailyXGBConfig()
