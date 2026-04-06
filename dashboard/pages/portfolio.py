"""
portfolio.py
------------
Dashboard 页面：组合总览

功能:
- 账户 KPI 卡片（权益/日盈亏/保证金/浮盈/VRP信号）
- 净值曲线
- 当前持仓表格
- Greeks 摘要卡片
- 近期成交记录
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# ──────────────────────────────────────────────
# DB helpers
# ──────────────────────────────────────────────

@st.cache_resource
def _get_db():
    try:
        from data.storage.db_manager import DBManager, get_db
        from config.config_loader import ConfigLoader
        config = ConfigLoader()
        db = get_db(config)
        db.initialize_tables()
        return db
    except Exception as e:
        st.error(f"数据库连接失败: {e}")
        return None


def _load_latest_account(db) -> dict | None:
    try:
        df = db.query_df(
            "SELECT * FROM account_snapshots ORDER BY trade_date DESC, snapshot_time DESC LIMIT 2"
        )
        if df is None or df.empty:
            return None
        return df.iloc[0].to_dict()
    except Exception:
        return None


def _load_equity_history(db) -> pd.DataFrame:
    try:
        df = db.query_df(
            "SELECT trade_date, balance FROM account_snapshots ORDER BY trade_date ASC"
        )
        if df is None:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


def _load_positions(db) -> pd.DataFrame:
    try:
        df = db.query_df(
            """SELECT * FROM position_snapshots
               WHERE trade_date = (SELECT MAX(trade_date) FROM position_snapshots)"""
        )
        if df is None:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


def _load_latest_model(db) -> dict | None:
    try:
        df = db.query_df(
            "SELECT * FROM daily_model_output ORDER BY trade_date DESC LIMIT 1"
        )
        if df is None or df.empty:
            return None
        return df.iloc[0].to_dict()
    except Exception:
        return None


def _load_prev_account(db) -> dict | None:
    """Load second-to-last account snapshot for delta calculation."""
    try:
        df = db.query_df(
            "SELECT * FROM account_snapshots ORDER BY trade_date DESC, snapshot_time DESC LIMIT 2"
        )
        if df is None or len(df) < 2:
            return None
        return df.iloc[1].to_dict()
    except Exception:
        return None


def _load_recent_trades(db, n: int = 10) -> pd.DataFrame:
    try:
        df = db.query_df(
            f"""SELECT trade_date, trade_time, symbol, direction, offset, volume, price, commission
                FROM trade_records
                ORDER BY trade_date DESC, trade_time DESC
                LIMIT {n}"""
        )
        if df is None:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


# ──────────────────────────────────────────────
# Render
# ──────────────────────────────────────────────

def render() -> None:
    st.title("📊 组合总览")

    db = _get_db()
    if db is None:
        st.error("无法连接数据库，请检查配置。")
        return

    account = _load_latest_account(db)
    prev_account = _load_prev_account(db)
    equity_df = _load_equity_history(db)
    positions_df = _load_positions(db)
    model_data = _load_latest_model(db)
    trades_df = _load_recent_trades(db)

    latest_date = account.get("trade_date", "N/A") if account else "N/A"
    st.caption(f"数据更新至: {latest_date}")

    # ── KPI Row ──
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        balance = account.get("balance", None) if account else None
        val = f"¥{balance:,.2f}" if balance is not None else "N/A"
        st.metric("账户权益", val)

    with col2:
        today_pnl = None
        delta_str = None
        if account:
            close_p = account.get("close_profit", 0) or 0
            float_p = account.get("float_profit", 0) or 0
            today_pnl = close_p + float_p
            if prev_account:
                prev_pnl = (prev_account.get("close_profit", 0) or 0) + (prev_account.get("float_profit", 0) or 0)
                delta_str = f"{today_pnl - prev_pnl:+,.2f}"
        val = f"¥{today_pnl:,.2f}" if today_pnl is not None else "N/A"
        st.metric("今日盈亏", val, delta=delta_str,
                  delta_color="inverse")  # A-share: profit=red, loss=green

    with col3:
        margin_ratio = account.get("margin_ratio", None) if account else None
        if margin_ratio is None:
            margin = account.get("margin", None) if account else None
            bal = account.get("balance", None) if account else None
            if margin is not None and bal and bal > 0:
                margin_ratio = margin / bal
        val = f"{margin_ratio*100:.1f}%" if margin_ratio is not None else "N/A"
        st.metric("保证金占用%", val)

    with col4:
        float_p = account.get("float_profit", None) if account else None
        val = f"¥{float_p:,.2f}" if float_p is not None else "N/A"
        st.metric("浮动盈亏", val)

    with col5:
        if model_data:
            vrp = model_data.get("vrp", None)
            signal = model_data.get("signal", None)
            if vrp is not None:
                arrow = "▲" if vrp > 0 else "▼"
                st.metric("VRP 信号", f"{vrp:.4f} {arrow}",
                          delta=str(signal) if signal else None)
            else:
                st.metric("VRP 信号", "待计算")
        else:
            st.metric("VRP 信号", "待计算")

    st.divider()

    # ── Equity curve + Positions ──
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("净值曲线")
        if equity_df.empty or len(equity_df) < 3:
            st.info("数据积累中，需要更多交易日数据")
            if not equity_df.empty:
                st.dataframe(equity_df, use_container_width=True)
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_df["trade_date"],
                y=equity_df["balance"],
                mode="lines",
                name="账户权益",
                line=dict(color="#3498db", width=2),
                fill="tozeroy",
                fillcolor="rgba(52,152,219,0.1)",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", title="权益 (元)"),
                margin=dict(l=10, r=10, t=30, b=10),
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

    with right_col:
        st.subheader("当前持仓")
        if positions_df.empty:
            st.info("暂无持仓数据")
        else:
            display_cols = [c for c in ["symbol", "direction", "volume", "open_price_avg",
                                         "last_price", "float_profit", "margin"] if c in positions_df.columns]
            disp_df = positions_df[display_cols].copy() if display_cols else positions_df.copy()

            col_labels = {
                "symbol": "合约",
                "direction": "方向",
                "volume": "手数",
                "open_price_avg": "均价",
                "last_price": "最新价",
                "float_profit": "浮动盈亏",
                "margin": "保证金",
            }
            disp_df = disp_df.rename(columns={k: v for k, v in col_labels.items() if k in disp_df.columns})

            def _color_profit(val):
                try:
                    v = float(val)
                    if v > 0:
                        return "color: #e74c3c"  # red = profit (A-share)
                    elif v < 0:
                        return "color: #2ecc71"  # green = loss
                    return ""
                except Exception:
                    return ""

            if "浮动盈亏" in disp_df.columns:
                styled = disp_df.style.applymap(_color_profit, subset=["浮动盈亏"])
                st.dataframe(styled, use_container_width=True, height=300)
            else:
                st.dataframe(disp_df, use_container_width=True, height=300)

    st.divider()

    # ── Greeks + Recent Trades ──
    left_col2, right_col2 = st.columns(2)

    with left_col2:
        st.subheader("组合 Greeks")
        g1, g2, g3, g4 = st.columns(4)
        if model_data:
            nd = model_data.get("net_delta")
            ng = model_data.get("net_gamma")
            nt = model_data.get("net_theta")
            nv = model_data.get("net_vega")
            g1.metric("Net Delta", f"{nd:.4f}" if nd is not None else "待计算")
            g2.metric("Net Gamma", f"{ng:.6f}" if ng is not None else "待计算")
            g3.metric("Net Theta", f"{nt:.2f}" if nt is not None else "待计算")
            g4.metric("Net Vega", f"{nv:.4f}" if nv is not None else "待计算")
        else:
            g1.metric("Net Delta", "待计算")
            g2.metric("Net Gamma", "待计算")
            g3.metric("Net Theta", "待计算")
            g4.metric("Net Vega", "待计算")

    with right_col2:
        st.subheader("近期成交")
        if trades_df.empty:
            st.info("暂无成交记录")
        else:
            col_labels2 = {
                "trade_date": "日期",
                "trade_time": "时间",
                "symbol": "合约",
                "direction": "方向",
                "offset": "开平",
                "volume": "手数",
                "price": "成交价",
                "commission": "手续费",
            }
            disp = trades_df.rename(columns={k: v for k, v in col_labels2.items() if k in trades_df.columns})
            st.dataframe(disp, use_container_width=True, height=300)


if __name__ == "__main__":
    render()
