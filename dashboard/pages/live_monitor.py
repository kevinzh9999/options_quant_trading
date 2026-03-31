"""
live_monitor.py
---------------
Dashboard 页面：盘中实时监控 (Live Monitor)

每30秒自动刷新，读取 signal_log / orderbook_snapshots / vol_monitor_snapshots。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ── 常量 ──────────────────────────────────────────────
PRODUCT_COLORS = {"IF": "#3498db", "IH": "#e74c3c", "IM": "#e67e22", "IC": "#9b59b6"}
_BG = "rgba(0,0,0,0)"
_GRID = "rgba(255,255,255,0.1)"


# ── DB ────────────────────────────────────────────────

@st.cache_resource
def _get_db():
    try:
        from data.storage.db_manager import DBManager
        from config.config_loader import ConfigLoader
        db = DBManager(ConfigLoader().get_db_path())
        db.initialize_tables()
        return db
    except Exception as e:
        st.error(f"数据库连接失败: {e}")
        return None


def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


# ── 数据加载 ──────────────────────────────────────────

def _load_signals(db, date_prefix: str) -> pd.DataFrame:
    df = db.query_df(
        f"SELECT * FROM signal_log WHERE datetime LIKE '{date_prefix}%' ORDER BY datetime"
    )
    return df if df is not None else pd.DataFrame()


def _load_orderbook(db, date_prefix: str) -> pd.DataFrame:
    df = db.query_df(
        f"SELECT * FROM orderbook_snapshots WHERE datetime LIKE '{date_prefix}%' ORDER BY datetime"
    )
    return df if df is not None else pd.DataFrame()


def _load_vol_snapshots(db, date_prefix: str) -> pd.DataFrame:
    df = db.query_df(
        f"SELECT * FROM vol_monitor_snapshots WHERE datetime LIKE '{date_prefix}%' ORDER BY datetime"
    )
    return df if df is not None else pd.DataFrame()


def _load_trade_decisions(db, date_prefix: str) -> pd.DataFrame:
    df = db.query_df(
        f"SELECT * FROM trade_decisions WHERE datetime LIKE '{date_prefix}%' ORDER BY datetime"
    )
    return df if df is not None else pd.DataFrame()


def _load_positions(db) -> pd.DataFrame:
    df = db.query_df(
        "SELECT * FROM position_snapshots "
        "WHERE trade_date = (SELECT MAX(trade_date) FROM position_snapshots)"
    )
    return df if df is not None else pd.DataFrame()


def _load_spot(db) -> float | None:
    df = db.query_df(
        "SELECT close FROM index_daily WHERE ts_code='000852.SH' ORDER BY trade_date DESC LIMIT 1"
    )
    if df is not None and not df.empty:
        return float(df["close"].iloc[0])
    return None


# ── 信号面板 ──────────────────────────────────────────

def _render_signal_panel(signals: pd.DataFrame):
    st.subheader("信号面板")

    if signals.empty:
        st.info("今日暂无盘中信号")
        return

    # 最新一轮信号（最后一个时间戳）
    latest_time = signals["datetime"].max()
    latest = signals[signals["datetime"] == latest_time].copy()

    rows = []
    for _, r in latest.iterrows():
        # 确定方向和得分
        direction = r.get("direction") or r.get("direction_v2") or r.get("direction_v3") or ""
        score = int(r.get("score") or r.get("score_v2") or r.get("score_v3") or 0)
        version = r.get("signal_version") or ""

        # 方向符号
        if direction == "LONG":
            dir_icon = "▲多"
        elif direction == "SHORT":
            dir_icon = "▼空"
        else:
            dir_icon = "—"

        # 维度明细
        dims = []
        for col, label in [
            ("score_breakout", "突破"),
            ("score_vwap", "VWAP"),
            ("score_multiframe", "多TF"),
            ("score_volume", "量"),
            ("score_daily", "日线"),
            ("score_orderbook", "盘口"),
        ]:
            v = r.get(col)
            if v and int(v) != 0:
                dims.append(f"{label}{int(v)}")

        rows.append({
            "品种": r.get("symbol", ""),
            "时间": latest_time[11:16] if len(latest_time) > 11 else latest_time,
            "方向": dir_icon,
            "得分": score,
            "版本": version,
            "维度明细": " | ".join(dims) if dims else "—",
            "动作": r.get("action_taken", ""),
        })

    df_display = pd.DataFrame(rows)

    def _color_score(val):
        try:
            v = int(val)
        except (ValueError, TypeError):
            return ""
        if v >= 55:
            return "background-color: rgba(46,204,113,0.3)"
        elif v >= 40:
            return "background-color: rgba(241,196,15,0.2)"
        return ""

    styled = df_display.style.applymap(_color_score, subset=["得分"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # 信号历史折线图
    st.markdown("**今日信号得分走势**")

    fig = go.Figure()
    for symbol in ["IF", "IH", "IM", "IC"]:
        sym_data = signals[signals["symbol"] == symbol].copy()
        if sym_data.empty:
            continue
        scores = sym_data.apply(
            lambda r: int(r.get("score") or r.get("score_v2") or r.get("score_v3") or 0),
            axis=1,
        )
        dirs = sym_data.apply(
            lambda r: r.get("direction") or r.get("direction_v2") or r.get("direction_v3") or "",
            axis=1,
        )
        signed = scores.copy()
        signed[dirs == "SHORT"] = -signed[dirs == "SHORT"]

        fig.add_trace(go.Scatter(
            x=sym_data["datetime"],
            y=signed,
            name=symbol,
            line=dict(color=PRODUCT_COLORS.get(symbol, "#999")),
        ))

    fig.add_hline(y=55, line_dash="dash", line_color="green", opacity=0.5,
                  annotation_text="多阈值55")
    fig.add_hline(y=-55, line_dash="dash", line_color="red", opacity=0.5,
                  annotation_text="空阈值-55")
    fig.add_hline(y=0, line_color="gray", opacity=0.3)
    fig.update_layout(
        height=300,
        plot_bgcolor=_BG, paper_bgcolor=_BG,
        xaxis=dict(gridcolor=_GRID),
        yaxis=dict(title="带方向得分", gridcolor=_GRID),
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=40, r=20, t=40, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── 盘口面板 ──────────────────────────────────────────

def _render_orderbook_panel(ob: pd.DataFrame):
    st.subheader("盘口买卖压力")

    if ob.empty:
        st.info("今日暂无盘口数据")
        return

    # 买卖压力比 = bid_volume / ask_volume
    ob = ob.copy()
    ob["pressure"] = ob["bid_volume1"] / ob["ask_volume1"].replace(0, np.nan)

    fig = go.Figure()
    for symbol in ["IF", "IH", "IM", "IC"]:
        sym_data = ob[ob["symbol"] == symbol]
        if sym_data.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sym_data["datetime"],
            y=sym_data["pressure"],
            name=symbol,
            line=dict(color=PRODUCT_COLORS.get(symbol, "#999")),
        ))

    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text="平衡线")
    fig.update_layout(
        height=280,
        plot_bgcolor=_BG, paper_bgcolor=_BG,
        xaxis=dict(gridcolor=_GRID),
        yaxis=dict(title="买/卖压力比 (>1偏多)", gridcolor=_GRID),
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=40, r=20, t=40, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── 波动率面板 ────────────────────────────────────────

def _render_vol_panel(vol_snaps: pd.DataFrame, positions: pd.DataFrame, spot: float | None, db=None):
    st.subheader("波动率面板")

    if vol_snaps.empty:
        st.info("今日暂无波动率快照")
        return

    col1, col2 = st.columns(2)

    with col1:
        # IV日内走势
        fig = go.Figure()
        if "atm_iv" in vol_snaps.columns:
            fig.add_trace(go.Scatter(
                x=vol_snaps["datetime"], y=vol_snaps["atm_iv"] * 100,
                name="ATM IV (结构)", line=dict(color="#3498db"),
            ))
        if "iv_m1" in vol_snaps.columns:
            m1 = vol_snaps["iv_m1"].dropna()
            if not m1.empty:
                fig.add_trace(go.Scatter(
                    x=vol_snaps.loc[m1.index, "datetime"], y=m1 * 100,
                    name="近月IV", line=dict(color="#e67e22", dash="dot"),
                ))
        if "rv_20d" in vol_snaps.columns:
            fig.add_trace(go.Scatter(
                x=vol_snaps["datetime"], y=vol_snaps["rv_20d"] * 100,
                name="RV20", line=dict(color="#95a5a6", dash="dash"),
            ))
        fig.update_layout(
            title="市场IV日内走势",
            height=280,
            plot_bgcolor=_BG, paper_bgcolor=_BG,
            xaxis=dict(gridcolor=_GRID),
            yaxis=dict(title="波动率 (%)", gridcolor=_GRID),
            legend=dict(orientation="h", y=1.02),
            margin=dict(l=40, r=20, t=50, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Skew日内走势
        fig = go.Figure()
        if "rr_25d" in vol_snaps.columns:
            fig.add_trace(go.Scatter(
                x=vol_snaps["datetime"], y=vol_snaps["rr_25d"] * 100,
                name="25D RR", line=dict(color="#e74c3c"),
            ))
        if "bf_25d" in vol_snaps.columns:
            fig.add_trace(go.Scatter(
                x=vol_snaps["datetime"], y=vol_snaps["bf_25d"] * 100,
                name="25D BF", line=dict(color="#2ecc71"),
            ))
        fig.update_layout(
            title="Skew日内走势",
            height=280,
            plot_bgcolor=_BG, paper_bgcolor=_BG,
            xaxis=dict(gridcolor=_GRID),
            yaxis=dict(title="Skew (%)", gridcolor=_GRID),
            legend=dict(orientation="h", y=1.02),
            margin=dict(l=40, r=20, t=50, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    # IV分位（从volatility_history）
    try:
        iv_hist = db.query_df(
            "SELECT atm_iv FROM volatility_history WHERE atm_iv > 0"
        )
        if iv_hist is not None and not iv_hist.empty and not vol_snaps.empty:
            last_snap = vol_snaps.iloc[-1]
            current_iv = last_snap.get("atm_iv")
            if current_iv and current_iv > 0:
                iv_pct_val = current_iv * 100 if current_iv < 1 else current_iv
                hist_vals = iv_hist["atm_iv"].values
                hist_pct = hist_vals if hist_vals.mean() > 1 else hist_vals * 100
                pctile = float(np.mean(hist_pct <= iv_pct_val) * 100)
                st.metric("ATM IV 历史百分位", f"{pctile:.0f}%",
                          delta=f"IV={iv_pct_val:.1f}%")
    except Exception:
        pass

    # 持仓安全距离
    if not positions.empty and spot is not None:
        st.markdown("**持仓安全距离**")
        _render_safety_distance(positions, spot)


def _render_safety_distance(positions: pd.DataFrame, spot: float):
    """用色块显示每个空头期权到现货的距离"""
    rows = []
    for _, pos in positions.iterrows():
        symbol = str(pos.get("symbol", ""))
        m = re.search(r"MO(\d{4})-(C|P)-(\d+)", symbol)
        if not m:
            continue
        cp = m.group(2)
        strike = float(m.group(3))
        direction = str(pos.get("direction", ""))
        vol = int(pos.get("volume", 0))

        if direction == "SHORT":
            if cp == "P":
                dist = spot - strike
            else:
                dist = strike - spot
            pct = dist / spot * 100

            if dist < 100:
                color = "🔴"
                status = "危险"
            elif dist < 250:
                color = "🟡"
                status = "警戒"
            else:
                color = "🟢"
                status = "安全"

            rows.append({
                "状态": color,
                "合约": symbol.split(".")[-1] if "." in symbol else symbol,
                "方向": f"空{cp}",
                "手数": vol,
                "行权价": f"{strike:.0f}",
                "距离": f"{dist:.0f}点",
                "距离%": f"{pct:.1f}%",
                "评级": status,
            })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("无空头期权持仓")


# ── 交易决策记录 ──────────────────────────────────────

def _render_decisions(decisions: pd.DataFrame):
    st.subheader("交易决策记录")

    if decisions.empty:
        st.info("今日暂无交易决策记录")
        return

    rows = []
    for _, r in decisions.iterrows():
        dt = str(r.get("datetime", ""))
        decision = str(r.get("decision", ""))
        if decision in ("EXECUTED", "OPEN"):
            icon = "✅"
        elif decision == "SKIPPED":
            icon = "⏭️"
        else:
            icon = "📝"

        rows.append({
            "时间": dt[11:19] if len(dt) > 11 else dt,
            "品种": r.get("symbol", ""),
            "方向": r.get("signal_direction", ""),
            "得分": r.get("signal_score", ""),
            "决策": f"{icon} {decision}",
            "备注": r.get("manual_note", "") or "",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── 主渲染 ────────────────────────────────────────────

def render():
    st.title("盘中实时监控")

    # 自动刷新
    try:
        from streamlit_autorefresh import st_autorefresh
        count = st_autorefresh(interval=30_000, limit=None, key="live_refresh")
    except ImportError:
        st.caption("💡 安装 streamlit-autorefresh 可启用自动刷新: pip install streamlit-autorefresh")

    db = _get_db()
    if db is None:
        return

    today = _today_str()
    st.caption(f"数据日期: {today} | 每30秒自动刷新")

    # 加载所有数据
    signals = _load_signals(db, today)
    orderbook = _load_orderbook(db, today)
    vol_snaps = _load_vol_snapshots(db, today)
    decisions = _load_trade_decisions(db, today)
    positions = _load_positions(db)
    spot = _load_spot(db)

    # 布局
    _render_signal_panel(signals)
    st.markdown("---")

    col1, col2 = st.columns([1, 1])
    with col1:
        _render_orderbook_panel(orderbook)
    with col2:
        _render_decisions(decisions)

    st.markdown("---")
    _render_vol_panel(vol_snaps, positions, spot, db)
