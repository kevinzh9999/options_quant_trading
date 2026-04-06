"""
quadrant_history.py
-------------------
策略象限监控历史：象限时间线、各象限P&L、当前象限仪表盘。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.config_loader import ConfigLoader
from data.storage.db_manager import DBManager, get_db


def _load_data():
    db = get_db()
    briefing = db.query_df(
        "SELECT trade_date, direction, confidence, score, ad_ratio, "
        "iv_percentile, vrp, daily_5d_mom, range_position "
        "FROM morning_briefing ORDER BY trade_date"
    )
    spot = db.query_df(
        "SELECT trade_date, close FROM index_daily "
        "WHERE ts_code='000852.SH' ORDER BY trade_date DESC LIMIT 30"
    )
    account = db.query_df(
        "SELECT trade_date, balance FROM account_snapshots ORDER BY trade_date"
    )
    snap = db.query_df(
        "SELECT iv_percentile, vrp, term_structure_shape "
        "FROM vol_monitor_snapshots ORDER BY datetime DESC LIMIT 1"
    )
    return briefing, spot, account, snap, db


def _infer_quadrant(row) -> str:
    """从briefing数据推算象限。"""
    ip = row.get("iv_percentile")
    score = row.get("score", 0)
    vrp = row.get("vrp")

    if ip is None:
        return "?"

    iv_high = ip >= 70
    iv_low = ip <= 30
    bullish = score >= 20
    bearish = score <= -20

    if iv_high:
        if bearish:
            if vrp is not None and ((abs(vrp) < 1 and vrp < 0) or (abs(vrp) >= 1 and vrp < 0)):
                return "B1"
            return "B2"
        elif bullish:
            return "A"
        return "A/B"
    elif iv_low:
        if bullish:
            return "C"
        elif bearish:
            return "D"
        return "C/D"
    else:
        if bullish:
            return "→A"
        elif bearish:
            return "→D"
        return "中"


QUAD_COLORS = {
    "A": "#4CAF50", "A/B": "#8BC34A", "B1": "#F44336", "B2": "#FF9800",
    "C": "#2196F3", "C/D": "#03A9F4", "D": "#9C27B0",
    "→A": "#81C784", "→D": "#CE93D8", "中": "#9E9E9E", "?": "#E0E0E0",
}


def render():
    st.title("策略象限监控")

    try:
        briefing, spot, account, snap, db = _load_data()
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return

    # 当前象限仪表盘
    st.subheader("当前象限")
    if snap is not None and not snap.empty and briefing is not None and not briefing.empty:
        latest = briefing.iloc[-1]
        ip = snap.iloc[0].get("iv_percentile")
        if ip is None or ip == "":
            ip = latest.get("iv_percentile", 50)
        ip = float(ip) if ip else 50

        quad = _infer_quadrant({"iv_percentile": ip, "score": latest["score"],
                                "vrp": latest.get("vrp")})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("当前象限", quad, help="A=高IV多 B=高IV空 C=低IV多 D=低IV空")
        c2.metric("IV分位", f"P{ip:.0f}")
        c3.metric("方向评分", f"{int(latest['score']):+d}")
        vrp_v = latest.get("vrp")
        vrp_s = f"{float(vrp_v)*100:.1f}%" if vrp_v and abs(float(vrp_v)) < 1 else "N/A"
        c4.metric("VRP", vrp_s)

        # IV仪表盘
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ip,
            title={"text": "IV 分位"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 30], "color": "#E3F2FD"},
                    {"range": [30, 70], "color": "#FFF9C4"},
                    {"range": [70, 100], "color": "#FFCDD2"},
                ],
                "threshold": {"line": {"color": "red", "width": 2}, "value": ip},
            },
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        st.info("暂无数据，请先运行 morning_briefing.py")

    # 象限时间线
    if briefing is not None and not briefing.empty:
        st.subheader("象限时间线")
        briefing["quadrant"] = briefing.apply(_infer_quadrant, axis=1)

        fig_tl = go.Figure()
        for q in briefing["quadrant"].unique():
            mask = briefing["quadrant"] == q
            sub = briefing[mask]
            fig_tl.add_trace(go.Bar(
                x=sub["trade_date"], y=[1] * len(sub),
                name=q, marker_color=QUAD_COLORS.get(q, "#999"),
                hovertemplate="%{x}: " + q,
            ))
        fig_tl.update_layout(
            barmode="stack", height=200,
            yaxis=dict(visible=False),
            margin=dict(t=30, b=30),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_tl, use_container_width=True)

        # 象限分布
        st.subheader("象限分布统计")
        quad_counts = briefing["quadrant"].value_counts()
        st.bar_chart(quad_counts)

    # 账户权益走势
    if account is not None and not account.empty:
        st.subheader("账户权益走势")
        account["balance"] = account["balance"].astype(float)
        fig_eq = go.Figure(go.Scatter(
            x=account["trade_date"], y=account["balance"],
            mode="lines+markers", name="权益",
            line=dict(color="blue", width=2),
        ))
        fig_eq.update_layout(height=300, margin=dict(t=30, b=30))
        fig_eq.update_yaxes(title_text="权益(元)")
        st.plotly_chart(fig_eq, use_container_width=True)
