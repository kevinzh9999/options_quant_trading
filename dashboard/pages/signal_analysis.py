"""
signal_analysis.py
------------------
日内信号质量分析：信号频率、质量维度、时段分布。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.config_loader import ConfigLoader
from data.storage.db_manager import DBManager, get_db


def _load_signal_log(db, days=30):
    df = db.query_df(
        f"SELECT * FROM signal_log "
        f"WHERE datetime >= date('now', '-{days} days') "
        f"ORDER BY datetime"
    )
    return df


def _load_order_log(db):
    df = db.query_df("SELECT * FROM order_log ORDER BY datetime")
    return df


def render():
    st.title("日内信号分析")

    db = get_db()

    # 侧边栏
    days = st.sidebar.slider("回看天数", 7, 90, 30)

    signal_log = _load_signal_log(db, days)

    if signal_log is None or signal_log.empty:
        st.warning("暂无 signal_log 数据。monitor 运行后会自动记录。")
        st.info("可以通过回测生成数据：python scripts/backtest_signals_day.py --symbol IM --date YYYYMMDD")
        return

    # 基础统计
    st.subheader("信号概览")
    total_signals = len(signal_log)

    # 尝试获取score列
    score_col = None
    for col in ["score", "v2_score", "v3_score"]:
        if col in signal_log.columns:
            signal_log[col] = pd.to_numeric(signal_log[col], errors="coerce").fillna(0)
            if score_col is None:
                score_col = col

    triggered = 0
    if score_col:
        triggered = int((signal_log[score_col] >= 60).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("总记录数", total_signals)
    c2.metric("达标信号(>=60)", triggered)
    c3.metric("触发率", f"{triggered/total_signals*100:.1f}%" if total_signals > 0 else "0%")

    # 信号频率：每日信号数
    if "datetime" in signal_log.columns:
        signal_log["date"] = signal_log["datetime"].str[:10]
        daily_counts = signal_log.groupby("date").size().reset_index(name="count")

        st.subheader("每日信号记录数")
        fig = go.Figure(go.Bar(
            x=daily_counts["date"], y=daily_counts["count"],
            marker_color="steelblue",
        ))
        fig.update_layout(height=250, margin=dict(t=20, b=30))
        st.plotly_chart(fig, use_container_width=True)

    # 品种分布
    if "symbol" in signal_log.columns:
        st.subheader("品种信号分布")
        sym_counts = signal_log["symbol"].value_counts()
        st.bar_chart(sym_counts)

    # Score分布
    if score_col:
        st.subheader("Score 分布直方图")
        scores = signal_log[signal_log[score_col] > 0][score_col]
        if not scores.empty:
            fig_hist = px.histogram(scores, nbins=20, title="Score分布")
            fig_hist.add_vline(x=60, line_dash="dash", line_color="red",
                               annotation_text="阈值=60")
            fig_hist.update_layout(height=300, margin=dict(t=40, b=30))
            st.plotly_chart(fig_hist, use_container_width=True)

    # 方向分布
    dir_col = None
    for col in ["direction", "v2_direction", "v3_direction"]:
        if col in signal_log.columns:
            dir_col = col
            break
    if dir_col:
        st.subheader("信号方向分布")
        dir_counts = signal_log[signal_log[dir_col] != ""][dir_col].value_counts()
        if not dir_counts.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.bar_chart(dir_counts)
            with c2:
                long_n = int(dir_counts.get("LONG", 0))
                short_n = int(dir_counts.get("SHORT", 0))
                total_dir = long_n + short_n
                if total_dir > 0:
                    st.metric("LONG占比", f"{long_n/total_dir*100:.0f}%")
                    st.metric("SHORT占比", f"{short_n/total_dir*100:.0f}%")

    # 时段分布热力图
    if "datetime" in signal_log.columns and score_col and "symbol" in signal_log.columns:
        st.subheader("时段 × 品种 信号强度热力图")
        sl = signal_log[signal_log[score_col] > 0].copy()
        if not sl.empty:
            sl["hour_min"] = sl["datetime"].str[11:16]
            # 按30分钟分组
            sl["time_slot"] = sl["hour_min"].apply(
                lambda x: x[:3] + ("00" if int(x[3:5]) < 30 else "30")
                if len(x) >= 5 else x
            )
            pivot = sl.groupby(["time_slot", "symbol"])[score_col].mean().reset_index()
            if not pivot.empty:
                pivot_wide = pivot.pivot(index="symbol", columns="time_slot", values=score_col)
                fig_hm = px.imshow(
                    pivot_wide.fillna(0),
                    labels=dict(x="时段", y="品种", color="平均Score"),
                    color_continuous_scale="YlOrRd",
                    aspect="auto",
                )
                fig_hm.update_layout(height=250, margin=dict(t=20, b=30))
                st.plotly_chart(fig_hm, use_container_width=True)

    # 下单记录
    order_log = _load_order_log(db)
    if order_log is not None and not order_log.empty:
        st.subheader("下单记录 (order_log)")
        st.dataframe(order_log.iloc[::-1], use_container_width=True, height=300)
