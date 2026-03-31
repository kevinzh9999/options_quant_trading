"""
briefing_history.py
-------------------
Morning Briefing 历史分析：方向准确率、评分走势、维度贡献。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config.config_loader import ConfigLoader
from data.storage.db_manager import DBManager


def _load_data():
    db = DBManager(ConfigLoader().get_db_path())
    briefing = db.query_df(
        "SELECT * FROM morning_briefing ORDER BY trade_date"
    )
    spot = db.query_df(
        "SELECT trade_date, close FROM index_daily "
        "WHERE ts_code='000852.SH' ORDER BY trade_date"
    )
    return briefing, spot, db


def render():
    st.title("Morning Briefing 历史分析")

    try:
        briefing, spot, db = _load_data()
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return

    if briefing is None or briefing.empty:
        st.warning("暂无 Morning Briefing 数据。请先运行 python scripts/morning_briefing.py")
        return

    # 合并现货数据计算实际涨跌
    if spot is not None and not spot.empty:
        spot = spot.sort_values("trade_date").reset_index(drop=True)
        spot["prev_close"] = spot["close"].shift(1)
        spot["actual_pct"] = (spot["close"] - spot["prev_close"]) / spot["prev_close"] * 100
        spot_map = dict(zip(spot["trade_date"], spot["actual_pct"]))
    else:
        spot_map = {}

    briefing["actual_pct"] = briefing["trade_date"].map(spot_map)
    briefing["actual_dir"] = briefing["actual_pct"].apply(
        lambda x: "涨" if x and x > 0.1 else ("跌" if x and x < -0.1 else "平")
    )
    briefing["correct"] = briefing.apply(
        lambda r: (r["direction"] == "偏多" and r.get("actual_dir") == "涨") or
                  (r["direction"] == "偏空" and r.get("actual_dir") == "跌") or
                  r["direction"] == "中性",
        axis=1,
    )

    # KPI
    total = len(briefing[briefing["actual_pct"].notna()])
    correct = int(briefing["correct"].sum()) if total > 0 else 0
    accuracy = correct / total * 100 if total > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总天数", total)
    c2.metric("准确率", f"{accuracy:.0f}%")
    c3.metric("最新评分", f"{int(briefing.iloc[-1]['score']):+d}")
    c4.metric("最新方向", briefing.iloc[-1]["direction"])

    # 评分走势 + 现货走势
    st.subheader("评分 vs 现货走势")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = ["red" if s > 0 else "green" for s in briefing["score"]]
    fig.add_trace(
        go.Bar(x=briefing["trade_date"], y=briefing["score"],
               marker_color=colors, name="Briefing评分", opacity=0.7),
        secondary_y=False,
    )
    if spot is not None and not spot.empty:
        merged = briefing.merge(spot[["trade_date", "close"]], on="trade_date", how="left")
        fig.add_trace(
            go.Scatter(x=merged["trade_date"], y=merged["close"],
                       name="000852现货", line=dict(color="blue", width=1.5)),
            secondary_y=True,
        )
    fig.update_layout(height=400, margin=dict(t=30, b=30))
    fig.update_yaxes(title_text="评分", secondary_y=False)
    fig.update_yaxes(title_text="现货价格", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # 方向判断明细表
    st.subheader("每日方向判断明细")
    display_cols = ["trade_date", "direction", "confidence", "score",
                    "actual_pct", "actual_dir", "correct"]
    avail_cols = [c for c in display_cols if c in briefing.columns]
    display = briefing[avail_cols].copy()
    if "actual_pct" in display.columns:
        display["actual_pct"] = display["actual_pct"].apply(
            lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
    if "correct" in display.columns:
        display["correct"] = display["correct"].apply(lambda x: "✅" if x else "❌")
    st.dataframe(display.iloc[::-1], use_container_width=True, height=400)

    # 关键原因词频
    if "reasons" in briefing.columns:
        st.subheader("高频原因词")
        all_reasons = " | ".join(briefing["reasons"].dropna().tolist())
        words = [w.strip() for w in all_reasons.split("|") if w.strip()]
        if words:
            from collections import Counter
            freq = Counter(words).most_common(15)
            freq_df = pd.DataFrame(freq, columns=["原因", "出现次数"])
            st.bar_chart(freq_df.set_index("原因"))
