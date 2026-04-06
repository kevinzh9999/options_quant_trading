"""
futures_monitor.py
------------------
Dashboard 页面：期货行情监控

功能:
- 归一化价格走势（多品种对比）
- 主力/次主力合约价差（基差）
- 跨品种价差 Z-score
- 20日 RV 对比
- 近 5 日 OHLC 表格
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ──────────────────────────────────────────────
# Helpers
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


def _get_ts_code(product: str) -> str:
    mapping = {
        "IF": "IF.CFX",
        "IH": "IH.CFX",
        "IC": "IC.CFX",
        "IM": "IM.CFX",
    }
    return mapping.get(product, f"{product}.CFX")


def _get_continuous_codes(product: str):
    """Return (L1, L2) continuous contract codes."""
    mapping = {
        "IF": ("IFL1.CFX", "IFL2.CFX"),
        "IH": ("IHL1.CFX", "IHL2.CFX"),
        "IC": ("ICL1.CFX", "ICL2.CFX"),
        "IM": ("IML1.CFX", "IML2.CFX"),
    }
    return mapping.get(product, (f"{product}L1.CFX", f"{product}L2.CFX"))


def _load_futures_daily(db, ts_code: str, days: int) -> pd.DataFrame:
    try:
        df = db.query_df(
            f"""SELECT trade_date, open, high, low, close, volume, oi
                FROM futures_daily
                WHERE ts_code = '{ts_code}'
                ORDER BY trade_date DESC
                LIMIT {days}"""
        )
        if df is None or df.empty:
            return pd.DataFrame()
        return df.sort_values("trade_date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def _compute_rv20(_close_tuple):
    close = pd.Series(_close_tuple)
    try:
        from models.volatility.realized_vol import RealizedVolCalculator
        rv = RealizedVolCalculator.from_daily(close, window=20)
        return list(rv.values)
    except Exception:
        log_ret = np.log(close / close.shift(1))
        return list((log_ret.rolling(20).std() * np.sqrt(252)).values)


def _days_from_range(r: str) -> int:
    return {"近3月": 63, "近6月": 126, "近1年": 252, "全部": 2000}.get(r, 126)


# ──────────────────────────────────────────────
# Render
# ──────────────────────────────────────────────

def render() -> None:
    st.title("📉 期货行情监控")

    db = _get_db()
    if db is None:
        st.error("无法连接数据库，请检查配置。")
        return

    ctrl1, ctrl2 = st.columns(2)
    with ctrl1:
        selected = st.multiselect(
            "品种",
            options=["IF", "IH", "IC", "IM"],
            default=["IF", "IH", "IC", "IM"],
        )
    with ctrl2:
        time_range = st.selectbox("时间范围", ["近3月", "近6月", "近1年", "全部"], index=1)

    if not selected:
        st.warning("请至少选择一个品种")
        return

    days = _days_from_range(time_range)

    # Load data for all selected products
    data = {}
    for prod in selected:
        ts = _get_ts_code(prod)
        df = _load_futures_daily(db, ts, days)
        if not df.empty:
            data[prod] = df

    latest_dates = [df["trade_date"].iloc[-1] for df in data.values() if not df.empty]
    latest_date = max(latest_dates) if latest_dates else "N/A"
    st.caption(f"数据更新至: {latest_date}")

    colors = {"IF": "#3498db", "IH": "#e74c3c", "IC": "#2ecc71", "IM": "#e67e22"}
    line_colors = [colors.get(p, "#9b59b6") for p in selected]

    # ── Chart 1: Normalized Price ──
    st.subheader("归一化价格走势 (基=100)")

    if not data:
        st.info("暂无期货行情数据")
    else:
        fig1 = go.Figure()
        for i, (prod, df) in enumerate(data.items()):
            if df.empty or len(df) < 2:
                continue
            base = df["close"].iloc[0]
            if base == 0:
                continue
            normalized = (df["close"] / base * 100).tolist()
            fig1.add_trace(go.Scatter(
                x=df["trade_date"].tolist(),
                y=normalized,
                mode="lines",
                name=prod,
                line=dict(color=line_colors[i % len(line_colors)], width=2),
            ))
        fig1.add_hline(y=100, line_dash="dot", line_color="rgba(255,255,255,0.3)")
        fig1.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", title="指数化价格"),
            legend=dict(orientation="h", y=1.02),
            margin=dict(l=10, r=10, t=40, b=10),
            height=320,
        )
        st.plotly_chart(fig1, use_container_width=True)

    # ── Chart 2: Basis Spread (L1 - L2) ──
    st.subheader("主力/次主力价差（基差）")

    basis_data = {}
    for prod in selected:
        l1_code, l2_code = _get_continuous_codes(prod)
        l1_df = _load_futures_daily(db, l1_code, days)
        l2_df = _load_futures_daily(db, l2_code, days)
        if not l1_df.empty and not l2_df.empty:
            merged = l1_df[["trade_date", "close"]].merge(
                l2_df[["trade_date", "close"]], on="trade_date", suffixes=("_l1", "_l2")
            )
            if not merged.empty:
                merged["spread"] = merged["close_l1"] - merged["close_l2"]
                basis_data[prod] = merged

    if not basis_data:
        st.info("暂无连续合约数据（需要 IML1.CFX / IML2.CFX 等），尝试用主力合约价格代替")
        # Fallback: show raw close prices
        if data:
            fig2b = go.Figure()
            for i, (prod, df) in enumerate(data.items()):
                fig2b.add_trace(go.Scatter(
                    x=df["trade_date"].tolist(),
                    y=df["close"].tolist(),
                    mode="lines", name=prod,
                    line=dict(color=line_colors[i % len(line_colors)]),
                ))
            fig2b.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_title="价格",
                margin=dict(l=10, r=10, t=30, b=10),
                height=280,
            )
            st.plotly_chart(fig2b, use_container_width=True)
    else:
        fig2 = go.Figure()
        for i, (prod, merged) in enumerate(basis_data.items()):
            color = colors.get(prod, "#9b59b6")
            fig2.add_trace(go.Scatter(
                x=merged["trade_date"].tolist(),
                y=merged["spread"].tolist(),
                mode="lines", name=f"{prod} 基差",
                line=dict(color=color, width=1.5),
            ))
        fig2.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", title="价差 (点)"),
            legend=dict(orientation="h", y=1.02),
            margin=dict(l=10, r=10, t=40, b=10),
            height=280,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 3: Cross-Product Spread Z-Score ──
    st.subheader("跨品种价差 Z-score（20日）")

    spread_pairs = []
    if "IC" in data and "IH" in data:
        spread_pairs.append(("IC-IH", data["IC"], data["IH"]))
    if "IC" in data and "IF" in data:
        spread_pairs.append(("IC-IF", data["IC"], data["IF"]))
    if "IM" in data and "IF" in data:
        spread_pairs.append(("IM-IF", data["IM"], data["IF"]))

    if not spread_pairs:
        st.info("跨品种 Z-score 需要至少两个品种数据（如 IC 和 IH）")
    else:
        fig3 = go.Figure()
        z_colors = ["#3498db", "#e74c3c", "#9b59b6"]
        for i, (label, df_a, df_b) in enumerate(spread_pairs):
            merged = df_a[["trade_date", "close"]].merge(
                df_b[["trade_date", "close"]], on="trade_date", suffixes=("_a", "_b")
            )
            if merged.empty or len(merged) < 20:
                continue
            spread = merged["close_a"] - merged["close_b"]
            roll_mean = spread.rolling(20).mean()
            roll_std = spread.rolling(20).std()
            z = (spread - roll_mean) / roll_std.replace(0, np.nan)
            fig3.add_trace(go.Scatter(
                x=merged["trade_date"].tolist(),
                y=z.tolist(),
                mode="lines", name=label,
                line=dict(color=z_colors[i % len(z_colors)], width=1.5),
            ))
        fig3.add_hline(y=2, line_dash="dash", line_color="rgba(231,76,60,0.6)",
                       annotation_text="+2σ")
        fig3.add_hline(y=-2, line_dash="dash", line_color="rgba(46,204,113,0.6)",
                       annotation_text="-2σ")
        fig3.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)")
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", title="Z-score"),
            legend=dict(orientation="h", y=1.02),
            margin=dict(l=10, r=10, t=40, b=10),
            height=280,
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Chart 4: RV20 Comparison ──
    st.subheader("20日已实现波动率对比")

    if not data:
        st.info("暂无数据")
    else:
        fig4 = go.Figure()
        any_rv = False
        for i, (prod, df) in enumerate(data.items()):
            if len(df) < 22:
                continue
            rv_vals = _compute_rv20(tuple(df["close"].values.tolist()))
            if not rv_vals:
                continue
            any_rv = True
            fig4.add_trace(go.Scatter(
                x=df["trade_date"].tolist(),
                y=[v * 100 if v is not None and not np.isnan(v) else None for v in rv_vals],
                mode="lines", name=f"{prod} RV20",
                line=dict(color=line_colors[i % len(line_colors)], width=1.5),
            ))
        if any_rv:
            fig4.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", title="RV20 (%)"),
                legend=dict(orientation="h", y=1.02),
                margin=dict(l=10, r=10, t=40, b=10),
                height=280,
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("RV 计算需要至少 22 行数据")

    # ── Recent OHLC Table ──
    st.subheader("近 5 日行情一览")
    if not data:
        st.info("暂无数据")
    else:
        for prod, df in data.items():
            st.markdown(f"**{prod}**")
            last5 = df.tail(5)[["trade_date", "open", "high", "low", "close", "volume"]].copy()
            last5.columns = ["日期", "开盘", "最高", "最低", "收盘", "成交量"]
            st.dataframe(last5, use_container_width=True, hide_index=True)

    st.info("技术指标模块待实现（MA/MACD/ADX 等指标将在后续版本添加）")


if __name__ == "__main__":
    render()
