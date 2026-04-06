"""
greeks_dashboard.py
-------------------
Dashboard 页面：Greeks 分析

功能:
- 组合净 Greeks 卡片
- Delta / Vega / Theta 情景分析
- Greeks 历史走势
- 持仓明细表
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
from datetime import datetime


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


def _load_positions_for_greeks(db) -> list[dict]:
    """Load latest positions enriched with options_contracts data."""
    try:
        df = db.query_df(
            """SELECT ps.symbol, ps.direction, ps.volume, ps.volume_today,
                      ps.open_price_avg, ps.last_price, ps.float_profit, ps.margin,
                      oc.exercise_price as strike_price, oc.call_put, oc.delist_date as expire_date,
                      oc.exercise_price, oc.underlying_symbol
               FROM position_snapshots ps
               LEFT JOIN options_contracts oc
                 ON oc.ts_code = REPLACE(ps.symbol, 'CFFEX.', '') || '.CFX'
               WHERE ps.trade_date = (SELECT MAX(trade_date) FROM position_snapshots)"""
        )
        if df is None or df.empty:
            return []
        positions = []
        for _, row in df.iterrows():
            direction = str(row.get("direction", "LONG")).upper()
            volume = int(row.get("volume", 0) or 0)
            net_vol = volume if direction == "LONG" else -volume
            positions.append({
                "ts_code": str(row.get("symbol", "")),
                "symbol": str(row.get("symbol", "")),
                "direction": direction,
                "volume": net_vol,
                "strike_price": row.get("strike_price") or row.get("exercise_price"),
                "call_put": str(row.get("call_put", "")) if row.get("call_put") else None,
                "expire_date": str(row.get("expire_date", "")) if row.get("expire_date") else None,
                "last_price": float(row.get("last_price", 0) or 0),
                "float_profit": float(row.get("float_profit", 0) or 0),
                "margin": float(row.get("margin", 0) or 0),
                "underlying_symbol": str(row.get("underlying_symbol", "")) if row.get("underlying_symbol") else None,
                "iv": None,  # Will be filled if possible
            })
        return positions
    except Exception:
        return []


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


def _load_model_history(db) -> pd.DataFrame:
    try:
        df = db.query_df(
            """SELECT trade_date, net_delta, net_gamma, net_theta, net_vega
               FROM daily_model_output
               ORDER BY trade_date ASC"""
        )
        if df is None:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


def _load_spot(db) -> float:
    try:
        df = db.query_df(
            """SELECT close FROM futures_daily
               WHERE ts_code = 'IM.CFX'
               ORDER BY trade_date DESC LIMIT 1"""
        )
        if df is not None and not df.empty:
            return float(df.iloc[0]["close"])
        return 0.0
    except Exception:
        return 0.0


def _is_option(pos: dict) -> bool:
    cp = pos.get("call_put")
    return cp in ("C", "P", "认购", "认沽") and pos.get("expire_date") is not None


def _reprice_position(pos: dict, S: float, iv_override: float | None = None,
                      T_offset_days: float = 0, r: float = 0.02) -> float:
    """Reprice a single option position and return total P&L change."""
    try:
        from models.pricing.black_scholes import BlackScholes
        K = float(pos.get("strike_price") or pos.get("exercise_price") or S)
        cp_raw = str(pos.get("call_put", "C")).upper()
        cp = "C" if "C" in cp_raw or "购" in cp_raw else "P"
        expire_str = str(pos.get("expire_date", ""))
        if len(expire_str) >= 8:
            exp_date = datetime.strptime(expire_str[:8], "%Y%m%d")
            T_base = max((exp_date - datetime.now()).days / 365.0, 1 / 365)
        else:
            T_base = 30 / 365
        T = max(T_base - T_offset_days / 365.0, 1 / 365)
        iv = iv_override if iv_override is not None else (pos.get("iv") or 0.25)
        price_new = BlackScholes.price(S=S, K=K, T=T, r=r, sigma=iv, q=0, option_type=cp)
        price_old = pos.get("last_price", 0) or 0
        vol = pos.get("volume", 0) or 0
        multiplier = 100  # typical A-share option multiplier
        return (price_new - price_old) * vol * multiplier
    except Exception:
        return 0.0


# ──────────────────────────────────────────────
# Render
# ──────────────────────────────────────────────

def render() -> None:
    st.title("🔢 Greeks 分析")

    db = _get_db()
    if db is None:
        st.error("无法连接数据库，请检查配置。")
        return

    model_data = _load_latest_model(db)
    positions = _load_positions_for_greeks(db)
    model_history = _load_model_history(db)
    spot = _load_spot(db)

    latest_date = model_data.get("trade_date", "N/A") if model_data else "N/A"
    st.caption(f"数据更新至: {latest_date}")

    # ── Top KPI Row ──
    k1, k2, k3, k4 = st.columns(4)

    def _safe_val(d, key, fmt=".4f"):
        if d:
            v = d.get(key)
            if v is not None and not pd.isna(float(v)):
                return format(float(v), fmt)
        return "待计算"

    k1.metric("Net Delta", _safe_val(model_data, "net_delta"))
    k2.metric("Net Gamma", _safe_val(model_data, "net_gamma", ".6f"))
    k3.metric("Net Theta", _safe_val(model_data, "net_theta", ".2f"))
    k4.metric("Net Vega", _safe_val(model_data, "net_vega", ".4f"))

    st.divider()

    # Filter option positions for scenario analysis
    option_positions = [p for p in positions if _is_option(p)]

    if not option_positions:
        st.warning("暂无期权持仓数据，无法进行情景分析")
        # Still show Greeks history and position table below
    else:
        # Assign default IV if missing
        default_iv = 0.25
        if model_data and model_data.get("atm_iv"):
            default_iv = float(model_data["atm_iv"])
        for p in option_positions:
            if not p.get("iv"):
                p["iv"] = default_iv

        if spot <= 0:
            st.info("暂无标的价格数据，无法进行情景分析")
        else:
            sc1, sc2 = st.columns(2)

            # Chart 1: Delta scenario (spot ±5%)
            with sc1:
                st.subheader("Delta 情景 P&L (标的价格 ±5%)")
                spot_range = np.linspace(spot * 0.95, spot * 1.05, 50)
                pnl_delta = []
                for s in spot_range:
                    total = sum(_reprice_position(p, S=s) for p in option_positions)
                    pnl_delta.append(total)
                fig_d = go.Figure()
                fig_d.add_trace(go.Scatter(
                    x=spot_range.tolist(), y=pnl_delta,
                    mode="lines", name="情景 P&L",
                    line=dict(color="#3498db", width=2),
                ))
                fig_d.add_vline(x=spot, line_dash="dash",
                                line_color="rgba(255,200,0,0.8)",
                                annotation_text="当前价格")
                fig_d.add_hline(y=0, line_dash="dot",
                                line_color="rgba(255,255,255,0.3)")
                fig_d.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="标的价格",
                    yaxis_title="P&L (元)",
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=300,
                )
                st.plotly_chart(fig_d, use_container_width=True)

            # Chart 2: Vega scenario (IV ±10pp)
            with sc2:
                st.subheader("Vega 情景 P&L (IV ±10pp)")
                iv_range = np.linspace(
                    max(default_iv - 0.10, 0.01),
                    default_iv + 0.10, 50
                )
                pnl_vega = []
                for iv_shift in iv_range:
                    total = sum(_reprice_position(p, S=spot, iv_override=iv_shift) for p in option_positions)
                    pnl_vega.append(total)
                fig_v = go.Figure()
                fig_v.add_trace(go.Scatter(
                    x=(iv_range * 100).tolist(), y=pnl_vega,
                    mode="lines", name="情景 P&L",
                    line=dict(color="#e67e22", width=2),
                ))
                fig_v.add_vline(x=default_iv * 100, line_dash="dash",
                                line_color="rgba(255,200,0,0.8)",
                                annotation_text="当前 IV")
                fig_v.add_hline(y=0, line_dash="dot",
                                line_color="rgba(255,255,255,0.3)")
                fig_v.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="IV (%)",
                    yaxis_title="P&L (元)",
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=300,
                )
                st.plotly_chart(fig_v, use_container_width=True)

            # Chart 3: Theta decay
            st.subheader("Theta 时间价值衰减")
            max_dte = 0
            for p in option_positions:
                expire_str = str(p.get("expire_date", ""))
                if len(expire_str) >= 8:
                    try:
                        exp_date = datetime.strptime(expire_str[:8], "%Y%m%d")
                        dte = (exp_date - datetime.now()).days
                        max_dte = max(max_dte, dte)
                    except Exception:
                        pass
            max_dte = max(max_dte, 1)

            days_range = list(range(0, max_dte + 1))
            pnl_theta = []
            for d in days_range:
                total = sum(_reprice_position(p, S=spot, T_offset_days=float(d)) for p in option_positions)
                pnl_theta.append(total)

            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(
                x=days_range, y=pnl_theta,
                mode="lines", name="Theta 衰减",
                line=dict(color="#2ecc71", width=2),
            ))
            fig_t.add_vline(x=0, line_dash="dash",
                            line_color="rgba(255,200,0,0.8)",
                            annotation_text="今日")
            fig_t.add_hline(y=0, line_dash="dot",
                            line_color="rgba(255,255,255,0.3)")
            fig_t.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="经过天数",
                yaxis_title="P&L (元)",
                margin=dict(l=10, r=10, t=40, b=10),
                height=280,
            )
            st.plotly_chart(fig_t, use_container_width=True)

    # ── Chart 4: Greeks history ──
    st.subheader("Greeks 历史走势")
    greeks_cols = ["net_delta", "net_gamma", "net_theta", "net_vega"]
    greeks_labels = {"net_delta": "Net Delta", "net_gamma": "Net Gamma",
                     "net_theta": "Net Theta", "net_vega": "Net Vega"}
    colors_g = ["#3498db", "#e67e22", "#e74c3c", "#9b59b6"]

    if model_history.empty or len(model_history) < 5:
        n = len(model_history)
        st.info(f"数据积累中（当前 {n} 天，需要 ≥ 5 天）")
    else:
        valid_cols = [c for c in greeks_cols if c in model_history.columns]
        if valid_cols:
            fig_gh = go.Figure()
            for i, col in enumerate(valid_cols):
                sub = model_history.dropna(subset=[col])
                if sub.empty:
                    continue
                fig_gh.add_trace(go.Scatter(
                    x=sub["trade_date"].tolist(),
                    y=sub[col].tolist(),
                    mode="lines",
                    name=greeks_labels.get(col, col),
                    line=dict(color=colors_g[i % len(colors_g)]),
                ))
            fig_gh.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                legend=dict(orientation="h", y=1.02),
                margin=dict(l=10, r=10, t=40, b=10),
                height=300,
            )
            st.plotly_chart(fig_gh, use_container_width=True)
        else:
            st.info("模型输出中无 Greeks 历史数据")

    # ── Position Detail Table ──
    st.subheader("持仓明细")
    if not positions:
        st.info("暂无持仓数据")
    else:
        disp_rows = []
        for p in positions:
            disp_rows.append({
                "合约": p.get("symbol", ""),
                "方向": p.get("direction", ""),
                "手数": p.get("volume", 0),
                "最新价": p.get("last_price", 0),
                "浮动盈亏": p.get("float_profit", 0),
                "保证金": p.get("margin", 0),
                "类型": f"{p.get('call_put', '')} K={p.get('strike_price', '')}" if _is_option(p) else "期货",
                "到期日": p.get("expire_date", ""),
            })
        disp_df = pd.DataFrame(disp_rows)

        def _color_profit(val):
            try:
                v = float(val)
                if v > 0:
                    return "color: #e74c3c"
                elif v < 0:
                    return "color: #2ecc71"
                return ""
            except Exception:
                return ""

        if "浮动盈亏" in disp_df.columns:
            styled = disp_df.style.applymap(_color_profit, subset=["浮动盈亏"])
            st.dataframe(styled, use_container_width=True)
        else:
            st.dataframe(disp_df, use_container_width=True)


if __name__ == "__main__":
    render()
