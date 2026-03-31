"""
discount_monitor.py
-------------------
Dashboard 页面：IM 期货贴水监控

功能:
1. KPI 卡片：IML1/IML2/IML3 当前年化贴水率 + 信号强度
2. 历史贴水率走势图：IML1/IML2/IML3 时间序列 + 百分位线
3. 情景 P&L 图表：有/无 Put 保护 vs 到期时现货价格变化
4. Put 候选比较表：不同行权价对比
5. 全合约贴水明细表
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
from plotly.subplots import make_subplots


# ──────────────────────────────────────────────────────────────────────────────
# 数据库连接（缓存）
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _get_db():
    try:
        from data.storage.db_manager import DBManager
        from config.config_loader import ConfigLoader
        cfg = ConfigLoader()
        db = DBManager(cfg.get_db_path())
        db.initialize_tables()
        return db
    except Exception as e:
        st.error(f"数据库连接失败: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 数据加载函数
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _load_current_discount(_db_key: str, trade_date: str) -> pd.DataFrame:
    """加载当前日的贴水数据。"""
    try:
        db = _get_db()
        if db is None:
            return pd.DataFrame()
        from strategies.discount_capture.signal import DiscountSignal
        sig = DiscountSignal(db)
        return sig.calculate_discount(trade_date)
    except Exception as e:
        st.warning(f"贴水计算失败: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600)
def _load_discount_history(_db_key: str, contract_type: str, start_date: str) -> pd.DataFrame:
    """加载历史贴水时间序列。"""
    try:
        db = _get_db()
        if db is None:
            return pd.DataFrame()
        from strategies.discount_capture.signal import DiscountSignal
        sig = DiscountSignal(db)
        return sig.get_discount_history(contract_type=contract_type, start_date=start_date)
    except Exception as e:
        st.warning(f"历史贴水加载失败: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def _load_options_chain(_db_key: str, trade_date: str, expire_month: str) -> pd.DataFrame:
    """加载指定到期月的 MO 期权链。"""
    try:
        db = _get_db()
        if db is None:
            return pd.DataFrame()
        import re

        df = db.query_df(
            f"SELECT ts_code, close, volume, oi FROM options_daily "
            f"WHERE ts_code LIKE 'MO{expire_month}%' "
            f"AND trade_date='{trade_date}' AND close > 0"
        )
        if df is None or df.empty:
            return pd.DataFrame()

        _mo_re = re.compile(r'^MO(\d{4})-([CP])-(\d+)\.CFX$')
        records = []
        for _, row in df.iterrows():
            m = _mo_re.match(str(row["ts_code"]))
            if m:
                records.append({
                    "ts_code":       row["ts_code"],
                    "expire_month":  m.group(1),
                    "call_put":      m.group(2),
                    "exercise_price": float(m.group(3)),
                    "close":         float(row["close"]),
                    "volume":        float(row.get("volume") or 0),
                    "oi":            float(row.get("oi") or 0),
                })
        return pd.DataFrame(records) if records else pd.DataFrame()
    except Exception as e:
        st.warning(f"期权链加载失败: {e}")
        return pd.DataFrame()


def _get_latest_trade_date(db) -> str:
    """获取最新交易日（IM.CFX）。"""
    try:
        row = db.query_df(
            "SELECT MAX(trade_date) as dt FROM futures_daily WHERE ts_code='IM.CFX'"
        )
        if row is not None and not row.empty and row["dt"].iloc[0]:
            return str(row["dt"].iloc[0]).replace("-", "")
    except Exception:
        pass
    from datetime import date as _date
    return _date.today().strftime("%Y%m%d")


def _signal_color(signal: str) -> str:
    return {
        "STRONG": "#e74c3c",
        "MEDIUM": "#e67e22",
        "WEAK":   "#f1c40f",
        "NONE":   "#95a5a6",
    }.get(signal, "#95a5a6")


def _signal_label_cn(signal: str) -> str:
    return {
        "STRONG": "强烈",
        "MEDIUM": "中等",
        "WEAK":   "偏弱",
        "NONE":   "无",
    }.get(signal, "无")


# ──────────────────────────────────────────────────────────────────────────────
# 主渲染函数
# ──────────────────────────────────────────────────────────────────────────────

def render() -> None:
    st.title("💰 贴水监控 — IM 期货折价捕获")

    db = _get_db()
    if db is None:
        st.error("无法连接数据库，请检查配置。")
        return

    trade_date = _get_latest_trade_date(db)
    db_key = id(db)

    st.caption(f"数据日期: {trade_date}")

    # ── 控制栏 ────────────────────────────────────────────────────────────────
    col_ctrl1, col_ctrl2 = st.columns([2, 2])
    with col_ctrl1:
        time_range = st.selectbox(
            "历史回溯范围",
            options=["近3月", "近6月", "近1年", "全部"],
            index=2,
        )
    with col_ctrl2:
        account_equity = st.number_input(
            "账户权益（元，用于仓位计算）",
            min_value=100_000,
            max_value=100_000_000,
            value=1_000_000,
            step=100_000,
        )

    range_map = {"近3月": "20251017", "近6月": "20250717", "近1年": "20250317", "全部": "20220722"}
    hist_start = range_map.get(time_range, "20250317")

    # ── 加载当日贴水数据 ──────────────────────────────────────────────────────
    discount_df = _load_current_discount(str(db_key), trade_date)

    # ── Section 1: KPI 卡片 ──────────────────────────────────────────────────
    st.subheader("当前贴水概览")

    if discount_df.empty:
        st.info("暂无贴水数据（可能当日行情未更新）")
    else:
        # 生成信号
        try:
            from strategies.discount_capture.signal import DiscountSignal
            sig_gen = DiscountSignal(db)
            sig_result = sig_gen.generate_signal(trade_date)
        except Exception as e:
            sig_result = {"signal": "NONE", "annualized_discount": 0}

        # KPI 卡片：每个活跃合约一个
        cols = st.columns(min(len(discount_df), 4))
        for idx, (_, row) in enumerate(discount_df.iterrows()):
            if idx >= len(cols):
                break
            with cols[idx]:
                month = str(row["contract_month"])
                iml_code = str(row["iml_code"])
                ann_rate = float(row["annualized_discount_rate"])
                abs_disc = float(row["absolute_discount"])
                dte = int(row["days_to_expiry"])

                # 信号强度
                if ann_rate > 0.15:
                    sig_str, sig_color = "强烈", "#e74c3c"
                elif ann_rate >= 0.10:
                    sig_str, sig_color = "中等", "#e67e22"
                elif ann_rate >= 0.05:
                    sig_str, sig_color = "偏弱", "#f1c40f"
                else:
                    sig_str, sig_color = "无信号", "#95a5a6"

                disc_sign = "贴水" if abs_disc < 0 else "升水"

                st.markdown(
                    f"""
                    <div style="border:1px solid {sig_color}; border-radius:8px;
                                padding:12px; text-align:center; margin:4px 0;">
                        <div style="font-size:0.85em; color:#aaa;">{iml_code}</div>
                        <div style="font-size:1.4em; font-weight:bold; color:{sig_color};">
                            {ann_rate*100:.1f}%
                        </div>
                        <div style="font-size:0.8em; color:#ccc;">年化{disc_sign}率</div>
                        <div style="font-size:0.9em; margin-top:4px;">
                            {abs_disc:+.1f}点 | {dte}天
                        </div>
                        <div style="color:{sig_color}; font-size:0.85em;">▶ {sig_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # 推荐合约提示
        if sig_result.get("signal", "NONE") != "NONE":
            st.success(
                f"推荐合约: {sig_result.get('recommended_contract', 'N/A')}  |  "
                f"年化贴水率: {sig_result.get('annualized_discount', 0)*100:.1f}%  |  "
                f"剩余天数: {sig_result.get('days_to_expiry', 0)} 天  |  "
                f"历史百分位: {sig_result.get('discount_percentile', 0):.0f}%"
            )
        else:
            st.info("当前无满足条件的贴水信号（年化贴水率 < 5% 或无贴水）")

    st.markdown("---")

    # ── Section 2: 历史贴水走势 ──────────────────────────────────────────────
    st.subheader("历史贴水率走势")

    contract_types = ["IML1", "IML2", "IML3"]
    colors_hist = {"IML1": "#3498db", "IML2": "#e74c3c", "IML3": "#2ecc71"}

    fig_hist = go.Figure()
    hist_data = {}

    for ct in contract_types:
        df_h = _load_discount_history(str(db_key), ct, hist_start)
        if df_h is None or df_h.empty:
            continue
        hist_data[ct] = df_h
        # 显示负的 raw_discount_rate * 100（贴水为正值更直观）
        disc_pct = (-df_h["raw_discount_rate"] * 100).values  # 正值 = 贴水
        fig_hist.add_trace(go.Scatter(
            x=df_h["trade_date"].tolist(),
            y=disc_pct.tolist(),
            mode="lines",
            name=ct,
            line=dict(color=colors_hist[ct], width=1.5),
        ))

    if hist_data:
        # 百分位参考线（用 IML1 数据）
        if "IML1" in hist_data:
            rates_all = (-hist_data["IML1"]["raw_discount_rate"] * 100).values
            for p, color, dash in [
                (75, "rgba(231,76,60,0.4)", "dash"),
                (50, "rgba(255,255,255,0.3)", "dot"),
                (25, "rgba(52,152,219,0.4)", "dash"),
            ]:
                pval = float(np.percentile(rates_all, p))
                fig_hist.add_hline(
                    y=pval,
                    line_dash=dash,
                    line_color=color,
                    annotation_text=f"IML1 P{p}={pval:.1f}%",
                    annotation_font_size=10,
                )

        fig_hist.add_hline(y=8.0, line_dash="longdash", line_color="rgba(231,76,60,0.6)",
                           annotation_text="入场阈值 8%", annotation_font_size=10)

        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(
                showgrid=True, gridcolor="rgba(255,255,255,0.1)",
                title="贴水率 (%，正值=贴水)",
            ),
            legend=dict(orientation="h", y=1.02),
            margin=dict(l=10, r=10, t=40, b=10),
            height=350,
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("暂无历史贴水数据（等待数据积累）")

    st.markdown("---")

    # ── Section 3: 情景 P&L 分析 ─────────────────────────────────────────────
    st.subheader("到期情景 P&L 分析")

    if discount_df.empty:
        st.info("无贴水数据，跳过情景分析")
    else:
        col_s1, col_s2, col_s3 = st.columns(3)

        # 默认取贴水最大的合约
        valid_rows = discount_df[discount_df["absolute_discount"] < 0]
        if not valid_rows.empty:
            best_row = valid_rows.loc[valid_rows["annualized_discount_rate"].idxmax()]
            default_futures_price = float(best_row["futures_price"])
            default_spot = float(best_row["spot_price"])
            default_month = str(best_row["contract_month"])
        else:
            best_row = discount_df.iloc[0]
            default_futures_price = float(best_row["futures_price"])
            default_spot = float(best_row["spot_price"])
            default_month = str(best_row["contract_month"])

        with col_s1:
            futures_entry = st.number_input(
                "期货建仓价",
                value=float(round(default_futures_price, 0)),
                min_value=1000.0, max_value=20000.0, step=10.0,
            )
        with col_s2:
            put_strike = st.number_input(
                "保护 Put 行权价",
                value=float(round(default_futures_price * 0.92, -2)),
                min_value=1000.0, max_value=20000.0, step=50.0,
            )
        with col_s3:
            put_premium = st.number_input(
                "Put 权利金（元/张）",
                value=50.0,
                min_value=0.0, max_value=1000.0, step=5.0,
            )

        col_s4, col_s5 = st.columns(2)
        with col_s4:
            futures_lots = st.number_input("期货手数", value=1, min_value=1, max_value=20, step=1)
        with col_s5:
            put_lots = st.number_input("Put 手数", value=1, min_value=1, max_value=20, step=1)

        try:
            from strategies.discount_capture.position import DiscountPosition
            pos = DiscountPosition(account_equity=float(account_equity))
            pnl_df = pos.calculate_strategy_pnl_scenarios(
                futures_price=futures_entry,
                put_strike=put_strike,
                put_premium=put_premium,
                spot_price=default_spot,
                futures_lots=futures_lots,
                put_lots=put_lots,
            )

            fig_pnl = go.Figure()

            # 只有期货
            only_fut_pnl = (pnl_df["spot_at_expiry"] - futures_entry) * 200 * futures_lots
            fig_pnl.add_trace(go.Scatter(
                x=(pnl_df["spot_at_expiry"] / default_spot - 1) * 100,
                y=only_fut_pnl.values / 1000,
                mode="lines",
                name="纯多期货",
                line=dict(color="#3498db", width=2, dash="dot"),
            ))

            # 期货 + Put 保护
            fig_pnl.add_trace(go.Scatter(
                x=(pnl_df["spot_at_expiry"] / default_spot - 1) * 100,
                y=pnl_df["total_pnl"].values / 1000,
                mode="lines",
                name="期货 + 保护 Put",
                line=dict(color="#2ecc71", width=2),
                fill="tozeroy",
                fillcolor="rgba(46,204,113,0.1)",
            ))

            fig_pnl.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_dash="dot")
            fig_pnl.add_vline(x=0, line_color="rgba(255,255,255,0.3)", line_dash="dot",
                              annotation_text="当前现货", annotation_font_size=10)

            fig_pnl.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(
                    showgrid=True, gridcolor="rgba(255,255,255,0.1)",
                    title="到期时现货涨跌幅 (%)",
                ),
                yaxis=dict(
                    showgrid=True, gridcolor="rgba(255,255,255,0.1)",
                    title="策略 P&L（千元）",
                ),
                legend=dict(orientation="h", y=1.02),
                margin=dict(l=10, r=10, t=40, b=10),
                height=350,
            )
            st.plotly_chart(fig_pnl, use_container_width=True)

            # 关键指标
            col_k1, col_k2, col_k3 = st.columns(3)
            put_total_cost = put_premium * 200 * put_lots
            floor_loss = (futures_entry - put_strike) * 200 * futures_lots + put_total_cost
            # 最大收益：大幅上涨时期货无限盈利（理论上）
            max_gain_idx = pnl_df["total_pnl"].idxmax()
            max_gain = float(pnl_df["total_pnl"].iloc[max_gain_idx])

            with col_k1:
                st.metric("Put 总成本", f"{put_total_cost:,.0f} 元")
            with col_k2:
                st.metric("最大亏损（期货在K时）", f"-{floor_loss:,.0f} 元")
            with col_k3:
                st.metric("保盈平衡点（相对期货）",
                          f"{put_premium*200*put_lots / (futures_entry*200*futures_lots)*100:.2f}%")

        except Exception as e:
            st.warning(f"P&L 情景计算失败: {e}")

    st.markdown("---")

    # ── Section 4: Put 候选比较表 ────────────────────────────────────────────
    st.subheader("Put 保护方案对比")

    if not discount_df.empty and not valid_rows.empty if not discount_df.empty else True:
        chain_df = pd.DataFrame()
        if not discount_df.empty:
            valid_r = discount_df[discount_df["absolute_discount"] < 0]
            if not valid_r.empty:
                best_month = str(valid_r.loc[valid_r["annualized_discount_rate"].idxmax(), "contract_month"])
                chain_df = _load_options_chain(str(db_key), trade_date, best_month)

        if chain_df.empty:
            st.info("无对应期权链数据（MO 期权链未加载）")
        else:
            put_chain = chain_df[chain_df["call_put"] == "P"].copy()
            put_chain = put_chain[put_chain["volume"] >= 100].copy()

            if put_chain.empty:
                st.info("无足够流动性的 Put 期权（成交量 < 100）")
            else:
                fut_price = float(best_row["futures_price"]) if not valid_rows.empty else 8000.0

                put_chain["行权价"] = put_chain["exercise_price"]
                put_chain["权利金"] = put_chain["close"]
                put_chain["成交量"] = put_chain["volume"].astype(int)
                put_chain["保护成本(元/手)"] = (put_chain["close"] * 200).round(0)
                put_chain["行权价距离%"] = (
                    (put_chain["exercise_price"] - fut_price) / fut_price * 100
                ).round(2)
                put_chain["最大亏损(元/手)"] = (
                    (fut_price - put_chain["exercise_price"]) * 200
                    + put_chain["close"] * 200
                ).round(0)

                display_cols = ["行权价", "行权价距离%", "权利金", "成交量",
                                "保护成本(元/手)", "最大亏损(元/手)"]
                put_display = put_chain[
                    put_chain["exercise_price"] < fut_price
                ].sort_values("exercise_price", ascending=False)[display_cols]

                st.dataframe(put_display.reset_index(drop=True), use_container_width=True)

    st.markdown("---")

    # ── Section 5: 全合约贴水明细 ────────────────────────────────────────────
    st.subheader("各合约贴水明细")

    if discount_df.empty:
        st.info("暂无贴水明细数据")
    else:
        display_df = discount_df.copy()
        display_df["年化贴水率"] = (display_df["annualized_discount_rate"] * 100).round(2)
        display_df["绝对贴水(点)"] = display_df["absolute_discount"].round(2)
        display_df["期货价格"] = display_df["futures_price"].round(2)
        display_df["现货价格"] = display_df["spot_price"].round(2)
        display_df["理论基差"] = display_df["theoretical_basis"].round(2)
        display_df["剩余天数"] = display_df["days_to_expiry"].astype(int)

        show_cols = [
            "contract_month", "iml_code", "期货价格", "现货价格",
            "绝对贴水(点)", "年化贴水率", "剩余天数", "理论基差",
        ]
        st.dataframe(
            display_df[show_cols].rename(columns={
                "contract_month": "合约月份",
                "iml_code": "连续合约代码",
            }).reset_index(drop=True),
            use_container_width=True,
        )

    # ── 回测结果（可选展开）──────────────────────────────────────────────────
    with st.expander("历史回测结果（IML2，年化贴水率 > 8%）", expanded=False):
        try:
            from strategies.discount_capture.backtest import DiscountBacktest
            bt = DiscountBacktest(db)
            with st.spinner("回测计算中..."):
                results = bt.run(contract_type="IML2", min_discount_rate=0.08)

            if results["n_trades"] > 0:
                col_b1, col_b2, col_b3, col_b4 = st.columns(4)
                with col_b1:
                    st.metric("总收益率", f"{results['total_return']*100:.1f}%")
                with col_b2:
                    st.metric("年化收益率", f"{results['annualized_return']*100:.1f}%")
                with col_b3:
                    st.metric("最大回撤", f"{results['max_drawdown']*100:.1f}%")
                with col_b4:
                    st.metric("胜率", f"{results['win_rate']*100:.0f}%")

                if not results["trades"].empty:
                    trades_show = results["trades"].rename(columns={
                        "entry_date": "入场日期",
                        "exit_date": "出场日期",
                        "entry_price": "入场价",
                        "exit_price": "出场价",
                        "annualized_discount_rate": "年化贴水率",
                        "pnl": "P&L(元)",
                    })
                    trades_show["年化贴水率"] = (trades_show["年化贴水率"] * 100).round(1)
                    trades_show["P&L(元)"] = trades_show["P&L(元)"].round(0)
                    st.dataframe(
                        trades_show[["入场日期", "出场日期", "入场价", "出场价",
                                     "年化贴水率", "P&L(元)"]].reset_index(drop=True),
                        use_container_width=True,
                    )
            else:
                st.info("回测期间无满足条件的信号（贴水率持续低于阈值或数据不足）")

        except Exception as e:
            st.warning(f"回测执行失败: {e}")
