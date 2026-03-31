"""
app.py
------
A股期权量化交易系统 - Streamlit Dashboard 主入口

启动方式:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import importlib
import streamlit as st

st.set_page_config(
    page_title="A股期权量化交易系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "📊 组合总览": "dashboard.pages.portfolio",
    "📈 波动率监控": "dashboard.pages.vol_monitor",
    "🔢 Greeks 分析": "dashboard.pages.greeks_dashboard",
    "📉 期货监控": "dashboard.pages.futures_monitor",
    "💰 贴水监控": "dashboard.pages.discount_monitor",
    "💰 策略绩效": "dashboard.pages.performance_dashboard",
    "🔬 模型诊断": "dashboard.pages.model_diagnostics",
    "🔴 盘中监控": "dashboard.pages.live_monitor",
    "📋 每日报告": "dashboard.pages.daily_reports",
    "🔍 复盘分析": "dashboard.pages.review_analysis",
    "🧪 波动率研究": "dashboard.pages.vol_lab",
    "🌅 Briefing历史": "dashboard.pages.briefing_history",
    "🎯 象限监控": "dashboard.pages.quadrant_history",
    "📡 信号分析": "dashboard.pages.signal_analysis",
}


def main() -> None:
    st.sidebar.title("导航菜单")
    st.sidebar.markdown("---")
    selection = st.sidebar.radio("选择页面", list(PAGES.keys()), label_visibility="collapsed")

    module_name = PAGES[selection]
    try:
        page_module = importlib.import_module(module_name)
        page_module.render()
    except NotImplementedError:
        st.warning(f"页面 [{selection}] 尚未实现，敬请期待。")
    except Exception as e:
        st.error(f"页面加载失败: {e}")
        import traceback
        st.code(traceback.format_exc())

    st.sidebar.markdown("---")
    st.sidebar.caption("A股期权量化交易系统 v0.1.0")
    st.sidebar.caption("A股惯例: 涨红跌绿")


if __name__ == "__main__":
    main()
