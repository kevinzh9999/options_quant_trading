PYTHON := /opt/homebrew/Caskroom/miniforge/base/envs/quant/bin/python
PYTEST := /opt/homebrew/Caskroom/miniforge/base/envs/quant/bin/pytest

.PHONY: test test-data test-all briefing monitor monitor-im monitor-ic vol quadrant executor eod backtest dashboard trading-day signal-quality sensitivity \
        dxgb-signal dxgb-executor dxgb-self-bt dxgb-tq-bt dxgb-status \
        dxgb-audit dxgb-repair-dirty dxgb-repair-all

test:
	$(PYTEST) tests/test_data/ -v

test-all:
	$(PYTEST) tests/ -v

test-data:
	$(PYTEST) tests/test_data/ -v

# 盘前
briefing:
	$(PYTHON) scripts/morning_briefing.py

# 盘中（单独启动各组件）
monitor-im:
	$(PYTHON) -m strategies.intraday.monitor --symbol IM

monitor-ic:
	$(PYTHON) -m strategies.intraday.monitor --symbol IC

monitor:
	@echo "Per-symbol monitor: make monitor-im / make monitor-ic"

vol:
	$(PYTHON) scripts/vol_monitor.py

quadrant:
	$(PYTHON) scripts/quadrant_monitor.py

executor:
	$(PYTHON) scripts/order_executor.py

# 收盘后
eod:
	$(PYTHON) scripts/daily_record.py eod

backtest:
	$(PYTHON) scripts/backtest_signals_day.py --symbol IM --date $$(date +%Y%m%d)

sensitivity:
	$(PYTHON) scripts/backtest_signals_day.py --symbol IM --date 20260220-20260327 --sensitivity

# 分析
dashboard:
	streamlit run dashboard/app.py

signal-quality:
	$(PYTHON) scripts/signal_quality_analysis.py --symbol IM

# === Daily XGB 跨日策略（独立 namespace） ===
dxgb-signal:
	$(PYTHON) -m strategies.daily_xgb.signal_generator

dxgb-executor:
	$(PYTHON) scripts/daily_xgb_executor.py

dxgb-self-bt:
	$(PYTHON) scripts/daily_xgb_self_backtest.py

dxgb-tq-bt:
	$(PYTHON) scripts/daily_xgb_tq_backtest.py --start 20250407 --end 20250425

dxgb-status:
	$(PYTHON) -c "from strategies.daily_xgb.risk_guard import status_report; from strategies.daily_xgb.config import DailyXGBConfig; print(status_report(DailyXGBConfig()))"

# === Daily XGB 数据完整性维护（建议每周或部署前跑一次） ===
dxgb-audit:
	$(PYTHON) scripts/audit_daily_model_output.py

dxgb-repair-dirty:
	$(PYTHON) scripts/repair_daily_model_output.py --dirty-only

dxgb-repair-all:
	$(PYTHON) scripts/repair_daily_model_output.py --rebuild-all

# 一键全天
trading-day:
	bash scripts/run_trading_day.sh
