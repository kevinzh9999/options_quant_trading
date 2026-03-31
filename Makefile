PYTHON := /opt/homebrew/Caskroom/miniforge/base/envs/quant/bin/python
PYTEST := /opt/homebrew/Caskroom/miniforge/base/envs/quant/bin/pytest

.PHONY: test test-data test-all briefing monitor vol quadrant executor eod backtest dashboard trading-day signal-quality sensitivity

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
monitor:
	$(PYTHON) -m strategies.intraday.monitor

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

# 一键全天
trading-day:
	bash scripts/run_trading_day.sh
