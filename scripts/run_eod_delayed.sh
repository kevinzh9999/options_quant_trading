#!/bin/bash
# 延迟EOD：18:30后运行，Tushare数据已更新
cd "$(dirname "$0")/.."
PYTHON="/opt/homebrew/Caskroom/miniforge/base/envs/quant/bin/python"
TODAY=$(date +%Y%m%d)
LOG_DIR="logs/runtime"
mkdir -p "$LOG_DIR"

echo "=== Delayed EOD | $TODAY | $(date +%H:%M:%S) ==="

# 1. 重跑EOD（补全Tushare数据）
echo "[$(date +%H:%M:%S)] Running delayed EOD..."
$PYTHON scripts/daily_record.py eod

# 2. 增量更新briefing历史数据（全部8个表）
echo "[$(date +%H:%M:%S)] Updating briefing history data..."
$PYTHON scripts/download_briefing_history.py --update

echo "[$(date +%H:%M:%S)] === Done ==="
