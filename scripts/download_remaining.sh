#!/bin/bash
# 串联执行: 期货K线 → 期权5m → IM/IC tick → 校验
set -e
PYTHON=/opt/homebrew/Caskroom/miniforge/base/envs/quant/bin/python
cd "$(dirname "$0")/.."

LOG=logs/download_remaining_$(date +%Y%m%d_%H%M%S).log
mkdir -p logs

echo "=== [1/4] 期货主连 K线 (1m+5m) ===" | tee -a "$LOG"
$PYTHON scripts/download_kline_pro.py --category futures --end 20260403 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== [2/4] 期权 5分钟K线 (MO/IO/HO) ===" | tee -a "$LOG"
$PYTHON scripts/download_kline_pro.py --category options --end 20260403 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== [3/4] 期货 Tick (IM/IC) ===" | tee -a "$LOG"
$PYTHON scripts/download_tick_pro.py --symbols IM,IC --end 20260403 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== [4/4] 校验 ===" | tee -a "$LOG"
$PYTHON scripts/download_verify.py 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "全部完成: $(date)" | tee -a "$LOG"
echo "日志: $LOG"
