#!/bin/bash
# download_all_pro.sh — 串联执行全部历史数据下载（3步）+ 校验
# 用法: bash scripts/download_all_pro.sh

set -e
PYTHON=/opt/homebrew/Caskroom/miniforge/base/envs/quant/bin/python
cd "$(dirname "$0")/.."

LOG=logs/download_all_$(date +%Y%m%d_%H%M%S).log
mkdir -p logs

echo "========================================" | tee -a "$LOG"
echo "  全量历史数据下载 - $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

# Step 1: 现货 + 期货 K线 (1m + 5m)
echo "" | tee -a "$LOG"
echo "[Step 1/3] 现货指数 K线..." | tee -a "$LOG"
$PYTHON scripts/download_kline_pro.py --category spot --end 20260403 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[Step 1/3] 期货主连 K线..." | tee -a "$LOG"
$PYTHON scripts/download_kline_pro.py --category futures --end 20260403 2>&1 | tee -a "$LOG"

# Step 2: 期权 (MO/IO/HO) 5m K线
echo "" | tee -a "$LOG"
echo "[Step 2/3] 期权 5分钟K线 (MO/IO/HO)..." | tee -a "$LOG"
$PYTHON scripts/download_kline_pro.py --category options --end 20260403 2>&1 | tee -a "$LOG"

# Step 3: Tick (IM/IC only)
echo "" | tee -a "$LOG"
echo "[Step 3/3] 期货 Tick (IM/IC)..." | tee -a "$LOG"
$PYTHON scripts/download_tick_pro.py --symbols IM,IC --end 20260403 2>&1 | tee -a "$LOG"

# 校验
echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "  校验 + 汇总" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
$PYTHON scripts/download_verify.py 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "全部完成: $(date)" | tee -a "$LOG"
echo "日志: $LOG"
