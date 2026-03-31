#!/bin/bash
# 一键启动交易日全流程
# 用法: ./scripts/run_trading_day.sh 或 make trading-day
set -e
cd "$(dirname "$0")/.."
PROJ_DIR=$(pwd)
PYTHON="/opt/homebrew/Caskroom/miniforge/base/envs/quant/bin/python"
LOG_DIR="$PROJ_DIR/logs/runtime"
mkdir -p "$LOG_DIR"
TODAY=$(date +%Y%m%d)

echo "========================================"
echo " Trading Day Runner | $TODAY"
echo " Project: $PROJ_DIR"
echo "========================================"

# ========== 检查tmux ==========
if ! command -v tmux &> /dev/null; then
    echo "[$(date +%H:%M:%S)] tmux未安装，正在自动安装..."
    if command -v brew &> /dev/null; then
        brew install tmux
    else
        echo "  错误：需要Homebrew来安装tmux"
        echo "  临时方案：手动在4个终端窗口中分别运行："
        echo "    终端1: $PYTHON -m strategies.intraday.monitor"
        echo "    终端2: $PYTHON scripts/vol_monitor.py"
        echo "    终端3: $PYTHON scripts/quadrant_monitor.py"
        echo "    终端4: $PYTHON scripts/order_executor.py"
        exit 1
    fi
fi

# ========== 盘前 ==========
echo ""
echo "[$(date +%H:%M:%S)] === 盘前准备 ==="

echo "[$(date +%H:%M:%S)] Checking morning briefing..."
if [ -f /tmp/morning_briefing.json ] && grep -q "\"date\": \"$TODAY\"" /tmp/morning_briefing.json 2>/dev/null; then
    echo "  Briefing已由launchd自动完成，显示结果："
    cat "logs/briefing/${TODAY}.md" 2>/dev/null || $PYTHON scripts/morning_briefing.py 2>&1 | tee "$LOG_DIR/${TODAY}_briefing.log"
else
    echo "  运行morning briefing..."
    $PYTHON scripts/morning_briefing.py 2>&1 | tee "$LOG_DIR/${TODAY}_briefing.log"
fi

echo "[$(date +%H:%M:%S)] Quadrant check..."
$PYTHON scripts/quadrant_monitor.py --once 2>&1 | tee -a "$LOG_DIR/${TODAY}_quadrant.log"

echo ""
echo "[$(date +%H:%M:%S)] === 盘前准备完成 ==="
echo ""

# ========== 盘中 ==========
echo "[$(date +%H:%M:%S)] === 启动盘中监控 ==="

SESSION="trading_$TODAY"
tmux kill-session -t "$SESSION" 2>/dev/null || true

CONDA_INIT="source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate quant"
TMUX_PREFIX="cd '$PROJ_DIR' && $CONDA_INIT &&"

tmux new-session -d -s "$SESSION" -n "monitor"
tmux send-keys -t "$SESSION:monitor" "$TMUX_PREFIX python -m strategies.intraday.monitor 2>&1 | tee $LOG_DIR/${TODAY}_monitor.log" Enter

tmux new-window -t "$SESSION" -n "vol"
tmux send-keys -t "$SESSION:vol" "$TMUX_PREFIX python scripts/vol_monitor.py 2>&1 | tee $LOG_DIR/${TODAY}_vol.log" Enter

tmux new-window -t "$SESSION" -n "quadrant"
tmux send-keys -t "$SESSION:quadrant" "$TMUX_PREFIX python scripts/quadrant_monitor.py 2>&1 | tee $LOG_DIR/${TODAY}_quadrant_live.log" Enter

tmux new-window -t "$SESSION" -n "executor"
tmux send-keys -t "$SESSION:executor" "$TMUX_PREFIX python scripts/order_executor.py 2>&1 | tee $LOG_DIR/${TODAY}_executor.log" Enter

echo "[$(date +%H:%M:%S)] tmux session '$SESSION' created:"
echo "  monitor  - 日内信号 (Ctrl-b 0)"
echo "  vol      - 波动率   (Ctrl-b 1)"
echo "  quadrant - 象限     (Ctrl-b 2)"
echo "  executor - 下单     (Ctrl-b 3)"
echo ""
echo "  Attach: tmux attach -t $SESSION"
echo ""
echo "按回车键继续..."
read

# ========== 等待收盘 ==========
echo "[$(date +%H:%M:%S)] === 等待收盘 (15:15自动执行EOD) ==="
echo "  Ctrl-C 跳过等待"

TARGET_TIME="15:15:00"
NOW=$(date +%s)
TARGET=$(date -j -f "%Y%m%d %H:%M:%S" "$TODAY $TARGET_TIME" +%s 2>/dev/null || echo "")
if [ -n "$TARGET" ] && [ "$TARGET" -gt "$NOW" ]; then
    WAIT=$((TARGET - NOW))
    echo "  距离15:15还有 $((WAIT/60)) 分钟..."
    sleep "$WAIT" || true
fi

# ========== 盘后 ==========
echo ""
echo "[$(date +%H:%M:%S)] === 盘后处理 ==="

echo "[$(date +%H:%M:%S)] Checking EOD..."
if [ -f "$LOG_DIR/${TODAY}_eod.log" ] && grep -q "模型输出已写入" "$LOG_DIR/${TODAY}_eod.log" 2>/dev/null; then
    echo "  EOD已执行（launchd或手动），跳过"
else
    echo "  运行EOD..."
    $PYTHON scripts/daily_record.py eod 2>&1 | tee "$LOG_DIR/${TODAY}_eod.log"
fi

echo "[$(date +%H:%M:%S)] Quadrant snapshot..."
$PYTHON scripts/quadrant_monitor.py --once 2>&1 | tee -a "$LOG_DIR/${TODAY}_quadrant.log"

echo "[$(date +%H:%M:%S)] Backtesting today..."
$PYTHON scripts/backtest_signals_day.py --symbol IM --date "$TODAY" 2>&1 | tee "$LOG_DIR/${TODAY}_backtest.log"

echo ""
echo "[$(date +%H:%M:%S)] === 盘后处理完成 ==="

echo ""
echo "清理tmux session? [y/n]"
read -r CLEANUP
if [ "$CLEANUP" = "y" ]; then
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    echo "  tmux session已关闭"
fi

echo ""
echo "========================================"
echo " Trading Day Complete | $TODAY"
echo " Logs: $LOG_DIR/${TODAY}_*.log"
echo "========================================"
