#!/bin/bash
# Real-time training monitoring dashboard

LOG_DIR=${1:-"logs"}

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║           GHOP-HOLD Training Monitor Dashboard                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"

while true; do
    clear
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # GPU Status
    echo ""
    echo "GPU STATUS:"
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu \
               --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  Utilization: %d%% GPU | %d%% Memory | %dMB / %dMB | %d°C\n", $1, $2, $3, $4, $5}'
    
    # Latest Training Metrics
    echo ""
    echo "LATEST TRAINING METRICS:"
    if [ -d "$LOG_DIR" ]; then
        LATEST_LOG=$(find $LOG_DIR -name "train.log" -type f | xargs ls -t | head -1)
        if [ -n "$LATEST_LOG" ]; then
            echo "  Log: $LATEST_LOG"
            tail -5 "$LATEST_LOG" | grep -E "(Epoch|Loss|SDS)" || echo "  Waiting for metrics..."
        else
            echo "  No training logs found"
        fi
    else
        echo "  Log directory not found: $LOG_DIR"
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Press Ctrl+C to exit | Refreshing every 2 seconds..."
    
    sleep 2
done
