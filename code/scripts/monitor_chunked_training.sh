# scripts/monitor_chunked_training.sh
# Monitor memory during chunked training

LOGFILE="../ghop_production_chunked_results/memory_monitor_$(date +%Y%m%d_%H%M%S).log"

{
    echo "Memory monitoring started: $(date)"
    echo ""

    while true; do
        timestamp=$(date +%H:%M:%S)
        mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        echo "[$timestamp] GPU Memory: $mem MiB"
        sleep 30
    done
} > "$LOGFILE" 2>&1 &

MONITOR_PID=$!
echo "Memory monitor running (PID: $MONITOR_PID)"
echo "Log: $LOGFILE"
echo ""
echo "To stop: kill $MONITOR_PID"