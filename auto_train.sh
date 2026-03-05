#!/bin/bash
# Auto-restart training loop. Checks every 60s, restarts if dead.
LOGFILE="/home/user/MATERIALIZE/gnn_train.log"
MONITOR_LOG="/home/user/MATERIALIZE/monitor.log"
WORKDIR="/home/user/MATERIALIZE/Phase 4"

echo "=== Monitor started at $(date) ===" >> "$MONITOR_LOG"

while true; do
    # Find the training python process
    PID=$(pgrep -f "python.*gnn.train" | head -1)

    if [ -z "$PID" ]; then
        echo "[$(date)] Training process dead. Restarting..." >> "$MONITOR_LOG"

        # Get checkpoint state before restart
        python3 -c "
import torch
ckpt = torch.load('$WORKDIR/checkpoints/latest_checkpoint.pt', weights_only=False)
print(f'  Checkpoint: epoch={ckpt[\"epoch\"]}, chunk_idx={ckpt.get(\"chunk_idx\")}, best_val_mse={ckpt[\"best_val_mse\"]:.6f}')
" >> "$MONITOR_LOG" 2>&1

        # Restart training
        cd "$WORKDIR"
        nohup python -u -m gnn.train \
            --data_dir ./processed \
            --output_dir ./checkpoints \
            --epochs 300 \
            --batch_size 32 \
            --patience 30 \
            --resume \
            --save_every 1 \
            > "$LOGFILE" 2>&1 &

        NEW_PID=$!
        echo "[$(date)] Restarted with PID $NEW_PID" >> "$MONITOR_LOG"
    else
        # Log last line of training output
        LAST=$(tail -1 "$LOGFILE" 2>/dev/null)
        echo "[$(date)] PID=$PID alive. Last: $LAST" >> "$MONITOR_LOG"
    fi

    sleep 60
done
