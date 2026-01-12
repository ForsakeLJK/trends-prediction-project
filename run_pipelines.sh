#!/bin/zsh

# ==============================
# CONFIG
# ==============================
PROJECT_DIR="$HOME/python_workspace/trends-prediction-project"
LOG_DIR="$PROJECT_DIR/logs"
LOCK_FILE="$HOME/tmp/trends_pipeline.lock"
# INTERVAL=15

mkdir -p "$LOG_DIR"

# ==============================
# Prevent overlapping runs
# ==============================
if [ -f "$LOCK_FILE" ]; then
    echo "$(date)  Previous run still active. Exiting." >> "$LOG_DIR/scheduler.log"
    exit 0
fi

touch "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

# ==============================
# Main loop (no sleep!)
# ==============================
while true; do
    RUN_TS=$(date +"%Y-%m-%d_%H-%M-%S")
    echo "==========================" >> "$LOG_DIR/scheduler.log"
    echo "Run started: $RUN_TS" >> "$LOG_DIR/scheduler.log"

    cd "$PROJECT_DIR" || exit 1

    # 1. Feature pipeline (includes 5-min scrape)
    echo "$(date) Running feature_pipeline..." >> "$LOG_DIR/scheduler.log"
    uv run feature_pipeline.py >> "$LOG_DIR/feature_pipeline_$RUN_TS.log" 2>&1
    FP_STATUS=$?

    if [ $FP_STATUS -ne 0 ]; then
        echo "$(date) feature_pipeline FAILED — skipping downstream steps" >> "$LOG_DIR/scheduler.log"
        continue
    else
        echo "$(date) feature_pipeline finished OK" >> "$LOG_DIR/scheduler.log"
    fi

    # 2. Batch inference
    echo "$(date) Running batch_inference..." >> "$LOG_DIR/scheduler.log"
    uv run batch_inference.py >> "$LOG_DIR/batch_inference_$RUN_TS.log" 2>&1
    BI_STATUS=$?

    if [ $BI_STATUS -ne 0 ]; then
        echo "$(date) batch_inference FAILED — skipping dashboard" >> "$LOG_DIR/scheduler.log"
        continue
    else
        echo "$(date) batch_inference finished OK" >> "$LOG_DIR/scheduler.log"
    fi

    # 3. Generate dashboard
    echo "$(date) Generating dashboard..." >> "$LOG_DIR/scheduler.log"
    uv run generate_dashboard.py >> "$LOG_DIR/dashboard_$RUN_TS.log" 2>&1

    if [ $? -ne 0 ]; then
        echo "$(date) dashboard generation FAILED" >> "$LOG_DIR/scheduler.log"
    else
        echo "$(date) dashboard generated OK" >> "$LOG_DIR/scheduler.log"
    fi
    
    # echo "$(date) Sleeping 15 seconds..." >> "$LOG_DIR/scheduler.log"
    # sleep $INTERVAL
done