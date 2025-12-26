#!/bin/bash

# Default experiment name
EXP_NAME=${1:-"baseline"}

# Remove the first argument from the list so we can pass the rest to python
if [ $# -gt 0 ]; then
    shift
fi

# Project Root
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$ROOT"

# 1. Generate Timestamp and Directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_DIR="${ROOT}/experiments/${EXP_NAME}_${TIMESTAMP}"

# 2. Create the directory
mkdir -p "$EXP_DIR"

echo "=================================================================="
echo "🚀 Launching Experiment: $EXP_NAME"
echo "📂 Project Root: $ROOT"
echo "📁 Output Dir:   $EXP_DIR"
echo "📝 Extra Args:   $@"
echo "=================================================================="

# 3. Run with nohup, redirecting ALL output to the experiment folder
# "$@" passes all remaining arguments to the python script
nohup python -u exploring/single_process_test.py --exp_name "$EXP_NAME" --exp_dir "$EXP_DIR" "$@" > "${EXP_DIR}/experiment.log" 2>&1 &

PID=$!

# Brief sleep to allow log file creation
sleep 1

if ps -p $PID > /dev/null
then
    echo "✅ Training started successfully with PID: $PID"
    echo ""
    echo "📊 MONITORING COMMAND:"
    echo "tail -f ${EXP_DIR}/experiment.log"
    echo ""
    echo "📂 EXPERIMENT FOLDER:"
    echo "$EXP_DIR"
    echo "=================================================================="
else
    echo "❌ Error: Training failed to start. Check ${EXP_DIR}/experiment.log for details."
    echo "=================================================================="
    exit 1
fi