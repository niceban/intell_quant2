#!/bin/bash

# Configuration
REMOTE_HOST="192.168.31.158"
REMOTE_USER="lch_1028"
REMOTE_DIR="~/intell_quant2"
CONDA_ACTIVATE="/home/lch_1028/anaconda3/bin/activate"

echo "========================================================"
echo "🔄 [1/2] Syncing Code to ${REMOTE_HOST}..."
echo "========================================================"

rsync -avz \
    --exclude '.git' \
    --exclude '.idea' \
    --exclude '__pycache__' \
    --exclude '.venv' \
    --exclude 'data' \
    --exclude 'experiments' \
    --exclude 'outputs' \
    ./ ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/

echo ""
echo "========================================================"
echo "🚀 [2/2] Starting Background Random Search..."
echo "========================================================"

ssh ${REMOTE_USER}@${REMOTE_HOST} "bash -c '
    cd ${REMOTE_DIR}
    source ${CONDA_ACTIVATE}
    conda activate intell_quant2
    export PYTHONPATH=\$PYTHONPATH:.
    
    # Start in background
    nohup python random_search/run_search.py > dispatcher.log 2>&1 &
    
    echo "✅ Task started in background!"
    echo "🔍 PID: \$!"
'"

echo ""
echo "Monitor command: ./monitor_search.sh"