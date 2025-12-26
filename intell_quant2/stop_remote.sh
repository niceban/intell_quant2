#!/bin/bash

# Configuration
REMOTE_HOST="192.168.31.158"
REMOTE_USER="lch_1028"

echo "========================================================"
echo "🛑 STOPPING REMOTE TASKS & CLEARING GPU (STRICT)"
echo "========================================================"

ssh ${REMOTE_USER}@${REMOTE_HOST} "bash -s" << 'EOF'
    echo "1. Killing Dispatcher and Workers..."
    pkill -9 -f 'random_search/run_search.py' || true
    pkill -9 -f 'random_search/train_worker.py' || true
    
    echo "2. Killing any process tied to project path..."
    pids=$(pgrep -f 'intell_quant2')
    if [ -n "$pids" ]; then
        kill -9 $pids 2>/dev/null
    fi

    echo "3. Clearing GPU Memory..."
    if command -v nvidia-smi &> /dev/null; then
        gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
        if [ -n "$gpu_pids" ]; then
            echo "Killing GPU PIDs: $gpu_pids"
            echo "$gpu_pids" | xargs kill -9 2>/dev/null
        fi
    fi

    echo ""
    echo "📊 VERIFICATION:"
    nvidia-smi
    echo ""
    echo "🔍 REMAINING PYTHON:"
    ps -ef | grep python | grep -v grep || echo "Clean."
EOF

echo ""
echo "✅ Remote server is now reset."

