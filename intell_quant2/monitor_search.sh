#!/bin/bash

# Configuration
REMOTE_HOST="192.168.31.158"
REMOTE_USER="lch_1028"
REMOTE_DIR="~/intell_quant2"

echo "========================================================"
echo "📊 MONITORING RANDOM SEARCH STATUS (Press Ctrl+C to exit)"
echo "========================================================"

# Using single quotes for the SSH command to prevent local expansion of $ variables
ssh -t ${REMOTE_USER}@${REMOTE_HOST} '
    REMOTE_DIR="~/intell_quant2"
    # Find the latest random_search directory
    latest_dir=$(ls -td ~/intell_quant2/experiments/random_search_* 2>/dev/null | head -n 1)
    
    if [ -z "$latest_dir" ]; then
        echo "❌ No search results found yet."
        exit 1
    fi
    
    echo "Watching: $latest_dir"
    echo "---"
    tail -f "$latest_dir/search_status.log"
'