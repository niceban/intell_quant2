
import subprocess
import sys
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime

# CONFIG
NUM_GPUS = 2
MAX_CONCURRENT_PER_GPU = 1 
TOTAL_EXPERIMENTS = 20 
# Move output out of code module
PROJECT_ROOT = Path(__file__).resolve().parent.parent
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SEARCH_DIR = PROJECT_ROOT / "experiments" / f"random_search_{timestamp}"
MONITOR_FILE = SEARCH_DIR / "search_status.log"
SEARCH_DIR.mkdir(parents=True, exist_ok=True)

# Define Search Space
def sample_hyperparams(exp_id):
    # LogUniform sampling for LR
    lr = 10 ** np.random.uniform(-6, -4) 
    # NEW: 80% chance gamma=0, 20% chance normal range
    if random.random() < 0.8:
        gamma = 0.0
    else:
        # Prevent round(0.99x, 2) -> 1.0 case
        raw_gamma = np.random.uniform(0.5, 0.99)
        gamma = round(raw_gamma, 2)
        gamma = min(0.99, gamma) # Double safety
        gamma = 0
        
    batch_size = random.choice([4096])#2048, 3072, 
    buffer_size = random.choice([100000, 500000]) # Smaller buffer for faster turnover
    tau = random.choice([0.001, 0.005, 0.01])
    
    # Architecture Search: Head Complexity
    # 1% Light (0), 99% Heavy (128/256) - Focus on Heavy based on initial results
    if random.random() < 0.01:
        head_hidden_dim = 0
    else:
        head_hidden_dim = random.choice([128, 256, 512])
    
    # Name format: exp_01_lr_1e-05_gm_0.85_bs_1024_bf_100k_hd_128
    exp_name = f"exp_{exp_id:02d}_lr_{lr:.1e}_gm_{gamma}_bs_{batch_size}_bf_{buffer_size}_hd_{head_hidden_dim}"
    
    return {
        "exp_name": exp_name,
        "lr": lr,
        "gamma": gamma,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "head_hidden_dim": head_hidden_dim,
        "tau": tau
    }

def main():
    header = """
================================================================================
RANDOM SEARCH STATUS LOG
VAL METRICS: ATR=AvgTradeRet, MKR=MarketRet, Win=WinRate, PF=ProfitFactor, 
             Shp=Sharpe, AHD=AvgHoldDays, EFF=DailyEfficiency, Trd=Trades
================================================================================
"""
    # Clear previous logs
    if MONITOR_FILE.exists():
        with open(MONITOR_FILE, "a") as f:
            f.write(f"\n\n=== NEW SESSION STARTED AT {datetime.now()} ===\n{header}\n")
    else:
        with open(MONITOR_FILE, "w") as f:
            f.write(f"=== RANDOM SEARCH SESSION STARTED AT {datetime.now()} ===\n{header}\n")
            
    print(f"Starting Random Search Dispatcher. Monitoring: {MONITOR_FILE}")
    print(f"Total Experiments: {TOTAL_EXPERIMENTS}. GPUs: {NUM_GPUS}")
    
    # Task Queue
    queue = [sample_hyperparams(i) for i in range(1, TOTAL_EXPERIMENTS + 1)]
    
    # Process Management
    # procs[gpu_id] = Popen object or None
    procs = [None] * NUM_GPUS
    
    while queue or any(p is not None for p in procs):
        for gpu_id in range(NUM_GPUS):
            # Check if current proc is done
            if procs[gpu_id] is not None:
                ret = procs[gpu_id].poll()
                if ret is not None: # Process finished
                    print(f"GPU {gpu_id} finished task. Return code: {ret}")
                    procs[gpu_id] = None
            
            # Assign new task if idle and queue not empty
            if procs[gpu_id] is None and queue:
                params = queue.pop(0)
                cmd = [
                    sys.executable, "-u", "random_search/train_worker.py",
                    "--exp_name", params["exp_name"],
                    "--gpu_id", str(gpu_id),
                    "--monitor_file", str(MONITOR_FILE),
                    "--lr", str(params["lr"]), 
                    "--gamma", str(params["gamma"]), 
                    "--batch_size", str(params["batch_size"]), 
                    "--buffer_size", str(params["buffer_size"]), 
                    "--head_hidden_dim", str(params["head_hidden_dim"]), 
                    "--tau", str(params["tau"]), 
                    # Reduce duration for search
                    "--max_steps", "200000" # 200k blocks = ~4M steps? No, 200k * 20 = 4M steps.
                                            # Let's do smaller: 50k blocks = 1M steps.
                ]
                
                log_msg = f"[Dispatcher] Launching {params['exp_name']} on GPU {gpu_id} | LR:{params['lr']:.2e} GM:{params['gamma']} BS:{params['batch_size']}"
                print(log_msg)
                with open(MONITOR_FILE, "a") as f: f.write(f"{datetime.now()} {log_msg}\n")
                
                # Redirect stdout/stderr to individual log file in the experiment folder
                # Create exp folder first
                exp_path = SEARCH_DIR / params["exp_name"]
                exp_path.mkdir(exist_ok=True)
                
                log_path = exp_path / "console.log"
                with open(log_path, "w") as out:
                    procs[gpu_id] = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=Path.cwd())
        
        time.sleep(5) # Poll every 5s

    print("All experiments completed.")

if __name__ == "__main__":
    main()
