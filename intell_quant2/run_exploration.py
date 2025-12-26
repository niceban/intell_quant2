import subprocess
import sys
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parent
FEATURE_DIR = ROOT / "data" / "A"

def check_features_ready():
    files = list(FEATURE_DIR.glob("*.duckdb"))
    if not files:
        return False
    # Check sample file
    try:
        con = duckdb.connect(str(files[0]), read_only=True)
        res = con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'features'").fetchone()
        con.close()
        return res[0] > 0
    except:
        return False

def main():
    # 1. Check Features
    if not check_features_ready():
        print("Features not integrated into data/A. Running feature generator...")
        subprocess.run([sys.executable, "exploring/features/feature_generator.py"], cwd=ROOT, check=True)
    else:
        print("Integrated features found in data/A. Skipping generation.")
        
    # 2. Run Training (Full Mode)
    print("Starting RL Training (FULL PRODUCTION MODE - v-Final Spec)...")
    cmd = [
        sys.executable, "-u", "exploring/single_process_test.py",
        "--exp_name", "v4_nonlinear_scaled_final",
        "--num_envs", "1024",
        "--batch_size", "1024",
        "--collect_steps_per_block", "20",
        "--warm_start_steps", "1000",
        "--eval_freq", "500",
        "--num_episodes", "5000000"
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)

if __name__ == "__main__":
    main()