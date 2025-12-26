
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.aggregators import monthly_snapshots_for_backtest, weekly_snapshots_for_backtest
from src.pipelines import compute_indicators
from src.config import STATUS_FINAL, STATUS_LATEST

def calculate_trend_state(
    values: pd.Series, 
    statuses: pd.Series, 
    status_final_code: int = STATUS_FINAL
) -> pd.Series:
    """
    Compute discrete trend state based on stream values and commit flags.
    
    States:
     2: Turn Up (Down -> Up)
     1: Keep Up (Up -> Up)
     0: Neutral/Same
    -1: Keep Down (Down -> Down)
    -2: Turn Down (Up -> Down)
    """
    states = np.zeros(len(values), dtype=int)
    
    # Tracking variables
    last_confirmed_val: float = np.nan
    last_trend_dir: int = 0 # 1 for up, -1 for down, 0 neutral
    
    # Convert to numpy for speed
    vals = values.values
    stats = statuses.values
    
    # Initialize with first valid value if possible
    # We need to find the first confirmed value to start "trend" logic properly?
    # Or just start from the beginning.
    
    for i in range(len(vals)):
        curr_val = vals[i]
        curr_stat = stats[i]
        
        if np.isnan(curr_val):
            states[i] = 0
            continue
            
        if np.isnan(last_confirmed_val):
            # First value encountered
            if curr_stat == status_final_code:
                last_confirmed_val = curr_val
            continue
            
        # Compare current 'live' value with last 'confirmed' value
        diff = curr_val - last_confirmed_val
        
        curr_dir = 0
        if diff > 1e-9: # Float tolerance
            curr_dir = 1
        elif diff < -1e-9:
            curr_dir = -1
            
        # Determine State
        if curr_dir == 1:
            if last_trend_dir == 1:
                states[i] = 1 # Keep Up
            elif last_trend_dir == -1:
                states[i] = 2 # Turn Up
            else:
                states[i] = 1 # Initial Up
        elif curr_dir == -1:
            if last_trend_dir == -1:
                states[i] = -1 # Keep Down
            elif last_trend_dir == 1:
                states[i] = -2 # Turn Down
            else:
                states[i] = -1 # Initial Down
        else:
            states[i] = 0
            
        # If this row is a commit point (Final), update the 'last' memory
        if curr_stat == status_final_code:
            last_confirmed_val = curr_val
            if curr_dir != 0:
                last_trend_dir = curr_dir
                
    return pd.Series(states, index=values.index)

def process_symbol(symbol: str, daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate feature dataframe for a single symbol.
    """
    # Ensure daily_df has status column (required by normalize_prices)
    if "status" not in daily_df.columns:
        daily_df = daily_df.copy()
        daily_df["status"] = STATUS_FINAL

    # 1. Generate Stream Snapshots
    weekly_df = weekly_snapshots_for_backtest(daily_df)
    monthly_df = monthly_snapshots_for_backtest(daily_df)
    
    # 2. Compute Indicators (Stream Mode)
    # We need to mark 'update_flag' for the pipeline to know when to commit
    weekly_df["update_flag"] = weekly_df["status"] >= STATUS_LATEST
    monthly_df["update_flag"] = monthly_df["status"] >= STATUS_LATEST
    
    weekly_ind = compute_indicators(weekly_df, mode="stream")
    monthly_ind = compute_indicators(monthly_df, mode="stream")
    
    # 3. Discretize Features
    # Full feature list:
    # MACD: macd_dif, macd_dea, macd_bar
    # SKDJ: skdj_k, skdj_d
    # DMA: dma, ama
    # BBIBOLL: bbiboll, bbiboll_upr, bbiboll_dwn
    
    indicators = [
        "macd_dif", "macd_dea", "macd_bar",
        "skdj_k", "skdj_d",
        "dma", "ama",
        "bbiboll", "bbiboll_upr", "bbiboll_dwn"
    ]
    
    feature_cols = []
    
    # Weekly Features
    for col in indicators:
        if col in weekly_ind.columns:
            state_col = f"week_{col}_state"
            weekly_ind[state_col] = calculate_trend_state(
                weekly_ind[col], weekly_ind["status"], STATUS_FINAL
            )
            feature_cols.append(state_col)
            
    # Monthly Features
    for col in indicators:
        if col in monthly_ind.columns:
            state_col = f"month_{col}_state"
            monthly_ind[state_col] = calculate_trend_state(
                monthly_ind[col], monthly_ind["status"], STATUS_FINAL
            )
            feature_cols.append(state_col)
            
    # 4. Merge back to Daily
    # Preserve status columns for env.py indexing
    weekly_ind = weekly_ind.rename(columns={"status": "week_status"})
    monthly_ind = monthly_ind.rename(columns={"status": "month_status"})
    
    # Keep state columns, status, and ts
    w_cols = ["ts", "week_status"] + [c for c in feature_cols if c.startswith("week_")]
    m_cols = ["ts", "month_status"] + [c for c in feature_cols if c.startswith("month_")]
    
    # Select columns from ind frames
    w_feat = weekly_ind[w_cols]
    m_feat = monthly_ind[m_cols]
    
    merged = pd.merge(daily_df[["ts", "open", "high", "low", "close", "volume"]], w_feat, on="ts", how="left")
    merged = pd.merge(merged, m_feat, on="ts", how="left")
    
    # Fill NA states with 0, NA status with 1 (Intra)
    state_cols = [c for c in merged.columns if c.endswith("_state")]
    merged[state_cols] = merged[state_cols].fillna(0).astype(int)
    merged["week_status"] = merged["week_status"].fillna(1).astype(int)
    merged["month_status"] = merged["month_status"].fillna(1).astype(int)
    
    # 5. Generate Warm Start Labels (Oracle)
    # Revised Rule based on Close prices:
    # Buy (1): Future 5D Max(Close) / Current(Close) > 1.10 AND Future 5D Min(Close) / Current(Close) > 0.95
    # Sell (2): (Future 5D Max(Close) / Current(Close) < 1.10 AND Min < 0.90) OR (Max <= 1.00)
    
    # Rolling window forward looking on CLOSE prices
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=5)
    fwd_max_close = merged["close"].rolling(window=indexer).max()
    fwd_min_close = merged["close"].rolling(window=indexer).min()
    
    curr_close = merged["close"]
    
    max_c_ratio = fwd_max_close / curr_close
    min_c_ratio = fwd_min_close / curr_close
    
    # Labels
    labels = np.zeros(len(merged), dtype=int)
    
    # Condition Buy
    cond_buy = (max_c_ratio > 1.10) & (min_c_ratio > 0.95)
    
    # Condition Sell
    cond_sell = ((max_c_ratio < 1.10) & (min_c_ratio < 0.90)) | (max_c_ratio <= 1.00)
    
    labels[cond_buy] = 1 # Buy
    labels[cond_sell] = 2 # Sell
    
    merged["warm_start_label"] = labels
    
    # Add symbol column
    merged["symbol"] = symbol
    
    return merged

def build_dataset(duckdb_dir: Path):
    import duckdb
    
    all_files = sorted(list(duckdb_dir.glob("*.duckdb")) + list(duckdb_dir.glob("*.db")))
    print(f"Found {len(all_files)} symbol files in {duckdb_dir}.")
    
    processed_count = 0
    
    for p in all_files:
        full_symbol = p.stem
        
        try:
            # 1. Load raw data
            con = duckdb.connect(str(p), read_only=False) # Open for writing
            df = con.execute("SELECT * FROM daily_data ORDER BY date").df()
            
            # Rename date to ts if needed
            if "date" in df.columns:
                df = df.rename(columns={"date": "ts"})
            
            df["ts"] = pd.to_datetime(df["ts"])
            
            # Process features on FULL history to ensure indicator accuracy (warmup)
            if len(df) < 100:
                con.close()
                continue
                
            feat_df = process_symbol(full_symbol, df)
            
            # Filter last 20 years for storage/training
            cutoff = pd.Timestamp.now() - pd.DateOffset(years=20)
            feat_df = feat_df[feat_df["ts"] >= cutoff].copy()
            
            if feat_df.empty:
                con.close()
                continue
            
            # 3. Save table back to the same file
            con.execute("CREATE OR REPLACE TABLE features AS SELECT * FROM feat_df")
            con.close()
            
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Updated {processed_count} symbols with features...")
                
        except Exception as e:
            print(f"Error processing {full_symbol}: {e}")
            
    print(f"Done. Integrated features into {processed_count} files.")

if __name__ == "__main__":
    # Adjust paths as needed
    ROOT = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = ROOT / "data" / "A"
    
    # NEW: Check for command line argument to process single symbol
    if len(sys.argv) > 1:
        target_symbol = sys.argv[1]
        print(f"Single symbol mode: Processing {target_symbol}")
        import duckdb
        p = DATA_DIR / f"{target_symbol}.duckdb"
        if not p.exists():
            # Support both .db and .duckdb
            p = DATA_DIR / f"{target_symbol}.db"
            
        if p.exists():
            con = duckdb.connect(str(p), read_only=False)
            df = con.execute("SELECT * FROM daily_data ORDER BY date").df()
            if "date" in df.columns: df = df.rename(columns={"date": "ts"})
            df["ts"] = pd.to_datetime(df["ts"])
            if len(df) >= 100:
                feat_df = process_symbol(target_symbol, df)
                cutoff = pd.Timestamp.now() - pd.DateOffset(years=20)
                feat_df = feat_df[feat_df["ts"] >= cutoff].copy()
                if not feat_df.empty:
                    con.execute("CREATE OR REPLACE TABLE features AS SELECT * FROM feat_df")
                    print(f"Successfully generated features for {target_symbol}")
            con.close()
        else:
            print(f"File not found for {target_symbol}")
    else:
        build_dataset(DATA_DIR)
