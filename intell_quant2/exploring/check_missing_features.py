
import duckdb
from pathlib import Path

DATA_DIR = Path("data/A")
all_files = sorted(list(DATA_DIR.glob("*.duckdb")))

missing_features = []
no_daily_data = []

for p in all_files:
    try:
        con = duckdb.connect(str(p), read_only=True)
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        if 'features' not in tables:
            missing_features.append(p.stem)
        
        if 'daily_data' not in tables:
            no_daily_data.append(p.stem)
        else:
            count = con.execute("SELECT count(*) FROM daily_data").fetchone()[0]
            if count < 100:
                no_daily_data.append(f"{p.stem} (count={count})")
        con.close()
    except Exception as e:
        print(f"Error checking {p.name}: {e}")

print("Symbols missing 'features' table:")
print(missing_features)
print("\nSymbols with insufficient/missing 'daily_data':")
print(no_daily_data)
