
import duckdb
from pathlib import Path

data_dir = Path("data/A")
files = list(data_dir.glob("*.duckdb"))
if files:
    con = duckdb.connect(str(files[0]), read_only=True)
    df = con.execute("SELECT * FROM features LIMIT 1").df()
    print("Columns found:")
    for c in sorted(df.columns):
        print(c)
    con.close()
