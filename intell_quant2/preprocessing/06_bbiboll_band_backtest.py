"""Run bbiboll-band analysis/backtest without touching core strategy."""

from pathlib import Path

from intell_quant2.analysis.bbiboll_bands import DEFAULT_DUCKDB_DIR, DEFAULT_OUTPUT_DIR, run_all


def main():
    run_all(duckdb_dir=DEFAULT_DUCKDB_DIR, output_dir=DEFAULT_OUTPUT_DIR)


if __name__ == "__main__":
    main()

