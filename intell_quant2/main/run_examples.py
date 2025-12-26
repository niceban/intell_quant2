"""Run all example scripts in order."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "preprocessing"
SCRIPTS = [
    "01_generate_mock_daily.py",
    "02_build_weekly_frames.py",
    "03_batch_indicators.py",
    "04_stream_indicators.py",
    "05_compare_weekly_modes.py",
    "06_backtest.py",
    "07_rules_sweep.py",
]


def run_script(path: Path) -> int:
    print(f"\n==> {path.name}")
    # Run from project root so that 'src' is importable if cwd is in sys.path
    # We also explicitly set PYTHONPATH to ensure src is found
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, str(path)], cwd=ROOT, env=env)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all example scripts in sequence.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining scripts even if one fails.",
    )
    args = parser.parse_args()

    for name in SCRIPTS:
        path = EXAMPLES_DIR / name
        if not path.exists():
            print(f"[skip] {name}: not found")
            continue
        code = run_script(path)
        if code != 0:
            print(f"[fail] {name} exited with code {code}")
            if not args.continue_on_error:
                sys.exit(code)

    print("\nDone.")


if __name__ == "__main__":
    main()
