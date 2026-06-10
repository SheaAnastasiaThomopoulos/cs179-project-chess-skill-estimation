"""
run_pipeline.py
---------------
Runs all four pipeline steps in order.

    python run_pipeline.py              # fetch from Lichess API
    python run_pipeline.py --synthetic  # skip API, use synthetic data

Author: Pragya Jhunjhunwala
"""

import subprocess
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent


def run(cmd):
    print(f"\n{'='*60}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\nERROR: step failed with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data instead of Lichess API")
    parser.add_argument("--draws",  type=int, default=1000)
    parser.add_argument("--tune",   type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    args = parser.parse_args()

    py = sys.executable

    fetch_cmd = [py, "src/fetch_data.py"]
    if args.synthetic:
        fetch_cmd.append("--synthetic")
    run(fetch_cmd)

    run([py, "src/preprocess.py"])

    run([py, "src/model.py",
         f"--draws={args.draws}",
         f"--tune={args.tune}",
         f"--chains={args.chains}"])

    run([py, "src/evaluate.py"])

    print("\n" + "="*60)
    print("  Pipeline complete! Figures saved to results/")
    print("="*60)


if __name__ == "__main__":
    main()