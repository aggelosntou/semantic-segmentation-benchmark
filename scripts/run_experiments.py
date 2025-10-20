#!/usr/bin/env python3
"""
Simple runner script for pseudo-labeling experiments.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run the experiments with different configurations."""

    print(" Starting Pseudo-Labeling Experiments")
    print("=" * 60)

    # Quick test run (fewer experiments / debug mode)
    args = ["python", "pseudo_labeling_experiments.py", "--output", "experiment_results.csv"]

    if "--test" in sys.argv:
        print("Running in TEST MODE (this just forwards to pseudo_labeling_experiments.py)")
        # if you wanted: add a flag like --test to pseudo_labeling_experiments.py

    print(f"Running command: {' '.join(args)}")
    print("=" * 60)

    # Run the experiments
    try:
        result = subprocess.run(args, check=True)
        print("\n All experiments completed successfully!")

        # Show results file
        results_file = Path("experiment_results.csv")
        if results_file.exists():
            print(f" Results saved to: {results_file}")
            print(f" Model checkpoints saved to: checkpoints/")

    except subprocess.CalledProcessError as e:
        print(f"\n Experiments failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n  Experiments interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage:")
        print("  python run_experiments.py        # Full run")
        print("  python run_experiments.py --test # Quick test (if supported)")
    else:
        main()
