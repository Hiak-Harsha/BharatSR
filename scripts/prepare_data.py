"""
BharatSR — Data Pipeline CLI Entrypoint
Prepares train, validation, and test datasets with scene-level separation.
Usage: python scripts/prepare_data.py [--scenes 25] [--seed 42]
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.scripts.prepare_data import prepare_dataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare BharatSR Training and Evaluation Datasets")
    parser.add_argument("--scenes", type=int, default=24, help="Number of scene pairs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--force_synthetic", action="store_true", default=False, help="Force procedural synthetic pairs")
    args = parser.parse_args()

    print(f"Executing BharatSR Data Preparation Pipeline (seed={args.seed}, scenes={args.scenes})...")
    prepare_dataset(force_synthetic=args.force_synthetic, seed=args.seed)
    print("Data preparation complete.")
