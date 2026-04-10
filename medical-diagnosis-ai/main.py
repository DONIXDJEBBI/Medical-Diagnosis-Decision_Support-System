#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.evaluate import run_evaluation
from src.compare import run_comparison


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Medical Diagnosis Decision Support System"
    )
    
    parser.add_argument(
        "--model",
        choices=["fuzzy", "tree"],
        help="Run specific model (fuzzy or tree)"
    )
    parser.add_argument(
        "--mode",
        choices=["compare"],
        help="Run comparison mode"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Decision tree max depth (default: 5)"
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=10,
        help="Decision tree min samples per leaf (default: 10)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "compare":
        print("\n" + "="*70)
        print("COMPARING FUZZY LOGIC vs DECISION TREE")
        print("="*70 + "\n")
        run_comparison(
            tree_max_depth=args.max_depth,
            tree_min_samples_leaf=args.min_samples_leaf,
        )
    elif args.model == "fuzzy":
        print("\n" + "="*70)
        print("RUNNING FUZZY LOGIC MODEL")
        print("="*70 + "\n")
        run_evaluation(model_type="fuzzy_only")
    elif args.model == "tree":
        print("\n" + "="*70)
        print("RUNNING DECISION TREE MODEL")
        print("="*70 + "\n")
        run_evaluation(
            model_type="tree_only",
            tree_max_depth=args.max_depth,
            tree_min_samples_leaf=args.min_samples_leaf,
        )
    else:
        # Default: run comparison
        print("\n" + "="*70)
        print("COMPARING FUZZY LOGIC vs DECISION TREE")
        print("="*70 + "\n")
        run_comparison(
            tree_max_depth=args.max_depth,
            tree_min_samples_leaf=args.min_samples_leaf,
        )


if __name__ == "__main__":
    main()
