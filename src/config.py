import argparse
import os
from dataclasses import dataclass

@dataclass
class ArgsType:
    mode: str
    dataset: str
    val_size: float
    relaxed_stopping: bool
    n_estimators: int
    voting: str
    bootstrap: bool
    max_samples: float
    cal_ratio: float
    epsilon: float
    seed: int
    max_depth: int
    min_samples: int

# Absolute path directory containing this
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level --> project root
PROJECT_ROOT = os.path.abspath(os.path.join(CONFIG_DIR, '..'))

config = {
    'PROJECT_ROOT': PROJECT_ROOT,
    'DATASET_FOLDER': os.path.join(PROJECT_ROOT, 'datasets')
}

def parse_args() -> ArgsType:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate PromptTree or RandomForest."
    )

    # Mode Selection
    parser.add_argument(
        "--mode",
        type=str,
        default="single_tree",
        choices=["single_tree", "forest", "conformal", "unsupervised"],
        help="Mode to run",
    )

    # Dataset Selection
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["ECG200", "ECG5000", "ECGFiveDays", "PenDigits", "Heartbeat"],
        help="Name of the dataset to load. Must match a known dataset in load_dataset.",
    )

    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="Proportion of data to use for validation set.",
    )

    # Single Tree Specific Arguments
    parser.add_argument(
        "--relaxed_stopping",
        action="store_true",
        help="For single_tree mode: Use relaxed stopping criteria before pruning.",
    )

    # Forest Specific Arguments
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="For forest mode: Number of trees in the forest.",
    )
    parser.add_argument(
        "--voting",
        type=str,
        default="weighted",
        choices=["majority", "weighted", "track-record"],
        help="For forest mode: Voting mechanism to use.",
    )

    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="For forest mode: Whether to use bootstrap sampling for each tree.",
    )

    parser.add_argument(
        "--max_samples",
        type=float,
        default=0.7,
        help="For forest mode: Proportion of samples to draw for each tree if bootstrap is True (0 < max_samples <= 1).",
    )

    # Conformal Specific Arguments
    parser.add_argument(
        "--cal_ratio", type=float, default=0.3,
        help="For conformal mode: Proportion of training data for calibration (0 < ratio < 1)."
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.1,
        help="For conformal mode: Desired significance level (0 < epsilon < 1)."
    )

    # General Arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data splitting and forest building.",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=5,
        help="Maximum depth of the tree(s) for stopping criterion.",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=2,
        help="Minimum number of samples required to split a node.",
    )

    args = parser.parse_args()

    # Argument Validation
    if args.mode == "conformal":
        if not (0 < args.cal_ratio < 1):
             parser.error("--cal_ratio must be between 0 and 1 (exclusive)")
        if not (0 < args.epsilon < 1):
             parser.error("--epsilon must be between 0 and 1 (exclusive)")
        if args.val_size != 0.0:
            print("Info: --val_size is ignored in conformal mode. Setting to 0 internally.")
            args.val_size = 0.0 # Ensure it's 0 for conformal split logic
    elif args.mode == "unsupervised":
         if args.val_size != 0.0:
            print("Info: --val_size is ignored in unsupervised mode. Setting to 0 internally.")
            args.val_size = 0.0
    if args.bootstrap and not (0 < args.max_samples <= 1):
         parser.error("--max_samples must be between 0 and 1 (exclusive) when bootstrap is True")
    if not (0 <= args.val_size < 1):
         parser.error("--val_size must be between 0 and 1 (inclusive of 0)")

    return args