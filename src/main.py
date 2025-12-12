import copy
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Datasets
from dataset.ecg import ECGDataset
from dataset.heartbeat import HeartbeatDataset
from dataset.penDigits import PenDigitsDataset

# Core
from core.tree import PromptTree
from core.forest import RandomForest
from core.utils import (
    evaluate_tree_accuracy,
    evaluate_forest_accuracy,
    calculate_nonconformity_scores,
    get_prediction_sets,
    calculate_cluster_distances,
    calculate_entropy,
    calculate_purity,
    visualize_linkage
)
from core.tree_functions import (
    fe_default,
    fo_unsupervised_random,
    fe_unsupervised_default,
)
from core.distances import (
    compute_breiman_distance_matrix,
    compute_zhu_distance_matrix,
    compute_ratioRF_distance_matrix,
)
from config import config, parse_args


def load_dataset(
    dataset_name: str, validation_size: float = 0.2, random_state: int = 42
):
    """Loads the specified dataset."""
    print(f"Loading dataset: {dataset_name}...")
    try:
        match dataset_name:
            case "ECG200":
                dataset = ECGDataset(
                    ecg_dataset_name=dataset_name,
                    validation_size=validation_size,
                    random_state=random_state,
                )
            case "ECG5000":
                dataset = ECGDataset(
                    ecg_dataset_name=dataset_name,
                    validation_size=validation_size,
                    random_state=random_state,
                )
            case "ECGFiveDays":
                dataset = ECGDataset(
                    ecg_dataset_name=dataset_name,
                    validation_size=validation_size,
                    random_state=random_state,
                )
            case "PenDigits":
                dataset = PenDigitsDataset(
                    validation_size=validation_size, random_state=random_state
                )
            case "Heartbeat":
                dataset = HeartbeatDataset(
                    validation_size=validation_size, random_state=random_state
                )
            case _:
                raise ValueError(
                    f"Unknown dataset name: {dataset_name}. Add it to load_dataset function."
                )

        print(f"Successfully loaded {dataset.dataset_name}.")
        print(f"  Train shape: {dataset.X_train.shape}")
        print(f"  Validation shape: {dataset.X_val.shape}")
        print(f"  Test shape: {dataset.X_test.shape}")
        print(f"  Classes: {dataset.get_class_names()}")
        dataset.true_num_classes = len(dataset.label_map)
        print(f"  True number of classes: {dataset.true_num_classes}")
        return dataset
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the dataset exists in the expected directory structure.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def main_single_tree(dataset, relaxed_stopping: bool):
    """Trains, prunes, evaluates, and visualizes a single PromptTree."""

    if relaxed_stopping:
        print("\nUsing relaxed stopping criteria...")
        stopper = lambda Path, X, Y, depth: fe_default(
            Path, X, Y, depth, max_depth=20, min_samples=2
        )
        original_tree = PromptTree(fe=stopper)
        stopping_path = "relaxed"
    else:
        print("\nUsing default stopping criteria...")
        original_tree = PromptTree()
        stopping_path = "default"

    print("\nFitting the tree...")
    original_tree.fit(dataset.X_train, dataset.Y_train)

    # Pruning
    pruned_tree = copy.deepcopy(original_tree)
    print("\nPruning the tree using the validation set...")
    pruned_tree.prune(dataset.X_val, dataset.Y_val)
    print("Pruning complete.")

    # Evaluation
    print(f"\nOriginal tree leaves: {len(original_tree.get_paths())}")
    print(f"Pruned tree leaves: {len(pruned_tree.get_paths())}")

    original_accuracy = evaluate_tree_accuracy(
        original_tree, dataset.X_test, dataset.Y_test
    )
    pruned_accuracy = evaluate_tree_accuracy(
        pruned_tree, dataset.X_test, dataset.Y_test
    )

    print("\n--- FINAL RESULTS (Single Tree) ---")
    print(f"{'Original Tree Test Accuracy:':<30} {original_accuracy * 100:.2f}%")
    print(f"{'Pruned Tree Test Accuracy:':<30} {pruned_accuracy * 100:.2f}%")

    if pruned_accuracy > original_accuracy:
        print("\nResult: Pruning improved generalization.")
    elif pruned_accuracy == original_accuracy:
        print(
            "\nResult: Pruning resulted in the same accuracy (but likely a simpler model)."
        )
    else:
        print("\nResult: Pruning did not improve accuracy for this dataset/split.")

    # Visualization
    print("\nVisualizing trees...")
    # Use config for base output path
    output_base_dir = os.path.join(config.get("PROJECT_ROOT"), "output")
    output_dir = os.path.join(output_base_dir, dataset.dataset_name, stopping_path)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to: {output_dir}")

    try:
        original_tree.visualize_tree(
            dataset.get_class_names(),
            filename=os.path.join(output_dir, f"original_{dataset.dataset_name}_tree"),
        )
    except Exception as e:
        print(f"Warning: Could not visualize original tree. Error: {e}")

    try:
        pruned_tree.visualize_tree(
            dataset.get_class_names(),
            filename=os.path.join(output_dir, f"pruned_{dataset.dataset_name}_tree"),
        )
    except Exception as e:
        print(f"Warning: Could not visualize pruned tree. Error: {e}")


def main_forest(dataset, n_estimators: int, voting_mechanism: str, random_state: int):
    """Main logic Random Forest: trains and evaluates a Random Forest."""
    print(f"\n--- Random Forest Experiment ---")
    print(f"  Dataset: {dataset.dataset_name}")
    print(f"  Num Estimators: {n_estimators}")
    print(f"  Voting: {voting_mechanism}")

    forest = RandomForest(
        n_estimators=n_estimators,
        mode="supervised",
        random_state=random_state,
    )

    print("\nFitting the forest...")
    forest.fit(dataset.X_train, dataset.Y_train)

    print("\nEvaluating forest on test set...")
    accuracy = evaluate_forest_accuracy(
        forest, dataset.X_test, dataset.Y_test, voting=voting_mechanism
    )

    print(f"\n--- FINAL RESULTS (Random Forest) ---")
    print(f"Forest Test Accuracy ({voting_mechanism} voting): {accuracy * 100:.2f}%")

    # Print OOB score if available
    if (
        hasattr(forest, "oob_scores")
        and forest.oob_scores is not None
        and np.any(forest.oob_scores > 0)
    ):
        oob_mean = np.mean(forest.oob_scores[forest.oob_scores > 0])
        print(f"Overall Forest OOB Accuracy Estimate: {oob_mean * 100:.2f}%")


def main_conformal(
    dataset,
    n_estimators: int,
    voting_mechanism: str,
    random_state: int,
    calibration_ratio: float = 0.3,
    significance_level: float = 0.1,
):
    """Performs conformal prediction using a Random Forest."""
    print(f"\n--- Conformal Prediction Experiment ---")
    print(f"  Dataset: {dataset.dataset_name}")
    print(f"  Num Estimators: {n_estimators}")
    print(f"  Voting: {voting_mechanism}")
    print(f"  Calibration Ratio: {calibration_ratio}")
    print(f"  Significance Level (epsilon): {significance_level}")

    # Split Training data into Proper Training and Calibration sets
    X_train_full = dataset.X_train
    Y_train_full = dataset.Y_train

    if len(X_train_full) < 2 or calibration_ratio <= 0.0 or calibration_ratio >= 1.0:
        print(
            "Error: Need at least 2 samples in training data and a valid calibration_ratio (0<ratio<1) for conformal split."
        )
        return

    X_train_prop, X_cal, Y_train_prop, Y_cal = train_test_split(
        X_train_full,
        Y_train_full,
        test_size=calibration_ratio,
        random_state=random_state,
        stratify=Y_train_full,
    )
    print(
        f"\nSplit data: Proper Training ({len(X_train_prop)}), Calibration ({len(X_cal)})"
    )
    if len(X_cal) == 0:
        print(
            "Error: Calibration set is empty after split. Check data and calibration_ratio."
        )
        return

    # Train the Random Forest on the Proper Training Set
    model = RandomForest(
        n_estimators=n_estimators,
        mode="supervised",
        random_state=random_state,
        bootstrap=True,  # Ensure bootstrapping is enabled if OOB is needed
    )
    print("\nFitting the model on the proper training set...")
    model.fit(X_train_prop, Y_train_prop)

    # Calculate Nonconformity Scores on the Calibration Set
    nonconformity_scores = calculate_nonconformity_scores(
        model, X_cal, Y_cal, voting=voting_mechanism
    )
    if len(nonconformity_scores) == 0:
        print("Error: Failed to calculate nonconformity scores. Aborting.")
        return

    # Generate Prediction Sets for the Test Set
    prediction_sets = get_prediction_sets(
        model,
        dataset.X_test,
        nonconformity_scores,
        significance_level,
        voting=voting_mechanism,
    )
    if not prediction_sets or len(prediction_sets) != len(dataset.X_test):
        print(
            "Error: Failed to generate prediction sets correctly. Aborting evaluation."
        )
        return

    # Evaluate Conformal Classifier
    print("\n--- Conformal Classifier Evaluation ---")

    # Check Miscalibration
    error_count = 0
    test_set_size = len(dataset.X_test)
    if test_set_size > 0:
        for i in range(test_set_size):
            true_label = dataset.Y_test[i]
            # Check if the true label is NOT in the prediction set
            if true_label not in prediction_sets[i]:
                error_count += 1
        empirical_error_rate = error_count / test_set_size
    else:
        empirical_error_rate = 0.0
        print("Warning: Test set is empty. Cannot calculate error rate.")

    print(f"  Significance Level (epsilon): {significance_level:.3f}")
    print(
        f"  Empirical Error Rate:       {empirical_error_rate:.3f} ({error_count}/{test_set_size} errors)"
    )
    # Check calibration allowing for slight numerical tolerance
    if empirical_error_rate <= significance_level + 1e-6:
        print("  Result: Classifier is adequately calibrated (Error <= epsilon).")
    else:
        print("  Result: Classifier appears miscalibrated (Error > epsilon).")

    # Compute Efficiency (Average Prediction Set Size)
    set_sizes = [
        len(s) for s in prediction_sets if s
    ]  # Avoid counting empty error sets if any
    average_set_size = np.mean(set_sizes) if set_sizes else 0.0
    median_set_size = np.median(set_sizes) if set_sizes else 0.0  # Also useful
    print(f"\n  Efficiency:")
    print(f"    Average Set Size: {average_set_size:.3f}")
    print(f"    Median Set Size:  {median_set_size:.1f}")

    # Standard Accuracy for comparison
    print("\n  For reference:")
    try:
        point_accuracy = evaluate_forest_accuracy(
            model, dataset.X_test, dataset.Y_test, voting=voting_mechanism
        )
        print(f"    Underlying Model Point Accuracy: {point_accuracy * 100:.2f}%")
    except Exception as e:
        print(f"    Could not compute point prediction accuracy: {e}")


def main_unsupervised(dataset, n_estimators: int, random_state: int):
    """Performs unsupervised clustering using Isolation Forest distances."""
    print(f"\n--- Unsupervised Clustering Experiment ---")
    print(f"  Dataset: {dataset.dataset_name}")
    print(f"  Num Estimators (Isolation Forest): {n_estimators}")
    print(f"  Target Clusters: {dataset.true_num_classes}")

    # Ensure validation size was 0, use full training data for clustering
    if dataset.X_val.shape[0] > 0:
        print(
            "Warning: Unsupervised mode typically uses the full training set. Validation split will be ignored."
        )
    X_cluster_data = dataset.X_train  # Use training data for clustering
    Y_true_labels = dataset.Y_train  # Keep true labels for EXTERNAL evaluation later

    # Initialize and Fit Isolation Forest
    print("\nFitting Isolation Forest...")
    isolation_forest = RandomForest(
        n_estimators=n_estimators,
        mode="unsupervised",
        random_state=random_state,
        bootstrap=True,
        max_samples=0.8, # Using 80% subsampling
        fo_unsupervised=fo_unsupervised_random,
        fe_unsupervised=fe_unsupervised_default,
    )
    isolation_forest.fit(X_cluster_data, Y=None)

    # Get Leaf Assignments
    print("\nGetting leaf node assignments...")
    try:
        leaf_assignments = isolation_forest.get_leaf_assignments(X_cluster_data)
        if leaf_assignments.size == 0 or leaf_assignments.shape[0] != len(
            X_cluster_data
        ):
            raise ValueError("Leaf assignments matrix is empty or has incorrect shape.")
    except Exception as e:
        print(f"Error getting leaf assignments: {e}. Cannot proceed with clustering.")
        return

    # Calculate Distances, Cluster, and Evaluate for each metric
    distance_functions = {
        "Breiman": compute_breiman_distance_matrix,
        "Zhu": compute_zhu_distance_matrix,
        "RatioRF": compute_ratioRF_distance_matrix,
    }

    results = {}  # Store clustering results (labels and evaluations)

    for dist_name, dist_func in distance_functions.items():
        print(f"\n--- Processing Distance Metric: {dist_name} ---")

        # Calculate Distance Matrix
        print(f"Calculating {dist_name} distance matrix...")
        try:
            if dist_name == "Zhu":
                # Zhu distance needs the forest and the data (X)
                distance_matrix = dist_func(isolation_forest, X_cluster_data)
            else:
                # Breiman and RatioRF just need the leaf assignments
                distance_matrix = dist_func(leaf_assignments)

            if distance_matrix.shape != (len(X_cluster_data), len(X_cluster_data)):
                raise ValueError("Distance matrix has incorrect shape.")
            if np.isnan(distance_matrix).any() or np.isinf(distance_matrix).any():
                print(
                    f"Warning: Distance matrix for {dist_name} contains NaN or Inf values. Clustering might fail."
                )
                # Let's try to proceed, clustering will raise the error if it fails
        
        except Exception as e:
            print(f"Error calculating {dist_name} distance: {e}. Skipping this metric.")
            continue

        # Perform Hierarchical Clustering
        print("Performing hierarchical clustering (average linkage)...")
        try:
            cluster_labels = perform_clustering(distance_matrix, dataset.true_num_classes, dist_name)
        except Exception as e:
            print(
                f"Error during clustering for {dist_name}: {e}. Skipping this metric."
            )
            continue

        # Evaluate Clustering
        print("Evaluating clustering results...")
        eval_results = evaluate_clustering(distance_matrix, cluster_labels, Y_true_labels)

        # Store results for this distance metric
        results[dist_name] = {"clusters": cluster_labels, "evaluation": eval_results}

    # Clustering Comparison (ARI among the different clusterings)
    print("\n--- Clustering Comparison (ARI) ---")
    dist_names = list(results.keys())
    if len(dist_names) >= 2:
        for i in range(len(dist_names)):
            for j in range(i + 1, len(dist_names)):
                name1 = dist_names[i]
                name2 = dist_names[j]
                try:
                    ari_comparison = adjusted_rand_score(
                        results[name1]["clusters"], results[name2]["clusters"]
                    )
                    print(f"  ARI between {name1} and {name2}: {ari_comparison:.4f}")
                except Exception as e:
                    print(f"  Could not compare {name1} and {name2}: {e}")
    else:
        print("  Not enough valid clustering results to compare.")

def perform_clustering(distance_matrix, num_classes, dist_name: str):
    """Helper to perform hierarchical clustering."""
    # Convert to condensed form for SciPy
    condensed_dist = squareform(distance_matrix, checks=True) # Enable checks
    
    # Check for non-finite values *before* linkage
    if not np.all(np.isfinite(condensed_dist)):
        # Replace inf with a large number, handle nan
        print("Warning: Non-finite values found in condensed distance matrix. Replacing with max finite value.")
        max_val = np.nanmax(condensed_dist[np.isfinite(condensed_dist)])
        condensed_dist[np.isinf(condensed_dist)] = max_val + 1 # Replace inf
        condensed_dist[np.isnan(condensed_dist)] = max_val + 1 # Replace nan
        if not np.all(np.isfinite(condensed_dist)): # If still bad (e.g., all inf/nan)
             raise ValueError("Distance matrix contains only non-finite values.")

    # Perform average linkage (UPGMA)
    linked = linkage(condensed_dist, method="average")
    # visualize dendrogram
    visualize_linkage(linked, dist_name)
    # Form flat clusters based on the true number of classes
    cluster_labels = fcluster(
        linked, t=num_classes, criterion="maxclust"
    )
    # Adjust labels to be 0-indexed if they start from 1
    if np.min(cluster_labels) == 1:
        cluster_labels = cluster_labels - 1
    print(
        f"Clustering complete. Found {len(np.unique(cluster_labels))} clusters."
    )
    return cluster_labels

def evaluate_clustering(distance_matrix, cluster_labels, y_true):
    """Helper to evaluate clustering results."""
    eval_results = {}
    try:
        # Internal Validation
        intra_dist, inter_dist = calculate_cluster_distances(
            distance_matrix, cluster_labels
        )
        eval_results["intra_cluster_dist"] = intra_dist
        eval_results["inter_cluster_dist"] = inter_dist

        # External Validation
        purity = calculate_purity(y_true, cluster_labels)
        entropy = calculate_entropy(y_true, cluster_labels)
        eval_results["purity"] = purity
        eval_results["entropy"] = entropy
        
        # ARI vs True Labels
        ari_vs_true = adjusted_rand_score(y_true, cluster_labels)
        eval_results["ari_vs_true"] = ari_vs_true

        print(f"  ARI vs True Labels: {ari_vs_true:.4f}")
        print(f"  Purity: {purity:.4f}")
        print(f"  Entropy: {entropy:.4f}")
        print(f"  Avg Intra-cluster Distance: {intra_dist:.4f}")
        print(f"  Avg Inter-cluster Distance: {inter_dist:.4f}")

    except Exception as e:
        print(f"Error during evaluation: {e}.")
        
    return eval_results

if __name__ == "__main__":
    args = parse_args()

    # Load the selected dataset
    dataset = load_dataset(
        args.dataset, validation_size=args.val_size, random_state=args.seed
    )

    # Run the selected mode
    if args.mode == "single_tree":
        main_single_tree(dataset, args.relaxed_stopping)
    elif args.mode == "forest":
        main_forest(dataset, args.n_estimators, args.voting, args.seed)
    elif args.mode == "conformal":
        main_conformal(
            dataset,
            args.n_estimators,
            args.voting,
            args.seed,
            args.cal_ratio,
            args.epsilon,
        )
    elif args.mode == "unsupervised":
        main_unsupervised(dataset, args.n_estimators, args.seed)
    else:
        print(f"Error: Unknown mode '{args.mode}'")
        sys.exit(1)
