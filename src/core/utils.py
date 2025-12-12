import math
import numpy as np
from collections import Counter, defaultdict
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from .tree import PromptTree
    from .forest import RandomForest


def class_distribution(Y):
    cnt = Counter(Y)
    n = sum(cnt.values())
    return {k: cnt[k] / n for k in cnt}


def gini(Y):
    p = np.array(list(class_distribution(Y).values()))
    return 1.0 - np.sum(p**2)


def split_by_test(X, Y, test):
    X_t, Y_t, X_f, Y_f = [], [], [], []
    c, x_ref, b, e, delta_func, eps = test
    if Y is None:
        Y_iterator = [None] * len(X) # Create a dummy iterator of Nones
        Y_t = None
        Y_f = None
    else:
        Y_iterator = Y
    for i, (xi, yi) in enumerate(zip(X, Y_iterator)):
        if e > len(
            xi[c]
        ):  # need more observations -> send to "false" by design or treat as too-early
            X_f.append(xi)
            if Y is not None:
                Y_f.append(yi)
            continue
        d = delta_func(xi[c][b:e], x_ref)
        if d <= eps:
            X_t.append(xi)
            if Y is not None:
                Y_t.append(yi)
        else:
            X_f.append(xi)
            if Y is not None:
                Y_f.append(yi)
    return (X_t, Y_t), (X_f, Y_f)


def evaluate_tree_accuracy(
    tree: "PromptTree", X_test: np.ndarray, Y_test: np.ndarray
) -> float:
    """
    Calculates the accuracy of a PromptTree on a test dataset.
    """
    correct_predictions = 0
    total_samples = len(Y_test)

    if total_samples == 0:
        return 0.0

    for x_sample, y_true in zip(X_test, Y_test):
        # predict_one returns the leaf node's distribution (dict) or "too-early"
        prediction_dist = tree.predict_one(x_sample)

        if prediction_dist == "too-early":
            # 'too-early' is considered an incorrect prediction
            continue

        if prediction_dist is not None:
            # Find the class with the highest probability in the distribution
            predicted_class = max(prediction_dist, key=prediction_dist.get)

            if predicted_class == y_true:
                correct_predictions += 1
        # else: prediction_dist is None (e.g., empty leaf), which is incorrect

    return correct_predictions / total_samples


def evaluate_forest_accuracy(
    forest: "RandomForest",
    X_test: np.ndarray,
    Y_test: np.ndarray,
    voting: Literal["weighted", "majority", "track-record"] = "weighted",
) -> float:
    """
    Calculates the accuracy of a RandomForest on a test dataset.

    Args:
        forest (RandomForest): The trained Random Forest model.
        X_test (np.ndarray): The test time series data.
        Y_test (np.ndarray): The true labels for the test data.
        voting (str): The voting mechanism to use for prediction ('majority', 'weighted', 'track-record').

    Returns:
        float: The accuracy of the forest on the test set.
    """
    if len(Y_test) == 0:
        return 0.0

    try:
        # Use the forest's predict method
        predictions = forest.predict(X_test, voting=voting)

        # Handle potential None predictions if predict handles errors that way
        valid_predictions_mask = predictions is not None
        if not np.all(valid_predictions_mask):
            print(
                f"Warning: {np.sum(~valid_predictions_mask)} samples had invalid predictions."
            )
            # Only compare where predictions are valid
            correct = np.sum(
                predictions[valid_predictions_mask] == Y_test[valid_predictions_mask]
            )
            total_valid = np.sum(valid_predictions_mask)
            accuracy = correct / total_valid if total_valid > 0 else 0.0
        else:
            # Standard comparison if all predictions are valid
            correct = np.sum(predictions == Y_test)
            accuracy = correct / len(Y_test)

        return accuracy

    except RuntimeError as e:
        print(f"Error during forest evaluation: {e}. Forest might not be fitted.")
        return 0.0
    except Exception as e:
        print(f"An unexpected error occurred during forest evaluation: {e}")
        return 0.0


def calculate_nonconformity_scores(
    model: "RandomForest",
    X_cal: np.ndarray,
    Y_cal: np.ndarray,
    voting: Literal["weighted", "majority", "track-record"] = "weighted",
) -> np.ndarray:
    """
    Calculates nonconformity scores (1 - P(y_true)) for calibration data using a fitted RandomForest.

    Args:
        model (RandomForest): The fitted RandomForest model. Must have predict_proba and classes_ attributes.
        X_cal (np.ndarray): Calibration data features.
        Y_cal (np.ndarray): Calibration data true labels.
        voting (str): Voting mechanism used by predict_proba.

    Returns:
        np.ndarray: Array of nonconformity scores for the calibration set.
    """
    print("Calculating nonconformity scores...")
    if not hasattr(model, "classes_") or model.classes_ is None:
        raise RuntimeError(
            "Model is missing 'classes_' attribute needed for score calculation. Ensure it was fitted in supervised mode."
        )

    try:
        # Get probabilities for the calibration set
        probabilities = model.predict_probabilities(
            X_cal, voting=voting
        )  # Shape (n_cal_samples, n_classes)

        scores = []
        class_list = list(model.classes_)  # Convert to list for efficient index lookup

        for i in range(len(X_cal)):
            true_label = Y_cal[i]
            try:
                # Find the index corresponding to the true label
                true_label_idx = class_list.index(true_label)
                # Ensure probabilities array has the expected shape
                if probabilities.shape[1] > true_label_idx:
                    prob_true_label = probabilities[i, true_label_idx]
                    score = (
                        1.0 - prob_true_label
                    )  # Nonconformity score: Higher score = less conformity
                else:
                    print(
                        f"Warning: Probability array shape mismatch. Cannot get score for label index {true_label_idx}. Assigning max nonconformity (1.0)."
                    )
                    score = 1.0
            except ValueError:
                print(
                    f"Warning: Calibration label {true_label} not found in model classes {model.classes_}. Assigning max nonconformity (1.0)."
                )
                score = 1.0

            scores.append(score)

        print(f"Calculated {len(scores)} nonconformity scores.")
        return np.array(scores)

    except Exception as e:
        print(f"Error calculating nonconformity scores: {e}")
        return np.array([])  # Return empty if error occurs


def get_prediction_sets(
    model: "RandomForest",
    X_test: np.ndarray,
    nonconformity_scores: np.ndarray,
    significance_level: float,
    voting: Literal["weighted", "majority", "track-record"] = "weighted",
) -> list:
    """
    Generates prediction sets for test data based on calibration scores.

    Args:
        model (RandomForest): The fitted RandomForest model.
        X_test (np.ndarray): Test data features.
        nonconformity_scores (np.ndarray): Scores calculated from the calibration set.
        significance_level (float): The desired significance level (epsilon or alpha), e.g., 0.1 for 90% confidence.
        voting (str): Voting mechanism used by predict_probabilities.

    Returns:
        list: A list where each element is a sorted list representing the prediction set for a test sample.
    """
    print(
        f"Generating prediction sets with significance level (epsilon): {significance_level}..."
    )

    if not hasattr(model, "classes_") or model.classes_ is None:
        raise RuntimeError(
            "Model is missing 'classes_' attribute needed for prediction sets."
        )
    if len(nonconformity_scores) == 0:
        print(
            "Warning: Received empty nonconformity scores. Cannot generate prediction sets."
        )
        return [[] for _ in range(len(X_test))]

    try:
        probabilities = model.predict_probabilities(
            X_test, voting=voting
        )  # Shape (n_test_samples, n_classes)

        # Calculate the threshold (quantile) based on calibration scores
        N = len(nonconformity_scores)
        quantile_level = np.ceil((1.0 - significance_level) * (N + 1)) / (N + 1)
        if quantile_level > 1.0:
            quantile_level = 1.0  # Ensure it's <= 1
        if quantile_level <= 0.0:
            quantile_level = 1 / (N + 1)  # Ensure it's > 0

        # Use np.quantile for potentially more stable calculation
        score_threshold = np.quantile(
            nonconformity_scores, quantile_level, method="higher"
        )  # 'higher' matches ceil((1-a)(N+1))
        # Add small epsilon for numerical stability if needed, though quantile should handle it
        # score_threshold += 1e-9

        print(
            f"  Score threshold (q-hat): {score_threshold:.4f} (based on {N} calibration scores)"
        )

        prediction_sets = []
        for i in range(len(X_test)):
            sample_set = []
            # Calculate hypothetical nonconformity for each possible class
            for class_idx, label in enumerate(model.classes_):
                if probabilities.shape[1] > class_idx:
                    prob_label = probabilities[i, class_idx]
                    hypothetical_score = 1.0 - prob_label
                    # Include class in set if its hypothetical score is <= threshold
                    if hypothetical_score <= score_threshold:
                        sample_set.append(label)
                else:
                    print(
                        f"Warning: Probability array shape mismatch. Cannot evaluate class index {class_idx} for sample {i}."
                    )

            # Handle empty sets (optional but recommended): force include the most likely class
            if not sample_set and probabilities.shape[1] > 0:
                # print(f"Warning: Sample {i} resulted in an empty prediction set. Forcing inclusion of most likely class.")
                most_likely_idx = np.argmax(probabilities[i])
                sample_set.append(model.classes_[most_likely_idx])

            prediction_sets.append(
                sorted(list(set(sample_set)))
            )  # Ensure unique and sorted

        print(f"Generated {len(prediction_sets)} prediction sets.")
        return prediction_sets

    except Exception as e:
        print(f"Error generating prediction sets: {e}")
        return [[] for _ in range(len(X_test))]  # Return empty sets if error

def calculate_cluster_distances(distance_matrix: np.ndarray, cluster_labels: np.ndarray):
    """
    Calculates average intra-cluster and inter-cluster distances.

    Args:
        distance_matrix (np.ndarray): Symmetric matrix (n_samples, n_samples) of distances between samples.
        cluster_labels (np.ndarray): Array (n_samples,) assigning each sample to a cluster ID (0-indexed).

    Returns:
        tuple: (average_intra_cluster_distance, average_inter_cluster_distance)
               Returns (0.0, 0.0) if calculation is not possible (e.g., single cluster).
    """
    n_samples = distance_matrix.shape[0]
    if n_samples == 0 or len(cluster_labels) != n_samples:
        return 0.0, 0.0

    unique_clusters = np.unique(cluster_labels)
    num_clusters = len(unique_clusters)

    if num_clusters <= 0:
        return 0.0, 0.0

    # Intra-cluster distance
    total_intra_dist = 0.0
    intra_pairs_count = 0
    for cluster_id in unique_clusters:
        indices_in_cluster = np.where(cluster_labels == cluster_id)[0]
        n_in_cluster = len(indices_in_cluster)
        if n_in_cluster > 1:
            # Extract sub-matrix for the cluster
            cluster_dist_matrix = distance_matrix[np.ix_(indices_in_cluster, indices_in_cluster)]
            # Sum upper triangle (excluding diagonal) to count each pair once
            # Note: np.sum gives sum of all elements, divide by 2 later if needed,
            # but counting pairs is more direct.
            cluster_sum = np.sum(np.triu(cluster_dist_matrix, k=1))
            num_pairs_in_cluster = n_in_cluster * (n_in_cluster - 1) // 2
            total_intra_dist += cluster_sum
            intra_pairs_count += num_pairs_in_cluster

    avg_intra_dist = total_intra_dist / intra_pairs_count if intra_pairs_count > 0 else 0.0

    # Inter-cluster distance
    if num_clusters <= 1:
        # Cannot calculate inter-cluster distance with only one cluster
        avg_inter_dist = 0.0
    else:
        total_inter_dist = 0.0
        inter_pairs_count = 0
        # Iterate through all unique pairs of clusters
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                cluster_id_1 = unique_clusters[i]
                cluster_id_2 = unique_clusters[j]
                indices_1 = np.where(cluster_labels == cluster_id_1)[0]
                indices_2 = np.where(cluster_labels == cluster_id_2)[0]

                # Extract the block of the distance matrix between the two clusters
                inter_cluster_dists = distance_matrix[np.ix_(indices_1, indices_2)]
                cluster_pair_sum = np.sum(inter_cluster_dists)
                num_pairs_between = len(indices_1) * len(indices_2)

                total_inter_dist += cluster_pair_sum
                inter_pairs_count += num_pairs_between

        avg_inter_dist = total_inter_dist / inter_pairs_count if inter_pairs_count > 0 else 0.0

    return avg_intra_dist, avg_inter_dist


def calculate_purity(y_true: np.ndarray, y_pred_clusters: np.ndarray) -> float:
    """
    Calculates the Purity clustering metric.

    Args:
        y_true (np.ndarray): Array (n_samples,) of true class labels.
        y_pred_clusters (np.ndarray): Array (n_samples,) of predicted cluster assignments.

    Returns:
        float: The purity score (between 0 and 1).
    """
    n_samples = len(y_true)
    if n_samples == 0 or len(y_pred_clusters) != n_samples:
        return 0.0

    # Group true labels by predicted cluster
    clusters = defaultdict(list)
    for i in range(n_samples):
        clusters[y_pred_clusters[i]].append(y_true[i])

    total_correct = 0
    for cluster_id in clusters:
        labels_in_cluster = clusters[cluster_id]
        if labels_in_cluster:
            # Find the most frequent true label in this cluster
            most_common_label, count = Counter(labels_in_cluster).most_common(1)[0]
            total_correct += count

    purity = total_correct / n_samples if n_samples > 0 else 0.0
    return purity


def calculate_entropy(y_true: np.ndarray, y_pred_clusters: np.ndarray) -> float:
    """
    Calculates the Entropy clustering metric (weighted average entropy per cluster).
    Lower entropy indicates better clustering.

    Args:
        y_true (np.ndarray): Array (n_samples,) of true class labels.
        y_pred_clusters (np.ndarray): Array (n_samples,) of predicted cluster assignments.

    Returns:
        float: The entropy score (>= 0).
    """
    n_samples = len(y_true)
    if n_samples == 0 or len(y_pred_clusters) != n_samples:
        return 0.0

    # Group true labels by predicted cluster
    clusters = defaultdict(list)
    for i in range(n_samples):
        clusters[y_pred_clusters[i]].append(y_true[i])

    total_entropy = 0.0
    unique_true_labels = np.unique(y_true)
    num_true_classes = len(unique_true_labels)

    for cluster_id in clusters:
        labels_in_cluster = clusters[cluster_id]
        n_cluster = len(labels_in_cluster)
        if n_cluster == 0:
            continue

        label_counts = Counter(labels_in_cluster)
        cluster_entropy = 0.0
        for label in label_counts:
            probability = label_counts[label] / n_cluster
            if probability > 0:
                cluster_entropy -= probability * math.log2(probability)

        # Weight the cluster's entropy by its size relative to the total dataset size
        total_entropy += (n_cluster / n_samples) * cluster_entropy

    return total_entropy
