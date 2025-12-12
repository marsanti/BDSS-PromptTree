import sys
import numpy as np
import typing

if typing.TYPE_CHECKING:
    from core.forest import RandomForest


def delta_l2(a, b):
    """
    L2 distance implementation: Euclidean Distance.

    Parameters
    ----------
        a : The point a
        b : The point b

    Returns
    ----------
        distance
            the euclidean distance for the given points.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    # Handle unequal lengths by truncating to the shorter length
    if len(a) != len(b):
        m = min(len(a), len(b))
        a, b = a[:m], b[:m]
    return float(np.linalg.norm(a - b))


def compute_breiman_distance_matrix(leaf_assignments: np.ndarray) -> np.ndarray:
    """
    Calculates the Breiman distance matrix based on leaf node assignments.

    Distance(i, j) = 1 - Proximity(i, j)
    Proximity(i, j) = Fraction of trees where sample i and sample j fall into the same leaf node.

    Args:
        leaf_assignments (np.ndarray): Array of shape (n_samples, n_estimators)
                                       where element [s, t] is the leaf ID for sample s in tree t.
                                       Assumes leaf IDs are non-negative integers.
                                       Negative IDs (e.g., -1 for 'too-early') are ignored.

    Returns:
        np.ndarray: A symmetric distance matrix of shape (n_samples, n_samples)
                    where element [i, j] is the Breiman distance between sample i and j.
    """
    n_samples, n_estimators = leaf_assignments.shape

    if n_estimators == 0:
        # Handle case with no trees, return matrix of 1s (max distance)
        dist_matrix = np.ones((n_samples, n_samples), dtype=float)
        np.fill_diagonal(dist_matrix, 0)  # Distance to self is 0
        return dist_matrix

    # Initialize counts matrix
    proximity_counts = np.zeros((n_samples, n_samples), dtype=np.float64)
    for t in range(n_estimators):
        # Get leaf assignments for this tree
        tree_leaves = leaf_assignments[:, t]

        # Create masks for valid leaves (ignore negative IDs like -1 for 'too-early')
        valid_mask = tree_leaves >= 0
        valid_leaves = tree_leaves[valid_mask]
        valid_indices = np.where(valid_mask)[0]

        # Need at least two samples in valid leaves to form pairs
        if len(valid_indices) < 2:
            continue

        # Use broadcasting to find pairs in the same leaf *among valid samples*
        # `valid_leaves[:, None] == valid_leaves` creates a boolean matrix for valid samples
        same_leaf_matrix_tree_valid = valid_leaves[:, None] == valid_leaves

        # Use advanced indexing to update the main proximity_counts matrix
        # Create an index grid corresponding to the valid samples
        row_indices, col_indices = np.meshgrid(
            valid_indices, valid_indices, indexing="ij"
        )

        # Increment counts where same_leaf_matrix_tree_valid is True
        proximity_counts[
            row_indices[same_leaf_matrix_tree_valid],
            col_indices[same_leaf_matrix_tree_valid],
        ] += 1

    # Normalize counts to get proximity
    # Need to handle the diagonal separately as it wasn't fully counted in Method 2 loop
    # Sample always in same leaf as itself
    np.fill_diagonal(proximity_counts, n_estimators)
    proximity_matrix = proximity_counts / n_estimators

    # Calculate the distance matrix
    distance_matrix = 1.0 - proximity_matrix

    # Ensure diagonal is exactly zero due to potential floating point inaccuracies
    np.fill_diagonal(distance_matrix, 0)

    # Ensure symmetry (should already be symmetric, but good practice)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2.0

    return distance_matrix


def compute_zhu_distance_matrix(forest: "RandomForest", X: np.ndarray) -> np.ndarray:
    """
    Calculates the Zhu distance matrix.

    Distance(i, j) = Average_over_trees[ max_depth(T) - depth_of_separation(i, j, T) ]

    Args:
        forest (RandomForest): The fitted unsupervised RandomForest (Isolation Forest).
                               Must have 'trees' and 'tree_max_depths' attributes.
        X (np.ndarray): The data samples (shape [n_samples, n_channels, n_timesteps])
                        used to calculate pairwise distances.

    Returns:
        np.ndarray: A symmetric distance matrix of shape (n_samples, n_samples).
    """
    n_samples = len(X)
    n_estimators = forest.n_estimators

    if n_estimators == 0:
        dist_matrix = np.zeros((n_samples, n_samples), dtype=float)
        return dist_matrix 

    # Let's process tree by tree to save memory
    sum_of_depth_diffs = np.zeros((n_samples, n_samples), dtype=float)

    print("Calculating Zhu distances (tree by tree)...")
    for t in range(n_estimators):
        if t % 10 == 0:
            print(f"  Processing tree {t+1}/{n_estimators}...")

        tree = forest.trees[t]
        max_depth_t = forest.tree_max_depths[t]

        # Get paths for all samples *for this tree*
        paths_t = [tree.get_path_nodes(x_sample) for x_sample in X]

        # Iterate through pairs of samples
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                path_i = paths_t[i]
                path_j = paths_t[j]

                # Find depth of separation d(i, j, T)
                depth_of_separation = 0
                # Check if paths are valid (not empty, potentially handle 'too-early' if needed)
                if path_i and path_j:
                    # Find the length of the common prefix path
                    min_len = min(len(path_i), len(path_j))
                    for k in range(min_len):
                        if path_i[k] == path_j[k]:
                            # The depth of this common node IS k
                            depth_of_separation = k
                        else:
                            break  # Diverged at previous depth
                else:
                    # Handle invalid paths (e.g., sample too short for root)
                    # Contribution should be maximal distance: max_depth - 0
                    depth_of_separation = -1  # Or 0, results in max_depth contribution

                # Let's refine d: Depth of last common ancestor node.
                common_depth = -1
                if path_i and path_j:
                    min_len = min(len(path_i), len(path_j))
                    for k in range(min_len):
                        if path_i[k] == path_j[k]:
                            common_depth = path_i[k].depth  # Use stored depth
                        else:
                            break

                # If common_depth is -1, they didn't even share the root (error?) or one path was empty.
                # Treat as separated at depth 0. Common node depth is 0.
                d_ij_t = max(0, common_depth)  # Ensure non-negative depth

                # Contribution to distance sum
                dist_contribution = max_depth_t - d_ij_t

                sum_of_depth_diffs[i, j] += dist_contribution
                sum_of_depth_diffs[j, i] += dist_contribution

    print("Averaging distances...")
    # Average over the number of estimators
    distance_matrix = sum_of_depth_diffs / n_estimators

    # Ensure diagonal is zero
    np.fill_diagonal(distance_matrix, 0)

    return distance_matrix


def compute_ratioRF_distance_matrix(leaf_assignments: np.ndarray) -> np.ndarray:
    """
    Calculates a ratio-based distance matrix (interpretation of RatioRF).

    Distance(i, j) = Count_Apart(i, j) / Count_Together(i, j)
    Where:
        Count_Together(i, j) = Number of trees where i and j are in the same leaf.
        Count_Apart(i, j) = N_estimators - Count_Together(i, j).

    Args:
        leaf_assignments (np.ndarray): Array of shape (n_samples, n_estimators)
                                       where element [s, t] is the leaf ID for sample s in tree t.
                                       Assumes leaf IDs are non-negative integers.
                                       Negative IDs (e.g., -1 for 'too-early') are ignored.

    Returns:
        np.ndarray: A symmetric distance matrix of shape (n_samples, n_samples).
                    Infinite distances (division by zero) are replaced by a large float value.
    """
    n_samples, n_estimators = leaf_assignments.shape

    if n_estimators == 0:
        # No trees, define distance? Let's return 0 matrix, implies max similarity? Or max distance?
        # Let's return max distance (like infinity simulation)
        dist_matrix = np.full((n_samples, n_samples), sys.float_info.max, dtype=float)
        np.fill_diagonal(dist_matrix, 0)
        return dist_matrix

    # Calculate Proximity Counts (same as in Breiman distance)
    proximity_counts = np.zeros((n_samples, n_samples), dtype=np.float64)
    for t in range(n_estimators):
        tree_leaves = leaf_assignments[:, t]
        valid_mask = tree_leaves >= 0
        valid_leaves = tree_leaves[valid_mask]
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) < 2:
            continue

        same_leaf_matrix_tree_valid = valid_leaves[:, None] == valid_leaves
        row_indices, col_indices = np.meshgrid(
            valid_indices, valid_indices, indexing="ij"
        )
        proximity_counts[
            row_indices[same_leaf_matrix_tree_valid],
            col_indices[same_leaf_matrix_tree_valid],
        ] += 1
    # End Proximity Counts Calculation

    # Ensure diagonal counts are correct (N_estimators)
    np.fill_diagonal(proximity_counts, n_estimators)

    # Calculate Counts Apart
    counts_apart = n_estimators - proximity_counts

    # Calculate the Ratio Distance
    # Use np.errstate to suppress warnings about division by zero temporarily
    with np.errstate(divide="ignore", invalid="ignore"):
        distance_matrix = counts_apart / proximity_counts

    # Handle Division by Zero (where proximity_counts was 0) -> Max distance
    # Find the maximum finite distance in the matrix
    finite_mask = np.isfinite(distance_matrix)
    max_finite_dist = np.max(distance_matrix[finite_mask]) if np.any(finite_mask) else 0.0

    # Define a replacement value that is large but not infinite
    # Use 2x the max finite distance, or n_estimators as a fallback if max_finite_dist is 0
    replacement_val = (max_finite_dist * 2) + 1.0 # Add 1 to ensure it's strictly larger
    if replacement_val <= 1.0: # Handle case where max_finite_dist was 0
        replacement_val = float(n_estimators * 2) # Use a large number relative to tree count

    # Replace all non-finite values (inf, nan)
    distance_matrix[~finite_mask] = replacement_val
    # Ensure distance is non-negative
    distance_matrix[distance_matrix < 0] = 0

    # Ensure diagonal is exactly zero
    np.fill_diagonal(distance_matrix, 0)

    # Ensure symmetry (should be symmetric, but enforce)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2.0

    return distance_matrix


DELTA_SET = {"l2": delta_l2}
