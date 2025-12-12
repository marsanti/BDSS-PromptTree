import numpy as np
from collections import Counter
from typing import Literal, List
import random

from .tree import PromptTree
from .tree_functions import (
    DELTA_SET,
    fo_unsupervised_random,
    fe_unsupervised_default,
    fe_default,
    fo_default,
)
from .utils import evaluate_tree_accuracy

def c_factor(n):
    """
    Average path length of unsuccessful search in BST (Normalization factor).
    n: number of external nodes (subsample size).
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    # Euler's constant approximation
    euler_gamma = 0.5772156649
    return 2.0 * (np.log(n - 1) + euler_gamma) - (2.0 * (n - 1) / n)

class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        mode: Literal["supervised", "unsupervised"] = "supervised",
        fp=None,
        fs=None,
        fc=None,
        fo=None,
        fe=None,
        fo_unsupervised=None,
        fe_unsupervised=None,
        bootstrap: bool = True,
        max_samples: float = 1.0,
        random_state: int = None,
    ):
        """
        Initializes the Random Forest.

        Args:
            n_estimators (int): Number of trees in the forest.
            mode (str): 'supervised' for classification, 'unsupervised' for isolation. [cite: 130]
            fp, fs, fc, fo, fe: Functions to customize PromptTree behavior.
                                If None, PromptTree defaults will be used.
            bootstrap (bool): Whether to use bootstrap samples when building trees.
            max_samples (float): If bootstrap is True, fraction of samples to draw for each tree.
            random_state (int): Controls randomness for bootstrapping and tree building.
        """
        self.n_estimators = n_estimators
        if mode not in ["supervised", "unsupervised"]:
            raise ValueError("mode must be 'supervised' or 'unsupervised'")
        self.mode = mode
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.random_state = random_state

        self.fo_unsupervised = (
            fo_unsupervised if fo_unsupervised is not None else fo_unsupervised_random
        )
        self.fe_unsupervised = (
            fe_unsupervised if fe_unsupervised is not None else fe_unsupervised_default
        )

        # Store tree customization functions
        self.tree_kwargs = {"fp": fp, "fs": fs, "fc": fc, "fo": fo, "fe": fe}

        self.trees: List[PromptTree] = []
        # For track-record voting
        self.oob_scores = None
        self.tree_max_depths = []

    def fit(self, X, Y=None):
        """
        Builds the forest of PromptTrees from the training set (X, Y).

        Args:
            X (list or np.array): Training time series data.
                                  Shape (n_samples, n_channels, n_timesteps).
            Y (list or np.array): Target values (class labels). Required for supervised mode.
                                  Shape (n_samples,).
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)

        if self.mode == "supervised":
            if Y is None:
                raise ValueError("Y cannot be None in supervised mode.")
            # Determine and store classes
            self.classes_ = np.unique(Y)
            self.n_classes_ = len(self.classes_)
            print(f"Detected {self.n_classes_} classes: {self.classes_}")
            tree_fe = self.tree_kwargs.get("fe", fe_default)
            tree_fo = self.tree_kwargs.get("fo", fo_default)
            is_unsup = False
        else:
            self.classes_ = None
            self.n_classes_ = None
            tree_fe = self.fe_unsupervised
            tree_fo = self.fo_unsupervised
            is_unsup = True
            Y = None

        n_samples = len(X)
        if Y is not None and len(Y) != n_samples:
            raise ValueError("X and Y must have the same number of samples.")

        self.oob_scores = np.zeros(self.n_estimators)
        oob_predictions = [[] for _ in range(n_samples)]
        oob_true_labels = [[] for _ in range(n_samples)]

        self.trees = []
        self.tree_max_depths = []
        for i in range(self.n_estimators):
            print(f"Fitting tree {i+1}/{self.n_estimators}...")

            # Bootstrapping
            if self.bootstrap:
                n_bootstrap_samples = int(self.max_samples * n_samples)
                indices = np.random.choice(
                    n_samples, size=n_bootstrap_samples, replace=True
                )
                X_sample = [X[j] for j in indices]
                Y_sample = [Y[j] for j in indices] if Y is not None else None

                # Identify OOB samples
                oob_indices = np.setdiff1d(
                    np.arange(n_samples), indices, assume_unique=False
                )
            else:
                X_sample = X
                Y_sample = Y
                oob_indices = []

            # Tree Initialization & Randomness
            current_tree_kwargs = self.tree_kwargs.copy()
            current_tree_kwargs["fe"] = tree_fe
            current_tree_kwargs["fo"] = tree_fo
            current_tree_kwargs["fs_random_state"] = self.random_state + i

            tree = PromptTree(**current_tree_kwargs)

            # Train Tree
            tree.fit(X_sample, Y_sample, is_unsupervised=is_unsup)
            self.trees.append(tree)
            self.tree_max_depths.append(tree.get_max_depth())

            # Calculate OOB Score (if supervised and bootstrapping)
            if self.bootstrap and len(oob_indices) > 0 and self.mode == "supervised":
                X_oob = [X[j] for j in oob_indices]
                Y_oob = [Y[j] for j in oob_indices]
                oob_acc = evaluate_tree_accuracy(tree, X_oob, Y_oob)
                self.oob_scores[i] = oob_acc
        print("Forest fitting complete.")
        return self

    def get_leaf_assignments(self, X):
        """Gets the leaf node ID for each sample in each tree."""
        if not self.trees:
            raise RuntimeError("Forest is not fitted yet.")
        if self.mode != "unsupervised":
            print("Warning: Calling get_leaf_assignments on a supervised forest.")

        n_samples = len(X)
        leaf_assignments = np.zeros((n_samples, self.n_estimators), dtype=int)

        for i, tree in enumerate(self.trees):
            for j, x_sample in enumerate(X):
                leaf_assignments[j, i] = tree.get_leaf_id(x_sample)

        return leaf_assignments

    def predict_probabilities(
        self, X, voting: Literal["weighted", "majority", "track-record"] = "weighted"
    ):
        """
        Predict class probabilities for X. (Supervised mode only)
        Uses Weighted Voting by default (averaging probabilities).
        """
        if self.mode != "supervised":
            raise AttributeError(
                "predict_probabilities is only available in supervised mode."
            )
        if not self.trees:
            raise RuntimeError("Forest is not fitted yet. Call fit() first.")

        # Collect probability distributions from each tree
        all_probabilities = []
        for x_sample in X:
            sample_probabilities = []
            for i, tree in enumerate(self.trees):
                dist = tree.predict_one(x_sample)
                if dist == "too-early" or dist is None:
                    # Handle too-early
                    # A more robust approach might be needed
                    sample_probabilities.append(None)
                else:
                    # Check if it's a valid distribution dict
                    if isinstance(dist, dict):
                        sample_probabilities.append(dist)
                    else:
                        sample_probabilities.append(None)
            # List of lists of distributions
            all_probabilities.append(sample_probabilities)

        # Aggregate probabilities based on voting scheme
        final_probabilities = []
        if not hasattr(self, "classes_") or self.classes_ is None:
            raise RuntimeError("Forest not fitted or missing class information.")

        n_classes = self.n_classes_
        class_labels = self.classes_

        # Aggregation Logic
        for sample_idx, tree_dists in enumerate(all_probabilities):
            aggregated_votes = np.zeros(n_classes)
            valid_votes = 0

            if voting == "weighted":
                # Average probabilities
                for dist in tree_dists:
                    if dist is not None:
                        valid_votes += 1
                        for class_idx, label in enumerate(class_labels):
                            aggregated_votes[class_idx] += dist.get(label, 0.0)
                if valid_votes > 0:
                    aggregated_votes /= valid_votes

            elif voting == "majority":
                # Majority vote on predicted class
                tree_preds = []
                for dist in tree_dists:
                    if dist is not None:
                        predicted_class = max(dist, key=dist.get)
                        tree_preds.append(predicted_class)
                if tree_preds:
                    vote_counts = Counter(tree_preds)
                    most_common_label = vote_counts.most_common(1)[0][0]
                    # Convert back to probability (1.0 for predicted, 0.0 for others)
                    for class_idx, label in enumerate(class_labels):
                        if label == most_common_label:
                            aggregated_votes[class_idx] = 1.0

            elif voting == "track-record":
                # Weighted average by OOB score
                if self.oob_scores is None:
                    raise ValueError(
                        "OOB scores not available. Fit with bootstrap=True."
                    )
                total_weight = 0
                for i, dist in enumerate(tree_dists):
                    if dist is not None:
                        # Use pre-calculated OOB score
                        weight = self.oob_scores[i]
                        valid_votes += 1
                        total_weight += weight
                        for class_idx, label in enumerate(class_labels):
                            aggregated_votes[class_idx] += dist.get(label, 0.0) * weight
                if total_weight > 0:
                    aggregated_votes /= total_weight

            else:
                raise ValueError(f"Unknown voting mechanism: {voting}")

            final_probabilities.append(aggregated_votes)

        return np.array(final_probabilities)

    def predict(
        self, X, voting: Literal["weighted", "majority", "track-record"] = "weighted"
    ):
        """
        Predict class labels for X. (Supervised mode only)
        """
        if self.mode != "supervised":
            raise AttributeError("predict is only available in supervised mode.")

        probabilities = self.predict_probabilities(X, voting=voting)

        # If classes_ were stored during fit, use them
        if not hasattr(self, "classes_") or self.classes_ is None:
            raise RuntimeError(
                "This RandomForest instance is not fitted yet or class information is missing. Call 'fit' with appropriate training data first."
            )

        # Find the index with the highest probability
        predicted_indices = np.argmax(probabilities, axis=1)
        predictions = [self.classes_[idx] for idx in predicted_indices]
        return np.array(predictions)

    # Methods for Unsupervised / Isolation Forest
    def get_path_lengths(self, X):
        """
        Calculates the path length (depth) for each sample in X for every tree.
        """
        if not self.trees:
            raise RuntimeError("Forest not fitted.")
        
        n_samples = len(X)
        path_lengths = np.zeros((n_samples, self.n_estimators))

        for i, tree in enumerate(self.trees):
            for j, x_sample in enumerate(X):
                # We use get_path_nodes (added for Zhu distance) to calculate length
                nodes = tree.get_path_nodes(x_sample)
                # Path length is number of edges, which is nodes - 1
                depth = len(nodes) - 1 if nodes else 0
                path_lengths[j, i] = depth
        
        return path_lengths

    def anomaly_score(self, X):
        """
        Calculates the standard Isolation Forest anomaly score.
        Score s(x, n) = 2^(-E(h(x)) / c(n))
        
        Returns:
            scores (np.array): Value between 0 and 1. 
                               Closer to 1 => Anomaly (short path).
                               Closer to 0.5 or 0 => Normal (long path).
        """
        # 1. Calculate Average Path Length E(h(x))
        all_path_lengths = self.get_path_lengths(X)
        avg_path_lengths = np.mean(all_path_lengths, axis=1)

        # 2. Determine subsample size 'n' used during training
        # Heuristic: Try to retrieve the count from the first tree's root
        n = 256 # Default fallback
        if self.trees and self.trees[0].root:
            # Check if root.dist is a dict with 'count' (unsupervised)
            if isinstance(self.trees[0].root.dist, dict):
                 n = self.trees[0].root.dist.get('count', 256)
            # Or if it's supervised, it might be the sum of class counts
            elif isinstance(self.trees[0].root.dist, dict):
                 n = sum(self.trees[0].root.dist.values())

        # 3. Calculate c(n) normalization factor
        c_n = c_factor(n)

        if c_n == 0:
            return np.ones(len(X)) * 0.5

        # 4. Calculate Score
        scores = 2 ** (-avg_path_lengths / c_n)
        return scores

    def get_leaf_assignments(self, X):
        """
        Find the leaf node ID each sample falls into for each tree.
        (Renamed from get_leaf_nodes to match main.py)
        """
        if not self.trees:
            raise RuntimeError("Forest not fitted.")

        leaf_indices = np.zeros((len(X), self.n_estimators), dtype=int)
        
        for i, tree in enumerate(self.trees):
            for j, x_sample in enumerate(X):
                # Retrieve the unique leaf ID assigned during tree building
                leaf_id = tree.get_leaf_id(x_sample)
                leaf_indices[j, i] = leaf_id
                
        return leaf_indices
