from .node import Node
from .tree_functions import (
    fp_default,
    fs_default,
    fc_default,
    fo_default,
    fe_default,
    fo_unsupervised_random,
    fe_unsupervised_default,
    DELTA_SET,
)
from .utils import split_by_test
from graphviz import Digraph


class PromptTree:
    def __init__(
        self,
        fp=None,
        fs=None,
        fc=None,
        fo=None,
        fe=None,
        fo_unsupervised=None,
        fe_unsupervised=None,
        fs_random_state=None,
    ):
        # Supervised defaults
        self.fp = fp if fp is not None else fp_default
        self.fs = fs if fs is not None else fs_default
        self.fc = fc if fc is not None else fc_default
        self.fo = fo if fo is not None else fo_default
        self.fe = fe if fe is not None else fe_default

        # Unsupervised defaults (assuming they are imported)
        self.fo_unsupervised = (
            fo_unsupervised if fo_unsupervised is not None else fo_unsupervised_random
        )
        self.fe_unsupervised = (
            fe_unsupervised if fe_unsupervised is not None else fe_unsupervised_default
        )

        self.root = None
        self.fs_random_state = fs_random_state 

    # Update fit to pass is_unsupervised and leaf_counter
    def fit(self, X, Y=None, is_unsupervised=False):
        """Fits the PromptTree to the data."""
        leaf_counter = [0]
        self.root = self._fit_recursive(
            X,
            Y,
            Path=[],
            depth=0,
            leaf_counter=leaf_counter,
            is_unsupervised=is_unsupervised,
        )
        print(f"Tree fitting complete. Created {leaf_counter[0]} leaves.")
        return self

    def _fit_recursive(self, X, Y, Path, depth, leaf_counter, is_unsupervised):
        """Recursive helper function to build the tree."""

        if is_unsupervised:
            current_fe = self.fe_unsupervised
            current_fo = self.fo_unsupervised
            current_fc = lambda x, y: {"count": len(x)}
            Y_for_split = None
        else:
            current_fe = self.fe
            current_fo = self.fo
            current_fc = self.fc
            Y_for_split = Y

        # Check stopping criterion
        if current_fe(Path, X, Y, depth):
            leaf_id = leaf_counter[0]
            leaf_counter[0] += 1
            node = Node(dist=current_fc(X, Y), depth=depth)
            node.leaf_id = leaf_id
            node.is_leaf = True

            return node

        # Prepare for split proposal
        if not Path:
            B = {0}
            E = set()
        else:
            try:
                B = {p["b"] for p in Path if isinstance(p, dict) and "b" in p}
                E = {p["e"] for p in Path if isinstance(p, dict) and "e" in p}
            except (TypeError, KeyError) as e:
                print(f"Error processing Path structure: {Path}. Error: {e}")
                leaf_id = leaf_counter[0]
                leaf_counter[0] += 1
                node = Node(dist=current_fc(X, Y))
                node.leaf_id = leaf_id
                node.is_leaf = True
                return node

        next_start = (max(E) + 1) if E else 0

        # Propose intervals and tests
        cand_intervals = self.fp(B, next_start, X, Y_for_split)
        if not cand_intervals:
            leaf_id = leaf_counter[0]
            leaf_counter[0] += 1
            node = Node(dist=current_fc(X, Y))
            node.leaf_id = leaf_id
            node.is_leaf = True
            return node

        # Pass random state if fs needs it
        cand_tests = self.fs(X, Y_for_split, cand_intervals, seed=self.fs_random_state)
        if not cand_tests:
            leaf_id = leaf_counter[0]
            leaf_counter[0] += 1
            node = Node(dist=current_fc(X, Y))
            node.leaf_id = leaf_id
            node.is_leaf = True
            return node

        # Select best test using appropriate optimization function
        best, gain = current_fo(X, Y_for_split, cand_tests)

        # Check if a valid split was found
        # For supervised, gain > 0 is often used. For unsupervised, just check if 'best' exists.
        make_leaf = best is None
        if not is_unsupervised and gain <= 0:  # Supervised check
            make_leaf = True

        if make_leaf:
            leaf_id = leaf_counter[0]
            leaf_counter[0] += 1
            node = Node(dist=current_fc(X, Y), depth=depth)
            node.leaf_id = leaf_id
            node.is_leaf = True
            return node

        # Create Internal Node and Split Data
        c, x_ref, eps, (b, e), delta_name = best
        if delta_name not in DELTA_SET:
            raise ValueError(
                f"Distance function '{delta_name}' not found in DELTA_SET."
            )
        delta_func = DELTA_SET[delta_name]

        node = Node(
            dist=None,
            c=c,
            x_ref=x_ref,
            b=b,
            e=e,
            delta_name=delta_name,
            eps=eps,
            depth=depth,
        )
        # Store pre-split distribution ONLY if supervised (useful for pruning)
        if not is_unsupervised:
            node.dist = self.fc(X, Y)  # Original self.fc for supervised distribution
        else:
            node.dist = current_fc(X, Y)  # Store count or empty dict

        # Use Y_for_split (None if unsupervised) for the split function
        (X_t, Y_t), (X_f, Y_f) = split_by_test(
            X, Y_for_split, (c, x_ref, b, e, delta_func, eps)
        )

        # Recurse
        new_step = dict(c=c, b=b, e=e, delta=delta_name, eps=eps)

        node.left = self._fit_recursive(
            X_t, Y_t, Path + [new_step], depth + 1, leaf_counter, is_unsupervised
        )
        node.right = self._fit_recursive(
            X_f, Y_f, Path + [new_step], depth + 1, leaf_counter, is_unsupervised
        )

        return node

    def get_leaf_id(self, x_sample):
        """Finds the leaf ID for a given sample."""
        node = self.root
        current_depth = 0
        while node is not None and not node.is_leaf:
            if node.e > x_sample[node.c].shape[-1]:
                return -1  # 'Too early' or cannot traverse further
            try:
                if node.delta_name not in DELTA_SET:
                    print(f"Error: Unknown distance '{node.delta_name}' in node.")
                    return -2
                delta_func = DELTA_SET[node.delta_name]
                d = delta_func(x_sample[node.c][node.b : node.e], node.x_ref)
                node = node.left if d <= node.eps else node.right
                current_depth += 1
            except IndexError:
                # This might happen if b/e are somehow invalid despite length check
                print(
                    f"Error: IndexError during distance calculation at depth {current_depth}. Slice [{node.b}:{node.e}], Channel {node.c}, Sample shape {x_sample.shape}"
                )
                return -2
            except Exception as e:
                print(f"Error during node traversal: {e}")
                return -2

        if node is not None and node.is_leaf:
            if hasattr(node, "leaf_id"):
                return node.leaf_id
            else:
                print("Error: Reached a leaf node without a 'leaf_id' attribute.")
                return -3
        elif node is None:
            print("Error: Traversal led to a None node.")
            return -4
        else:
            print("Error: Unknown state after traversal loop.")
            return -5

    def predict_one(self, x):
        """
        Parameters
        ----------
            x : np.ndarray
                its shape is (C, T_obs)

        Returns
        ----------
            label distribution or 'too-early'.
        """
        v = self.root
        while not v.is_leaf:
            if v.e > x[v.c].shape[-1]:
                return "too-early"
            delta_func = DELTA_SET[v.delta_name]
            d = delta_func(x[v.c][v.b : v.e], v.x_ref)
            v = v.left if d <= v.eps else v.right
        return v.dist  # distribution

    def get_paths(self):
        res = []

        def dfs(node, cur):
            if node is None:
                return
            if node.is_leaf:
                res.append((list(cur), node))
            else:
                step = (node.c, node.b, node.e, node.delta_name, node.eps)
                cur.append(step)
                dfs(node.left, cur)
                dfs(node.right, cur)
                cur.pop()

        dfs(self.root, [])
        return res

    def print_tree(self):
        paths = self.get_paths()
        for i, path in enumerate(paths):
            print(f"Leaf {i}:")
            for depth, (c, b, e, delta_name, eps) in enumerate(path):
                indent = "  " * depth
                print(
                    f"{indent}if channel {c}, interval [{b}:{e}], delta={delta_name}, eps={eps}"
                )
            print(f"{'  ' * len(path)}--> LEAF\n")

    def prune(self, X_val, Y_val):
        """
        Prunes the tree in-place using a validation dataset to prevent overfitting.
        This method uses the Reduced Error Pruning algorithm.
        """
        # Start the recursive pruning process from the root
        self._prune_recursive(self.root, X_val, Y_val)

    def _calculate_accuracy(self, node, X, Y):
        """
        Calculate accuracy of a subtree for a given dataset.
        """
        correct = 0
        for x_sample, y_sample in zip(X, Y):
            # Predict using the subtree starting from 'node'
            current_node = node
            while not current_node.is_leaf:
                # Handle promptness: if data is too short, prediction is wrong by default
                if current_node.e > x_sample[current_node.c].shape[-1]:
                    break

                delta_func = DELTA_SET[current_node.delta_name]
                d = delta_func(
                    x_sample[current_node.c][current_node.b : current_node.e],
                    current_node.x_ref,
                )
                current_node = (
                    current_node.left if d <= current_node.eps else current_node.right
                )

            if current_node.is_leaf and current_node.predicted_class == y_sample:
                correct += 1

        return correct / len(Y) if len(Y) > 0 else 0.0

    def _prune_recursive(self, node, X_val, Y_val):
        # Base case: if no validation data, return
        if len(Y_val) == 0:
            return

        # If it's a leaf, we can't prune
        if node.is_leaf:
            return

        # Recurse down to the children first (Post-order traversal)
        # This ensures we prune from the bottom of the tree upwards
        (X_val_left, Y_val_left), (X_val_right, Y_val_right) = split_by_test(
            X_val,
            Y_val,
            (node.c, node.x_ref, node.b, node.e, DELTA_SET[node.delta_name], node.eps),
        )
        self._prune_recursive(node.left, X_val_left, Y_val_left)
        self._prune_recursive(node.right, X_val_right, Y_val_right)

        # To prune the current node : calculate accuracy of the full subtree at this node.
        accuracy_subtree = self._calculate_accuracy(node, X_val, Y_val)

        # Calculate accuracy IF we turned this node into a leaf.
        majority_class = max(node.dist, key=node.dist.get)
        accuracy_as_leaf = sum(1 for y in Y_val if y == majority_class) / len(Y_val)

        # 5. The Pruning Decision
        if accuracy_as_leaf >= accuracy_subtree:
            print(
                f"Pruning node at slice [{node.b}:{node.e}]. Accuracy change: {accuracy_subtree:.3f} -> {accuracy_as_leaf:.3f}"
            )
            node.is_leaf = True
            node.left = None
            node.right = None

    def visualize_tree(self, class_names: dict, filename="prompt_tree") -> None:
        """
        Generates a visualization of the decision tree and saves it as a PNG file.

        Parameters
        ----------
            class_names : dict
                A dictionary mapping class labels to their names.
            filename : str
                The name of the output file (without extension).
        """
        dot = Digraph(comment="Prompt Tree")
        dot.attr("node", shape="oval", style="filled", color="lightblue")
        dot.attr("edge", fontsize="10")
        dot.attr(rankdir="TB", splines="ortho")  # Top-to-Bottom layout

        node_counter = 0

        def get_unique_id():
            nonlocal node_counter
            node_counter += 1
            return f"node{node_counter}"

        def add_nodes_edges(node, parent_id=None, edge_label=""):
            if node is None:
                return

            current_id = get_unique_id()

            # Define node label based on whether it's a leaf or internal node
            if node.is_leaf:
                pred_class = node.predicted_class
                class_name = class_names.get(pred_class, "Unknown")
                # More detailed leaf info: show distribution
                dist_str = "\\n".join(
                    [f"{class_names.get(k, k)}: {v:.2f}" for k, v in node.dist.items()]
                )
                label = f"Leaf: {class_name}\\n({dist_str})"
                dot.node(current_id, label, shape="box", color="lightgreen")
            else:
                # Round epsilon for cleaner visualization
                eps_rounded = round(node.eps, 2)
                label = f"Channel: {node.c}\\nSlice: [{node.b}:{node.e}]\\nδ={node.delta_name}, ε≤{eps_rounded}"
                dot.node(current_id, label)

            # Add an edge from the parent to the current node
            if parent_id is not None:
                dot.edge(parent_id, current_id, xlabel=edge_label)

            # Recurse for children
            if not node.is_leaf:
                add_nodes_edges(node.left, current_id, "True")
                add_nodes_edges(node.right, current_id, "False")

        # Start the recursive process from the root
        add_nodes_edges(self.root)

        # Render and view the graph
        dot.render(filename, view=True, format="png", cleanup=True)
        print(f"Tree visualization saved to {filename}.png")


    def get_path_nodes(self, x_sample):
        """Returns the list of nodes traversed by a sample to reach a leaf (or failure point)."""
        path = []
        node = self.root
        while node is not None:
            path.append(node)
            if node.is_leaf:
                # Reached leaf
                break

            if node.e > x_sample[node.c].shape[-1]:
                # Sample is too short for this node's test
                break

            try:
                if node.delta_name not in DELTA_SET:
                    print(f"Error: Unknown distance '{node.delta_name}' in node.")
                    return []
                delta_func = DELTA_SET[node.delta_name]
                d = delta_func(x_sample[node.c][node.b : node.e], node.x_ref)
                node = node.left if d <= node.eps else node.right
            except Exception as e:
                print(f"Error during node traversal for path: {e}")
                return []

        return path


    def get_max_depth(self):
        """Calculates the maximum depth of the tree."""
        if self.root is None:
            return 0

        max_depth = 0
        stack = [(self.root, 0)]

        while stack:
            node, current_depth = stack.pop()
            if node is not None:
                max_depth = max(max_depth, current_depth)
                # If it's an internal node, add children
                if not node.is_leaf:
                    # Check if children exist before adding
                    if node.right:
                        stack.append((node.right, current_depth + 1))
                    if node.left:
                        stack.append((node.left, current_depth + 1))
        return max_depth
