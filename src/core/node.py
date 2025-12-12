# ---- tree node ----
class Node:
    __slots__ = (
        "is_leaf",
        "dist",
        "c",
        "x_ref",
        "b",
        "e",
        "delta_name",
        "eps",
        "left",
        "right",
        "leaf_id",
        "depth"
    )

    # what is *? force all the parameters after the * 
    #            to be specified as keyword arguments
    def __init__(
        self,
        dist=None,
        *,
        c=None,
        x_ref=None,
        b=None,
        e=None,
        delta_name=None,
        eps=None,
        depth=-1
    ):
        self.is_leaf = dist is not None
        # class distribution on the node
        self.dist = dist
        #
        self.c, self.x_ref, self.b, self.e = c, x_ref, b, e
        self.delta_name, self.eps = delta_name, eps
        # store the children if is not a leaf
        self.left = None
        self.right = None
        # Attributes added for unsupervised/Zhu
        self.leaf_id = None
        self.depth = depth

    @property
    def predicted_class(self):
        if self.dist is not None:
            return max(self.dist, key=self.dist.get)
        return None    