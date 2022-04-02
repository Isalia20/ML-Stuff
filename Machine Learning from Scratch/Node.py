class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, n_count = None,
                 value_for_pruning=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.n_count = n_count
        self.value_for_pruning = value_for_pruning

    def is_leaf(self):
        if self.value is None:
            return False
        return True
