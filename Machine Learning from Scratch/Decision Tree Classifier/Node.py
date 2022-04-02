class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None,n_count=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.n_count = n_count

    def is_leaf(self):
        if self.value is not None:
            return True
        return False
