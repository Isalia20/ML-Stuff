import numpy as np
from Node import Node


class DecisionTreeClassifier:

    def __init__(self,
                 criterion="cross_entropy",
                 splitter="best", # done
                 max_depth=6,  # done
                 min_samples_split=2,  # done
                 min_samples_leaf=2,
                 max_features= None, # done
                 min_impurity_decrease=0):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.root = None
        print("Decision Tree Classifier")


    def _is_finished(self, tree_depth):
        if (self.max_depth <= tree_depth or
                self.n_samples < self.min_samples_split or
                self.n_class_labels == 1):
            return True
        return False

    @staticmethod
    def _calc_entropy(y):
        probabilities = np.bincount(y) / len(y)
        entropy = -np.sum([probability * np.math.log(probability) for probability in probabilities if probability > 0])
        return entropy

    @staticmethod
    def _create_split(x, threshold):
        left_ids = np.argwhere(x <= threshold).flatten()
        right_ids = np.argwhere(x > threshold).flatten()
        return left_ids, right_ids

    def _information_gain(self, x, y, threshold):
        parent_loss = self._calc_entropy(y)
        left_ids, right_ids = self._create_split(x, threshold)
        n, n_left, n_right = len(y), len(left_ids), len(right_ids)

        if n_left == 0 or n_right == 0:
            return 0

        child_loss = n_left / n * self._calc_entropy(y[left_ids]) + n_right / n * self._calc_entropy(y[right_ids])

        return parent_loss - child_loss

    def _best_split(self, x, y, features):
        split = {"score": -1,
                 "feature": None,
                 "threshold": None}

        if self.splitter == "best":
            for feature in features:
                x_feature = x[:, feature]
                thresholds = np.unique(x_feature)
                for threshold in thresholds:
                    loss_improvement = self._information_gain(x_feature, y, threshold)
                    if split["score"] <= loss_improvement:
                        split["score"] = loss_improvement
                        split["feature"] = feature
                        split["threshold"] = threshold
        elif self.splitter == "random":
            num_features = len(features)
            feature = np.random.randint(0, num_features)
            x_feature = x[:, feature]

            feature_mean = np.mean(x_feature)
            feature_stdev = np.std(x_feature)
            threshold = np.random.normal(feature_mean, feature_stdev)

            loss_improvement = self._information_gain(x_feature, y, threshold)
            split["score"] = loss_improvement
            split["feature"] = feature
            split["threshold"] = threshold
        else:
            raise Exception ("invalid splitter selected")

        return split["feature"], split["threshold"]

    def _build_tree(self, x, y, depth=0):
        # we build the tree recursively
        self.n_samples = x.shape[0]
        self.n_features = x.shape[1]
        self.n_class_labels = len(np.unique(y))

        if self._is_finished(depth):
            most_common_label = np.argmax(np.bincount(y))
            return Node(value=most_common_label)

        all_feature_indices = np.array(list(range(x.shape[1])))

        if self.max_features is None:
            features = all_feature_indices
        elif self.max_features == "auto" or self.max_features == "sqrt":
            features = np.random.choice(all_feature_indices, int((self.n_features ** (1/2) // 1) + 1))
        elif self.max_features == "log2":
            features = np.random.choice(all_feature_indices, int(np.math.log2(self.n_features) // 1 + 1))
        else:
            raise Exception("invalid max features selected")

        # Create split
        best_feature_for_split, threshold_split = self._best_split(x, y, features)
        left_ids, right_ids = self._create_split(x[:, best_feature_for_split], threshold_split)
        left_child = self._build_tree(x[left_ids, :], y[left_ids], depth + 1)
        right_child = self._build_tree(x[right_ids, :], y[right_ids], depth + 1)

        return Node(feature=best_feature_for_split, threshold=threshold_split, left=left_child, right=right_child,
                    value=None)

    def _traverse_tree(self, x, node):
        # If it's a leaf
        if node.is_leaf():
            return node.value

        # x is a single sample here
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def fit(self, x, y):
        self.root = self._build_tree(x, y)

    def predict(self, x):
        predictions = [self._traverse_tree(x_scalar, self.root) for x_scalar in x]
        return np.array(predictions)
