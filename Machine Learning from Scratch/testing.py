from sklearn import datasets
from DecisionTreeClassifier import DecisionTreeClassifier
import numpy as np

clf = DecisionTreeClassifier(max_depth=1000)
data = datasets.load_breast_cancer()
X, y = data.data, data.target

clf.fit(X, y)


def accuracy(y_true, y_pred):
    calc_accuracy = np.sum(y_true == y_pred) / len(y_true)
    return calc_accuracy


accuracy(y, clf.predict(X))
