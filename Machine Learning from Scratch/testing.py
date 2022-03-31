from sklearn import datasets
from DecisionTreeClassifier import DecisionTreeClassifier
import numpy as np

clf = DecisionTreeClassifier(max_depth= 2)
data = datasets.load_breast_cancer()
X, y = data.data, data.target

clf.fit(X, y)

sum(clf.predict(X))


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

accuracy(y, clf.predict(X))

