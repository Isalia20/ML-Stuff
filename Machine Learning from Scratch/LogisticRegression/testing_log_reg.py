from LogisticRegression.LogisticRegression import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

log_reg = LogisticRegression()
X, y = make_classification(n_samples=100, n_features=5)

log_reg.fit(X, y)

y_pred = log_reg.predict(X)

(y_pred >= 0.5) * 1

accuracy_score(y, (y_pred >= 0.5) * 1)