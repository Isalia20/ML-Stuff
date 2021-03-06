from LogisticRegression.LogisticRegression import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

log_reg = LogisticRegression(max_iter=1000,tol = 0)
X, y = make_classification(n_samples=100, n_features=5)

log_reg.fit(X, y)

y_pred = log_reg.predict(X)

accuracy_score(y, (y_pred >= 0.5) * 1)

y_pred_proba = log_reg.predict_proba(X)
                                                                                                                   