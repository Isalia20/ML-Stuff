import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dt = pd.read_csv("MSFT_price_history.csv",index_col = 0)
X, y = dt.drop(["price_increase_bin"],axis = 1), dt["price_increase_bin"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

xgb_clf = XGBClassifier(max_depth=2, n_estimators=5)
xgb_clf.fit(X_train, y_train)

y_pred = xgb_clf.predict(X_train)
accuracy_score(y_train, y_pred)

y_pred_test = xgb_clf.predict(X_test)
accuracy_score(y_test, y_pred_test)
