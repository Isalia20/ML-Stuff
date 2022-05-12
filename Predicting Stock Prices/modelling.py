import numpy as np
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

from sklearn.tree import DecisionTreeClassifier

X_train_dec = X_train.dropna()
y_train_dec = y_train[y_train.index.isin(X_train_dec.index)]
X_test_dec = X_test.dropna()
y_test_dec = y_test[y_test.index.isin(X_test_dec.index)]

dec_clf = DecisionTreeClassifier(random_state=42)

dec_clf.fit(X_train_dec, y_train_dec)

y_train_pred = dec_clf.predict(X_train_dec)
y_test_pred = dec_clf.predict(X_test_dec)

accuracy_score(y_train_dec, y_train_pred)
accuracy_score(y_test_dec, y_test_pred)  #48% here but 100% on train, need to reduce some overfit

dec_clf = DecisionTreeClassifier(random_state=42, max_depth=10)

dec_clf.fit(X_train_dec, y_train_dec)

y_train_pred = dec_clf.predict(X_train_dec)
y_test_pred = dec_clf.predict(X_test_dec)

accuracy_score(y_train_dec, y_train_pred)
accuracy_score(y_test_dec, y_test_pred)  #48% here but 100% on train, need to reduce some overfit

# Meh tree methods are not working
from sklearn.svm import SVC

svc_clf = SVC(class_weight="balanced", degree=5, kernel="poly")

svc_clf.fit(X_train_dec, y_train_dec)

y_train_pred = svc_clf.predict(X_train_dec)
y_test_pred = svc_clf.predict(X_test_dec)

accuracy_score(y_train_dec, y_train_pred)
accuracy_score(y_test_dec, y_test_pred)  # 50% accuracy not impressive can be achieved by simply saying 1 for everything

y_train_dec.value_counts()

y_train_pred.bincount()

np.bincount(y_train_pred) # Well as expected, didn't quite work
