from LinearRegression.Linear_Regression import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

lin_reg = LinearRegression()
X, y = make_regression(n_samples=100,n_features=5)
lin_reg.fit(X, y)
y_pred = lin_reg.predict(X)
mean_squared_error(y, y_pred)
