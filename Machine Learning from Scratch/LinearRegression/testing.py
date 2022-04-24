from LinearRegression.Linear_Regression import LinearRegression
from sklearn.datasets import make_regression

lin_reg = LinearRegression()
X, y = make_regression(n_samples = 100,n_features= 5)

lin_reg.fit(X, y)

y_pred = lin_reg.predict(X)


from sklearn.metrics import mean_squared_error


mean_squared_error(y, y_pred)

lin_reg.weight_matrix