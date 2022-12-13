# polynomial linear regression 多项式线性回归

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

dataset = pd.read_csv("data/polynomial_linear_regression/Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X, Y)

from sklearn.preprocessing import PolynomialFeatures

polynomial_features = PolynomialFeatures(degree=4)  # max power
x_poly = polynomial_features.fit_transform(X)

polynomial_regression = LinearRegression()
polynomial_regression.fit(x_poly, Y)

plt.scatter(X, Y, c="red")
plt.plot(X, regressor.predict(X), c="blue")
plt.title("Truth or Bluff(Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# visualising
x_grid = np.arange(min(X), max(X), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(X, Y, c="red")
plt.plot(x_grid, polynomial_regression.predict(polynomial_features.fit_transform(x_grid)), c="blue")
plt.title("Truth or Bluff(Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Predicting
test = polynomial_features.fit_transform([[6.5]])
rest_result = polynomial_regression.predict(test)
