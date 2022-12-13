# Assumptions
# 1. Linearity 线性
# 2. Homoscedasticity 同方差性
# 3. Multivariate normality 多元正态分布
# 4. Independence of errors 误差独立
# 5. Lack of multi-collinearity 无多重共线性

# methods of building models
# 1. All-in
# 2. Backward Elimination 反向淘汰 应用最多
# 3. Forward Selection 顺向选择
# 4. Bidirectional Elimination 双向淘汰
# 5. Score Comparison 信息量比较，自变量纬度大的时候不可用 2^N-1

# significance level 显著性

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

dataset = pd.read_csv("data/multiple_linear_regression/50_Startups.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

# categorical data
li_x = LabelEncoder()
X[:, 3] = li_x.fit_transform(X[:, 3])

# dummy variables
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],
    remainder='passthrough'  # Leave the rest of the columns untouched
)
X = ct.fit_transform(X)

# avoid dummy variable trap
X = X[:, 1:]

X = np.array(X, dtype=float)

# split data to training set and test set
from sklearn.model_selection import train_test_split

x_tran, x_test, y_tran, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# regression
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_tran, y_tran)

# predicting the test set
y_pred = regressor.predict(x_test)

# Building the optimal model use Backward-Elimination
import statsmodels.api as sm

# add new column as the constant
x_tran = np.append(arr=np.ones((40, 1)), values=x_tran, axis=1)
# x_optimal = x_tran[:, [0, 1, 2, 3, 4, 5]]
# regressor_ols = sm.OLS(y_tran, x_optimal)
# regressor_ols = regressor_ols.fit()
# print(regressor_ols.summary())
#
# x_optimal = x_tran[:, [0, 1, 3, 4, 5]]
# regressor_ols = sm.OLS(y_tran, x_optimal)
# regressor_ols = regressor_ols.fit()
# print(regressor_ols.summary())
#
# x_optimal = x_tran[:, [0, 3, 4, 5]]
# regressor_ols = sm.OLS(y_tran, x_optimal)
# regressor_ols = regressor_ols.fit()
# print(regressor_ols.summary())

# x_optimal = x_tran[:, [0, 3, 5]]
# regressor_ols = sm.OLS(y_tran, x_optimal)
# regressor_ols = regressor_ols.fit()
# print(regressor_ols.summary())

x_optimal = x_tran[:, [0, 3]]
regressor_ols = sm.OLS(y_tran, x_optimal)
regressor_ols = regressor_ols.fit()
print(regressor_ols.summary())
