import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

dataset = pd.read_csv("data/linear_regression/Salary_Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# split data to training set and test set
from sklearn.model_selection import train_test_split

x_tran, x_test, y_tran, y_test = train_test_split(X, Y, test_size=1 / 3, random_state=0)

# feature scaling, line regression already include feature scaling
# fit: 拟合
# from sklearn.preprocessing import StandardScaler
#
# sc_x = StandardScaler()
# x_tran = sc_x.fit_transform(x_tran)
# x_test = sc_x.transform(x_test)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_tran, y_tran)

# Predicting the Test set results
y_pred = regressor.predict(x_test)

# Visualising
plt.scatter(x_tran, y_tran, c="red")
plt.scatter(x_test, y_test, c="black")
plt.plot(x_tran, regressor.predict(x_tran), c="blue")
plt.title("Salary VS Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
