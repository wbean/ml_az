import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

dataset = pd.read_csv("data/data_preprocessing/Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# dirty data process
si = SimpleImputer(missing_values=np.nan, strategy='mean')
si.fit(X[:, 1:3])
X[:, 1:3] = si.transform(X[:, 1:3])

# categorical data
li_x = LabelEncoder()
X[:, 0] = li_x.fit_transform(X[:, 0])

# dummy variables
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],
    remainder='passthrough'  # Leave the rest of the columns untouched
)
X = ct.fit_transform(X)

li_y = LabelEncoder()
Y = li_y.fit_transform(Y)

# split data to training set and test set
from sklearn.model_selection import train_test_split

x_tran, x_test, y_tran, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# feature scaling
# fit: 拟合
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
x_tran = sc_x.fit_transform(x_tran)
x_test = sc_x.transform(x_test)
