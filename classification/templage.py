# Sigmoid Function S 函数
# y = b0 + b1X
# p = 1 / (1 + e^-y)
# ln(p/(1-p)) = b0 + b1X 概率函数

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("../data/logic_regression/Social_Network_Ads.csv")
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values

# split data to training set and test set
from sklearn.model_selection import train_test_split

x_tran, x_test, y_tran, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# feature scaling
# fit: 拟合
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
x_tran = sc_x.fit_transform(x_tran)
x_test = sc_x.transform(x_test)

# Logistic regression fit
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(x_tran, y_tran)

# Logistic regression predict test set
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix to view the result
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# Visualing
x_set, y_set = x_tran, y_tran
from matplotlib.colors import ListedColormap

X1 = np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01)
X2 = np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01)
X1, X2 = np.meshgrid(X1, X2)
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha=0.75,
             cmap=ListedColormap(('red', 'green')))

# 设置坐标轴范围
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# 画点
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c=ListedColormap(('orange', 'blue'))(i), label=j)

plt.title('Classifier Training set')
plt.xlabel('Age')
plt.ylabel('Salary')
# 图例
plt.legend()
plt.show()

x_set, y_set = x_test, y_test
from matplotlib.colors import ListedColormap

X1 = np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01)
X2 = np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01)
X1, X2 = np.meshgrid(X1, X2)
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha=0.75,
             cmap=ListedColormap(('red', 'green')))

# 设置坐标轴范围
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# 画点
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c=ListedColormap(('orange', 'blue'))(i), label=j)

plt.title('Classifier Test set')
plt.xlabel('Age')
plt.ylabel('Salary')
# 图例
plt.legend()
plt.show()
