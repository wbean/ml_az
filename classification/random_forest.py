"""
Ensemble Learning 集成学习
采用多个分类器预测，最后再对结果进行组合（投票，平均）
建立在决策树上

step1: Pick at random K data points from the Training set.
step2: Build the Decision Tree associated to these K data points.
step3: Choose the number NTree of trees you want to build and repeat step1 and step2
step4: For a new data point, make each one of your NTree trees predict the category
       to which the data point belongs, and assign the new data point to the category
       that wins the majority vote
"""

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

# Fitting Random Forest to the training set
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(
    n_estimators=100,  # 决策树数量
    criterion='entropy',
    random_state=0,
    max_depth=4
)
classifier.fit(x_tran, y_tran)

# Random Forest predict test set
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
