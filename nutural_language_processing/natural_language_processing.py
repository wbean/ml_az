# Natural Language Processing
"""
磁带模型
构造稀疏矩阵，每个单词构成一列
矩阵中的值是单词出现的次数
"""
import pandas as pd

# Importing dataset
dataset = pd.read_csv('../data/natural_language_processing/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Cleaning the text
import re
import nltk

# 下载虚词库
nltk.download('stopwords')
from nltk.corpus import stopwords

# 词根化
# 词根广泛存在于拉丁语系， 日耳曼语系 等
# love loved lovely
# 对中文文本没有作用
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()

    # drop all 虚词 && 取词根
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    # to string
    review = ' '.join(review)
    corpus.append(review)

# Create the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# split data to training set and test set
from sklearn.model_selection import train_test_split

x_tran, x_test, y_tran, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling
# fit: 拟合
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
x_tran = sc_x.fit_transform(x_tran)
x_test = sc_x.transform(x_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(x_tran, y_tran)

# Naive Bayes predict test set
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix to view the result
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
