# Kaggle competetion data science London + Scikit learn
# Colab live project link is: https://colab.research.google.com/drive/18Cr5T3fcIChIM3rWusv1I6SoBR8NghS1?usp=sharing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv', header=None)
test = pd.read_csv('test.csv', header=None)
trainLabel = pd.read_csv('trainLabels.csv', header=None)

train.head()

train.info()

train.describe()

test.tail()

trainLabel.describe()

train.isnull().sum()

X, y = train, trainLabel

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)

from sklearn.svm import SVC

m = SVC()

m.fit(X_train,y_train)
m.score(X_test,y_test)

m.score(X_train,y_train)

from sklearn.metrics import accuracy_score

predict = m.predict(X_test)
accuracy = accuracy_score(y_test, predict)
print(accuracy)

from sklearn.metrics import classification_report
pred = m.predict(X_test)
print(m.__class__.__name__)
print(classification_report(y_test,pred))