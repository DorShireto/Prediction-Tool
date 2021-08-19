import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')

svm_model = svm.SVC(kernel='rbf')

x_train = train_set.iloc[:,0:12].values
y_train = train_set.iloc[:,12].values

x_test = test_set.iloc[:,0:12].values
y_test = test_set.iloc[:,12].values

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

svm_model.fit(x_train,y_train)
y_prediction = svm_model.predict(x_test)
print("Accuracy: \n", metrics.accuracy_score(y_test,y_prediction))
