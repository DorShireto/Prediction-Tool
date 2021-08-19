def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, average_precision_score, f1_score, precision_score
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier



""" MODELS METHODS"""
def RF_Classifier(x_train,y_train,x_test,y_test):
    """
    This method will train and test the prediction using Random forest algorithm
    :param x_train: Features to train the model on, can be Standard Scaled or not
    :param y_train: Results to the x_train for the model to train on
    :param x_test: Features from games that needed to be predict
    :param y_test: Actual results
    :return: Prints the Accuracy, Recall, Precision and F1 score
    """
    RF_model = RandomForestClassifier(n_estimators=1200, random_state=0)
    RF_model.fit(x_train, y_train)
    y_prediction = RF_model.predict(x_test)
    print("Accuracy: \n", metrics.accuracy_score(y_test, y_prediction))
    print("Recall score: ", recall_score(y_test, y_prediction, average='macro'))
    print("Precision score: ", average_precision_score(y_test, y_prediction, average='macro'))
    print("F1 score: ", f1_score(y_test, y_prediction, average='macro'))

def KNN_Classifier(x_train,y_train,x_test,y_test):
    """
    This method will train and test the prediction using k-Nearest Neighbours algorithm with 750 neighbours
    :param x_train: Features to train the model on, can be Standard Scaled or not
    :param y_train: Results to the x_train for the model to train on
    :param x_test: Features from games that needed to be predict
    :param y_test: Actual results
    :return: Prints the Accuracy, Recall, Precision and F1 score
    """
    KNN_model = KNeighborsClassifier(n_neighbors=750)
    KNN_model.fit(x_train, y_train)
    y_prediction = KNN_model.predict(x_test)
    print("Accuracy: \n", metrics.accuracy_score(y_test, y_prediction))
    print("Recall score: ", recall_score(y_test, y_prediction, average='macro'))
    print("Precision score: ", average_precision_score(y_test, y_prediction, average='macro'))
    print("F1 score: ", f1_score(y_test, y_prediction, average='macro'))


def SVM_Classifier(x_train,y_train,x_test,y_test):
    """
    This method will train and test the prediction using Support Vector Machine algorithm
    :param x_train: Features to train the model on, can be Standard Scaled or not
    :param y_train: Results to the x_train for the model to train on
    :param x_test: Features from games that needed to be predict
    :param y_test: Actual results
    :return: Prints the Accuracy, Recall, Precision and F1 score
    """
    svm_model = svm.SVC(kernel='rbf')
    svm_model.fit(x_train, y_train)
    y_prediction = svm_model.predict(x_test)
    prediction = [round(val) for val in y_prediction]
    print("Accuracy: \n", metrics.accuracy_score(y_test, y_prediction))
    print("Recall score: ", recall_score(y_test, y_prediction, average='macro'))
    print("Precision score: ", precision_score(y_test, prediction, average='macro'))
    print("F1 score: ", f1_score(y_test, y_prediction, average='macro'))

def GaussianNB_Classifier(x_train,y_train,x_test,y_test):
    """
    This method will train and test the prediction using Gaussian Naive Bayes algorithm
    :param x_train: Features to train the model on, can be Standard Scaled or not
    :param y_train: Results to the x_train for the model to train on
    :param x_test: Features from games that needed to be predict
    :param y_test: Actual results
    :return: Prints the Accuracy, Recall, Precision and F1 score
    """
    GBN_model = GaussianNB()
    GBN_model.fit(x_train, y_train)
    y_prediction = GBN_model.predict(x_test)
    print("Accuracy: \n", metrics.accuracy_score(y_test, y_prediction))
    print("Recall score: ", recall_score(y_test, y_prediction, average='macro'))
    print("Precision score: ", precision_score(y_test, y_prediction, average='macro'))
    print("F1 score: ", f1_score(y_test, y_prediction, average='macro'))

def LDA_Classifier(x_train,y_train,x_test,y_test):
    """
    This method will train and test the prediction using Linear Discriminant Analysis algorithm
    :param x_train: Features to train the model on, can be Standard Scaled or not
    :param y_train: Results to the x_train for the model to train on
    :param x_test: Features from games that needed to be predict
    :param y_test: Actual results
    :return: Prints the Accuracy, Recall, Precision and F1 score
    """
    LDA_model = LinearDiscriminantAnalysis()
    LDA_model.fit(x_train, y_train)
    y_prediction = LDA_model.predict(x_test)
    print("Accuracy: \n", metrics.accuracy_score(y_test, y_prediction))
    print("Recall score: ", recall_score(y_test, y_prediction, average='macro'))
    print("Precision score: ", precision_score(y_test, y_prediction, average='macro'))
    print("F1 score: ", f1_score(y_test, y_prediction, average='macro'))

def LogisticReg_Classifier(x_train,y_train,x_test,y_test):
    """
    This method will train and test the prediction using Logistic Regression algorithm
    :param x_train: Features to train the model on, can be Standard Scaled or not
    :param y_train: Results to the x_train for the model to train on
    :param x_test: Features from games that needed to be predict
    :param y_test: Actual results
    :return: Prints the Accuracy, Recall, Precision and F1 score
    """
    LR_model = LogisticRegression()
    LR_model.fit(x_train, y_train)
    y_prediction = LR_model.predict(x_test)
    print("Accuracy: \n", metrics.accuracy_score(y_test, y_prediction))
    print("Recall score: ", recall_score(y_test, y_prediction, average='macro'))
    print("Precision score: ", precision_score(y_test, y_prediction, average='macro'))
    print("F1 score: ", f1_score(y_test, y_prediction, average='macro'))


def XGBoost_Classifier(x_train,y_train,x_test,y_test):
    """
    This method will train and test the prediction using XG Boost algorithm
    :param x_train: Features to train the model on, can be Standard Scaled or not
    :param y_train: Results to the x_train for the model to train on
    :param x_test: Features from games that needed to be predict
    :param y_test: Actual results
    :return: Prints the Accuracy, Recall, Precision and F1 score
    """
    XGB_model = XGBClassifier()
    XGB_model.fit(x_train, y_train)
    y_prediction = XGB_model.predict(x_test)
    print("Accuracy: \n", metrics.accuracy_score(y_test, y_prediction))
    print("Recall score: ", recall_score(y_test, y_prediction, average='macro'))
    print("Precision score: ", precision_score(y_test, y_prediction, average='macro'))
    print("F1 score: ", f1_score(y_test, y_prediction, average='macro'))


def ADABoost_Classifier(x_train,y_train,x_test,y_test):
    """
    This method will train and test the prediction using ADA Boost algorithm
    :param x_train: Features to train the model on, can be Standard Scaled or not
    :param y_train: Results to the x_train for the model to train on
    :param x_test: Features from games that needed to be predict
    :param y_test: Actual results
    :return: Prints the Accuracy, Recall, Precision and F1 score
    """
    AdaB_model = AdaBoostClassifier(n_estimators=200, random_state=2)
    AdaB_model.fit(x_train, y_train)
    y_prediction = AdaB_model.predict(x_test)
    print("Accuracy: \n", metrics.accuracy_score(y_test, y_prediction))
    print("Recall score: ", recall_score(y_test, y_prediction, average='macro'))
    print("Precision score: ", precision_score(y_test, y_prediction, average='macro'))
    print("F1 score: ", f1_score(y_test, y_prediction, average='macro'))

""" SCRIPT PART """
train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')
x_train = train_set.iloc[:, 3:13].values
y_train = train_set.iloc[:, 13].values
x_test = test_set.iloc[:, 3:13].values
y_test = test_set.iloc[:, 13].values
sc = StandardScaler()
x_train_sc,x_test_sc = sc.fit_transform(x_train),sc.fit_transform(x_test)



""" CALL ALL MODELS """
print("\n\n RandomForestClassifier with StandardScaler n_estimators=1200:")
RF_Classifier(x_train_sc,y_train,x_test_sc,y_test)
print("\n\n RandomForestClassifier without StandardScaler n_estimators=1200:")
RF_Classifier(x_train,y_train,x_test,y_test)
print("\n\nKNNeighborsClassifier with StandardScaler n_neighbors=750:")
KNN_Classifier(x_train_sc,y_train,x_test_sc,y_test)
print("\n\nKNNeighborsClassifier without StandardScaler n_neighbors=750:")
KNN_Classifier(x_train,y_train,x_test,y_test)
print("\n\nSVM model with StandardScaler:")
SVM_Classifier(x_train_sc,y_train,x_test_sc,y_test)
print("\n\nSVM model without StandardScaler:")
SVM_Classifier(x_train,y_train,x_test,y_test)
print("\n\nGaussianNB model with StandardScaler:")
GaussianNB_Classifier(x_train_sc,y_train,x_test_sc,y_test)
print("\n\nGaussianNB model without StandardScaler:")
GaussianNB_Classifier(x_train,y_train,x_test,y_test)
print("\n\nLinearDiscriminantAnalysis model with StandardScaler:")
LDA_Classifier(x_train_sc,y_train,x_test_sc,y_test)
print("\n\nLinearDiscriminantAnalysis model without StandardScaler:")
LDA_Classifier(x_train,y_train,x_test,y_test)
print("\n\nLogistic Reg model with StandardScaler:")
LogisticReg_Classifier(x_train_sc,y_train,x_test_sc,y_test)
print("\n\nLogistic Reg model without StandardScaler:")
LogisticReg_Classifier(x_train,y_train,x_test,y_test)
print("\n\nXGBoost model with StandardScaler:")
XGBoost_Classifier(x_train_sc,y_train,x_test_sc,y_test)
print("\n\nXGBoost model without StandardScaler:")
XGBoost_Classifier(x_train,y_train,x_test,y_test)
print("\n\nAdaBoost model with StandardScaler:")
ADABoost_Classifier(x_train_sc,y_train,x_test_sc,y_test)
print("\n\nAdaBoost model without StandardScaler:")
ADABoost_Classifier(x_train,y_train,x_test,y_test)
