import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

# load data (have to be classification data set)
cancer = datasets.load_breast_cancer()
# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data
y = cancer.target

# split data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classes = ['malignant', 'benign'] #classifer names

# implement classifer
# SVC = support vector classification.
# In SVC() : If we leave no parameter, we are not tweaking anything => prediction comes out would merely like guesing
# The more parameters we put it, the more accuracy we get for svm
# kernal = function applied;  C =soft margin
clf = svm.SVC(kernel="linear", C=2)
#clf = svm.SVC(kernel="poly") #polynomial might be more accurate, but since it applies more math => take more time
clf.fit(x_train,y_train)

# predic
predictions= clf.predict(x_test)

# get accuracy
acc= metrics.accuracy_score(y_test,predictions)
print("The accuracy of prediction: ",acc)

# -----------------------------------------------
# --Compare with KNearNeighbor-------------------
# NOTE: KNN does not work as well as SVM on multi dimensional data set
model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train,y_train)
knn_predictions = model.predict(x_test)

# get accuracy
knn_acc= metrics.accuracy_score(y_test,knn_predictions)
print("The accuracy of prediction of KNN: ",knn_acc)


