from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import numpy as np

from utils import *

'''
R Noah Padilla submission for final exam coding prob 1
'''

X = np.load('q1_X.npy')
Y = np.load('q1_y.npy')
X_train, X_test, y_train, y_test = split_train_test(X,Y,seed=20)

'''
Part 1 - 
    1. (50 points) Write a program to compare the accuracy of decision tree, k-nearest neighbor, and support vector
classifier to classify the data contained in the files q1 X.npy and q1 y.npy. You can use the sklearn implementations of
the algorithms with default parameters. See the expected output for guidance.
'''

algos = ["DTree" , "KNN" , "SVC"]
algos_preds = []

#-----------------Decision Tree-----------------
print("> Evaluating Decision Tree")
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)
pred = model.predict(X_train)
print('\tAccuracy on training set: {0:.6f}'.format(accuracy(y_train,pred)))
pred = model.predict(X_test)
print('\tAccuracy on test set: {0:.6f}'.format(accuracy(y_test,pred)))
algos_preds.append(accuracy(y_test,pred))

#-----------------KNN-----------------
print("> Evaluating KNN")
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
pred = model.predict(X_train)
print('\tAccuracy on training set: {0:.6f}'.format(accuracy(y_train,pred)))
pred = model.predict(X_test)
print('\tAccuracy on test set: {0:.6f}'.format(accuracy(y_test,pred)))
algos_preds.append(accuracy(y_test,pred))

#-----------------SVC-----------------
print("> Evaluating SVC")
model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
model.fit(X_train, y_train)
pred = model.predict(X_train)
print('\tAccuracy on training set: {0:.6f}'.format(accuracy(y_train,pred)))
pred = model.predict(X_test)
print('\tAccuracy on test set: {0:.6f}'.format(accuracy(y_test,pred)))
algos_preds.append(accuracy(y_test,pred))


print("The best model was", algos[np.argmax(algos_preds)] , "with a test accuracy of",max(algos_preds))

'''
** FUENTES Sample output: **
    
Evaluating Decision Tree
Accuracy on training set: 1.000000
Accuracy on test set: 0.960000

Evaluating k-nearest Neighbor
Accuracy on training set: 0.710000
Accuracy on test set: 0.500000

Evaluating Support Vector Classifier
Accuracy on training set: 0.933333
Accuracy on test set: 0.510000

The best model was Decision Tree with a test set accuracy of 0.960000

** My (Noah) output **

> Evaluating Decision Tree
	Accuracy on training set: 1.000000
	Accuracy on test set: 0.960000
> Evaluating KNN
	Accuracy on training set: 0.710000
	Accuracy on test set: 0.500000
> Evaluating SVC
	Accuracy on training set: 0.932222
	Accuracy on test set: 0.500000
The best model was DTree with a test accuracy of 0.96
'''
