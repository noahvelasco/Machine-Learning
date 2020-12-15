from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import numpy as np

from utils import *

X_train = np.load('q2_X_train.npy').astype(np.float32).reshape(-1,28*28)
X_test = np.load('q2_X_test.npy').astype(np.float32).reshape(-1,28*28)
y_train = np.load('q2_y_train.npy')
y_test = np.load('q2_y_test.npy')

"""
************************************* PART A *************************************
"""
algos = ["MLP" , "KNN" , "SVC"]
algos_preds = []

#-----------------MLP-----------------
print("> Evaluating MLP")
model = MLPClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_train)
print('\tAccuracy on training set: {0:.6f}'.format(accuracy(y_train,pred)))
pred = model.predict(X_test)
print('\tAccuracy on test set: {0:.6f}'.format(accuracy(y_test,pred)))
algos_preds.append(accuracy(y_test,pred))

#-----------------KNN-----------------
print("> Evaluating KNN")
model = KNeighborsClassifier()
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

print("** The best model was", algos[np.argmax(algos_preds)] , "with a test accuracy of",max(algos_preds), "**")

"""
************************************* PART B *************************************
"""
print("\nAdding mirrored images to training set")

x_train_flipped = np.flip(np.load('q2_X_train.npy')).astype(np.float32).reshape(-1,28*28)

X_train = np.vstack((X_train,x_train_flipped)) # goes from shape (900,784) to (18000,784)
y_train = np.append(y_train,y_train)

algos = ["MLP" , "KNN" , "SVC"]
algos_preds = []

#-----------------MLP-----------------
print("> Evaluating MLP")
model = MLPClassifier(learning_rate='adaptive')
model.fit(X_train, y_train)
pred = model.predict(X_train)
print('\tAccuracy on training set: {0:.6f}'.format(accuracy(y_train,pred)))
pred = model.predict(X_test)
print('\tAccuracy on test set: {0:.6f}'.format(accuracy(y_test,pred)))
algos_preds.append(accuracy(y_test,pred))

#-----------------KNN-----------------
print("> Evaluating KNN")
model = KNeighborsClassifier()
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
print("** The best model was", algos[np.argmax(algos_preds)] , "with a test accuracy of",max(algos_preds), "**")


"""
************************************* PART C *************************************
"""
print('\nCompressing data to 20 principal components')

pca = PCA(n_components=20)
pca.fit(X_train)
ev = pca.explained_variance_ratio_
cum_ev = np.cumsum(ev)
cum_ev = cum_ev/cum_ev[-1]

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

algos = ["MLP" , "KNN" , "SVC"]
algos_preds = []

#-----------------MLP-----------------
print("> Evaluating MLP")
model = MLPClassifier(learning_rate='adaptive')
model.fit(X_train, y_train)
pred = model.predict(X_train)
print('\tAccuracy on training set: {0:.6f}'.format(accuracy(y_train,pred)))
pred = model.predict(X_test)
print('\tAccuracy on test set: {0:.6f}'.format(accuracy(y_test,pred)))
algos_preds.append(accuracy(y_test,pred))

#-----------------KNN-----------------
print("> Evaluating KNN")
model = KNeighborsClassifier()
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
print("** The best model was", algos[np.argmax(algos_preds)] , "with a test accuracy of",max(algos_preds), "**")


'''
** FUENTES SAMPLE OUTPUT **

Evaluating Multi-layer perceptron
Accuracy on training set: 1.000000
Accuracy on test set: 0.495000

Evaluating k-nearest Neighbor
Accuracy on training set: 0.910000
Accuracy on test set: 0.610000

Evaluating Support Vector Classifier
Accuracy on training set: 0.987778
Accuracy on test set: 0.670000

The best model was Support Vector Classifier with a test set accuracy of 0.670000

Adding mirrored images to training set

Evaluating Multi-layer perceptron
Accuracy on training set: 1.000000
Accuracy on test set: 0.730000

Evaluating k-nearest Neighbor
Accuracy on training set: 0.903333
Accuracy on test set: 0.800000

Evaluating Support Vector Classifier
Accuracy on training set: 0.957778
Accuracy on test set: 0.860000

The best model was Support Vector Classifier with a test set accuracy of 0.860000

Compressing data to 20 principal components

Evaluating Multi-layer perceptron
Accuracy on training set: 1.000000
Accuracy on test set: 0.725000

Evaluating k-nearest Neighbor
Accuracy on training set: 0.913889
Accuracy on test set: 0.880000

Evaluating Support Vector Classifier
Accuracy on training set: 0.950556
Accuracy on test set: 0.890000

The best model was Support Vector Classifier with a test set accuracy of 0.890000
------------------------------------------------------------------------------------------------------------------

** My (Noah)  **

> Evaluating MLP
	Accuracy on training set: 1.000000
	Accuracy on test set: 0.520000
> Evaluating KNN
	Accuracy on training set: 0.910000
	Accuracy on test set: 0.610000
> Evaluating SVC
	Accuracy on training set: 0.973333
	Accuracy on test set: 0.620000
The best model was SVC with a test accuracy of 0.62

Adding mirrored images to training set
> Evaluating MLP
	Accuracy on training set: 0.901111
	Accuracy on test set: 0.495000
> Evaluating KNN
	Accuracy on training set: 0.609444
	Accuracy on test set: 0.585000
> Evaluating SVC
	Accuracy on training set: 0.709444
	Accuracy on test set: 0.615000
** The best model was SVC with a test accuracy of 0.615 **

Compressing data to 20 principal components
> Evaluating MLP
	Accuracy on training set: 0.827222
	Accuracy on test set: 0.495000
> Evaluating KNN
	Accuracy on training set: 0.633333
	Accuracy on test set: 0.580000
> Evaluating SVC
	Accuracy on training set: 0.672778
	Accuracy on test set: 0.620000
** The best model was SVC with a test accuracy of 0.62 **
'''


