import numpy as np
import matplotlib.pyplot as plt
import time 
import math 

def split_train_test(X,y,percent_train=0.9):
    ind = np.random.permutation(X.shape[0])
    train = ind[:int(X.shape[0]*percent_train)]
    test = ind[int(X.shape[0]*percent_train):]
    return X[train],X[test], y[train], y[test]

def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/y_true.shape[0]

def confusion_matrix(y_true,y_pred):
    cm = np.zeros((np.amax(y_true)+1,np.amax(y_true)+1),dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i],y_pred[i]]+=1
    return cm

def precision(y_true,y_pred,positive_class):
    tp = np.sum((y_true==positive_class) & (y_pred==positive_class))
    pred_p = np.sum(y_pred==positive_class)
    return tp/pred_p

def recall(y_true,y_pred,positive_class):
    tp = np.sum((y_true==positive_class) & (y_pred==positive_class))
    true_p = np.sum(y_true==positive_class)
    return tp/true_p

def classify_nearest_example_fast(X_train,y_train, X_test):
    # Uses formula (X_train - X_test)**2 = X_train**2 - 2*X_train*X_test + X_test**2
    # Each of the 3 elements can be computed without loops. Third term is not necessary
    # Addition of 3 terms is done with broadcasting without loops
    X_train =  X_train.reshape(-1,784)
    X_test =  X_test.reshape(-1,784).T
    dist = np.sum(X_train**2,axis=1).reshape(-1,1) # dist = X_train**2
    dist = dist - 2*np.matmul(X_train,X_test) # dist = X_train**2  - 2*X_train*X_test
    dist = dist + np.sum(X_test**2,axis=0).reshape(1,-1) # dist = X_train**2  - 2*X_train*X_test + X_test**2 - Not really necessary
    dist = np.sqrt(dist) #  Not really necessary
    nn = np.argmin(dist,axis=0)
    return y_train[nn]

def classify_nearest_example_loocv(X,y):
    X =  X.reshape(-1,784)
    dist = np.sum(X**2,axis=1).reshape(-1,1) # dist = X_train**2
    dist = dist - 2*np.matmul(X,X.T) # dist = X_train**2  - 2*X_train*X_test
    dist = dist + np.sum(X.T**2,axis=0).reshape(1,-1) # dist = X_train**2  - 2*X_train*X_test + X_test**2 - Not really necessary
    dist = np.sqrt(dist) #  Not really necessary
    dist[np.arange(dist.shape[0]),np.arange(dist.shape[0])]= math.inf
    nn = np.argmin(dist,axis=0)
    return y[nn]

def classify_nearest_example_kfcv(X,y,k=5):
    # Modify this function
    pred = np.zeros_like(y)
    for i in range(k):
        train = np.arange(len(y))
        test = np.arange(len(y))
        pred[test] = classify_nearest_example_fast(X[train],y[train], X[test])
    return pred

if __name__ == '__main__':
    # Read the data
    np.random.seed(4361) # Set seed to obtain repeatable results
    n = 10000
    X = np.load('mnist_X.npy').astype(np.float32)
    y = np.load('mnist_y.npy')
    ind = np.random.permutation(len(y))
    X=X[ind[:n]]
    y=y[ind[:n]]
    
    print('Evaluating Algorithm 2')
    start = time.time()
    print('Leave-one-out cross validation')
    p2 = classify_nearest_example_loocv(X,y)
    elapsed_time = time.time() - start 
    print('Accuracy:',accuracy(y,p2))
    print('Elapsed time: ',elapsed_time)
    print('Confusion matrix:')
    print(confusion_matrix(y,p2))
    print('Precision and recall:')
    for i in range(np.amax(y)+1):
        print('Positive class {}, precision: {:.4f}, recall: {:.4f}'.format(i, precision(y,p2,i),recall(y,p2,i)))
    
    print('k-fold cross validation')
    p2 = classify_nearest_example_kfcv(X,y)
    elapsed_time = time.time() - start 
    print('Accuracy:',accuracy(y,p2))
    print('Elapsed time: ',elapsed_time)
    print('Confusion matrix:')
    print(confusion_matrix(y,p2))
    print('Precision and recall:')
    for i in range(np.amax(y)+1):
        print('Positive class {}, precision: {:.4f}, recall: {:.4f}'.format(i, precision(y,p2,i),recall(y,p2,i)))
    
'''
Evaluating Algorithm 2
Leave-one-out cross validation
Accuracy: 0.9522
Elapsed time:  2.4264791011810303
Confusion matrix:
[[ 986    1    0    0    1    4    4    1    2    0]
 [   0 1147    1    1    3    0    1    3    0    3]
 [  11    9  950    4    5    3    4   14    4    6]
 [   0    4    7  981    0   21    2    7   16    3]
 [   0   10    4    0  917    0    3    6    0   41]
 [   5    5    2   27    1  823   14    4    7    5]
 [   2    1    0    0    0    6  955    0    3    0]
 [   1    3    4    1    4    0    0  971    1   23]
 [   8   14    5   19    3   22    5    3  895   10]
 [   4    3    1    4   23    2    1   20    3  897]]
Precision and recall:
Positive class 0, precision: 0.9695, recall: 0.9870
Positive class 1, precision: 0.9582, recall: 0.9896
Positive class 2, precision: 0.9754, recall: 0.9406
Positive class 3, precision: 0.9460, recall: 0.9424
Positive class 4, precision: 0.9582, recall: 0.9348
Positive class 5, precision: 0.9342, recall: 0.9216
Positive class 6, precision: 0.9656, recall: 0.9876
Positive class 7, precision: 0.9436, recall: 0.9633
Positive class 8, precision: 0.9613, recall: 0.9096
Positive class 9, precision: 0.9079, recall: 0.9363
k-fold cross validation
Accuracy: 0.9467
Elapsed time:  4.4121668338775635
Confusion matrix:
[[ 986    1    0    0    1    3    3    1    4    0]
 [   0 1146    1    1    4    0    1    4    0    2]
 [  12   12  938    4    5    4    5   21    3    6]
 [   0    4    8  976    0   22    1    7   16    7]
 [   0   12    3    0  913    0    3    5    0   45]
 [   5    6    2   34    2  811   17    3    7    6]
 [   5    1    0    0    1    6  951    0    3    0]
 [   1    5    4    1    5    0    0  967    1   24]
 [   9   13    4   18    3   25    5    6  888   13]
 [   4    6    2    4   23    1    2   22    3  891]]
Precision and recall:
Positive class 0, precision: 0.9648, recall: 0.9870
Positive class 1, precision: 0.9502, recall: 0.9888
Positive class 2, precision: 0.9751, recall: 0.9287
Positive class 3, precision: 0.9403, recall: 0.9376
Positive class 4, precision: 0.9540, recall: 0.9307
Positive class 5, precision: 0.9300, recall: 0.9082
Positive class 6, precision: 0.9626, recall: 0.9835
Positive class 7, precision: 0.9334, recall: 0.9593
Positive class 8, precision: 0.9600, recall: 0.9024
Positive class 9, precision: 0.8964, recall: 0.9301
''' 