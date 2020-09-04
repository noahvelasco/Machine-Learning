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
    
    #Fuentes next 2 lines from class
    dist[np.arange(dist.shape[0]),np.arange(dist.shape[0])] = math.inf
    nn = np.argmin(dist,axis=0)
    
    '''
    No modifications - still trying to understand his code
    '''
    return y_train[nn]

if __name__ == '__main__':
    # Read the data
    n = 100
    X = np.load('mnist_X.npy').astype(np.float32)
    y = np.load('mnist_y.npy')
    ind = np.random.permutation(len(y))
    X=X[ind[:n]]
    y=y[ind[:n]]
    
    

    print('Evaluating Algorithm 2')
    start = time.time()
    p2 = classify_nearest_example_fast(X,y,X)
    elapsed_time = time.time() - start 
    print('Accuracy:',accuracy(y,p2))
    print('Elapsed time: ',elapsed_time)
    print('Confusion matrix:')
    print(confusion_matrix(y,p2))
    print('Precision and recall:')
    for i in range(np.amax(y)+1):
        print('Positive class {}, precision: {:.4f}, recall: {:.4f}'.format(i, precision(y,p2,i),recall(y,p2,i)))
   